"""CNN backbone + GeM pooling + linear or µMoE (CP / Tensor Ring) classification head.

Backbones come from torchvision, or from timm when torchvision does not have them
(ConvNeXt V2).
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

from mumoe_layer import MuMoE, CPMuMoE, TRMuMoE


# ---------------------------------------------------------------------------
# GeM pooling
# ---------------------------------------------------------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, p_trainable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if p_trainable else p
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

    def __repr__(self):
        p_val = round(self.p.item(), 4) if isinstance(self.p, nn.Parameter) else self.p
        return f"GeM(p={p_val}, eps={self.eps})"


# ---------------------------------------------------------------------------
# Generic backbone loader — works with (almost) any torchvision CNN
# ---------------------------------------------------------------------------
def get_backbone(name, pretrained=True):
    try:
        weights = tv_models.get_model_weights(name).DEFAULT if pretrained else None
        model = tv_models.get_model(name, weights=weights)
    except ValueError:
        import timm
        return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")

    if hasattr(model, "features"):
        return model.features

    children = list(model.children())
    if len(children) >= 3:
        return nn.Sequential(*children[:-2])

    raise ValueError(
        f"Don't know how to extract spatial (B,C,H,W) features from backbone "
        f"'{name}'. This helper covers CNNs with a `.features` attribute or a "
        f"ResNet-style (..., avgpool, fc) tail. For transformer backbones "
        f"(vit_*, swin_*) are not supported."
    )


def get_feature_dim(backbone, img_size=224):
    """Detect the channel dimension of the backbone's spatial output via a
    dummy CPU forward pass."""
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        out = backbone(torch.zeros(1, 3, img_size, img_size))
    backbone.train(was_training)
    if out.dim() != 4:
        raise ValueError(
            f"Expected a 4D (B,C,H,W) spatial feature map from the backbone, "
            f"got shape {tuple(out.shape)}. This backbone likely already "
            f"includes global pooling internally and isn't supported here."
        )
    return out.shape[1]


# ---------------------------------------------------------------------------
# µMoE classifier head
# ---------------------------------------------------------------------------
def _build_tr_ranks(expert_dims, input_dim, output_dim, r_expert, r_io):
    """Build the Tensor-Ring core-shape list."""
    n_modes = len(expert_dims)
    if n_modes == 1:
        r1, r2, r3 = r_expert, r_expert, r_io
        return [[r1, expert_dims[0], r2], [r2, input_dim, r3], [r3, output_dim, r1]]
    elif n_modes == 2:
        r1, r2, r3, r4 = r_expert, r_expert, r_expert, r_io
        return [[r1, expert_dims[0], r2], [r2, expert_dims[1], r3],
                [r3, input_dim, r4], [r4, output_dim, r1]]
    else:
        raise NotImplementedError("Only hierarchy 1 or 2 are wired up here.")


class MuMoEHead(nn.Module):
    """Drop-in replacement for nn.Linear (or a 2-layer MLP head, via second_dim > 0) using the µMoE layer."""
    def __init__(self, input_dim, output_dim, layer_type="cp",
                 n_experts=64, rank=256, hierarchy=1, second_experts=None,
                 second_dim=0, normalization="bn"):
        super().__init__()
        expert_dims = [n_experts] if hierarchy == 1 else [n_experts, second_experts]

        if layer_type == "cp":
            layer_cls = CPMuMoE
            ranks = rank  # single int CP rank
        elif layer_type == "tr":
            layer_cls = TRMuMoE
            r_expert, r_io = rank
            ranks = _build_tr_ranks(expert_dims, input_dim, output_dim, r_expert, r_io)
        else:
            raise ValueError(f"Unknown µMoE layer_type: {layer_type!r} (use 'cp' or 'tr')")

        self.mumoe = MuMoE(
            input_dim=input_dim,
            output_dim=output_dim,
            MuMoE_layer=layer_cls,
            expert_dims=expert_dims,
            ranks=ranks,
            normalization=normalization,
            hierarchy=hierarchy,
            second_dim=second_dim,
        )
        self.hierarchy = hierarchy
        self.n_experts = n_experts

    def forward(self, x):
        return self.mumoe(x).squeeze(1)

    def expert_coefficients(self):
        return self.mumoe.a


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class MuMoEConceptDetectionModel(nn.Module):
    """
    CNN backbone -> pooling -> dropout -> classifier head.
    head_type: 'linear' | 'cpmumoe' | 'trmumoe'.
    """
    def __init__(self, config):
        super().__init__()
        m_cfg = config["model"]
        d_cfg = config["data"]

        self.backbone = get_backbone(m_cfg["backbone"], pretrained=m_cfg.get("pretrained", True))
        feat_dim = get_feature_dim(self.backbone, img_size=d_cfg["image_size"])
        self.feat_dim = feat_dim

        pool_type = m_cfg.get("pool_type", "gem")
        if pool_type == "gem":
            self.pool = GeM(p=m_cfg.get("gem_p", 3.0), p_trainable=True)
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=m_cfg.get("dropout", 0.0))

        head_type = m_cfg.get("head_type", "linear")
        num_classes = d_cfg["num_classes"]

        if head_type == "linear":
            self.classifier = nn.Linear(feat_dim, num_classes)
        elif head_type in ("cpmumoe", "trmumoe"):
            mm = m_cfg["mumoe"]
            layer_type = "cp" if head_type == "cpmumoe" else "tr"
            rank = mm["rank"] if layer_type == "cp" else (mm["r_expert"], mm["r_io"])
            self.classifier = MuMoEHead(
                input_dim=feat_dim,
                output_dim=num_classes,
                layer_type=layer_type,
                n_experts=mm["n_experts"],
                rank=rank,
                hierarchy=mm.get("hierarchy", 1),
                second_experts=mm.get("second_experts", None),
                second_dim=mm.get("second_dim", 0),
                normalization=mm.get("normalization", "bn"),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type!r} (use 'linear', 'cpmumoe', or 'trmumoe')")

        self.head_type = head_type

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)

    # param groups for the optimizer (see train.py: build_optimizer)
    def backbone_named_parameters(self):
        return self.backbone.named_parameters(prefix="backbone")

    def head_named_parameters(self):
        for n, p in self.pool.named_parameters(prefix="pool"):
            yield n, p
        for n, p in self.classifier.named_parameters(prefix="classifier"):
            yield n, p


def build_model(config):
    return MuMoEConceptDetectionModel(config)


def count_parameters(model):
    backbone_p = sum(p.numel() for p in model.backbone.parameters())
    head_p = sum(p.numel() for _, p in model.head_named_parameters())
    total_p = sum(p.numel() for p in model.parameters())
    return {"backbone": backbone_p, "head": head_p, "total": total_p}
