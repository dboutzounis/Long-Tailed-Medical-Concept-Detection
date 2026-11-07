"""Max Constraint Module and MCLoss (Giunchiglia & Lukasiewicz, NeurIPS 2020), ported from
https://github.com/EGiunchiglia/C-HMCNN, plus build_criterion(), which selects BCE, ASL
or MCLoss from the config for both training and evaluation.

The hierarchy is a descendant matrix D (C x C), with D[a, c] = 1 if concept c is a or one
of its descendants (built by data/umls_hierarchy/).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from asl_loss import AsymmetricLoss


def _load_descendant_matrix(path: str) -> torch.Tensor:
    D_sparse = sparse.load_npz(path)
    D_dense = torch.tensor(D_sparse.toarray(), dtype=torch.float32)
    return D_dense


def _masked_max(values: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    masked = values.unsqueeze(1) * D.unsqueeze(0)  # [batch, N, N], broadcast
    return masked.max(dim=2).values


class MCLoss(nn.Module):
    """
    Drop-in replacement for BCEWithLogitsLoss that enforces and trains
    toward hierarchy coherence using the UMLS descendant matrix.
    """

    def __init__(self, descendant_matrix_path: str, device=None):
        super().__init__()
        D = _load_descendant_matrix(descendant_matrix_path)
        if device is not None:
            D = D.to(device)
        self.register_buffer("D", D)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)

        # MCM_A = max over D_A of h_B -- used for the negative-class branch
        constr_output = _masked_max(probs, self.D)

        # max over D_A of (y_B * h_B) -- used for the positive-class branch
        masked_probs = targets * probs
        positive_branch = _masked_max(masked_probs, self.D)

        train_output = (1 - targets) * constr_output + targets * positive_branch
        train_output = train_output.clamp(min=1e-7, max=1 - 1e-7)  # guard log(0)

        with torch.autocast(device_type=logits.device.type, enabled=False):
            return F.binary_cross_entropy(train_output, targets)

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return _masked_max(probs, self.D)


def build_criterion(cfg, device):
    loss_cfg = cfg["loss"]
    loss_type = loss_cfg["type"]
    pos_weight_enabled = cfg["pos_weight"]["enabled"]

    if loss_type == "bce":
        pos_weight = None
        if pos_weight_enabled:
            pos_weight = torch.load(cfg["paths"]["pos_weight"], weights_only=True).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        get_probs = torch.sigmoid
        return criterion, get_probs

    elif loss_type == "mcloss":
        if pos_weight_enabled:
            raise ValueError(
                "pos_weight.enabled=true is not supported with loss.type='mcloss'. "
                "MCLoss's masked-max construction (see mcloss.py's module docstring) "
                "has no defined per-class weighting term in the reference formulation. "
                "Set pos_weight.enabled: false, or switch loss.type back to 'bce'."
            )
        matrix_path = cfg["paths"]["descendant_matrix"]
        criterion = MCLoss(matrix_path, device=device)
        get_probs = criterion.predict
        return criterion, get_probs

    elif loss_type == "asl":
        if pos_weight_enabled:
            raise ValueError(
                "pos_weight.enabled=true is not supported with loss.type='asl'. "
                "ASL already has its own, more targeted imbalance-handling mechanisms "
                "(asymmetric clipping + asymmetric focal weighting, see asl_loss.py) -- "
                "combining it with pos_weight would conflate two different reweighting "
                "schemes with no defined interaction. Set pos_weight.enabled: false, or "
                "switch loss.type back to 'bce'."
            )
        asl_cfg = loss_cfg.get("asl", {})
        criterion = AsymmetricLoss(
            gamma_neg=asl_cfg.get("gamma_neg", 4),
            gamma_pos=asl_cfg.get("gamma_pos", 1),
            clip=asl_cfg.get("clip", 0.05),
        ).to(device)
        get_probs = torch.sigmoid
        return criterion, get_probs

    else:
        raise ValueError(f"Unknown loss.type: {loss_type!r} (use 'bce', 'mcloss', or 'asl')")
