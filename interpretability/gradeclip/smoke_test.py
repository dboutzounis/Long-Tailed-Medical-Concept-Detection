#!/usr/bin/env python
"""CPU smoke test: no weights, no data. Builds an open_clip TimmModel with
BiomedCLIP's image-tower configuration (timm vit_base_patch16_224, CLS pooling,
linear projection to 512) with RANDOM weights and runs every code path of
run_gradeclip.py except data loading and the encoder backends.

  python smoke_test.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_gradeclip as R  # noqa: E402
from core import TimmViTTap, denormalize_to_uint8, find_normalize  # noqa: E402


class FakeBackend:
    def __init__(self):
        from open_clip.timm_model import TimmModel
        from open_clip.transform import image_transform
        torch.manual_seed(0)
        # BiomedCLIP open_clip config: timm vit_base_patch16_224, pool "", proj linear
        self.visual = TimmModel("vit_base_patch16_224", embed_dim=512, image_size=224,
                                pool="", proj="linear").eval()
        self.eval_transform = image_transform(224, is_train=False)

    def encode_image(self, x):
        return self.visual(x)


def main():
    # login nodes limit CPU time and memory per process: few threads, tiny batches
    torch.set_num_threads(int(os.environ.get("SMOKE_THREADS", "2")))
    torch.use_deterministic_algorithms(True, warn_only=True)
    be = FakeBackend()
    for p in be.visual.parameters():
        p.requires_grad_(False)
    tap = TimmViTTap(be.visual.trunk, -1)
    print("grid", tap.grid, "patch", tap.patch, "heads", tap.num_heads)
    C, d = 20, 512
    g = torch.Generator().manual_seed(1)
    T = F.normalize(torch.randn(C, d, generator=g), dim=-1)
    W, b = torch.randn(C, d, generator=g) * 0.1, torch.zeros(C)
    x = torch.randn(1, 3, 224, 224, generator=g)

    maps, scores, fn = R.explain_image(be, tap, x, [3, 7], T, W, b, "smoke")
    d_layout = (tap.recompute_attn_out() - tap.attn_out.detach()).abs().max().item()
    print("hook layout max abs diff %.3g" % d_layout)
    assert d_layout < 1e-4
    for k, v in maps.items():
        assert v.shape == (2, 196), (k, v.shape)
        assert torch.isfinite(v).all(), k
        print("%-22s min %.3g max %.3g nonzero %d" % (k, v.min(), v.max(), (v > 0).sum()))
    assert maps["gradeclip__lin"].abs().sum() > 0
    # maps of different CUIs on the same image must differ
    assert not torch.allclose(maps["gradeclip__lin"][0], maps["gradeclip__lin"][1])

    # the score explained is the unmodified model's score
    with torch.no_grad():
        ref = F.normalize(be.encode_image(x).float(), dim=-1)
    assert torch.allclose(ref[0], fn, atol=1e-6)
    assert torch.allclose(scores["zs"], (ref @ T.t())[0, [3, 7]], atol=1e-6)

    again, _, _ = R.explain_image(be, tap, x, [3], T, W, b, "smoke")
    assert torch.equal(again["gradeclip__lin"][0], maps["gradeclip__lin"][0]), "not deterministic"

    fcfg = {"n_steps": 4, "blur_sigma": 10.0, "batch_size": 4}
    print("faithfulness: %d forward passes" % (2 * len(R.ORDERS) * 2 * 5), flush=True)
    curves = R.faithfulness(be, tap, x, maps, [3, 7], T, W, b, fcfg)
    print("curves", curves.shape)
    assert curves.shape == (2, len(R.ORDERS), 2, 5, 2)
    # deletion at step 0 == insertion at last step == original image score
    with torch.no_grad():
        p0 = torch.sigmoid(ref @ W.t() + b)[0, [3, 7]].numpy()
    assert np.allclose(curves[:, :, 0, 0, 0], p0[:, None], atol=1e-5)
    assert np.allclose(curves[:, :, 1, -1, 0], p0[:, None], atol=1e-5)
    # all orders start/end at the same images -> identical endpoints
    assert np.allclose(curves[:, :, 0, -1, :], curves[:, :1, 0, -1, :], atol=1e-5)

    tap.remove()
    rvis = R.randomized_copy(be.visual, -1, 0).eval()
    rtap = TimmViTTap(rvis.trunk, -1)
    rm, _, _ = R.explain_image(R._VisualOnly(rvis), rtap, x, [3, 7], T, W, b, "r")
    rho = R.spearman(maps["gradeclip__lin"][0], rm["gradeclip__lin"][0])
    print("randomization spearman (random model vs re-initialised block): %.3f" % rho)
    assert len(be.visual.trunk.blocks[-1]._forward_pre_hooks) == 0, "hooks leaked"

    mean, std = find_normalize(be.eval_transform)
    img = denormalize_to_uint8(x[0], mean, std)
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8

    # summary on fake outputs
    out = tempfile.mkdtemp()
    np.savez_compressed(os.path.join(out, "curves.npz"),
                        curves=np.concatenate([curves, curves]),
                        orders=np.array([R.order_name(m, t) for m, t in R.ORDERS]),
                        modes=np.array(R.MODES), targets=np.array(R.TARGETS), n_steps=4)
    with open(os.path.join(out, "pairs.csv"), "w") as f:
        f.write("key,cui\na,C1\na,C2\nb,C1\nb,C2\n")
    R.summarize(out, n_boot=200)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
