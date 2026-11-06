#!/usr/bin/env python3
"""Download the ImageNet weights of the 14 CNN backbones (Section 4.3). Run once on a node
with internet access; the SLURM jobs then read them offline from TORCH_HOME and HF_HOME.

Usage:
    python scripts/download_weights.py
"""
import os
import sys

TORCH_HOME = os.path.expandvars("$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/torch")
HF_HOME = os.path.expandvars("$WORK/Long-Tailed-Medical-Concept-Detection/model_cache/hf")

BACKBONES = [
    "efficientnet_b0",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "convnext_tiny",
    "convnext_small",
    "densenet121",
    "densenet169",
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "resnext101_32x8d",
]

TIMM_BACKBONES = [
    "convnextv2_tiny.fcmae_ft_in1k",
    "convnextv2_base.fcmae_ft_in1k",
]


def download_torchvision():
    os.environ["TORCH_HOME"] = TORCH_HOME
    os.makedirs(TORCH_HOME, exist_ok=True)

    import torchvision.models as tv_models  # imported after TORCH_HOME is set

    print(f"TORCH_HOME = {TORCH_HOME}")
    print(f"Downloading {len(BACKBONES)} torchvision backbone(s)...\n")

    failed = []
    for name in BACKBONES:
        print(f"--- {name} ---")
        try:
            weights = tv_models.get_model_weights(name).DEFAULT
            _ = tv_models.get_model(name, weights=weights)
            print(f"OK: cached {name}\n")
        except Exception as e:
            print(f"FAILED: {name}: {e}\n", file=sys.stderr)
            failed.append(name)

    cache_dir = os.path.join(TORCH_HOME, "hub", "checkpoints")
    if os.path.isdir(cache_dir):
        print(f"Contents of {cache_dir}:")
        for f in sorted(os.listdir(cache_dir)):
            print(f"  {f}")

    return failed


def download_timm():
    if not TIMM_BACKBONES:
        return []

    os.environ["HF_HOME"] = HF_HOME
    os.makedirs(HF_HOME, exist_ok=True)

    import timm  # imported after HF_HOME is set

    print(f"\nHF_HOME = {HF_HOME}")
    print(f"Downloading {len(TIMM_BACKBONES)} timm backbone(s)...\n")

    failed = []
    for name in TIMM_BACKBONES:
        print(f"--- {name} ---")
        try:
            _ = timm.create_model(name, pretrained=True, num_classes=0, global_pool="")
            print(f"OK: cached {name}\n")
        except Exception as e:
            print(f"FAILED: {name}: {e}\n", file=sys.stderr)
            failed.append(name)

    return failed


def main():
    failed = []
    failed += download_torchvision()
    failed += download_timm()

    total = len(BACKBONES) + len(TIMM_BACKBONES)
    if failed:
        print(f"\n{len(failed)}/{total} backbone(s) FAILED to download: {failed}",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {total} backbones cached successfully "
          f"({len(BACKBONES)} under {TORCH_HOME}, {len(TIMM_BACKBONES)} under {HF_HOME}).")


if __name__ == "__main__":
    main()
