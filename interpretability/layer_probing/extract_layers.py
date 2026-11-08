"""Cache the CLS token of every BiomedCLIP transformer block (Section 4.6.1).

Writes {split}_layer_{i}_embs.pt (L2-normalized, 768-d) for i = 0..11 and {split}_labels.pt.

Usage:
    python extract_layers.py --config ../../vlm/common/base_config.yaml \
        --cache-dir $WORK/Long-Tailed-Medical-Concept-Detection/model_cache --output-dir LAYER_DIR
"""

import argparse
import os
import sys

import torch
from open_clip import create_model_from_pretrained
from tqdm import tqdm

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vlm", "common")
)
from loaders import build_biomedclip_dataloader, load_config

intermediate_features = {}  # filled by the forward hooks


def get_activation(layer_name):
    def hook(model, input, output):
        out = output[0] if isinstance(output, tuple) else output

        # 197 tokens = CLS + 14 x 14 patches; handle both batch- and sequence-first layouts
        cls_token = out[0, :, :] if out.shape[0] == 197 and out.shape[1] != 197 else out[:, 0, :]

        intermediate_features[layer_name] = cls_token.detach().cpu().float()

    return hook


def extract_and_save_layers(model, loader, device, split, output_dir):
    model.eval()

    num_layers = len(model.visual.trunk.blocks)
    all_embs = {f"layer_{i}": [] for i in range(num_layers)}
    all_labels = []

    use_amp = device.type == "cuda"

    print(f"Extracting all layers for '{split}' split...")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split}"):
            imgs, labels = batch[0].to(device), batch[1]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _ = model.encode_image(imgs)

            for layer_name in all_embs.keys():
                norm_emb = torch.nn.functional.normalize(intermediate_features[layer_name], dim=-1)
                all_embs[layer_name].append(norm_emb)

            all_labels.append(labels.cpu())

    os.makedirs(output_dir, exist_ok=True)

    labels_tensor = torch.cat(all_labels, dim=0)
    torch.save(labels_tensor, os.path.join(output_dir, f"{split}_labels.pt"))

    for layer_name, embs_list in all_embs.items():
        layer_tensor = torch.cat(embs_list, dim=0)
        torch.save(layer_tensor, os.path.join(output_dir, f"{split}_{layer_name}_embs.pt"))
        print(f"Saved {split} {layer_name}: {layer_tensor.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading BiomedCLIP...")
    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        cache_dir=os.path.expandvars(args.cache_dir),
    )
    model = model.to(device)

    print(f"Found {len(model.visual.trunk.blocks)} transformer blocks. Attaching hooks...")
    for i, block in enumerate(model.visual.trunk.blocks):
        block.register_forward_hook(get_activation(f"layer_{i}"))

    train_loader = build_biomedclip_dataloader("train", cfg, preprocess, return_keys=False)
    extract_and_save_layers(model, train_loader, device, "train", args.output_dir)

    val_loader = build_biomedclip_dataloader("val", cfg, preprocess, return_keys=False)
    extract_and_save_layers(model, val_loader, device, "val", args.output_dir)

    test_loader = build_biomedclip_dataloader("test", cfg, preprocess, return_keys=False)
    extract_and_save_layers(model, test_loader, device, "test", args.output_dir)


if __name__ == "__main__":
    main()
