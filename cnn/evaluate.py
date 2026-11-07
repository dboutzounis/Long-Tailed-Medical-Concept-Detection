"""Evaluate a trained CNN checkpoint.

On val, sweeps a global threshold and reports the one with the best sample-averaged F1.
On test, applies a given threshold and writes the predictions as an "ID,CUIs" CSV.

Usage:
    python evaluate.py --config CONFIG --checkpoint CKPT --split val
    python evaluate.py --config CONFIG --checkpoint CKPT --split test --threshold 0.47 --output preds.csv
"""
import argparse
import csv
import os
import pickle

import torch
import yaml
from tqdm import tqdm

from dataset import build_dataloader
from mcloss import build_criterion
from models import build_model


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def expand(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        return obj

    return expand(cfg)


def write_submission(cfg, all_probs, all_labels, all_keys, threshold, output, split_name):
    with open(cfg["paths"]["mlb"], "rb") as f:
        mlb = pickle.load(f)

    preds = (all_probs >= threshold).float().numpy()
    cui_lists = mlb.inverse_transform(preds)

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "CUIs"])
        for key, cuis in zip(all_keys, cui_lists):
            writer.writerow([key, ";".join(cuis)])

    print(f"Written {len(all_keys)} rows -> {output}")

    if all_labels.sum() > 0:
        eps = 1e-8
        preds_t = torch.tensor(preds)
        tp = (preds_t * all_labels).sum(dim=1)
        fp = (preds_t * (1 - all_labels)).sum(dim=1)
        fn = ((1 - preds_t) * all_labels).sum(dim=1)
        prec_i = tp / (tp + fp + eps)
        rec_i = tp / (tp + fn + eps)
        f1_i = 2 * prec_i * rec_i / (prec_i + rec_i + eps)
        print(
            f"\n{split_name} metrics @ threshold={threshold:.2f} | "
            f"F1={f1_i.mean():.4f}  Prec={prec_i.mean():.4f}  "
            f"Rec={rec_i.mean():.4f}  MeanCUIs={preds_t.sum(dim=1).mean():.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    args.checkpoint = os.path.expandvars(args.checkpoint)
    if args.output:
        args.output = os.path.expandvars(args.output)

    if args.split == "test":
        if args.threshold is None:
            parser.error("--threshold is required for --split test")
        if args.output is None:
            parser.error("--output is required for --split test")
    elif args.threshold is not None and args.output is None:
        parser.error("--threshold with --split val only makes sense together with --output")

    cfg = load_config(args.config)

    # Device — evaluate.py is always single-GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        local = False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        local = True
    else:
        device = torch.device("cpu")
        local = True

    print(f"Device: {device}")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    _, get_probs = build_criterion(cfg, device)
    print(f"Loss: {cfg['loss']['type']}")

    # Dataloader
    return_keys = args.split == "test" or args.output is not None
    loader = build_dataloader(args.split, cfg, local=local, return_keys=return_keys)

    print(f"Running inference on {args.split} split...")
    all_probs = []
    all_labels = []
    all_keys = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  infer"):
            if return_keys:
                imgs, labels, keys = batch
                all_keys.extend(keys)
            else:
                imgs, labels = batch

            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(imgs)
            probs = get_probs(logits.float())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)     # (N, C)
    all_labels = torch.cat(all_labels)   # (N, C)

    # -------------------------------------------------------------------------
    # Val path: vectorized threshold sweep
    # -------------------------------------------------------------------------
    if args.split == "val":
        T = 100
        CHUNK = 512
        eps = 1e-8
        thresholds = torch.linspace(0.01, 1, T)
        N = all_probs.size(0)

        f1_sums = torch.zeros(T)
        prec_sums = torch.zeros(T)
        rec_sums = torch.zeros(T)
        mc_sums = torch.zeros(T)

        for start in range(0, N, CHUNK):
            p = all_probs[start:start + CHUNK].unsqueeze(2)             # (chunk, C, 1)
            l = all_labels[start:start + CHUNK].unsqueeze(2).bool()     # (chunk, C, 1)
            preds_c = p >= thresholds.view(1, 1, T)                     # (chunk, C, T)

            tp = (preds_c & l).sum(dim=1).float()
            fp = (preds_c & ~l).sum(dim=1).float()
            fn = (~preds_c & l).sum(dim=1).float()

            prec_i = tp / (tp + fp + eps)
            rec_i = tp / (tp + fn + eps)
            f1_i = 2 * prec_i * rec_i / (prec_i + rec_i + eps)

            f1_sums += f1_i.sum(dim=0)
            prec_sums += prec_i.sum(dim=0)
            rec_sums += rec_i.sum(dim=0)
            mc_sums += preds_c.sum(dim=1).float().sum(dim=0)

        rows = [
            (
                round(thresholds[i].item(), 2),
                (f1_sums[i] / N).item(),
                (prec_sums[i] / N).item(),
                (rec_sums[i] / N).item(),
                (mc_sums[i] / N).item(),
            )
            for i in range(T)
        ]

        print(f"\n{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MeanCUIs':>10}")
        print("-" * 52)
        for thr, f1, prec, rec, mc in rows:
            print(f"{thr:>10.2f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {mc:>10.2f}")

        best = max(rows, key=lambda x: x[1])
        print(
            f"\n*** Best threshold: {best[0]:.2f} -> "
            f"F1={best[1]:.4f}  Prec={best[2]:.4f}  Rec={best[3]:.4f}  MeanCUIs={best[4]:.2f} ***"
        )

        if args.output is not None:
            thr = args.threshold if args.threshold is not None else best[0]
            print(f"\nWriting val submission CSV at threshold={thr:.2f}"
                  f"{'' if args.threshold is not None else ' (val-best, no --threshold given)'}")
            write_submission(cfg, all_probs, all_labels, all_keys, thr, args.output, split_name="Val")

    # -------------------------------------------------------------------------
    # Test path: CSV output
    # -------------------------------------------------------------------------
    else:
        write_submission(cfg, all_probs, all_labels, all_keys, args.threshold, args.output, split_name="Test")


if __name__ == "__main__":
    main()
