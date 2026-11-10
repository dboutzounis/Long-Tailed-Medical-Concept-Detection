"""Routing statistics and counterfactual expert ablation of a trained µMoE head (Section 4.5.2).

On the validation split:
  - routing: top-1 expert load, dead experts, effective number of experts, and the
    enrichment of each concept among the images routed to an expert;
  - ablation: each expert is removed in turn (its slice of the expert factor is zeroed)
    and the relative per-concept drop in AP (or F1 at the decision threshold) is measured,
    giving the polysemanticity of every expert.

Usage:
    python analyze_experts.py --config configs/tr_n32.yaml --checkpoint CKPT \
        --decision-threshold 0.77 --concept-list concepts_fixed.txt --metric ap \
        --ablate-top-k -1 --output report.csv --summary-json summary.json
"""
import argparse
import contextlib
import csv
import json
import math
import os
import pickle
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cnn"))
from dataset import build_dataloader
from models import build_model, MuMoEHead

EPS = 1e-8


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


# ---------------------------------------------------------------------------
# Counterfactual ablation: temporarily zero one expert's factor contribution
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def ablate_expert(head: MuMoEHead, expert_idx: int):
    """
    CPMuMoE: factors[0] has shape (N, R); zeroing row n zeroes W_n exactly.
    TRMuMoE: factors[0] has shape (r1, N, r2); zeroing [:, n, :] zeroes W_n.
    Restores the original values on exit.
    """
    if head.hierarchy != 1:
        raise NotImplementedError("Ablation is only implemented for hierarchy=1 heads.")
    factor0 = head.mumoe.MuMoE.factors[0]
    with torch.no_grad():
        if factor0.dim() == 2:
            backup = factor0[expert_idx].clone()
            factor0[expert_idx] = 0.0
        elif factor0.dim() == 3:
            backup = factor0[:, expert_idx, :].clone()
            factor0[:, expert_idx, :] = 0.0
        else:
            raise ValueError(f"Unexpected factor shape: {tuple(factor0.shape)}")
    try:
        yield
    finally:
        with torch.no_grad():
            if factor0.dim() == 2:
                factor0[expert_idx] = backup
            else:
                factor0[:, expert_idx, :] = backup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def f1_from_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn + EPS)


def counts_from_logits(logits, labels, thr, device, bs=4096):
    """Per-class TP/FP/FN from precomputed logits (CPU tensors), chunked."""
    n_classes = labels.shape[1]
    tp = torch.zeros(n_classes, device=device)
    fp = torch.zeros(n_classes, device=device)
    fn = torch.zeros(n_classes, device=device)
    for i in range(0, len(labels), bs):
        y = labels[i:i + bs].to(device)
        p = (torch.sigmoid(logits[i:i + bs].to(device)) >= thr).float()
        tp += (p * y).sum(0)
        fp += (p * (1 - y)).sum(0)
        fn += ((1 - p) * y).sum(0)
    return tp.cpu(), fp.cpu(), fn.cpu()


def counts_from_head(head, feats, labels, thr, device, bs=4096):
    """Per-class TP/FP/FN running only the head on cached features."""
    n_classes = labels.shape[1]
    tp = torch.zeros(n_classes, device=device)
    fp = torch.zeros(n_classes, device=device)
    fn = torch.zeros(n_classes, device=device)
    with torch.no_grad():
        for i in range(0, len(labels), bs):
            x = feats[i:i + bs].to(device)
            y = labels[i:i + bs].to(device)
            p = (torch.sigmoid(head(x).float()) >= thr).float()
            tp += (p * y).sum(0)
            fp += (p * (1 - y)).sum(0)
            fn += ((1 - p) * y).sum(0)
    return tp.cpu(), fp.cpu(), fn.cpu()


def counts_full_pass(model, loader, thr, device, n_classes, max_batches=None):
    """Fallback: full forward passes through backbone + head."""
    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            p = (torch.sigmoid(model(imgs.to(device, non_blocking=True)).float()).cpu() >= thr).float()
            labels = labels.float()
            tp += (p * labels).sum(0)
            fp += (p * (1 - labels)).sum(0)
            fn += ((1 - p) * labels).sum(0)
    return tp, fp, fn


def average_precision(scores, labels, device, col_chunk=256):
    n_cols = scores.shape[1]
    out = torch.zeros(n_cols)
    for j in range(0, n_cols, col_chunk):
        s = scores[:, j:j + col_chunk].to(device).float()
        y = labels[:, j:j + col_chunk].to(device).float()
        order = torch.argsort(s, dim=0, descending=True)
        y_sorted = torch.gather(y, 0, order)
        cum_tp = torch.cumsum(y_sorted, dim=0)
        ranks = torch.arange(1, s.shape[0] + 1, device=device, dtype=torch.float32).unsqueeze(1)
        n_pos = y_sorted.sum(0)
        ap = (cum_tp / ranks * y_sorted).sum(0) / n_pos.clamp(min=1)
        out[j:j + col_chunk] = torch.where(n_pos > 0, ap, torch.zeros_like(ap)).cpu()
    return out


def head_logits_cols(head, feats, cols, device, bs=4096):
    """Logits of selected concept columns, running only the head on cached features."""
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            out.append(head(feats[i:i + bs].to(device)).float()[:, cols].cpu())
    return torch.cat(out)


def full_pass_logits_cols(model, loader, cols, device, max_batches=None):
    """Fallback: logits of selected columns (and labels) from full forward passes."""
    logits, labels = [], []
    with torch.no_grad():
        for i, (imgs, lab) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            logits.append(model(imgs.to(device, non_blocking=True)).float()[:, cols].cpu())
            labels.append(lab.float()[:, cols])
    return torch.cat(logits), torch.cat(labels)


def scan_val(model, head, loader, device, max_batches=None):
    """One full pass: cache head inputs, expert coefficients, logits, labels."""
    captured = {}

    def pre_hook(module, args):
        captured["x"] = args[0].detach()

    handle = head.register_forward_pre_hook(pre_hook)
    feats, coefs, logits, labels = [], [], [], []
    with torch.no_grad():
        for i, (imgs, lab) in enumerate(tqdm(loader, desc="  scan")):
            if max_batches and i >= max_batches:
                break
            out = model(imgs.to(device, non_blocking=True))
            coefs.append(head.expert_coefficients()[0].squeeze(1).float().cpu())
            feats.append(captured["x"].float().cpu())
            logits.append(out.float().cpu())
            labels.append(lab.float())
    handle.remove()
    return torch.cat(feats), torch.cat(coefs), torch.cat(logits), torch.cat(labels)


def safe_corr(a, b):
    if len(a) < 3 or a.std() < EPS or b.std() < EPS:
        return float("nan")
    return torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decision-threshold", type=float, required=True,
                   help="Validation-calibrated sigmoid threshold of this model.")
    p.add_argument("--output", default=None, help="CSV path for the per-expert report.")
    p.add_argument("--summary-json", default=None, help="Optional JSON with run-level numbers.")
    p.add_argument("--dump-base-f1", default=None,
                   help="Save per-concept baseline F1/support to this .pt file and exit.")
    p.add_argument("--cui-mapping", default=None, help="cui_mapping.csv (CUI, Canonical name).")
    p.add_argument("--top-k-cuis", type=int, default=8)
    p.add_argument("--load-mode", choices=["top1", "threshold"], default="top1")
    p.add_argument("--active-threshold", type=str, default="auto",
                   help="Only used with --load-mode threshold ('auto' or a number).")
    p.add_argument("--min-enrich-count", type=int, default=10,
                   help="Min co-occurrences of a concept with an expert to report enrichment.")
    p.add_argument("--concept-list", default=None,
                   help="Fixed concept set for the polysemanticity metric (one CUI per line). "
                        "Overrides --min-base-f1 / --min-base-support for the metric.")
    p.add_argument("--metric", choices=["f1", "ap"], default="f1",
                   help="Per-concept metric for the ablation: F1 at --decision-threshold, or "
                        "threshold-free average precision (AP).")
    p.add_argument("--min-base-f1", type=float, default=0.05)
    p.add_argument("--min-base-ap", type=float, default=0.10,
                   help="With --metric ap and no --concept-list: minimum baseline AP.")
    p.add_argument("--min-effect", type=float, default=None,
                   help="Largest relative drop below which an expert counts as having no "
                        "measurable effect. Default 1e-6 for f1 (discrete), 1e-3 for ap "
                        "(any weight change perturbs the ranking slightly).")
    p.add_argument("--min-base-support", type=int, default=30)
    p.add_argument("--ablate-top-k", type=int, default=-1,
                   help="Ablate the K most-loaded experts; -1 = all experts, 0 = no ablation.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the head-only fast path (always run full forward passes).")
    p.add_argument("--max-val-batches", type=int, default=None, help="Smoke-test cap.")
    args = p.parse_args()

    if args.dump_base_f1 is None and args.output is None:
        p.error("--output is required unless --dump-base-f1 is given")
    if args.min_effect is None:
        args.min_effect = 1e-3 if args.metric == "ap" else 1e-6
    metric_name = "AP" if args.metric == "ap" else "F1"

    args.checkpoint = os.path.expandvars(args.checkpoint)
    for k in ("output", "summary_json", "dump_base_f1", "cui_mapping", "concept_list"):
        if getattr(args, k):
            setattr(args, k, os.path.expandvars(getattr(args, k)))

    cfg = load_config(args.config)
    if cfg["model"]["head_type"] not in ("cpmumoe", "trmumoe"):
        raise SystemExit("analyze_experts.py requires a µMoE head (cpmumoe / trmumoe).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    head: MuMoEHead = model.classifier
    n_experts = head.n_experts
    print(f"Loaded {args.checkpoint} | µMoE head, {n_experts} experts (hierarchy={head.hierarchy})")
    print(f"Decision threshold: {args.decision_threshold}")

    with open(cfg["paths"]["mlb"], "rb") as f:
        mlb = pickle.load(f)
    cui_names = [str(c) for c in mlb.classes_]
    n_classes = len(cui_names)

    cui_to_name = {}
    if args.cui_mapping:
        with open(args.cui_mapping, newline="") as f:
            for row in csv.DictReader(f):
                cui_to_name[row["CUI"]] = row.get("Canonical name", row["CUI"])

    def cui_label(cui):
        return f"{cui} ({cui_to_name[cui]})" if cui in cui_to_name else cui

    loader = build_dataloader("val", cfg, local=(device.type != "cuda"))

    # -------------------------------------------------------------------
    # Pass 1: single full scan of the validation split
    # -------------------------------------------------------------------
    print("Scanning validation split...")
    feats, a_all, logits_model, labels_all = scan_val(model, head, loader, device, args.max_val_batches)
    n_total = len(labels_all)
    support = labels_all.sum(0)
    print(f"Val images scanned: {n_total}")

    # Baseline per-concept F1 at the calibrated threshold (from the full model)
    base_tp, base_fp, base_fn = counts_from_logits(logits_model, labels_all, args.decision_threshold, device)
    base_f1 = f1_from_counts(base_tp, base_fp, base_fn)
    # Baseline per-concept AP (threshold-free), computed on logits to minimise ties
    base_ap = average_precision(logits_model, labels_all, device) if args.metric == "ap" else None

    if args.dump_base_f1:
        if base_ap is None:
            base_ap = average_precision(logits_model, labels_all, device)
        os.makedirs(os.path.dirname(args.dump_base_f1) or ".", exist_ok=True)
        torch.save({"cuis": cui_names, "f1": base_f1, "ap": base_ap, "support": support,
                    "threshold": args.decision_threshold, "checkpoint": args.checkpoint},
                   args.dump_base_f1)
        print(f"Saved baseline per-concept F1 and AP -> {args.dump_base_f1}")
        return

    base_metric = base_ap if args.metric == "ap" else base_f1
    min_base = args.min_base_ap if args.metric == "ap" else args.min_base_f1

    # Verify the head-only fast path
    use_cache = False
    if not args.no_cache:
        try:
            with torch.no_grad():
                hl = torch.cat([head(feats[i:i + 4096].to(device)).float().cpu()
                                for i in range(0, n_total, 4096)])
            diff = (hl - logits_model).abs().max().item()
            use_cache = diff < 1e-3
            print(f"Head-only cache check: max |logit diff| = {diff:.2e} -> "
                  f"{'using cached features' if use_cache else 'FALLING BACK to full passes'}")
        except Exception as e:  # noqa: BLE001
            print(f"Head-only cache check failed ({e!r}) -> falling back to full passes")

    # -------------------------------------------------------------------
    # Routing statistics
    # -------------------------------------------------------------------
    max_coef = a_all.max(dim=1).values
    q = torch.quantile(max_coef, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    print("Per-image largest expert coefficient: "
          f"mean={max_coef.mean():.3f} p10={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} "
          f"p75={q[3]:.3f} p90={q[4]:.3f} (uniform would be {1.0 / n_experts:.4f})")

    if args.load_mode == "top1":
        top1 = a_all.argmax(dim=1)
        active = torch.nn.functional.one_hot(top1, n_experts).float()
        load_desc = "top-1"
        chosen_threshold = None
    else:
        if args.active_threshold.strip().lower() == "auto":
            chosen_threshold = max(q[2].item() - 0.01, 0.01)
        else:
            chosen_threshold = float(args.active_threshold)
        active = (a_all >= chosen_threshold).float()
        load_desc = f"coefficient >= {chosen_threshold:.3f}"

    expert_load = active.sum(0)                         # (N,)
    expert_cui_pos = active.t() @ labels_all            # (N, C)
    mean_coef = a_all.mean(0)                           # (N,)
    n_dead = int((expert_load == 0).sum().item())
    load_frac = expert_load / expert_load.sum().clamp(min=1)
    nz = load_frac[load_frac > 0]
    eff_experts = math.exp(-(nz * nz.log()).sum().item())   # perplexity of the load distribution
    print(f"Load mode: {load_desc} | dead experts: {n_dead}/{n_experts} "
          f"({100.0 * n_dead / n_experts:.1f}%) | effective number of experts: {eff_experts:.1f}")

    base_rate = support / max(n_total, 1)

    # -------------------------------------------------------------------
    # Counterfactual ablation
    # -------------------------------------------------------------------
    k = n_experts if args.ablate_top_k < 0 else min(args.ablate_top_k, n_experts)
    ablation = {}
    summary_abl = {}
    if k > 0:
        if args.concept_list:
            keep = {l.strip() for l in open(args.concept_list) if l.strip()}
            valid_mask = torch.tensor([c in keep for c in cui_names])
            src = f"fixed list {os.path.basename(args.concept_list)}"
        else:
            valid_mask = (base_metric > min_base) & (support >= args.min_base_support)
            src = f"baseline {metric_name} > {min_base} and support >= {args.min_base_support}"
        zero_base = valid_mask & (base_metric <= EPS)
        if zero_base.any():
            print(f"  WARNING: {int(zero_base.sum())} listed concepts have zero baseline "
                  f"{metric_name} in this model and are excluded.")
        valid_mask = valid_mask & (base_metric > EPS)
        n_valid = int(valid_mask.sum().item())
        valid_cols = valid_mask.nonzero().flatten()
        print(f"Polysemanticity concept set: {n_valid} concepts ({src}) | metric: {metric_name} "
              f"| min effect: {args.min_effect:g}")

        experts = torch.argsort(expert_load, descending=True)[:k].tolist()
        for e in tqdm(experts, desc="  ablating"):
            with ablate_expert(head, e):
                if args.metric == "ap":
                    if use_cache:
                        lg = head_logits_cols(head, feats, valid_cols.to(device), device)
                        lb = labels_all[:, valid_cols]
                    else:
                        lg, lb = full_pass_logits_cols(model, loader, valid_cols.to(device), device,
                                                       args.max_val_batches)
                    metric_after = torch.zeros(n_classes)
                    metric_after[valid_cols] = average_precision(lg, lb, device)
                else:
                    if use_cache:
                        tp, fp, fn = counts_from_head(head, feats, labels_all, args.decision_threshold, device)
                    else:
                        tp, fp, fn = counts_full_pass(model, loader, args.decision_threshold, device,
                                                      n_classes, args.max_val_batches)
                    metric_after = f1_from_counts(tp, fp, fn)
            drop = torch.zeros(n_classes)
            # Relative drop (Eq. 4 with per-concept F1 or AP), clamped: only damage counts
            drop[valid_mask] = ((base_metric[valid_mask] - metric_after[valid_mask])
                                / base_metric[valid_mask]).clamp(min=0)
            max_drop = drop.max().item() if n_valid else 0.0
            if n_valid == 0 or max_drop < args.min_effect:
                poly = None
            else:
                onehot = torch.zeros(n_classes)
                onehot[torch.argmax(drop)] = 1.0
                poly = torch.norm(drop - onehot, p=2).item()
            top = torch.topk(drop, k=min(args.top_k_cuis, n_classes))
            ablation[e] = {
                "polysemanticity": poly,
                "max_drop": max_drop,
                "top_cuis": [(cui_names[c], drop[c].item()) for c in top.indices.tolist()
                             if drop[c].item() > 0],
            }

        scored = [(e, v) for e, v in ablation.items() if v["polysemanticity"] is not None]
        if scored:
            loads = torch.tensor([expert_load[e].item() for e, _ in scored])
            polys = torch.tensor([v["polysemanticity"] for _, v in scored])
            mdrops = torch.tensor([v["max_drop"] for _, v in scored])
            above = loads > loads.median()
            summary_abl = {
                "n_concepts_metric": n_valid,
                "n_ablated": len(ablation),
                "n_measurable": len(scored),
                "mean_polysemanticity": polys.mean().item(),
                "mean_polysemanticity_above_median_load": polys[above].mean().item() if above.any() else float("nan"),
                "mean_max_drop": mdrops.mean().item(),
                "corr_load_polysemanticity": safe_corr(loads, polys),
            }
            print(f"\nMean polysemanticity: {summary_abl['mean_polysemanticity']:.4f} "
                  f"({len(scored)}/{len(ablation)} ablated experts with measurable impact)")
            print(f"  Mean polysemanticity, experts above median load: "
                  f"{summary_abl['mean_polysemanticity_above_median_load']:.4f}")
            print(f"  Mean largest relative {metric_name} drop per expert: {summary_abl['mean_max_drop']:.4f}")
            print(f"  Correlation(load, polysemanticity): {summary_abl['corr_load_polysemanticity']:.3f}")
        else:
            summary_abl = {"n_concepts_metric": n_valid, "n_ablated": len(ablation), "n_measurable": 0}
            print("\nNo ablated expert had a measurable effect on the concept set.")

    # -------------------------------------------------------------------
    # Per-expert report
    # -------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        f.write(f"# load_mode={args.load_mode}"
                + (f" active_threshold={chosen_threshold:.4f}" if chosen_threshold is not None else "")
                + f" decision_threshold={args.decision_threshold} n_experts={n_experts}"
                + f" min_enrich_count={args.min_enrich_count} metric={args.metric}\n")
        w = csv.writer(f)
        header = ["expert_id", "n_images_loaded", "pct_load", "mean_coefficient",
                  f"top_enriched_cuis (count>={args.min_enrich_count}, support>={args.min_base_support})"]
        if k > 0:
            header += ["polysemanticity (lower = more monosemantic)", f"max_relative_{args.metric}_drop",
                       f"counterfactual_top_affected_cuis (relative {metric_name} drop)"]
        w.writerow(header)

        for e in torch.argsort(expert_load, descending=True).tolist():
            n_act = int(expert_load[e].item())
            if n_act == 0:
                enr_str = "(dead expert)"
            else:
                enr = (expert_cui_pos[e] / n_act) / (base_rate + EPS)
                ok = (expert_cui_pos[e] >= args.min_enrich_count) & (support >= args.min_base_support)
                enr = torch.where(ok, enr, torch.zeros_like(enr))
                top = torch.topk(enr, k=min(args.top_k_cuis, n_classes))
                enr_str = "; ".join(
                    f"{cui_label(cui_names[c])} [{enr[c].item():.1f}x, n={int(expert_cui_pos[e, c].item())}]"
                    for c in top.indices.tolist() if enr[c].item() > 0) or "(no concept passes the filter)"
            row = [e, n_act, f"{100.0 * n_act / max(n_total, 1):.2f}", f"{mean_coef[e].item():.4f}", enr_str]
            if k > 0:
                if e in ablation:
                    v = ablation[e]
                    row += [f"{v['polysemanticity']:.4f}" if v["polysemanticity"] is not None
                            else "n/a (no measurable effect)",
                            f"{v['max_drop']:.4f}",
                            "; ".join(f"{cui_label(c)} [{d:.2f}]" for c, d in v["top_cuis"])]
                else:
                    row += ["", "", ""]
            w.writerow(row)
    print(f"Wrote report -> {args.output}")

    if args.summary_json:
        summary = {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "head_type": cfg["model"]["head_type"],
            "n_experts": n_experts,
            "decision_threshold": args.decision_threshold,
            "val_images": n_total,
            "used_feature_cache": use_cache,
            "max_coef_mean": max_coef.mean().item(),
            "max_coef_median": q[2].item(),
            "uniform_coef": 1.0 / n_experts,
            "load_mode": args.load_mode,
            "metric": args.metric,
            "min_effect": args.min_effect,
            "n_dead": n_dead,
            "pct_dead": 100.0 * n_dead / n_experts,
            "effective_experts": eff_experts,
            "max_load_pct": 100.0 * expert_load.max().item() / max(n_total, 1),
            **summary_abl,
        }
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
