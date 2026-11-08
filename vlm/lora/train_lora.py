#!/usr/bin/env python
"""LoRA fine-tuning of one vision encoder with a jointly trained linear head (Section 4.4.6, Table 5.10).

LoRA is applied to every linear layer of every transformer block; the encoder stays in eval
mode. The head starts from the frozen linear probe (vlm/frozen_heads), and since B = 0 the model
at step 0 is exactly that probe, which is checked on val before training. ASL, AdamW, early
stopping on val loss; global threshold on val, test evaluated once.

Usage (4 GPUs):
    torchrun --standalone --nproc_per_node=4 train_lora.py --config config.yaml --model medsiglip
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA init

import argparse
import csv
import hashlib
import json
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from lora_common import (AsymmetricLoss, EncoderClassifier, all_reduce_, barrier,
                         best_threshold, build_backend, env_info, f1_sums, import_common,
                         init_distributed, load_config, make_loader, set_determinism)
from lora_modules import (attach_lora, enable_block_checkpointing, lora_blocks,
                          lora_state_dict)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def run_name(model, lcfg, tag):
    return "%s__lora_r%d_a%g%s%s" % (model, lcfg["rank"], lcfg["alpha"],
                                     "_dora" if lcfg.get("dora") else "", tag or "")


@torch.no_grad()
def evaluate(core, loader, crit, device, chunk, max_batches=None):
    """Distributed eval: per-rank sums, all-reduced. Returns (loss, f1_vec, n)."""
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    n = torch.zeros((), dtype=torch.float64, device=device)
    f1 = None
    for b, (x, y, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        y = y.to(device, non_blocking=True)
        logits = torch.cat([core(x[s:s + chunk].to(device, non_blocking=True))
                            for s in range(0, x.shape[0], chunk)])
        loss_sum += crit(logits, y).double() * y.shape[0]
        n += y.shape[0]
        s = f1_sums(torch.sigmoid(logits), y.bool())
        f1 = s if f1 is None else f1 + s
    if f1 is None:
        f1 = torch.zeros(99, dtype=torch.float64, device=device)
    all_reduce_(loss_sum)
    all_reduce_(n)
    all_reduce_(f1)
    return (loss_sum / n).item(), (f1 / n).cpu(), int(n.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True,
                    choices=["biomedclip", "medsiglip", "medimageinsight", "raddino"])
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument("--overwrite", action="store_true")
    # debug knobs (a debug run's metrics are NOT comparable)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--eval-max-batches", type=int, default=None)
    ap.add_argument("--deterministic-attention", choices=["true", "false"], default=None,
                    help="override training.deterministic_attention from the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.deterministic_attention is not None:
        cfg["training"]["deterministic_attention"] = args.deterministic_attention == "true"
    P, T, L, D = cfg["paths"], cfg["training"], cfg["lora"], cfg["data"]
    M = cfg["models"][args.model]
    rank, world, device = init_distributed()
    is_main = rank == 0
    seed = int(T["seed"])
    set_determinism(seed, T.get("deterministic", True), T.get("deterministic_attention", True))

    def log(msg):
        if is_main:
            print(msg, flush=True)

    name = run_name(args.model, L, args.tag)
    run_dir = os.path.join(P["out_root"], name)
    if os.path.exists(os.path.join(run_dir, "results.json")) and not args.overwrite:
        log("results exist in %s (use --overwrite); skipping" % run_dir)
        return
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    barrier()

    backends_mod, common_mod = import_common()
    classes = list(common_mod.load_mlb(P["mlb"]).classes_)
    C = len(classes)

    # encoder + LoRA (identical init on every rank)
    backend = build_backend(backends_mod, args.model, cfg)
    blocks = lora_blocks(backend)
    torch.manual_seed(seed)
    n_wrapped = attach_lora(blocks, L["rank"], L["alpha"], L.get("dora", False))
    if M.get("grad_ckpt", False):
        enable_block_checkpointing(blocks)
    backend.visual.to(device).eval()

    # head from the frozen linear probe
    probe_dir = os.path.join(P["frozen_heads_root"], args.model, "linear")
    probe_sd = torch.load(os.path.join(probe_dir, "head.pt"), map_location="cpu")
    w, b = probe_sd["fc.weight"], probe_sd["fc.bias"]
    if w.shape[0] != C:
        raise ValueError("frozen probe has %d classes, mlb has %d" % (w.shape[0], C))
    head = nn.Linear(w.shape[1], C)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
    amp = T.get("precision", "bf16") == "bf16" and device.type == "cuda"
    model = EncoderClassifier(backend, head, amp).to(device)
    model.eval()

    n_lora = sum(p.numel() for n_, p in model.named_parameters() if ".lora_" in n_)
    n_total = sum(p.numel() for p in model.visual.parameters())
    log("model %s | LoRA on %d linears, r=%d alpha=%g dora=%s | %.2fM LoRA params "
        "(%.2f%% of encoder) | world %d"
        % (args.model, n_wrapped, L["rank"], L["alpha"], L.get("dora", False),
           n_lora / 1e6, 100 * n_lora / max(n_total, 1), world))

    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
              broadcast_buffers=False) if world > 1 else model

    # optimiser: decay on LoRA matrices + head weight only
    groups = {"lora_decay": [], "lora_nodecay": [], "head_decay": [], "head_nodecay": []}
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n_.startswith("head."):
            groups["head_decay" if p.dim() >= 2 else "head_nodecay"].append(p)
        else:
            groups["lora_decay" if p.dim() >= 2 else "lora_nodecay"].append(p)
    wd = T["weight_decay"]
    opt = torch.optim.AdamW([g for g in [
        {"params": groups["lora_decay"], "lr": T["lr_lora"], "weight_decay": wd},
        {"params": groups["lora_nodecay"], "lr": T["lr_lora"], "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": T["lr_head"], "weight_decay": wd},
        {"params": groups["head_nodecay"], "lr": T["lr_head"], "weight_decay": 0.0},
    ] if g["params"]])
    trainable = [p for p in model.parameters() if p.requires_grad]

    A = cfg["asl"]
    crit = AsymmetricLoss(A["gamma_neg"], A["gamma_pos"], A["clip"])

    # data
    micro, eff = M["batch_size"], T["effective_batch"]
    if eff % (micro * world):
        raise ValueError("effective_batch %d not divisible by micro %d x world %d" % (eff, micro, world))
    accum = eff // (micro * world)
    steps = args.steps_per_epoch or D["train_size"] // eff
    max_epochs = args.max_epochs or T["max_epochs"]
    train_loader = make_loader(P["train_shards"], classes, backend.eval_transform,
                               common_mod.parse_cuis, micro, D["num_workers"],
                               D["prefetch_factor"], train=True, seed=seed,
                               shuffle_buffer=D["shuffle_buffer"], distributed=world > 1)
    E = cfg["extract"]

    def eval_loader(split):
        return make_loader(P[split + "_shards"], classes, backend.eval_transform,
                           common_mod.parse_cuis, E["batch_size"], E["num_workers"],
                           E["prefetch_factor"], train=False, distributed=world > 1)

    log("micro-batch %d x %d GPUs x %d accum = %d | %d steps/epoch | max %d epochs | patience %d"
        % (micro, world, accum, eff, steps, max_epochs, T["patience"]))

    # provenance
    if is_main:
        conf = {"args": vars(args), "config": cfg, "run_name": name, "env": env_info(device, world),
                "frozen_probe": os.path.join(probe_dir, "head.pt"),
                "frozen_probe_sha256": sha256_file(os.path.join(probe_dir, "head.pt")),
                "lora_linears": n_wrapped, "lora_params": n_lora, "accum": accum,
                "steps_per_epoch": steps}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(conf, f, indent=2)
    frozen_ref = None
    ref_path = os.path.join(probe_dir, "results.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            frozen_ref = json.load(f)

    chunk = M["eval_chunk"]

    # epoch 0: must reproduce the frozen linear probe
    t_start = time.time()
    v_loss, v_f1, n_val = evaluate(model, eval_loader("val"), crit, device, chunk, args.eval_max_batches)
    i0, thr0, f10 = best_threshold(v_f1)
    log("epoch   0 | val loss %.4f | val F1 %.4f @ %.2f | %d val images (init = frozen probe)"
        % (v_loss, f10, thr0, n_val))
    if frozen_ref is not None:
        log("            frozen probe reference: val loss %.4f | val F1 %.4f @ %.2f"
            % (frozen_ref["best_val_loss"], frozen_ref["val_f1"], frozen_ref["threshold"]))
    init_stats = {"val_loss": v_loss, "val_f1": f10, "threshold": thr0, "n_val": n_val}

    log_f = open(os.path.join(run_dir, "log.csv"), "w", newline="") if is_main else None
    writer = csv.writer(log_f) if is_main else None
    if is_main:
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1_tuned", "val_thr", "minutes"])
        writer.writerow([0, "", "%.6f" % v_loss, "%.6f" % f10, thr0, "0.0"])
        log_f.flush()

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()
                if ".lora_" in k or k.startswith("head.")}

    best = {"loss": v_loss, "epoch": 0, "state": snapshot(), "f1": v_f1}
    bad, epoch = 0, 0
    it = iter(train_loader)
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        run_loss = torch.zeros((), dtype=torch.float64, device=device)
        t_warm = None
        for step in range(steps):
            if step == min(5, steps - 1):
                t_warm = time.time()  # step timing excludes data-loader warm-up
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                x, y, _ = next(it)
                sync = nullcontext() if (world == 1 or a == accum - 1) else ddp.no_sync()
                with sync:
                    loss = crit(ddp(x.to(device, non_blocking=True)),
                                y.to(device, non_blocking=True)) / accum
                    loss.backward()
                run_loss += loss.detach().double()
            if T["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(trainable, T["grad_clip"])
            opt.step()
            if is_main and (step + 1) % 25 == 0:
                print("  epoch %d step %d/%d | %.1f s" % (epoch, step + 1, steps, time.time() - t0),
                      flush=True)
        n_timed = steps - min(5, steps - 1)
        s_per_step = (time.time() - t_warm) / n_timed
        peak_gb = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
        train_loss = (all_reduce_(run_loss) / (world * steps)).item()
        t_eval = time.time()

        v_loss, v_f1, _ = evaluate(model, eval_loader("val"), crit, device, chunk, args.eval_max_batches)
        _, thr, f1 = best_threshold(v_f1)
        improved = v_loss < best["loss"]  # identical on every rank (all-reduced)
        if improved:
            best = {"loss": v_loss, "epoch": epoch, "state": snapshot(), "f1": v_f1}
            bad = 0
        else:
            bad += 1
        minutes = (time.time() - t0) / 60
        log("epoch %3d | train %.4f | val loss %.4f | val F1 %.4f @ %.2f | %.1f min%s"
            % (epoch, train_loss, v_loss, f1, thr, minutes, "  *" if improved else ""))
        log("            %.2f s/step (after warm-up) | val pass %.1f min | peak GPU mem %.1f GiB"
            % (s_per_step, (time.time() - t_eval) / 60, peak_gb))
        if is_main:
            writer.writerow([epoch, "%.6f" % train_loss, "%.6f" % v_loss, "%.6f" % f1, thr,
                             "%.1f" % minutes])
            log_f.flush()
        if bad >= T["patience"]:
            break
    hit_cap = bad < T["patience"]
    if hit_cap:
        log("WARNING: reached max epochs without triggering early stopping")

    # restore best, save, test once at the val-tuned threshold
    model.load_state_dict(best["state"], strict=False)
    if is_main:
        torch.save({"lora": lora_state_dict(model.visual),
                    "head": {k: v.cpu() for k, v in model.head.state_dict().items()},
                    "lora_cfg": {"rank": L["rank"], "alpha": L["alpha"], "dora": L.get("dora", False)},
                    "model": args.model, "epoch": best["epoch"]},
                   os.path.join(run_dir, "best.pt"))
    i_best, thr, val_f1 = best_threshold(best["f1"])
    _, t_f1, n_test = evaluate(model, eval_loader("test"), crit, device, chunk, args.eval_max_batches)
    test_f1 = t_f1[i_best].item()
    results = {"model": args.model, "run": name, "best_epoch": best["epoch"],
               "epochs_run": epoch, "hit_max_epochs": hit_cap, "best_val_loss": best["loss"],
               "val_f1": val_f1, "threshold": thr, "test_f1": test_f1, "n_test": n_test,
               "init": init_stats, "frozen_probe_reference": frozen_ref and {
                   k: frozen_ref[k] for k in ("best_val_loss", "val_f1", "threshold", "test_f1")},
               "lora_params": n_lora, "minutes_total": round((time.time() - t_start) / 60, 1),
               "debug": any(v is not None for v in (args.max_epochs, args.steps_per_epoch,
                                                     args.eval_max_batches))}
    if is_main:
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        log_f.close()
    log("DONE %s | best epoch %d | val F1 %.4f @ %.2f | test F1 %.4f | %.1f min"
        % (name, best["epoch"], val_f1, thr, test_f1, results["minutes_total"]))
    barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
