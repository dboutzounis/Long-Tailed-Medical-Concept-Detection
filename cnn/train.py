"""Train a CNN backbone with a linear or µMoE head (Sections 4.1-4.3, 4.5).

DDP with torchrun, BF16 autocast, ReduceLROnPlateau and early stopping on validation
loss. The checkpoint with the lowest validation loss is kept, and the run resumes
from <checkpoints>/last.pt if one exists.

Usage:
    torchrun --nproc_per_node=4 train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --local    # short CPU/1-GPU debug run
"""
import argparse
import contextlib
import os
import random
import warnings

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import build_dataloader
from mcloss import build_criterion
from models import build_model, count_parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def run_tag(cfg):
    m_cfg = cfg["model"]
    tag = f"{m_cfg['backbone']}__{m_cfg['head_type']}"
    if m_cfg["head_type"] in ("cpmumoe", "trmumoe"):
        mm = m_cfg["mumoe"]
        tag += f"__n{mm['n_experts']}"
        if m_cfg["head_type"] == "cpmumoe":
            tag += f"__r{mm['rank']}"
        else:
            tag += f"__re{mm['r_expert']}_rio{mm['r_io']}"
        if mm.get("hierarchy", 1) == 2:
            tag += f"__h2n{mm['second_experts']}"
    loss_type = cfg["loss"]["type"]
    if loss_type != "bce":
        tag += f"__{loss_type}"
    return tag


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _no_decay(name, param):
    """Biases, norm layers, and any scalar/1D parameter (e.g. GeM's p,
    µMoE gating BatchNorm — though affine=False means it has none) skip
    weight decay."""
    return name.endswith("bias") or "norm" in name.lower() or param.ndim <= 1


def build_optimizer(model, cfg):
    """Four param groups: {backbone, head} x {weight_decay, no_weight_decay}."""
    t_cfg = cfg["training"]
    raw_model = model.module if isinstance(model, DDP) else model

    backbone_wd, backbone_no_wd = [], []
    for n, p in raw_model.backbone_named_parameters():
        (backbone_no_wd if _no_decay(n, p) else backbone_wd).append(p)

    head_wd, head_no_wd = [], []
    for n, p in raw_model.head_named_parameters():
        (head_no_wd if _no_decay(n, p) else head_wd).append(p)

    # Force float casting to protect against PyYAML string/list parsing
    lr_bb = float(t_cfg["lr_backbone"])
    lr_h = float(t_cfg["lr_head"])
    wd = float(t_cfg["weight_decay"])

    param_groups = [
        {"params": backbone_wd,    "lr": lr_bb, "weight_decay": wd},
        {"params": backbone_no_wd, "lr": lr_bb, "weight_decay": 0.0},
        {"params": head_wd,        "lr": lr_h,  "weight_decay": wd},
        {"params": head_no_wd,     "lr": lr_h,  "weight_decay": 0.0},
    ]

    optimizer_type = t_cfg.get("optimizer", "adamw").lower()
    if optimizer_type == "adamw":
        return AdamW(param_groups)
    elif optimizer_type == "adam":
        return Adam(param_groups)
    else:
        raise ValueError(f"Unknown training.optimizer: {optimizer_type!r} (use 'adam' or 'adamw')")


def build_scheduler(optimizer, cfg):
    t_cfg = cfg["training"]
    scheduler_type = t_cfg.get("scheduler_type", "reduce_on_plateau")

    if scheduler_type == "reduce_on_plateau":
        metric = t_cfg.get("early_stop_metric", "val_loss")
        mode = "max" if metric == "val_f1" else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=t_cfg.get("plateau_factor", 0.5),
            patience=t_cfg.get("plateau_patience", 1),
        )
    elif scheduler_type == "warmup_cosine":
        warmup_epochs = t_cfg["warmup_epochs"]
        total_epochs = t_cfg["epochs"]
        min_lr = t_cfg["min_lr"]
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}")


def scheduler_step(scheduler, scheduler_type, metric_value):
    if scheduler is None:
        return
    if scheduler_type == "reduce_on_plateau":
        scheduler.step(metric_value)
    else:
        scheduler.step()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, no_improve, cfg, metric="val_loss"):
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "metric": metric,
        "no_improve": no_improve,
        "config": cfg,
    }
    if metric == "val_loss":
        ckpt["best_val_loss"] = best_score  # legacy key, same value
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, device, cfg=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])

    if cfg is not None and "config" in ckpt:
        old_opt = ckpt["config"].get("training", {}).get("optimizer", "adamw")
        new_opt = cfg.get("training", {}).get("optimizer", "adamw")
        if old_opt != new_opt:
            print(f"  WARNING: resuming a checkpoint saved with training.optimizer={old_opt!r} "
                  f"but the current config specifies {new_opt!r}. Adam/AdamW optimizer states "
                  f"aren't guaranteed compatible -- optimizer.load_state_dict() below may raise, "
                  f"or silently leave the optimizer in an inconsistent state.")

    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "best_score" in ckpt:
        best_score, ckpt_metric = ckpt["best_score"], ckpt.get("metric", "val_loss")
    else:
        best_score, ckpt_metric = ckpt["best_val_loss"], "val_loss"
    return ckpt["epoch"] + 1, best_score, ckpt["no_improve"], ckpt_metric


# ---------------------------------------------------------------------------
# Training / validation passes
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, cfg, use_amp, rank=0, max_batches=None):
    model.train()
    t_cfg = cfg["training"]
    accum_steps = t_cfg["grad_accum_steps"]
    max_norm = t_cfg["max_grad_norm"]
    is_ddp = isinstance(model, DDP)

    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  train", leave=False, disable=(rank != 0), total=max_batches)
    for i, (imgs, labels) in enumerate(pbar):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        is_sync_step = (i + 1) % accum_steps == 0
        sync_ctx = contextlib.nullcontext() if (is_sync_step or not is_ddp) else model.no_sync()

        with sync_ctx:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels) / accum_steps
            loss.backward()

        if is_sync_step:
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_batches += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, get_probs, device, cfg, use_amp, is_ddp, rank=0, max_batches=None,
              threshold=None):
    """get_probs: callable (logits) -> probabilities, [batch, N]."""
    model.eval()
    t_cfg = cfg["training"]
    metric = t_cfg.get("early_stop_metric", "val_loss")
    eps = 1e-8

    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    if metric == "val_f1":
        T = 100
        CHUNK = 512
        thresholds = torch.linspace(0.01, 1, T, device=device)
        f1_sums = torch.zeros(T, device=device, dtype=torch.float64)
        prec_sums = torch.zeros(T, device=device, dtype=torch.float64)
        rec_sums = torch.zeros(T, device=device, dtype=torch.float64)
        mc_sums = torch.zeros(T, device=device, dtype=torch.float64)
    else:
        threshold = threshold if threshold is not None else t_cfg["threshold"]
        tp = fp = fn = 0.0
        n_pred_concepts = 0.0

    pbar = tqdm(loader, desc="  val", leave=False, disable=(rank != 0), total=max_batches)
    for i, (imgs, labels) in enumerate(pbar):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        probs = get_probs(logits.float())
        n_samples += imgs.size(0)

        if metric == "val_f1":
            B = probs.size(0)
            for start in range(0, B, CHUNK):
                p = probs[start:start + CHUNK].unsqueeze(2)             # (chunk, C, 1)
                l = labels[start:start + CHUNK].unsqueeze(2).bool()     # (chunk, C, 1)
                preds_c = p >= thresholds.view(1, 1, T)                  # (chunk, C, T)

                tp_i = (preds_c & l).sum(dim=1).double()                 # (chunk, T)
                fp_i = (preds_c & ~l).sum(dim=1).double()
                fn_i = (~preds_c & l).sum(dim=1).double()

                prec_i = tp_i / (tp_i + fp_i + eps)
                rec_i = tp_i / (tp_i + fn_i + eps)
                f1_i = 2 * prec_i * rec_i / (prec_i + rec_i + eps)

                f1_sums += f1_i.sum(dim=0)
                prec_sums += prec_i.sum(dim=0)
                rec_sums += rec_i.sum(dim=0)
                mc_sums += preds_c.sum(dim=1).double().sum(dim=0)
        else:
            preds = (probs >= threshold).float()
            tp += (preds * labels).sum().item()
            fp += (preds * (1 - labels)).sum().item()
            fn += ((1 - preds) * labels).sum().item()
            n_pred_concepts += preds.sum().item()

        total_loss += loss.item()
        n_batches += 1

    if metric == "val_f1":
        if is_ddp:
            scalar_stats = torch.tensor([total_loss, n_batches, n_samples],
                                         device=device, dtype=torch.float64)
            dist.all_reduce(scalar_stats, op=dist.ReduceOp.SUM)
            total_loss, n_batches, n_samples = scalar_stats.tolist()
            dist.all_reduce(f1_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(prec_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(rec_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(mc_sums, op=dist.ReduceOp.SUM)

        avg_loss = total_loss / max(n_batches, 1)
        f1_per_t = f1_sums / max(n_samples, 1)
        best_idx = torch.argmax(f1_per_t).item()

        f1 = f1_per_t[best_idx].item()
        precision = (prec_sums[best_idx] / max(n_samples, 1)).item()
        recall = (rec_sums[best_idx] / max(n_samples, 1)).item()
        mean_concepts = (mc_sums[best_idx] / max(n_samples, 1)).item()
        val_threshold = thresholds[best_idx].item()

        return avg_loss, f1, precision, recall, mean_concepts, val_threshold

    else:
        if is_ddp:
            stats = torch.tensor([total_loss, n_batches, tp, fp, fn, n_pred_concepts, n_samples],
                                  device=device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, n_batches, tp, fp, fn, n_pred_concepts, n_samples = stats.tolist()

        avg_loss = total_loss / max(n_batches, 1)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        mean_concepts = n_pred_concepts / max(n_samples, 1)

        return avg_loss, f1, precision, recall, mean_concepts, threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--resume", default=None, metavar="PATH")
    parser.add_argument("--no-scheduler", action="store_true",
                         help="Disable LR scheduler; use fixed lr_backbone / lr_head throughout")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

    cfg = load_config(args.config)
    t_cfg = cfg["training"]

    if not args.local and t_cfg.get("train_batches_per_epoch") is None:
        raise SystemExit(
            "training.train_batches_per_epoch is null/unset in the config, but the "
            "train split streams infinitely (see dataset.py's .repeat()). Set it to "
            "a concrete integer approximating one pass over the training set: "
            "n_train_images / (data.batch_size * num_gpus). Check the 'Total samples' "
            "line of data/fit_label_binarizer.py for n_train_images."
        )

    # Device & DDP
    if args.local or not torch.cuda.is_available():
        is_ddp = False
        rank = 0
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        use_amp = False
    else:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_ddp = True
        use_amp = True

    set_seeds(t_cfg["seed"] + rank)

    # Data
    train_loader = build_dataloader("train", cfg, local=args.local)
    val_loader = build_dataloader("val", cfg, local=args.local)

    # Model
    model = build_model(cfg).to(device)
    if rank == 0:
        pc = count_parameters(model)
        print(f"Backbone: {cfg['model']['backbone']}  |  Head: {cfg['model']['head_type']}")
        print(f"Params -- backbone: {pc['backbone']:,}  head: {pc['head']:,}  total: {pc['total']:,}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    criterion, get_probs = build_criterion(cfg, device)
    if rank == 0:
        print(f"Loss: {cfg['loss']['type']}")
    optimizer = build_optimizer(model, cfg)
    scheduler = None if args.no_scheduler else build_scheduler(optimizer, cfg)

    # Resume
    ckpt_dir = cfg["paths"]["checkpoints"]
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    metric = t_cfg.get("early_stop_metric", "val_loss")
    higher_is_better = metric == "val_f1"

    start_epoch = 0
    best_score = float("-inf") if higher_is_better else float("inf")
    no_improve = 0

    resume_path = args.resume or (last_ckpt if os.path.exists(last_ckpt) else None)
    if resume_path:
        if rank == 0:
            print(f"Resuming from {resume_path}")
        start_epoch, ckpt_best_score, no_improve, ckpt_metric = load_checkpoint(
            resume_path, model, optimizer, scheduler, device, cfg=cfg
        )
        if ckpt_metric != metric:
            if rank == 0:
                print(f"  WARNING: resuming a checkpoint saved with early_stop_metric={ckpt_metric!r} "
                      f"but the current config specifies {metric!r}. best_score isn't comparable across "
                      f"metrics -- starting a fresh best_score={best_score} and no_improve=0 instead of "
                      f"the checkpoint's; epoch/model/optimizer/scheduler state still resumed normally.")
            no_improve = 0
        else:
            best_score = ckpt_best_score

    # TensorBoard
    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = cfg["paths"]["logs"]
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

    max_epochs = t_cfg["local_dev_epochs"] if args.local else t_cfg["epochs"]
    max_train_b = t_cfg["local_dev_max_train_batches"] if args.local else t_cfg.get("train_batches_per_epoch")
    max_val_b = t_cfg["local_dev_max_val_batches"] if args.local else None
    patience = t_cfg["early_stopping_patience"]

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{max_epochs}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg, use_amp, rank,
            max_batches=max_train_b,
        )

        val_loss, val_f1, val_prec, val_rec, mean_concepts, val_threshold = val_epoch(
            model, val_loader, criterion, get_probs, device, cfg, use_amp, is_ddp, rank,
            max_batches=max_val_b,
        )

        if is_ddp:
            dist.barrier()

        if scheduler is not None:
            metric_value = val_f1 if higher_is_better else val_loss
            scheduler_step(scheduler, cfg["training"].get("scheduler_type", "reduce_on_plateau"), metric_value)

        if rank == 0:
            lr_bb = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[2]["lr"]

            print(
                f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"val_f1={val_f1:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f} "
                f"val_threshold={val_threshold:.2f} mean_concepts={mean_concepts:.2f} | "
                f"lr_bb={lr_bb:.2e} lr_head={lr_head:.2e}"
            )

            if writer:
                step = epoch + 1
                writer.add_scalar("loss/train", train_loss, step)
                writer.add_scalar("loss/val", val_loss, step)
                writer.add_scalar("f1/val", val_f1, step)
                writer.add_scalar("precision/val", val_prec, step)
                writer.add_scalar("recall/val", val_rec, step)
                writer.add_scalar("val/threshold", val_threshold, step)
                writer.add_scalar("val/mean_concepts", mean_concepts, step)
                writer.add_scalar("lr/backbone", lr_bb, step)
                writer.add_scalar("lr/head", lr_head, step)

            score = val_f1 if higher_is_better else val_loss
            improved = (score > best_score) if higher_is_better else (score < best_score)

            if improved:
                best_score = score
                no_improve = 0
                if higher_is_better:
                    best_name = (
                        f"{run_tag(cfg)}__seed{t_cfg['seed']}__epoch{epoch+1:02d}"
                        f"__f1_{val_f1:.4f}__thr{val_threshold:.2f}.pt"
                    )
                else:
                    best_name = (
                        f"{run_tag(cfg)}__seed{t_cfg['seed']}__epoch{epoch+1:02d}__loss_{val_loss:.4f}.pt"
                    )
                save_checkpoint(
                    os.path.join(ckpt_dir, best_name),
                    model, optimizer, scheduler, epoch, best_score, no_improve, cfg, metric=metric,
                )
                print(f"  Saved best checkpoint -> {best_name}")
            else:
                no_improve += 1

            save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best_score, no_improve, cfg, metric=metric)

        if is_ddp:
            stop = torch.tensor(int(no_improve >= patience), device=device)
            dist.broadcast(stop, src=0)
            should_stop = stop.item() == 1
        else:
            should_stop = no_improve >= patience

        if should_stop:
            if rank == 0:
                print(f"Early stopping: no improvement for {patience} epochs.")
            break

    if writer:
        writer.close()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
