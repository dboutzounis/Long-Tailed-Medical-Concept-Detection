"""PCA dimensionality ablation of the BiomedCLIP image embedding (Section 4.7, Table 5.17).

The cached last-layer CLS tokens are passed through the final LayerNorm and image projection
(512-d), PCA is fitted on the training split, and for each d in {512, ..., 8} an MLP probe is
trained with ASL on the first d components. A global threshold is selected on val and the test
F1 is reported per taxonomy category.

Usage:
    python pca_ablation.py --data-dir LAYER_DIR --mlb-path mlb.pkl \
        --json-path ../../data/taxonomy/concept_classifications.json --output-prefix dim_ablation --min-samples 0
"""

import argparse
import copy
import json
import os
import pickle

import matplotlib
import open_clip
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.02, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


def forward_projection(embs, model):
    """Final LayerNorm + image projection of BiomedCLIP: 768-d CLS token -> 512-d, L2-normalized."""
    final_embs = embs
    if final_embs.dim() == 3:
        final_embs = final_embs[:, 0, :]

    if hasattr(model.visual, "trunk"):
        if hasattr(model.visual.trunk, "norm") and model.visual.trunk.norm is not None:
            final_embs = model.visual.trunk.norm(final_embs)
        if hasattr(model.visual, "head") and model.visual.head is not None:
            final_embs = model.visual.head(final_embs)
    else:
        if hasattr(model.visual, "ln_post"):
            final_embs = model.visual.ln_post(final_embs)
        if hasattr(model.visual, "proj") and model.visual.proj is not None:
            final_embs = final_embs @ model.visual.proj

    return final_embs / final_embs.norm(dim=-1, keepdim=True)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_targets, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mlb-path", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--output-prefix", default="dim_ablation")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PCA dimensionality ablation on {device}")

    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    model.eval()

    with open(args.mlb_path, "rb") as f:
        mlb = pickle.load(f)
    cui_order = list(mlb.classes_)

    with open(args.json_path, "r") as f:
        raw_data = json.load(f)

    concept_categories = {}
    for item in raw_data.get("classifications", []):
        cui = str(item.get("cui", "")).strip().upper()
        if cui:
            cat = str(item.get("category", "")).strip().lower()
            if "anatomy" in cat:
                cat = "Anatomy"
            elif "pathology" in cat:
                cat = "Pathology"
            elif "finding" in cat:
                cat = "Radiological Finding"
            elif "modality" in cat:
                cat = "Imaging Modality"
            elif "device" in cat:
                cat = "Device & Intervention"
            elif "clinical context" in cat:
                cat = "Clinical Context"
            else:
                cat = "Other"
            concept_categories[cui] = cat

    print("Loading pre-computed layer 11 embeddings (train, val, test)...")
    train_embs = torch.load(
        os.path.join(args.data_dir, "train_layer_11_embs.pt"), map_location="cpu", weights_only=True
    ).float()
    train_labels = torch.load(
        os.path.join(args.data_dir, "train_labels.pt"), map_location="cpu", weights_only=True
    )
    val_embs = torch.load(
        os.path.join(args.data_dir, "val_layer_11_embs.pt"), map_location="cpu", weights_only=True
    ).float()
    val_labels = torch.load(
        os.path.join(args.data_dir, "val_labels.pt"), map_location="cpu", weights_only=True
    )
    test_embs = torch.load(
        os.path.join(args.data_dir, "test_layer_11_embs.pt"), map_location="cpu", weights_only=True
    ).float()
    test_labels = torch.load(
        os.path.join(args.data_dir, "test_labels.pt"), map_location="cpu", weights_only=True
    )

    valid_cols = []
    target_categories = []

    for col, cui in enumerate(cui_order):
        cui = str(cui).strip().upper()
        if cui in concept_categories and concept_categories[cui] != "Other":
            if (
                val_labels[:, col].sum() >= args.min_samples
                and train_labels[:, col].sum() >= args.min_samples
                and test_labels[:, col].sum() >= args.min_samples
            ):
                valid_cols.append(col)
                target_categories.append(concept_categories[cui])

    num_targets = len(valid_cols)
    print(f"Evaluating {num_targets} concepts across {len(set(target_categories))} categories.")

    Y_train = train_labels[:, valid_cols].float()
    Y_val = val_labels[:, valid_cols].float()
    Y_test = test_labels[:, valid_cols].float()

    print("Projecting embeddings to multimodal 512d space...")
    with torch.no_grad():

        def batch_project(embs, batch_size=2048):
            out = []
            for i in range(0, len(embs), batch_size):
                batch = embs[i : i + batch_size].to(device)
                proj = forward_projection(batch, model)
                out.append(proj.cpu())
            return torch.cat(out, dim=0)

        X_train_512 = batch_project(train_embs)
        X_val_512 = batch_project(val_embs)
        X_test_512 = batch_project(test_embs)

    del train_embs, val_embs, test_embs

    print("Fitting PCA on Training data...")
    pca = PCA(n_components=512, random_state=42, svd_solver="randomized")
    X_train_pca = torch.tensor(pca.fit_transform(X_train_512.numpy())).float()
    X_val_pca = torch.tensor(pca.transform(X_val_512.numpy())).float()
    X_test_pca = torch.tensor(pca.transform(X_test_512.numpy())).float()

    dimensions = [512, 256, 128, 64, 32, 16, 8]
    results = []

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.02)

    for d in dimensions:
        print(f"\nd = {d}")
        X_train_d = X_train_pca[:, :d]
        X_train_d = X_train_d / X_train_d.norm(dim=-1, keepdim=True)

        X_val_d = X_val_pca[:, :d]
        X_val_d = X_val_d / X_val_d.norm(dim=-1, keepdim=True)

        X_test_d = X_test_pca[:, :d]
        X_test_d = X_test_d / X_test_d.norm(dim=-1, keepdim=True)

        train_loader = DataLoader(
            TensorDataset(X_train_d, Y_train), batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_d, Y_val), batch_size=args.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test_d, Y_test), batch_size=args.batch_size, shuffle=False
        )

        mlp_probe = MLPProbe(input_dim=d, num_targets=num_targets, hidden_dim=512).to(device)
        optimizer = optim.Adam(mlp_probe.parameters(), lr=1e-3)

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(args.max_epochs):
            mlp_probe.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = mlp_probe(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            mlp_probe.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    out = mlp_probe(batch_x)
                    loss = criterion(out, batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = copy.deepcopy(mlp_probe.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(
                    f"  Early stopping triggered at epoch {epoch + 1} (Val Loss: {best_val_loss:.4f})"
                )
                break

        if patience_counter < args.patience:
            print(f"  Finished {args.max_epochs} epochs (Val Loss: {best_val_loss:.4f})")

        mlp_probe.load_state_dict(best_weights)
        mlp_probe.eval()

        # one global threshold, selected on val
        print("  Sweeping for global threshold on validation set...")
        val_logits = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                val_logits.append(mlp_probe(batch_x.to(device)).cpu())

        val_probs = torch.sigmoid(torch.cat(val_logits, dim=0)).numpy()
        y_val_np = Y_val.numpy()

        best_f1 = 0
        best_th = 0.5
        thresholds_to_test = [x / 100.0 for x in range(1, 100, 2)]

        for th in thresholds_to_test:
            pred = (val_probs >= th).astype(int)
            f1 = f1_score(y_val_np, pred, average="samples", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        print(f"  Selected global threshold: {best_th:.2f} (Val F1: {best_f1:.4f})")

        # test F1 per category, at the val threshold
        print("  Evaluating final metrics on test set...")
        test_logits = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                test_logits.append(mlp_probe(batch_x.to(device)).cpu())

        test_probs = torch.sigmoid(torch.cat(test_logits, dim=0)).numpy()
        y_test_np = Y_test.numpy()

        y_pred_test = (test_probs >= best_th).astype(int)

        cat_indices = {cat: [] for cat in set(target_categories)}
        for idx, cat in enumerate(target_categories):
            cat_indices[cat].append(idx)

        for cat, indices in cat_indices.items():
            y_t_cat = y_test_np[:, indices]
            y_p_cat = y_pred_test[:, indices]

            f1_samples = f1_score(y_t_cat, y_p_cat, average="samples", zero_division=0)
            p_samples = precision_score(y_t_cat, y_p_cat, average="samples", zero_division=0)
            r_samples = recall_score(y_t_cat, y_p_cat, average="samples", zero_division=0)

            results.append(
                {
                    "Dimensions": d,
                    "Category": cat,
                    "Samples_F1": f1_samples,
                    "Samples_Precision": p_samples,
                    "Samples_Recall": r_samples,
                }
            )
            print(f"    {cat} -> F1: {f1_samples:.4f} | P: {p_samples:.4f} | R: {r_samples:.4f}")

    df = pd.DataFrame(results)
    csv_out = f"{args.output_prefix}.csv"
    df.to_csv(csv_out, index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Dimensions", y="Samples_F1", hue="Category", marker="o", linewidth=2)
    plt.title("Test Set F1 Score Decay by Concept Category (Fixed Capacity Probe)")
    plt.xlabel("Vision Embedding Dimensions")
    plt.ylabel("Samples Average F1 Score")
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}.png", dpi=300)
    plt.close()

    print(f"\nAblation complete. Data saved to {csv_out} and plot to {args.output_prefix}.png")


if __name__ == "__main__":
    main()
