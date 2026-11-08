"""Layer-wise linear probes on BiomedCLIP by taxonomy category (Section 4.6.2, Figure 5.4).

One linear probe per (layer, category) pair, 12 x 6 = 72 probes, trained with ASL on the cached
CLS tokens of extract_layers.py. The epoch and threshold with the best validation F1 are kept
and the test split is evaluated once.

Usage:
    python probe_hierarchy.py --data-dir LAYER_DIR --mlb-path mlb.pkl \
        --json-path ../../data/taxonomy/concept_classifications.json --output-csv hierarchy_layer_f1.csv
"""

import argparse
import copy
import csv
import json
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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

        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1

        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        loss *= one_sided_w
        return -loss.mean()


class LinearProbe(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def calculate_metrics(logits, labels, threshold=0.5, eps=1e-8):
    """Sample-averaged F1, precision and recall at one threshold."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    tp = (preds * labels).sum(dim=1)
    fp = (preds * (1 - labels)).sum(dim=1)
    fn = ((1 - preds) * labels).sum(dim=1)

    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    return torch.nanmean(f1).item(), torch.nanmean(prec).item(), torch.nanmean(rec).item()


def find_best_threshold(logits, labels, thresholds):
    """Metrics at the threshold with the best F1."""
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thresh = 0.5

    for t in thresholds:
        f1, prec, rec = calculate_metrics(logits, labels, threshold=t)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thresh = t

    return best_f1, best_prec, best_rec, best_thresh


def train_and_eval(
    train_loader,
    val_loader,
    test_loader,
    input_dim,
    num_classes,
    device,
    epochs=15,
):
    model = LinearProbe(input_dim, num_classes).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_val_f1 = 0.0
    best_val_prec = 0.0
    best_val_rec = 0.0
    best_threshold = 0.5
    best_weights = None

    thresholds_to_test = [x / 100.0 for x in range(30, 100, 1)]

    for epoch in range(epochs):
        model.train()
        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits, all_targs = [], []
        with torch.no_grad():
            for embs, labels in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                all_logits.append(model(embs).cpu())
                all_targs.append(labels.cpu())

        epoch_f1, epoch_prec, epoch_rec, epoch_thresh = find_best_threshold(
            torch.cat(all_logits), torch.cat(all_targs), thresholds_to_test
        )

        if epoch_f1 > best_val_f1:
            best_val_f1 = epoch_f1
            best_val_prec = epoch_prec
            best_val_rec = epoch_rec
            best_threshold = epoch_thresh
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is None:
        best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    test_logits, test_targs = [], []
    with torch.no_grad():
        for embs, labels in test_loader:
            embs, labels = embs.to(device), labels.to(device)
            test_logits.append(model(embs).cpu())
            test_targs.append(labels.cpu())

    test_f1, test_prec, test_rec = calculate_metrics(
        torch.cat(test_logits), torch.cat(test_targs), threshold=best_threshold
    )

    return best_val_f1, best_val_prec, best_val_rec, test_f1, test_prec, test_rec, best_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mlb-path", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.mlb_path, "rb") as f:
        mlb = pickle.load(f)
    cui_order = list(mlb.classes_)

    with open(args.json_path, "r") as f:
        raw_data = json.load(f)

    classifications_list = raw_data.get("classifications", [])
    concept_map = {}
    for item in classifications_list:
        cui = str(item.get("cui", "")).strip().upper()
        cat = str(item.get("category", "")).strip().lower()
        if cui:
            concept_map[cui] = cat

    category_aliases = {
        "Modality": ["modality", "imaging modality"],
        "Anatomy": ["anatomy"],
        "Radiological Finding": ["radiological finding"],
        "Pathology": ["pathology"],
        "Device & Intervention": ["device", "device & intervention"],
        "Clinical Context": ["clinical context", "clinical"],
    }
    cat_indices = {key: [] for key in category_aliases.keys()}

    for idx, cui in enumerate(cui_order):
        cui_clean = str(cui).strip().upper()
        json_cat = concept_map.get(cui_clean, "")
        if json_cat:
            for target_cat, aliases in category_aliases.items():
                if any(alias in json_cat for alias in aliases):
                    cat_indices[target_cat].append(idx)
                    break

    train_labels = torch.load(os.path.join(args.data_dir, "train_labels.pt"), weights_only=True)
    val_labels = torch.load(os.path.join(args.data_dir, "val_labels.pt"), weights_only=True)
    test_labels = torch.load(os.path.join(args.data_dir, "test_labels.pt"), weights_only=True)

    results = []

    for layer_idx in range(12):
        layer_name = f"layer_{layer_idx}"
        print(f"\n--- Training Probes for {layer_name.upper()} ---")

        train_embs = torch.load(
            os.path.join(args.data_dir, f"train_{layer_name}_embs.pt"),
            weights_only=True,
        )
        val_embs = torch.load(
            os.path.join(args.data_dir, f"val_{layer_name}_embs.pt"),
            weights_only=True,
        )
        test_embs = torch.load(
            os.path.join(args.data_dir, f"test_{layer_name}_embs.pt"),
            weights_only=True,
        )
        input_dim = train_embs.shape[1]

        for cat in category_aliases.keys():
            idx_list = cat_indices[cat]
            if not idx_list:
                continue

            sliced_train_labels = train_labels[:, idx_list]
            sliced_val_labels = val_labels[:, idx_list]
            sliced_test_labels = test_labels[:, idx_list]

            train_loader = DataLoader(
                TensorDataset(train_embs, sliced_train_labels),
                batch_size=1024,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_embs, sliced_val_labels),
                batch_size=1024,
                shuffle=False,
            )
            test_loader = DataLoader(
                TensorDataset(test_embs, sliced_test_labels),
                batch_size=1024,
                shuffle=False,
            )

            val_f1, val_prec, val_rec, test_f1, test_prec, test_rec, best_thresh = train_and_eval(
                train_loader,
                val_loader,
                test_loader,
                input_dim,
                len(idx_list),
                device,
            )

            print(f"  {cat:<24}: (Threshold: {best_thresh:.2f})")
            print(f"    Val  | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
            print(f"    Test | F1: {test_f1:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f}")

            results.append(
                [
                    layer_idx,
                    cat,
                    best_thresh,
                    val_f1,
                    val_prec,
                    val_rec,
                    test_f1,
                    test_prec,
                    test_rec,
                ]
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Layer",
                "Category",
                "Best_Threshold",
                "Val_Samples_F1",
                "Val_Samples_Precision",
                "Val_Samples_Recall",
                "Test_Samples_F1",
                "Test_Samples_Precision",
                "Test_Samples_Recall",
            ]
        )
        writer.writerows(results)

    print(f"\nHierarchy evaluation complete! Data written to: {args.output_csv}")


if __name__ == "__main__":
    main()
