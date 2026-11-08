"""Plot the layer-wise probing results of probe_hierarchy.py (Figure 5.4).

Usage:
    python plot_probe_performance.py --csv-path hierarchy_layer_f1.csv --output-image probing_f1_layers.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_layer_performance(csv_path, output_image_path="probing_f1_layers.png"):
    df = pd.read_csv(csv_path)

    color_map = {
        "Modality": "#176BD7",
        "Anatomy": "#009E73",
        "Radiological Finding": "#E69F00",
        "Pathology": "#D55E00",
        "Device & Intervention": "#8E44AD",
        "Clinical Context": "#56B4E9",
    }

    marker_map = {
        "Modality": "o",
        "Anatomy": "s",
        "Radiological Finding": "^",
        "Pathology": "D",
        "Device & Intervention": "v",
        "Clinical Context": "P",
    }

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = [
        "Modality",
        "Anatomy",
        "Radiological Finding",
        "Pathology",
        "Device & Intervention",
        "Clinical Context",
    ]

    for cat in categories:
        cat_df = df[df["Category"] == cat].sort_values("Layer")
        if cat_df.empty:
            continue

        ax.plot(
            cat_df["Layer"],
            cat_df["Test_Samples_F1"],
            label=cat,
            color=color_map.get(cat, "#333333"),
            marker=marker_map.get(cat, "o"),
            markersize=7,
            linewidth=2.2,
            alpha=0.92,
        )

    ax.set_title(
        "BiomedCLIP Vision Encoder Layer-Wise Probing Performance",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Transformer Layer Index", fontsize=12, fontweight="semibold")
    ax.set_ylabel(
        "Linear Probe Samples-Averaged F1 Score (Test)", fontsize=12, fontweight="semibold"
    )

    ax.set_xticks(range(12))
    ax.set_xticklabels([f"Layer {i}" for i in range(12)], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(True, linestyle="--", alpha=0.5, color="#c0c0c0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Taxonomy Category",
        title_fontsize=11,
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="#dcdcdc",
        loc="center right",
    )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")

    pdf_path = output_image_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plots saved to:\n  - {output_image_path}\n  - {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, help="Path to probing results CSV")
    parser.add_argument(
        "--output-image",
        default="probing_f1_layers.png",
        help="Output image path",
    )
    args = parser.parse_args()
    plot_layer_performance(args.csv_path, args.output_image)
