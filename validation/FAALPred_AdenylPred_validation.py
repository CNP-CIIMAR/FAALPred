#!/usr/bin/env python3
"""
faal_adenyl_confusion.py

Script to:
  1. Read a table with FAALPred / AdenylPred results and literature substrates.
  2. Convert literature substrates into "true intervals" compatible with FAALPred
     and AdenylPred (based on overlap of carbon-chain ranges).
  3. Build confusion matrices (true vs predicted intervals).
  4. Plot publication-quality confusion-matrix heatmaps:
       - Axes in English: "True label" (y), "Predicted label" (x)
       - X tick labels vertical
       - Cell values as row-wise percentages (each row sums to 100%)
       - Export as TIFF, PNG (600 dpi) and SVG.
  5. Compute per-class metrics: Sensitivity, Specificity, Precision, F1.
  6. Create a single 2x2 panel figure:
       Panel A: FAALPred confusion matrix
       Panel B: FAALPred per-class metrics (bar plot)
       Panel C: AdenylPred confusion matrix
       Panel D: AdenylPred per-class metrics (bar plot)

Expected columns in the input table (TSV/CSV):
  - "Protein"
  - "Refseq/GenBank"
  - "Substrate in literature"
  - "Species"
  - "FAALPred, Prediction Score"
  - "AdenylPred, Prediction Score"

Usage example:
  python faal_adenyl_confusion.py -i data.tsv -o results --sep "\\t"

Author: (coloque o seu nome aqui)
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend não interativo (evita erros de Qt/xcb em servidores)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configurações globais de figura
# ---------------------------------------------------------------------
DPI_FIG = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


# ---------------------------------------------------------------------
# 1. Parsing da literatura: extrair números de carbono
# ---------------------------------------------------------------------
def extract_carbons(substrate: str) -> List[int]:
    """
    Extract all carbon chain lengths from the 'Substrate in literature' string.

    Examples:
      "C8:0, C10:0, C12:0 (mainly C:10)" -> [8, 10, 12]
      "C12:0, C14:0, C16:0, C48:0 to C62:0" -> [12, 14, 16, 48, 62]

    We deliberately ignore 'mainly', 'preferred', etc.
    """
    if pd.isna(substrate):
        return []
    # Matches C10, C10:0, C:10 etc.; captures only the number
    nums = re.findall(r"C:?(\d+)", str(substrate))
    return [int(n) for n in nums]


# ---------------------------------------------------------------------
# 2. Escolher intervalo verdadeiro compatível com cada método
# ---------------------------------------------------------------------
def choose_interval(
    carbons: List[int],
    ranges_dict: Dict[str, Tuple[int, int]],
    low_clip: Optional[int] = None,
    high_clip: Optional[int] = None,
) -> Optional[str]:
    """
    Map a set of carbon lengths from the literature to the closest interval in
    ranges_dict, based on *discrete overlap*.

    - carbons: list of carbon chain lengths from the literature, e.g. [8, 10, 12]
    - ranges_dict: e.g. for FAALPred:
          {
            "C4-C6-C8": (4, 8),
            "C8-C10-C12": (8, 12),
            "C12-C14-C16": (12, 16),
            "C14-C16-C18": (14, 18),
          }
    - low_clip / high_clip: optional limits for carbons to keep (e.g. 4–18
      for FAALPred, 6–17 for AdenylPred). Values outside this range are dropped
      before computing the overlap.

    Overlap score:
      We treat chain lengths as discrete integers and compute overlap as the
      number of integers in common between [lit_min, lit_max] and [rmin, rmax]:

        overlap = max(0, inter_max - inter_min + 1)

      where:
        inter_min = max(lit_min, rmin)
        inter_max = min(lit_max, rmax)

    The label whose interval has the largest overlap is returned.
    If carbons is empty or nothing remains after clipping, returns None.
    """
    if not carbons:
        return None

    # Clip to the relevant range for this method, if requested
    if low_clip is not None or high_clip is not None:
        filtered = [
            c
            for c in carbons
            if (low_clip is None or c >= low_clip)
            and (high_clip is None or c <= high_clip)
        ]
        if filtered:
            carbons = filtered
        else:
            # All carbons are outside method range
            return None

    lit_min, lit_max = min(carbons), max(carbons)

    best_label = None
    best_overlap = -1.0

    for label, (rmin, rmax) in ranges_dict.items():
        inter_min = max(lit_min, rmin)
        inter_max = min(lit_max, rmax)
        if inter_max >= inter_min:
            overlap = inter_max - inter_min + 1  # discrete number of integers
        else:
            overlap = 0

        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    return best_label


# ---------------------------------------------------------------------
# 3. Métricas por classe: sensibilidade, especificidade, F1
# ---------------------------------------------------------------------
def per_class_metrics(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class metrics from a confusion matrix DataFrame.

    The confusion matrix must have:
      - rows = true labels
      - columns = predicted labels

    For each class c (over the union of rows and columns), we compute:
      - TP, FP, FN, TN
      - Sensitivity (Recall)
      - Specificity
      - Precision
      - F1-score

    Returns a DataFrame indexed by class label.
    """
    # Ensure all labels appear in both rows and columns
    labels = sorted(set(cm.index) | set(cm.columns))
    cm_full = cm.reindex(index=labels, columns=labels, fill_value=0)

    total = cm_full.values.sum()
    metrics = []

    for c in labels:
        TP = cm_full.loc[c, c]
        FN = cm_full.loc[c, :].sum() - TP
        FP = cm_full.loc[:, c].sum() - TP
        TN = total - (TP + FN + FP)

        # Avoid division by zero
        sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else np.nan

        metrics.append(
            {
                "Class": c,
                "TP": TP,
                "FN": FN,
                "FP": FP,
                "TN": TN,
                "Sensitivity": sens,
                "Specificity": spec,
                "Precision": prec,
                "F1": f1,
            }
        )

    return pd.DataFrame(metrics).set_index("Class")


# ---------------------------------------------------------------------
# 4. Plot da matriz de confusão em porcentagem (em um eixo)
# ---------------------------------------------------------------------
def plot_confusion_matrix_percent_ax(
    cm: pd.DataFrame,
    title: str,
    ax: plt.Axes,
):
    """
    Plot a confusion matrix as percentages (row-wise) with annotations
    on a given Matplotlib axis.

    - cm: confusion matrix (counts), rows=true labels, cols=predicted labels
    - title: title for the plot
    - ax: Matplotlib axis on which to draw
    """
    # Ensure fixed ordering
    cm = cm.sort_index(axis=0).sort_index(axis=1)

    # Row-wise percent
    row_sums = cm.sum(axis=1).replace(0, np.nan)
    cm_pct = cm.div(row_sums, axis=0) * 100.0
    cm_pct = cm_pct.fillna(0.0)

    im = ax.imshow(cm_pct.values, vmin=0, vmax=100, cmap="viridis")

    # Ticks and labels
    ax.set_xticks(np.arange(cm_pct.shape[1]))
    ax.set_yticks(np.arange(cm_pct.shape[0]))
    # Confusion matrix: labels do eixo x em 90° (vertical, como pedido antes)
    ax.set_xticklabels(cm_pct.columns, rotation=90, ha="center")
    ax.set_yticklabels(cm_pct.index)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, pad=10)

    # Annotate percentages in each cell with color-contrast
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            value = cm_pct.values[i, j]
            text_color = "white" if value > 50 else "black"
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    # Colorbar (local to this axis)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-wise percentage", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------
# 5. Plot de barras das métricas por classe (em um eixo)
# ---------------------------------------------------------------------
def plot_metrics_bars_ax(
    metrics_df: pd.DataFrame,
    model_name: str,
    ax: plt.Axes,
):
    """
    Plot per-class bar charts of Sensitivity, Specificity and F1 on a given axis.

    - metrics_df: DataFrame returned by per_class_metrics()
    - model_name: string used in the figure title (e.g. 'FAALPred')
    - ax: Matplotlib axis on which to draw

    The figure has:
      - x-axis: classes
      - y-axis: metric value (0–1)
      - three bars per class: Sensitivity, Specificity, F1
      - legend placed on the right side of the plot
    """
    metrics_to_plot = ["Sensitivity", "Specificity", "F1"]

    # Ensure numeric
    plot_df = metrics_df[metrics_to_plot].astype(float)

    classes = plot_df.index.tolist()
    x = np.arange(len(classes))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]  # azul, verde, vermelho suave

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        values = plot_df[metric].values
        offset = (i - 1) * width  # positions: -width, 0, +width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        # Annotate each bar with the metric value (inside or above)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            height = bar.get_height()
            # Se a barra é alta, escreve dentro; se baixa, escreve acima
            ypos = height - 0.06 if height > 0.15 else height + 0.02
            va = "top" if height > 0.15 else "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                ypos,
                f"{v:.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.set_xticks(x)
    # Aqui padronizamos para 45° no eixo x (barplots)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)

    ax.set_ylabel("Score")
    ax.set_xlabel("Class")
    ax.set_title(f"Per-class performance – {model_name}", pad=10)

    # Grade leve no eixo y
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Legenda à direita do gráfico, fora da área das barras
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )


# ---------------------------------------------------------------------
# 6. Painel único 2x2: A/B/C/D
# ---------------------------------------------------------------------
def plot_combined_panel(
    faal_cm: pd.DataFrame,
    adenyl_cm: pd.DataFrame,
    faal_metrics: pd.DataFrame,
    adenyl_metrics: pd.DataFrame,
    output_prefix: str,
):
    """
    Create a single 2x2 panel figure with:

      A: FAALPred confusion matrix
      B: FAALPred per-class metrics
      C: AdenylPred confusion matrix
      D: AdenylPred per-class metrics

    and save as PNG, TIFF and SVG at high resolution.
    """
    # Figura um pouco mais larga e alta para acomodar legendas e labels
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # ~ 25.4 x 20.3 cm

    axA = axes[0, 0]
    axB = axes[0, 1]
    axC = axes[1, 0]
    axD = axes[1, 1]

    # Panel A: FAALPred confusion matrix
    plot_confusion_matrix_percent_ax(
        faal_cm,
        title="Confusion matrix – FAALPred",
        ax=axA,
    )

    # Panel B: FAALPred metrics
    plot_metrics_bars_ax(
        faal_metrics,
        model_name="FAALPred",
        ax=axB,
    )

    # Panel C: AdenylPred confusion matrix
    plot_confusion_matrix_percent_ax(
        adenyl_cm,
        title="Confusion matrix – AdenylPred",
        ax=axC,
    )

    # Panel D: AdenylPred metrics
    plot_metrics_bars_ax(
        adenyl_metrics,
        model_name="AdenylPred",
        ax=axD,
    )

    # Panel labels A, B, C, D (no canto superior esquerdo de cada subplot)
    panel_labels = ["A", "B", "C", "D"]
    for label, ax in zip(panel_labels, [axA, axB, axC, axD]):
        ax.text(
            -0.15,
            1.12,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Ajustar espaçamento:
    # - left maior (labels y)
    # - bottom maior (labels x)
    # - hspace maior (mais espaço entre A/B e C/D)
    # - right < 1 para acomodar as legendas à direita dos barplots
    plt.subplots_adjust(
        left=0.14,
        right=0.86,   # deixa espaço à direita para legendas B e D
        top=0.93,
        bottom=0.14,
        wspace=0.55,
        hspace=0.60,  # mais espaço entre a linha de cima e a de baixo
    )

    png_path = f"{output_prefix}_combined_panel.png"
    tiff_path = f"{output_prefix}_combined_panel.tiff"
    svg_path = f"{output_prefix}_combined_panel.svg"

    fig.savefig(png_path, dpi=DPI_FIG)
    fig.savefig(tiff_path, dpi=DPI_FIG)
    fig.savefig(svg_path)

    plt.close(fig)

    print(
        f"\nSaved combined panel figure (A–D) to:\n"
        f"  {png_path}\n  {tiff_path}\n  {svg_path}"
    )


# ---------------------------------------------------------------------
# 7. Função principal
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build FAALPred/AdenylPred confusion matrices and per-class metrics."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_table",
        required=True,
        help="Path to input table (TSV/CSV) with substrates and predictions.",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Field separator for the input table (default: '\\t' for TSV).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=".",
        help="Directory to save output files (default: current directory).",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 7.1 Prepare output directory and base prefix
    # -------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_table))[0]
    output_prefix_base = os.path.join(args.output_dir, base_name)

    # -------------------------------------------------------------------
    # 7.2 Ler a tabela
    # -------------------------------------------------------------------
    df = pd.read_csv(args.input_table, sep=args.sep)

    # Normalizar nomes de coluna (tirar espaços extras)
    df.columns = [c.strip() for c in df.columns]

    # Mapear nomes "feios" para nomes internos mais simples
    COLUMN_MAP = {
        "Refseq/GenBank": "Refseq_GenBank",
        "Substrate in literature": "Substrate_lit",
        "FAALPred, Prediction Score": "FAALPred_raw",
        "AdenylPred, Prediction Score": "AdenylPred_raw",
    }
    df = df.rename(columns=COLUMN_MAP)

    required = ["Substrate_lit", "FAALPred_raw", "AdenylPred_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input table: {missing}")

    # -------------------------------------------------------------------
    # 7.3 Limpar intervalos previstos
    # -------------------------------------------------------------------
    df["FAALPred_interval"] = df["FAALPred_raw"].str.split("(").str[0].str.strip()
    df["AdenylPred_interval"] = df["AdenylPred_raw"].str.split("(").str[0].str.strip()

    # -------------------------------------------------------------------
    # 7.4 Extrair tamanhos de carbono da literatura
    # -------------------------------------------------------------------
    df["lit_carbons"] = df["Substrate_lit"].apply(extract_carbons)

    # -------------------------------------------------------------------
    # 7.5 Definir bins de FAALPred e AdenylPred e mapear intervalos verdadeiros
    # -------------------------------------------------------------------
    faal_ranges = {
        "C4-C6-C8": (4, 8),
        "C8-C10-C12": (8, 12),
        "C12-C14-C16": (12, 16),
        "C14-C16-C18": (14, 18),
    }

    adenyl_ranges = {
        "C6 through C12": (6, 12),
        "C13 through C17": (13, 17),
    }

    df["True_FAAL_interval"] = df["lit_carbons"].apply(
        lambda cs: choose_interval(cs, faal_ranges, low_clip=4, high_clip=18)
    )

    df["True_Adenyl_interval"] = df["lit_carbons"].apply(
        lambda cs: choose_interval(cs, adenyl_ranges, low_clip=6, high_clip=17)
    )

    # -------------------------------------------------------------------
    # 7.6 Construir matrizes de confusão (contagens)
    # -------------------------------------------------------------------
    faal_cm = pd.crosstab(df["True_FAAL_interval"], df["FAALPred_interval"])
    adenyl_cm = pd.crosstab(df["True_Adenyl_interval"], df["AdenylPred_interval"])

    print("\n=== Confusion matrix (FAALPred) - counts ===")
    print(faal_cm)

    print("\n=== Confusion matrix (AdenylPred) - counts ===")
    print(adenyl_cm)

    # -------------------------------------------------------------------
    # 7.7 Métricas por classe
    # -------------------------------------------------------------------
    faal_metrics = per_class_metrics(faal_cm)
    adenyl_metrics = per_class_metrics(adenyl_cm)

    print("\n=== Per-class metrics (FAALPred) ===")
    print(faal_metrics)

    print("\n=== Per-class metrics (AdenylPred) ===")
    print(adenyl_metrics)

    # Save metrics to CSV for later use
    faal_metrics_csv = f"{output_prefix_base}_faal_metrics.csv"
    adenyl_metrics_csv = f"{output_prefix_base}_adenyl_metrics.csv"
    faal_metrics.to_csv(faal_metrics_csv)
    adenyl_metrics.to_csv(adenyl_metrics_csv)
    print(f"\nSaved metrics to:\n  {faal_metrics_csv}\n  {adenyl_metrics_csv}")

    # -------------------------------------------------------------------
    # 7.8 Criar figura única A–D (600 dpi)
    # -------------------------------------------------------------------
    plot_combined_panel(
        faal_cm=faal_cm,
        adenyl_cm=adenyl_cm,
        faal_metrics=faal_metrics,
        adenyl_metrics=adenyl_metrics,
        output_prefix=output_prefix_base,
    )


if __name__ == "__main__":
    main()
