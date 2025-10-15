# -*- coding: utf-8 -*-
#Autor: Leandro de Mattos Pereira
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
faal_adenyl_metrics.py
Compute Sørensen–Dice (per protein) for FAALPred and AdenylPred and
produce a single figure with two side-by-side boxplots (publication-ready).

Usage:
  python faal_adenyl_metrics.py --input ./input_table.tsv --outdir ./results --dpi 900
"""

import argparse
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------- Parsing helpers -----------------
def ints_from_c_tokens(text: str) -> List[int]:
    """Extract integers after 'C' (e.g., 'C12:0' -> 12, 'C10' -> 10)."""
    return [int(x) for x in re.findall(r'C\s*(\d+)', text or "", flags=re.IGNORECASE)]

def expand_range(a: int, b: int) -> Set[int]:
    """Inclusive integer range between a and b (order-agnostic)."""
    if a > b:
        a, b = b, a
    return set(range(a, b + 1))

def parse_experimental_set(text: str) -> Set[int]:
    """Build E from 'Substrate in literature' by expanding ranges and collecting discrete tokens."""
    E: Set[int] = set()
    s = (text or "").replace("\u2013", "-").replace("\u2014", "-")
    # Ranges like "C10–C18", "C10:0 to C18:0", "C10 through C18"
    for m in re.finditer(r'C\s*(\d+)\s*(?::0)?\s*(?:-|to|through)\s*C?\s*(\d+)', s, flags=re.IGNORECASE):
        a, b = int(m.group(1)), int(m.group(2))
        E |= expand_range(a, b)
    # Discrete tokens
    E |= set(ints_from_c_tokens(s))
    return E

def parse_faal_set(text: str) -> Set[int]:
    """FAALPred discrete predictions like 'C12; C14; C16' -> {12,14,16}."""
    return set(ints_from_c_tokens(text or ""))

def parse_adenyl_set(text: str) -> Set[int]:
    """
    AdenylPred may return a range 'C13–C17' or discrete values.
    Expand ranges to full integer sets; use tokens directly otherwise.
    """
    s = (text or "").replace("\u2013", "-")
    r = re.search(r'C\s*(\d+)\s*(?:to|through|-)\s*C?\s*(\d+)', s, flags=re.IGNORECASE)
    if r:
        a, b = int(r.group(1)), int(r.group(2))
        return expand_range(a, b)
    return set(ints_from_c_tokens(s))

def dice(A: Set[int], B: Set[int]) -> float:
    """Sørensen–Dice = 2|A∩B| / (|A| + |B|)."""
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return (2.0 * inter) / (len(A) + len(B))


# ----------------- Robust reading & headers -----------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = s.strip().lower()
    s = re.sub(r'[\s_/,-]+', ' ', s)
    s = s.replace(':', '')
    s = re.sub(r'\s+', ' ', s)
    return s

SYNONYMS = {
    "protein":   ["protein", "protein name", "enzyme", "name"],
    "accession": ["refseq/genbank", "refseq genbank", "protein accession", "current accession", "accession", "refseq", "genbank"],
    "substrate": ["substrate in literature", "substrates in literature", "experimental substrate", "substrate"],
    "species":   ["species", "organism name", "organism", "strain"],
    "faalpred":  ["faalpred, prediction score", "faalpred prediction score", "faalpred", "faalpred score"],
    "adenylpred":["adenylpred, prediction score", "adenylpred prediction score", "adenylpred", "adenylpred score"],
}

def map_headers_strict(columns) -> Dict[str, Optional[str]]:
    norm_to_actual = {_norm(c): c for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    for key, cands in SYNONYMS.items():
        found = None
        for cand in cands:
            n = _norm(cand)
            if n in norm_to_actual:
                found = norm_to_actual[n]
                break
        mapping[key] = found
    return mapping

def ensure_columns(df: pd.DataFrame) -> Dict[str, str]:
    m = map_headers_strict(df.columns)
    missing = [k for k, v in m.items() if v is None]
    if missing:
        raise ValueError(
            "Could not find the required columns: {}.\nFound columns: {}\n"
            "Hint: final table must have exactly:\n"
            "Protein | Refseq/GenBank | Substrate in literature | Species | "
            "FAALPred, Prediction Score | AdenylPred, Prediction Score".format(missing, list(df.columns))
        )
    return m  # type: ignore[return-value]

def read_table_robust(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".xls") or lower.endswith(".xlsx"):
        df = pd.read_excel(path).fillna("")
        _ = ensure_columns(df)
        return df
    candidates = [
        {"sep": "\t", "engine": "python"},
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
    ]
    for cand in candidates:
        try:
            df = pd.read_csv(path, sep=cand["sep"], engine=cand["engine"], encoding="utf-8-sig").fillna("")
            m = map_headers_strict(df.columns)
            if any(v is None for v in m.values()):
                continue
            return df
        except Exception:
            continue
    raise ValueError("Could not read table with a valid delimiter and header mapping.")


# ----------------- Figure (two side-by-side boxplots) -----------------
def save_boxplot_side_by_side(
    faal_values: np.ndarray,
    aden_values: np.ndarray,
    title: str,
    out_no_ext: str,
    dpi: int
) -> None:
    # Publication-friendly typography
    plt.rcParams.update({
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(6.2, 4.8))  # good for single-column width
    data = [faal_values[~np.isnan(faal_values)], aden_values[~np.isnan(aden_values)]]

    # Colorblind-friendly palette with good print contrast
    facecolors = ["#2F78C4", "#E07B39"]  # blue, orange
    edgecolors = ["#1F4F83", "#9B4F22"]

    bp = ax.boxplot(
        data,
        notch=True,
        vert=True,
        patch_artist=True,
        widths=0.6,
        whis=1.5,
        showfliers=True
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(facecolors[i])
        patch.set_edgecolor(edgecolors[i])
        patch.set_alpha(0.88)
        patch.set_linewidth(1.6)

    for whisker in bp["whiskers"]:
        whisker.set_color("#333333")
        whisker.set_linewidth(1.2)

    for cap in bp["caps"]:
        cap.set_color("#333333")
        cap.set_linewidth(1.2)

    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(2.2)

    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markerfacecolor("#555555")
        flier.set_markeredgecolor("#333333")
        flier.set_alpha(0.55)
        flier.set_markersize(4)

    ax.set_xticklabels(["FAALPred", "AdenylPred"])
    ax.set_ylabel("Sørensen–Dice (per protein)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_no_ext + ".svg", format="svg")
    fig.savefig(out_no_ext + ".png", format="png", dpi=dpi)
    try:
        fig.savefig(out_no_ext + ".tiff", format="tiff", dpi=dpi)
    except Exception:
        fig.savefig(out_no_ext + ".tif", format="tiff", dpi=dpi)
    plt.close(fig)


# ----------------- Core -----------------
def run(input_path: str, outdir: str, dpi: int):
    os.makedirs(outdir, exist_ok=True)

    df = read_table_robust(input_path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    cols = ensure_columns(df)

    dice_faal: List[float] = []
    dice_aden: List[float] = []

    for _, row in df.iterrows():
        E = parse_experimental_set(row[cols["substrate"]])
        P_faal = parse_faal_set(row[cols["faalpred"]])
        P_aden = parse_adenyl_set(row[cols["adenylpred"]])

        dice_faal.append(dice(P_faal, E))
        dice_aden.append(dice(P_aden, E))

    arr_faal = np.array(dice_faal, dtype=float)
    arr_aden = np.array(dice_aden, dtype=float)

    # Save single figure with two side-by-side boxplots
    save_boxplot_side_by_side(
        faal_values=arr_faal,
        aden_values=arr_aden,
        title="FAALPred vs AdenylPred — Sørensen–Dice (per protein)",
        out_no_ext=os.path.join(outdir, "Dice_boxplots_side_by_side"),
        dpi=dpi
    )

    # Save per-case table (only Dice)
    pd.DataFrame({
        "Dice_FAALPred": arr_faal,
        "Dice_AdenylPred": arr_aden
    }).to_csv(os.path.join(outdir, "per_case_dice.tsv"), sep="\t", index=False)

    # Optional: print macro means to stdout (no files/plots for these)
    macro_faal = float(np.nanmean(arr_faal)) if arr_faal.size else float("nan")
    macro_aden = float(np.nanmean(arr_aden)) if arr_aden.size else float("nan")
    print({"FAALPred_Macro_Dice": macro_faal, "AdenylPred_Macro_Dice": macro_aden})
    print("Saved:",
          os.path.join(outdir, "Dice_boxplots_side_by_side.{svg,png,tiff}"),
          os.path.join(outdir, "per_case_dice.tsv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input TSV/CSV/XLSX (final 6-column table)")
    parser.add_argument("--outdir", "-o", default=".", help="Output directory (default: .)")
    parser.add_argument("--dpi", type=int, default=900, help="DPI for PNG/TIFF (default: 900)")
    args = parser.parse_args()
    run(args.input, args.outdir, dpi=args.dpi)

if __name__ == "__main__":
    main()

