# -*- coding: utf-8 -*-
#Autor: Leandro de Mattos Pereira
# -*- coding: utf-8 -*-
"""
FAALPred vs AdenylPred — Complementary analyses (CLI, robust headers, multi-format figures)

Usage:
  python faal_adenyl_metrics_cli.py --input ./input_table.tsv --outdir ./results --dpi 900

Expected input columns (6 columns; minor header variations are accepted):
  - Protein
  - Refseq/GenBank
  - Substrate in literature
  - Species
  - FAALPred, Prediction Score
  - AdenylPred, Prediction Score

Reported metrics:
  - FAALPred: Exact-Hit (prediction equals literature-preferred FA)
  - AdenylPred: Jaccard (set overlap) + Preferred-in-Bin

Outputs (written to --outdir; default = .):
  - per_case_metrics.tsv            (per-protein metrics in TSV)
  - summary_metrics.json            (full structure)
  - summary_metrics_minimal.json    (key metrics only)
  - AdenylPred_overlap_and_bin.svg/.png/.tiff
  - FAALPred_exact_hit.svg/.png/.tiff
"""

import argparse
import json
import re
import unicodedata
import os
from typing import List, Set, Optional, Tuple, Dict

# Headless backend (no GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# ----------------- Parsing helpers -----------------
def ints_from_c_tokens(s: str) -> List[int]:
    """Extract integers after 'C' (e.g., 'C12:0' -> 12, 'C10' -> 10)."""
    return [int(x) for x in re.findall(r'C\s*(\d+)', s, flags=re.IGNORECASE)]


def expand_range(a: int, b: int) -> Set[int]:
    """Return the inclusive integer range between a and b (order-agnostic)."""
    if a > b:
        a, b = b, a
    return set(range(a, b + 1))


def parse_experimental_set(s: str) -> Set[int]:
    """
    Build the experimental set E from text in “Substrate in literature”:
      - Expands ranges such as 'C10-C18', 'C10:0 to C18:0', 'C10 through C18'
      - Also includes discrete mentions such as 'C10', 'C12:0', etc.
    """
    E: Set[int] = set()
    s_norm = s.replace('\u2013', '-').replace('\u2014', '-')

    # Ranges
    for m in re.finditer(r'C\s*(\d+)\s*(?::0)?\s*(?:-|to|through)\s*C?\s*(\d+)', s_norm, flags=re.IGNORECASE):
        a, b = int(m.group(1)), int(m.group(2))
        E |= expand_range(a, b)

    # Discrete tokens
    E |= set(ints_from_c_tokens(s_norm))
    return E


def parse_preferred_fa(s: str, E: Set[int]) -> Optional[int]:
    """
    Extract the literature-preferred FA when indicated:
      - If the word 'preferred' appears, use the last range before it (rounded midpoint),
        or the first discrete mention before it.
      - Otherwise, if E is a singleton, return that single value; else None.
    """
    idx = s.lower().find('preferred')
    if idx != -1:
        left = s[:idx]
        m = None
        for m in re.finditer(r'C\s*(\d+)\s*(?::0)?\s*(?:-|to|through)\s*C?\s*(\d+)', left, flags=re.IGNORECASE):
            pass
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return int(round((a + b) / 2.0))
        nums = [int(x) for x in re.findall(r'C\s*(\d+)', left, flags=re.IGNORECASE)]
        if nums:
            return nums[0]

    if len(E) == 1:
        return list(E)[0]
    return None


def parse_faal_pred_set(s: str) -> Tuple[Set[int], Optional[float]]:
    """
    FAALPred: 'C12-C14-C16 (0.74)' -> discrete set {12,14,16} + float score (if present).
    """
    score = None
    m = re.search(r'\(([\d.]+)\)', s)
    if m:
        try:
            score = float(m.group(1))
        except Exception:
            score = None
    preds = set(ints_from_c_tokens(s))
    return preds, score


def parse_adenyl_pred_set(s: str) -> Tuple[Set[int], Optional[float], Optional[Tuple[int, int]]]:
    """
    AdenylPred: 'C13 through C17 (47%)' -> set {13..17}, score in [0,1], and bin (low, high).
    Accepts 'C13 through C17(47%)' (no space). If only discrete tokens appear,
    uses (min, max) as a pseudo-bin.
    """
    score = None
    m = re.search(r'\((\d+)\s*%?\)', s)
    if m:
        try:
            score = float(m.group(1)) / 100.0
        except Exception:
            score = None

    s_norm = s.replace('\u2013', '-')
    range_m = re.search(r'C\s*(\d+)\s*(?:to|through|-)\s*C?\s*(\d+)', s_norm, flags=re.IGNORECASE)
    if range_m:
        a, b = int(range_m.group(1)), int(range_m.group(2))
        aset = expand_range(a, b)
        return aset, score, (min(a, b), max(a, b))

    toks = set(ints_from_c_tokens(s_norm))
    if toks:
        lo, hi = min(toks), max(toks)
        return toks, score, (lo, hi)

    return set(), score, None


def choose_top_from_set(vals: Set[int]) -> Optional[int]:
    """Choose a deterministic 'top-1' from a set: the lower median."""
    if not vals:
        return None
    arr = sorted(vals)
    n = len(arr)
    mid = (n - 1) // 2
    return arr[mid]


def jaccard(P: Set[int], E: Set[int]) -> float:
    """Jaccard index between sets P and E."""
    if not P and not E:
        return 1.0
    if not P and E:
        return 0.0
    if P and not E:
        return 0.0
    inter = len(P & E)
    union = len(P | E)
    return inter / union if union > 0 else 0.0


# ----------------- Header normalization -----------------
def _norm_text(s: str) -> str:
    """Normalize header strings for robust matching."""
    s = unicodedata.normalize("NFKD", s)
    s = s.strip().lower()
    s = re.sub(r'[\s_/,-]+', ' ', s)
    s = s.replace(':', '').replace('%', '').replace('(', '').replace(')', '')
    s = re.sub(r'\s+', ' ', s)
    return s


def map_headers(cols) -> Dict[str, str]:
    """
    Map actual column names to canonical keys:
      protein, accession, substrate, species, faalpred, adenylpred
    """
    norm_to_actual = {_norm_text(c): c for c in cols}

    def pick(cands: List[str]) -> Optional[str]:
        # Exact match
        for c in cands:
            n = _norm_text(c)
            if n in norm_to_actual:
                return norm_to_actual[n]
        # Relaxed (substring) match
        for norm, actual in norm_to_actual.items():
            for c in cands:
                n = _norm_text(c)
                if n in norm or norm in n:
                    return actual
        return None

    return {
        "protein":   pick(["protein", "enzyme", "name", "protein name"]),
        "accession": pick(["refseq/genbank", "refseq genbank", "protein accession", "current accession", "accession", "refseq", "genbank"]),
        "substrate": pick(["substrate in literature", "experimental substrate", "substrate", "substrates in literature"]),
        "species":   pick(["species", "organism name", "organism", "strain"]),
        "faalpred":  pick(["faalpred, prediction score", "faalpred prediction score", "faalpred", "faalpred score"]),
        "adenylpred":pick(["adenylpred, prediction score", "adenylpred prediction score", "adenylpred", "adenylpred score"]),
    }


def ensure_columns(df) -> Dict[str, str]:
    """Ensure all required columns exist; raise a clear error otherwise."""
    mapping = map_headers(df.columns)
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(
            "Could not find the required columns: {}.\n"
            "Found columns: {}\n"
            "Hint: the final table must have exactly:\n"
            "Protein | Refseq/GenBank | Substrate in literature | Species | "
            "FAALPred, Prediction Score | AdenylPred, Prediction Score".format(missing, list(df.columns))
        )
    return mapping


# ----------------- IO helpers -----------------
def read_table_any(input_path: str) -> pd.DataFrame:
    """
    Read TSV/CSV with automatic delimiter detection; if extension is .xls/.xlsx, read Excel.
    """
    lower = input_path.lower()
    if lower.endswith(".xls") or lower.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        try:
            df = pd.read_csv(input_path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(input_path, sep="\t", engine="python", encoding="utf-8-sig")
    return df.fillna("")


def macro_mean(series: pd.Series) -> float:
    """Mean over non-NaN values; returns NaN (float) if empty."""
    vals = series.dropna().values
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


# ----------------- Save figures in multiple formats -----------------
def save_multiformats(fig: plt.Figure, base_path_no_ext: str, dpi: int = 900) -> None:
    """
    Save the figure as:
      - SVG (vector)
      - PNG (specified dpi)
      - TIFF (specified dpi; falls back to .tif if needed)
    """
    svg_path  = f"{base_path_no_ext}.svg"
    png_path  = f"{base_path_no_ext}.png"
    tiff_path = f"{base_path_no_ext}.tiff"

    # SVG
    fig.savefig(svg_path, format="svg")

    # PNG (e.g., 900 dpi)
    fig.savefig(png_path, format="png", dpi=dpi)

    # TIFF (e.g., 900 dpi)
    try:
        fig.savefig(tiff_path, format="tiff", dpi=dpi)
    except Exception:
        fig.savefig(f"{base_path_no_ext}.tif", format="tiff", dpi=dpi)

    plt.close(fig)


# ----------------- Core runner -----------------
def run(input_path: str, outdir: str, dpi: int):
    df = read_table_any(input_path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    cols = ensure_columns(df)
    recs = []

    for _, row in df.iterrows():
        exp_text  = row[cols["substrate"]]
        faal_text = row[cols["faalpred"]]
        aden_text = row[cols["adenylpred"]]

        # Experimental
        E = parse_experimental_set(exp_text)
        preferred = parse_preferred_fa(exp_text, E)

        # FAALPred (discrete): Exact-Hit
        P_faal, faal_score = parse_faal_pred_set(faal_text)
        faal_top = choose_top_from_set(P_faal)
        faal_exact_hit = int(faal_top == preferred) if (preferred is not None and faal_top is not None) else np.nan

        # AdenylPred (bin): Jaccard + Preferred-in-Bin
        P_aden, aden_score, aden_bin = parse_adenyl_pred_set(aden_text)
        aden_pref_in_bin = np.nan
        if aden_bin and preferred is not None:
            low, high = aden_bin
            aden_pref_in_bin = int(low <= preferred <= high)

        recs.append({
            "Protein": row[cols["protein"]],
            "Refseq/GenBank": row[cols["accession"]],
            "Species": row[cols["species"]],
            "E_exp": sorted(E),
            "Preferred_FA": preferred,
            # FAALPred
            "FAALPred_set": sorted(P_faal),
            "FAALPred_score": faal_score,
            "FAALPred_top": faal_top,
            "FAALPred_ExactHit": faal_exact_hit,
            # AdenylPred
            "AdenylPred_set": sorted(P_aden),
            "AdenylPred_score": aden_score,
            "AdenylPred_bin": aden_bin,
            "AdenylPred_Jaccard": jaccard(P_aden, E),
            "AdenylPred_PreferredInBin": aden_pref_in_bin,
        })

    res = pd.DataFrame.from_records(recs)

    # ----- Summary with the requested metrics only -----
    summary = {
        "FAALPred": {
            "Exact_Hit_rate": macro_mean(res["FAALPred_ExactHit"]),
        },
        "AdenylPred": {
            "Overlap_Jaccard_mean": macro_mean(res["AdenylPred_Jaccard"]),
            "Preferred_in_Bin_rate": macro_mean(res["AdenylPred_PreferredInBin"]),
        }
    }

    minimal_summary = {
        "FAALPred_Exact_Hit_rate": summary["FAALPred"]["Exact_Hit_rate"],
        "AdenylPred_Jaccard_mean": summary["AdenylPred"]["Overlap_Jaccard_mean"],
        "AdenylPred_Preferred_in_Bin_rate": summary["AdenylPred"]["Preferred_in_Bin_rate"],
    }

    # ----- Write outputs -----
    os.makedirs(outdir, exist_ok=True)
    per_case_path   = os.path.join(outdir, "per_case_metrics.tsv")   # TSV output
    summary_path    = os.path.join(outdir, "summary_metrics.json")
    summary_minimal = os.path.join(outdir, "summary_metrics_minimal.json")

    # Per-protein metrics TSV
    res.to_csv(per_case_path, sep="\t", index=False)
    # JSON summaries
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_minimal, "w") as f:
        json.dump(minimal_summary, f, indent=2)

    # ----- Figures -----
    # A) AdenylPred: Jaccard + Preferred-in-Bin
    cats_ap = ["Jaccard (mean)", "Preferred-in-Bin"]
    vals_ap = [
        summary["AdenylPred"]["Overlap_Jaccard_mean"],
        summary["AdenylPred"]["Preferred_in_Bin_rate"],
    ]
    fig_ap, ax_ap = plt.subplots(figsize=(6, 4))
    ax_ap.bar(range(len(cats_ap)), vals_ap)
    ax_ap.set_xticks(range(len(cats_ap)))
    ax_ap.set_xticklabels(cats_ap, rotation=10)
    ax_ap.set_ylim(0, 1.05)
    ax_ap.set_ylabel("Score")
    ax_ap.set_title("AdenylPred – Overlap & Preferred-in-Bin")
    fig_ap.tight_layout()
    save_multiformats(fig_ap, os.path.join(outdir, "AdenylPred_overlap_and_bin"), dpi=dpi)

    # B) FAALPred: Exact-Hit
    fig_fp, ax_fp = plt.subplots(figsize=(5, 4))
    ax_fp.bar([0], [summary["FAALPred"]["Exact_Hit_rate"]])
    ax_fp.set_xticks([0])
    ax_fp.set_xticklabels(["FAALPred Exact-Hit"])
    ax_fp.set_ylim(0, 1.05)
    ax_fp.set_ylabel("Rate")
    ax_fp.set_title("FAALPred – Exact-FA Hit Rate")
    fig_fp.tight_layout()
    save_multiformats(fig_fp, os.path.join(outdir, "FAALPred_exact_hit"), dpi=dpi)

    return per_case_path, summary_path, summary_minimal, summary, minimal_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input TSV/CSV/XLSX in the final 6-column format")
    parser.add_argument("--outdir", "-o", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--dpi", type=int, default=900, help="DPI for PNG/TIFF (default: 900)")
    args = parser.parse_args()

    per_case, summary_json, summary_minimal, summary, minimal = run(args.input, args.outdir, dpi=args.dpi)
    print(json.dumps(minimal, indent=2))
    print("\nSaved to:\n", per_case, "\n", summary_json, "\n", summary_minimal,
          "\n", os.path.join(args.outdir, "AdenylPred_overlap_and_bin.{svg,png,tiff}"),
          "\n", os.path.join(args.outdir, "FAALPred_exact_hit.{svg,png,tiff}"))


if __name__ == "__main__":
    main()
