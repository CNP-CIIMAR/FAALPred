# -*- coding: utf-8 -*-
#Autor: Leandro de Mattos Pereira
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
FAALPred vs AdenylPred — Complementary analyses (CLI, robust headers, multi-format figures)

This version Hides Exact-Hit entirely:
- Reports Jaccard for BOTH FAALPred (discrete set) and AdenylPred (bin -> set).
- Keeps "preferred" logic: if text has "(preferred)", ALL values before the keyword
  are preferred (ranges expanded). Otherwise, preferred_set = E if |E|=1, else empty.
- Reports FAALPred "preferred-in-set" and AdenylPred "preferred-in-bin".
- DOES NOT compute or display Exact-Hit or the "selected FA" column.

Usage:
  python faal_adenyl_metrics_cli.py --input ./input_table.tsv --outdir ./results --dpi 900

Expected input columns (6 columns; minor header variations are accepted):
  - Protein
  - Refseq/GenBank
  - Substrate in literature
  - Species
  - FAALPred, Prediction Score
  - AdenylPred, Prediction Score

Outputs (written to --outdir; default = .):
  - per_case_metrics.tsv            (human-readable per-protein metrics, TSV)
  - per_case_metrics_raw.tsv        (raw per-protein metrics, TSV)
  - summary_metrics.json            (full structure)
  - summary_metrics_minimal.json    (key metrics only)
  - AdenylPred_overlap_and_bin.svg/.png/.tiff
  - FAALPred_jaccard_prefset.svg/.png/.tiff
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
    Build the experimental set E from “Substrate in literature”:
      - Expands ranges like 'C10–C18', 'C10:0 to C18:0', 'C10 through C18'
      - Includes discrete mentions like 'C10', 'C12:0', etc.
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


def parse_preferred_set(s: str, E: Set[int]) -> Set[int]:
    """
    Return the set of literature-preferred FA(s).

    Rule:
      - If the string contains 'preferred', take ALL tokens/ranges that appear
        BEFORE the word 'preferred' as the preferred set (ranges expanded).
        e.g., 'C12–C16 (preferred), C10–C18 tested' -> {12,13,14,15,16}
              'C8:0, C10:0 (preferred); others tested' -> {8,10}
      - If there is NO 'preferred':
          * if E is a singleton, preferred_set = E
          * else preferred_set = empty set (no unique preferred reported)
    """
    idx = s.lower().find('preferred')
    if idx != -1:
        left = s[:idx]
        left_norm = left.replace('\u2013', '-')
        pref: Set[int] = set()

        # Ranges in the left segment
        for m in re.finditer(r'C\s*(\d+)\s*(?::0)?\s*(?:-|to|through)\s*C?\s*(\d+)', left_norm, flags=re.IGNORECASE):
            a, b = int(m.group(1)), int(m.group(2))
            pref |= expand_range(a, b)

        # Discrete tokens in the left segment
        pref |= set(ints_from_c_tokens(left_norm))

        return pref

    # No 'preferred' marker
    if len(E) == 1:
        return set(E)
    return set()


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
    If only discrete tokens appear, uses (min, max) as a pseudo-bin (and the set
    equals those discrete tokens).
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


def jaccard(P: Set[int], E: Set[int]) -> float:
    """Jaccard index between sets P and E."""
    if not P and not E:
        return 1.0
    if not P or not E:
        return 0.0
    inter = len(P & E)
    union = len(P | E)
    return inter / union if union > 0 else 0.0


# ----------------- Header normalization (strict) -----------------
def _norm_text(s: str) -> str:
    """Normalize header strings for robust, strict matching."""
    s = unicodedata.normalize("NFKD", s)
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


def map_headers_strict(cols) -> Dict[str, Optional[str]]:
    """Strict header mapping: only exact normalized matches against known synonyms."""
    norm_to_actual = {_norm_text(c): c for c in cols}
    mapping = {}
    for key, cand_list in SYNONYMS.items():
        found = None
        for cand in cand_list:
            n = _norm_text(cand)
            if n in norm_to_actual:
                found = norm_to_actual[n]
                break
        mapping[key] = found
    return mapping


def ensure_columns(df) -> Dict[str, str]:
    """Ensure all required columns exist; raise a clear error otherwise."""
    mapping = map_headers_strict(df.columns)
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


# ----------------- Accession utilities & robust reading -----------------
ACCESSION_PAT = re.compile(
    r'\b(?!C\d+\b)(?:[A-Z]{2,6}_[0-9]+(?:\.\d+)?|[A-Z]{2,6}[0-9]{3,}(?:\.\d+)?)\b'
)

def extract_accession(val: str) -> Optional[str]:
    """Extract a plausible accession (e.g., WP_012409759.1, AXN93619.1) from a string."""
    if not isinstance(val, str):
        val = str(val)
    m = ACCESSION_PAT.search(val)
    return m.group(0) if m else None


def accession_match_rate(series: pd.Series) -> float:
    """Fraction of values that look like accessions."""
    if series.empty:
        return 0.0
    total = min(len(series), 1000)
    hits = 0
    for v in series.head(total):
        if extract_accession(str(v)):
            hits += 1
    return hits / float(total)


def read_table_robust(input_path: str) -> pd.DataFrame:
    """
    Try multiple separators and validate by:
      1) strict header mapping, and
      2) accession match rate in the mapped 'accession' column.
    """
    lower = input_path.lower()
    if lower.endswith(".xls") or lower.endswith(".xlsx"):
        df = pd.read_excel(input_path).fillna("")
        _ = ensure_columns(df)
        return df

    candidates = [
        {"sep": "\t", "engine": "python"},
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
    ]

    best_df = None
    best_rate = -1.0
    for cand in candidates:
        try:
            df = pd.read_csv(input_path, sep=cand["sep"], engine=cand["engine"], encoding="utf-8-sig").fillna("")
            mapping = map_headers_strict(df.columns)
            if any(v is None for v in mapping.values()):
                continue
            rate = accession_match_rate(df[mapping["accession"]])
            if rate > best_rate:
                best_rate = rate
                best_df = df
                if rate >= 0.3:
                    break
        except Exception:
            continue

    if best_df is None:
        raise ValueError("Could not read table with a valid delimiter and header mapping.")
    return best_df


# ----------------- Formatting helpers -----------------
def fmt_fa_list(values) -> str:
    if not isinstance(values, (list, tuple, set)) or len(values) == 0:
        return "-"
    return "; ".join(f"C{int(v)}" for v in sorted(int(x) for x in values))

def fmt_bin(b) -> str:
    if not isinstance(b, (list, tuple)) or len(b) != 2 or any(pd.isna(x) for x in b):
        return "-"
    a, c = int(b[0]), int(b[1])
    return f"C{a}\u2013C{c}"

def fmt_score(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.2f}"

def fmt_jaccard(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.2f}"

def yesno(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return "Yes" if int(x) == 1 else "No"
    except Exception:
        return "-"


# ----------------- Save figures in multiple formats -----------------
def save_multiformats(fig: plt.Figure, base_path_no_ext: str, dpi: int = 900) -> None:
    """Save figure as SVG + PNG/TIFF (with specified dpi)."""
    svg_path  = f"{base_path_no_ext}.svg"
    png_path  = f"{base_path_no_ext}.png"
    tiff_path = f"{base_path_no_ext}.tiff"

    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=dpi)
    try:
        fig.savefig(tiff_path, format="tiff", dpi=dpi)
    except Exception:
        fig.savefig(f"{base_path_no_ext}.tif", format="tiff", dpi=dpi)
    plt.close(fig)


# ----------------- Core runner -----------------
def run(input_path: str, outdir: str, dpi: int):
    df = read_table_robust(input_path)

    # Trim strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    cols = ensure_columns(df)

    def is_blank_row(vals: List[str]) -> bool:
        blanks = {"", "-", "–", "—", "NA", "na", "None", "none"}
        return all((str(v).strip() in blanks) for v in vals)

    recs = []
    for _, row in df.iterrows():
        protein_raw   = row[cols["protein"]]
        accession_raw = row[cols["accession"]]
        species_raw   = row[cols["species"]]
        substrate_raw = row[cols["substrate"]]
        faal_raw      = row[cols["faalpred"]]
        aden_raw      = row[cols["adenylpred"]]

        # Skip fully blank/toy lines
        if is_blank_row([protein_raw, accession_raw, species_raw, substrate_raw, faal_raw, aden_raw]):
            continue

        # Experimental & preferred set
        E = parse_experimental_set(substrate_raw)
        preferred_set = parse_preferred_set(substrate_raw, E)

        # FAALPred (discrete) — set and score
        P_faal, faal_score = parse_faal_pred_set(faal_raw)

        # FAALPred Jaccard (set vs. E) and preferred-in-set
        faal_jaccard = jaccard(P_faal, E)
        faal_pref_in_set = np.nan
        if preferred_set:
            faal_pref_in_set = int(any(v in P_faal for v in preferred_set))

        # AdenylPred (bin -> set) — set, score, preferred-in-bin
        P_aden, aden_score, aden_bin = parse_adenyl_pred_set(aden_raw)
        aden_pref_in_bin = np.nan
        if aden_bin and preferred_set:
            low, high = aden_bin
            aden_pref_in_bin = int(any(low <= v <= high for v in preferred_set))

        # AdenylPred Jaccard (set vs. E)
        aden_jaccard = jaccard(P_aden, E)

        # Robust accession extraction
        acc = extract_accession(accession_raw)
        if acc is None:
            for candidate in [protein_raw, substrate_raw, faal_raw, aden_raw, species_raw]:
                acc = extract_accession(str(candidate))
                if acc:
                    break

        recs.append({
            # Raw identification & audit
            "Protein_raw": protein_raw,
            "Accession_raw": accession_raw,
            "Species_raw": species_raw,
            "Substrate_in_Literature_raw": substrate_raw,
            "FAALPred_raw": faal_raw,
            "AdenylPred_raw": aden_raw,
            # Parsed / derived
            "Accession_parsed": acc if acc else "",
            "E_exp": sorted(E),
            "Preferred_FA_set": sorted(preferred_set),
            "FAALPred_set": sorted(P_faal),
            "FAALPred_score": faal_score,
            "FAALPred_PreferredInSet": faal_pref_in_set,
            "FAALPred_Jaccard": faal_jaccard,
            "AdenylPred_set": sorted(P_aden),
            "AdenylPred_score": aden_score,
            "AdenylPred_bin": aden_bin,
            "AdenylPred_Jaccard": aden_jaccard,
            "AdenylPred_PreferredInBin": aden_pref_in_bin,
        })

    res = pd.DataFrame.from_records(recs)

    # ----- Summary -----
    def safe_mean(s: pd.Series) -> float:
        return float(np.nanmean(s.values)) if len(s) else float("nan")

    summary = {
        "FAALPred": {
            "Overlap_Jaccard_mean": safe_mean(res["FAALPred_Jaccard"]),
            "Preferred_in_Set_rate": safe_mean(res["FAALPred_PreferredInSet"]),
        },
        "AdenylPred": {
            "Overlap_Jaccard_mean": safe_mean(res["AdenylPred_Jaccard"]),
            "Preferred_in_Bin_rate": safe_mean(res["AdenylPred_PreferredInBin"]),
        }
    }

    minimal_summary = {
        "FAALPred_Jaccard_mean": summary["FAALPred"]["Overlap_Jaccard_mean"],
        "FAALPred_Preferred_in_Set_rate": summary["FAALPred"]["Preferred_in_Set_rate"],
        "AdenylPred_Jaccard_mean": summary["AdenylPred"]["Overlap_Jaccard_mean"],
        "AdenylPred_Preferred_in_Bin_rate": summary["AdenylPred"]["Preferred_in_Bin_rate"],
    }

    # ----- Write outputs -----
    os.makedirs(outdir, exist_ok=True)

    # RAW TSV
    per_case_raw = os.path.join(outdir, "per_case_metrics_raw.tsv")
    res.to_csv(per_case_raw, sep="\t", index=False)

    # Human-readable TSV
    pretty_rows = []
    for _, r in res.iterrows():
        acc = r.get("Accession_parsed") or ""
        pretty_rows.append({
            "Protein ID (Refseq/GenBank accession)": acc if acc else "-",
            "Protein name (from input)": r.get("Protein_raw", ""),
            "Species": r.get("Species_raw", ""),
            "Substrate in Literature (as provided)": r.get("Substrate_in_Literature_raw", ""),
            "Experimental FA set (expanded)": fmt_fa_list(r.get("E_exp", [])),
            "Literature-preferred FA(s)": fmt_fa_list(r.get("Preferred_FA_set", [])),
            "FAALPred prediction (discrete)": fmt_fa_list(r.get("FAALPred_set", [])),
            "FAALPred score": fmt_score(r.get("FAALPred_score")),
            "FAALPred Jaccard (overlap)": fmt_jaccard(r.get("FAALPred_Jaccard")),
            "FAALPred preferred in set?": yesno(r.get("FAALPred_PreferredInSet")),
            "AdenylPred predicted bin": fmt_bin(r.get("AdenylPred_bin")),
            "AdenylPred score": fmt_score(r.get("AdenylPred_score")),
            "AdenylPred Jaccard (overlap)": fmt_jaccard(r.get("AdenylPred_Jaccard")),
            "AdenylPred preferred in bin?": yesno(r.get("AdenylPred_PreferredInBin")),
        })
    pretty_df = pd.DataFrame(pretty_rows)
    per_case_pretty = os.path.join(outdir, "per_case_metrics.tsv")
    pretty_df.to_csv(per_case_pretty, sep="\t", index=False)

    # JSON summaries
    with open(os.path.join(outdir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(outdir, "summary_metrics_minimal.json"), "w") as f:
        json.dump(minimal_summary, f, indent=2)

    # ----- Figures -----
    # AdenylPred: Jaccard + Preferred-in-Bin
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
    ax_ap.set_title("AdenylPred – Jaccard & Preferred-in-Bin")
    fig_ap.tight_layout()
    save_multiformats(fig_ap, os.path.join(outdir, "AdenylPred_overlap_and_bin"), dpi=dpi)

    # FAALPred: Jaccard + Preferred-in-Set
    cats_fp = ["Jaccard (mean)", "Preferred-in-Set"]
    vals_fp = [
        summary["FAALPred"]["Overlap_Jaccard_mean"],
        summary["FAALPred"]["Preferred_in_Set_rate"],
    ]
    fig_fp, ax_fp = plt.subplots(figsize=(6, 4))
    ax_fp.bar(range(len(cats_fp)), vals_fp)
    ax_fp.set_xticks(range(len(cats_fp)))
    ax_fp.set_xticklabels(cats_fp, rotation=10)
    ax_fp.set_ylim(0, 1.05)
    ax_fp.set_ylabel("Score")
    ax_fp.set_title("FAALPred – Jaccard & Preferred-in-Set")
    fig_fp.tight_layout()
    save_multiformats(fig_fp, os.path.join(outdir, "FAALPred_jaccard_prefset"), dpi=dpi)

    return per_case_pretty, per_case_raw, summary, minimal_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input TSV/CSV/XLSX in the final 6-column format")
    parser.add_argument("--outdir", "-o", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--dpi", type=int, default=900, help="DPI for PNG/TIFF (default: 900)")
    args = parser.parse_args()

    per_case, per_case_raw, summary, minimal = run(args.input, args.outdir, dpi=args.dpi)
    print(json.dumps(minimal, indent=2))
    print("\nSaved to:\n",
          per_case, "\n",
          per_case_raw, "\n",
          os.path.join(args.outdir, "summary_metrics.json"), "\n",
          os.path.join(args.outdir, "summary_metrics_minimal.json"), "\n",
          os.path.join(args.outdir, "AdenylPred_overlap_and_bin.{svg,png,tiff}"), "\n",
          os.path.join(args.outdir, "FAALPred_jaccard_prefset.{svg,png,tiff}"))


if __name__ == "__main__":
    main()
