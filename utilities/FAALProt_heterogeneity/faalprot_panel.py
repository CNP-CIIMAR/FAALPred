#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mpl_colors
from sklearn.manifold import TSNE

RANDOM_STATE = 42

# Tamanhos de fonte para publicação
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 10
LEGEND_TITLE_FONTSIZE = 11

# Domínios para filtrar (não contam como Phylum)
DOMAIN_TOKS = {
    "bacteria",
    "archaea",
    "eukaryota",
    "eukarya",
    "viruses",
    "virus",
    "viroids",
}

# Paleta fixa original para Phylum
PHYLUM_PALETTE = {
    "Myxococcota": "#0000ff","Candidatus Riflebacteria": "#008000","Candidatus Tectomicrobia": "#008b8b",
    "Candidatus Eremiobacterota": "#00bfff","Cyanobacteriota": "#00ff4f","Ignavibacteriota": "#00ffff",
    "Euryarchaeota": "#040404","Bacillota": "#073763","Thermodesulfobacteriota": "#15ffff",
    "Deinococcota": "#16537e","Bdellovibrionota": "#1e90ff","Armatimonadota": "#20b2aa",
    "Viridiplantae": "#274e13","Planctomycetota": "#331900","Sar": "#331900","Pseudomonadati": "#34ae8f",
    "NA": "#36abaf","Metazoa": "#36abb2","Actinobacteria": "#37a9c3","Amoebozoa": "#39a6d3",
    "Candidatus Neomarinimicrobiota": "#3ba3ec","Cyanobacteria": "#46a0f4","Abditibacteriota": "#483d8b",
    "Candidatus Aerophobetes": "#4b0082","Candidatus Blackallbacteria": "#556b2f","Deltaproteobacteria": "#5f9ea0",
    "Candidatus Margulisiibacteriota": "#61ae31","Actinomycetota": "#6a329f","Candidatus Deferrimicrobiota": "#719af4",
    "Apicomplexa": "#744700","Candidatus Margulisbacteria": "#7cfc00","Lentisphaerota": "#7fffd4",
    "Rhodothermota": "#808080","candidate division KSB3": "#8096f4","Candidatus Shapirobacteria": "#87cefa",
    "Candidatus Tectimicrobiota": "#89a631","Candidatus Aminicenantes": "#8b008b","Candidatus Woesearchaeota": "#8fbc8f",
    "Candidatus Nealsonbacteria": "#9400d3","Candidatus Binatota": "#9932cc","Candidatus Moduliflexota": "#998ff4",
    "Candidatus Latescibacterota": "#a38cf4","uncultured bacterium": "#a52a2a","Candidatus Cloacimonadota": "#b19b31",
    "Calditrichota": "#b8860b","Fungi": "#bb9731","Chloroflexota": "#c90076","Candidatus Hydrogenedentota": "#cb9131",
    "Nitrospinota": "#d2691e","Chlamydiota": "#dcdcdc","Candidatus Sericytochromatia": "#dd6ef4",
    "Metamonada": "#de8731","Gemmatimonadota": "#deb887","Vulcanimicrobiota": "#e68231","Chlorobiota": "#e6e6fa",
    "Nitrospirota": "#ea9999","Discoba": "#ef7c32","Candidatus Melainabacteria": "#f0fff0","Candidatus Hinthialibacterota": "#f25cf4",
    "Acidobacteriota": "#f47936","Candidatus Moraniibacteriota": "#f562d9","Candidatus Omnitrophota": "#f569ba",
    "Candidatus Rokuibacteriota": "#f66bab","Candidatus Tharpellota": "#f6717e","Haptista": "#f77272",
    "candidate division KSB1": "#f77553","Bacillati": "#f77639","Candidatus Sumerlaeota": "#f8f8ff",
    "Elusimicrobiota": "#faf0e6","Pseudomonadota": "#ff0000","Candidatus Krumholzibacteriota": "#ff00ff",
    "Campylobacterota": "#ff69b4","bacterium": "#ff7f50","Spirochaetota": "#ff8c00","environmental samples": "#ffa07a",
    "Candidatus Rokubacteria": "#ffa312","Verrucomicrobiota": "#ffd34b","Caldisericota": "#fff0f5",
    "candidate division NC10": "#fff2cc","Candidatus Gracilibacteria": "#fff8dc","Candidatus Eisenbacteria": "#fffaf0",
    "Bacteroidota": "#ffff00",
}

def ensure_outdir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _save_multiformat_figure(fig, out_prefix: str) -> None:
    """Salva SVG/PNG/TIFF com tight layout."""
    ensure_outdir(out_prefix)
    svg = out_prefix + ".svg"
    png = out_prefix + ".png"
    tiff = out_prefix + ".tiff"
    fig.tight_layout()
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=900, bbox_inches="tight")
    fig.savefig(tiff, dpi=900, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", os.path.basename(svg),
          os.path.basename(png),
          os.path.basename(tiff))

def _rgba_tuple(c):
    if isinstance(c, tuple) and len(c) in (3, 4):
        if len(c) == 3:
            return (c[0], c[1], c[2], 1.0)
        return c
    try:
        r, g, b, a = mpl_colors.to_rgba(c)
        return (r, g, b, a)
    except Exception:
        return (0.5, 0.5, 0.5, 1.0)

def _distinct_palette(n: int):
    """Gera paleta extra (quando o PHYLUM_PALETTE não cobre todos)."""
    if n <= 0:
        return []
    base = []
    for name in ["tab20", "tab20b", "tab20c"]:
        cmap = cm.get_cmap(name)
        if hasattr(cmap, "colors"):
            base += list(cmap.colors)
        else:
            base += [cmap(i) for i in np.linspace(0, 1, 20)]
    base = [_rgba_tuple(c) for c in base]
    if n <= len(base):
        return base[:n]
    extra = []
    for i in range(n - len(base)):
        h = (i * 0.61803398875) % 1.0
        s = 0.88
        v = 0.92 if (i % 2 == 0) else 0.78
        r, g, b = mpl_colors.hsv_to_rgb((h, s, v))
        extra.append((r, g, b, 1.0))
    return base + extra

def normalize_seq_id(s: str) -> str:
    """Normaliza IDs (tira '>', pega primeiro token, remove '|', etc.)."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if t.startswith(">"):
        t = t[1:]
    t = t.split()[0]
    if "|" in t:
        t = t.split("|")[0]
    return t

def normalize_phylum_name(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "Unknown"
    t = s.strip()
    if t.lower() == "actinobacteria":
        return "Actinomycetota"
    return t

def _is_excluded_phylum(s) -> bool:
    """Remove Unknown, uncultured, domínios, etc."""
    if not isinstance(s, str):
        return True
    sl = s.strip().lower()
    if sl == "":
        return True
    if "unknown" in sl:
        return True
    if "uncultured" in sl:
        return True
    if "uncultivated" in sl:
        return True
    if "uncultived" in sl:
        return True
    if sl in DOMAIN_TOKS:
        return True
    return False

def _extract_phylum_from_lineage(lineage: str) -> str:
    """Extrai Phylum a partir da coluna 'Species Lineage'."""
    if not isinstance(lineage, str) or not lineage.strip():
        return "Unknown"
    toks = [t.strip() for t in lineage.split(";") if t.strip()]
    if len(toks) >= 2:
        cand = toks[1]
        if cand.lower() in DOMAIN_TOKS:
            return "Unknown"
        return normalize_phylum_name(cand)
    if len(toks) == 1:
        cand = toks[0]
        if cand.lower() in DOMAIN_TOKS:
            return "Unknown"
        return normalize_phylum_name(cand)
    return "Unknown"

def read_metadata_table(meta_path: str) -> pd.DataFrame:
    """Lê meta-table e devolve (sequence_id, phylum) usando Species Lineage."""
    if not meta_path or not os.path.exists(meta_path):
        raise SystemExit(f"Meta-table não encontrada: {meta_path}")
    df = pd.read_csv(meta_path, sep="\t", dtype=str, engine="python")
    if df.empty:
        raise SystemExit(f"Meta-table vazia: {meta_path}")

    cols_lc = {c.lower().strip(): c for c in df.columns}

    # Coluna de ID
    acc_col = None
    for key in [
        "protein accession",
        "protein_accession",
        "protein accession version",
        "signature.accession",
        "protein id",
        "protein_id",
    ]:
        if key in cols_lc:
            acc_col = cols_lc[key]
            break
    if acc_col is None:
        for c in df.columns:
            if "accession" in c.lower():
                acc_col = c
                break
    if acc_col is None:
        raise SystemExit("Não encontrei coluna de ID (Protein Accession) na tabela de metadados.")

    # Coluna de lineage
    lin_col = None
    for key in ["species lineage", "species_lineage"]:
        if key in cols_lc:
            lin_col = cols_lc[key]
            break
    if lin_col is None:
        for c in df.columns:
            if "lineage" in c.lower():
                lin_col = c
                break
    if lin_col is None:
        raise SystemExit("Não encontrei coluna 'Species Lineage' (ou similar) na tabela de metadados.")

    tmp = df[[acc_col, lin_col]].copy()
    tmp[acc_col] = tmp[acc_col].astype(str).map(normalize_seq_id)
    tmp["phylum"] = tmp[lin_col].astype(str).map(_extract_phylum_from_lineage)
    tmp = tmp.rename(columns={acc_col: "sequence_id"})
    tmp = tmp[["sequence_id", "phylum"]].dropna().drop_duplicates()
    return tmp

def rebuild_subset_phylum_map(subset_dir: str, meta_table: str) -> pd.DataFrame:
    """Recria subset_phylum_map.csv usando a meta-table (Species Lineage)."""
    pairs_csv = os.path.join(subset_dir, "pairs.csv")
    if not os.path.exists(pairs_csv):
        raise SystemExit(f"pairs.csv não encontrado: {pairs_csv}")
    pairs_df = pd.read_csv(pairs_csv)
    if pairs_df.empty:
        raise SystemExit("pairs.csv está vazio, impossível recriar mapa de phylum.")

    ids = pd.unique(pd.concat([pairs_df["seq_i"], pairs_df["seq_j"]], ignore_index=True))
    ids = [normalize_seq_id(s) for s in ids]

    meta_df = read_metadata_table(meta_table)
    meta_df["sequence_id"] = meta_df["sequence_id"].astype(str).map(normalize_seq_id)
    meta_map = meta_df.set_index("sequence_id")["phylum"].to_dict()

    rows = []
    for sid in ids:
        ph = normalize_phylum_name(meta_map.get(sid, "Unknown"))
        rows.append({"sequence_id": sid, "phylum": ph})
    phylum_map_df = pd.DataFrame(rows).drop_duplicates()
    out_csv = os.path.join(subset_dir, "subset_phylum_map.csv")
    phylum_map_df.to_csv(out_csv, index=False)
    print("[meta] subset_phylum_map.csv recriado com",
          len(phylum_map_df), "linhas ->", out_csv)
    return phylum_map_df

def build_global_phylum_palette(phylum_series: pd.Series) -> dict:
    """Constrói paleta global para todos os phyla presentes."""
    phyla = set()
    for p in phylum_series:
        if not isinstance(p, str):
            continue
        pn = normalize_phylum_name(p)
        if _is_excluded_phylum(pn):
            continue
        phyla.add(pn)
    phyla = sorted(phyla)

    palette = {}
    for p in phyla:
        if p in PHYLUM_PALETTE:
            palette[p] = mpl_colors.to_rgba(PHYLUM_PALETTE[p])

    missing = [p for p in phyla if p not in palette]
    extra = _distinct_palette(len(missing))
    for p, c in zip(missing, extra):
        palette[p] = c
    return palette

def _safe_tsne_perplexity(n: int) -> float:
    return max(5.0, min(30.0, (n - 1) / 3.0))

def build_dissimilarity_matrix_from_pairs(ids, pairs_df: pd.DataFrame) -> np.ndarray:
    """Constroi matriz de dissimilaridade (1 - identity/100) para um conjunto de IDs."""
    n = len(ids)
    D = np.ones((n, n), dtype=float)
    np.fill_diagonal(D, 0.0)
    idx = {sid: i for i, sid in enumerate(ids)}

    for _, row in pairs_df.iterrows():
        a = row["seq_i"]; b = row["seq_j"]
        if a not in idx or b not in idx:
            continue
        i = idx[a]; j = idx[b]
        idp = float(row["identity_percent"])
        d = 1.0 - (idp / 100.0)
        if d < D[i, j]:
            D[i, j] = d
            D[j, i] = d
    return D

def top_phyla_from_ids(seq_ids, seq_to_phylum, top_n: int):
    """Retorna os top_n phyla com mais sequências entre seq_ids."""
    from collections import Counter
    counter = Counter()
    for sid in seq_ids:
        ph = normalize_phylum_name(seq_to_phylum.get(sid, "Unknown"))
        if _is_excluded_phylum(ph):
            continue
        counter[ph] += 1
    if not counter:
        return []
    return [p for p, _ in counter.most_common(top_n)]

def panel_label(ax, label: str):
    """Imprime letras A,B,C,... do lado de fora (superior esquerdo) do painel."""
    ax.text(
        -0.12, 1.02, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=16, fontweight="bold",
    )

# ------------ Painel A: histograma de identidade -----------------

def plot_bar_identity_distribution_ax(pairs_df: pd.DataFrame, ax):
    if pairs_df is None or pairs_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    ids = pairs_df["identity_percent"].to_numpy(dtype=float)
    bins = np.arange(10, 101, 5)
    hist, edges = np.histogram(ids, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(
        centers,
        hist,
        width=4.8,
        edgecolor="black",
        color="#4C72B0",
        alpha=0.85,
    )
    ax.set_xlabel("Identity (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Pair count (all pairs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(10, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

# ------------ Painel B: intra-cluster por cutoff -----------------

def _counts_intra_by_cutoff_deciles(pairs_df: pd.DataFrame,
                                    clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Conta pares intra-cluster por cutoff e por decil de identidade."""
    bins = np.arange(10, 101, 10)
    rows = []
    if pairs_df.empty or clusters_df.empty:
        return pd.DataFrame(rows)

    P = pairs_df.copy()
    P["a"] = P[["seq_i", "seq_j"]].min(axis=1)
    P["b"] = P[["seq_i", "seq_j"]].max(axis=1)
    P = P.drop(columns=["seq_i", "seq_j"]).drop_duplicates(["a", "b"])

    for c in sorted(clusters_df["cutoff"].unique()):
        mem = clusters_df[clusters_df["cutoff"] == c][
            ["sequence_id", "cluster_id"]
        ].drop_duplicates()
        if mem.empty:
            continue
        m = (
            P.merge(
                mem.rename(columns={"sequence_id": "a"}),
                on="a",
                how="inner",
            )
            .merge(
                mem.rename(
                    columns={"sequence_id": "b", "cluster_id": "cluster_id_b"}
                ),
                on="b",
                how="inner",
            )
        )
        m = m[m["cluster_id"] == m["cluster_id_b"]]
        if m.empty:
            continue
        vals = m["identity_percent"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        hist, edges = np.histogram(vals, bins=bins)
        for b_idx in range(len(hist)):
            rows.append(
                {
                    "cutoff": float(c),
                    "bin_left": float(edges[b_idx]),
                    "bin_right": float(edges[b_idx + 1]),
                    "x_value": int(edges[b_idx]),
                    "count": int(hist[b_idx]),
                }
            )
    return pd.DataFrame(rows)

def plot_intra_identity_counts_by_cutoff_lines_ax(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    ax,
):
    df = _counts_intra_by_cutoff_deciles(pairs_df, clusters_df)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    x_ticks = sorted(df["x_value"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    for idx, c in enumerate(sorted(df["cutoff"].unique())):
        sub = df[df["cutoff"] == c].sort_values("x_value")
        color = None
        if colors is not None and idx < len(colors):
            color = colors[idx]
        ax.plot(
            sub["x_value"],
            sub["count"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"{int(c)}%",
            color=color,
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], rotation=0)
    ax.set_xlabel("Identity (%) — decile bins left edges (10, 20, …)",
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Intra-cluster pair count",
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(
        bbox_to_anchor=(0.5, -0.22),
        loc="upper center",
        ncol=min(7, len(df["cutoff"].unique())),
        fontsize=LEGEND_FONTSIZE,
    )

# ------------ Painel C: t-SNE (1 - identidade) -------------------

def plot_tsne_dissimilarity_top_phyla_ax(
    pairs_df: pd.DataFrame,
    phylum_map_df: pd.DataFrame,
    top_n: int,
    ax,
    global_palette: dict,
):
    """t-SNE de (1 - identidade) colorido por Phylum (top_n phyla)."""
    phylum_map_df = phylum_map_df.copy()
    phylum_map_df["phylum"] = phylum_map_df["phylum"].apply(
        normalize_phylum_name
    )
    seq_to_phylum = dict(
        zip(phylum_map_df["sequence_id"], phylum_map_df["phylum"])
    )

    ids_all = pd.unique(
        pd.concat([pairs_df["seq_i"], pairs_df["seq_j"]], ignore_index=True)
    ).tolist()
    ids_all = [sid for sid in ids_all if sid in seq_to_phylum]

    if len(ids_all) < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return set()

    top_phyla = top_phyla_from_ids(ids_all, seq_to_phylum, top_n)
    if not top_phyla:
        top_phyla = sorted(
            set(
                normalize_phylum_name(seq_to_phylum[s])
                for s in ids_all
            )
        )

    ids_filtered = [
        sid for sid in ids_all
        if normalize_phylum_name(seq_to_phylum.get(sid, "Unknown")) in top_phyla
    ]
    if len(ids_filtered) < 2:
        ax.text(0.5, 0.5, "No data (top phyla)", ha="center", va="center")
        return set()

    D = build_dissimilarity_matrix_from_pairs(ids_filtered, pairs_df)
    perp = _safe_tsne_perplexity(len(ids_filtered))
    reducer = TSNE(
        n_components=2,
        metric="precomputed",
        random_state=RANDOM_STATE,
        init="random",
        learning_rate="auto",
        perplexity=perp,
    )
    Z = reducer.fit_transform(D)

    phyla_used = sorted(
        set(
            normalize_phylum_name(seq_to_phylum[sid])
            for sid in ids_filtered
        )
    )
    cmap = {p: global_palette.get(p, (0.5, 0.5, 0.5, 1.0)) for p in phyla_used}

    C = [
        cmap.get(
            normalize_phylum_name(seq_to_phylum.get(sid, "Unknown")),
            (0.5, 0.5, 0.5, 1.0),
        )
        for sid in ids_filtered
    ]
    ax.scatter(Z[:, 0], Z[:, 1], c=C, s=18, alpha=0.95, linewidths=0)
    ax.set_xlabel("t-SNE-1", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("t-SNE-2", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("t-SNE (1 - identity) — top phyla")
    return set(phyla_used)

# ------------ Painel D: t-SNE W2V -------------------

def plot_tsne_w2v_top_phyla_ax(
    w2v_df: pd.DataFrame,
    phylum_map_df: pd.DataFrame,
    top_n: int,
    ax,
    global_palette: dict,
):
    """t-SNE dos embeddings W2V (média), usando Phylum da meta-table."""
    if "sequence_id" not in w2v_df.columns:
        ax.text(0.5, 0.5, "No 'sequence_id' column", ha="center", va="center")
        return set()

    pm = phylum_map_df[["sequence_id", "phylum"]].copy()
    pm["sequence_id"] = pm["sequence_id"].astype(str).map(normalize_seq_id)
    pm["phylum"] = pm["phylum"].apply(normalize_phylum_name)

    w2v_df = w2v_df.copy()
    w2v_df["sequence_id"] = w2v_df["sequence_id"].astype(str).map(normalize_seq_id)
    if "phylum" in w2v_df.columns:
        w2v_df = w2v_df.drop(columns=["phylum"])
    w2v_df = w2v_df.merge(pm, on="sequence_id", how="left")
    w2v_df["phylum"] = w2v_df["phylum"].fillna("Unknown").apply(normalize_phylum_name)

    feat_cols = [c for c in w2v_df.columns if c.startswith("f")]
    if not feat_cols:
        ax.text(0.5, 0.5, "No feature columns", ha="center", va="center")
        return set()

    df_valid = w2v_df[~w2v_df["phylum"].apply(_is_excluded_phylum)].copy()
    if df_valid.empty:
        df_valid = w2v_df.copy()

    counts = df_valid["phylum"].value_counts()
    top_phyla = list(counts.head(top_n).index)
    if not top_phyla:
        ax.text(0.5, 0.5, "No top phyla", ha="center", va="center")
        return set()

    df_top = df_valid[df_valid["phylum"].isin(top_phyla)].reset_index(drop=True)
    if len(df_top) < 2:
        ax.text(0.5, 0.5, "No data (top phyla)", ha="center", va="center")
        return set()

    X = df_top[feat_cols].to_numpy(float)
    perp = _safe_tsne_perplexity(len(df_top))
    reducer = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
        perplexity=perp,
    )
    Z = reducer.fit_transform(X)

    phyla_used = sorted(df_top["phylum"].unique())
    cmap = {p: global_palette.get(p, (0.5, 0.5, 0.5, 1.0)) for p in phyla_used}
    C = [cmap[p] for p in df_top["phylum"]]
    ax.scatter(Z[:, 0], Z[:, 1], c=C, s=14, alpha=0.95, linewidths=0)
    ax.set_xlabel("t-SNE-1", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("t-SNE-2", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("t-SNE (W2V) — top phyla")
    return set(phyla_used)

# ------------ E/F: cosine vs identity -------------------

def plot_cosine_vs_identity_type_ax(
    cos_csv: str,
    pair_type: str,
    ax,
    title: str,
    phylum_map_df: pd.DataFrame | None = None,
):
    """Cosine vs identity para Intra- ou Inter-phylum.

    Se `type` estiver ausente ou inconsistente no CSV, ele é recalculado a
    partir do mapeamento de phylum (subset_phylum_map.csv).
    """
    df = pd.read_csv(cos_csv)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    # Recalcula 'type' a partir do mapeamento de Phylum (robusto)
    if phylum_map_df is not None:
        ph = phylum_map_df.copy()
        ph["sequence_id"] = ph["sequence_id"].astype(str).map(normalize_seq_id)
        ph["phylum"] = ph["phylum"].apply(normalize_phylum_name)
        ph_map = ph.set_index("sequence_id")["phylum"].to_dict()

        def _row_type(row):
            a = normalize_seq_id(row["seq_i"])
            b = normalize_seq_id(row["seq_j"])
            pa = normalize_phylum_name(ph_map.get(a, "Unknown"))
            pb = normalize_phylum_name(ph_map.get(b, "Unknown"))
            if pa == "Unknown" or pb == "Unknown":
                return "Unknown"
            if _is_excluded_phylum(pa) or _is_excluded_phylum(pb):
                return "Unknown"
            return "Intra-phylum" if pa == pb else "Inter-phylum"

        df["type"] = df.apply(_row_type, axis=1)
        df = df[df["type"] != "Unknown"]

    if "type" not in df.columns:
        ax.text(0.5, 0.5, "No 'type' column", ha="center", va="center")
        return

    type_norm = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
    )
    target = pair_type.strip().lower().replace("_", "-")

    sub = df[type_norm == target].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"No data ({pair_type})", ha="center", va="center")
        return

    hb = ax.hexbin(
        sub["identity_percent"],
        sub["cosine"],
        gridsize=50,
        mincnt=2,
        cmap="viridis",
    )
    ax.set_xlabel("Identity (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Cosine similarity (W2V mean)", fontsize=AXIS_LABEL_FONTSIZE)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Pair count", fontsize=AXIS_LABEL_FONTSIZE)

    # Eixo X 10–100 de 10 em 10
    ax.set_xlim(10, 100)
    ax.set_xticks(np.arange(10, 101, 10))

    # Curva média por decil
    bins = np.arange(10, 101, 10)
    sub["bin"] = pd.cut(
        sub["identity_percent"],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    grp = (
        sub.groupby("bin", observed=False)
        .agg(x=("identity_percent", "mean"),
             y=("cosine", "mean"))
        .dropna()
    )
    if not grp.empty:
        ax.plot(
            grp["x"],
            grp["y"],
            marker="o",
            linewidth=2,
            color="deepskyblue",
        )
    ax.set_title(title)

# --------- localizar arquivo pairs_identity_cosine_full ---------

def find_pairs_identity_cosine_file(
    subset_dir: str,
    w2v_cos_dir: str,
    w2v_tsne_dir: str,
) -> str:
    """Localiza o pairs_identity_cosine_full.csv de forma robusta."""
    candidates: list[str] = []

    # Se o usuário passou diretamente o CSV
    if os.path.isfile(w2v_cos_dir) and w2v_cos_dir.endswith(".csv"):
        candidates.append(w2v_cos_dir)

    # Tenta tratar w2v_cos_dir e w2v_tsne_dir como diretórios W2V
    for base in {w2v_cos_dir, w2v_tsne_dir}:
        if not base:
            continue
        if os.path.isdir(base):
            candidates.append(os.path.join(base, "pairs_identity_cosine_full.csv"))
            candidates.append(os.path.join(base, "plots", "pairs_identity_cosine_full.csv"))

    # Último recurso: procurar dentro do subset_dir por um único arquivo
    for root, _, files in os.walk(subset_dir):
        if "pairs_identity_cosine_full.csv" in files:
            candidates.append(os.path.join(root, "pairs_identity_cosine_full.csv"))
            break

    for p in candidates:
        if p and os.path.exists(p):
            print("[cosine] Usando arquivo:", p)
            return p

    raise SystemExit(
        "Não encontrei 'pairs_identity_cosine_full.csv'. "
        "Verifique se a Parte B (W2V) rodou e se o diretório w2v está correto."
    )

# ------------ MAIN ----------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera painéis A–F (identidade + W2V) a partir dos CSVs do FAALProt."
    )
    parser.add_argument(
        "--subset-dir",
        required=True,
        help="Diretório do subset (contendo pairs.csv, clusters_multi_threshold.csv).",
    )
    parser.add_argument(
        "--meta-table",
        required=True,
        help="Tabela de metadados (Table_S2.tsv) com colunas 'Protein Accession' e 'Species Lineage'.",
    )
    parser.add_argument(
        "--w2v-tsne-dir",
        help="Diretório W2V para o painel D (t-SNE dos embeddings). "
             "EX: <subset-dir>/w2v_dim390_ep2500.",
    )
    parser.add_argument(
        "--w2v-cos-dir",
        help="Diretório W2V ou diretório 'plots' contendo pairs_identity_cosine_full.csv "
             "para os painéis E/F. EX: <subset-dir>/w2v_dim390_ep2500.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefixo de saída (sem extensão) para a figura painel (SVG/PNG/TIFF).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Número de phyla a manter nos t-SNE (C e D). Default = 10.",
    )
    parser.add_argument(
        "--layout",
        default="AF",
        choices=["AB", "AD", "AF", "ABEF", "CDEF", "ALL"],
        help=(
            "Layout da figura final:\n"
            "  AB    -> painéis A,B\n"
            "  AD    -> painéis A–D (A,B,C,D)\n"
            "  AF    -> painéis A–F (A,B,C,D,E,F)\n"
            "  ABEF  -> A,B,E,F renomeados A,B,C,D\n"
            "  CDEF  -> C,D,E,F renomeados A,B,C,D\n"
            "  ALL   -> alias de AF"
        ),
    )
    args = parser.parse_args(argv)

    subset_dir = os.path.abspath(args.subset_dir)
    pairs_csv = os.path.join(subset_dir, "pairs.csv")
    clusters_csv = os.path.join(subset_dir, "clusters_multi_threshold.csv")

    if not os.path.exists(pairs_csv):
        raise SystemExit(f"pairs.csv não encontrado: {pairs_csv}")
    if not os.path.exists(clusters_csv):
        raise SystemExit(f"clusters_multi_threshold.csv não encontrado: {clusters_csv}")

    pairs_df = pd.read_csv(pairs_csv)
    clusters_df = pd.read_csv(clusters_csv)

    # (1) Recria subset_phylum_map.csv usando Species Lineage
    phylum_map_df = rebuild_subset_phylum_map(subset_dir, args.meta_table)

    # (2) Paleta global de phylum
    global_palette = build_global_phylum_palette(phylum_map_df["phylum"])

    # (3) Diretórios de W2V (t-SNE e cosine)
    if args.w2v_tsne_dir:
        w2v_tsne_dir = os.path.abspath(args.w2v_tsne_dir)
    else:
        w2v_tsne_dir = os.path.join(subset_dir, "w2v_dim390_ep2500")

    if args.w2v_cos_dir:
        w2v_cos_dir = os.path.abspath(args.w2v_cos_dir)
    else:
        w2v_cos_dir = os.path.join(subset_dir, "w2v_dim390_ep2500")

    if not os.path.isdir(w2v_tsne_dir):
        raise SystemExit(f"Diretório w2v-tsne-dir não existe: {w2v_tsne_dir}")

    # Arquivo de embeddings W2V
    cand_emb = [
        f for f in os.listdir(w2v_tsne_dir)
        if f.startswith("w2v_mean_embeddings_") and f.endswith(".csv")
    ]
    if not cand_emb:
        raise SystemExit(f"Nenhum w2v_mean_embeddings_*.csv encontrado em {w2v_tsne_dir}")
    if len(cand_emb) > 1:
        print("[WARN] Vários embeddings encontrados, usando o primeiro:", cand_emb[0])
    emb_csv = os.path.join(w2v_tsne_dir, cand_emb[0])
    w2v_df = pd.read_csv(emb_csv)

    # Arquivo de cosine vs identity (robusto)
    cos_csv = find_pairs_identity_cosine_file(subset_dir, w2v_cos_dir, w2v_tsne_dir)

    # Layout
    layout = args.layout.upper()
    if layout == "ALL":
        layout = "AF"

    if layout == "AF":
        panel_map = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("E", "E"), ("F", "F")]
    elif layout == "AB":
        panel_map = [("A", "A"), ("B", "B")]
    elif layout == "AD":
        panel_map = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")]
    elif layout == "ABEF":
        panel_map = [("A", "A"), ("B", "B"), ("E", "C"), ("F", "D")]
    elif layout == "CDEF":
        panel_map = [("C", "A"), ("D", "B"), ("E", "C"), ("F", "D")]
    else:
        raise SystemExit(f"Layout inválido: {layout}")

    # Cria figura
    n_panels = len(panel_map)
    if n_panels == 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.subplots_adjust(left=0.07, right=0.80, wspace=0.40, hspace=0.40)
    elif n_panels == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.subplots_adjust(left=0.08, right=0.82, wspace=0.35, hspace=0.35)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.subplots_adjust(left=0.07, right=0.93, wspace=0.35)

    axes_flat = np.atleast_1d(axes).ravel()

    phyla_c = set()
    phyla_d = set()
    axD_for_legend = None

    for (content_id, label), ax in zip(panel_map, axes_flat):
        if content_id == "A":
            plot_bar_identity_distribution_ax(pairs_df, ax)
            ax.set_title("Identity distribution")

        elif content_id == "B":
            plot_intra_identity_counts_by_cutoff_lines_ax(pairs_df, clusters_df, ax)
            ax.set_title("Intra-cluster identity by cutoff")

        elif content_id == "C":
            phyla_c = plot_tsne_dissimilarity_top_phyla_ax(
                pairs_df,
                phylum_map_df,
                args.top_n,
                ax,
                global_palette,
            )

        elif content_id == "D":
            phyla_d = plot_tsne_w2v_top_phyla_ax(
                w2v_df,
                phylum_map_df,
                args.top_n,
                ax,
                global_palette,
            )
            axD_for_legend = ax

        elif content_id == "E":
            plot_cosine_vs_identity_type_ax(
                cos_csv,
                "Intra-phylum",
                ax,
                "Cosine vs identity (Intra-phylum)",
                phylum_map_df,
            )

        elif content_id == "F":
            plot_cosine_vs_identity_type_ax(
                cos_csv,
                "Inter-phylum",
                ax,
                "Cosine vs identity (Inter-phylum)",
                phylum_map_df,
            )

        panel_label(ax, label)  # A,B,C,... no canto superior esquerdo externo

    # Legenda única de Phylum à direita do t-SNE W2V (painel D)
    from matplotlib.lines import Line2D
    if axD_for_legend is not None:
        phyla_union = sorted(set(list(phyla_c) + list(phyla_d)))
        phyla_union = [p for p in phyla_union if not _is_excluded_phylum(p)]
        if phyla_union:
            handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label=p,
                    markerfacecolor=global_palette.get(p, (0.5, 0.5, 0.5, 1.0)),
                    markersize=6,
                )
                for p in phyla_union
            ]
            axD_for_legend.legend(
                handles=handles,
                title="Phylum",
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                ncol=2,          # legenda em duas colunas
                fontsize=LEGEND_FONTSIZE,
                title_fontsize=LEGEND_TITLE_FONTSIZE,
            )

    _save_multiformat_figure(fig, args.out_prefix)
    print("[OK] Painel criado em prefixo:", args.out_prefix)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
