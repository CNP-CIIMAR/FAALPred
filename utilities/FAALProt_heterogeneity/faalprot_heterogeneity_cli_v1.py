#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAALProt Heterogeneity Pipeline — COMPLETO (com grade W2V por default, agora robusto)

Defaults da grade (Parte B):
  Subsets: 0.25, 0.50, 0.75, 1.00
  W2V dims: 100, 200, 390
  W2V epochs: 200, 500, 1500, 2500

Para cada subset e (dim, epochs):
  - Treina W2V sobre k-mers do MSA (k=3, step=1, filtra k-mers só de gaps)
  - Extrai embeddings por média com padding até min_kmers do subset
  - Salva TODOS os pares de cosseno (pairs_cosine_dimXXX_epYYYY.csv)
  - Faz UMAP/t-SNE e dendrograma por Filo (W2V)
  - Junta identidade × cosseno (pairs_identity_cosine_full.csv) e gera hexbins intra/inter

Parte A (Identidade):
  - MAFFT → identidade par-a-par de TODOS os pares (pairs.csv)
  - MMseqs2 clusters (FIXED_CUTOFFS)
  - UMAP/t-SNE e dendrograma por Filo (identidade)
  - Gráficos de distribuição/heatmap e estatísticas de cluster

Figuras em PNG/TIFF(900dpi)/SVG. Tabelas em CSV para reprodutibilidade.
"""

from __future__ import annotations

import os, sys, re, json, shutil, random, warnings, argparse, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mpl_colors

# Seeds globais para reprodutibilidade
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Optional deps
try:
    from Bio import SeqIO
    HAS_BIO = True
except Exception:
    HAS_BIO = False

# UMAP (optional)
try:
    import umap
    HAS_UMAP = True
except Exception:
    try:
        from umap import UMAP as _UMAP

        class umap:
            UMAP = _UMAP
        HAS_UMAP = True
    except Exception:
        HAS_UMAP = False

# t-SNE
from sklearn.manifold import TSNE

# W2V (optional)
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

# Dendrogram + distances
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from sklearn.metrics import pairwise_distances
    HAS_SK_PWD = True
except Exception:
    HAS_SK_PWD = False


# =============================
# Configurações globais
# =============================

FIXED_CUTOFFS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
DOMAIN_TOKS = {'bacteria', 'archaea', 'eukaryota', 'eukarya', 'viruses', 'virus', 'viroids'}

# Parte B (defaults solicitados)
DEFAULT_W2V_DIMS = [100, 200, 390]
DEFAULT_W2V_EPOCHS_LIST = [200, 500, 1500, 2500]
DEFAULT_SUBSET_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

# Hiperparâmetros W2V fixos
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_SG = 1            # skip-gram
W2V_HS = 0            # hierarchical softmax off
W2V_NEGATIVE = 5
W2V_WORKERS = min(48, os.cpu_count() or 2)  # seguro
W2V_K = 3
W2V_STEP_SIZE = 1


# =============================
# Utilities
# =============================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_multiformat(out_base: str) -> Dict[str, str]:
    out = {}
    base = os.path.splitext(out_base)[0]
    svg = base + '.svg'
    png = base + '.png'
    tif = base + '.tiff'
    plt.tight_layout()
    plt.savefig(svg, bbox_inches='tight')
    plt.savefig(png, dpi=900, bbox_inches='tight')
    plt.savefig(tif, dpi=900, bbox_inches='tight')
    plt.close()
    out['svg'] = svg
    out['png'] = png
    out['tiff'] = tif
    print('[saved]', os.path.basename(svg), os.path.basename(png), os.path.basename(tif))
    return out


def _is_exec(p: str) -> bool:
    try:
        return p and os.path.isfile(p) and os.access(p, os.X_OK)
    except Exception:
        return False


def _tokens_from_whereis(name: str) -> List[str]:
    try:
        out = subprocess.run(['whereis', name], capture_output=True, text=True, check=False)
        toks = out.stdout.strip().split()
        return [t for t in toks[1:] if os.path.isabs(t)]
    except Exception:
        return []


def find_executable(name: str, user_hint: Optional[str] = None,
                    extra_candidates: Optional[List[str]] = None) -> Optional[str]:
    env_hint = os.environ.get(f"{name.upper()}_BIN") or os.environ.get(f"{name.lower()}_bin")
    for cand in [user_hint, env_hint]:
        if cand and _is_exec(cand):
            print(f"[bin] Using explicit {name}: {cand}")
            return cand
    w = shutil.which(name)
    if _is_exec(w or ""):
        print(f"[bin] Found {name} in PATH: {w}")
        return w
    cp = os.environ.get('CONDA_PREFIX')
    if cp:
        cand = os.path.join(cp, 'bin', name)
        if _is_exec(cand):
            print(f"[bin] Found {name} in CONDA_PREFIX: {cand}")
            return cand
    for cand in (extra_candidates or []):
        if _is_exec(cand):
            print(f"[bin] Found {name} in candidates: {cand}")
            return cand
    for cand in _tokens_from_whereis(name):
        if _is_exec(cand):
            print(f"[bin] Found {name} via whereis: {cand}")
            return cand
    print(f"[bin][WARN] Could not locate '{name}'.")
    return None


def _safe_tsne_perplexity(n: int) -> float:
    # Mantém perplexidade válida e razoável (evita ValueError)
    return max(5.0, min(30.0, (n - 1) / 3.0))


# =============================
# Checkpoints
# =============================

def checkpoint_exists(path: str) -> bool:
    return os.path.exists(path)


def write_checkpoint(path: str, info: Optional[str] = None) -> None:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            if info is not None:
                f.write(str(info) + '\n')
    except Exception as e:
        print(f"[checkpoint][WARN] Could not write checkpoint {path}: {e}")


# =============================
# Normalização de IDs
# =============================

def normalize_seq_id(s: Optional[str]) -> str:
    """
    Normaliza IDs de sequência para garantir match FASTA ↔ tabela.

    Regras:
      - remove '>' se existir
      - remove espaços nas extremidades
      - pega só o primeiro token antes de espaço/tab
      - se existir '|', pega só o que vem antes do primeiro '|'

    Exemplos:
      '>WP_193920795.1 GCF_015207005.2'  -> 'WP_193920795.1'
      'WP_193920795.1|GCF_015207005.2'   -> 'WP_193920795.1'
    """
    if not isinstance(s, str):
        return ''
    s = s.strip()
    if s.startswith('>'):
        s = s[1:]
    # primeiro token
    s = s.split()[0]
    # se tiver pipes, ficar só com o primeiro bloco
    if '|' in s:
        s = s.split('|')[0]
    return s


# =============================
# IO FASTA (mínima)
# =============================

def read_fasta_minimal(fasta_path: str) -> pd.DataFrame:
    """
    Lê FASTA e aplica normalize_seq_id ao header para manter consistência com a tabela.
    """
    ids, seqs = [], []
    with open(fasta_path, 'r', encoding='utf-8') as f:
        cur, buf = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur is not None:
                    seqs.append(''.join(buf))
                    buf = []
                header = line[1:]
                cur = normalize_seq_id(header)
                ids.append(cur)
            else:
                buf.append(line)
        if cur is not None:
            seqs.append(''.join(buf))
    return pd.DataFrame({'sequence_id': ids, 'sequence': seqs})


def write_fasta(df: pd.DataFrame, out_fa: str) -> None:
    with open(out_fa, 'w', encoding='utf-8') as f:
        for _, r in df.iterrows():
            sid = str(r['sequence_id'])
            seq = str(r['sequence'])
            f.write(">" + sid + "\n" + seq + "\n")


# =============================
# Phylum helpers
# =============================

def normalize_phylum_name(s: Optional[str]) -> str:
    if not isinstance(s, str) or not s.strip():
        return 'Unknown'
    t = s.strip()
    if t.lower() == 'actinobacteria':
        return 'Actinomycetota'
    return t


def _extract_phylum_from_lineage(lineage: str) -> str:
    if not isinstance(lineage, str) or not lineage.strip():
        return 'Unknown'
    toks = [t.strip() for t in lineage.split(';') if t.strip()]
    if len(toks) >= 2:
        ph = toks[1]
        return normalize_phylum_name(ph) if ph and ph.lower() not in DOMAIN_TOKS else 'Unknown'
    if len(toks) == 1:
        first = toks[0]
        if first.lower() in DOMAIN_TOKS:
            return 'Unknown'
        return normalize_phylum_name(first)
    return 'Unknown'


def read_table_s2(table_path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Lê tabela de metadados (por ex. NCBI) e extrai (sequence_id, phylum).

    Funciona com tabelas com cabeçalhos como:
      - 'Protein Accession'
      - 'Species Lineage'
      - 'Assembly BioProject Lineage Title'
      - 'Organism Name', etc.

    Estratégia:
      1. Detectar coluna de ID (Protein Accession, Protein_id, Sequence ID, etc.).
      2. Detectar coluna de lineage (Species Lineage, *Lineage*, Taxonomy, ...).
      3. Extrair phylum a partir da lineage.
      4. Se não houver lineage, tentar coluna explícita de phylum.
    """
    if not table_path or not os.path.exists(table_path):
        return None

    ext = Path(table_path).suffix.lower()
    if ext == '.tsv':
        df = pd.read_csv(table_path, sep='\t', dtype=str, engine='python')
    else:
        try:
            df = pd.read_csv(table_path, sep=None, dtype=str, engine='python')
        except Exception:
            df = pd.read_csv(table_path, dtype=str)

    # Map de nomes em minúsculas/strip -> nome original
    cols_lc = {c.lower().strip(): c for c in df.columns}

    # 1) Coluna de ID
    id_priority = [
        'protein accession',
        'protein_accession',
        'protein accession version',
        'protein_id',
        'protein id',
        'signature.accession',
        'sequence id',
        'sequence_id',
        'seqid',
        'accession',
        'id',
    ]
    acc_col = None
    for key in id_priority:
        if key in cols_lc:
            acc_col = cols_lc[key]
            break
    if acc_col is None:
        # fallback: qualquer coluna contendo 'accession'
        for c in df.columns:
            if 'accession' in c.lower():
                acc_col = c
                break

    # 2) Coluna de lineage (preferindo Species Lineage, etc.)
    lineage_priority = [
        'species lineage',
        'assembly bioproject lineage title',
        'lineage',
        'taxonomic lineage',
        'taxonomic_lineage',
        'ncbi_lineage',
        'full_lineage',
        'taxonomy',
        'taxon_lineage',
    ]
    lin_col = None
    for key in lineage_priority:
        if key in cols_lc:
            lin_col = cols_lc[key]
            break
    if lin_col is None:
        # fallback geral: primeira coluna com 'lineage' no nome
        for c in df.columns:
            if 'lineage' in c.lower():
                lin_col = c
                break

    # Normaliza IDs se tivermos uma coluna de acesso
    if acc_col is not None:
        df[acc_col] = df[acc_col].astype(str).map(normalize_seq_id)

    # Caminho preferencial: ID + lineage
    if acc_col is not None and lin_col is not None:
        tmp = df[[acc_col, lin_col]].copy().rename(columns={acc_col: 'sequence_id',
                                                            lin_col: 'Lineage'})
        tmp['phylum'] = tmp['Lineage'].apply(_extract_phylum_from_lineage).apply(normalize_phylum_name)
        tmp = tmp[['sequence_id', 'phylum']].dropna().drop_duplicates()
        return tmp

    # 3) Fallback: coluna explícita de phylum
    phylum_priority = ['phylum', 'tax_phylum', 'taxonomy_phylum']
    ph_col = None
    for key in phylum_priority:
        if key in cols_lc:
            ph_col = cols_lc[key]
            break

    if ph_col is not None:
        if acc_col is not None:
            tmp = df[[acc_col, ph_col]].copy().rename(columns={acc_col: 'sequence_id',
                                                               ph_col: 'phylum'})
        elif 'sequence_id' in df.columns:
            df['sequence_id'] = df['sequence_id'].astype(str).map(normalize_seq_id)
            tmp = df[['sequence_id', ph_col]].copy().rename(columns={ph_col: 'phylum'})
        else:
            return None

        tmp['sequence_id'] = tmp['sequence_id'].astype(str).map(normalize_seq_id)
        tmp['phylum'] = tmp['phylum'].apply(
            lambda s: 'Unknown'
            if isinstance(s, str) and s.lower().strip() in DOMAIN_TOKS
            else normalize_phylum_name(s)
        )
        return tmp[['sequence_id', 'phylum']].dropna().drop_duplicates()

    # Se nada foi encontrado, retorna None
    return None


def merge_phylum_info(seqs_df: pd.DataFrame, meta_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Faz o merge entre as sequências do FASTA e os metadados (phylum),
    garantindo normalização dos IDs nos dois lados e reportando estatísticas.
    """
    out = seqs_df.copy()
    out['sequence_id'] = out['sequence_id'].astype(str).map(normalize_seq_id)

    if meta_df is None or meta_df.empty:
        out['phylum'] = 'Unknown'
        print("[meta] No metadata table provided or empty. All phylum set to 'Unknown'.")
        return out

    meta = meta_df.copy()
    meta['sequence_id'] = meta['sequence_id'].astype(str).map(normalize_seq_id)
    meta = meta[['sequence_id', 'phylum']].dropna().drop_duplicates()

    out = out.merge(meta, on='sequence_id', how='left')
    out['phylum'] = out['phylum'].fillna('Unknown').apply(normalize_phylum_name)

    n_tot = len(out)
    n_mapped = int((out['phylum'] != 'Unknown').sum())
    print(f"[meta] Phylum successfully mapped for {n_mapped}/{n_tot} sequences "
          f"({100.0 * n_mapped / max(1, n_tot):.1f}%).")

    return out


# =============================
# Paletas e filtros
# =============================

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
    if n <= 0:
        return []
    base = []
    for name in ['tab20', 'tab20b', 'tab20c']:
        cmap = cm.get_cmap(name)
        base += list(cmap.colors) if hasattr(cmap, 'colors') else [cmap(i) for i in np.linspace(0, 1, 20)]
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


def _is_excluded_phylum(s: Optional[str]) -> bool:
    if not isinstance(s, str):
        return True
    sl = s.strip().lower()
    if sl == '':
        return True
    return ('unknown' in sl) or ('uncultured' in sl) or ('uncultivated' in sl) or ('uncultived' in sl)


def build_global_phylum_palette(seqs_df: pd.DataFrame) -> Dict[str, tuple]:
    uniq = sorted({normalize_phylum_name(p)
                   for p in seqs_df['phylum'].astype(str).unique()
                   if not _is_excluded_phylum(p)})
    cols = _distinct_palette(len(uniq))
    return {p: cols[i] for i, p in enumerate(uniq)}


# =============================
# Subsets
# =============================

@dataclass
class SubsetParams:
    size: Optional[int] = None
    fraction: Optional[float] = None
    seed: int = 42


def apply_subset_random(seqs_df: pd.DataFrame, params: SubsetParams) -> Tuple[pd.DataFrame, str]:
    n = len(seqs_df)
    if params.size is None and params.fraction is None:
        return seqs_df.copy(), 'full'
    rng = random.Random(params.seed)
    if params.size is not None:
        k = min(n, int(params.size))
    else:
        k = max(1, min(n, int(round(n * float(params.fraction)))))
    idx = list(range(n))
    rng.shuffle(idx)
    take = sorted(idx[:k])
    out = seqs_df.iloc[take].copy()
    label = f"size_{k}" if params.size is not None else f"frac_{params.fraction:.2f}".replace('.', 'p')
    return out, label


# =============================
# MAFFT & Identities
# =============================

def run_mafft_msa(seqs_df: pd.DataFrame, run_dir: str, mafft_bin: str,
                  mafft_opts: Optional[List[str]] = None) -> str:
    ensure_outdir(run_dir)
    fa_in = os.path.join(run_dir, 'sequences_for_mafft.fasta')
    fa_out = os.path.join(run_dir, 'sequences_mafft_aligned.fasta')
    write_fasta(seqs_df, fa_in)
    if mafft_opts is None:
        mafft_opts = ['--auto', '--thread', str(os.cpu_count() or 1)]
    cmd = [mafft_bin] + list(mafft_opts) + [fa_in]
    print('[mafft]', ' '.join(cmd))
    with open(fa_out, 'w', encoding='utf-8') as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed:\n{proc.stderr}")
    return fa_out


def read_aligned_fasta_to_array(fa_aligned: str) -> Tuple[List[str], np.ndarray]:
    ids, seqs = [], []
    if HAS_BIO:
        for rec in SeqIO.parse(fa_aligned, 'fasta'):
            ids.append(str(rec.id))
            seqs.append(str(rec.seq))
    else:
        with open(fa_aligned, 'r', encoding='utf-8') as f:
            cur, buf = None, []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if cur is not None:
                        seqs.append(''.join(buf))
                        buf = []
                    cur = line[1:].split()[0]
                    ids.append(cur)
                else:
                    buf.append(line)
            if cur is not None:
                seqs.append(''.join(buf))
    arr = np.array([list(s) for s in seqs], dtype='<U1')
    return ids, arr


def msa_identities_allpairs_to_csv(ids: List[str], arr: np.ndarray, out_csv: str) -> None:
    n = len(ids)
    ensure_outdir(os.path.dirname(out_csv))
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('seq_i,seq_j,identity_percent\n')
        for i in range(n):
            Ai = arr[i]
            for j in range(i + 1, n):
                Aj = arr[j]
                both = (Ai != '-') & (Aj != '-')
                denom = int(both.sum())
                idp = 0.0 if denom == 0 else 100.0 * int(((Ai == Aj) & both).sum()) / denom
                f.write(f"{ids[i]},{ids[j]},{idp:.6f}\n")


# =============================
# MMseqs2
# =============================

def mmseqs_prepare_db(mmseqs_bin: str, fasta_path: str, run_dir: str) -> Tuple[str, str]:
    db = os.path.join(run_dir, 'mmseqs_db')
    tmp = os.path.join(run_dir, 'mmseqs_tmp')
    ensure_outdir(tmp)
    subprocess.run([mmseqs_bin, 'createdb', fasta_path, db], check=True)
    subprocess.run([mmseqs_bin, 'createindex', db, tmp], check=True)
    return db, tmp


def mmseqs_cluster_for_cutoff(mmseqs_bin: str, db: str, tmp: str, run_dir: str,
                              cutoff: float, extra_opts: Optional[List[str]] = None) -> str:
    outbase = os.path.join(run_dir, f"mmseqs_clu_{int(cutoff)}")
    out = outbase
    min_id = float(cutoff) / 100.0
    cmd = [mmseqs_bin, 'cluster', db, out, tmp, '--min-seq-id', str(min_id)]
    if extra_opts is None:
        extra_opts = ['--cov-mode', '0']
    cmd += extra_opts
    print('[mmseqs]', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tsv = outbase + '.tsv'
        cmd_tsv = [mmseqs_bin, 'createtsv', db, db, out, tsv]
        subprocess.run(cmd_tsv, check=True, capture_output=True, text=True)
        return tsv
    except subprocess.CalledProcessError as e:
        print('[mmseqs][cluster][error]:', (e.stderr or e.stdout or str(e)))
        fasta_in = os.path.join(run_dir, 'sequences_for_mmseqs.fasta')
        easy_dir = os.path.join(run_dir, f"easy_{int(cutoff)}")
        ensure_outdir(easy_dir)
        easy_cmd = [mmseqs_bin, 'easy-cluster', fasta_in, easy_dir, tmp,
                    '--min-seq-id', str(min_id)] + extra_opts
        subprocess.run(easy_cmd, check=True, capture_output=True, text=True)
        for cand in [os.path.join(easy_dir, 'cluster.tsv'),
                     easy_dir + '_cluster.tsv',
                     os.path.join(easy_dir, 'easy_cluster.tsv')]:
            if os.path.exists(cand):
                return cand
        return outbase + '.tsv'


def parse_mmseqs_tsv_to_clusters(tsv_path: str, cutoff: float) -> pd.DataFrame:
    rows = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            rep, mem = parts[0], parts[1]
            rows.append({'sequence_id': mem, 'cutoff': float(cutoff), 'cluster_id': rep})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['cluster_size'] = df.groupby(['cutoff', 'cluster_id'])['sequence_id'].transform('size')
    return df


def generate_clusters_with_mmseqs(mmseqs_bin: str, seqs_df: pd.DataFrame, run_dir: str,
                                  cutoffs: List[int], extra_opts: Optional[List[str]] = None) -> pd.DataFrame:
    fasta_in = os.path.join(run_dir, 'sequences_for_mmseqs.fasta')
    write_fasta(seqs_df, fasta_in)
    db, tmp = mmseqs_prepare_db(mmseqs_bin, fasta_in, run_dir)
    parts = []
    for c in cutoffs:
        tsv = mmseqs_cluster_for_cutoff(mmseqs_bin, db, tmp, run_dir, c, extra_opts=extra_opts)
        parts.append(parse_mmseqs_tsv_to_clusters(tsv, c))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['sequence_id', 'cutoff', 'cluster_id', 'cluster_size'])
    out_csv = os.path.join(run_dir, 'clusters_multi_threshold.csv')
    out.to_csv(out_csv, index=False)
    print('[ok] clusters ->', out_csv)
    return out


# =============================
# Identities helpers
# =============================

def build_dissimilarity_matrix_from_pairs(ids: List[str], pairs_df: pd.DataFrame) -> np.ndarray:
    n = len(ids)
    D = np.ones((n, n), dtype=float)
    np.fill_diagonal(D, 0.0)
    index = {sid: i for i, sid in enumerate(ids)}
    for _, row in pairs_df.iterrows():
        a, b, idp = row['seq_i'], row['seq_j'], float(row['identity_percent'])
        if a in index and b in index:
            i, j = index[a], index[b]
            d = 1.0 - (idp / 100.0)
            if d < D[i, j]:
                D[i, j] = D[j, i] = d
    return D


# =============================
# W2V — extração e treinamento
# =============================

def build_kmer_sentences_from_msa(
    ids: List[str],
    arr: np.ndarray,
    k: int,
    step_size: int = 1
) -> Dict[str, List[str]]:
    tokens_by_seq: Dict[str, List[str]] = {}
    for i, sid in enumerate(ids):
        seq = ''.join(arr[i])
        if len(seq) < k:
            tokens_by_seq[sid] = ['PAD']
            continue
        kmers = [seq[j:j + k] for j in range(0, len(seq) - k + 1, step_size)]
        kmers = [kmer for kmer in kmers if kmer.count('-') < k]
        tokens_by_seq[sid] = kmers if kmers else ['PAD']
    return tokens_by_seq


def train_w2v_model(
    tokens_by_seq: Dict[str, List[str]],
    vector_size: int,
    window: int,
    min_count: int,
    sg: int,
    hs: int,
    negative: int,
    workers: int,
    epochs: int
):
    if not HAS_GENSIM:
        warnings.warn('gensim not available — skipping Part B (W2V).')
        return None
    workers = min(workers, os.cpu_count() or 2)
    sentences = list(tokens_by_seq.values())
    if len(sentences) == 0:
        warnings.warn('No token sentences available for W2V.')
        return None
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        hs=hs,
        negative=negative,
        workers=workers,
        epochs=epochs,
        seed=RANDOM_STATE
    )
    return model


def build_w2v_mean_embeddings_with_min_kmers(
    tokens_by_seq: Dict[str, List[str]],
    model,
    run_dir: str,
    min_kmers: int
) -> pd.DataFrame:
    sent_dir = os.path.join(run_dir, 'w2v_sentences')
    ensure_outdir(sent_dir)
    rows = []
    for sid, toks in tokens_by_seq.items():
        toks = toks or []
        selected = toks[:min_kmers] if len(toks) >= min_kmers else (toks + ['PAD'] * (min_kmers - len(toks)))
        vecs = []
        for t in selected:
            if t in model.wv:
                vecs.append(model.wv[t])
            else:
                vecs.append(np.zeros((model.vector_size,), dtype=float))
        emb = np.stack(vecs, axis=0) if len(vecs) else np.zeros((1, model.vector_size), dtype=float)
        np.save(os.path.join(sent_dir, f'{sid}.npy'), emb)
        mean_vec = emb.mean(axis=0)
        row = {'sequence_id': sid}
        for i, v in enumerate(mean_vec):
            row[f'f{i:03d}'] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.DataFrame({
        'sequence_id': list(tokens_by_seq.keys()),
        'num_kmers': [len(tokens_by_seq[s]) for s in tokens_by_seq.keys()]
    }).to_csv(os.path.join(run_dir, 'w2v_sentence_index.csv'), index=False)
    return df


# =============================
# Plots (identidade)
# =============================

def plot_bar_identity_distribution(pairs_df: pd.DataFrame, outdir: str) -> Dict[str, str]:
    ensure_outdir(outdir)
    if pairs_df is None or pairs_df.empty:
        return {}
    ids = pairs_df['identity_percent'].to_numpy(dtype=float)
    bins = np.arange(10, 101, 5)
    hist, edges = np.histogram(ids, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    plt.figure(figsize=(10.5, 6.4))
    plt.bar(centers, hist, width=4.8, edgecolor='black')
    plt.xlabel('Identity (%)')
    plt.ylabel('Pair count (all pairs)')
    plt.xlim(10, 100)
    out = os.path.join(outdir, '01_bar_identity_10_100')
    return _save_multiformat(out)


def _counts_intra_by_cutoff_deciles(pairs_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(10, 101, 10)
    rows = []
    if pairs_df.empty or clusters_df.empty:
        return pd.DataFrame(rows)
    P = pairs_df.copy()
    P['a'] = P[['seq_i', 'seq_j']].min(axis=1)
    P['b'] = P[['seq_i', 'seq_j']].max(axis=1)
    P = P.drop(columns=['seq_i', 'seq_j']).drop_duplicates(['a', 'b'])
    for c in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        mem = clusters_df[clusters_df['cutoff'] == c][['sequence_id', 'cluster_id']].drop_duplicates()
        if mem.empty:
            continue
        m = (
            P.merge(mem.rename(columns={'sequence_id': 'a'}), on='a', how='inner')
             .merge(mem.rename(columns={'sequence_id': 'b', 'cluster_id': 'cluster_id_b'}),
                    on='b', how='inner')
        )
        m = m[m['cluster_id'] == m['cluster_id_b']]
        if m.empty:
            continue
        vals = m['identity_percent'].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        hist, edges = np.histogram(vals, bins=bins)
        for b_idx in range(len(hist)):
            rows.append({
                'cutoff': int(c),
                'bin_left': float(edges[b_idx]),
                'bin_right': float(edges[b_idx + 1]),
                'x_value': int(edges[b_idx]),
                'count': int(hist[b_idx]),
            })
    return pd.DataFrame(rows)


def plot_intra_identity_counts_by_cutoff_lines(pairs_df: pd.DataFrame, clusters_df: pd.DataFrame,
                                               outdir: str) -> Dict[str, str]:
    ensure_outdir(outdir)
    df = _counts_intra_by_cutoff_deciles(pairs_df, clusters_df)
    if df.empty:
        return {}
    plt.figure(figsize=(11.5, 7.8))
    x_ticks = sorted(df['x_value'].unique())
    for c in sorted(df['cutoff'].unique()):
        sub = df[df['cutoff'] == c].sort_values('x_value')
        plt.plot(sub['x_value'], sub['count'], marker='o', label=f'{c}%')
    plt.xticks(x_ticks, [str(x) for x in x_ticks], rotation=0)
    plt.xlabel('Identity (%) — decile bins left edges (10, 20, …)')
    plt.ylabel('Intra-cluster pair count')
    plt.legend(bbox_to_anchor=(0.5, -0.16), loc='upper center', ncol=7)
    out = os.path.join(outdir, '06a_counts_intra_identity_bins_by_cutoff_lines')
    return _save_multiformat(out)


def plot_intra_identity_counts_by_cutoff_heatmap(pairs_df: pd.DataFrame, clusters_df: pd.DataFrame,
                                                 outdir: str) -> Dict[str, str]:
    ensure_outdir(outdir)
    df = _counts_intra_by_cutoff_deciles(pairs_df, clusters_df)
    if df.empty:
        return {}
    piv = df.pivot_table(index='cutoff', columns='x_value', values='count',
                         aggfunc='sum', fill_value=0).sort_index()
    xvals = list(piv.columns)
    plt.figure(figsize=(max(10.5, piv.shape[1] * 0.46), 7.9))
    plt.imshow(piv.to_numpy(), aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Count')
    plt.yticks(range(piv.shape[0]), piv.index)
    plt.xticks(range(piv.shape[1]), [str(x) for x in xvals], rotation=0)
    plt.xlabel('Identity (%) — decile bins (10, 20, …)')
    plt.ylabel('Cutoff (%)')
    out = os.path.join(outdir, '06b_counts_intra_identity_bins_by_cutoff_heatmap')
    return _save_multiformat(out)


# =============================
# UMAP/TSNE (identidade)
# =============================

def add_umap_annotation(ax, distance_method_text: str, sequence_subset_label: str,
                        pair_sample_label: Optional[str] = None):
    lines = [f"Distance method: {distance_method_text}"]
    if sequence_subset_label:
        lines.append(f"Sequence subset: {sequence_subset_label}")
    if pair_sample_label:
        lines.append(f"Pair sample: {pair_sample_label}")
    text = "\n".join(lines)
    ax.text(0.99, 0.99, text, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.35',
                                  facecolor='white', edgecolor='black', alpha=0.8))


def build_info_suffix(distance_method_text: Optional[str],
                      subset_label: Optional[str],
                      pair_sample_label: Optional[str]) -> str:
    def _slugify(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
        return s

    tokens = []
    if distance_method_text:
        tokens.append(f"dist-{_slugify(distance_method_text)}")
    if subset_label:
        tokens.append(f"subset-{_slugify(subset_label)}")
    if pair_sample_label:
        tokens.append(f"pairs-{_slugify(pair_sample_label.replace(',', ''))}")
    return ("__" + "_".join(tokens)) if tokens else ""


def plot_umap_from_pairs(pairs_df: pd.DataFrame, seqs_df: pd.DataFrame, out_dir: str,
                         sequence_subset_label: str,
                         annotate_on_canvas: bool = True,
                         append_info_to_filename: bool = False) -> Dict[str, str]:
    ensure_outdir(out_dir)
    if pairs_df is None or pairs_df.empty or not HAS_UMAP:
        return {}
    ids_all = pd.unique(pd.concat([pairs_df['seq_i'], pairs_df['seq_j']],
                                  ignore_index=True)).tolist()
    present = set(seqs_df['sequence_id'].tolist())
    ids_all = [sid for sid in ids_all if sid in present]
    if len(ids_all) < 2:
        return {}
    phylum_map = seqs_df.set_index('sequence_id')['phylum'].to_dict()
    ids_filtered = [sid for sid in ids_all
                    if not _is_excluded_phylum(phylum_map.get(sid, 'Unknown'))]
    if len(ids_filtered) < 2:
        ids_filtered = ids_all
    D = build_dissimilarity_matrix_from_pairs(ids_filtered, pairs_df)
    nn = max(5, min(20, len(ids_filtered) - 1))
    reducer = umap.UMAP(metric='precomputed', random_state=RANDOM_STATE,
                        n_neighbors=nn, min_dist=0.05)
    Z = reducer.fit_transform(D)
    phyla = sorted({phylum_map.get(sid, 'Unknown') for sid in ids_filtered})
    colors = _distinct_palette(len(phyla)) if len(phyla) > 0 else [(0, 0, 0, 1)]
    cmap = {p: colors[i] for i, p in enumerate(phyla)}
    point_colors = [cmap.get(phylum_map.get(sid, 'Unknown'),
                             (0.5, 0.5, 0.5, 1)) for sid in ids_filtered]
    plt.figure(figsize=(13.0, 13.0))
    ax = plt.gca()
    ax.scatter(Z[:, 0], Z[:, 1], c=point_colors, s=18, alpha=0.95, linewidths=0)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', label=p,
                      markerfacecolor=cmap[p], markersize=6)
               for p in phyla]
    if handles:
        ax.legend(handles=handles, title='Phylum',
                  bbox_to_anchor=(0.5, -0.12), loc='upper center',
                  ncol=min(6, max(1, len(phyla))))
    distance_text = "Alignment dissimilarity (1 − identity)"
    if annotate_on_canvas:
        add_umap_annotation(ax, distance_method_text=distance_text,
                            sequence_subset_label=sequence_subset_label.replace('_', ' '),
                            pair_sample_label=None)
    base = os.path.join(out_dir, '04_umap_dissimilarity_by_phylum')
    if append_info_to_filename:
        suffix = build_info_suffix(distance_text, sequence_subset_label, None)
        base = base + suffix
    return _save_multiformat(base)


def plot_tsne_from_pairs(pairs_df: pd.DataFrame, seqs_df: pd.DataFrame, out_dir: str,
                         sequence_subset_label: str) -> Dict[str, str]:
    ensure_outdir(out_dir)
    if pairs_df is None or pairs_df.empty:
        return {}
    ids_all = pd.unique(pd.concat([pairs_df['seq_i'], pairs_df['seq_j']],
                                  ignore_index=True)).tolist()
    present = set(seqs_df['sequence_id'].tolist())
    ids_all = [sid for sid in ids_all if sid in present]
    if len(ids_all) < 2:
        return {}
    phylum_map = seqs_df.set_index('sequence_id')['phylum'].to_dict()
    ids_filtered = [sid for sid in ids_all
                    if not _is_excluded_phylum(phylum_map.get(sid, 'Unknown'))]
    if len(ids_filtered) < 2:
        ids_filtered = ids_all
    D = build_dissimilarity_matrix_from_pairs(ids_filtered, pairs_df)
    perp = _safe_tsne_perplexity(len(ids_filtered))
    # metric='precomputed' → init não pode ser 'pca'
    reducer = TSNE(
        n_components=2,
        metric='precomputed',
        random_state=RANDOM_STATE,
        init='random',
        learning_rate='auto',
        perplexity=perp
    )
    Z = reducer.fit_transform(D)
    phyla = sorted({phylum_map.get(sid, 'Unknown') for sid in ids_filtered})
    colors = _distinct_palette(len(phyla)) if len(phyla) > 0 else [(0, 0, 0, 1)]
    cmap = {p: colors[i] for i, p in enumerate(phyla)}
    point_colors = [cmap.get(phylum_map.get(sid, 'Unknown'),
                             (0.5, 0.5, 0.5, 1)) for sid in ids_filtered]
    plt.figure(figsize=(13.0, 13.0))
    ax = plt.gca()
    ax.scatter(Z[:, 0], Z[:, 1], c=point_colors, s=18, alpha=0.95, linewidths=0)
    ax.set_xlabel('t-SNE-1')
    ax.set_ylabel('t-SNE-2')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', label=p,
                      markerfacecolor=cmap[p], markersize=6)
               for p in phyla]
    if handles:
        ax.legend(handles=handles, title='Phylum',
                  bbox_to_anchor=(0.5, -0.12), loc='upper center',
                  ncol=min(6, max(1, len(phyla))))
    base = os.path.join(out_dir, '11_tsne_dissimilarity_by_phylum')
    return _save_multiformat(base)


# =============================
# Dendrogramas (identidade e W2V)
# =============================

def plot_dendrogram_from_pairs(pairs_df: pd.DataFrame, seqs_df: pd.DataFrame, outdir: str,
                               method: str = 'average',
                               max_leaves: Optional[int] = None) -> Dict[str, str]:
    ensure_outdir(outdir)
    if not HAS_SCIPY or pairs_df is None or pairs_df.empty:
        return {}
    ph_map = seqs_df.set_index('sequence_id')['phylum'].to_dict()
    ids = pd.unique(pd.concat([pairs_df['seq_i'], pairs_df['seq_j']],
                              ignore_index=True)).tolist()
    present = set(seqs_df['sequence_id'].tolist())
    ids = [sid for sid in ids if sid in present]
    ids_filt = [sid for sid in ids
                if not _is_excluded_phylum(ph_map.get(sid, 'Unknown'))]
    if len(ids_filt) >= 3:
        ids = ids_filt
    if len(ids) < 3:
        return {}
    if max_leaves is not None and len(ids) > max_leaves:
        rng = np.random.default_rng(RANDOM_STATE)
        ids = list(sorted(rng.choice(ids, size=max_leaves, replace=False)))
    D = build_dissimilarity_matrix_from_pairs(ids, pairs_df)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    n = len(ids)
    leaf_phyla = {i: {ph_map.get(ids[i], 'Unknown')} for i in range(n)}
    node_sets = dict(leaf_phyla)
    for idx, (a, b, _, _) in enumerate(Z):
        a = int(a)
        b = int(b)
        node = n + idx
        node_sets[node] = node_sets.get(a, set()) | node_sets.get(b, set())
    uniq = sorted(list({ph_map.get(sid, 'Unknown')
                        for sid in ids
                        if not _is_excluded_phylum(ph_map.get(sid, 'Unknown'))}))
    pal_list = _distinct_palette(len(uniq))
    palette = {p: pal_list[i] for i, p in enumerate(uniq)}
    hexmap = {k: mpl_colors.to_hex((v[0], v[1], v[2], 1.0), keep_alpha=False)
              for k, v in palette.items()}

    def link_color_func(k):
        s = node_sets.get(int(k), set())
        if len(s) == 1:
            ph = next(iter(s))
            return hexmap.get(ph, '#999999')
        return '#B0B0B0'

    plt.figure(figsize=(14, 6.5))
    dendrogram(Z, no_labels=True, color_threshold=0,
               above_threshold_color='#B0B0B0',
               link_color_func=link_color_func)
    plt.xlabel('Sequences (leaves)')
    plt.ylabel('Distance (1 − identity)')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=hexmap.get(p, '#999999'),
                      lw=2, label=p) for p in uniq]
    if handles:
        plt.legend(handles=handles,
                   bbox_to_anchor=(0.5, -0.18), loc='upper center',
                   ncol=min(6, max(1, len(handles))))
    out = os.path.join(outdir, '03_dendrogram_branches_by_phylum')
    return _save_multiformat(out)


def plot_dendrogram_from_w2v_means(emb_df: pd.DataFrame, outdir: str,
                                   palette: Optional[Dict[str, tuple]] = None,
                                   metric: str = 'cosine',
                                   method: str = 'average',
                                   max_leaves: Optional[int] = 800) -> Dict[str, str]:
    ensure_outdir(outdir)
    if not HAS_SCIPY or not HAS_SK_PWD:
        return {}
    feat = [c for c in emb_df.columns if c.startswith('f')]
    if not feat:
        return {}
    meta = emb_df[['sequence_id', 'phylum']].copy()
    meta['phylum'] = meta['phylum'].apply(normalize_phylum_name)
    x = emb_df[feat].to_numpy(float)
    mask = ~meta['phylum'].apply(_is_excluded_phylum).to_numpy()
    if mask.sum() < 3:
        mask = np.ones(len(meta), dtype=bool)
    X = x[mask, :]
    M = meta[mask].reset_index(drop=True)
    if len(M) < 3:
        return {}
    if max_leaves is not None and len(M) > max_leaves:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(np.arange(len(M)), size=max_leaves, replace=False))
        X = X[idx, :]
        M = M.iloc[idx].reset_index(drop=True)
    D = pairwise_distances(X, metric=metric)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    n = len(M)
    leaf_phyla = {i: {M.iloc[i]['phylum']} for i in range(n)}
    node_sets = dict(leaf_phyla)
    for idx, (a, b, _, _) in enumerate(Z):
        a = int(a)
        b = int(b)
        node = n + idx
        node_sets[node] = node_sets.get(a, set()) | node_sets.get(b, set())
    uniq = sorted(M['phylum'].astype(str).unique())
    if palette is None:
        pal_list = _distinct_palette(len(uniq))
        palette = {p: pal_list[i] for i, p in enumerate(uniq)}
    hexmap = {k: mpl_colors.to_hex((v[0], v[1], v[2], 1.0), keep_alpha=False)
              for k, v in palette.items()}

    def link_color_func(k):
        s = node_sets.get(int(k), set())
        if len(s) == 1:
            ph = next(iter(s))
            return hexmap.get(ph, '#999999')
        return '#B0B0B0'

    plt.figure(figsize=(14, 6.5))
    dendrogram(Z, no_labels=True, color_threshold=0,
               above_threshold_color='#B0B0B0',
               link_color_func=link_color_func)
    plt.xlabel('Sequences (leaves)')
    yl = 'Cosine distance (1 − cosine sim)' if metric == 'cosine' else f'{metric.capitalize()} distance'
    plt.ylabel(yl)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=hexmap.get(p, '#999999'),
                      lw=2, label=p) for p in uniq]
    if handles:
        plt.legend(handles=handles, bbox_to_anchor=(0.5, -0.18),
                   loc='upper center', ncol=min(6, max(1, len(handles))))
    out = os.path.join(outdir, '13_dendrogram_w2v_mean_by_phylum')
    return _save_multiformat(out)


# =============================
# W2V — UMAP/TSNE de médias
# =============================

def reduce_and_plot_w2v(df_mean: pd.DataFrame, method: str, out_dir: str,
                        sequence_subset_label: str,
                        annotate_on_canvas: bool,
                        append_info_to_filename: bool) -> Dict[str, str]:
    ensure_outdir(out_dir)
    feat = [c for c in df_mean.columns if c.startswith('f')]
    if not feat:
        return {}
    X = df_mean[feat].to_numpy(float)
    meta = df_mean[['sequence_id', 'phylum']].reset_index(drop=True)
    meta['phylum'] = meta['phylum'].apply(normalize_phylum_name)
    filt_mask = ~meta['phylum'].apply(_is_excluded_phylum)
    if filt_mask.sum() < 2:
        filt_mask = pd.Series([True] * len(meta))
    Xf = X[filt_mask.to_numpy()]
    metaf = meta[filt_mask].reset_index(drop=True)
    if len(metaf) < 2:
        return {}

    if method == 'tsne':
        perp = _safe_tsne_perplexity(len(metaf))
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE,
                       init='pca', learning_rate='auto', perplexity=perp)
        Z = reducer.fit_transform(Xf)
        base = os.path.join(out_dir, '09_tsne_w2v_mean_by_phylum')
    else:
        if not HAS_UMAP:
            return {}
        nn = max(5, min(20, Xf.shape[0] - 1))
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                            n_neighbors=nn, min_dist=0.05, metric='euclidean')
        Z = reducer.fit_transform(Xf)
        base = os.path.join(out_dir, '10_umap_w2v_mean_by_phylum')

    df_plot = pd.DataFrame({'x': Z[:, 0], 'y': Z[:, 1], 'phylum': metaf['phylum']})
    uniq = sorted(df_plot['phylum'].astype(str).unique())
    colors = _distinct_palette(len(uniq))
    color_map = {p: colors[i] for i, p in enumerate(uniq)}
    C = [color_map[p] for p in df_plot['phylum']]
    plt.figure(figsize=(12.5, 12.5))
    ax = plt.gca()
    ax.scatter(df_plot['x'], df_plot['y'], s=14, c=C, alpha=0.95, linewidths=0)
    ax.set_xlabel('UMAP-1' if method == 'umap' else 't-SNE-1')
    ax.set_ylabel('UMAP-2' if method == 'umap' else 't-SNE-2')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', label=p,
                      markerfacecolor=color_map[p], markersize=6)
               for p in uniq]
    if handles:
        ax.legend(handles=handles, title='Phylum',
                  bbox_to_anchor=(0.5, -0.12), loc='upper center',
                  ncol=min(6, max(1, len(uniq))))

    if method == 'umap' and annotate_on_canvas:
        distance_text = "Euclidean on Word2Vec mean embeddings"
        add_umap_annotation(ax, distance_method_text=distance_text,
                            sequence_subset_label=sequence_subset_label.replace('_', ' '),
                            pair_sample_label=None)

    if method == 'umap' and append_info_to_filename:
        distance_text = "Euclidean on Word2Vec mean embeddings"
        suffix = build_info_suffix(distance_text, sequence_subset_label, None)
        base = base + suffix

    return _save_multiformat(base)


def plot_tsne_umap_w2v_means(df_mean: pd.DataFrame, out_dir: str, sequence_subset_label: str,
                             annotate_on_canvas: bool, append_info_to_filename: bool) -> None:
    ensure_outdir(out_dir)
    reduce_and_plot_w2v(df_mean, 'tsne', out_dir, sequence_subset_label,
                        annotate_on_canvas=False,
                        append_info_to_filename=False)
    reduce_and_plot_w2v(df_mean, 'umap', out_dir, sequence_subset_label,
                        annotate_on_canvas=annotate_on_canvas,
                        append_info_to_filename=append_info_to_filename)


# =============================
# Cosine vs Identity
# =============================

def save_allpairs_cosine(emb_df: pd.DataFrame, out_csv: str) -> None:
    feat_cols = [c for c in emb_df.columns if c.startswith('f')]
    M = emb_df.set_index('sequence_id')[feat_cols]
    ids = list(M.index)
    X = M.to_numpy(float)
    n = len(ids)
    ensure_outdir(os.path.dirname(out_csv))
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('seq_i,seq_j,cosine\n')
        for i in range(n):
            vi = X[i]
            ni = np.linalg.norm(vi)
            for j in range(i + 1, n):
                vj = X[j]
                nj = np.linalg.norm(vj)
                if ni == 0 or nj == 0:
                    continue
                cos = float(np.dot(vi, vj) / (ni * nj))
                f.write(f"{ids[i]},{ids[j]},{cos:.6f}\n")


def cosine_vs_identity_plots_full_pairs(pairs_df_all: pd.DataFrame, emb_df: pd.DataFrame,
                                        seqs_df: pd.DataFrame, outdir: str
                                        ) -> Tuple[Dict[str, str], Dict[str, str]]:
    ensure_outdir(outdir)
    feat_cols = [c for c in emb_df.columns if c.startswith('f')]
    if not feat_cols:
        return {}, {}
    M = emb_df.set_index('sequence_id')[feat_cols]
    ph = seqs_df.set_index('sequence_id')['phylum'].to_dict()

    rows = []
    present = set(M.index)
    for _, r in pairs_df_all.iterrows():
        a, b = r['seq_i'], r['seq_j']
        if a in present and b in present:
            va = M.loc[a].to_numpy(float)
            vb = M.loc[b].to_numpy(float)
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            if na == 0 or nb == 0:
                continue
            cos = float(np.dot(va, vb) / (na * nb))
            t = 'Intra-phylum' if ph.get(a, 'Unknown') == ph.get(b, 'Unknown') else 'Inter-phylum'
            rows.append({
                'seq_i': a,
                'seq_j': b,
                'identity_percent': float(r['identity_percent']),
                'cosine': cos,
                'type': t
            })
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return {}, {}
    df.to_csv(os.path.join(outdir, 'pairs_identity_cosine_full.csv'), index=False)

    outs = {}
    for t, fname in [('Intra-phylum', '07_cosine_vs_identity_intra'),
                     ('Inter-phylum', '08_cosine_vs_identity_inter')]:
        sub = df[df['type'] == t]
        if sub.empty:
            outs[fname] = {}
            continue
        plt.figure(figsize=(10.2, 7.4))
        hb = plt.hexbin(sub['identity_percent'], sub['cosine'],
                        gridsize=50, mincnt=2)
        plt.xlabel('Real identity (%)')
        plt.ylabel('Cosine similarity (W2V mean)')
        cb = plt.colorbar(hb)
        cb.set_label('Pair count')
        bins = np.arange(10, 101, 10)
        sub2 = sub.copy()
        sub2['bin'] = pd.cut(sub2['identity_percent'], bins=bins,
                             include_lowest=True, right=False)
        grp = sub2.groupby('bin', observed=False).agg(
            x=('identity_percent', 'mean'),
            y=('cosine', 'mean')
        ).dropna()
        if not grp.empty:
            plt.plot(grp['x'], grp['y'], marker='o', linewidth=2)
        out = os.path.join(outdir, fname)
        outs[fname] = _save_multiformat(out)
    return outs.get('07_cosine_vs_identity_intra', {}), outs.get('08_cosine_vs_identity_inter', {})


# =============================
# Orquestração principal (com checkpoints)
# =============================

def run_pipeline_for_subset(
    seqs_full: pd.DataFrame,
    subset_fraction: float,
    outdir_root: str,
    table_s2_df: Optional[pd.DataFrame],
    mafft_bin: str,
    mmseqs_bin: str,
    mafft_opts: Optional[List[str]],
    mmseqs_extra_opts: Optional[List[str]],
    w2v_dims: List[int],
    w2v_epochs_list: List[int],
    run_part_b: bool = True,
    w2v_dendro_max_leaves: Optional[int] = 800
) -> str:
    ensure_outdir(outdir_root)
    subset = SubsetParams(size=None, fraction=subset_fraction, seed=RANDOM_STATE)
    seqs_df, label = apply_subset_random(seqs_full, subset)
    subset_dir = os.path.join(outdir_root, f"subset_{label}")
    ensure_outdir(subset_dir)

    write_fasta(seqs_df, os.path.join(subset_dir, 'input_subset.fasta'))
    seqs_df.to_csv(os.path.join(subset_dir, 'subset_ids.csv'), index=False)
    seqs_df[['sequence_id', 'phylum']].to_csv(os.path.join(subset_dir, 'subset_phylum_map.csv'),
                                              index=False)

    # Checkpoint 1: MAFFT + pairs.csv
    fa_aln = os.path.join(subset_dir, 'sequences_mafft_aligned.fasta')
    pairs_csv = os.path.join(subset_dir, 'pairs.csv')
    mafft_ck = os.path.join(subset_dir, 'CHECKPOINT_mafft_pairs.done')

    if checkpoint_exists(mafft_ck) and os.path.exists(fa_aln) and os.path.exists(pairs_csv):
        print(f"[checkpoint] MAFFT + identity pairs already computed for subset {label}.")
        ids, arr = read_aligned_fasta_to_array(fa_aln)
        pairs_df_all = pd.read_csv(pairs_csv)
    elif os.path.exists(fa_aln) and os.path.exists(pairs_csv):
        print(f"[checkpoint] Found existing MAFFT outputs for subset {label}, assuming done.")
        ids, arr = read_aligned_fasta_to_array(fa_aln)
        pairs_df_all = pd.read_csv(pairs_csv)
        write_checkpoint(mafft_ck, "ok")
    else:
        fa_aln = run_mafft_msa(
            seqs_df,
            subset_dir,
            mafft_bin=mafft_bin,
            mafft_opts=mafft_opts or ['--auto', '--thread', str(os.cpu_count() or 1)]
        )
        ids, arr = read_aligned_fasta_to_array(fa_aln)
        msa_identities_allpairs_to_csv(ids, arr, pairs_csv)
        pairs_df_all = pd.read_csv(pairs_csv)
        write_checkpoint(mafft_ck, "ok")

    # Checkpoint 2: MMseqs2 clusters
    clusters_csv = os.path.join(subset_dir, 'clusters_multi_threshold.csv')
    mmseqs_ck = os.path.join(subset_dir, 'CHECKPOINT_mmseqs_clusters.done')

    if checkpoint_exists(mmseqs_ck) and os.path.exists(clusters_csv):
        print(f"[checkpoint] MMseqs2 clusters already computed for subset {label}.")
        clusters_df = pd.read_csv(clusters_csv)
    elif os.path.exists(clusters_csv):
        print(f"[checkpoint] Found existing MMseqs2 clusters for subset {label}, assuming done.")
        clusters_df = pd.read_csv(clusters_csv)
        write_checkpoint(mmseqs_ck, "ok")
    else:
        clusters_df = generate_clusters_with_mmseqs(
            mmseqs_bin,
            seqs_df,
            subset_dir,
            FIXED_CUTOFFS,
            extra_opts=mmseqs_extra_opts or ['--cov-mode', '0']
        )
        write_checkpoint(mmseqs_ck, "ok")

    global_palette = build_global_phylum_palette(seqs_df)

    # Checkpoint 3: plots de identidade
    plot_dir = os.path.join(subset_dir, 'plots_identity')
    ensure_outdir(plot_dir)
    plots_ck = os.path.join(subset_dir, 'CHECKPOINT_identity_plots.done')

    def plot_cluster_size_vs_cutoff(clusters_df_local: pd.DataFrame, outdir: str):
        if clusters_df_local.empty:
            return False
        agg = clusters_df_local.groupby(['cutoff', 'cluster_id'])['sequence_id'].count() \
            .reset_index(name='cluster_size')
        stat = agg.groupby('cutoff')['cluster_size'] \
            .agg(['mean', 'median', lambda x: np.percentile(x, 95)]).reset_index()
        stat = stat.rename(columns={'<lambda_0>': 'p95'})
        stat.to_csv(os.path.join(outdir, 'cluster_size_stats_by_cutoff.csv'), index=False)

        plt.figure(figsize=(9.6, 6.0))
        plt.plot(stat['cutoff'], stat['mean'], marker='o', label='Mean')
        plt.plot(stat['cutoff'], stat['median'], marker='o', label='Median')
        plt.plot(stat['cutoff'], stat['p95'], marker='o', label='P95')
        plt.xlabel('Identity cutoff (%)')
        plt.ylabel('Cluster size')
        plt.legend(bbox_to_anchor=(0.5, -0.16), loc='upper center', ncol=3)
        plt.grid(True, alpha=0.3)
        _save_multiformat(os.path.join(outdir, '04_cluster_size_vs_identity_cutoff'))

        largest = agg.groupby('cutoff')['cluster_size'].max().reset_index()
        largest.to_csv(os.path.join(outdir, 'largest_cluster_by_cutoff.csv'), index=False)
        plt.figure(figsize=(9.6, 6.0))
        plt.plot(largest['cutoff'], largest['cluster_size'], marker='o')
        plt.xlabel('Identity cutoff (%)')
        plt.ylabel('Largest cluster size')
        plt.grid(True, alpha=0.3)
        _save_multiformat(os.path.join(outdir, '05_largest_cluster_by_identity_cutoff'))
        return True

    if checkpoint_exists(plots_ck):
        print(f"[checkpoint] Identity plots already generated for subset {label}.")
    else:
        print(f"[plots] Generating identity plots for subset {label}...")
        plot_bar_identity_distribution(pairs_df_all, plot_dir)
        plot_umap_from_pairs(
            pairs_df_all,
            seqs_df,
            plot_dir,
            sequence_subset_label=label,
            annotate_on_canvas=True,
            append_info_to_filename=False
        )
        plot_tsne_from_pairs(pairs_df_all, seqs_df, plot_dir, sequence_subset_label=label)
        plot_dendrogram_from_pairs(
            pairs_df_all,
            seqs_df,
            plot_dir,
            method='average',
            max_leaves=800
        )
        plot_intra_identity_counts_by_cutoff_lines(pairs_df_all, clusters_df, plot_dir)
        plot_intra_identity_counts_by_cutoff_heatmap(pairs_df_all, clusters_df, plot_dir)
        plot_cluster_size_vs_cutoff(clusters_df, subset_dir)
        write_checkpoint(plots_ck, "ok")

    # Checkpoint 4: Parte B (W2V)
    if run_part_b and HAS_GENSIM:
        print(f"[W2V] Building k-mer sentences from MSA for subset {label}...")
        tokens_by_seq = build_kmer_sentences_from_msa(ids, arr, k=W2V_K, step_size=W2V_STEP_SIZE)
        kmers_counts = [len(v) for v in tokens_by_seq.values() if len(v) > 0]
        if kmers_counts:
            min_kmers = min(kmers_counts)
            print(f"[W2V] min_kmers: {min_kmers}")
            for dim in (w2v_dims or DEFAULT_W2V_DIMS):
                for epochs in (w2v_epochs_list or DEFAULT_W2V_EPOCHS_LIST):
                    cfg_label = f"dim{dim}_ep{epochs}"
                    w2v_run_dir = os.path.join(subset_dir, f"w2v_{cfg_label}")
                    ensure_outdir(w2v_run_dir)
                    w2v_ck = os.path.join(w2v_run_dir, f"CHECKPOINT_w2v_{cfg_label}.done")

                    if checkpoint_exists(w2v_ck):
                        print(f"[checkpoint][W2V] Skipping W2V config {cfg_label} "
                              f"for subset {label} (already done).")
                        continue

                    print(f"[W2V] Training model ({cfg_label}) for subset {label}...")
                    model = train_w2v_model(
                        tokens_by_seq=tokens_by_seq,
                        vector_size=dim,
                        window=W2V_WINDOW,
                        min_count=W2V_MIN_COUNT,
                        sg=W2V_SG,
                        hs=W2V_HS,
                        negative=W2V_NEGATIVE,
                        workers=W2V_WORKERS,
                        epochs=epochs
                    )
                    if model is None:
                        continue

                    emb_df = build_w2v_mean_embeddings_with_min_kmers(
                        tokens_by_seq, model, w2v_run_dir, min_kmers)
                    emb_df = emb_df.merge(
                        seqs_df[['sequence_id', 'phylum']],
                        on='sequence_id',
                        how='left'
                    ).fillna({'phylum': 'Unknown'})
                    emb_df['phylum'] = emb_df['phylum'].apply(normalize_phylum_name)
                    emb_df.to_csv(
                        os.path.join(w2v_run_dir, f'w2v_mean_embeddings_{cfg_label}.csv'),
                        index=False
                    )

                    cosine_csv = os.path.join(w2v_run_dir, f'pairs_cosine_{cfg_label}.csv')
                    save_allpairs_cosine(emb_df, cosine_csv)

                    plots_dir_w2v = os.path.join(w2v_run_dir, 'plots')
                    ensure_outdir(plots_dir_w2v)
                    cosine_vs_identity_plots_full_pairs(pairs_df_all, emb_df, seqs_df,
                                                        plots_dir_w2v)

                    plot_tsne_umap_w2v_means(
                        emb_df[['sequence_id', 'phylum'] +
                               [c for c in emb_df.columns if c.startswith('f')]],
                        w2v_run_dir,
                        sequence_subset_label=label,
                        annotate_on_canvas=True,
                        append_info_to_filename=False
                    )

                    plot_dendrogram_from_w2v_means(
                        emb_df[['sequence_id', 'phylum'] +
                               [c for c in emb_df.columns if c.startswith('f')]],
                        w2v_run_dir,
                        palette=global_palette,
                        metric='cosine',
                        method='average',
                        max_leaves=w2v_dendro_max_leaves
                    )

                    write_checkpoint(w2v_ck, "ok")

    with open(os.path.join(subset_dir, 'FIGURE_MANIFEST_identity.txt'),
              'w', encoding='utf-8') as f:
        f.write("01_bar_identity_10_100\tBarplot of identity distribution (all pairs)\n")
        f.write("03_dendrogram_branches_by_phylum\tDendrogram (1 − identity)\n")
        f.write("04_umap_dissimilarity_by_phylum\tUMAP (1 − identity)\n")
        f.write("06a_counts_intra_identity_bins_by_cutoff_lines\tLines\n")
        f.write("06b_counts_intra_identity_bins_by_cutoff_heatmap\tHeatmap\n")
        f.write("11_tsne_dissimilarity_by_phylum\tt-SNE (1 − identity)\n")
        f.write("04_cluster_size_vs_identity_cutoff\tMean/Median/P95 vs cutoff\n")
        f.write("05_largest_cluster_by_identity_cutoff\tLargest cluster vs cutoff\n")

    return subset_dir


def run_pipeline(
    sequences_path: str,
    outdir: str,
    table_s2_path: Optional[str] = None,
    subset_fractions: Optional[List[float]] = None,
    mafft_opts: Optional[List[str]] = None,
    mmseqs_extra_opts: Optional[List[str]] = None,
    run_part_b: bool = False,
    w2v_dims: Optional[List[int]] = None,
    w2v_epochs_list: Optional[List[int]] = None,
    mafft_bin: Optional[str] = None,
    mmseqs_bin: Optional[str] = None,
    w2v_dendro_max_leaves: Optional[int] = 800
) -> str:
    ensure_outdir(outdir)
    print('[i] Reading sequences...')
    seqs_full = read_fasta_minimal(sequences_path)
    meta = read_table_s2(table_s2_path)
    seqs_full = merge_phylum_info(seqs_full, meta)
    seqs_full['phylum'] = seqs_full['phylum'].apply(normalize_phylum_name)

    mafft_bin = find_executable('mafft', user_hint=mafft_bin,
                                extra_candidates=['/usr/bin/mafft', '/usr/local/bin/mafft'])
    mmseqs_bin = find_executable('mmseqs', user_hint=mmseqs_bin, extra_candidates=[
        '/usr/bin/mmseqs', '/usr/local/bin/mmseqs',
        os.path.join(os.environ.get('HOME', ''), 'anaconda3', 'envs', 'faal', 'bin', 'mmseqs'),
        os.path.join(os.environ.get('HOME', ''), 'anaconda3', 'envs', 'ai_agent', 'bin', 'mmseqs')
    ])
    if mafft_bin is None:
        raise RuntimeError("MAFFT not found. Set MAFFT_BIN or pass --mafft-bin '/usr/bin/mafft'.")
    if mmseqs_bin is None:
        raise RuntimeError("MMseqs2 not found. Set MMSEQS_BIN or pass --mmseqs-bin '.../bin/mmseqs'.")

    subset_fractions = subset_fractions or DEFAULT_SUBSET_FRACTIONS
    w2v_dims = w2v_dims or DEFAULT_W2V_DIMS
    w2v_epochs_list = w2v_epochs_list or DEFAULT_W2V_EPOCHS_LIST

    results_index = []
    for frac in subset_fractions:
        print(f"\n=== Running subset fraction {frac:.2f} ===")
        subset_dir = run_pipeline_for_subset(
            seqs_full=seqs_full,
            subset_fraction=frac,
            outdir_root=outdir,
            table_s2_df=meta,
            mafft_bin=mafft_bin,
            mmseqs_bin=mmseqs_bin,
            mafft_opts=mafft_opts or ['--auto', '--thread', str(os.cpu_count() or 1)],
            mmseqs_extra_opts=mmseqs_extra_opts or ['--cov-mode', '0'],
            w2v_dims=w2v_dims,
            w2v_epochs_list=w2v_epochs_list,
            run_part_b=run_part_b,
            w2v_dendro_max_leaves=w2v_dendro_max_leaves
        )
        results_index.append({'subset_fraction': frac, 'subset_dir': subset_dir})
    pd.DataFrame(results_index).to_csv(os.path.join(outdir, 'SUBSETS_INDEX.csv'), index=False)

    print('[OK] Done. Outputs in:', outdir)
    return outdir


# =============================
# CLI
# =============================

def _parse_list_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [tok.strip() for tok in s.split(',') if tok.strip()]


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out = []
    for tok in s.split(','):
        tok = tok.strip().replace('_', '')
        if tok:
            out.append(int(tok))
    return out or None


def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out or None


def main(argv: Optional[List[str]] = None) -> int:
    epilog_txt = """
Analyses (PNG/TIFF 900dpi e SVG) por subset e por (dim,epochs):
  Identidade:
    01_bar_identity_10_100
    03_dendrogram_branches_by_phylum
    04_umap_dissimilarity_by_phylum
    06a_counts_intra_identity_bins_by_cutoff_lines
    06b_counts_intra_identity_bins_by_cutoff_heatmap
    11_tsne_dissimilarity_by_phylum
    04_cluster_size_vs_identity_cutoff
    05_largest_cluster_by_identity_cutoff
    pairs.csv (todos os pares com identidade)
    clusters_multi_threshold.csv

  Parte B (W2V) — para cada (dim, epochs):
    w2v_mean_embeddings_dimXXX_epYYYY.csv
    pairs_cosine_dimXXX_epYYYY.csv (todos os pares com cosseno)
    plots/07_cosine_vs_identity_intra  | plots/08_cosine_vs_identity_inter
    09_tsne_w2v_mean_by_phylum         | 10_umap_w2v_mean_by_phylum
    13_dendrogram_w2v_mean_by_phylum
    pairs_identity_cosine_full.csv (tabela identidade×cosseno)

Flags importantes:
  --run-part-b         Ativa a Parte B (Word2Vec). Por padrão, NÃO roda.
"""

    parser = argparse.ArgumentParser(
        prog="faalprot_heterogeneity",
        description="FAALProt heterogeneity pipeline (grade W2V completa por subset) — "
                    "robusto a metadados ausentes.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog_txt
    )
    parser.add_argument('--fasta', required=True, help='Path do FASTA de proteínas.')
    parser.add_argument('--table', default=None, help='Tabela/metadados com Phylum (opcional).')
    parser.add_argument('--outdir', required=True, help='Diretório de saída.')
    parser.add_argument('--subset-fractions', default=None,
                        help='Frações CSV (ex: "0.25,0.5,0.75,1.0").')
    parser.add_argument('--mafft-opts', default=None,
                        help='Opções MAFFT CSV (ex: "--auto,--maxiterate,2").')
    parser.add_argument('--mmseqs-extra-opts', default=None,
                        help='Opções extras MMseqs2 CSV (ex: "--cov-mode,0").')
    parser.add_argument('--run-part-b', action='store_true',
                        help='Se presente, roda a Parte B (W2V). Default: OFF.')
    parser.add_argument('--w2v-dims', default=None,
                        help='Dimensões CSV (ex: "100,200,390").')
    parser.add_argument('--w2v-epochs-list', default=None,
                        help='Epochs CSV (ex: "200,500,1500,2500").')
    parser.add_argument('--w2v-dendro-max-leaves', type=int, default=800,
                        help='Máx. folhas no dendrograma W2V.')
    parser.add_argument('--mafft-bin', default=None, help='Caminho para mafft (override).')
    parser.add_argument('--mmseqs-bin', default=None, help='Caminho para mmseqs (override).')

    args = parser.parse_args(argv)

    mafft_opts = _parse_list_csv(args.mafft_opts) if args.mafft_opts else \
        ['--auto', '--thread', str(os.cpu_count() or 1)]
    mmseqs_extra_opts = _parse_list_csv(args.mmseqs_extra_opts) if args.mmseqs_extra_opts else \
        ['--cov-mode', '0']

    subset_fractions = _parse_float_list(args.subset_fractions) if args.subset_fractions else None
    w2v_dims = _parse_int_list(args.w2v_dims) if args.w2v_dims else None
    w2v_epochs_list = _parse_int_list(args.w2v_epochs_list) if args.w2v_epochs_list else None

    run_dir = run_pipeline(
        sequences_path=args.fasta,
        outdir=args.outdir,
        table_s2_path=args.table,
        subset_fractions=subset_fractions,
        mafft_opts=mafft_opts,
        mmseqs_extra_opts=mmseqs_extra_opts,
        run_part_b=args.run_part_b,
        w2v_dims=w2v_dims,
        w2v_epochs_list=w2v_epochs_list,
        mafft_bin=args.mafft_bin,
        mmseqs_bin=args.mmseqs_bin,
        w2v_dendro_max_leaves=args.w2v_dendro_max_leaves
    )
    print('Output directory:', run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main()
