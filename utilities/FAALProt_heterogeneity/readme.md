# FAALProt Heterogeneity Analysis

**Complete documentation for the FAAL heterogeneity CLI (`faalprot_heterogeneity_cli_v2.py`) and the underlying approach.**

---

## Table of Contents

- [1. Overview](#1-overview)  
- [2. Approach](#2-approach)  
- [3. Input Data & Metadata](#3-input-data--metadata)  
- [4. Software & Environment](#4-software--environment)  
- [5. Mathematical Definitions](#5-mathematical-definitions)  
  - [5.1 Aligned Sequences](#51-aligned-sequences)  
  - [5.2 Pairwise Identity](#52-pairwise-identity)  
  - [5.3 Dissimilarity Matrix](#53-dissimilarity-matrix)  
  - [5.4 k-mers and Word2Vec Embeddings](#54-k-mers-and-word2vec-embeddings)  
  - [5.5 Identity Bins and Pair Sampling](#55-identity-bins-and-pair-sampling)  
  - [5.6 Correlation Identity vs Cosine Similarity](#56-correlation-identity-vs-cosine-similarity)  
- [6. Pipeline Stages](#6-pipeline-stages)  
  - [6.1 Multiple Sequence Alignment (MAFFT)](#61-multiple-sequence-alignment-mafft)  
  - [6.2 Pairwise Identities & Dissimilarity Matrix](#62-pairwise-identities--dissimilarity-matrix)  
  - [6.3 Subsampling of Sequences and Pairs](#63-subsampling-of-sequences-and-pairs)  
  - [6.4 MMseqs2 Clustering Across Identity Cutoffs](#64-mmseqs2-clustering-across-identity-cutoffs)  
  - [6.5 Heterogeneity Visualizations](#65-heterogeneity-visualizations)  
  - [6.6 Part B — Word2Vec Analyses and Grid Search](#66-part-b--word2vec-analyses-and-grid-search)  
- [7. Color Policy (Distinct Phylum Colors)](#7-color-policy-distinct-phylum-colors)  
- [8. Outputs & Directory Layout](#8-outputs--directory-layout)  
- [9. Reproducibility, Performance & Practical Notes](#9-reproducibility-performance--practical-notes)  
- [10. How to Run — CLI Version](#10-how-to-run--cli-version)  
- [11. FAQ](#11-faq)  
- [12. References](#12-references)  

---

## 1. Overview

This repository provides a complete, publication-grade pipeline to quantify and visualize **heterogeneity in FAAL proteins** across thousands of sequences. It integrates:

- **MAFFT** for multiple sequence alignment (MSA)  
- **Streaming pairwise identity** estimation with deterministic subsampling of the upper triangle  
- **UMAP (precomputed metric)** on *dissimilarity* derived from alignment identity  
- **Hierarchical clustering** and **dendrograms** with **branch colors by Phylum**  
- **MMseqs2 clustering** over fixed identity thresholds (10–90%) and diagnostic plots  
- **Word2Vec (Part B)**: k-mer embeddings from aligned sequences using the **same MSA**, with configurable dimension and epochs (grid search)  

All figures are exported as **PNG (900 dpi)**, **TIFF (900 dpi)** and **SVG**, without titles, and with legends **below** the plots.

> **Note**: Any Phylum string equal to `Actinobacteria` is normalized to `Actinomycetota`. Phyla labelled as `Unknown` / `uncultured` / `uncultivated` are excluded from color-coded plots and legends, though they are still used in numeric computations.

---

## 2. Approach

1. **Read FASTA** and an optional **Table_S2**-like metadata file.  
2. Derive **Phylum** from `Lineage` (if provided) and normalize names.  
3. Define **subsets** of sequences (25%, 50%, 75%, 100%) and run the full pipeline for each subset.  
4. For each subset, run **MAFFT** → compute **pairwise identities** for all pairs (streamed with deterministic sampling).  
5. Convert identity to **dissimilarity** and build UMAP + dendrograms.  
6. Run **MMseqs2 clustering** at fixed identity cutoffs (10–90%) on each subset.  
7. **Part B**: using the *same* aligned sequences, generate **k-mers** (k=3 by default), train **Word2Vec** with:
   - **default configuration**:
     - window = 5  
     - `sg = 1` (skip-gram)  
     - `hs = 0` (hierarchical softmax OFF)  
     - `negative = 5` (negative sampling ON)  
     - `min_count = 1`  
     - `workers = 48`  
   - **dimensions grid**: by default `100` and `390`, but fully configurable via CLI (`--w2v-dims`).  
   - **epochs grid**: configurable via CLI (`--w2v-epochs-list`).  
8. For **every combination** of:
   - subset ∈ {25%, 50%, 75%, 100%},  
   - embedding dimension ∈ {100, 390, ...},  
   - epochs ∈ user-specified list,  

   we generate the full set of figures, including:
   - `03_dendrogram_branches_by_phylum`  
   - `07_cosine_vs_identity_intra`  
   - `08_cosine_vs_identity_inter`  
   - `01_bar_identity_10_100`  
   - `06a_counts_intra_identity_bins_by_cutoff_lines_REALX`  
   - `06b_counts_intra_identity_bins_by_cutoff_heatmap`  
   - identity-based plots and MMseqs2 diagnostics,  

   plus a **summary table** with the Pearson correlation between alignment identity and cosine similarity for each (subset, dim, epochs) combination.

---

## 3. Input Data & Metadata

- **FASTA**: protein sequences (headers’ first token is used as `sequence_id`).  
- **Metadata table (optional)**: tab-separated file with at least:
  - `Protein Accession` or equivalent ID that matches FASTA headers  
  - `Lineage` (e.g., `Bacteria; Pseudomonadota; Gammaproteobacteria; ...`)  

Phylum is extracted from `Lineage` **position 2** if present; otherwise we fall back to a single-token lineage when it is not a domain-level token.

---

## 4. Software & Environment

- **MAFFT** (e.g., `/usr/bin/mafft`)  
- **MMseqs2** (e.g., `<conda-env>/bin/mmseqs`)  
- **Python 3.9+**, with libraries:
  - `numpy`, `pandas`, `matplotlib`  
  - `umap-learn` (for UMAP)  
  - `scikit-learn` (t-SNE, pairwise distances, metrics)  
  - `scipy` (hierarchical clustering/dendrogram)  
  - `biopython` (FASTA parsing; optional but recommended)  
  - `gensim` (Word2Vec; needed for Part B)  

> The CLI tries to auto-locate MAFFT and MMseqs2 via `PATH`, `CONDA_PREFIX`, and `whereis`. You can force paths with `--mafft-bin` and `--mmseqs-bin`.

---

## 5. Mathematical Definitions

### 5.1 Aligned Sequences

Let the aligned sequence for protein \(i\) be

$$
S_i = [s_{i,1}, s_{i,2}, \dots, s_{i,L}],
$$

where each position \(s_{i,t}\) is either an amino acid or a gap:

$$
s_{i,t} \in \{	ext{amino acids}\} \cup \{-\}.
$$

For **Word2Vec**, we first remove gaps from the MSA to obtain a gapped-free sequence:

$$
	ilde{S}_i = [	ilde{s}_{i,1}, 	ilde{s}_{i,2}, \dots, 	ilde{s}_{i,	ilde{L}_i}],
\qquad
	ilde{s}_{i,u} \in \{	ext{amino acids}\},
$$

where \(	ilde{L}_i\) is the length of the non-gapped sequence.

### 5.2 Pairwise Identity

Let \(A_i(t)\) be the residue of sequence \(i\) at aligned position \(t\), and let `'-'` denote a gap.  
The identity between sequences \(i\) and \(j\) is:

$$
\mathrm{Id}(i,j) \;=\; 100 	imes
rac{\displaystyle \sum_{t=1}^L \mathbf{1}ig(A_i(t)=A_j(t),\,A_i(t)
eq -,\,A_j(t)
eq -ig)}
{\displaystyle \sum_{t=1}^L \mathbf{1}ig(A_i(t)
eq -,\,A_j(t)
eq -ig)}.
$$

Here, \(\mathbf{1}(\cdot)\) is the indicator function.

### 5.3 Dissimilarity Matrix

We define a **dissimilarity** \(d_{ij}\) from identity:

$$
d_{ij} = 1 - rac{\mathrm{Id}(i,j)}{100}.
$$

The matrix \(\mathbf{D} = [d_{ij}]\) is used as a **precomputed** distance matrix in UMAP (metric=`precomputed`) and for hierarchical clustering.

### 5.4 k-mers and Word2Vec Embeddings

From each gapped-free sequence \(	ilde{S}_i\), we generate **overlapping k-mers** of size \(k\) (default \(k=3\)) with step size 1:

$$
k_{i,u} = 	ilde{s}_{i,u} 	ilde{s}_{i,u+1} \dots 	ilde{s}_{i,u+k-1},
\qquad
u = 1, \dots, M_i,
$$

where

$$
M_i = 	ilde{L}_i - k + 1
$$

is the number of k-mers for sequence \(i\).  
Each sequence \(S_i\) is thus mapped to a **sentence** of tokens:

$$
\mathcal{K}_i = [k_{i,1}, k_{i,2}, \dots, k_{i,M_i}].
$$

These sentences are used to train a **Word2Vec** model:

- dimension \(d \in \{100, 390, \dots\}\) (grid via `--w2v-dims`)  
- window size \(w = 5\)  
- skip-gram (`sg = 1`)  
- hierarchical softmax disabled (`hs = 0`)  
- negative sampling (`negative = 5`)  
- `min_count = 1`, `workers = 48`  

Once the W2V model is trained, each k-mer \(k\) has an embedding vector \(\mathbf{w}(k) \in \mathbb{R}^d\).

#### Standardization by \(m_{\min}\) (min\_kmers)

Let \(M_i\) be the number of k-mers in sequence \(i\), and let

$$
m_{\min} = \min_i M_i
$$

be the minimum number of k-mers across all sequences in the training set. For each sequence:

- If \(M_i \ge m_{\min}\), we keep **only the first** \(m_{\min}\) k-mers.  
- If \(M_i < m_{\min}\), we **pad** with zero vectors to reach \(m_{\min}\) k-mers.

Thus, for each sequence we obtain exactly \(m_{\min}\) k-mer embeddings \(\mathbf{w}(k_{i,u})\), \(u = 1,\dots,m_{\min}\).

We then build a **sequence-level embedding**. In this project, we use the **mean embedding**:

$$
\mathbf{v}_i
=
rac{1}{m_{\min}}
\sum_{u=1}^{m_{\min}} \mathbf{w}(k_{i,u}) \in \mathbb{R}^d.
$$

These \(\mathbf{v}_i\) are the W2V sequence embeddings used in cosine-similarity analyses and W2V-based dendrograms.

### 5.5 Identity Bins and Pair Sampling

Pairwise identities \(\mathrm{Id}(i,j)\) are grouped into **identity bins** in the range 10–100%, typically in steps of 5% or 10%, depending on the figure (e.g., 10–20, 20–30, …, 90–100).

To limit the number of plotted pairs while keeping coverage of the full identity space, we deterministically **subsample** the upper triangle of all pairs \((i,j)\), \(i<j\), using a stride.

Let \(N\) be the number of sequences, and

$$
P = rac{N(N-1)}{2}
$$

the total number of unordered pairs. Given a desired number of sampled pairs \(S\) (e.g., \(600{,}000\) or \(1{,}000{,}000\)), we define:

$$
	ext{stride} = \left\lfloor rac{P}{S} 
ight
floor.
$$

Then we enumerate all pairs in a fixed (deterministic) order and keep every `stride`-th pair. This yields a **deterministic, evenly spaced** subsample without bias toward any region of the triangle.

### 5.6 Correlation Identity vs Cosine Similarity

To quantify how well the embedding space reflects alignment identity, we compute the **Pearson correlation** between:

- \(x_n = \mathrm{Id}(i_n, j_n)\) (identity, typically in % or normalized to \([0,1]\)),  
- \(y_n = \cos(	heta_{i_n j_n})\) (cosine similarity of W2V mean embeddings),

for sampled pairs \((i_n, j_n)\), \(n = 1,\dots,N\).

Cosine similarity between two embedding vectors \(\mathbf{v}_i\) and \(\mathbf{v}_j\) is:

$$
\cos(	heta_{ij}) =
rac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\|\,\|\mathbf{v}_j\|}.
$$

Let

$$
ar{x} = rac{1}{N}\sum_{n=1}^N x_n, \qquad
ar{y} = rac{1}{N}\sum_{n=1}^N y_n.
$$

The **Pearson correlation coefficient** is

$$
r_{xy} =
rac{\displaystyle \sum_{n=1}^N (x_n - ar{x})(y_n - ar{y})}
{\displaystyle
\sqrt{\sum_{n=1}^N (x_n - ar{x})^2}\,
\sqrt{\sum_{n=1}^N (y_n - ar{y})^2}
}.
$$

This value is reported in a **summary table** for each combination of:

- subset ∈ {25%, 50%, 75%, 100%},  
- W2V dimension (e.g., 100, 390, …),  
- W2V epochs (from `--w2v-epochs-list`),

allowing direct comparison of which configuration best captures the identity signal.

---

## 6. Pipeline Stages

### 6.1 Multiple Sequence Alignment (MAFFT)

- Aligns each subset with **MAFFT** (default `--auto`, configurable via CLI).  
- The **gapped alignment** is used for identity calculations.  
- For **Word2Vec**, we remove gaps (as in Section 5.1) to build k-mer sentences, but the alignment itself is always the same underlying MSA for both identity and embeddings.

### 6.2 Pairwise Identities & Dissimilarity Matrix

- We stream the upper triangle of all pairs to compute identity percentages \(\mathrm{Id}(i,j)\).  
- **Dissimilarity** is derived as \(d_{ij}=1-\mathrm{Id}(i,j)/100\).  
- For UMAP, we pass a **precomputed** dissimilarity matrix as input (metric=`precomputed`).  
- For hierarchical clustering, we use linkage on the same dissimilarity matrix.

### 6.3 Subsampling of Sequences and Pairs

- The pipeline runs separately for subsets of **25%, 50%, 75% and 100%** of all sequences.
- For each subset:
  - We run the full MSA, identity, and clustering pipeline.  
  - We apply pairwise subsampling as in Section 5.5 for the large scatter/hexbin plots.  

This allows us to inspect heterogeneity and embedding behavior across different dataset sizes.

### 6.4 MMseqs2 Clustering Across Identity Cutoffs

- For each subset, we cluster with **MMseqs2** using identity cutoffs:
  - 10%, 20%, 30%, ..., 90%.  
- Diagnostics include:
  - **Intra-cluster** pair counts over identity **deciles** (10–100) → **line plots** and **heatmap**.  
  - **Cluster size** vs **cutoff** (mean/median/P95) and **largest cluster** vs cutoff.

### 6.5 Heterogeneity Visualizations

For each subset and configuration, the following plots are generated (filenames may be prefixed with subset, dim, and epochs information):

- **01_bar_identity_10_100**:  
  - Barplot of identity distribution (10–100%), often in 5% bins.  
- **03_dendrogram_branches_by_phylum**:  
  - Dendrogram built from the dissimilarity matrix, with branches colored by Phylum where clades are homogeneous; mixed clades are colored gray.  
- **04_umap_dissimilarity_by_phylum**:  
  - UMAP 2D projection (metric=`precomputed`, using \(d_{ij}\)) colored by Phylum.  
- **06a_counts_intra_identity_bins_by_cutoff_lines_REALX**:  
  - Line plots of intra-cluster pair counts binned by identity for each MMseqs2 identity cutoff.  
- **06b_counts_intra_identity_bins_by_cutoff_heatmap**:  
  - Heatmap of counts of intra-cluster pairs across (identity bin × cutoff).  
- **04_cluster_size_vs_identity_cutoff**:  
  - Cluster size statistics vs identity cutoff.  
- **05_largest_cluster_by_identity_cutoff**:  
  - Size of the largest cluster vs identity cutoff.

All figures are exported as:

- `.png` (900 dpi)  
- `.tiff` (900 dpi)  
- `.svg`  

with **no plot title** and legends placed **below** the plot.

### 6.6 Part B — Word2Vec Analyses and Grid Search

Part B adds Word2Vec-based analyses to the heterogeneity pipeline.  
The **logic is the same** for all subsets and follows the alignment → k-mers → W2V steps described earlier:

1. **Alignment (same as Part A)**:  
   We always use the same MAFFT alignment already computed for the given subset.

2. **k-mer extraction** (from gapped-free sequences):  
   - \(k = 3\)  
   - step size = 1  
   - ignore k-mers consisting only of gaps (after removal, gaps are not present).

3. **Word2Vec training** for each combination of:
   - subset ∈ {25%, 50%, 75%, 100%}  
   - dimension \(d \in\) `--w2v-dims` (default includes 100 and 390)  
   - epochs ∈ `--w2v-epochs-list` (e.g., 200, 500, 1500, 2500)  

   with:

   - `vector_size = d`  
   - `window = 5`  
   - `sg = 1` (skip-gram)  
   - `hs = 0`  
   - `negative = 5`  
   - `min_count = 1`  
   - `workers = 48`  

   Embeddings are standardized by \(m_{\min}\) (`min_kmers`), as in Section 5.4.

4. **Sequence embeddings**:  
   - mean of k-mer embeddings per sequence (dimension \(d\)), \(\mathbf{v}_i \in \mathbb{R}^d\).

5. **Figures generated for each (subset, dim, epochs)**:

   - **07_cosine_vs_identity_intra**:  
     - Cosine similarity vs identity for **intra-phylum** pairs; typically plotted as density/hexbins with identity on the x-axis and cosine on the y-axis.  
   - **08_cosine_vs_identity_inter**:  
     - Same as above, but for **inter-phylum** pairs.  
   - **09_tsne_w2v_mean_by_phylum** (optional if t-SNE enabled):  
     - 2D t-SNE embedding of \(\mathbf{v}_i\), colored by Phylum.  
   - **10_umap_w2v_mean_by_phylum**:  
     - 2D UMAP embedding of \(\mathbf{v}_i\), colored by Phylum.  
   - **13_dendrogram_w2v_mean_by_phylum**:  
     - Dendrogram based on distances \(1 - \cos(	heta_{ij})\) between W2V sequence embeddings, with branches colored by Phylum.  

6. **Correlation summary table**:

   For each combination of:

   - subset (25/50/75/100%),  
   - W2V dimension,  
   - epochs,

   we compute the Pearson correlation coefficient \(r_{xy}\) (Section 5.6) between identity and cosine similarity over a sampled set of pairs and write a summary table, e.g.:

   ```text
   subset   w2v_dim   epochs   pearson_r_identity_cosine
   0.25     100       200      0.78
   0.25     100       500      0.81
   ...
   1.00     390       2500     0.86
   ```

   This allows direct inspection of which configuration best captures the identity signal.

---

## 7. Color Policy (Distinct Phylum Colors)

- We construct a **global palette** across all phyla using `tab20`, `tab20b`, `tab20c` as base, with golden-ratio HSV fallbacks when needed.  
- **Normalization**:  
  - `Actinobacteria` → `Actinomycetota`.  
- **Excluded from color plots**:  
  - labels containing `Unknown`, `uncultured`, `uncultivated` (they remain in numeric analyses, but not in color legends).  
- The same palette is reused across all plots for visual consistency.

---

## 8. Outputs & Directory Layout

For an output directory `results_faal`, and a subset label (e.g., `subset_25pct`, `subset_50pct`, `subset_75pct`, `subset_100pct`):

```text
results_faal/
  subset_25pct_dim100_epochs200/
    input_subset.fasta
    pairs.csv
    clusters_multi_threshold.csv
    subset_ids.csv
    subset_phylum_map.csv
    plots_sample_600000/
      01_bar_identity_10_100.(png|tiff|svg)
      03_dendrogram_branches_by_phylum.(png|tiff|svg)
      04_umap_dissimilarity_by_phylum.(png|tiff|svg)
      06a_counts_intra_identity_bins_by_cutoff_lines_REALX.(png|tiff|svg)
      06b_counts_intra_identity_bins_by_cutoff_heatmap.(png|tiff|svg)
      07_cosine_vs_identity_intra.(png|tiff|svg)
      08_cosine_vs_identity_inter.(png|tiff|svg)
    04_cluster_size_vs_identity_cutoff.(png|tiff|svg)
    05_largest_cluster_by_identity_cutoff.(png|tiff|svg)
    09_tsne_w2v_mean_by_phylum.(png|tiff|svg)
    10_umap_w2v_mean_by_phylum.(png|tiff|svg)
    13_dendrogram_w2v_mean_by_phylum.(png|tiff|svg)
    correlation_summary.tsv
    FIGURE_MANIFEST.txt
  subset_25pct_dim100_epochs500/
    ...
  subset_25pct_dim390_epochs200/
    ...
  subset_100pct_dim390_epochs2500/
    ...
```

Each `(subset, dim, epochs)` combination has its own subdirectory, so results are clearly separated.

---

## 9. Reproducibility, Performance & Practical Notes

- **Random seed** fixed to **42** for subsetting, UMAP, and t-SNE (where applicable).  
- **Scalability**:  
  - Pairwise identity computation streams the upper triangle and writes results incrementally.  
  - Subsampling by stride (Section 5.5) controls plot sizes (e.g., 600k and 1M sampled pairs).  
- **MSA cost** still dominates for very large \(N\).  
  - Consider subset fractions (25–75%) for exploratory analyses, then run 100% once.  
- **MMseqs2** is run on each subset separately with the same identity thresholds.  
- If metadata is missing, Phylum is `Unknown`; these entries are excluded from color plots but remain in numeric computations.  
- Use `--mafft-bin` and `--mmseqs-bin` to force binary paths when Conda/`PATH` resolution differs by system.

---

## 10. How to Run — CLI Version

The main CLI script is `faalprot_heterogeneity_cli_v2.py` (check `--help` for full options).

### 10.1 Minimal run (Part A only, default subset = 100%)

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta   --table Table_S2.tsv   --outdir results_faal
```

### 10.2 Enable Word2Vec Part B (default dims 100 and 390)

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta   --table Table_S2.tsv   --outdir results_faal_w2v   --run-part-b
```

### 10.3 Custom Word2Vec grid (dimensions and epochs)

For example, to test dimensions 100, 200, 390 and epochs 200, 500, 1500, 2500:

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta my_sequences.fasta   --table table_S2.tsv   --outdir resultados_grid   --run-part-b   --w2v-dims 100,200,390   --w2v-epochs-list 200,500,1500,2500
```

The script will:

- run alignment → k-mers → W2V **with the same logic** as FAALPred (alignment-based k-mers, min\_kmers cropping/padding, skip-gram, `hs=0`, `negative=5`, window=5),  
- for **all subsets (25%, 50%, 75%, 100%)**,  
- for **all combinations** of dimensions and epochs provided.

### 10.4 Explicit binary paths

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta   --table Table_S2.tsv   --outdir results_bins   --mafft-bin /usr/bin/mafft   --mmseqs-bin /home/USER/anaconda3/envs/faal/bin/mmseqs   --run-part-b   --w2v-dims 100,390   --w2v-epochs-list 500,1500,2500
```

---

## 11. FAQ

**Q1. Why sample pairs instead of plotting all?**  
The number of pairs grows as \(N(N-1)/2\). Even when we compute all identities, we **plot** only a **deterministic subsample** for clarity and performance.

**Q2. Why exclude “Unknown/uncultured” from color plots?**  
Color encodes discrete taxa (Phylum). Unknown labels would receive arbitrary colors and clutter legends without adding biological signal.

**Q3. Why use Word2Vec on gapped-free sequences, but identity on the gapped alignment?**  
Word2Vec models sequence **content/context** (k-mer tokens) and benefits from removing alignment-introduced gaps. Identity must respect aligned positions, hence uses the **gapped** MSA.

**Q4. How are dendrogram colors assigned?**  
Branches are colored by the **unique** phylum if all descendants share it; otherwise the branch is gray, highlighting mixed clades.

**Q5. How do I compare different W2V configs (dims, epochs)?**  
Check the **correlation summary table** (`correlation_summary.tsv`) for each (subset, dim, epochs). Higher Pearson \(r\) between identity and cosine typically indicates a better embedding for capturing the alignment signal.

---

## 12. References

- Katoh, K. & Standley, D.M. (2013). MAFFT multiple sequence alignment software version 7: improvements in performance and usability. *Mol. Biol. Evol.*  
- Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching. *Nat. Biotechnol.*  
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.  
- van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*.  
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *NIPS* / arXiv.  
- Virtanen, P. *et al.* (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat. Methods*.  
- Pedregosa, F. *et al.* (2011). Scikit-learn: Machine Learning in Python. *JMLR*.  
- Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*.  
- Cock, P.J.A. *et al.* (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*.  
