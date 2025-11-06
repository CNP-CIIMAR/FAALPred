# FAALProt Heterogeneity Analysis
**Complete documentation for both the Jupyter notebook and the CLI script (v2, with Word2Vec grid search)**  

---

## Table of Contents
- [1. Overview](#1-overview)
- [2. Approach](#2-approach)
- [3. Input Data & Metadata](#3-input-data--metadata)
- [4. Software & Environment](#4-software--environment)
- [5. Mathematical Definitions](#5-mathematical-definitions)
  - [5.1 Pairwise Identity](#51-pairwise-identity)
  - [5.2 Dissimilarity](#52-dissimilarity)
  - [5.3 k-mer Tokenization (Aligned Sequences)](#53-kmer-tokenization-aligned-sequences)
  - [5.4 Word2Vec Sequence Embeddings](#54-word2vec-sequence-embeddings)
  - [5.5 Cosine Similarity on Embeddings](#55-cosine-similarity-on-embeddings)
  - [5.6 Binned Identity & Counts](#56-binned-identity--counts)
  - [5.7 Correlation Between Identity and Cosine Similarity](#57-correlation-between-identity-and-cosine-similarity)
- [6. Pipeline Stages](#6-pipeline-stages)
  - [6.1 Multiple Sequence Alignment (MAFFT)](#61-multiple-sequence-alignment-mafft)
  - [6.2 Pairwise Identities & Dissimilarity Matrix](#62-pairwise-identities--dissimilarity-matrix)
  - [6.3 Subsampling of Pairs](#63-subsampling-of-pairs)
  - [6.4 MMseqs2 Clustering Across Identity Cutoffs](#64-mmseqs2-clustering-across-identity-cutoffs)
  - [6.5 Visualizations (Alignment-based)](#65-visualizations-alignment-based)
  - [6.6 Optional Part B — Word2Vec Analyses (Alignment → k-mers → Embeddings)](#66-optional-part-b--word2vec-analyses-alignment--k-mers--embeddings)
- [7. Color Policy (Distinct Phylum Colors)](#7-color-policy-distinct-phylum-colors)
- [8. Outputs & Directory Layout](#8-outputs--directory-layout)
- [9. Reproducibility, Performance & Practical Notes](#9-reproducibility-performance--practical-notes)
- [10. How to Run — Jupyter Version](#10-how-to-run--jupyter-version)
- [11. How to Run — CLI Version (v2)](#11-how-to-run--cli-version-v2)
- [12. FAQ](#12-faq)
- [13. References](#13-references)

---

## 1. Overview

This repository provides a complete, publication-grade pipeline to quantify and visualize **heterogeneity in FAAL proteins** across thousands of sequences. It integrates:

- **MAFFT** for multiple sequence alignment (MSA)
- **Streaming pairwise identity** estimation with deterministic subsampling of the upper triangle
- **UMAP** (precomputed metric) on *dissimilarity* derived from alignment identity
- **Hierarchical clustering** and **dendrograms** with **branch colors by Phylum**
- **MMseqs2 clustering** over fixed identity thresholds (10–90%) and diagnostic plots
- **(Optional) Word2Vec Part B**, now extended to a **grid search** over:
  - embedding dimensions (e.g. **100**, **390**, and user-defined),
  - number of training epochs (e.g. **200, 500, 1500, 2500**),
  - always following the logic: **alignment → k-mers → Word2Vec → cosine similarity**

All figures are exported as **PNG (900 dpi)**, **TIFF (900 dpi)** and **SVG**, **without titles**, and with legends **below** the plots.  

> **Taxonomic normalization:** Any Phylum string equal to `Actinobacteria` is normalized to `Actinomycetota`. Phyla labelled as `Unknown` / `uncultured` / `uncultivated` are excluded from color-coded plots and legends.

---

## 2. Approach

For a given FASTA + metadata table, the pipeline:

1. **Reads FASTA** and an optional **Table_S2** metadata file.
2. Derives **Phylum** from `Lineage` (if provided) and normalizes names.
3. Automatically defines **four sequence subsets**:  
   **25%, 50%, 75%, and 100%** of all sequences (deterministic subsampling with fixed seed).
4. For **each subset**:
   - Runs **MAFFT**.
   - Computes **all-vs-all pairwise identities** (streamed).
   - Builds a **dissimilarity matrix** \(d_{ij}=1-\mathrm{Id}(i,j)/100\).
   - Runs **MMseqs2 clustering** at identity cutoffs 10–90%.
   - Generates alignment-based figures:
     - **01_bar_identity_10_100**
     - **03_dendrogram_branches_by_phylum**
     - **04_umap_dissimilarity_by_phylum**
     - **06a_counts_intra_identity_bins_by_cutoff_lines_REALX**
     - **06b_counts_intra_identity_bins_by_cutoff_heatmap**
5. If **Part B** is enabled (`--run-part-b`), still per subset:
   - **Removes gaps** from the aligned sequences.
   - Extracts **k-mers** (default \(k=3\)) with a sliding window (step size 1).
   - Calculates `min_kmers` (minimum number of valid k-mers across all sequences) and
     **truncates/pads all sequences to exactly `min_kmers` tokens**.
   - Trains **Word2Vec** for each combination of:
     - dimension \(d \in 	exttt{--w2v-dims}\) (default: `100,390`),
     - epochs \(E \in 	exttt{--w2v-epochs-list}\) (default: `2500`),
     using **skip-gram**, `hs=0`, `negative=5`, `window=5`, `min_count=1`.
   - Computes **mean embedding** per sequence, **cosine similarity** for sequence pairs,
     and all W2V-based figures for each *(subset, dim, epochs)* combination:
     - **07_cosine_vs_identity_intra**
     - **08_cosine_vs_identity_inter**
     - **09_tsne_w2v_mean_by_phylum**
     - **10_umap_w2v_mean_by_phylum**
     - **13_dendrogram_w2v_mean_by_phylum**
   - Fills a **summary table** of **correlations between identity and cosine similarity**:
     `summary_identity_vs_cosine_correlations.csv`.

---

## 3. Input Data & Metadata

- **FASTA**: protein sequences (the header’s first token is used as `sequence_id`).
- **Table_S2.tsv (optional but recommended)**: a tab-separated table with at least:
  - `Protein Accession` (or a column that can be mapped to `sequence_id`)
  - `Lineage` (e.g., `Bacteria; Pseudomonadota; Gammaproteobacteria; ...`)

**Phylum extraction**:

- Phylum is extracted from `Lineage` **position 2** (0-based index 1) if it exists.
- If `Lineage` has fewer levels, we use the last level if it is not a known domain-level token.
- `Actinobacteria` is normalized to `Actinomycetota`.

---

## 4. Software & Environment

- **MAFFT** (e.g., `/usr/bin/mafft`)
- **MMseqs2** (e.g., `<conda-env>/bin/mmseqs`)
- **Python 3.9+**, with libraries:
  - `numpy`, `pandas`, `matplotlib`
  - `umap-learn` (UMAP)
  - `scikit-learn` (t-SNE, distances, clustering utilities)
  - `scipy` (hierarchical clustering, linkage, dendrogram)
  - `biopython` (FASTA parsing, optional but recommended)
  - `gensim` (Word2Vec; required for Part B)

The CLI tries to auto-locate MAFFT and MMseqs2 via `PATH`, `CONDA_PREFIX`, and `whereis`.  
You can force paths with `--mafft-bin` and `--mmseqs-bin`.

---

## 5. Mathematical Definitions

### 5.1 Pairwise Identity

Let \(A_i(t)\) be the residue of sequence \(i\) at aligned position \(t\), and `-` a gap.  
The **pairwise identity** between sequences \(i\) and \(j\) is:

\[
\mathrm{Id}(i,j) \;=\; 100 	imes
rac{\displaystyle\sum_{t} \mathbf{1}ig[\,A_i(t)=A_j(t) \wedge A_i(t)
eq - \wedge A_j(t)
eq -\,ig]}
     {\displaystyle\sum_{t} \mathbf{1}ig[\,A_i(t)
eq - \wedge A_j(t)
eq -\,ig]}
\, .
\]

Only aligned positions where **neither sequence has a gap** are considered in the denominator.

---

### 5.2 Dissimilarity

From the pairwise identity we define a **dissimilarity** in \([0,1]\):

\[
d_{ij} \;=\; 1 \;-\; rac{\mathrm{Id}(i,j)}{100} \,.
\]

This is the input to **UMAP** when using `metric="precomputed"` and to hierarchical clustering for the **MSA-based dendrogram**.

---

### 5.3 k-mer Tokenization (Aligned Sequences)

Let a **gapped alignment** sequence be:

\[
S_i = [s_{i,1}, s_{i,2}, \dots, s_{i,L}], \quad s_{i,t} \in \{	ext{amino acids} \cup \{-\}\}.
\]

We first **remove gaps** to obtain a gapped-free sequence:

\[
	ilde{S}_i = [ 	ilde{s}_{i,1}, 	ilde{s}_{i,2}, \dots, 	ilde{s}_{i,	ilde{L}} ], 
\quad 	ilde{s}_{i,u} \in \{	ext{amino acids}\}.
\]

For a fixed **k-mer size** \(k\) (default \(k=3\)) and **step size** \(s=1\), we generate tokens:

\[
	ext{kmer}_{i,u} = (	ilde{s}_{i,u}, 	ilde{s}_{i,u+1}, \dots, 	ilde{s}_{i,u+k-1}),
\quad u = 1, 1+s, 1+2s, \dots, 	ilde{L} - k + 1 \,.
\]

In code, this is implemented as a sliding window with optional filtering of degenerate patterns.  
For each subset, we compute:

\[
m \;=\; \min_i |\{	ext{kmer}_{i,\cdot}\}| \,,
\]

the **minimum number of valid k-mers across all sequences** (`min_kmers`).  
Later we **truncate**/pad each sequence to exactly \(m\) k-mers (Section 5.4).

---

### 5.4 Word2Vec Sequence Embeddings

Let \(\phi(\cdot)\) be the trained **Word2Vec embedding** that maps each k-mer token to \(\mathbb{R}^d\):

\[
\phi:\; 	ext{k-mer string} \;\longrightarrow\; \mathbb{R}^d,
\quad d \in 	exttt{--w2v-dims}.
\]

After training, each sequence \(i\) has a list of valid k-mers \(\{	ext{kmer}_{i,1}, \dots, 	ext{kmer}_{i,n_i}\}\).  
We truncate or pad to exactly \(m = 	exttt{min\_kmers}\) tokens:

- if \(n_i > m\): keep the **first** \(m\) tokens,
- if \(n_i < m\): pad with “missing” tokens mapped to the **zero vector**.

We obtain a length-\(m\) list of vectors:

\[
ig(\phi(	ext{kmer}_{i,1}), \dots, \phi(	ext{kmer}_{i,m})ig)
\in (\mathbb{R}^d)^m.
\]

For the **mean embedding** used in this pipeline:

\[
\mathbf{v}_i
= rac{1}{m} \sum_{u=1}^{m} \phi(	ext{kmer}_{i,u}) 
\;\;\in\; \mathbb{R}^d.
\]

This ensures each sequence contributes **exactly the same number of tokens** (`min_kmers`) to its embedding, reducing bias from different sequence lengths.

---

### 5.5 Cosine Similarity on Embeddings

Given mean embeddings \(\mathbf{v}_i, \mathbf{v}_j \in \mathbb{R}^d\), the **cosine similarity** is:

\[
\cos(	heta_{ij})
\;=\;
rac{\mathbf{v}_i \cdot \mathbf{v}_j}
     {\|\mathbf{v}_i\|_2 \,\|\mathbf{v}_j\|_2} \,.
\]

For **W2V-based dendrograms**, we use a distance derived from the cosine:

\[
\delta_{ij}^{(\mathrm{w2v})}
\;=\;
1 - \cos(	heta_{ij}) \,,
\]

and build a hierarchical clustering on the matrix \(\delta_{ij}^{(\mathrm{w2v})}\).

---

### 5.6 Binned Identity & Counts

For diagnostics and heatmaps, we discretize identity into **deciles**:

\[
	ext{bin}(i,j) = 
\left\lfloor rac{\mathrm{Id}(i,j)}{10} 
ight
floor,
\quad
	ext{clipped to } 1,\dots,10.
\]

For a given identity cutoff \(c\) and an identity bin \(b\), the **count of intra-cluster pairs** is:

\[
N_{	ext{intra}}(c,b) 
= 
\sum_{(i,j) \in \mathcal{P}(c)}
\mathbf{1}ig[	ext{bin}(i,j) = big],
\]

where \(\mathcal{P}(c)\) is the set of pairs in clusters produced at cutoff \(c\)
(e.g., MMseqs2 cluster identity ≥ \(c\)).

These counts build the **line plot** and the **heatmap**:

- **06a_counts_intra_identity_bins_by_cutoff_lines_REALX**
- **06b_counts_intra_identity_bins_by_cutoff_heatmap**.

---

### 5.7 Correlation Between Identity and Cosine Similarity

To quantify how well the embedding space reflects alignment identity, we compute the **Pearson correlation** between:

- \(x_n = \mathrm{Id}(i_n,j_n)\) (identity, typically in % or normalized to \([0,1]\)),
- \(y_n = \cos(	heta_{i_n j_n})\) (cosine similarity of W2V mean embeddings),

for sampled pairs \((i_n,j_n)\).  

Let:

\[
ar{x} = rac{1}{N}\sum_{n=1}^N x_n, 
\qquad
ar{y} = rac{1}{N}\sum_{n=1}^N y_n.
\]

The **Pearson correlation coefficient** is:

\[
r
=
rac{
  \sum_{n=1}^N (x_n - ar{x})(y_n - ar{y})
}{
  \sqrt{\sum_{n=1}^N (x_n - ar{x})^2}
  \,\sqrt{\sum_{n=1}^N (y_n - ar{y})^2}
}.
\]

The pipeline computes \(r\) separately for:

- **intra-phylum pairs**,
- **inter-phylum pairs**,
- optionally **all pairs combined**.

The results across **subsets × dimensions × epochs** are stored in:
`summary_identity_vs_cosine_correlations.csv`.

---

## 6. Pipeline Stages

### 6.1 Multiple Sequence Alignment (MAFFT)

For each subset (25%, 50%, 75%, 100%) the script:

1. Selects a deterministic subset of sequences (fixed random seed 42).
2. Runs **MAFFT** (default `--auto`) on the subset:
   - MSA is used for **pairwise identity**.
   - Gaps are **removed only for Word2Vec tokenization** in Part B; the identity still uses the full, gapped alignment.

---

### 6.2 Pairwise Identities & Dissimilarity Matrix

For a subset with \(N\) sequences:

- The script walks over the **upper triangle** of the \(N	imes N\) matrix
  and computes \(\mathrm{Id}(i,j)\) for all pairs \((i<j)\) using the formula in [5.1].
- It then derives the **dissimilarity matrix** \(d_{ij}=1-\mathrm{Id}(i,j)/100\) (Section 5.2).
- This matrix is used for:
  - **UMAP** with `metric="precomputed"`,
  - **hierarchical clustering** (linkage, default `average`) for `03_dendrogram_branches_by_phylum`.

---

### 6.3 Subsampling of Pairs

To keep the scatter and density plots tractable while preserving global structure, we deterministically subsample pairs.

Let:

- total number of pairs: \( M = N(N-1)/2 \),
- target number of pairs to plot: \(S\) (e.g., 600,000 or 1,000,000).

We define the **stride**:

\[
	ext{stride} = \left\lfloor rac{M}{S} 
ight
floor.
\]

We then enumerate pairs \((i,j)\) in a fixed order and keep every pair whose index \(k\) satisfies:

\[
k \equiv 0 \pmod{	ext{stride}}.
\]

This yields a deterministic, evenly spaced coverage over the upper triangle, without bias toward a specific block of the matrix.

---

### 6.4 MMseqs2 Clustering Across Identity Cutoffs

For each subset, the pipeline runs **MMseqs2** clustering at identity thresholds:

\[
p \in \{10, 20, \dots, 90\}\%.
\]

For each cutoff \(p\):

1. Clusters are computed using MMseqs2 with the corresponding minimum identity.
2. The script collects:
   - cluster assignments,
   - **intra-cluster pairs** and their identities,
   - **cluster size** statistics (mean, median, 95th percentile, maximum).

The resulting diagnostics feed into:

- **04_cluster_size_vs_identity_cutoff**  
  (mean, median, P95 cluster size vs cutoff),
- **05_largest_cluster_by_identity_cutoff**  
  (size of the largest cluster vs cutoff),
- **06a/06b** (intra-cluster identity distributions).

---

### 6.5 Visualizations (Alignment-based)

For each subset, the following figures are generated (all without titles, legends below):

1. **01_bar_identity_10_100**  
   - Histogram / barplot of pairwise identities (10–100%, typically in 5% bins).
2. **03_dendrogram_branches_by_phylum**  
   - Hierarchical clustering on \(d_{ij}\) (Section 5.2) with branches colored by phylum:
     - a clade is colored if all leaves share the same phylum;
     - mixed-phylum clades are drawn in neutral gray.
3. **04_umap_dissimilarity_by_phylum**  
   - UMAP on precomputed dissimilarity:
     - `n_neighbors=20`, `min_dist=0.05`, `metric="precomputed"`, `random_state=42`.
     - Points colored by phylum with a **globally consistent palette**.
4. **06a_counts_intra_identity_bins_by_cutoff_lines_REALX**  
   - Line plots of intra-cluster counts per identity bin and cutoff.
5. **06b_counts_intra_identity_bins_by_cutoff_heatmap**  
   - Heatmap of intra-cluster counts, identity bins (10–100) × cutoffs (10–90).

Additionally, for each subset, **subset-level** plots are generated in the subset folder:

- **04_cluster_size_vs_identity_cutoff**
- **05_largest_cluster_by_identity_cutoff**

---

### 6.6 Optional Part B — Word2Vec Analyses (Alignment → k-mers → Embeddings)

Part B is activated via `--run-part-b` in the CLI.

For each subset (25%, 50%, 75%, 100%):

1. **Gap removal & k-mer generation**

   - Remove gaps from the aligned sequences.
   - Generate k-mers of size \(k\) (default `--kmer-size 3`) with step size `--kmer-step 1`.
   - Keep only valid k-mers (optionally filtering those dominated by gaps).
   - Compute `min_kmers` = minimum number of valid k-mers across all sequences.

2. **Standardization via `min_kmers`**

   - For each sequence, its list of k-mers is:
     - **truncated** to the first `min_kmers` tokens if it has more; or
     - **padded** with “missing” tokens mapped to the **zero vector** if it has fewer.
   - This ensures each sequence contributes **exactly `min_kmers` tokens** to its embedding, aligning the logic with FAALPred.

3. **Word2Vec training (per (dim, epochs) combination)**

   The script iterates over:

   - embedding dimensions:  
     \[
     d \in 	exttt{--w2v-dims}
     \quad	ext{(default: } d\in\{100,390\}	ext{)}
     \]
   - epochs:  
     \[
     E \in 	exttt{--w2v-epochs-list}
     \quad	ext{(default: } E = 2500	ext{; user can set e.g. } 200,500,1500,2500).
     \]

   For each pair \((d,E)\) and each subset, it trains a Word2Vec model with:

   - `vector_size = d`
   - `window = 5`
   - `sg = 1` (skip-gram)
   - `hs = 0` (hierarchical softmax disabled)
   - `negative = 5` (negative sampling)
   - `min_count = 1`
   - `workers = 48` (default; configurable in the code/CLI if exposed)
   - fixed random seed = `42` for reproducibility.

4. **Sequence embeddings**

   - For each sequence, compute the **mean embedding** of its `min_kmers` token embeddings as in [5.4].
   - Store \(\mathbf{v}_i \in \mathbb{R}^d\) for downstream analyses.

5. **Cosine vs Identity plots (intra / inter)**

   For each (subset, \(d\), \(E\)):

   - Compute cosine similarity \(\cos(	heta_{ij})\) between mean embeddings (Section 5.5).
   - Join with pairwise identity \(\mathrm{Id}(i,j)\) and phylum labels.
   - Produce:

     - **07_cosine_vs_identity_intra**  
       Scatter/hexbin of identity vs cosine similarity for **intra-phylum** pairs.  
       Includes a trend line (e.g., binned average of cosine over identity deciles).

     - **08_cosine_vs_identity_inter**  
       Same but **inter-phylum** pairs.

   - These figures use the same **pair subsampling** strategy (Section 6.3).

6. **Low-dimensional projections based on embeddings**

   For each (subset, \(d\), \(E\)):

   - **t-SNE** on mean embeddings (`perplexity=30`) →  
     **09_tsne_w2v_mean_by_phylum_dim{d}_ep{E}_subset{fraction}**.
   - **UMAP** on mean embeddings (Euclidean metric) →  
     **10_umap_w2v_mean_by_phylum_dim{d}_ep{E}_subset{fraction}**.

   In both, points are colored by phylum using the same global palette.

7. **W2V-based dendrogram**

   - Compute pairwise distances \(\delta_{ij}^{(\mathrm{w2v})}=1-\cos(	heta_{ij})\).
   - Build a hierarchical clustering (e.g., `average` linkage).
   - Produce:

     - **13_dendrogram_w2v_mean_by_phylum_dim{d}_ep{E}_subset{fraction}**

   Branch coloring follows the **same phylum rules** as in the MSA-based dendrogram (Section 6.5).

8. **Correlation summary table**

   - For each *(subset, dim, epochs)* and each pair type (intra, inter, combined),
     compute the **Pearson correlation coefficient \(r\)** between identity and cosine similarity (Section 5.7).
   - Append a row to:

     - `summary_identity_vs_cosine_correlations.csv`

   with columns such as:

   - `subset_fraction` (e.g., 0.25, 0.50, 0.75, 1.00)
   - `embedding_dim`
   - `epochs`
   - `pair_type` (intra / inter / all)
   - `pearson_r`
   - `n_pairs` (number of pairs used to compute \(r\))

This table allows direct comparison of **which embedding dimension / training regime best captures alignment signal**.

---

## 7. Color Policy (Distinct Phylum Colors)

- A **global palette** is built across all phyla using `tab20`, `tab20b`, `tab20c`, and HSV fallbacks.
- `Actinobacteria` is always mapped to `Actinomycetota`.
- Any phylum string containing `Unknown`, `uncultured`, or `uncultivated` is:
  - **excluded** from color-coded plots and legends,
  - but still included in numerical computations (identities, clustering, etc.).
- The same color mapping is reused in **all figures** involving phylum labels:
  - UMAPs, t-SNEs, dendrograms, barplots, etc.

---

## 8. Outputs & Directory Layout

For an output directory `results_faal`, the CLI v2 organizes results approximately as:

```text
results_faal/
  subset_0.25/
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

      # Only if --run-part-b, and for each (dim, epochs):
      07_cosine_vs_identity_intra_dim{d}_ep{E}.(png|tiff|svg)
      08_cosine_vs_identity_inter_dim{d}_ep{E}.(png|tiff|svg)
      09_tsne_w2v_mean_by_phylum_dim{d}_ep{E}.(png|tiff|svg)
      10_umap_w2v_mean_by_phylum_dim{d}_ep{E}.(png|tiff|svg)
      13_dendrogram_w2v_mean_by_phylum_dim{d}_ep{E}.(png|tiff|svg)

    04_cluster_size_vs_identity_cutoff.(png|tiff|svg)
    05_largest_cluster_by_identity_cutoff.(png|tiff|svg)
    FIGURE_MANIFEST.txt

  subset_0.50/
    ...
  subset_0.75/
    ...
  subset_1.00/
    ...

  # Global W2V correlation summary (all subsets × dims × epochs)
  summary_identity_vs_cosine_correlations.csv
```

The exact filenames may have additional suffixes (e.g., `_subset0.25`) to make the combination *(subset, dim, epochs)* explicit.

---

## 9. Reproducibility, Performance & Practical Notes

- **Random seed**:
  - Subsetting, UMAP, t-SNE, and Word2Vec all use **seed = 42** for reproducibility.
- **Scalability**:
  - MAFFT and all-vs-all identities are the main bottlenecks.
  - Working with subsets at 25–75% helps keep runtime and memory usage reasonable.
  - Pairwise **identities are streamed**, and plots use **deterministic subsampling** (Section 6.3).
- **MMseqs2**:
  - Runs on each subset separately; cluster IDs and sizes are reported per subset.
- **Word2Vec**:
  - Training can be heavy for large datasets, especially with many combinations of `--w2v-dims` and `--w2v-epochs-list`.
  - You can start with fewer combinations (e.g., only 100 and 390 dimensions; only 2500 epochs) and then expand.
- **Metadata gaps**:
  - Sequences lacking phylum information are included in numerical analyses but omitted from color legends.
- **Binary paths**:
  - Use `--mafft-bin` and `--mmseqs-bin` if your binaries are not discoverable via `PATH`.

---

## 10. How to Run — Jupyter Version

1. Open the Jupyter notebook (English version) and set:

   ```python
   FASTA_PATH = "FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta"
   TABLE_S2_PATH = "Table_S2.tsv"   # optional but recommended
   OUTDIR = "results_faal"
   ```

2. The notebook internally handles the **four subset fractions**: 0.25, 0.50, 0.75, 1.00.

3. Optionally tune:

   ```python
   KMER_SIZE = 3
   KMER_STEP = 1
   W2V_DIMS = [100, 390]          # or add more, e.g. [100, 200, 390]
   W2V_EPOCHS_LIST = [2500]       # or [200, 500, 1500, 2500]
   RUN_PART_B = True              # to enable W2V analyses
   ```

4. Run the main orchestration cell (e.g. `run_pipeline(...)`).

The notebook exports all figures as `.png`/`.tiff` (900 dpi) and `.svg`, prints summary tables, and fills `summary_identity_vs_cosine_correlations.csv` if Part B is active.

---

## 11. How to Run — CLI Version (v2)

The CLI script is:

```text
faalprot_heterogeneity_cli_v2.py
```

You can inspect all options with:

```bash
python faalprot_heterogeneity_cli_v2.py --help
```

### 11.1 Minimal run (alignment, identities, MMseqs2; no Word2Vec)

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta   --table Table_S2.tsv   --outdir results_faal
```

This will:

- Build the four subsets (25%, 50%, 75%, 100%),
- Run MAFFT, pairwise identities, MMseqs2,
- Generate figures: 01, 03, 04, 06a, 06b, 04_cluster_size, 05_largest_cluster,
- **Not** run any Word2Vec analyses.

---

### 11.2 With Word2Vec Part B (default dimensions and epochs)

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta   --table Table_S2.tsv   --outdir results_faal_w2v   --run-part-b
```

Defaults (in v2):

- `--w2v-dims 100,390`
- `--w2v-epochs-list 2500`
- `--kmer-size 3`
- `--kmer-step 1`
- `window=5`, `sg=1`, `hs=0`, `negative=5`, `min_count=1`, `workers=48`, `seed=42`.

For each subset and each (dim, epochs) pair, the script:

- Trains Word2Vec,
- Computes mean embeddings,
- Generates figures 07, 08, 09, 10, 13,
- Updates `summary_identity_vs_cosine_correlations.csv`.

---

### 11.3 Full W2V grid (example: multiple dimensions and epochs)

To explicitly test several dimensions and epoch counts:

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta my_sequences.fasta   --outdir resultados_grid   --table table_S2.tsv   --run-part-b   --w2v-dims 100,200,390   --w2v-epochs-list 200,500,1500,2500
```

This will, for each subset fraction (0.25, 0.50, 0.75, 1.00):

- Train Word2Vec for:
  - dims: 100, 200, 390
  - epochs: 200, 500, 1500, 2500
- Generate all W2V-dependent figures (07/08/09/10/13) for each combination.
- Fill the correlation summary table with one row per combination and pair type.

---

### 11.4 Custom pair sampling & binary paths (optional)

If exposed in your version of the CLI, you can tune pair sampling sizes and binary paths, e.g.:

```bash
python faalprot_heterogeneity_cli_v2.py   --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta   --table Table_S2.tsv   --outdir results_custom   --run-part-b   --mafft-bin /usr/bin/mafft   --mmseqs-bin /home/USER/miniconda3/envs/faal/bin/mmseqs
```

Consult `--help` for the exact option names available in your installed version.

---

## 12. FAQ

**Q1. Why sample pairs instead of showing all?**  
The number of pairs grows as \(N(N-1)/2\). We **compute all** identities for correctness, but plots would be too dense and slow. We therefore use a **deterministic stride** over the upper triangle (Section 6.3) to sample a fixed number of pairs (e.g., 600k, 1M) for visualization.

---

**Q2. Why exclude “Unknown/uncultured” from color plots?**  
Color is used to distinguish **taxonomic phyla**. Labels like `Unknown` or `uncultured` do not add interpretable phylogenetic structure. They are still used in **numerical analyses**, but their points are either dropped from colored plots or shown in neutral tones without cluttering the legend.

---

**Q3. Why is Word2Vec trained on gap-free sequences but identity on gapped alignments?**  
Word2Vec models local **sequence context** in the amino acid chain and benefits from **removing alignment-induced gaps**. In contrast, alignment identity quantifies conservation of aligned positions, which correctly uses the **registered, gapped coordinates** from MAFFT. The pipeline keeps both representations consistent by:

- using gapped alignment for identity (Section 5.1),
- using gap-free sequences for k-mer generation and Word2Vec (Section 5.3),
- standardizing token counts with `min_kmers` (Section 5.4).

---

**Q4. How should I interpret the correlation summary table?**  
`summary_identity_vs_cosine_correlations.csv` reports the Pearson correlation \(r\) between identity and cosine similarity for each *(subset fraction, dimension, epochs)* and pair type (intra / inter / all).  

Roughly:

- **Higher \(r\)** (closer to 1) means the embedding space preserves alignment identity better.
- Comparing rows lets you see whether:
  - higher dimensions (e.g. 390 vs 100),
  - more epochs (e.g. 2500 vs 200),
  improve or degrade the alignment-based signal.

This table is designed to help you choose the **best embedding regime** for downstream tasks.

---

**Q5. Why four fixed subsets (25%, 50%, 75%, 100%)?**  
These fractions provide a **systematic view of how dataset size affects heterogeneity**:

- 25% and 50% allow much faster exploration and debugging,
- 75% and 100% show the behavior near the full dataset,
- Word2Vec and MMseqs2 are all evaluated consistently across these four regimes.

---

## 13. References

- Katoh, K. & Standley, D.M. (2013). MAFFT multiple sequence alignment software version 7: improvements in performance and usability. *Mol. Biol. Evol.*
- Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching. *Nat. Biotechnol.*
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NIPS*.
- Virtanen, P. *et al.* (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nat. Methods*.
- Pedregosa, F. *et al.* (2011). Scikit-learn: Machine learning in Python. *JMLR*.
- Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*.
- Cock, P.J.A. *et al.* (2009). Biopython: Freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*.
