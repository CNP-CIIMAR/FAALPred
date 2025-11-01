# FAALProt Heterogeneity Analysis
**Complete documentation for both the Jupyter notebook and the CLI script**  
---

## Table of Contents
- [1. Overview](#1-overview)
- [2. Approach at a Glance](#2-approach-at-a-glance)
- [3. Input Data & Metadata](#3-input-data--metadata)
- [4. Software & Environment](#4-software--environment)
- [5. Mathematical Definitions](#5-mathematical-definitions)
- [6. Pipeline Stages](#6-pipeline-stages)
  - [6.1 Multiple Sequence Alignment (MAFFT)](#61-multiple-sequence-alignment-mafft)
  - [6.2 Pairwise Identities & Dissimilarity Matrix](#62-pairwise-identities--dissimilarity-matrix)
  - [6.3 Subsampling of Pairs](#63-subsampling-of-pairs)
  - [6.4 MMseqs2 Clustering Across Identity Cutoffs](#64-mmseqs2-clustering-across-identity-cutoffs)
  - [6.5 Visualizations](#65-visualizations)
  - [6.6 Optional Part B — Word2Vec Analyses](#66-optional-part-b--word2vec-analyses)
- [7. Color Policy (Distinct Phylum Colors)](#7-color-policy-distinct-phylum-colors)
- [8. Outputs & Directory Layout](#8-outputs--directory-layout)
- [9. Reproducibility, Performance & Practical Notes](#9-reproducibility-performance--practical-notes)
- [10. How to Run — Jupyter Version](#10-how-to-run--jupyter-version)
- [11. How to Run — CLI Version](#11-how-to-run--cli-version)
- [12. FAQ](#12-faq)
- [13. References](#13-references)

---

## 1. Overview
This repository provides a complete, publication‑grade pipeline to quantify and visualize **heterogeneity in FAAL proteins** across thousands of sequences. It integrates:

- **MAFFT** for multiple sequence alignment (MSA)
- **Streaming pairwise identity** estimation with deterministic subsampling of the upper triangle
- **UMAP (precomputed metric)** on *dissimilarity* derived from alignment identity
- **Hierarchical clustering** and **dendrograms** with **branch colors by Phylum**
- **MMseqs2 clustering** over fixed identity thresholds (10–90%) and diagnostic plots
- **(Optional) Word2Vec** (k‑mer embeddings from aligned sequences without gaps): cosine‑vs‑identity, t‑SNE/UMAP, and a W2V‑based dendrogram

All figures are exported as **PNG (900 dpi)**, **TIFF (900 dpi)** and **SVG**, without titles, and with legends **below** the plots.

> **Note**: Any Phylum string equal to `Actinobacteria` is normalized to `Actinomycetota`. Phyla labelled as `Unknown` / `uncultured` / `uncultivated` are excluded from color‑coded plots and legends.

---

## 2. Approach at a Glance
1. **Read FASTA** and an optional **Table_S2** metadata file.
2. Derive **Phylum** from `Lineage` (if provided) and normalize names.
3. (Optional) **Subsample** sequences for tractable MSA and visualization.
4. Run **MAFFT** → produce **pairwise identities** for all pairs (streamed with sampling).
5. Build **dissimilarity** = \(1 - \tfrac{\text{identity}}{100}\).
6. Visualize **UMAP** of dissimilarity, **dendrogram** from dissimilarity, and **identity distribution**.
7. Run **MMseqs2 clustering** across identity thresholds and plot intra‑cluster diagnostics.
8. *(Optional)* Train **Word2Vec** from gapped‑free aligned sequences; visualize 2‑D reductions and a W2V dendrogram.

---

## 3. Input Data & Metadata
- **FASTA**: protein sequences (headers’ first token used as `sequence_id`).
- **Table_S2.tsv (optional)**: a tab‑separated table with at least:
  - `Protein Accession` (or `sequence_id`)
  - `Lineage` (e.g., `Bacteria; Pseudomonadota; Gammaproteobacteria; ...`)
- Phylum is extracted from `Lineage` **position 2** if present; otherwise falls back to a single‑token lineage when it is not a domain‑level token.

---

## 4. Software & Environment
- **MAFFT** (e.g., `/usr/bin/mafft`)
- **MMseqs2** (e.g., `<conda‑env>/bin/mmseqs`)
- **Python 3.9+**, with libraries:
  - `numpy`, `pandas`, `matplotlib`
  - `umap-learn` (for UMAP)
  - `scikit-learn` (t‑SNE, pairwise distances)
  - `scipy` (hierarchical clustering/dendrogram)
  - `biopython` (FASTA parsing; optional but recommended)
  - `gensim` (Word2Vec; only needed for Part B)

> The CLI tries to auto‑locate MAFFT and MMseqs2 via `PATH`, `CONDA_PREFIX`, and `whereis`. You can force paths with `--mafft-bin` and `--mmseqs-bin`.

---




## 5. Mathematical Definitions

### 5.1 Pairwise Identity (from MSA)
Let A_i(t) be the residue of sequence i at aligned position t, and let '-' denote a gap.
Identity between sequences i and j is:
  
$$
\mathrm{Id}(i,j)
=
100 \times
\frac{\sum_{t} \mathbf{1}[\, A_i(t)=A_j(t) \wedge A_i(t)\neq - \wedge A_j(t)\neq - \,]}
     {\sum_{t} \mathbf{1}[\, A_i(t)\neq - \wedge A_j(t)\neq - \,]}
\, .
$$

### 5.2 Dissimilarity
  
$$
d_{ij} = 1 - \frac{\mathrm{Id}(i,j)}{100} \; .
$$

### 5.3 Cosine Similarity on W2V Mean Embeddings
For mean vectors v_i, v_j in R^d:
  
$$
\cos(\theta_{ij}) = \frac{v_i \cdot v_j}{\|v_i\|\,\|v_j\|} \; .
$$

For dendrograms based on W2V, we use distances such as
  
$$
1 - \cos(\theta_{ij})\, .
$$

---

## 6. Pipeline Stages

### 6.1 Multiple Sequence Alignment (MAFFT)
- Aligns the selected subset with **MAFFT** (default `--auto`).
- Removes gaps only for **Word2Vec** tokenization; identities continue to use the **gapped alignment** as above.

### 6.2 Pairwise Identities & Dissimilarity Matrix
- We stream the upper triangle of all pairs to compute identity percentages.
- **Dissimilarity** is derived as \(d_{ij}=1-\mathrm{Id}(i,j)/100\).
- For UMAP (identity‑based), we pass a **precomputed** dissimilarity matrix.

### 6.3 Subsampling of Pairs
To limit the number of plotted pairs while keeping global coverage, we deterministically sample pairs using a **stride** over the upper triangle:
\[
\text{stride} = \left\lfloor \frac{N(N-1)/2}{S} \right\rfloor,
\]
where \(N\) is the number of sequences and \(S\) the target pair sample size (e.g., **600,000** and **1,000,000**). Every \(k\)-th pair (modulo `stride`) is retained. This yields evenly‑spaced coverage without bias toward any region of the triangle.

### 6.4 MMseqs2 Clustering Across Identity Cutoffs
- We cluster with **MMseqs2** for cutoffs: **10, 20, …, 90%**.
- Diagnostics:
  - **Intra‑cluster** pair counts over identity **deciles** (10–100) → **lines** and **heatmap**.
  - **Cluster size** vs **cutoff** (mean/median/P95) and **largest cluster** vs cutoff.

### 6.5 Visualizations
- **UMAP of dissimilarity** (metric=`precomputed`):
  - `n_neighbors=20`, `min_dist=0.05`, random seed 42.
  - Colored by **Phylum** with a **distinct, stable palette**.
- **Dendrogram from dissimilarity**:
  - Linkage (default `average`), branches colored by **Phylum** when a clade is homogeneous; mixed clades are shown in neutral gray.
- **Identity distribution barplot** (10–100%, bin 5%).
- **Intra‑cluster counts** (lines + heatmap) by identity deciles × cutoff.
- **Cluster size metrics** (mean/median/P95, largest cluster).

> All plots have **no title**; legends are placed **below** to maximize figure area for data. Axes labels remain.

### 6.6 Optional Part B — Word2Vec Analyses
- Build k‑mer **sentences** from **aligned sequences without gaps**; default \(k=3\).
- Train **Word2Vec** (`vector_size=200`, `window=5`, `sg=1`, `epochs=20`).
- Compute **mean embedding** per sequence.
- Visualizations:
  - **Cosine vs Identity**: 2D hexbin for **intra‑** and **inter‑phylum** pairs, with trend line across identity deciles.
  - **t‑SNE** (`perplexity=30`) and **UMAP** (Euclidean) of mean embeddings by Phylum.
  - **W2V dendrogram** using pairwise distances (default **cosine**).

---

## 7. Color Policy (Distinct Phylum Colors)
- We construct a **global palette** across all phyla using `tab20`, `tab20b`, `tab20c` as base (non‑overlapping), with golden‑ratio HSV fallbacks when needed.
- **Normalization**: `Actinobacteria` → `Actinomycetota`.
- **Excluded from color plots**: labels containing `Unknown`, `uncultured`, `uncultivated`.
- The same palette is reused across all plots for visual consistency.

---

## 8. Outputs & Directory Layout
For an output directory `results_faal`, and a subsample label (e.g., `subset_size_5000`):
```
results_faal/
  subset_size_5000/
    input_subset.fasta
    pairs.csv                       # all streamed pairs
    clusters_multi_threshold.csv    # MMseqs2 clusters for all cutoffs
    subset_ids.csv
    subset_phylum_map.csv
    plots_sample_600000/
      01_bar_identity_10_100.(png|tiff|svg)
      03_dendrogram_branches_by_phylum.(png|tiff|svg)
      04_umap_dissimilarity_by_phylum.(png|tiff|svg)
      06a_counts_intra_identity_bins_by_cutoff_lines_REALX.(png|tiff|svg)
      06b_counts_intra_identity_bins_by_cutoff_heatmap.(png|tiff|svg)
      07_cosine_vs_identity_intra.(png|tiff|svg)   # if --run-part-b
      08_cosine_vs_identity_inter.(png|tiff|svg)   # if --run-part-b
    04_cluster_size_vs_identity_cutoff.(png|tiff|svg)
    05_largest_cluster_by_identity_cutoff.(png|tiff|svg)
    09_tsne_w2v_mean_by_phylum.(png|tiff|svg)      # if --run-part-b
    10_umap_w2v_mean_by_phylum.(png|tiff|svg)      # if --run-part-b
    13_dendrogram_w2v_mean_by_phylum.(png|tiff|svg)# if --run-part-b
    FIGURE_MANIFEST.txt
```

---

## 9. Reproducibility, Performance & Practical Notes
- **Random seed** fixed to **42** for subsetting and reductions.
- **Scalability**: The pairwise identity computation streams the upper triangle and writes results incrementally. Subsampling by stride controls plot sizes (e.g., **600k** and **1M** sampled pairs).
- **MSA cost** still dominates for very large \(N\). Consider subsetting by `--subset-fraction` or `--subset-size` (e.g., 20–40%) before alignment.
- **MMseqs2** is run on the **subset** to match the identities/plots for that subset.
- If metadata is missing, Phylum is `Unknown` and those entries are filtered out of color plots (but still included in numerical computations).
- Use `--mafft-bin` and `--mmseqs-bin` to force binary paths when Conda/`PATH` resolution differs by system.

---

## 10. How to Run — Jupyter Version
1. Open the provided notebook (English) and set:
   - `FASTA_PATH = "FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta"`
   - `TABLE_S2_PATH = "Table_S2.tsv"` (optional but recommended)
   - `OUTDIR = "results_faal"`
2. Choose subsetting parameters (examples):
   - `subset = SubsetParams(fraction=0.20, seed=42)`
   - or `subset = SubsetParams(size=5000, seed=42)`
3. Set pair sample sizes, e.g.:
   - `sample_sizes = [600_000, 1_000_000]`
4. Run the **orchestration cell**: `run_pipeline(...)` with your preferred arguments.
5. (Optional) Enable Part B by `run_part_b=True`.

> The notebook exports all figures in `.png`/`.tiff` (900 dpi) and `.svg`, prints where files are saved, and keeps legends below plots. The two composition‑based UMAPs are intentionally **removed**.

---

## 11. How to Run — CLI Version
The CLI script is `faalprot_heterogeneity_cli_v1.py` (see `--help`).

**Minimal run**:
```bash
python faalprot_heterogeneity_cli_v1.py \
  --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta \
  --table Table_S2.tsv \
  --outdir results_faal
```

**With Word2Vec Part B**:
```bash
python faalprot_heterogeneity_cli_v1.py \
  --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta \
  --table Table_S2.tsv \
  --outdir results_faal_w2v \
  --run-part-b
```

**Custom subsampling + pair sampling**:
```bash
python faalprot_heterogeneity_cli_v1.py \
  --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta \
  --table Table_S2.tsv \
  --outdir results_custom \
  --subset-fraction 0.20 \
  --sample-sizes 600_000,1_000_000 \
  --run-part-b
```

**Explicit binary paths**:
```bash
python faalprot_heterogeneity_cli_v1.py \
  --fasta FAAL_NR_MIBIG_LEGE_FFT_NS_2_raw_data.fasta \
  --table Table_S2.tsv \
  --outdir results_bins \
  --mafft-bin /usr/bin/mafft \
  --mmseqs-bin /home/USER/anaconda3/envs/fall/bin/mmseqs
```

---

## 12. FAQ
**Q1. Why sample pairs instead of computing all?**  
The number of pairs grows as \(N(N-1)/2\). Even when we compute all identities, we **plot** only a **deterministic subsample** (e.g., 600k and 1M) for clarity and performance.

**Q2. Why exclude “Unknown/uncultured” from color plots?**  
Color encodes discrete taxa (Phylum). Unknown labels would receive arbitrary colors and clutter legends without adding biological signal.

**Q3. Why Word2Vec on gapped‑free sequences, but identity on gapped alignment?**  
W2V models sequence **content/context** (token k‑mers) and benefits from removing gaps introduced by alignment. Identity must respect aligned positions, hence uses the **gapped** MSA.

**Q4. How are dendrogram colors assigned?**  
Branches are colored by the **unique** phylum if all descendants share it; otherwise the branch is gray, highlighting mixed clades.

---

## 13. References
- Katoh, K. & Standley, D.M. (2013). **MAFFT** multiple sequence alignment software version 7: improvements in performance and usability. *Mol. Biol. Evol.*
- Steinegger, M. & Söding, J. (2017). **MMseqs2** enables sensitive protein sequence searching. *Nat. Biotechnol.*
- McInnes, L., Healy, J., & Melville, J. (2018). **UMAP**: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- van der Maaten, L. & Hinton, G. (2008). **t‑SNE**. *JMLR*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Word2Vec**/Distributed Representations of Words and Phrases. *NIPS*.
- Virtanen, P. *et al.* (2020). **SciPy** 1.0: fundamental algorithms for scientific computing in Python. *Nat. Methods*.
- Pedregosa, F. *et al.* (2011). **scikit‑learn**: Machine Learning in Python. *JMLR*.
- Hunter, J.D. (2007). **Matplotlib**: A 2D graphics environment. *Computing in Science & Engineering*.
- Cock, P.J.A. *et al.* (2009). **Biopython**: freely available Python tools for computational biology. *Bioinformatics*.


