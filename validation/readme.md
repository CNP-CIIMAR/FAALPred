# External validation of FAALPred and AdenylPred substrate chain-length predictions

This directory contains the code used for the external evaluation of FAALPred and AdenylPred substrate chain-length predictions for FAAL enzymes, as described in:

> **Diversity of FAAL enzymes and prediction of their substrate specificity using FAALPred**  
> Anne Liong¹²†, Leandro de Mattos Pereira¹† and Pedro N. Leão¹*  
> ¹ Interdisciplinary Centre of Marine and Environmental Research (CIIMAR/CIMAR), University of Porto, Matosinhos, 4450-208, Portugal  
> ² ICBAS – School of Medicine and Biomedical Sciences, University of Porto, Porto, 4050-313, Portugal  
> \* To whom correspondence should be addressed. Email: pleao@ciimar.up.pt  
> † Anne Liong and Leandro de Mattos Pereira contributed equally to this work.

The script `faal_adenyl_validation.py` reproduces the analysis used to generate the confusion matrices and per-class performance metrics (Sensitivity, Specificity, Precision and F1-score) shown in **Supplementary Fig. S48** and summarized in **Table 7** of the manuscript, using **Supplementary Table S6** as input.

Repository location:  
`https://github.com/CNP-CIIMAR/FAALPred/tree/main/validation`

---

## Contents

- `faal_adenyl_validation.py`  
  Main analysis script to:
  - parse experimental substrates from the literature/in vitro data,
  - map them onto FAALPred and AdenylPred chain-length bins,
  - build confusion matrices (true vs predicted bins),
  - compute per-class performance metrics,
  - generate a 2×2 panel figure (FAALPred and AdenylPred).

- `Supplementary_Table_S6.tsv` (or equivalent)  
  Input table containing the curated external validation set, including:
  - protein identifiers,
  - experimental substrates (literature and in vitro),
  - FAALPred and AdenylPred predictions.

---

## Dependencies

The script was written for **Python ≥ 3.8** and uses:

- `numpy`
- `pandas`
- `matplotlib`

You can install these packages either in your base environment or, preferably, in a dedicated **conda** environment (see below).

---

## Creating a conda environment

To reproduce the analysis in an isolated environment, you can create a conda environment as follows:

```bash
# Create a new environment with Python 3.10 (or another supported version)
conda create -n faalpred_validation python=3.10

# Activate the environment
conda activate faalpred_validation

# Install required packages
conda install numpy pandas matplotlib
```

Alternatively, you can install the packages via `pip` inside the activated environment:

```bash
conda activate faalpred_validation
pip install numpy pandas matplotlib
```

After this, you should be able to run `faal_adenyl_validation.py` from within the `faalpred_validation` environment.

---

## Input format

The script expects a tabular file (TSV/CSV) with at least the following columns:

- `Protein`  
- `Refseq/GenBank`  
- `Substrate in literature`  
- `Species`  
- `FAALPred, Prediction Score`  
- `AdenylPred, Prediction Score`  

In the repository, this corresponds to **Supplementary Table S6**, placed in the same directory as the script (e.g. `validation/`).

The FAALPred and AdenylPred columns are strings such as:

- `C12-C14-C16 (0.74)`  
- `C13 through C17 (47%)`

Only the categorical label (before the parenthesis) is used in the analysis.

---

## Overview of the analysis

### 1. Parsing experimental substrates

For each protein, the script extracts all carbon chain lengths from the field **“Substrate in literature”** using a regular expression. Examples:

- `"C8:0, C10:0, C12:0 (mainly C:10)" → [8, 10, 12]`  
- `"C12:0, C14:0, C16:0, C48:0 to C62:0" → [12, 14, 16, 48, 62]`

Qualitative modifiers such as “mainly”, “preferred”, “tested”, etc. are intentionally ignored, so that all explicitly reported chain lengths contribute equally.

We denote the set of experimental chain lengths for a given enzyme by:

$$
\{c_1, c_2, \dots, c_n\}
$$

and define:

$$
c_{\min} = \min_i c_i, \quad
c_{\max} = \max_i c_i.
$$

### 2. Chain-length bins for FAALPred and AdenylPred

FAALPred predictions are reported as triplet labels, which are interpreted as closed integer ranges:

- `C4-C6-C8` → $[4, 8]$  
- `C8-C10-C12` → $[8, 12]$  
- `C12-C14-C16` → $[12, 16]$  
- `C14-C16-C18` → $[14, 18]$  

AdenylPred predictions use broader categories:

- `C6 through C12` → $[6, 12]$  
- `C13 through C17` → $[13, 17]$  

Before binning, experimental chain lengths are clipped to the range relevant for each method:

- **FAALPred**: only chain lengths in $[4, 18]$ are retained.  
- **AdenylPred**: only chain lengths in $[6, 17]$ are retained.

If all experimental chain lengths fall outside the clipping range, the enzyme is excluded from the corresponding method’s evaluation.

### 3. Overlap-based mapping of experimental data to model bins

For each enzyme and each model bin with range $[r_{\min}, r_{\max}]$, we compute the **discrete overlap** between the experimental range $[c_{\min}, c_{\max}]$ and the model bin as:

$$
\text{inter}_{\min} = \max(c_{\min}, r_{\min}), \quad
\text{inter}_{\max} = \min(c_{\max}, r_{\max}),
$$

$$
\text{overlap} = \max\left(0,\ \text{inter}_{\max} - \text{inter}_{\min} + 1\right).
$$

This corresponds to the number of integer chain lengths shared by the two intervals (e.g. the overlap between $[6, 8]$ and $[4, 8]$ is 3, corresponding to C6, C7 and C8).

For each enzyme, the **true bin** is defined as the model bin with the largest overlap:

- `True_FAAL_interval` for FAALPred,  
- `True_Adenyl_interval` for AdenylPred.

The script simultaneously reads the **predicted** bin from the FAALPred/AdenylPred output strings (by discarding the numerical score and keeping only the categorical label), yielding:

- `FAALPred_interval`  
- `AdenylPred_interval`.

### 4. Confusion matrices and row-wise percentages

For each method, a confusion matrix is built with rows corresponding to true bins and columns to predicted bins:

- FAALPred: `crosstab(True_FAAL_interval, FAALPred_interval)`  
- AdenylPred: `crosstab(True_Adenyl_interval, AdenylPred_interval)`  

Let $N_{ij}$ denote the count of enzymes whose **true** bin is $i$ and **predicted** bin is $j$. To facilitate interpretation, the script converts counts into row-wise percentages:

$$
P_{ij} = 100 \times \frac{N_{ij}}{\sum_j N_{ij}}.
$$

Each row of $P_{ij}$ therefore sums to 100%, showing how each true class is distributed over the predicted classes.

The script plots these matrices as heatmaps with:

- y-axis: **True label**  
- x-axis: **Predicted label** (tick labels vertical)  
- each cell annotated with $P_{ij}$ (one decimal place)  
- a color bar labelled “Row-wise percentage”.

### 5. Per-class performance metrics

Using a one-vs-all scheme, the script computes, for each class $c$:

- True positives: $TP_c$  
- False negatives: $FN_c$  
- False positives: $FP_c$  
- True negatives: $TN_c$  

from the full confusion matrix.

From these, the following metrics are calculated:

$$
\text{Sensitivity}_c = \frac{TP_c}{TP_c + FN_c},
$$

$$
\text{Specificity}_c = \frac{TN_c}{TN_c + FP_c},
$$

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c},
$$

$$
F1_c = \frac{2 \times \text{Precision}_c \times \text{Sensitivity}_c}
            {\text{Precision}_c + \text{Sensitivity}_c}.
$$

If any denominator is zero, the corresponding metric is reported as `NaN` (not defined).

The script saves these per-class metrics to CSV files and creates grouped bar plots showing Sensitivity, Specificity and F1 per class.

---

## Combined panel figure (Supplementary Fig. S48)

The function `plot_combined_panel` creates a single **2×2 panel** figure:

- **Panel A**: FAALPred confusion matrix (row-wise percentages)  
- **Panel B**: FAALPred per-class metrics (bar plot)  
- **Panel C**: AdenylPred confusion matrix (row-wise percentages)  
- **Panel D**: AdenylPred per-class metrics (bar plot)

Panels are labelled “A”, “B”, “C”, “D” in the upper-left corner of each subplot. The layout is optimized for journal-quality output (increased left and bottom margins, extra vertical spacing between the top and bottom rows, and legends placed to the right of the bar plots).

The combined figure is exported in three formats:

- `<prefix>_combined_panel.png` (600 dpi)  
- `<prefix>_combined_panel.tiff` (600 dpi)  
- `<prefix>_combined_panel.svg`  

where `<prefix>` is derived from the input file name and output directory.

---

## Running the script

From the `validation/` directory (where the script and Supplementary Table S6 are located):

```bash
conda activate faalpred_validation  # if using the suggested conda env

python faal_adenyl_validation.py \
  -i Supplementary_Table_S6.tsv \
  -o results \
  --sep "\t"
```

Command-line arguments:

- `-i, --input`  
  Path to the input table (TSV/CSV) with substrates and predictions  
  (e.g. `Supplementary_Table_S6.tsv`).

- `--sep`  
  Field separator for the input file. Default: `"\t"` (TSV).

- `-o, --output_dir`  
  Directory to save output files. Default: current directory (`"."`).

### Output files

For an input file named `Supplementary_Table_S6.tsv` and `-o results`, the script will produce, among others:

- `results/Supplementary_Table_S6_faal_metrics.csv`  
  Per-class metrics for FAALPred.

- `results/Supplementary_Table_S6_adenyl_metrics.csv`  
  Per-class metrics for AdenylPred.

- `results/Supplementary_Table_S6_combined_panel.png`  
- `results/Supplementary_Table_S6_combined_panel.tiff`  
- `results/Supplementary_Table_S6_combined_panel.svg`  

These correspond to **Supplementary Fig. S48** in the manuscript.

---

## Reproducibility

Running `faal_adenyl_validation.py` on the curated data set (Supplementary Table S6) reproduces:

- the overlap-based mapping of experimental substrates onto FAALPred and AdenylPred bins,
- the confusion matrices used to quantify agreement between predicted and experimental chain-length ranges,
- the per-class metrics reported in **Table 7**, and
- the combined panel shown as **Supplementary Fig. S48** in the article  
  *“Diversity of FAAL enzymes and prediction of their substrate specificity using FAALPred”*.
