# FAALPred: Fatty Acyl-AMP Ligases (FAAL) Prediction Tool

This document provides a comprehensive Markdown-friendly explanation and usage guide for the **FAALPred** code. You can copy and paste this document into your GitHub repository’s README or any publication. The document is structured to make it easy to understand what each function does, how to run the code, and how the workflow is organized.

Supplementary_methodology.dox

1. [Detailed Explanation of Each Function](#ExplanationofEachFunction)
   - `are_sequences_aligned`
   - `create_unique_model_directory`
   - `realign_sequences_with_mafft`
   - `perform_clustering`
   - `plot_dual_tsne`
   - `plot_roc_curve_global`
   - `get_class_rankings_global`
   - `calculate_roc_values`
   - `visualize_latent_space_with_similarity`
   - `format_and_sum_probabilities`
2. [Support (Class)](#support-class)
   - `_oversample_single_sample_classes`
   - `fit`
   - `_perform_grid_search`
   - `get_best_param`
   - `plot_learning_curve` (inside Support)
   - `get_class_rankings` (inside Support)
   - `test_best_RF`
   - `_calculate_score`
   - `plot_roc_curve` (inside Support)
3. [ProteinEmbeddingGenerator (Class)](#proteinembeddinggenerator-class)
   - `generate_embeddings`
   - `get_embeddings_and_labels`
   - `compute_perplexity`
   - `plot_dual_umap`
   - `plot_predictions_scatterplot_custom`
   - `adjust_predictions_global`
   - `main`
4. [Additional Streamlit Setup and Theming](#AdditionalStreamlitSetupandTheming)

## Overview
**FAALPred** is a comprehensive bioinformatics tool designed to predict fatty acid chain-length specificity (ranging from C4 to C18) of **Fatty Acyl-AMP Ligases (FAALs)**. It integrates:


## Features
- **MAFFT alignment** (if sequences are unaligned)
- **Word2Vec** to generate embeddings from protein sequences
- **Random Forest** and calibration strategies
- **Oversampling** (`RandomOverSampler`, `SMOTE`) for balancing classes
- **UMAP** and **t-SNE** for dimensionality reduction and visualization
- **ROC** and **Precision-Recall AUC** metrics for performance evaluation
- **A Streamlit interface** that guides through training and prediction

---

## Requirements and Installation

### Python Version:
- `>= 3.8`

### Required Libraries:
Install the following libraries using `pip`:

```bash
pip install streamlit numpy pandas scikit-learn scipy gensim plotly \
matplotlib joblib biopython imblearn umap-learn Pillow tabulate base64



# Step 1: 

```bash
# sudo nano ~/.streamlit/config.toml
```
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 8501

# Step 2:

# Run the Application (Streamlit Approach)

## Start the application:
```bash
streamlit run faalpred.py
```

# Step3:
```bash
http://localhost:8501/
  ```




