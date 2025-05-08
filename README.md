# FAALPred: Fatty Acyl-AMP Ligases (FAAL) Prediction Tool

This document provides a comprehensive, Markdown-friendly explanation and usage guide for the FAALPred code. The Supplementary Methodology is structured to clearly outline the main functions and workflow of FAALPred. 
The file Supplementary_methodology.dox serves as a detailed reference for users.



1. [Function](#ExplanationofEachFunction)
   - `are_sequences_aligned`
   - `create_unique_model_directory`
   - `realign_sequences_with_mafft`
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
   - `plot_predictions_scatterplot_custom`
   - `adjust_predictions_global`
   - `main`
4. [Additional Streamlit Setup and Theming](#AdditionalStreamlitSetupandTheming)

## Overview
**FAALPred** is a comprehensive bioinformatics tool designed to predict fatty acid chain-length specificity (ranging from C4 to C18) of **Fatty Acyl-AMP Ligases (FAALs)**. It integrates:

## MainFeatures
- **MAFFT alignment** (if sequences are unaligned)
- **Word2Vec** to generate embeddings from protein sequences
- **Random Forest** and calibration strategies
- **Oversampling** (`RandomOverSampler`, `SMOTE`) for balancing classes
- **UMAP** for dimensionality reduction and visualization
- **ROC** and **Precision-Recall AUC** metrics for performance evaluation
- **A Streamlit interface** that guides through training and prediction

---

## Requirements and Installation

### Python Version:
- `>= 3.8`


## 1. Install Anaconda (if not already installed)
If Anaconda is not installed, download and install it from [here](https://www.anaconda.com/download).

Alternatively, you can install **Miniconda** (a lightweight version of Anaconda) from [here](https://docs.conda.io/en/latest/miniconda.html).

## 2. Create a New Conda Environment

### Run the following command to create a Conda environment named **faalpred_env**:

```bash
conda create -n faalpred_env python=3.9 -y
```
```
conda activate faalpred_env
```
conda install -y numpy pandas scikit-learn matplotlib seaborn joblib biopython umap-learn imbalanced-learn pillow tabulate plotly -c conda-forge
```

### Install Additional Libraries via pip

Some packages are not available in Conda by default and must be installed using pip:

```bash
pip install streamlit gensim argparse base64
```

### Verify Installation

- After installing the required packages, you can verify the installation by running:

```bash
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, joblib, Bio, umap, imblearn, PIL, tabulate, plotly, streamlit, gensim; print('All packages installed successfully!')"
```

### Each time you need to work with FAALPred, activate the environment with:
```bash
conda activate faalpred_env
```
```bash
To deactivate it, simply run:
```
### Configure the TOML File

To configure Streamlit, edit the `config.toml` file by running the following command:

```bash
sudo nano ~/.streamlit/config.toml
```
### Streamlit Server Configuration

Add the following content to your `config.toml` file:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 8501


### Run the Application (Streamlit Approach)

## Start the application in the conda environment:
```bash
streamlit run faalpred.py
```
### Access in your web browser:

```bash
http://localhost:8501/
  ```




