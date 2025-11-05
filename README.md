
# FAALPred: Fatty Acyl-AMP Ligases (FAAL) Prediction Tool

This document provides a comprehensive, Markdown-friendly explanation and usage guide for the FAALPred code.  
The Supplementary Methodology is structured to clearly outline the main functions and workflow of FAALPred.  
The file **`Supplementary_methodology.dox`** serves as a detailed reference for users.

---

## Functions and Classes

### Standalone Functions

- `are_sequences_aligned`
- `create_unique_model_directory`
- `realign_sequences_with_mafft`
- `plot_roc_curve_global`
- `get_class_rankings_global`
- `calculate_roc_values`
- `visualize_latent_space_with_similarity`
- `format_and_sum_probabilities`
- `plot_predictions_scatterplot_custom`
- `adjust_predictions_global`
- `main`

### Support (class)

- `_oversample_single_sample_classes`
- `fit`
- `_perform_grid_search`
- `get_best_param`
- `plot_learning_curve` (method of `Support`)
- `get_class_rankings` (method of `Support`)
- `test_best_RF`
- `_calculate_score`
- `plot_roc_curve` (method of `Support`)

### ProteinEmbeddingGenerator (class)

- `generate_embeddings`
- `get_embeddings_and_labels`

### Additional Streamlit Setup and Theming

The codebase also includes extended Streamlit configuration and theming to provide:

- A customized sidebar and main layout
- Styled tables and buttons
- Downloadable plots and result tables
- Integration with external tools (InterProScan, bedtools) for FAAL domain extraction

---

## Overview

FAALPred is a comprehensive bioinformatics tool designed to predict fatty-acid chain-length specificity (ranging from C4 to C18) of Fatty Acyl-AMP Ligases (FAALs).  
It integrates several computational approaches in a single workflow:

### Main Features

- **MAFFT alignment** for realigning sequences when they are unaligned.
- **Word2Vec embeddings** to represent protein sequences via sliding-window k-mers.
- **Random Forest classifier** with probability calibration for substrate prediction.
- **Oversampling strategies** using `RandomOverSampler` and `SMOTE` to balance classes in small datasets.
- **UMAP-based dimensionality reduction and visualization** to inspect latent spaces and synthetic vs. original samples.
- **Performance evaluation** with ROC curves, Precision–Recall AUC, F1 scores, calibration curves, and confusion matrices.
- **Interactive Streamlit interface** guiding the user through training, prediction, visualization, and exporting results.

---

## Requirements and Installation

FAALPred is available as:

- A public web server: **https://faalpred.ciimar.up.pt/**
- A standalone package from GitHub: **https://github.com/CNP-CIIMAR/FAALPred**

It is implemented in Python and distributed with a Conda environment file  
**`faalpred_environment.yml`** that installs all required dependencies.

### Python version

FAALPred was developed and tested with **Python 3.9**  
(Python ≥ 3.8 should also work, but 3.9 is recommended).

---

## 0. Install Git and Conda (if needed)

### Install Git

- **Ubuntu / Debian (Linux)**:
  ```bash
  sudo apt update
  sudo apt install git
  ```
- **macOS**:
  - Either install Xcode Command Line Tools:
    ```bash
    xcode-select --install
    ```
  - Or install via Homebrew:
    ```bash
    brew install git
    ```
- **Windows**:
  - Download and install Git from: https://git-scm.com/downloads

### Install Anaconda or Miniconda

If Conda is not installed, install either:

- **Anaconda**: https://www.anaconda.com/download  
- **Miniconda** (lighter): https://docs.conda.io/en/latest/miniconda.html  

After installation, open a **terminal** (or Anaconda Prompt on Windows).

---

## 1. Clone the FAALPred repository

In your terminal, run:

```bash
git clone https://github.com/CNP-CIIMAR/FAALPred.git
cd FAALPred
```

After these commands:

- You will be inside the `FAALPred` directory.
- The file **`faalpred_environment.yml`** will be in this directory.

---

## 2. Create and activate the Conda environment

From inside the `FAALPred` directory, create the environment using the provided YAML file:

```bash
conda env create -f faalpred_environment.yml
conda activate faalpred
```

The environment name is **`faalpred`** (defined in the YAML file).

Each time you want to work with FAALPred, you must activate the environment:

```bash
conda activate faalpred
```

To deactivate it when you are done:

```bash
conda deactivate
```

---

## 3. (Optional) Verify the installation

You can quickly verify that the most important packages are correctly installed:

```bash
python -c "import numpy, pandas, sklearn, matplotlib, plotly, streamlit, gensim, Bio; print('All key packages imported successfully!')"
```

If you see the message **`All key packages imported successfully!`**, the environment is correctly set up.

---

## 4. Configure Streamlit (optional, for server deployment)

To customize Streamlit’s server behavior (useful on remote servers), create or edit the file  
`~/.streamlit/config.toml`:

```bash
mkdir -p ~/.streamlit
nano ~/.streamlit/config.toml
```

Add the following content:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 8501
```

Save and close the file.

This step is **optional** for local use but helpful if running FAALPred on a server.

---

## 5. Run the FAALPred application (Streamlit)

With the **`faalpred`** environment activated and from inside the `FAALPred` directory, run:

```bash
streamlit run faalpred.py
```

Streamlit will start the web application and print a URL in the terminal, typically:

```text
  Local URL: http://localhost:8501
```

Open this URL in your web browser to use **FAALPred** locally.

If you are running on a remote server, use the appropriate host/port combination (for example, `http://server_ip:8501`) depending on your network and firewall settings.

---

## 6. Use the public FAALPred web server

If you do not wish to install anything locally, you can use FAALPred directly via the public web interface:

👉 **https://faalpred.ciimar.up.pt/**
