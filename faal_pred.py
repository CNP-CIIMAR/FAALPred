import logging
import os
import sys
import subprocess
import random
import zipfile
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import shutil
import time
import argparse 
import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MafftCommandline
import joblib
import plotly.io as pio
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize, MultiLabelBinarizer
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from matplotlib import ticker
import umap.umap_ as umap  # Import for UMAP
import umap
import base64
from plotly.graph_objs import Figure
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import plotly.express as px
import streamlit as st
import pandas as pd
import requests
import os
import shutil
import time
import requests
import subprocess
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration  import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

# ============================================
# Streamlit Configuration and Interface
# ============================================
st.set_page_config(
    page_title="FAAL_Pred",
    #page_icon="ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬",  # DNA Symbol
    page_icon="Ã°Å¸Â§Â¬", 
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    /* =======================
       FULL SIDEBAR RESET
       ======================= */
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] div[role="presentation"],
    [data-testid="stSidebar"] div[role="toolbar"],
    [data-testid="stSidebar"] .stElementContainer,
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] .css-1minutp {
        margin: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] .stButton {
        margin: 0 !important;
    }
    [data-testid="stSidebar"] button[aria-label="Run Analysis"] {
        background-color: #000 !important;
        color: #FFF !important;
        border: none !important;
    }
    [data-testid="stSidebar"] button[aria-label="Run Analysis"]:hover {
        background-color: #222 !important;
    }

    /* =======================
   Ã‚Â«    APP STYLES
       ======================= */
    .stApp {
        background: linear-gradient(145deg, #fafafa 0%, #f0f0f0 100%) fixed !important;
        box-shadow: inset 0 4px 12px rgba(0, 0, 0, 0.06) !important;
        color: #000000 !important;
    }
    [data-testid="stAppViewContainer"] .block-container {
        background: #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05) !important;
        padding: 1.5rem 2rem !important;
        margin-bottom: 1.5rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
    }
    div[role="alert"] {
        background: #fff7e6 !important;
        border-left: 4px solid #f0ad4e !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        padding: 0.75rem 1rem !important;
    }
    div[role="alert"] p {
        color: #000000 !important;
    }

    /* =======================
       SIDEBAR STYLES
       ======================= */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f8f8 100%) !important;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1) !important;
        border-radius: 0 12px 12px 0 !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] * {
        background: #ffffff !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox span {
        color: #000000 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stTextInput > div,
    [data-testid="stSidebar"] .stNumberInput > div,
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stSlider > div,
    [data-testid="stSidebar"] .stFileUploader > div {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        border-radius: 6px !important;
    }

    /* =======================
       SIDEBAR ACTION BUTTONS
       ======================= */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #383838 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08) !important;
        border-radius: 8px !important;
        transition: transform 0.1s ease-out, box-shadow 0.1s ease-out !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12) !important;
        background-color: #4a4a4a !important;
    }

    /* =======================
       GLOBAL BUTTON STYLES
       ======================= */
    .stButton > button {
        background-color: #383838 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 12px 28px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
        transition: transform 0.15s ease-out, box-shadow 0.15s ease-out !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.16) !important;
        background-color: #4a4a4a !important;
    }

    /* =======================
      TABLE STYLES
       ======================= */
    .dataframe-container {
        background: #ffffff !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 1.5rem !important;
    }
    table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
    }
    table thead th {
        background: linear-gradient(90deg, #e0e5f1 0%, #f5f7fb 100%) !important;
        color: #333333 !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        border-bottom: 2px solid #d1d7e0 !important;
        text-align: center !important;
    }
    table tbody tr:nth-child(odd)  { background: #ffffff !important; }
    table tbody tr:nth-child(even) { background: #f9f9fa !important; }
    table tbody td {
        padding: 0.65rem !important;
        color: #333333 !important;
        border-bottom: 1px solid #ececec !important;
        text-align: center !important;
    }
    table tbody tr:hover { background: #f1f4f8 !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================
# Definitions of Functions and Classes
# ============================================

# Setting seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# FunÃƒÂ§ÃƒÂ£o para verificar se as sequÃƒÂªncias estÃƒÂ£o alinhadas
def are_sequences_aligned(fasta_file: str) -> bool:
    """
    Checks if all sequences in a FASTA file are aligned by verifying they have the same length.
    
    Parameters:
    - fasta_file (str): Path to the FASTA file.
    
    Returns:
    - bool: True if all sequences are aligned (same length), False otherwise.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Returns True if all sequences have the same length

# FunÃƒÂ§ÃƒÂ£o para criar diretÃƒÂ³rio ÃƒÂºnico para modelos
def create_unique_model_directory(base_dir: str, aggregation_method: str) -> str:
    """
    Creates a unique directory for models based on the aggregation method.
    
    Parameters:
    - base_dir (str): Base directory for models.
    - aggregation_method (str): Aggregation method used.
    
    Returns:
    - str: Path to the unique model directory.
    """
    model_dir = os.path.join(base_dir, f"models_{aggregation_method}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

# FunÃƒÂ§ÃƒÂ£o para realinhar sequÃƒÂªncias utilizando MAFFT
def realign_sequences_with_mafft(input_path: str, output_path: str, threads: int = 8) -> None:
    """
    Realigns sequences using MAFFT.
    
    Parameters:
    - input_path (str): Path to the input file.
    - output_path (str): Path to save the realigned file.
    - threads (int): Number of threads for MAFFT.
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Realigned sequences saved to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing MAFFT: {e.stderr.decode()}")
        sys.exit(1)

def plot_roc_curve_global(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
    """
    Plots the ROC curve for binary or multiclass classifications.
    
    Parameters:
    - y_true (np.ndarray): True labels.
    - y_pred_proba (np.ndarray): Predicted probabilities.
    - title (str): Title of the plot.
    - save_as (str): Path to save the plot.
    - classes (list): List of classes (for multiclass).
    """
    lw = 2  # Line width

    # Check if it's binary or multiclass classification
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
    else:  # Multiclass classification
        y_bin = label_binarize(y_true, classes=unique_classes)
        n_classes = y_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            class_label = classes[i] if classes is not None else unique_classes[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f'ROC Curve for class {class_label} (area = {roc_auc[i]:0.2f})')

    # Diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    # Limits e rÃ³tulos
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.title(title, color='white')

    # Legenda Ã  direita do grÃ¡fico
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='white')  # Match background color
    plt.close()

def plot_precision_recall_curve_global(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str,
    save_as: str = None,
    classes: list = None
) -> None:
    """
    Plots the Precision-Recall curve for binary or multiclass classifications,
    com a legenda posicionada fora, Ã  direita do grÃ¡fico.
    """
    lw = 2  # Line width
    unique_classes = np.unique(y_true)

    # Cria figura e eixo
    fig, ax = plt.subplots()
    # Empurra o eixo para deixar espaÃ§o Ã  direita
    fig.subplots_adjust(right=0.75)

    if len(unique_classes) == 2:  # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        ap = average_precision_score(y_true, y_pred_proba[:, 1])
        ax.plot(recall, precision,
                color='darkorange', lw=lw,
                label=f'PR Curve (AP = {ap:0.2f})')
    else:  # Multiclass
        y_bin = label_binarize(y_true, classes=unique_classes)
        n_classes = y_bin.shape[1]
        precision = {}
        recall = {}
        ap = {}
        colors = plt.cm.plasma(np.linspace(0, 1, n_classes))

        for i, color in enumerate(colors):
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
            ap[i] = average_precision_score(y_bin[:, i], y_pred_proba[:, i])
            class_label = classes[i] if classes is not None else unique_classes[i]
            ax.plot(recall[i], precision[i],
                    color=color, lw=lw,
                    label=f'PR Curve for {class_label} (AP = {ap[i]:0.2f})')

    # RÃ³tulos e tÃ­tulo
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)

    # Legenda fora do grÃ¡fico, Ã  direita
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Salva e fecha
    if save_as:
        fig.savefig(save_as, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def get_class_rankings_global(model: RandomForestClassifier, X: np.ndarray) -> list:
    """
    Obtains class rankings based on the predicted probabilities from the model.
    
    Parameters:
    - model (RandomForestClassifier): Trained model.
    - X (np.ndarray): Data to obtain predictions.
    
    Returns:
    - list: List of formatted class rankings for each sample.
    """
    if model is None:
        raise ValueError("Model not trained. Please train the model first.")

    # Obtain probabilities for each class
    y_pred_proba = model.predict_proba(X)

    # Ranking classes based on probabilities
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings
 
def visualize_latent_space_with_similarity(
    X_original: np.ndarray, 
    X_synthetic: np.ndarray, 
    y_original: np.ndarray, 
    y_synthetic: np.ndarray, 
    protein_ids_original: list, 
    protein_ids_synthetic: list, 
    var_assoc_original: list, 
    var_assoc_synthetic: list, 
    output_dir: str = None
) -> Figure:
    """
    Visualizes the latent space using 3D UMAP with similarity measures between original and synthetic samples.
    Uses a single fit_transform on the combined data so que ambas as populaÃ§Ãµes compartilham a mesma projeÃ§Ã£o.
    """
    n_orig = X_original.shape[0]
    n_syn = X_synthetic.shape[0]

    # 1) Combina os dados
    X_combined = np.vstack([X_original, X_synthetic])

    # 2) Aplica UMAP uma Ãºnica vez
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    X_transformed = umap_reducer.fit_transform(X_combined)

    # 3) Separa de volta
    X_transformed_original = X_transformed[:n_orig]
    X_transformed_synthetic = X_transformed[n_orig:]

    # 4) Calcula similaridades no espaÃ§o original
    similarities = cosine_similarity(X_synthetic, X_original)
    max_similarities = similarities.max(axis=1)
    closest_indices = similarities.argmax(axis=1)

    # 5) Monta DataFrames
    df_original = pd.DataFrame({
        'x': X_transformed_original[:, 0],
        'y': X_transformed_original[:, 1],
        'z': X_transformed_original[:, 2],
        'Protein ID': protein_ids_original,
        'Associated Variable': var_assoc_original,
        'Type': 'Original'
    })
    df_synthetic = pd.DataFrame({
        'x': X_transformed_synthetic[:, 0],
        'y': X_transformed_synthetic[:, 1],
        'z': X_transformed_synthetic[:, 2],
        'Protein ID': protein_ids_synthetic,
        'Associated Variable': var_assoc_synthetic,
        'Similarity': max_similarities,
        'Closest Protein': [protein_ids_original[idx] for idx in closest_indices],
        'Closest Variable': [var_assoc_original[idx] for idx in closest_indices],
        'Type': 'Synthetic'
    })

    # 6) Cria a figura interativa
    fig = go.Figure()

    # Originais
    fig.add_trace(go.Scatter3d(
        x=df_original['x'], y=df_original['y'], z=df_original['z'],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.7),
        name='Original',
        text=df_original.apply(
            lambda r: f"Protein ID: {r['Protein ID']}<br>Associated Variable: {r['Associated Variable']}", 
            axis=1
        ),
        hoverinfo='text'
    ))

    # SintÃ©ticos
    fig.add_trace(go.Scatter3d(
        x=df_synthetic['x'], y=df_synthetic['y'], z=df_synthetic['z'],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.7),
        name='Synthetic',
        text=df_synthetic.apply(
            lambda r: (
                f"Protein ID: {r['Protein ID']}<br>"
                f"Associated Variable: {r['Associated Variable']}<br>"
                f"Similarity: {r['Similarity']:.4f}<br>"
                f"Closest Protein: {r['Closest Protein']}<br>"
                f"Closest Variable: {r['Closest Variable']}"
            ), 
            axis=1
        ),
        hoverinfo='text'
    ))

    # Layout
    fig.update_layout(
        title="Latent Space Visualization with Similarity (UMAP 3D)",
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3"
        ),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )

    # Salvamento
    if output_dir:
        out_path = os.path.join(output_dir, "umap_similarity_3D.html")
        fig.write_html(out_path)
        logging.info(f"UMAP plot salvo em {out_path}")

    return fig

def format_and_sum_probabilities(associated_rankings: list) -> tuple:
    """
    Formats and sums probabilities for each category, returning only the main category (SS Prediction Specificity)
    and a normalized prediction confidence score (0-1) based on the sum of the top 3 categories.
    Os valores extraÃƒÂ­dos (em percentuais) sÃƒÂ£o convertidos para decimais e, em seguida, normalizados
    dividindo-se pela soma total (garantindo que a soma dos top 3 seja uma fraÃƒÂ§ÃƒÂ£o da distribuiÃƒÂ§ÃƒÂ£o).
    
    Parameters:
    - associated_rankings (list): List of associated rankings.
    
    Returns:
    - tuple: (main category, prediction confidence as a decimal value)
    """
    category_sums = {}
    categories = ['C4-C6-C8', 'C6-C8-C10', 'C8-C10-C12', 'C10-C12-C14', 'C12-C14-C16', 'C14-C16-C18']
    pattern_mapping = {
        'C4-C6-C8': ['C4', 'C6', 'C8'],
        'C6-C8-C10': ['C6', 'C8', 'C10'],
        'C8-C10-C12': ['C8', 'C10', 'C12'],
        'C10-C12-C14': ['C10', 'C12', 'C14'],
        'C12-C14-C16': ['C12', 'C14', 'C16'],
        'C14-C16-C18': ['C14', 'C16', 'C18'],
    }
    for category in categories:
        category_sums[category] = 0.0
    for rank in associated_rankings:
        try:
            prob = float(rank.split(": ")[1].replace("%", "")) / 100.0
        except (IndexError, ValueError):
            logging.error(f"Error processing ranking string: {rank}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob
    if not category_sums:
        return None, None
    total_sum = sum(category_sums.values())
    if total_sum > 0:
        normalized_category_sums = {cat: val / total_sum for cat, val in category_sums.items()}
    else:
        normalized_category_sums = category_sums.copy()
    main_category = max(normalized_category_sums, key=normalized_category_sums.get)
    sorted_normalized_values = sorted(normalized_category_sums.values(), reverse=True)
    top3_sum = sum(sorted_normalized_values[:3])
    return main_category, top3_sum



def lighten_color(hex_color, amount=0.5):
    """
    Returns a lighter shade of the given hex color.
    
    Parameters:
    - hex_color (str): Base color in hex format (e.g., "#1f78b4").
    - amount (float): Fraction to lighten the color (0 to 1).
    
    Returns:
    - str: Lightened hex color.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02x}{g:02x}{b:02x}'
def plot_umap_3d_combined(
    X_original: np.ndarray,
    X_synthetic: np.ndarray,
    var_assoc_original: list,
    var_assoc_synthetic: list,
    output_path: str = None
) -> go.Figure:
    """
    Generates a 3D UMAP plot combining original (pre-oversampling) and synthetic (post-oversampling) samples.
    Uses a single fit_transform on the combined data to ensure ambos os conjuntos compartilham a mesma projeÃ§Ã£o.
    """
    n_orig = X_original.shape[0]
    n_syn = X_synthetic.shape[0]

    # Combina os dados e aplica um Ãºnico fit_transform
    X_combined = np.vstack([X_original, X_synthetic])
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.05)
    X_umap_combined = umap_reducer.fit_transform(X_combined)

    # Separa de volta em originais e sintÃ©ticos
    X_umap_orig = X_umap_combined[:n_orig]
    X_umap_syn  = X_umap_combined[n_orig:]

    # Cria dataframes
    df_orig = pd.DataFrame({
        'UMAP1': X_umap_orig[:, 0],
        'UMAP2': X_umap_orig[:, 1],
        'UMAP3': X_umap_orig[:, 2],
        'Associated Variable': var_assoc_original,
        'Group': ['Original'] * n_orig
    })
    df_syn = pd.DataFrame({
        'UMAP1': X_umap_syn[:, 0],
        'UMAP2': X_umap_syn[:, 1],
        'UMAP3': X_umap_syn[:, 2],
        'Associated Variable': var_assoc_synthetic,
        'Group': ['Synthetic'] * n_syn
    })
    df = pd.concat([df_orig, df_syn], ignore_index=True)

    # Cores
    unique_vars = sorted(set(var_assoc_original).union(var_assoc_synthetic))
    palette = px.colors.qualitative.Plotly
    base_colors = {var: palette[i % len(palette)] for i, var in enumerate(unique_vars)}
    df['Color'] = df.apply(
        lambda r: base_colors.get(r['Associated Variable'], '#808080')
                  if r['Group']=='Original'
                  else lighten_color(base_colors.get(r['Associated Variable'],'#808080'), 0.5),
        axis=1
    )

    # Monta figura
    fig = go.Figure()
    for var in unique_vars:
        for grp, opac in [('Original', 0.8), ('Synthetic', 0.5)]:
            sub = df[(df['Group']==grp)&(df['Associated Variable']==var)]
            if not sub.empty:
                fig.add_trace(go.Scatter3d(
                    x=sub['UMAP1'], y=sub['UMAP2'], z=sub['UMAP3'],
                    mode='markers',
                    marker=dict(size=6, color=sub['Color'], opacity=opac),
                    name=f'{grp} {var} ({len(sub)})',
                    text=sub.apply(lambda r: f"Group: {r['Group']}<br>Var: {r['Associated Variable']}", axis=1),
                    hoverinfo='text'
                ))

    fig.update_layout(
        title=f"3D UMAP (Original: {n_orig}, Synthetic: {n_syn})",
        scene=dict(
            xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"
        ),
        template="plotly_dark"
    )

    if output_path:
        fig.write_html(output_path, auto_open=False)
        logging.info(f"3D UMAP plot salvo em {output_path}")

    return fig
def plot_oversampling_quality(
    X_original: np.ndarray,
    X_synthetic: np.ndarray,
    var_assoc_original: list,
    var_assoc_synthetic: list,
    output_path: str
) -> None:
    """
    Generates quality-assessment plots by class:
      1) Histogram of cosine similarities of each synthetic sample to its closest original, split by class.
      2) Silhouette boxplot per class (Original vs. Synthetic).
    Saves PNG and SVG for each.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # ---- 1) Cosine Similarity Histograms Per Class ----
    sim_matrix = cosine_similarity(X_synthetic, X_original)
    max_similarities = sim_matrix.max(axis=1)
    classes = sorted(set(var_assoc_synthetic))
    n_classes = len(classes)
    fig1, axes = plt.subplots(n_classes, 1, figsize=(8, 4 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        idx = [i for i, v in enumerate(var_assoc_synthetic) if v == cls]
        values = max_similarities[idx]
        ax.hist(values, bins=30, edgecolor='black')
        ax.set_title(f"Histogram of Cosine Similarities for class {cls}")
        ax.set_xlabel("Max Cosine Similarity")
        ax.set_ylabel("Count")
    fig1.tight_layout()

    # Save histograms
    hist_png = os.path.join(output_path, "cosine_similarity_histogram_by_class.png")
    hist_svg = os.path.join(output_path, "cosine_similarity_histogram_by_class.svg")
    fig1.savefig(hist_png, dpi=600, bbox_inches='tight')
    fig1.savefig(hist_svg, dpi=600, bbox_inches='tight')
    plt.close(fig1)

    # ---- 2) Silhouette Boxplot per Class ----
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.array(var_assoc_original + var_assoc_synthetic)
    source_flags = np.array(['Original'] * len(X_original) + ['Synthetic'] * len(X_synthetic))
    sil_vals = silhouette_samples(X_combined, y_combined)

    df_sil = pd.DataFrame({
        'silhouette': sil_vals,
        'class': y_combined,
        'source': source_flags
    })
    unique_classes = sorted(df_sil['class'].unique())
    fig2, axes2 = plt.subplots(1, len(unique_classes), figsize=(4 * len(unique_classes), 6), sharey=True)
    if len(unique_classes) == 1:
        axes2 = [axes2]
    for ax, cls in zip(axes2, unique_classes):
        subset = df_sil[df_sil['class'] == cls]
        data = [
            subset[subset['source']=='Original']['silhouette'],
            subset[subset['source']=='Synthetic']['silhouette']
        ]
        ax.boxplot(data, labels=['Orig', 'Synth'], patch_artist=True,
                   boxprops=dict(facecolor='lightgray'), medianprops=dict(color='orange'))
        ax.set_title(f'Class {cls}')
        ax.tick_params(axis='x', labelrotation=90)
        if ax is axes2[0]:
            ax.set_ylabel('Silhouette Coefficient')
    fig2.suptitle('Silhouette Coefficients per Class: Original vs. Synthetic')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    box_png = os.path.join(output_path, "silhouette_per_class_boxplot.png")
    box_svg = os.path.join(output_path, "silhouette_per_class_boxplot.svg")
    fig2.savefig(box_png, dpi=600, bbox_inches='tight')
    fig2.savefig(box_svg, dpi=600, bbox_inches='tight')
    plt.close(fig2)


def plot_confusion_and_calibration(
    y_true, 
    y_pred, 
    y_pred_proba, 
    classes, 
    model_name_prefix, 
    output_dir, 
    normalize_cm=True, 
    n_bins=10
):
    """
    Save side-by-side Confusion Matrix and Calibration Curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(
        y_true, 
        y_pred, 
        labels=classes, 
        normalize='true' if normalize_cm else None
    )
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title(f'{model_name_prefix}: Confusion Matrix' + (' (norm.)' if normalize_cm else ''))
    ax1.set_xlabel('Predicted', rotation=0, labelpad=15)
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_ylabel('True')

    # ---- Calibration Curve ----
    if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
        prob_pos = y_pred_proba.ravel()
        frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins)
        ax2.plot(mean_pred, frac_pos, 's-', label='Model')
    else:
        for i, cls in enumerate(classes):
            prob_pos = y_pred_proba[:, i]
            frac_pos, mean_pred = calibration_curve((y_true==cls).astype(int), prob_pos, n_bins=n_bins)
            ax2.plot(mean_pred, frac_pos, 's-', label=str(cls))
    ax2.plot([0,1],[0,1],'k--', label='Perfectly calibrated')
    ax2.set_title(f'{model_name_prefix}: Calibration Curve')
    ax2.set_xlabel('Mean predicted prob.')
    ax2.set_ylabel('Fraction positive')
    ax2.tick_params(axis='x')
    ax2.legend(loc='best')
    # Save
    cm_path = os.path.join(output_dir, f'{model_name_prefix}_confusion_matrix.png')
    cal_path = os.path.join(output_dir, f'{model_name_prefix}_calibration_curve.png')
    fig.savefig(os.path.join(output_dir, f'{model_name_prefix}_cm_calibration.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return cm_path, cal_path
    
def adjust_predictions_global(
    predicted_proba: np.ndarray, 
    method: str = 'normalize', 
    alpha: float = 1.0
) -> np.ndarray:
    """
    Adjusts the predicted probabilities from the model.
    
    Parameters:
    - predicted_proba (np.ndarray): Predicted probabilities from the model.
    - method (str): Adjustment method ('normalize', 'smoothing', 'none').
    - alpha (float): Parameter for smoothing (used if method='smoothing').
    
    Returns:
    - np.ndarray: Adjusted probabilities.
    """
    if method == 'normalize':
        # Normalize probabilities so that they sum to 1 for each sample
        logging.info("Normalizing predicted probabilities.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)

    elif method == 'smoothing':
        # Apply smoothing to probabilities to avoid extreme values
        logging.info(f"Applying smoothing to predicted probabilities with alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])

    elif method == 'none':
        # Do not apply any adjustment
        logging.info("No adjustment applied to predicted probabilities.")
        adjusted_proba = predicted_proba.copy()

    else:
        logging.warning(f"Unknown adjustment method '{method}'. No adjustment will be applied.")
        adjusted_proba = predicted_proba.copy()

    return adjusted_proba


class Support:
    """
    Support class for training and evaluating Random Forest models with oversampling techniques.
    """

    def __init__(self, cv: int = 5, seed: int = SEED, n_jobs: int = 8):
        self.cv = cv
        self.model = None
        self.seed = seed
        self.n_jobs = n_jobs
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []
        self.roc_results = []
        self.train_sizes = np.linspace(.1, 1.0, 5)
        self.standard = StandardScaler()

        self.best_params = {}

        self.init_params = {
            "n_estimators": 100,  # Keeping a moderate number
            "max_depth": 2,  # Reducing depth to prevent overfitting
            "min_samples_split": 2,  # Increasing to prevent overfitting
            "min_samples_leaf": 2,  # Keeping high to avoid overfitting
            "criterion": "entropy",  # Keeping as is
            "max_features": "sqrt",  # Maintaining a good default choice
            "class_weight": "balanced_subsample",  # Testing without class balancing
            "max_leaf_nodes": 5,  # Reducing leaf nodes to prevent overfitting
            "min_impurity_decrease": 0.01,  # Increasing value to prevent overfitting
            "bootstrap": True,  # Testing with bootstrap
            "ccp_alpha": 0.005,  # Including tree pruning            
       }

        self.parameters = {
            "n_estimators": [250,300],
            "max_depth": [10, 20],
            "min_samples_split": [4,6],
            "min_samples_leaf": [4, 6],
            "criterion": ["gini", "entropy"],
            "max_features": ["log2"],
            "class_weight": ["balanced",  None],
            "max_leaf_nodes": [10, 20, None],
            "min_impurity_decrease": [0.0],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0],
        }

    def _oversample_single_sample_classes(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        protein_ids: list = None, 
        var_assoc: list = None
    ) -> tuple:
        """
        Customizes oversampling to ensure all classes have at least 'self.cv + 1' samples.
        Also captures synthetic data information such as protein IDs and associated variables.
    
        Parameters:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        - protein_ids (list): List of original protein IDs.
        - var_assoc (list): List of original associated variables.
    
        Returns:
        - tuple: (X after oversampling, y after oversampling, synthetic protein IDs, synthetic associated variables)
        """
        logging.info("Starting oversampling process...")

        # Count initial class distribution
        counter = Counter(y)
        
        logging.info(f"Initial class distribution: {counter}")

        # Define oversampling strategy to ensure at least 'self.cv + 1' samples per class
        classes_to_oversample = {cls: max(self.cv + 1, count) for cls, count in counter.items()}
        logging.info(f"Oversampling strategy for RandomOverSampler: {classes_to_oversample}")

        try:
            # Apply RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=classes_to_oversample, random_state=self.seed)
            X_ros, y_ros = ros.fit_resample(X, y)
            logging.info(f"Class distribution after RandomOverSampler: {Counter(y_ros)}")
        except ValueError as e:
            logging.error(f"Error during RandomOverSampler: {e}")
            sys.exit(1)

        # Capture synthetic data from RandomOverSampler
        synthetic_protein_ids = []
        synthetic_var_assoc = []
        if protein_ids and var_assoc:
            for idx in range(len(X), len(X_ros)):
                synthetic_protein_ids.append(f"synthetic_ros_{idx}")
                synthetic_var_assoc.append(var_assoc[idx % len(var_assoc)])

        try:
            # Apply SMOTE for further class balancing
            smote = SMOTE(random_state=self.seed)
            X_smote, y_smote = smote.fit_resample(X_ros, y_ros)
            logging.info(f"Class distribution after SMOTE: {Counter(y_smote)}")
        except ValueError as e:
            logging.error(f"Error during SMOTE: {e}")
            sys.exit(1)

        # Capture synthetic data from SMOTE
        if protein_ids and var_assoc:
            for idx in range(len(X_ros), len(X_smote)):
                synthetic_protein_ids.append(f"synthetic_smote_{idx}")
                synthetic_var_assoc.append(var_assoc[idx % len(var_assoc)])

        # Save class counts to a file
        with open("oversampling_counts.txt", "a") as f:
            f.write("Class Distribution after Oversampling:\n")
            for cls, count in Counter(y_smote).items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote, synthetic_protein_ids, synthetic_var_assoc


    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        protein_ids: list = None, 
        var_assoc: list = None, 
        model_name_prefix: str = 'model', 
        model_dir: str = None, 
        min_kmers: int = None
    ) -> RandomForestClassifier:
        """
        Trains the model with oversampling and cross-validation.
        
        Parameters:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        - protein_ids (list): Protein IDs.
        - var_assoc (list): Associated variables.
        - model_name_prefix (str): Prefix for saving the model.
        - model_dir (str): Directory to save the model.
        - min_kmers (int): Minimum number of k-mers.
        
        Returns:
        - RandomForestClassifier: Trained and calibrated model.
        """
        logging.info(f"Starting fit method for {model_name_prefix}...")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Determine min_kmers
        if min_kmers is not None:
            logging.info(f"Using provided min_kmers: {min_kmers}")
        else:
            min_kmers = len(X)
            logging.info(f"min_kmers not provided. Setting to size of X: {min_kmers}")
            
            
        logging.info(f"oversampling...")    

        # Initial oversampling
        X_smote, y_smote, synthetic_protein_ids, synthetic_var_assoc = self._oversample_single_sample_classes(
            X, y, protein_ids, var_assoc
        )

		

        # Combine original and synthetic data
        if protein_ids and var_assoc:
            combined_protein_ids = protein_ids + synthetic_protein_ids
            combined_var_assoc = var_assoc + synthetic_var_assoc
        else:
            combined_protein_ids = None
            combined_var_assoc = None

        # Visualization of latent space after oversampling
        if protein_ids and var_assoc:
            save_path = f"umap_similarity_visualization_{model_name_prefix}.html"
            logging.info("Visualizing latent space with similarity measures...")
            fig = visualize_latent_space_with_similarity(
                X_original=X, 
                X_synthetic=X_smote[len(y):],  # Adjust for y
                y_original=y, 
                y_synthetic=y_smote[len(y):],
                protein_ids_original=protein_ids,
                protein_ids_synthetic=synthetic_protein_ids,
                var_assoc_original=var_assoc,
                var_assoc_synthetic=synthetic_var_assoc,
                output_dir=model_dir
            )
            logging.info("UMAP visualization generated successfully.")

        X_original = X
        X_synthetic = X_smote[len(X):]  # Assumindo que os dados sintÃƒÂ©ticos vÃƒÂªm apÃƒÂ³s os originais
        var_assoc_original = var_assoc  # Lista original das variÃƒÂ¡veis associadas
        var_assoc_synthetic = synthetic_var_assoc  # Lista das variÃƒÂ¡veis associadas para os sintÃƒÂ©ticos


        fig_umap_3d = plot_umap_3d_combined(
            X_original, X_synthetic, var_assoc_original, var_assoc_synthetic,
            output_path=os.path.join(model_dir, "umap_3d_pre_cv.html")
        )
        
        # supondo que vocÃª tenha X_orig, X_syn, var_orig, var_syn
        figures_dir = os.path.join(model_dir, "quality_figures")	
# DiretÃ³rio onde os grÃ¡ficos de qualidade serÃ£o salvos
# Chamada da funÃ§Ã£o revisada:
        plot_oversampling_quality(
            X_original=X_original,
            X_synthetic=X_synthetic,
            var_assoc_original=var_assoc_original,
            var_assoc_synthetic=var_assoc_synthetic,
            output_path=figures_dir
        )


        # Cross-validation with StratifiedKFold
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []

        # Check if all classes have at least self.cv +1 samples
        class_counts = Counter(y_smote)
        min_class_count = min(class_counts.values())
        adjusted_n_splits = min(self.cv, min_class_count - 1)  # Because SMOTE requires n_samples > n_neighbors
        if adjusted_n_splits < self.cv:
            logging.warning(f"Adjusting n_splits from {self.cv} to {adjusted_n_splits} due to class size constraints.")
            skf = StratifiedKFold(n_splits=adjusted_n_splits, random_state=self.seed, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        for fold_number, (train_index, test_index) in enumerate(skf.split(X_smote, y_smote), start=1):
            X_train, X_test = X_smote[train_index], X_smote[test_index]
            y_train, y_test = y_smote[train_index], y_smote[test_index]

            # Check class distribution in test set
            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test set class distribution: {fold_class_distribution}")

            # Oversampling for training
            X_train_resampled, y_train_resampled, synthetic_protein_ids_train, synthetic_var_assoc_train = self._oversample_single_sample_classes(
                X_train, 
                y_train, 
                protein_ids=protein_ids if protein_ids else None, 
                var_assoc=var_assoc if var_assoc else None
            )

            # Check class distribution after oversampling in training set
            train_sample_counts = Counter(y_train_resampled)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Training set class distribution after oversampling: {train_sample_counts}")

            with open("training_sample_counts_after_oversampling.txt", "a") as f:
                f.write(f"Fold {fold_number} Training sample counts after oversampling for {model_name_prefix}:\n")
                for cls, count in train_sample_counts.items():
                    f.write(f"{cls}: {count}\n")

            # Model Training
            self.model = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs)
            self.model.fit(X_train_resampled, y_train_resampled)

            # Evaluation
            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)
            y_pred = self.model.predict(X_test)

            # Metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

            # Precision-Recall AUC
            if len(np.unique(y_test)) > 1:
                pr_auc = average_precision_score(y_test, self.model.predict_proba(X_test), average='macro')
            else:
                pr_auc = 0.0  # Cannot calculate PR AUC for a single class
            self.pr_auc_scores.append(pr_auc)

            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Training Score: {train_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test Score: {test_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: F1 Score: {f1}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Precision-Recall AUC: {pr_auc}")

            # Calculate ROC AUC
            try:
                if len(np.unique(y_test)) == 2:
                    fpr, tpr, thresholds = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
                    roc_auc_score_value = auc(fpr, tpr)
                    self.roc_results.append((fpr, tpr, roc_auc_score_value))
                else:
                    y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                    roc_auc_score_value = roc_auc_score(y_test_bin, self.model.predict_proba(X_test), multi_class='ovo', average='macro')
                    self.roc_results.append(roc_auc_score_value)
            except ValueError:
                logging.warning(f"Unable to calculate ROC AUC for fold {fold_number} [{model_name_prefix}] due to insufficient class representation.")

            # Perform Grid Search and save the best model
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            self.best_params = best_params

            if model_dir:
                best_model_filename = os.path.join(model_dir, f'model_best_{model_name_prefix}.pkl')
                # Ensure the directory exists
                os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved as {best_model_filename} for {model_name_prefix}")
            else:
                best_model_filename = f'model_best_{model_name_prefix}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved as {best_model_filename} for {model_name_prefix}")

            if best_params is not None:
                self.best_params = best_params
                logging.info(f"Best parameters for {model_name_prefix}: {self.best_params}")
            else:
                logging.warning(f"No best parameters found in grid search for {model_name_prefix}.")

            # Integrate Probability Calibration
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=5, n_jobs=self.n_jobs)
            calibrator.fit(X_train_resampled, y_train_resampled)

            self.model = calibrator

            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'calibrated_model_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Calibrated model saved as {calibrated_model_filename} for {model_name_prefix}")

        return self.model

    def _perform_grid_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray) -> tuple:
        """
        Performs Grid Search to find the best hyperparameters.
        
        Parameters:
        - X_train_resampled (np.ndarray): Training features after oversampling.
        - y_train_resampled (np.ndarray): Training labels after oversampling.
        
        Returns:
        - tuple: (Best model, Best parameters)
        """
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.seed),
            self.parameters,
            cv=skf,
            n_jobs=self.n_jobs,
            scoring='roc_auc_ovo',
            verbose=1
        )

        grid_search.fit(X_train_resampled, y_train_resampled)
        logging.info(f"Best parameters from grid search: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def get_best_param(self, param_name: str, default = None):
        """
        Retrieves the best parameter found in grid search.
        
        Parameters:
        - param_name (str): Parameter name.
        - default: Default value if parameter is not found.
        
        Returns:
        - Value of the parameter or default.
        """
        return self.best_params.get(param_name, default)

    def plot_learning_curve(self, output_path: str) -> None:
        """
        Plots the learning curve of the model.
        
        Parameters:
        - output_path (str): Path to save the plot.
        """
        plt.figure()
        plt.figure(facecolor='white')
        plt.plot(self.train_scores, label='Training Score')
        plt.plot(self.test_scores, label='Cross-Validation Score')
        plt.plot(self.f1_scores, label='F1 Score')
        plt.plot(self.pr_auc_scores, label='Precision-Recall AUC')
        plt.title("Learning Curve", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='white')  # Match background color
        plt.close()

    def get_class_rankings(self, X: np.ndarray) -> list:
        """
        Obtains class rankings for the provided data.
        
        Parameters:
        - X (np.ndarray): Data to obtain predictions.
        
        Returns:
        - list: List of formatted class rankings for each sample.
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Obtain probabilities for each class
        y_pred_proba = self.model.predict_proba(X)

        # Ranking classes based on probabilities
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings
############################################# Save the model in the final test
    def test_best_RF(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        scaler_dir: str = '.'
    ) -> tuple:
        """
        Tests the best Random Forest model with the provided data.
        
        Parameters:
        - X (np.ndarray): Features for testing.
        - y (np.ndarray): True labels for testing.
        - scaler_dir (str): Directory where the scaler is saved.
        
        Returns:
        - tuple: (Score, F1 Score, Precision-Recall AUC, Best parameters, Calibrated model, X_test, y_test)
        """
        # Ajuste aqui para sempre usar scaler_dir como base
#        scaler_path = os.path.join(scaler_dir, 'scaler_associated.pkl')
        scaler_path = os.path.join(model_dir, 'scaler_associated.pkl')
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found at {scaler_path}. It should be scaler_associated.pkl.")
            sys.exit(1)

        X_scaled = scaler.transform(X)

        # Apply oversampling on the entire set before splitting
      ###  X_resampled, y_resampled, _, _ = self._oversample_single_sample_classes(X_scaled, y)
      # Apply oversampling on the entire set before splitting
        X_resampled, y_resampled, synthetic_protein_ids, synthetic_var_assoc = self._oversample_single_sample_classes(X_scaled, y)      
                      
    # Log the total number of samples after oversampling
        total_samples = len(y_resampled)
        logging.info(f"Total number of samples after oversampling: {total_samples}")

    # Visualize the latent space used for training:
    # Assumes that the first len(y) samples in X_resampled correspond to the original data,
    # and the rest are synthetic.
    
        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        logging.info(f"Number of training samples: {len(y_train)}")
        logging.info(f"Number of test samples: {len(y_test)}")
    
        # Train RandomForestClassifier with the best parameters
        model = RandomForestClassifier(
            n_estimators=self.best_params.get('n_estimators', 100),
            max_depth=self.best_params.get('max_depth', 10),
            min_samples_split=self.best_params.get('min_samples_split', 2),
            min_samples_leaf=self.best_params.get('min_samples_leaf', 4),
            criterion=self.best_params.get('criterion', 'gini'),
            max_features=self.best_params.get('max_features', 'log2'),
            class_weight=self.best_params.get('class_weight', 'balanced'),
            max_leaf_nodes=self.best_params.get('max_leaf_nodes', 20),
            min_impurity_decrease=self.best_params.get('min_impurity_decrease', 0.00),
            bootstrap=self.best_params.get('bootstrap', True),
            ccp_alpha=self.best_params.get('ccp_alpha', 0.0),
            random_state=self.seed,
            n_jobs=self.n_jobs
        )
        model.fit(X_train, y_train)  # Fit the model on training data

        # Integrate Probability Calibration in Test Model
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        # Make predictions
        y_pred = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred, method='normalize')

          # Calculate global score (e.g., AUC)	
        score = self._calculate_score(y_pred_adjusted, y_test)

         # Calculate global score (e.g., AUC)
        y_pred_classes = calibrated_model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
                    
    # Calculate per-class F1 scores:
        unique_classes = np.unique(y_test)
        f1_per_class = f1_score(y_test, y_pred_classes, average=None, labels=unique_classes)
        f1_class_dict = dict(zip(unique_classes, f1_per_class))
    
        logging.info(f"Global F1 Score (weighted): {f1}")

        logging.info(f"F1 Scores per class: {f1_class_dict}")

        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(y_test, y_pred_adjusted, average='macro')
        else:
            pr_auc = 0.0  # Cannot calculate PR AUC for a single class                
# Cria o grÃƒÂ¡fico de barras
# Cria o grÃƒÂ¡fico de barras
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
# Define as cores da paleta viridis para cada classe
        unique_classes = list(f1_class_dict.keys())
        n_classes = len(unique_classes)
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

# Gera as barras com as cores correspondentes
# Desenha as barras e coloca o valor dentro delas, sem rotaÃ§Ã£o
        for i, cls in enumerate(unique_classes):
            score_val = f1_class_dict[cls]
            ax_bar.bar(i, score_val, color=viridis_colors[i])
            ax_bar.text(
                i, 
                score_val / 2, 
                f"{score_val:.2f}", 
                ha='center', 
                va='center', 
                fontsize=12, 
                rotation=0  # texto horizontal
            )

# RÃ³tulos do eixo X na vertical
        ax_bar.set_xticks(range(n_classes))
        ax_bar.set_xticklabels(unique_classes, rotation=90, fontsize=12)

# TÃ­tulo mais acima
        ax_bar.set_title('F1 Score per Class', fontsize=16, pad=20)

        ax_bar.set_xlabel('Associated Variable (Class)', fontsize=14)
        ax_bar.set_ylabel('F1 Score', fontsize=14)
        ax_bar.set_ylim(0, 1)

        plt.tight_layout()

# Define os caminhos para salvar os arquivos em PNG e SVG no diretÃƒÂ³rio model_dir
        png_path = os.path.join(model_dir, "f1_per_class.png")
        svg_path = os.path.join(model_dir, "f1_per_class.svg")

# Salva a figura nos dois formatos
        fig_bar.savefig(png_path, dpi=300, bbox_inches='tight')
        fig_bar.savefig(svg_path, format='svg', bbox_inches='tight')
        plt.close(fig_bar)

#############
    # --- Generate Violin Plots for Predicted Probabilities (True Class) in Train and Test Sets ---
    # For each sample in training set, get the predicted probability of its true class.
        train_pred_proba = calibrated_model.predict_proba(X_train)
        test_pred_proba = calibrated_model.predict_proba(X_test)
        true_probs_train = []
        for i, true_label in enumerate(y_train):
            idx = list(calibrated_model.classes_).index(true_label)
            true_probs_train.append(train_pred_proba[i][idx])
        true_probs_test = []
        for i, true_label in enumerate(y_test):
            idx = list(calibrated_model.classes_).index(true_label)
            true_probs_test.append(test_pred_proba[i][idx])
        
        import seaborn as sns
    # Cria DataFrames para training e test
        df_train = pd.DataFrame({
            'Probability': true_probs_train,
            'Dataset': ['Train'] * len(true_probs_train),
            'Associated Variable': y_train  # assume that y_train contains the associated variable labels
        })
        df_test = pd.DataFrame({
            'Probability': true_probs_test,
            'Dataset': ['Test'] * len(true_probs_test),
            'Associated Variable': y_test
        })
        df_violin = pd.concat([df_train, df_test], ignore_index=True)
    
        fig_violin, ax_violin = plt.subplots(figsize=(12, 6))
        sns.violinplot(x='Associated Variable', y='Probability', hue='Dataset', data=df_violin,
                       split=False,inner='quartile', scale='width', bw=0.2, palette='viridis', ax=ax_violin)
                       
        ax_violin.tick_params(axis='x', labelrotation=90)
                       
        ax_violin.set_title('Distribution of Predicted Probabilities for True Class', fontsize=16)
        ax_violin.set_xlabel('Associated Variable', fontsize=14)
        ax_violin.set_ylabel('Predicted Probability', fontsize=14)
        plt.tight_layout()
        violin_png_path = os.path.join(model_dir, "true_probability_violin.png")
        violin_svg_path = os.path.join(model_dir, "true_probability_violin.svg")
        fig_violin.savefig(violin_png_path, dpi=300, bbox_inches='tight')
        fig_violin.savefig(violin_svg_path, format='svg', bbox_inches='tight')
        plt.close(fig_violin)
                
         # --- save confusion & calibration plots ---
        classes = list(calibrated_model.classes_)
        cm_file, cal_file = plot_confusion_and_calibration(
            y_true=y_test,
            y_pred=y_pred_classes,
            y_pred_proba=y_pred,        # raw predict_proba output
            classes=classes,
            model_name_prefix='test_best_RF',
            output_dir=model_dir,
            normalize_cm=True,
            n_bins=10
        )
        logging.info(f"Saved confusion matrix at {cm_file}")
        logging.info(f"Saved calibration curve at {cal_file}")

        # Return the score, best parameters, trained model, and test sets
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculates the score (e.g., ROC AUC) based on predictions and true labels.
        
        Parameters:
        - y_pred (np.ndarray): Adjusted predicted probabilities.
        - y_test (np.ndarray): True labels.
        
        Returns:
        - float: Score value.
        """
        n_classes = len(np.unique(y_test))
        if y_pred.ndim == 1 or n_classes == 2:
            return roc_auc_score(y_test, y_pred)
        elif y_pred.ndim == 2 and n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            return roc_auc_score(y_test_bin, y_pred, multi_class='ovo', average='macro')
        else:
            logging.warning(f"Unexpected shape or number of classes: y_pred shape: {y_pred.shape}, number of classes: {n_classes}")
            return 0


class ProteinEmbeddingGenerator:
    """
    Class to generate protein embeddings using Word2Vec.
    """

    def __init__(self, sequences_path: str, table_data: pd.DataFrame = None, aggregation_method: str = 'none'):
        """
        Initializes the embedding generator.
        
        Parameters:
        - sequences_path (str): Path to the sequences file.
        - table_data (pd.DataFrame): Associated table data.
        - aggregation_method (str): Aggregation method ('none' or 'mean').
        """
        aligned_path = sequences_path
        if not are_sequences_aligned(sequences_path):
            realign_sequences_with_mafft(sequences_path, sequences_path.replace(".fasta", "_aligned.fasta"), threads=1)
            aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")
        else:
            logging.info(f"Sequences are already aligned: {sequences_path}")

        self.alignment = AlignIO.read(aligned_path, 'fasta')
        self.table_data = table_data
        self.embeddings = []
        self.models = {}
        self.aggregation_method = aggregation_method  # Aggregation method: 'none' or 'mean'
        self.min_kmers = None  # To store min_kmers

    def generate_embeddings(
        self, 
        k: int = 3, 
        step_size: int = 1, 
        word2vec_model_path: str = "word2vec_model.bin", 
        model_dir: str = None, 
        min_kmers: int = None, 
        save_min_kmers: bool = False,
        window: int = 5,
        workers: int = 48,
        epochs: int = 2500
    ) -> None:
        """
        Generates embeddings for protein sequences using Word2Vec, standardizing the number of k-mers.
        
        Parameters:
        - k (int): Size of the k-mer.
        - step_size (int): Step size for generating k-mers.
        - word2vec_model_path (str): Filename for the Word2Vec model.
        - model_dir (str): Directory to save the Word2Vec model.
        - min_kmers (int): Minimum number of k-mers to use.
        - save_min_kmers (bool): If True, saves min_kmers to a file.
        - window (int): Word2Vec window size.
        - workers (int): Number of worker threads for Word2Vec.
        - epochs (int): Number of training epochs for Word2Vec.
        """
        # Define the full path for the Word2Vec model
        if model_dir:
            word2vec_model_full_path = os.path.join(model_dir, word2vec_model_path)
        else:
            word2vec_model_full_path = word2vec_model_path

        # Check if the Word2Vec model already exists
        if os.path.exists(word2vec_model_full_path):
            logging.info(f"Word2Vec model found at {word2vec_model_full_path}. Loading the model.")
            model = Word2Vec.load(word2vec_model_full_path)
            self.models['global'] = model
        else:
            logging.info("Word2Vec model not found. Training a new model.")
            # Initialize variables
            kmer_groups = {}
            all_kmers = []
            kmers_counts = []

            # Generate k-mers
            for record in self.alignment:
                sequence = str(record.seq)
                seq_len = len(sequence)
                protein_accession_alignment = record.id.split()[0]

                # If table data is not provided, skip matching
                if self.table_data is not None:
                    matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                    matching_info = self.table_data[matching_rows]

                    if matching_info.empty:
                        logging.warning(f"No matching table data for {protein_accession_alignment}")
                        continue  # Skip to next iteration

                    target_variable = matching_info['Target variable'].values[0]  # Can be removed if not necessary
                    associated_variable = matching_info['Associated variable'].values[0]
                else:
                    # If no table, use default values or None
                    target_variable = None  # Can be removed
                    associated_variable = None

                logging.info(f"Processing {protein_accession_alignment} with sequence length {seq_len}")

                if seq_len < k:
                    logging.warning(f"Sequence too short for {protein_accession_alignment}. Length: {seq_len}")
                    continue

                # Generate k-mers, allowing k-mers with fewer que k gaps
                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Allow k-mers with fewer than k gaps

                if not kmers:
                    logging.warning(f"No valid k-mers for {protein_accession_alignment}")
                    continue

                all_kmers.append(kmers)  # Add the list of k-mers as a sentence
                kmers_counts.append(len(kmers))  # Store the count of k-mers

                embedding_info = {
                    'protein_accession': protein_accession_alignment,
                    'target_variable': target_variable,
                    'associated_variable': associated_variable,
                    'kmers': kmers  # Store k-mers for later use
                }
                kmer_groups[protein_accession_alignment] = embedding_info

            # Determine the minimum number of k-mers
            if not kmers_counts:
                logging.error("No k-mers were collected. Please check your sequences and k-mer parameters.")
                sys.exit(1)

            if min_kmers is not None:
                self.min_kmers = min_kmers
                logging.info(f"Using provided min_kmers: {self.min_kmers}")
            else:
                self.min_kmers = min(kmers_counts)
                logging.info(f"Minimum number of k-mers in any sequence: {self.min_kmers}")

            # Save min_kmers if necessary
            if save_min_kmers and model_dir:
                min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
                with open(min_kmers_path, 'w') as f:
                    f.write(str(self.min_kmers))
                logging.info(f"min_kmers saved at {min_kmers_path}")

            # Train the Word2Vec model using all k-mers
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=390,  # Alojado no script
                window=window,
                min_count=1,
                workers=workers,
                sg=1,
                hs=1,  # Hierarchical softmax enabled
                negative=0,  # Negative sampling disabled
                epochs=epochs,
                seed=SEED  # Fixed seed for reproducibility
            )

            # Create directory for the Word2Vec model if necessary
            if model_dir:
                os.makedirs(os.path.dirname(word2vec_model_full_path), exist_ok=True)

            # Save the Word2Vec model
            model.save(word2vec_model_full_path)
            self.models['global'] = model
            logging.info(f"Word2Vec model saved at {word2vec_model_full_path}")

        # Generate standardized embeddings
        kmer_groups = {}
        kmers_counts = []
        all_kmers = []

        for record in self.alignment:
            sequence_id = record.id.split()[0]  # Use consistent sequence IDs
            sequence = str(record.seq)

            # If table data is not provided, skip matching
            if self.table_data is not None:
                matching_rows = self.table_data['Protein.accession'].str.split().str[0] == sequence_id
                matching_info = self.table_data[matching_rows]

                if matching_info.empty:
                    logging.warning(f"No matching table data for {sequence_id}")
                    continue  # Skip to next iteration

                target_variable = matching_info['Target variable'].values[0]  # Can be removed
                associated_variable = matching_info['Associated variable'].values[0]
            else:
                # If no table, use default values or None
                target_variable = None  # Can be removed
                associated_variable = None

            kmers = [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, step_size)]
            kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Allow k-mers with fewer than k gaps

            if not kmers:
                logging.warning(f"No valid k-mers for {sequence_id}")
                continue

            all_kmers.append(kmers)
            kmers_counts.append(len(kmers))

            embedding_info = {
                'protein_accession': sequence_id,
                'target_variable': target_variable,
                'associated_variable': associated_variable,
                'kmers': kmers
            }
            kmer_groups[sequence_id] = embedding_info

        # Determine the minimum number of k-mers
        if not kmers_counts:
            logging.error("No k-mers were collected. Please check your sequences and k-mer parameters.")
            sys.exit(1)

        if min_kmers is not None:
            self.min_kmers = min_kmers
            logging.info(f"Using provided min_kmers: {self.min_kmers}")
        else:
            self.min_kmers = min(kmers_counts)
            logging.info(f"Minimum number of k-mers in any sequence: {self.min_kmers}")

        # Generate standardized embeddings
        for record in self.alignment:
            sequence_id = record.id.split()[0]  # Use consistent sequence IDs
            embedding_info = kmer_groups.get(sequence_id, {})
            kmers_for_protein = embedding_info.get('kmers', [])

            if len(kmers_for_protein) == 0:
                if self.aggregation_method == 'none':
                    embedding_concatenated = np.zeros(self.models['global'].vector_size * self.min_kmers)
                else:
                    embedding_concatenated = np.zeros(self.models['global'].vector_size)
                self.embeddings.append({
                    'protein_accession': sequence_id,
                    'embedding': embedding_concatenated,
                    'target_variable': embedding_info.get('target_variable'),
                    'associated_variable': embedding_info.get('associated_variable')
                })
                continue

            # Select the first min_kmers k-mers
            selected_kmers = kmers_for_protein[:self.min_kmers]

            # Pad with zeros if necessary
            if len(selected_kmers) < self.min_kmers:
                padding = [np.zeros(self.models['global'].vector_size)] * (self.min_kmers - len(selected_kmers))
                selected_kmers.extend(padding)

            # Obtain embeddings for selected k-mers
            selected_embeddings = [self.models['global'].wv[kmer] if kmer in self.models['global'].wv else np.zeros(self.models['global'].vector_size) for kmer in selected_kmers]

            if self.aggregation_method == 'none':
                # Concatenate embeddings of selected k-mers
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                # Aggregate embeddings of selected k-mers by mean
                embedding_concatenated = np.mean(selected_embeddings, axis=0)
            else:
                # If the method is unrecognized, use concatenation as default
                logging.warning(f"Unknown aggregation method '{self.aggregation_method}'. Using concatenation.")
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)

            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': embedding_concatenated,
                'target_variable': embedding_info.get('target_variable'),
                'associated_variable': embedding_info.get('associated_variable')
            })

            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {embedding_concatenated.shape}")

        # Fit StandardScaler with embeddings for training/prediction
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])

        # Check if all embeddings have the same shape
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Inconsistent embedding shapes detected: {embedding_shapes}")
            raise ValueError("Embeddings have inconsistent shapes.")
        else:
            logging.info(f"All embeddings have the shape: {embedding_shapes.pop()}")

        # Define the full path for the scaler
        scaler_full_path = os.path.join(model_dir, 'scaler_associated.pkl') if model_dir else 'scaler_associated.pkl'

        # Check if the scaler already exists
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler found at {scaler_full_path}. Loading the scaler.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler not found. Training a new scaler.")
            scaler = StandardScaler().fit(embeddings_array_train)
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler saved at {scaler_full_path}")

    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        """
        Returns embeddings and associated labels.
        
        Parameters:
        - label_type (str): Type of label ('associated_variable').
        
        Returns:
        - tuple: (Embeddings, Labels)
        """
        embeddings = []
        labels = []

        for embedding_info in self.embeddings:
            embeddings.append(embedding_info['embedding'])
            labels.append(embedding_info[label_type])  # Use the specified label type

        return np.array(embeddings), np.array(labels)


def plot_predictions_scatterplot_custom(
    results: dict, 
    output_path: str, 
    top_n: int = 1
) -> None:
    """
    Generates a scatter plot showing only the main category with the highest sum of probabilities for each protein.
    
    Y-Axis: Protein accession ID
    X-Axis: Specificities from C4 to C18 (fixed scale)
    Each point represents the corresponding specificity for the protein.
    Only the main category (top 1) is plotted per protein.
    Points are colored in a single uniform color, styled for scientific publication.
    
    Parameters:
    - results (dict): Dictionary containing predictions and rankings for proteins.
    - output_path (str): Path to save the scatter plot.
    - top_n (int): Number of top main categories to plot (default is 1).
    """
    # Prepare data
    protein_specificities = {}
    
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No ranking data associated with the protein. {seq_id}. Skipping...")
            continue

        # Use the function format_and_sum_probabilities to get the main category and normalized confidence
        main_category, normalized_confidence = format_and_sum_probabilities(associated_rankings)
        if main_category is None:
            logging.warning(f"No valid formatting for protein {seq_id}. Skipping...")
            continue

        protein_specificities[seq_id] = {
            'top_category': main_category,
            'confidence': normalized_confidence
        }

    if not protein_specificities:
        logging.warning("No data available to plot the scatter plot.")
        return

    # Sort protein IDs for better visualization
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))  # Adjust height based on number of proteins

    # Fixed scale for X-axis from C4 to C18 (only even numbers are shown)
    x_values = list(range(4, 19, 2))

    # Plot points for all proteins with their main category
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        confidence = data['confidence']

        # Extract specificities from the category string (e.g., "C4-C6-C8" -> [4,6,8])
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]

        for spec in specificities:
            ax.scatter(
                spec, y,
                color='black',  # Uniform color
                edgecolors='black',
                linewidth=0.5,
                s=100,
                label='_nolegend_'  # Avoid duplication in legend
            )

        # Connect points with lines if multiple specificities
        if len(specificities) > 1:
            ax.plot(
                specificities,
                [y] * len(specificities),
                color='black',
                linestyle='-',
                linewidth=1.0,
                alpha=0.7
            )

    # Customize the plot for better scientific publication quality
    ax.set_xlabel('Specificity (C4 to C18)', fontsize=14, fontweight='bold', color='black')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold', color='black')
    ax.set_title('Scatter Plot of Predictions for New Sequences (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='black')

    # Define fixed scale and formatting for the X-axis
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='black')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='black')

    # Define grid and remove unnecessary spines for a clean look
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Minor ticks on the X-axis for better visibility
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure in high quality for publication
    plt.savefig(output_path, facecolor='white', dpi=600, bbox_inches='tight')  # Match background color
    plt.close()
    logging.info(f"Scatter plot saved at {output_path}")

def plot_prediction_confidence_bar(df_results: pd.DataFrame) -> None:
    """
    Creates an interactive horizontal bar chart to visualize prediction confidence.
    The x-axis ranges from 0 to 1, and each bar represents the 'Prediction Confidence' value.
    Each bar is colored according to the confidence level:
      - Low Confidence (0 - 0.3)
      - Medium Confidence (0.3 - 0.5)
      - High Confidence (0.5 - 1)
    
    A single legend explains the confidence intervals.
    
    Parameters:
    - df_results (pd.DataFrame): DataFrame containing the columns 
      "Query Name" and "Prediction Confidence ( range: 0 - 1 )"
    """
    # Convert the confidence column to float
    df_results['Prediction Confidence'] = df_results['Prediction Confidence ( range: 0 - 1 )'] \
        .astype(float)
    
    # Function to determine the confidence level
    def get_confidence_level(confidence):
        if confidence < 0.3:
            return "Low Confidence (0 - 0.3)"
        elif confidence < 0.5:
            return "Medium Confidence (0.3 - 0.5)"
        else:
            return "High Confidence (0.5 - 1)"
    
    # Apply the function
    df_results['confidence_level'] = df_results['Prediction Confidence'] \
        .apply(get_confidence_level)
    
    # Define color mapping (same colors, on a white background)
    color_map = {
        "Low Confidence (0 - 0.3)": "#F4D03F",   # Golden Yellow
        "Medium Confidence (0.3 - 0.5)": "#E67E22",  # Orange
        "High Confidence (0.5 - 1)": "#3498DB"      # Light Blue
    }
    
    # Create the bar chart
    fig_conf = px.bar(
        df_results,
        x='Prediction Confidence',
        y='Query Name',
        orientation='h',
        text=df_results['Prediction Confidence'].apply(lambda x: f"{x:.2f}"),
        color='confidence_level',
        color_discrete_map=color_map
    )
    
    # Update layout for white background and black text
    fig_conf.update_layout(
        title={
            'text': "Prediction Confidence Horizontal Bar Chart",
            'font': {'color': 'black', 'size': 20}
        },
        xaxis=dict(
            range=[0, 1],
            title={'text': "Prediction Confidence", 'font': {'color': 'black', 'size': 16}},
            tickfont={'color': 'black'}
        ),
        yaxis=dict(
            title={'text': "Query Name", 'font': {'color': 'black', 'size': 16}},
            tickfont={'color': 'black'}
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend_title_text="Confidence Intervals",
        legend=dict(font=dict(color='black'))
    )
    
    # Display in Streamlit
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Generate PNG for download
    img_bytes = fig_conf.to_image(format="png")
    st.download_button(
        label="Download Bar Chart",
        data=img_bytes,
        file_name="prediction_confidence_bar_chart.png",
        mime="image/png"
    )


def main(args: argparse.Namespace) -> None:
    """
    Main function coordinating the workflow.
    
    Parameters:
    - args (argparse.Namespace): Input arguments.
    """
    model_dir = args.model_dir

    # Initialize progress variables
    total_steps = 5  # Adjusted to include plot_dual_umap
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # =============================
    # STEP 1: Training the Model for associated_variable
    # =============================

    # Load training data
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table

    # Check if training sequences are aligned
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Training sequences are not aligned. Realigning with MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)  # Threads set to 1
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Aligned training file found or sequences already aligned: {train_alignment_path}")

    # Load training table data
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Training table data loaded successfully.")

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Initialize and generate embeddings for training
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, 
        table_data=train_table_data, 
        aggregation_method=args.aggregation_method  # Passing the aggregation method ('none' or 'mean')
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        save_min_kmers=True,  # Save min_kmers after training
        window=5,   # Ajustar se necessÃƒÂ¡rio (ou via args)
        workers=48, # Ajustar se necessÃƒÂ¡rio
        epochs=2500 # Ajustar se necessÃƒÂ¡rio
    )
    logging.info(f"Number of training embeddings generated: {len(protein_embedding_train.embeddings)}")

    # Save min_kmers to ensure consistency
    min_kmers = protein_embedding_train.min_kmers

    # Obtaining protein IDs and associated variables from the training set
    protein_ids_associated = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]
    var_assoc_associated = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]

    logging.info(f"Protein IDs for associated_variable extracted: {len(protein_ids_associated)}")
    logging.info(f"Associated variables for associated_variable extracted: {len(var_assoc_associated)}")

    # Get embeddings and labels for associated_variable
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"Shape of X_associated: {X_associated.shape}")

    # Create scaler for X_associated
    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info("Scaler for X_associated created and saved.")

    # Scale the X_associated data
    X_associated_scaled = scaler_associated.transform(X_associated)    

    # Full paths for associated_variable models
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Check if the calibrated model for associated_variable already exists
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Random Forest calibrated model for associated_variable loaded from {calibrated_model_associated_full_path}")
    else:
        # Train the model for associated_variable
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(
            X_associated_scaled, 
            y_associated, 
            protein_ids=protein_ids_associated,  # Pass protein IDs for visualization
            var_assoc=var_assoc_associated, 
            model_name_prefix='associated', 
            model_dir=model_dir, 
            min_kmers=min_kmers
        )

        logging.info("Training and calibration for associated_variable completed.")

        # Plot the learning curve
        logging.info("Plotting Learning Curve for associated_variable")
        learning_curve_associated_path = args.learning_curve_associated
        support_model_associated.plot_learning_curve(learning_curve_associated_path)

        # Save the calibrated model
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Random Forest calibrated model for associated_variable saved at {calibrated_model_associated_full_path}")

        # Test the model
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(
            X_associated_scaled, 
            y_associated
        )

        logging.info(f"Best ROC AUC for associated_variable: {best_score_associated}")
        logging.info(f"Best F1 Score for associated_variable: {best_f1_associated}")
        logging.info(f"Best Precision-Recall AUC for associated_variable: {best_pr_auc_associated}")
        logging.info(f"Best Parameters: {best_params_associated}")

        for param, value in best_params_associated.items():
            logging.info(f"{param}: {value}")

        # Accessing class_weight from best_params_associated
        class_weight = best_params_associated.get('class_weight', None)
        # Printing results
        logging.info(f"Class weight used: {class_weight}")

        # Save the trained model for associated_variable
        joblib.dump(best_model_associated, rf_model_associated_full_path)
        logging.info(f"Random Forest model for associated_variable saved at {rf_model_associated_full_path}")

        # Plot ROC curve for associated_variable
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
            classes_associated = None  # Binary classification
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve_global(
            y_test_associated, 
            y_pred_proba_associated, 
            'ROC Curve for Associated Variable', 
            save_as=args.roc_curve_associated, 
            classes=classes_associated
        )

        plot_precision_recall_curve_global(y_test_associated, y_pred_proba_associated, title='Precision-Recall Curve', save_as=os.path.join(model_dir, 'pr_curve_associated.png'), classes=classes_associated)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # STEP 2: Classifying New Sequences for associated_variable
    # =============================

    # Load min_kmers
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers loaded: {min_kmers_loaded}")
    else:
        logging.error(f"min_kmers file not found at {min_kmers_path}. Please make sure the training was completed successfully.")
        sys.exit(1)

    # Load prediction data
    predict_alignment_path = args.predict_fasta

    # Check if prediction sequences are aligned
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Sequences for prediction are not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Threads set to 1
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Aligned file for prediction found or sequences already aligned: {predict_alignment_path}")

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Initialize ProteinEmbedding for prediction, no need for the table
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, 
        table_data=None,
        aggregation_method=args.aggregation_method  # Passing the aggregation method ('none' or 'mean')
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded  # Use the same min_kmers from training
    )
    logging.info(f"Number of embeddings generated for prediction: {len(protein_embedding_predict.embeddings)}")

    # Get embeddings for prediction
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Load scalers for associated_variable
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')

    if os.path.exists(scaler_associated_path):
        scaler_associated = joblib.load(scaler_associated_path)
        logging.info("Scaler for associated_variable loaded successfully.")
    else:
        logging.error("Scalers not found. Please make sure the training was completed successfully.")
        sys.exit(1)

    # Scale prediction embeddings using scaler_associated
    X_predict_scaled_associated = scaler_associated.transform(X_predict)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Perform prediction for associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled_associated)
    
    predict_labels = [str(label) for label in predictions_associated]
    print("O valor is", predict_labels)


    # Get class rankings using the global function
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled_associated)

    # Process and save results
    results = {}
    for entry, pred_associated, ranking_associated in zip(
        protein_embedding_predict.embeddings,
        predictions_associated,
        rankings_associated
    ):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "associated_prediction": pred_associated,
            "associated_ranking": ranking_associated
        }

    # Save results to a file
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tAssociated_Prediction\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['associated_prediction']}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Associated Variable: {result['associated_prediction']}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

    # Generate the Scatter Plot of Predictions
    logging.info("Generating scatter plot of predictions for new sequences...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatter plot saved at {args.scatterplot_output}")
    train_labels = y_associated
    predict_labels = predictions_associated
    train_protein_ids = protein_ids_associated
    predict_protein_ids = [entry['protein_accession'] for entry in protein_embedding_predict.embeddings]

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    st.success("Analysis completed successfully!")

    # Display scatter plot
    st.header("Scatter Plot of Predictions")
    scatterplot_path = args.scatterplot_output
    if os.path.exists(scatterplot_path):
        st.image(scatterplot_path, use_container_width=True)
    else:
        st.error(f"Scatter plot not found at {scatterplot_path}")

    # Format the results
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No ranking data associated with the protein. {sequence_id}. Skipping...")
            continue

        # Use the function format_and_sum_probabilities to get the main category and normalized confidence
        main_category, normalized_confidence = format_and_sum_probabilities(associated_rankings)
        if main_category is None:
            logging.warning(f"No valid formatting for protein {sequence_id}. Skipping...")
            continue
        # Ajustando a terceira coluna para "Prediction Confidence ( range: 0 - 1 )"
        formatted_results.append([
            sequence_id,
            main_category,
            f"{normalized_confidence:.2f}"  # Exemplo: 0.80
        ])

    # Ajustar para 3 colunas (removendo a 'Heatmap')
    headers = ["Query Name", "SS Prediction Specificity", "Prediction Confidence ( range: 0 - 1 )"]
    df_results = pd.DataFrame(formatted_results, columns=headers)

    # FunÃƒÂ§ÃƒÂ£o para aplicar estilos personalizados
# FunÃƒÂ§ÃƒÂ£o para aplicar estilos personalizados com fundo branco
    def highlight_table(df):
        return df.style.set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#f2f2f2'),  # Cinza muito claro para cabeÃƒÂ§alho
                    ('color', '#000000'),              # Texto preto
                    ('border', '1px solid #dddddd'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('background-color', '#ffffff'),  # Branco para linhas ÃƒÂ­mpares
                    ('color', '#000000'),             # Texto preto
                    ('border', '1px solid #dddddd'),
                    ('text-align', 'center'),
                    ('font-family', 'Arial'),
                    ('font-size', '12px'),
                ]
            },
            {
                'selector': 'tr:nth-child(even) td',
                'props': [
                    ('background-color', '#f9f9f9')   # Cinza claro para linhas pares
                ]
            },
            {
                'selector': 'tr:hover td',
                'props': [
                    ('background-color', '#e6e6e6')   # Cinza um pouco mais escuro ao passar o mouse
                ]
            },
    ])


    # Mantemos a funÃƒÂ§ÃƒÂ£o, mas sem subset de background_gradient
    styled_df = highlight_table(df_results)

    html = styled_df.to_html(index=False, escape=False)
##########################  Injetar CSS para download buttons e ajustar estilos adicionais
 

    st.header("Formatted Results")


    st.markdown(
        f"""
        <div class="dataframe-container">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # BotÃƒÂ£o para download em CSV
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name='results.csv',
        mime='text/csv',
    )

    # BotÃƒÂ£o para download em Excel
    output = BytesIO() 
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Results')
        writer.close()

        processed_data = output.getvalue()

    st.download_button(
        label="Download Results as Excel",
        data=processed_data,
        file_name='results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    # NOVA SEÃƒâ€¡ÃƒÆ’O: Exibir grÃƒÂ¡fico de barras horizontal para Prediction Confidence
    st.header("Prediction Confidence Bar Chart")
    plot_prediction_confidence_bar(df_results)

    # Prepare results.zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for folder_name, subfolders, filenames in os.walk(args.output_dir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, args.output_dir))
    zip_buffer.seek(0)

    # Provide download link
    st.header("Download All Results")
    st.download_button(
        label="Download All Results as results.zip",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )

    # Salvar os resultados em um arquivo Excel
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Resultados salvos em {args.excel_output}")

    # Salvar a tabela em formato tabulado
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved at {args.formatted_results_table}")
    
    umap_similarity_path = os.path.join(args.output_dir, "umap_similarity_3D.html")
    # Salvar os Dual UMAP plots na pasta de saÃƒÂ­da para inclusÃƒÂ£o no ZIP
    

# =======================
# Auxiliary Tool: Get your FAAL domain
# =======================
st.sidebar.header("Auxiliary Tools")

# ==============================================
# Auxiliary Functions for InterProScan & FAAL
# ==============================================
def submit_job(fasta_file, email="user@example.com", title="InterProScanJob"):
    """
    Submits the FASTA file to the InterProScan REST API and returns the generated job_id.
    """
    url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run"
    with open(fasta_file, "r") as f:
        sequence_data = f.read()
    data = {
        "sequence": sequence_data,
        "email": email,
        "title": title,
        "format": "tsv"
    }

    st.info("Submitting job to InterProScan REST API...")
    response = requests.post(url, data=data)
    if response.status_code != 200:
        st.error(f"Error submitting job. Status code: {response.status_code}")
        return None

    job_id = response.text.strip()
    st.success(f"Job submitted successfully. Job ID: {job_id}")
    return job_id


def poll_status(job_id, poll_interval=30):
    """
    Periodically checks the status of the InterProScan job.
    Returns True if finished, or False if an error/failure occurs.
    """
    status_url = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/status/{job_id}"
    while True:
        response = requests.get(status_url)
        if response.status_code != 200:
            st.error(f"Error getting job status. Status code: {response.status_code}")
            return False

        status = response.text.strip()
        st.info(f"Job status: {status}")

        if status == "FINISHED":
            return True
        elif status in ["ERROR", "FAILURE"]:
            st.error("Job failed.")
            return False

        time.sleep(poll_interval)


def retrieve_result(job_id, result_filename):
    """
    Downloads the job result (TSV) from the InterProScan REST API into `result_filename`.
    """
    result_url = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/result/{job_id}/tsv"
    st.info("Retrieving job result...")
    response = requests.get(result_url)
    if response.status_code != 200:
        st.error(f"Error retrieving result. Status code: {response.status_code}")
        return None

    with open(result_filename, "w") as f:
        f.write(response.text)

    st.success(f"Result saved in {result_filename}")
    return result_filename


def extract_faal(tsv_file, faal_output):
    """
    Filters lines containing 'FAAL' from the TSV and writes them to faal_output.
    """
    st.info("Extracting lines containing 'FAAL'...")
    with open(tsv_file, "r") as fin, open(faal_output, "w") as fout:
        for line in fin:
            if "FAAL" in line:
                fout.write(line)

    st.success(f"File {faal_output} generated.")


def create_bed(faal_file, bed_file):
    """
    Creates a BED file from the lines in `faal_file`.
    """
    st.info("Creating BED file from FAAL file...")
    with open(faal_file, "r") as fin, open(bed_file, "w") as fout:
        for line in fin:
            fields = line.strip().split("\t")
            if len(fields) >= 8:
                fout.write("\t".join([fields[0], fields[6], fields[7], fields[0]]) + "\n")
            else:
                st.warning(f"Insufficient columns in line: {line.strip()}")

    st.success(f"BED file {bed_file} generated.")


def run_bedtools(ref_fasta, bed_file, fasta_output):
    """
    Runs bedtools getfasta to extract sequences from `ref_fasta` (FAAL regions).
    """
    cmd = f"bedtools getfasta -name -fi {ref_fasta} -bed {bed_file} -fo {fasta_output}"
    st.info("Running bedtools getfasta...")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running bedtools getfasta: {e}")
        return False

    st.success(f"FASTA file generated: {fasta_output}")
    return True


def process_faal_domain(uploaded_fasta, email, title, output_dir):
    """
    Main flow for FAAL domain extraction using InterProScan + bedtools.
    
    1. Ensure 'faal' output directory is clean (remove old files if any).
    2. Save uploaded FASTA.
    3. Submit InterProScan job & poll status.
    4. Retrieve results in TSV.
    5. Extract lines containing 'FAAL'.
    6. Create BED file.
    7. bedtools getfasta to generate final FASTA with the FAAL domain.
    
    Returns the path to the processed FASTA or None if any step fails.
    """
    # 1) Clean output_dir if it already exists and has files
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        if os.listdir(output_dir):  # if not empty
            st.info(f"Cleaning old files in {output_dir} ...")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    else:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    # 2) Save the uploaded FASTA to 'faal_input.fasta'
    fasta_path = os.path.join(output_dir, "faal_input.fasta")
    with open(fasta_path, "wb") as f:
        f.write(uploaded_fasta.getbuffer())

    base_name = os.path.splitext(os.path.basename(fasta_path))[0]
    iprscan_result_file = os.path.join(output_dir, f"{base_name}.iprscan_result.tsv")
    faal_tsv = os.path.join(output_dir, f"{base_name}.interpro.tsv")
    bed_file = os.path.join(output_dir, f"{base_name}.interpro.bed")
    tmp_output = os.path.join(output_dir, f"{base_name}.temp.fasta")

    # 3) InterProScan job submission
    job_id = submit_job(fasta_path, email=email, title=title)
    if not job_id:
        return None

    # 4) Poll status until FINISHED (or ERROR)
    if not poll_status(job_id):
        return None

    # 5) Retrieve InterProScan results
    if not retrieve_result(job_id, iprscan_result_file):
        return None

    # 6) Extract FAAL lines & create BED
    extract_faal(iprscan_result_file, faal_tsv)
    create_bed(faal_tsv, bed_file)

    # 7) Run bedtools to get final FASTA
    if not run_bedtools(fasta_path, bed_file, tmp_output):
        return None

    # Replace the original 'faal_input.fasta' with the newly extracted domain
    os.replace(tmp_output, fasta_path)
    st.success(f"Processed FASTA (FAAL domains extracted): {fasta_path}")
    return fasta_path


# ==============================================
# Streamlit Sidebar for FAAL Domain Extraction
# ==============================================
get_faal_domain = st.sidebar.checkbox("Get your FAAL domain")
if get_faal_domain:
    st.sidebar.markdown("### Upload FASTA for FAAL domain extraction")
    uploaded_fasta_faal = st.sidebar.file_uploader("Upload FASTA", type=["fasta", "fa", "fna"], key="faal_domain")
    email_faal = st.sidebar.text_input("Email", value="user@example.com", key="faal_email")
    title_faal = st.sidebar.text_input("Job Title", value="InterProScanJob", key="faal_title")

    if uploaded_fasta_faal is not None and st.sidebar.button("Process FAAL Domain"):
        output_faal_dir = os.path.join("results", "faal")

        with st.spinner("Processing FAAL domain... (this may take several minutes)"):
            result_fasta = process_faal_domain(uploaded_fasta_faal, email_faal, title_faal, output_faal_dir)

        if result_fasta:
            # Read final FASTA content
            with open(result_fasta, "r") as f:
                result_content = f.read()

            # Download button for user
            st.download_button(
                label="Download Processed FASTA",
                data=result_content,
                file_name=os.path.basename(result_fasta),
                mime="text/plain"
            )

            # Text area to display file
            st.text_area("Processed FASTA", result_content, height=300)

            
def get_base64_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.

    
    Parameters:
    - image_path (str): Path to the image file.

    
    Retorna:
    - str: String base64 da imagem.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return ""

# Caminho para a imagem
image_path = "./images/faal.png"
image_base64 = get_base64_image(image_path)
# Using HTML with st.markdown to align title and text
st.markdown(
    f"""
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 30px; background: linear-gradient(to bottom, #f9f9f9, #ffffff); border-radius: 15px; border: 2px solid #dddddd; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); position: relative;">
        <p style="color: black; font-size: 1.5em; font-weight: bold; margin: 0;">
            FAALPred: Predicting Fatty Acyl Chain Specificities in Fatty Acyl-AMP Ligases (FAALs) Using Integrated Approaches of Neural Networks, Bioinformatics, and Machine Learning
        </p>
        <p style="color: #2c3e50; font-size: 1.2em; font-weight: normal; margin-top: 10px;">
            Leandro de Mattos Pereira, Anne Liong and Pedro LeÃ£o
        </p>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8;">
            <strong>FAALPred</strong> is a comprehensive bioinformatics tool designed to predict the fatty acyl chain length specificity of FAALs, ranging from C4 to C18.
        </p>
        <h5 style="color: #2c3e50; font-size: 20px; font-weight: bold; margin-top: 25px;">ABSTRACT</h5>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8; text-align: justify;">
            Fatty Acyl-AMP Ligases (FAALs), identified by Zhang et al. (2011), activate fatty acids of varying lengths for the biosynthesis of natural products. 
            These substrates enable the production of compounds such as nocuolin (<em>Nodularia sp.</em>, Martins et al., 2022) 
            and sulfolipid-1 (<em>Mycobacterium tuberculosis</em>, Yan et al., 2023), with applications in cancer and tuberculosis treatment 
            (Kurt et al., 2017; Gilmore et al., 2012). Dr. Pedro LeÃ£o team (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" style="color: #3498db; text-decoration: none;">visit here</a>)
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{image_base64}" alt="FAAL Domain" style="width: auto; height: 120px; object-fit: contain;">
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>FAAL Domain of Synechococcus sp. PCC7002, link: <a href="https://www.rcsb.org/structure/7R7F" target="_blank" style="color: #3498db; text-decoration: none;">https://www.rcsb.org/structure/7R7F</a></em>
            </p>
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>Note: Before running FAALPred, extract the FAAL domain from your protein using the Auxiliary Tools. The model was trained and developed for specificity prediction using only the FAAL domain with signatures from CDD protein databases (cd05931: FAAL).</em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to save uploaded files
def save_uploaded_file(uploaded_file, save_path: str) -> str:
    """
    Saves a file uploaded by the user.
    
    Parameters:
    - uploaded_file: File uploaded by the user.
    - save_path (str): Path to save the file.
    
    Returns:
    - str: Path to the saved file.
    """
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path


# =====================================
# SessÃƒÂ£o de Input (barra lateral)
# =====================================
st.sidebar.header("Input Parameters")

st.sidebar.header("Use Default Training Data")
# ParÃƒÂ¢metro 1: Escolha se deseja usar dados de treinamento padrÃƒÂ£o
use_default_train = st.sidebar.checkbox("Use Default Training Data", value=True)

#use_default_train = st.sidebar.file_uploader("Use Default Training Data")

if not use_default_train:
    # Se o usuÃƒÂ¡rio desmarcar, pedimos o upload dos arquivos
    train_fasta_file = st.sidebar.file_uploader("Upload Training FASTA File", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("Upload Training Table File (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

# ParÃƒÂ¢metro 2: FASTA de previsÃƒÂ£o
predict_fasta_file = st.sidebar.file_uploader("Upload Prediction FASTA File", type=["fasta", "fa", "fna"])

# ParÃƒÂ¢metro 3: K-mer e Step
kmer_size = st.sidebar.number_input("K-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)


aggregation_method = st.sidebar.selectbox(
    "Aggregation Method",
    options=['mean'],  # Only 'none' and 'mean' are options
    index=0
)

custom_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)


if custom_word2vec:
    window = st.sidebar.number_input(
        "Window Size", min_value=5, max_value=20, value=10, step=5
    )
    workers = st.sidebar.number_input(
        "Workers", min_value=1, max_value=112, value=8, step=8
    )
    epochs = st.sidebar.number_input(
        "Epochs", min_value=1, max_value=2500, value=2500, step=100
    )
else:
    window = 5  # Default value
    workers = 48  # Default value
    epochs = 2500  # Default value

# Output directory based on aggregation method
model_dir = create_unique_model_directory("results", aggregation_method)
output_dir = model_dir



# Button to start processing
if st.sidebar.button("Run Analysis"):
    # Paths for internal data
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"

    # Handling training data
    if use_default_train:
        train_fasta_path = internal_train_fasta
        train_table_path = internal_train_table
        st.markdown("<span style='color:white'>Using default training data.</span>", unsafe_allow_html=True)

    else:
        if train_fasta_file is not None and train_table_file is not None:
            train_fasta_path = os.path.join(output_dir, "uploaded_train.fasta")
            train_table_path = os.path.join(output_dir, "uploaded_train_table.tsv")
            save_uploaded_file(train_fasta_file, train_fasta_path)
            save_uploaded_file(train_table_file, train_table_path)
            st.markdown("<span style='color:white'>Uploaded training data will be used.</span>", unsafe_allow_html=True)

        else:
            st.error("Please upload both the training FASTA file and the training TSV table file.")
            st.stop()

    # Handling prediction data
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(output_dir, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Please upload a FASTA file for prediction.")
        st.stop()
        
    # Remaining parameters
    args = argparse.Namespace(
        train_fasta=train_fasta_path,
        train_table=train_table_path,
        predict_fasta=predict_fasta_path,
        kmer_size=kmer_size,
        step_size=step_size,
        aggregation_method=aggregation_method,
        results_file=os.path.join(output_dir, "predictions.tsv"),
        output_dir=output_dir,
        scatterplot_output=os.path.join(output_dir, "scatterplot_predictions.png"),
        excel_output=os.path.join(output_dir, "results.xlsx"),
        formatted_results_table=os.path.join(output_dir, "formatted_results.txt"),
        roc_curve_associated=os.path.join(output_dir, "roc_curve_associated.png"),
        learning_curve_associated=os.path.join(output_dir, "learning_curve_associated.png"),
        roc_values_associated=os.path.join(output_dir, "roc_values_associated.csv"),
        rf_model_associated="rf_model_associated.pkl",
        word2vec_model="word2vec_model.bin",
        scaler="scaler_associated.pkl",  # Corrected scaler name
        model_dir=model_dir,
    )

    # Create model directory if it does not exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Run the main analysis function
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(args)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")

# Function to load and resize images with DPI adjustment
def load_and_resize_image_with_dpi(image_path: str, base_width: int, dpi: int = 300) -> Image.Image:
    """
    Loads and resizes an image with DPI adjustment.
    
    Parameters:
    - image_path (str): Path to the image file.
    - base_width (int): Base width for resizing.
    - dpi (int): DPI for the image.
    
    Returns:
    - Image.Image: Resized image object.
    """
    try:
        # Load the image
        image = Image.open(image_path)
        # Calculate the new height proportionally based on the base width
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Resize the image
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

# Function to encode images to base64
def encode_image(image: Image.Image) -> str:
    """
    Encodes an image as a base64 string.
    
    Parameters:
    - image (Image.Image): Image object.
    
    Returns:
    - str: Base64 string of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Definitions of image paths
image_dir = "images"
image_paths = [
    os.path.join(image_dir, "lab_logo.png"),
    os.path.join(image_dir, "ciimar.png"),
    os.path.join(image_dir, "faal_pred_logo.png"), 
    os.path.join(image_dir, "bbf4.png"),
    os.path.join(image_dir, "google.png"),
    os.path.join(image_dir, "uniao.png"),
]

# Load and resize all images
images = [load_and_resize_image_with_dpi(path, base_width=150, dpi=300) for path in image_paths]

# Encode images to base64
encoded_images = [encode_image(img) for img in images if img is not None]

# CSS for footer layout
st.markdown(
    """
    <style>
    .footer-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }
    .footer-text {
        text-align: center;
        color: white;
        font-size: 12px;
        margin-top: 10px;
    }
    .support-text {
        text-align: center;
        color: white;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# HTML para exibir imagens no rodapÃƒÂ©
footer_html = """
<div class="support-text">Supported by:</div>
<div class="footer-container">
    {}
</div>
<div class="footer-text">
    CIIMAR - Pedro LeÃƒÂ£o @CNP - 2024 - All rights reserved.
</div>

"""

# Gera tags <img> para cada imagem
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Renderiza o rodapÃƒÂ©
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)
def update_visit_count(file_path=os.path.join("logs", "visit_count.txt")):
    if not os.path.exists(file_path):
        count = 0
    else:
        with open(file_path, "r") as f:
            try:
                count = int(f.read().strip())
            except ValueError:
                count = 0
    count += 1
    with open(file_path, "w") as f:
        f.write(str(count))
    return count

def get_visitor_country():
    try:
        response = requests.get("https://ipapi.co/json/")
        if response.status_code == 200:
            data = response.json()
            return data.get("country_name", "Desconhecido")
        else:
            return "Desconhecido"
    except Exception as e:
        return "Desconhecido"

# Atualiza a contagem e obtÃƒÂ©m o paÃƒÂ­s do visitante
visit_count = update_visit_count()
visitor_country = get_visitor_country()

# Exibe a contagem de visitas e o paÃƒÂ­s no rodapÃƒÂ©
st.markdown(
    f"""
    <div style="text-align: center; color: black; font-size: 14px; margin-top: 10px;">
        Visit Count: {visit_count} | Country: {visitor_country}
    </div>
    """,
    unsafe_allow_html=True
)


