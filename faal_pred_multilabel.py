import logging
import os
import sys
import subprocess
import random
import zipfile
from collections import Counter, defaultdict
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
# Precisamos das bibliotecas abaixo para oversampling multilabel
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from matplotlib import ticker
import umap.umap_ as umap
import umap
import base64
from plotly.graph_objs import Figure
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.metrics import make_scorer

# ============================================
# Setting seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)

# ============================================
# Streamlit Configuration and Interface
# ============================================
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded",
)

def are_sequences_aligned(fasta_file: str) -> bool:
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1

def create_unique_model_directory(base_dir: str, aggregation_method: str) -> str:
    model_dir = os.path.join(base_dir, f"models_{aggregation_method}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def realign_sequences_with_mafft(input_path: str, output_path: str, threads: int = 8) -> None:
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Realigned sequences saved to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing MAFFT: {e.stderr.decode()}")
        sys.exit(1)

from sklearn.cluster import DBSCAN, KMeans

def perform_clustering(data: np.ndarray, method: str = "DBSCAN", eps: float = 0.5, min_samples: int = 5, n_clusters: int = 3) -> np.ndarray:
    if method == "DBSCAN":
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "K-Means":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Invalid clustering method: {method}")
    labels = clustering_model.fit_predict(data)
    return labels

def plot_dual_tsne(train_embeddings: np.ndarray, train_labels: list, train_protein_ids: list,
                   predict_embeddings: np.ndarray, predict_labels: list, predict_protein_ids: list,
                   output_dir: str) -> tuple:
    from sklearn.manifold import TSNE
    tsne_train = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_train_result = tsne_train.fit_transform(train_embeddings)
    tsne_predict = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_predict_result = tsne_predict.fit_transform(predict_embeddings)
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter3d(
        x=tsne_train_result[:, 0],
        y=tsne_train_result[:, 1],
        z=tsne_train_result[:, 2],
        mode='markers',
        marker=dict(size=5, color=train_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {lbl}" for pid, lbl in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Training Data'
    ))
    fig_train.update_layout(
        title='t-SNE 3D: Training Data',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    fig_predict = go.Figure()
    fig_predict.add_trace(go.Scatter3d(
        x=tsne_predict_result[:, 0],
        y=tsne_predict_result[:, 1],
        z=tsne_predict_result[:, 2],
        mode='markers',
        marker=dict(size=5, color=predict_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {lbl}" for pid, lbl in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Predictions'
    ))
    fig_predict.update_layout(
        title='t-SNE 3D: Predictions',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")
    pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
    pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)
    logging.info(f"t-SNE Training plot saved as {tsne_train_html}")
    logging.info(f"t-SNE Predictions plot saved as {tsne_predict_html}")
    return fig_train, fig_predict

def plot_roc_curve_global(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
    lw = 2
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=unique_classes)
        n_classes = y_bin.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            class_label = classes[i] if classes is not None else unique_classes[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f'ROC Curve for class {class_label} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.title(title, color='white')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')
    plt.close()

def get_class_rankings_global(model, X: np.ndarray) -> list:
    if model is None:
        raise ValueError("Model not trained. Please train the model first.")
    y_pred_proba = model.predict_proba(X)
    if hasattr(model, "classes_"):
        class_labels = list(model.classes_)
    else:
        class_labels = [f"Label_{i}" for i in range(y_pred_proba.shape[1])]
    rankings = []
    for sample_proba in y_pred_proba:
        pairs = sorted(zip(class_labels, sample_proba), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in pairs]
        rankings.append(formatted_rankings)
    return rankings

def calculate_roc_values(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    n_classes = len(np.unique(y_test))
    y_pred_proba = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        logging.info(f"For class {i}:")
        logging.info(f"FPR: {fpr[i]}")
        logging.info(f"TPR: {tpr[i]}")
        logging.info(f"ROC AUC: {roc_auc[i]}")
        logging.info("--------------------------")
    roc_df = pd.DataFrame(list(roc_auc.items()), columns=['Class', 'ROC AUC'])
    return roc_df

def visualize_latent_space_with_similarity(X_original: np.ndarray, X_synthetic: np.ndarray, 
                                             y_original: np.ndarray, y_synthetic: np.ndarray, 
                                             protein_ids_original: list, protein_ids_synthetic: list, 
                                             var_assoc_original: list, var_assoc_synthetic: list, 
                                             output_dir: str = None) -> Figure:
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.vstack([y_original, y_synthetic])
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    X_transformed = umap_reducer.fit_transform(X_combined)
    X_transformed_original = X_transformed[:len(X_original)]
    X_transformed_synthetic = X_transformed[len(X_original):]
    similarities = cosine_similarity(X_synthetic, X_original)
    max_similarities = similarities.max(axis=1)
    closest_indices = similarities.argmax(axis=1)
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
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df_original['x'],
        y=df_original['y'],
        z=df_original['z'],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.7),
        name='Original',
        text=df_original.apply(lambda row: f"Protein ID: {row['Protein ID']}<br>Associated Variable: {row['Associated Variable']}", axis=1),
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter3d(
        x=df_synthetic['x'],
        y=df_synthetic['y'],
        z=df_synthetic['z'],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.7),
        name='Synthetic',
        text=df_synthetic.apply(lambda row: f"Protein ID: {row['Protein ID']}<br>Associated Variable: {row['Associated Variable']}<br>Similarity: {row['Similarity']:.4f}<br>Closest Protein: {row['Closest Protein']}<br>Closest Variable: {row['Closest Variable']}", axis=1),
        hoverinfo='text'
    ))
    fig.update_layout(
        title="Latent Space Visualization with Similarity (UMAP 3D)",
        scene=dict(xaxis_title="UMAP Dimension 1",
                   yaxis_title="UMAP Dimension 2",
                   zaxis_title="UMAP Dimension 3"),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )
    if output_dir:
        umap_similarity_path = os.path.join(output_dir, "umap_similarity_3D.html")
        fig.write_html(umap_similarity_path)
        logging.info(f"UMAP plot saved at {umap_similarity_path}")
    return fig

def format_and_sum_probabilities(associated_rankings: list) -> tuple:
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
            prob = float(rank.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Error processing ranking string: {rank}")
            continue
        for cat, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[cat] += prob
    if not category_sums:
        return None, None, None
    top_category, top_sum = max(category_sums.items(), key=lambda x: x[1])
    sorted_categories = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)
    top_two = sorted_categories[:2] if len(sorted_categories) >= 2 else sorted_categories
    top_two_categories = [f"{cat} ({prob:.2f}%)" for cat, prob in top_two]
    top_category_with_confidence = f"{top_category} ({top_sum:.2f}%)"
    return top_category_with_confidence, top_sum, top_two_categories

# ============================================
# Classe MLSMOTE (Custom Multi-Label SMOTE)
# ============================================
class MLSMOTE:
    """
    Classe para implementar a técnica MLSMOTE (Multi-Label Synthetic Minority Over-sampling Technique).
    Essa implementação realiza primeiro um random oversampling para as classes minoritárias (definidas como aquelas
    com número de exemplos abaixo de um limiar) e, em seguida, gera exemplos sintéticos via interpolação (MLSMOTE).
    Ao final, todas as classes presentes nos dados são incluídas.
    """
    def __init__(self, n_neighbors=5, random_state=None, min_samples=5):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.min_samples = min_samples
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
    def fit_resample(self, X, y, n_samples):
        # Converter X e y para DataFrames e resetar índices
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            X = X.reset_index(drop=True)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        else:
            y = y.reset_index(drop=True)
        print(f"[DEBUG] Dimensões iniciais: X={X.shape}, y={y.shape}")
        X_orig = X.copy()
        y_orig = y.copy()
        # Oversampling para cada classe minoritária
        for col in y.columns:
            count_col = y[col].sum()
            threshold = y.shape[0] / len(y.columns)
            if count_col < threshold:
                needed = self.min_samples - int(count_col)
                if needed > 0:
                    idx_positive = y.index[y[col] == 1].tolist()
                    if len(idx_positive) > 0:
                        oversampled_idx = resample(idx_positive, replace=True, n_samples=needed, random_state=self.random_state)
                        X_oversampled = X.loc[oversampled_idx]
                        y_oversampled = y.loc[oversampled_idx]
                        y_oversampled[col] = 1
                        X = pd.concat([X, X_oversampled], ignore_index=True)
                        y = pd.concat([y, y_oversampled], ignore_index=True)
                        print(f"[DEBUG] Oversampled coluna {col}: adicionadas {needed} amostras.")
        # Aplicar MLSMOTE para gerar amostras sintéticas adicionais
        X_sub, y_sub = self.get_minority_instance(X, y)
        print(f"[DEBUG] Após identificação de amostras minoritárias: X_sub={X_sub.shape}, y_sub={y_sub.shape}")
        if X_sub.shape[0] < self.min_samples:
            overX, overY = self.random_oversample(X_sub, y_sub, self.min_samples)
            X_sub = pd.concat([X_sub, overX], axis=0, ignore_index=True)
            y_sub = pd.concat([y_sub, overY], axis=0, ignore_index=True)
            print(f"[DEBUG] Após Random Oversampling: X_sub={X_sub.shape}, y_sub={y_sub.shape}")
        neighbors = self.find_nearest_neighbors(X_sub)
        print("[DEBUG] Vizinhos mais próximos encontrados para as amostras minoritárias.")
        new_X, new_y = self.generate_synthetic_samples(X_sub, y_sub, neighbors, n_samples)
        print(f"[DEBUG] Novas amostras sintéticas geradas: new_X={new_X.shape}, new_y={new_y.shape}")
        X_orig_arr = X_orig.to_numpy()
        y_orig_arr = y_orig.to_numpy()
        X_new_arr = new_X.to_numpy()
        y_new_arr = new_y.to_numpy()
        X_combined = np.vstack([X_orig_arr, X_new_arr])
        y_combined = np.vstack([y_orig_arr, y_new_arr])
        print(f"[DEBUG] Após concatenação via np.vstack: X_combined={X_combined.shape}, y_combined={y_combined.shape}")
        return {"original": (X_orig_arr, y_orig_arr),
                "synthetic": (X_new_arr, y_new_arr),
                "combined": (X_combined, y_combined)}
    def get_tail_label(self, y):
        cols = y.columns
        counts = np.array([y[col].sum() for col in cols], dtype=float)
        counts[counts == 0] = 1e-9
        ratios = counts.max() / counts
        avg_ratio = ratios.mean()
        tail_labels = [cols[i] for i in range(len(cols)) if ratios[i] > avg_ratio]
        print(f"[DEBUG] Classes minoritárias identificadas: {tail_labels}")
        return tail_labels
    def get_index(self, y):
        tail_labels = self.get_tail_label(y)
        indices = set()
        for tl in tail_labels:
            indices = indices.union(set(y.index[y[tl] == 1]))
        if len(indices) == 0:
            indices = set(y.index)
        print(f"[DEBUG] Número total de amostras minoritárias: {len(indices)}")
        return list(indices)
    def get_minority_instance(self, X, y):
        idx = self.get_index(y)
        X_sub = X.loc[idx].reset_index(drop=True)
        y_sub = y.loc[idx].reset_index(drop=True)
        return X_sub, y_sub
    def random_oversample(self, X_sub, y_sub, target_samples):
        current = X_sub.shape[0]
        needed = target_samples - current
        if needed <= 0:
            return pd.DataFrame(), pd.DataFrame()
        overX, overY = resample(X_sub, y_sub, replace=True, n_samples=needed, random_state=self.random_state)
        print(f"[DEBUG] Random OverSampling: Geradas {needed} novas amostras.")
        return overX, overY
    def find_nearest_neighbors(self, X_sub):
        k = min(self.n_neighbors, X_sub.shape[0])
        if k < 2:
            raise ValueError(f"Não há amostras suficientes para encontrar vizinhos: n_neighbors={self.n_neighbors}, n_samples={X_sub.shape[0]}")
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='kd_tree').fit(X_sub)
        distances, indices = nbrs.kneighbors(X_sub)
        return indices
    def generate_synthetic_samples(self, X_sub, y_sub, neighbors, n_samples):
        synth_X = []
        synth_y = []
        n_minority = X_sub.shape[0]
        for _ in range(n_samples):
            ref_idx = random.randint(0, n_minority - 1)
            neighbor_pool = list(neighbors[ref_idx])
            if ref_idx in neighbor_pool:
                neighbor_pool.remove(ref_idx)
            if not neighbor_pool:
                continue
            nb_idx = random.choice(neighbor_pool)
            ref_feat = X_sub.iloc[ref_idx].values
            nb_feat = X_sub.iloc[nb_idx].values
            alpha = random.random()
            new_feat = ref_feat + alpha * (nb_feat - ref_feat)
            synth_X.append(new_feat)
            ref_label = y_sub.iloc[ref_idx].values.astype(float)
            nb_label = y_sub.iloc[nb_idx].values.astype(float)
            new_label = (ref_label + nb_label) / 2.0
            new_label = (new_label > 0.5).astype(int)
            synth_y.append(new_label)
        if synth_X:
            new_X = pd.DataFrame(synth_X, columns=X_sub.columns)
        else:
            new_X = pd.DataFrame(columns=X_sub.columns)
        if synth_y:
            new_y = pd.DataFrame(synth_y, columns=y_sub.columns)
        else:
            new_y = pd.DataFrame(columns=y_sub.columns)
        return new_X, new_y

# ============================================
# Função scorer customizada para multilabel
# ============================================
def multilabel_f1_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples', zero_division=0)

# ============================================
# Classe Support
# ============================================
class Support:
    """
    Classe para treinar e avaliar RandomForest multi-label, usando Random OverSampling e MLSMOTE para gerar amostras sintéticas.
    """
    def __init__(self, cv: int = 5, seed: int = SEED, n_jobs: int = 8):
        self.cv = cv
        self.seed = seed
        self.n_jobs = n_jobs
        self.model = None
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []
        self.roc_results = []
        self.train_sizes = np.linspace(0.1, 1.0, 5)
        self.standard = StandardScaler()
        self.best_params = {}
        self.min_samples = 5
        self.init_params = {
            "n_estimators": 100,
            "max_depth": 2,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "criterion": "entropy",
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "max_leaf_nodes": 5,
            "min_impurity_decrease": 0.01,
            "bootstrap": True,
            "ccp_alpha": 0.005,
       }
        self.parameters = {
            "estimator__n_estimators": [250, 300],
            "estimator__max_depth": [10, 20],
            "estimator__min_samples_split": [4, 6],
            "estimator__min_samples_leaf": [4, 6],
            "estimator__criterion": ["gini", "entropy"],
            "estimator__max_features": ["log2"],
            "estimator__class_weight": ["balanced", None],
            "estimator__max_leaf_nodes": [10, 20, None],
            "estimator__min_impurity_decrease": [0.0],
            "estimator__bootstrap": [True, False],
            "estimator__ccp_alpha": [0.0],
        }
    def fit(self, X: np.ndarray, y: np.ndarray, protein_ids: list = None, var_assoc: list = None,
            model_name_prefix: str = 'model', model_dir: str = None, min_kmers: int = None):
        logging.info(f"Starting fit method for {model_name_prefix}...")
        if min_kmers is not None:
            logging.info(f"Using provided min_kmers: {min_kmers}")
        else:
            min_kmers = len(X)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)
        # Oversampling para cada classe minoritária
        for col in y_df.columns:
            count_col = y_df[col].sum()
            threshold = y_df.shape[0] / len(y_df.columns)
            if count_col < threshold:
                needed = self.min_samples - int(count_col)
                if needed > 0:
                    idx_positive = y_df.index[y_df[col] == 1].tolist()
                    if len(idx_positive) > 0:
                        oversampled_idx = resample(idx_positive, replace=True, n_samples=needed, random_state=self.seed)
                        X_oversampled = X_df.loc[oversampled_idx]
                        y_oversampled = y_df.loc[oversampled_idx]
                        y_oversampled[col] = 1
                        X_df = pd.concat([X_df, X_oversampled], ignore_index=True)
                        y_df = pd.concat([y_df, y_oversampled], ignore_index=True)
                        print(f"[DEBUG] Coluna {col}: Oversampled {needed} amostras.")
        # Aplicar MLSMOTE para gerar amostras sintéticas adicionais
        mlsmote = MLSMOTE(n_neighbors=5, random_state=self.seed, min_samples=self.min_samples)
        n_extra = max(int(len(X_df) * 0.3), 1)
        resampled = mlsmote.fit_resample(X_df, y_df, n_extra)
        X_smote, y_smote = resampled["combined"]
        logging.info(f"MLSMOTE applied. Final shapes - X: {X_smote.shape}, y: {y_smote.shape}")
        # Corrigido: para visualização, usar a quantidade de linhas de X_df (oversampled) para gerar os rótulos originais
        if protein_ids is not None and var_assoc is not None:
            visualize_latent_space_with_similarity(
                X_original=X_df.to_numpy(),
                X_synthetic=X_smote[len(X_df):],
                y_original=y_df.to_numpy(),
                y_synthetic=y_smote[len(y_df):],
                protein_ids_original=[f"orig_{i}" for i in range(len(X_df))],
                protein_ids_synthetic=[f"synthetic_{i}" for i in range(len(X_df), len(X_smote))],
                var_assoc_original=[f"orig_var" for i in range(len(X_df))],
                var_assoc_synthetic=[f"synthetic_var" for i in range(len(X_df), len(X_smote))],
                output_dir=model_dir
            )
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []
        fold_number = 0
        for train_idx, test_idx in kf.split(X_smote, y_smote):
            fold_number += 1
            X_train, X_test = X_smote[train_idx], X_smote[test_idx]
            y_train, y_test = y_smote[train_idx], y_smote[test_idx]
            logging.debug(f"Fold X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            base_rf = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs, random_state=self.seed)
            self.model = OneVsRestClassifier(base_rf)
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            y_pred = self.model.predict(X_test)
            f1_metric = f1_score(y_test, y_pred, average='samples', zero_division=0)
            self.f1_scores.append(f1_metric)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)
            y_pred_proba = self.model.predict_proba(X_test)
            pr_auc = average_precision_score(y_test, y_pred_proba, average='samples')
            self.pr_auc_scores.append(pr_auc)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Train Score: {train_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test Score: {test_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: F1 score (samples average): {f1_metric}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: PR AUC (samples average): {pr_auc}")
            try:
                roc_auc_score_value = roc_auc_score(y_test, y_pred_proba, average='samples')
                self.roc_results.append(roc_auc_score_value)
            except ValueError:
                logging.warning(f"Unable to calculate ROC AUC for fold {fold_number}.")
            scorer = make_scorer(multilabel_f1_scorer)
            kf_inner = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
            grid_search = GridSearchCV(
                OneVsRestClassifier(RandomForestClassifier(random_state=self.seed)),
                self.parameters,
                cv=kf_inner,
                n_jobs=self.n_jobs,
                scoring=scorer,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model, best_params = grid_search.best_estimator_, grid_search.best_params_
            self.model = best_model
            self.best_params = best_params
            if model_dir:
                best_model_filename = os.path.join(model_dir, f'model_best_{model_name_prefix}.pkl')
                os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved as {best_model_filename} for {model_name_prefix}")
            else:
                best_model_filename = f'model_best_{model_name_prefix}.pkl'
                joblib.dump(best_model, best_model_filename)
            if best_params is not None:
                self.best_params = best_params
                logging.info(f"Best parameters for {model_name_prefix}: {self.best_params}")
            else:
                logging.warning(f"No best parameters found in grid search for {model_name_prefix}.")
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=3, n_jobs=self.n_jobs)
            calibrator.fit(X_train, y_train)
            self.model = calibrator
            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'calibrated_model_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Calibrated model saved as {calibrated_model_filename} for {model_name_prefix}")
        return self.model
    def _perform_grid_search(self, X_train, y_train) -> tuple:
        assert X_train.ndim == 2, f"X_train deve ser 2D, mas tem {X_train.ndim} dimensões."
        assert y_train.ndim == 2, f"y_train deve ser 2D, mas tem {y_train.ndim} dimensões."


        base_rf = RandomForestClassifier(random_state=self.seed)
        estimator = OneVsRestClassifier(base_rf)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        scorer = make_scorer(multilabel_f1_scorer)
        logging.debug(f"X_train shape: {X_train.shape}")
        logging.debug(f"y_train shape: {y_train.shape}")
        logging.debug(f"Sample of y_train: {y_train[:5]}")  # Apenas para verificar o formato
        grid_search = GridSearchCV(
            estimator,
            self.parameters,
            cv=kf,
            n_jobs=self.n_jobs,
            scoring=scorer,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters from grid search: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_
    def get_class_rankings(self, X: np.ndarray) -> list:
        if self.model is None:
            raise ValueError("Model not trained.")
        y_pred_proba = self.model.predict_proba(X)
        if hasattr(self.model, "classes_"):
            class_labels = self.model.classes_
        else:
            class_labels = [f"Label_{i}" for i in range(y_pred_proba.shape[1])]
        rankings = []
        for sample_proba in y_pred_proba:
            pairs = sorted(zip(class_labels, sample_proba), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in pairs]
            rankings.append(formatted_rankings)
        return rankings
    def plot_learning_curve(self, output_path: str) -> None:
        plt.figure()
        plt.plot(self.train_scores, label='Train Score')
        plt.plot(self.test_scores, label='Test Score')
        plt.plot(self.f1_scores, label='F1 score (samples average)')
        plt.plot(self.pr_auc_scores, label='PR AUC (samples average)')
        plt.title("Learning Curve (Multilabel + MLSMOTE)", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')
        plt.close()
    def test_best_RF(self, X: np.ndarray, y: np.ndarray, scaler_dir: str = '.'):
        scaler_path = os.path.join(model_dir, 'scaler_associated.pkl') if model_dir else 'scaler_associated.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found at {scaler_path}.")
            sys.exit(1)
        X_scaled = scaler.transform(X)
        X_res, y_res = X_scaled, y
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=self.seed)
        base_rf = RandomForestClassifier(
            n_estimators=self.best_params.get('estimator__n_estimators', 100),
            max_depth=self.best_params.get('estimator__max_depth', 10),
            min_samples_split=self.best_params.get('estimator__min_samples_split', 2),
            min_samples_leaf=self.best_params.get('estimator__min_samples_leaf', 4),
            criterion=self.best_params.get('estimator__criterion', 'gini'),
            max_features=self.best_params.get('estimator__max_features', 'log2'),
            class_weight=self.best_params.get('estimator__class_weight', 'balanced'),
            max_leaf_nodes=self.best_params.get('estimator__max_leaf_nodes', 20),
            min_impurity_decrease=self.best_params.get('estimator__min_impurity_decrease', 0.0),
            bootstrap=self.best_params.get('estimator__bootstrap', True),
            ccp_alpha=self.best_params.get('estimator__ccp_alpha', 0.0),
            random_state=self.seed,
            n_jobs=self.n_jobs
        )
        model_ovr = OneVsRestClassifier(base_rf)
        model_ovr.fit(X_train, y_train)
        calibrator = CalibratedClassifierCV(model_ovr, method='isotonic', cv=3, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator
        y_pred_proba = calibrated_model.predict_proba(X_test)
        y_pred = calibrated_model.predict(X_test)
        f1_val = f1_score(y_test, y_pred, average='samples', zero_division=0)
        pr_auc = average_precision_score(y_test, y_pred_proba, average='samples')
        # Correção: calcular ROC AUC para cada rótulo e fazer a média
        try:
            roc_vals = []
            for i in range(y_test.shape[1]):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
                roc_vals.append(auc(fpr, tpr))
            roc_val = np.mean(roc_vals)
        except ValueError:
            roc_val = 0.0
        return roc_val, f1_val, pr_auc, self.best_params, calibrated_model, X_test, y_test

class ProteinEmbeddingGenerator:
    """
    Class to generate protein embeddings using Word2Vec.
    """
    def __init__(self, sequences_path: str, table_data: pd.DataFrame = None, aggregation_method: str = 'none'):
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
        self.aggregation_method = aggregation_method
        self.min_kmers = None
    def generate_embeddings(self, k: int = 3, step_size: int = 1,
                            word2vec_model_path: str = "word2vec_model.bin",
                            model_dir: str = None, min_kmers: int = None,
                            save_min_kmers: bool = False) -> None:
        if model_dir:
            word2vec_model_full_path = os.path.join(model_dir, word2vec_model_path)
        else:
            word2vec_model_full_path = word2vec_model_path
        if os.path.exists(word2vec_model_full_path):
            logging.info(f"Word2Vec model found at {word2vec_model_full_path}. Loading the model.")
            model = Word2Vec.load(word2vec_model_full_path)
            self.models['global'] = model
        else:
            logging.info("Word2Vec model not found. Training a new model.")
            kmer_groups = {}
            all_kmers = []
            kmers_counts = []
            for record in self.alignment:
                sequence = str(record.seq)
                seq_len = len(sequence)
                protein_accession_alignment = record.id.split()[0]
                if self.table_data is not None:
                    matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                    matching_info = self.table_data[matching_rows]
                    if matching_info.empty:
                        logging.warning(f"No matching table data for {protein_accession_alignment}")
                        continue
                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]
                else:
                    target_variable = None
                    associated_variable = None
                logging.info(f"Processing {protein_accession_alignment} with sequence length {seq_len}")
                if seq_len < k:
                    logging.warning(f"Sequence too short for {protein_accession_alignment}. Length: {seq_len}")
                    continue
                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]
                if not kmers:
                    logging.warning(f"No valid k-mers for {protein_accession_alignment}")
                    continue
                all_kmers.append(kmers)
                kmers_counts.append(len(kmers))
                embedding_info = {
                    'protein_accession': protein_accession_alignment,
                    'target_variable': target_variable,
                    'associated_variable': associated_variable,
                    'kmers': kmers
                }
                kmer_groups[protein_accession_alignment] = embedding_info
            if not kmers_counts:
                logging.error("No k-mers were collected. Please check your sequences and k-mer parameters.")
                sys.exit(1)
            if min_kmers is not None:
                self.min_kmers = min_kmers
                logging.info(f"Using provided min_kmers: {self.min_kmers}")
            else:
                self.min_kmers = min(kmers_counts)
                logging.info(f"Minimum number of k-mers in any sequence: {self.min_kmers}")
            if save_min_kmers and model_dir:
                min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
                with open(min_kmers_path, 'w') as f:
                    f.write(str(self.min_kmers))
                logging.info(f"min_kmers saved at {min_kmers_path}")
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=390,
                window=window,
                min_count=1,
                workers=workers,
                sg=1,
                hs=1,
                negative=0,
                epochs=epochs,
                seed=SEED
            )
            if model_dir:
                os.makedirs(os.path.dirname(word2vec_model_full_path), exist_ok=True)
            model.save(word2vec_model_full_path)
            self.models['global'] = model
            logging.info(f"Word2Vec model saved at {word2vec_model_full_path}")
        # Regenera o dicionário de k-mer groups para criar os embeddings
        kmer_groups = {}
        kmers_counts = []
        all_kmers = []
        for record in self.alignment:
            sequence_id = record.id.split()[0]
            sequence = str(record.seq)
            if self.table_data is not None:
                matching_rows = self.table_data['Protein.accession'].str.split().str[0] == sequence_id
                matching_info = self.table_data[matching_rows]
                if matching_info.empty:
                    logging.warning(f"No matching table data for {sequence_id}")
                    continue
                target_variable = matching_info['Target variable'].values[0]
                associated_variable = matching_info['Associated variable'].values[0]
            else:
                target_variable = None
                associated_variable = None
            kmers = [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, step_size)]
            kmers = [kmer for kmer in kmers if kmer.count('-') < k]
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
        if not kmers_counts:
            logging.error("No k-mers were collected. Please check your sequences and k-mer parameters.")
            sys.exit(1)
        if min_kmers is not None:
            self.min_kmers = min_kmers
            logging.info(f"Using provided min_kmers: {self.min_kmers}")
        else:
            self.min_kmers = min(kmers_counts)
            logging.info(f"Minimum number of k-mers in any sequence: {self.min_kmers}")
        for record in self.alignment:
            sequence_id = record.id.split()[0]
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
            selected_kmers = kmers_for_protein[:self.min_kmers]
            if len(selected_kmers) < self.min_kmers:
                padding = [np.zeros(self.models['global'].vector_size)] * (self.min_kmers - len(selected_kmers))
                selected_kmers.extend(padding)
            selected_embeddings = [self.models['global'].wv[kmer] if kmer in self.models['global'].wv 
                                     else np.zeros(self.models['global'].vector_size) for kmer in selected_kmers]
            if self.aggregation_method == 'none':
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                embedding_concatenated = np.mean(selected_embeddings, axis=0)
            else:
                logging.warning(f"Unknown aggregation method '{self.aggregation_method}'. Using concatenation.")
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': embedding_concatenated,
                'target_variable': embedding_info.get('target_variable'),
                'associated_variable': embedding_info.get('associated_variable')
            })
            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {embedding_concatenated.shape}")
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Inconsistent embedding shapes detected: {embedding_shapes}")
            raise ValueError("Embeddings have inconsistent shapes.")
        else:
            logging.info(f"All embeddings have the shape: {embedding_shapes.pop()}")
        scaler_full_path = os.path.join(model_dir, 'scaler_associated.pkl') if model_dir else 'scaler_associated.pkl'
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler found at {scaler_full_path}. Loading the scaler.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler not found. Training a new scaler.")
            scaler = StandardScaler().fit(embeddings_array_train)
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler saved at {scaler_full_path}")
    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        embeddings = []
        labels_raw = []
        for embedding_info in self.embeddings:
            embeddings.append(embedding_info['embedding'])
            label_str = embedding_info[label_type]
            if isinstance(label_str, str):
                split_labels = [lbl.strip() for lbl in label_str.split(',')]
                try:
                    split_labels = [int(lbl) for lbl in split_labels]
                except ValueError:
                    pass
                labels_raw.append(split_labels)
            else:
                if not label_str:
                    labels_raw.append([])
                else:
                    try:
                        labels_raw.append([int(label_str)])
                    except ValueError:
                        labels_raw.append([str(label_str)])
        embeddings_array = np.array(embeddings)
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(labels_raw)
        self.mlb_classes_ = mlb.classes_
        return embeddings_array, Y

def compute_perplexity(n_samples: int) -> int:
    return max(5, min(50, n_samples // 100))

def plot_dual_umap(train_embeddings: np.ndarray, train_labels: list, train_protein_ids: list,
                   predict_embeddings: np.ndarray, predict_labels: list, predict_protein_ids: list,
                   output_dir: str) -> tuple:
    umap_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_train_result = umap_train.fit_transform(train_embeddings)
    umap_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_predict_result = umap_predict.fit_transform(predict_embeddings)
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter3d(
        x=umap_train_result[:, 0],
        y=umap_train_result[:, 1],
        z=umap_train_result[:, 2],
        mode='markers',
        marker=dict(size=5, color=train_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {lbl}" for pid, lbl in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Training Data'
    ))
    fig_train.update_layout(
        title='UMAP 3D: Training Data',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    fig_predict = go.Figure()
    fig_predict.add_trace(go.Scatter3d(
        x=umap_predict_result[:, 0],
        y=umap_predict_result[:, 1],
        z=umap_predict_result[:, 2],
        mode='markers',
        marker=dict(size=5, color=predict_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {lbl}" for pid, lbl in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Predictions'
    ))
    fig_predict.update_layout(
        title='UMAP 3D: Predictions',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    umap_train_html = os.path.join(output_dir, "umap_train_3d.html")
    umap_predict_html = os.path.join(output_dir, "umap_predict_3d.html")
    pio.write_html(fig_train, file=umap_train_html, auto_open=False)
    pio.write_html(fig_predict, file=umap_predict_html, auto_open=False)
    logging.info(f"UMAP Training plot saved as {umap_train_html}")
    logging.info(f"UMAP Predictions plot saved as {umap_predict_html}")
    return fig_train, fig_predict

def plot_predictions_scatterplot_custom(results: dict, output_path: str, top_n: int = 1) -> None:
    protein_specificities = {}
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No ranking data associated with the protein {seq_id}. Skipping...")
            continue
        top_category_with_confidence, top_sum, top_two_categories = format_and_sum_probabilities(associated_rankings)
        if top_category_with_confidence is None:
            logging.warning(f"No valid formatting for protein {seq_id}. Skipping...")
            continue
        category = top_category_with_confidence.split(" (")[0]
        protein_specificities[seq_id] = {'top_category': category, 'confidence': top_sum}
    if not protein_specificities:
        logging.warning("No data available to plot the scatter plot.")
        return
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))
    x_values = list(range(4, 19))
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]
        for spec in specificities:
            ax.scatter(spec, y, color='#1E3A8A', edgecolors='black', linewidth=0.5, s=100, label='_nolegend_')
        if len(specificities) > 1:
            ax.plot(specificities, [y] * len(specificities), color='#1E3A8A', linestyle='-', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Specificity (C4 to C18)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Scatter Plot of Predictions for New Sequences (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='white')
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='white')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='white')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Scatter plot saved at {output_path}")

def adjust_predictions_global(predicted_proba: np.ndarray, method: str = 'normalize', alpha: float = 1.0) -> np.ndarray:
    if method == 'normalize':
        logging.info("Normalizing predicted probabilities.")
        sums = predicted_proba.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1e-9
        adjusted_proba = predicted_proba / sums
    elif method == 'smoothing':
        logging.info(f"Applying smoothing to predicted probabilities with alpha={alpha}.")
        sums = predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1]
        adjusted_proba = (predicted_proba + alpha) / sums
    elif method == 'none':
        logging.info("No adjustment applied to predicted probabilities.")
        adjusted_proba = predicted_proba.copy()
    else:
        logging.warning(f"Unknown adjustment method '{method}'. No adjustment will be applied.")
        adjusted_proba = predicted_proba.copy()
    return adjusted_proba

def main(args: argparse.Namespace) -> None:
    model_dir = args.model_dir
    total_steps = 5
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()
    # STEP 1: Training the Model
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Training sequences not aligned. Realigning with MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Aligned training file found or sequences already aligned: {train_alignment_path}")
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Training table data loaded successfully.")
    current_step += 1
    progress_bar.progress(min(current_step/total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step/total_steps*100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    protein_embedding_train = ProteinEmbeddingGenerator(train_alignment_path, table_data=train_table_data, aggregation_method=args.aggregation_method)
    protein_embedding_train.generate_embeddings(k=args.kmer_size, step_size=args.step_size, word2vec_model_path=args.word2vec_model, model_dir=model_dir, save_min_kmers=True)
    logging.info(f"Train embeddings generated: {len(protein_embedding_train.embeddings)}")
    min_kmers = protein_embedding_train.min_kmers
    protein_ids_associated = [e['protein_accession'] for e in protein_embedding_train.embeddings]
    var_assoc_associated = [e['associated_variable'] for e in protein_embedding_train.embeddings]
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"X_associated shape: {X_associated.shape}, y_associated shape: {y_associated.shape}")
    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info("Scaler for X_associated saved.")
    X_associated_scaled = scaler_associated.transform(X_associated)
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')
    current_step += 1
    progress_bar.progress(min(current_step/total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step/total_steps*100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Loaded existing calibrated model from {calibrated_model_associated_full_path}")
    else:
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(X_associated_scaled, y_associated,
                                                                  protein_ids=protein_ids_associated,
                                                                  var_assoc=var_assoc_associated,
                                                                  model_name_prefix='associated',
                                                                  model_dir=model_dir,
                                                                  min_kmers=min_kmers)
        support_model_associated.plot_learning_curve(args.learning_curve_associated)
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated model saved at {calibrated_model_associated_full_path}")
        best_roc, best_f1, best_pr_auc, best_params, best_calibrated_model, X_test_, y_test_ = support_model_associated.test_best_RF(X_associated_scaled, y_associated)
        logging.info(f"Best ROC (samples average): {best_roc}")
        logging.info(f"Best F1 (samples average): {best_f1}")
        logging.info(f"Best Precision-Recall AUC (samples average): {best_pr_auc}")
        logging.info(f"Best Parameters: {best_params}")
        joblib.dump(best_calibrated_model, rf_model_associated_full_path)
        logging.info(f"Random Forest model (multilabel) saved at {rf_model_associated_full_path}")
        y_pred_proba_test = best_calibrated_model.predict_proba(X_test_)
        if hasattr(protein_embedding_train, 'mlb_classes_'):
            mlb_classes = protein_embedding_train.mlb_classes_
        else:
            mlb_classes = [f"Label_{i}" for i in range(y_pred_proba_test.shape[1])]
        plot_roc_curve_global(y_test_, y_pred_proba_test, title="ROC Curve Multilabel (Associated)", save_as=args.roc_curve_associated, classes=mlb_classes)
    current_step += 1
    progress_bar.progress(min(current_step/total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step/total_steps*100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    # STEP 2: Classifying New Sequences
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers loaded: {min_kmers_loaded}")
    else:
        logging.error("min_kmers file not found. Ensure training was completed.")
        sys.exit(1)
    predict_alignment_path = args.predict_fasta
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Prediction sequences not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Aligned predict file found: {predict_alignment_path}")
    current_step += 1
    progress_bar.progress(min(current_step/total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step/total_steps*100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    protein_embedding_predict = ProteinEmbeddingGenerator(predict_alignment_path, table_data=None, aggregation_method=args.aggregation_method)
    protein_embedding_predict.generate_embeddings(k=args.kmer_size, step_size=args.step_size, word2vec_model_path=args.word2vec_model, model_dir=model_dir, min_kmers=min_kmers_loaded)
    logging.info(f"Prediction embeddings: {len(protein_embedding_predict.embeddings)}")
    X_predict = np.array([ent['embedding'] for ent in protein_embedding_predict.embeddings])
    scaler_path_ = os.path.join(model_dir, 'scaler_associated.pkl')
    if os.path.exists(scaler_path_):
        scaler_associated = joblib.load(scaler_path_)
        logging.info("Scaler loaded for prediction.")
    else:
        logging.error("Scaler not found, training incomplete.")
        sys.exit(1)
    X_predict_scaled = scaler_associated.transform(X_predict)
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled)
    results = {}
    for ent, pred, ranking in zip(protein_embedding_predict.embeddings, predictions_associated, rankings_associated):
        seq_id = ent['protein_accession']
        results[seq_id] = {"associated_prediction": pred, "associated_ranking": ranking}
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tAssociated_Prediction\tAssociated_Ranking\n")
        for seq_id, info in results.items():
            f.write(f"{seq_id}\t{info['associated_prediction']}\t{'; '.join(info['associated_ranking'])}\n")
            logging.info(f"{seq_id} => {info['associated_prediction']}, {info['associated_ranking']}")
    logging.info("Generating scatter plot of predictions for new sequences...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatter plot saved at {args.scatterplot_output}")
    logging.info("Generating Dual UMAP plots for training/prediction data...")
    if hasattr(protein_embedding_train, 'mlb_classes_'):
        mlb_classes = protein_embedding_train.mlb_classes_
    else:
        mlb_classes = [f"Label_{i}" for i in range(y_associated.shape[1])]
    train_labels_str = []
    for row in y_associated:
        active_labels = [mlb_classes[i] for i, val in enumerate(row) if val == 1]
        if not active_labels:
            train_labels_str.append("NoLabel")
        else:
            train_labels_str.append(",".join(active_labels))
    predict_labels_str = []
    for row in predictions_associated:
        active_labels = [mlb_classes[i] for i, val in enumerate(row) if val == 1]
        if not active_labels:
            predict_labels_str.append("NoLabel")
        else:
            predict_labels_str.append(",".join(active_labels))
    train_protein_ids = protein_ids_associated
    predict_protein_ids = [ent['protein_accession'] for ent in protein_embedding_predict.embeddings]
    plot_dual_umap(train_embeddings=X_associated_scaled, train_labels=train_labels_str, train_protein_ids=train_protein_ids,
                   predict_embeddings=X_predict_scaled, predict_labels=predict_labels_str, predict_protein_ids=predict_protein_ids,
                   output_dir=model_dir)
    logging.info("Dual UMAP done.")
    fig_tsne_train, fig_tsne_predict = plot_dual_tsne(train_embeddings=X_associated_scaled, train_labels=train_labels_str,
                                                      train_protein_ids=train_protein_ids, predict_embeddings=X_predict_scaled,
                                                      predict_labels=predict_labels_str, predict_protein_ids=predict_protein_ids,
                                                      output_dir=model_dir)
    logging.info("Dual TSNE done.")
    current_step += 1
    progress_bar.progress(min(current_step/total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step/total_steps*100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    st.success("Analysis completed successfully!")
    st.header("Scatter Plot of Predictions")
    scatterplot_path = args.scatterplot_output
    if os.path.exists(scatterplot_path):
        st.image(scatterplot_path, use_column_width=True)
    else:
        st.error(f"Scatter plot not found at {scatterplot_path}")
    formatted_results = []
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            continue
        top_specificity, confidence, top_two_specificities = format_and_sum_probabilities(associated_rankings)
        if top_specificity is None:
            continue
        formatted_results.append([seq_id, top_specificity, f"{confidence:.2f}%", "; ".join(top_two_specificities)])
    headers = ["Query Name", "SS Prediction Specificity", "Prediction Confidence", "Top 2 Specificities"]
    df_results = pd.DataFrame(formatted_results, columns=headers)
    def highlight_table(df):
        return df.style.set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#1E3A8A'),
                       ('color', 'white'),
                       ('border', '1px solid white'),
                       ('font-weight', 'bold'),
                       ('text-align', 'center')]},
            {'selector': 'td',
             'props': [('background-color', '#0B3C5D'),
                       ('color', 'white'),
                       ('border', '1px solid white'),
                       ('text-align', 'center'),
                       ('font-family', 'Arial'),
                       ('font-size', '12px')]},
            {'selector': 'tr:nth-child(even) td',
             'props': [('background-color', '#145B9C')]},
            {'selector': 'tr:hover td',
             'props': [('background-color', '#0D4F8B')]}
        ])
    styled_df = highlight_table(df_results)
    html = styled_df.to_html(index=False, escape=False)
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #1E3A8A;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .stButton > button:hover {
            background-color: #0B3C5D;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        .dataframe-container {
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header("Formatted Results")
    st.markdown(f"""<div class="dataframe-container">{html}</div>""", unsafe_allow_html=True)
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name='results.csv',
        mime='text/csv',
    )
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
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for folder_name, subfolders, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, output_dir))
    zip_buffer.seek(0)
    st.header("Download All Results")
    st.download_button(
        label="Download All Results as results.zip",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Resultados salvos em {args.excel_output}")
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved at {args.formatted_results_table}")
    umap_similarity_path = os.path.join(output_dir, "umap_similarity_3D.html")
    dual_umap_train_path = os.path.join(output_dir, "umap_train_3d.html")
    dual_umap_predict_path = os.path.join(output_dir, "umap_predict_3d.html")
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")

# --- Streamlit UI ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B3C5D;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stFileUploader,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stSlider {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] div {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stFileUploader,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stSlider {
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stCheckbox input[type="checkbox"] + div:first-of-type,
    [data-testid="stSidebar"] .stRadio input[type="radio"] + div:first-of-type {
        background-color: #1E3A8A !important;
    }
    [data-testid="stSidebar"] .stSlider > div:first-of-type {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider .st-bo {
        background-color: #1E3A8A !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    div[role="alert"] p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return ""

image_path = "./images/faal.png"
image_base64 = get_base64_image(image_path)
st.markdown(
    f"""
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 30px; 
         background: linear-gradient(to bottom, #f9f9f9, #ffffff); border-radius: 15px; 
         border: 2px solid #dddddd; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); 
         position: relative;">
        <p style="color: black; font-size: 1.5em; font-weight: bold; margin: 0;">
            FAALPred: Predicting Fatty Acyl Chain Specificities in Fatty Acyl-AMP Ligases (FAALs) 
            Using Integrated Approaches of Neural Networks, Bioinformatics, and Machine Learning
        </p>
        <p style="color: #2c3e50; font-size: 1.2em; font-weight: normal; margin-top: 10px;">
            Anne Liong, Leandro de Mattos Pereira, and Pedro Leão
        </p>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8;">
            <strong>FAALPred</strong> is a comprehensive bioinformatics tool designed to 
            predict the fatty acid chain length specificity of substrates, ranging from C4 to C18.
        </p>
        <h5 style="color: #2c3e50; font-size: 20px; font-weight: bold; margin-top: 25px;">ABSTRACT</h5>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8; text-align: justify;">
            Fatty Acyl-AMP Ligases (FAALs), identified by Zhang et al. (2011), activate 
            fatty acids of varying lengths for the biosynthesis of natural products. 
            These substrates enable the production of compounds such as nocuolin 
            (<em>Nodularia sp.</em>, Martins et al., 2022) and sulfolipid-1 
            (<em>Mycobacterium tuberculosis</em>, Yan et al., 2023), with applications in 
            cancer and tuberculosis treatment (Kurt et al., 2017; Gilmore et al., 2012). 
            Dr. Pedro Leão and his team identified several of these natural products in 
            cyanobacteria (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" 
            style="color: #3498db; text-decoration: none;">visit here</a>), and FAALPred 
            classifies FAALs by their substrate specificity.
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{image_base64}" alt="FAAL Domain" 
                 style="width: auto; height: 120px; object-fit: contain;">
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>FAAL Domain of Synechococcus sp. PCC7002, link: 
                <a href="https://www.rcsb.org/structure/7R7F" target="_blank" 
                   style="color: #3498db; text-decoration: none;">
                   https://www.rcsb.org/structure/7R7F
                </a></em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Input Parameters")

def save_uploaded_file(uploaded_file, save_path: str) -> str:
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path

use_default_train = st.sidebar.checkbox("Use Default Training Data", value=True)
if not use_default_train:
    train_fasta_file = st.sidebar.file_uploader("Upload Training FASTA File", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("Upload Training Table File (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Upload Prediction FASTA File", type=["fasta", "fa", "fna"])
kmer_size = st.sidebar.number_input("K-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)
aggregation_method = st.sidebar.selectbox("Aggregation Method", options=['none', 'mean'], index=0)
st.sidebar.header("Customize Word2Vec Parameters")
custom_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)
if custom_word2vec:
    window = st.sidebar.number_input("Window Size", min_value=5, max_value=20, value=10, step=5)
    workers = st.sidebar.number_input("Workers", min_value=1, max_value=112, value=8, step=8)
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=2500, value=2500, step=100)
else:
    window = 10
    workers = 48
    epochs = 2500
model_dir = create_unique_model_directory("results", aggregation_method)
output_dir = model_dir
if st.sidebar.button("Run Analysis"):
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"
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
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(output_dir, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Please upload a FASTA file for prediction.")
        st.stop()
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
        scaler="scaler_associated.pkl",
        model_dir=model_dir,
    )
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(args)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")

def load_and_resize_image_with_dpi(image_path: str, base_width: int, dpi: int = 300) -> Image.Image:
    try:
        image = Image.open(image_path)
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

image_dir = "images"
image_paths = [
    os.path.join(image_dir, "lab_logo.png"),
    os.path.join(image_dir, "ciimar.png"),
    os.path.join(image_dir, "faal_pred_logo.png"),
    os.path.join(image_dir, "bbf4.png"),
    os.path.join(image_dir, "google.png"),
    os.path.join(image_dir, "uniao.png"),
]
images = [load_and_resize_image_with_dpi(path, base_width=150, dpi=300) for path in image_paths]
encoded_images = [encode_image(img) for img in images if img is not None]
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
    unsafe_allow_html=True
)
footer_html = """
<div class="support-text">Supported by:</div>
<div class="footer-container">
    {}
</div>
<div class="footer-text">
    CIIMAR - Pedro Leão @CNP - 2024 - All rights reserved.
</div>
"""
img_tags = "".join(f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images)
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)
