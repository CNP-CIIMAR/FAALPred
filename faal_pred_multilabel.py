import logging
import os
import sys
import subprocess
import random
import zipfile
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from io import BytesIO
import shutil
import time
import argparse
from PIL import Image
import base64

from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MafftCommandline

from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt
from matplotlib import ticker
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure
import streamlit as st

# Scikit-learn and imbalanced-learn imports
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_curve, auc, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tabulate import tabulate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors

# UMAP imports
import umap.umap_ as umap
import umap

# --------------------------------------------
# Global Configuration and Reproducibility Settings
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)

# Streamlit Configuration
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------
# Auxiliary Functions

def are_sequences_aligned(fasta_file_path: str) -> bool:
    """Check if sequences in the FASTA file are aligned."""
    lengths = set()
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1

def create_unique_model_directory(base_directory: str, aggregation_method: str) -> str:
    """Create a unique model directory based on the aggregation method."""
    model_directory = os.path.join(base_directory, f"models_{aggregation_method}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    return model_directory

def realign_sequences_with_mafft(input_path: str, output_path: str, threads: int = 8) -> None:
    """Realign sequences using the MAFFT command line tool."""
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Realigned sequences saved at {output_path}")
    except subprocess.CalledProcessError as error_exception:
        logging.error(f"Error executing MAFFT: {error_exception.stderr.decode()}")
        sys.exit(1)

def perform_clustering(data: np.ndarray, method: str = "DBSCAN", eps: float = 0.5,
                       min_samples: int = 5, num_clusters: int = 3) -> np.ndarray:
    """Perform clustering using either DBSCAN or K-Means."""
    if method == "DBSCAN":
        from sklearn.cluster import DBSCAN
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "K-Means":
        from sklearn.cluster import KMeans
        clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
    else:
        raise ValueError(f"Invalid clustering method: {method}")
    labels = clustering_model.fit_predict(data)
    return labels

def plot_dual_tsne(train_embeddings: np.ndarray, train_labels: list, train_protein_ids: list,
                     predict_embeddings: np.ndarray, predict_labels: list, predict_protein_ids: list,
                     output_directory: str) -> tuple:
    """Perform 3D t-SNE visualization for training and prediction embeddings."""
    from sklearn.manifold import TSNE
    tsne_model_train = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_result_train = tsne_model_train.fit_transform(train_embeddings)
    tsne_model_predict = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_result_predict = tsne_model_predict.fit_transform(predict_embeddings)
    
    unique_train_labels = sorted(list(set(train_labels)))
    train_color_map = px.colors.qualitative.Dark24
    train_color_dict = {label: train_color_map[i % len(train_color_map)] for i, label in enumerate(unique_train_labels)}
    unique_predict_labels = sorted(list(set(predict_labels)))
    predict_color_map = px.colors.qualitative.Light24
    predict_color_dict = {label: predict_color_map[i % len(predict_color_map)] for i, label in enumerate(unique_predict_labels)}
    
    train_colors = [train_color_dict.get(label, 'gray') for label in train_labels]
    predict_colors = [predict_color_dict.get(label, 'gray') for label in predict_labels]
    
    train_figure = go.Figure()
    train_figure.add_trace(go.Scatter3d(
        x=tsne_result_train[:, 0],
        y=tsne_result_train[:, 1],
        z=tsne_result_train[:, 2],
        mode='markers',
        marker=dict(size=5, color=train_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {label}" for pid, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Training Data'
    ))
    train_figure.update_layout(
        title='t-SNE 3D: Training Data',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    
    predict_figure = go.Figure()
    predict_figure.add_trace(go.Scatter3d(
        x=tsne_result_predict[:, 0],
        y=tsne_result_predict[:, 1],
        z=tsne_result_predict[:, 2],
        mode='markers',
        marker=dict(size=5, color=predict_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {label}" for pid, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Predictions'
    ))
    predict_figure.update_layout(
        title='t-SNE 3D: Predictions',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    
    train_tsne_html = os.path.join(output_directory, "tsne_train_3d.html")
    predict_tsne_html = os.path.join(output_directory, "tsne_predict_3d.html")
    pio.write_html(train_figure, file=train_tsne_html, auto_open=False)
    pio.write_html(predict_figure, file=predict_tsne_html, auto_open=False)
    logging.info(f"t-SNE training plot saved as {train_tsne_html}")
    logging.info(f"t-SNE predictions plot saved as {predict_tsne_html}")
    return train_figure, predict_figure

def plot_global_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          title: str, save_as: str = None, classes: list = None) -> None:
    """Plot a global ROC curve (handles binary and multiclass cases)."""
    line_width = 2
    unique_classes = np.unique(y_true)
    plt.figure()
    if len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=line_width, label='ROC Curve (area = %0.2f)' % roc_auc)
    else:
        from sklearn.preprocessing import label_binarize
        y_binarized = label_binarize(y_true, classes=unique_classes)
        num_classes = y_binarized.shape[1]
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        for i, color in zip(range(num_classes), colors):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_binarized[:, i], y_pred_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            class_label = classes[i] if classes is not None else unique_classes[i]
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=line_width,
                     label=f'ROC Curve for class {class_label} (area = {roc_auc_dict[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.title(title, color='white')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')
    plt.close()

def get_global_class_rankings(model, X: np.ndarray, mlb_classes: list = None) -> list:
    """Compute class rankings from predicted probabilities.
    If mlb_classes is provided, use those as the labels."""
    y_pred_proba = model.predict_proba(X)
    if mlb_classes is not None:
        class_labels = mlb_classes
    elif hasattr(model, "classes_"):
        class_labels = list(model.classes_)
    else:
        class_labels = [f"Label_{i}" for i in range(y_pred_proba.shape[1])]
    rankings = []
    for sample_proba in y_pred_proba:
        sorted_pairs = sorted(zip(class_labels, sample_proba), key=lambda x: x[1], reverse=True)
        formatted_ranking = [f"{label}: {probability*100:.2f}%" for label, probability in sorted_pairs]
        rankings.append(formatted_ranking)
    return rankings

def calculate_global_roc_values(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Calculate ROC AUC for each label individually and return a DataFrame of AUC values."""
    num_classes = len(np.unique(y_test))
    y_pred_proba = model.predict_proba(X_test)
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc_dict[i] = auc(fpr, tpr)
        logging.info(f"For class {i}:")
        logging.info(f"FPR: {fpr}")
        logging.info(f"TPR: {tpr}")
        logging.info(f"ROC AUC: {roc_auc_dict[i]}")
        logging.info("--------------------------")
    roc_dataframe = pd.DataFrame(list(roc_auc_dict.items()), columns=['Class', 'ROC AUC'])
    return roc_dataframe

def visualize_latent_space_with_similarity(X_original: np.ndarray, X_synthetic: np.ndarray, 
                                             y_original: np.ndarray, y_synthetic: np.ndarray, 
                                             original_protein_ids: list, synthetic_protein_ids: list, 
                                             original_associated_variables: list, synthetic_associated_variables: list, 
                                             output_directory: str = None) -> Figure:
    """Visualize latent space with similarity overlay using UMAP."""
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.vstack([y_original, y_synthetic])
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    X_transformed = umap_reducer.fit_transform(X_combined)
    X_transformed_original = X_transformed[:len(X_original)]
    X_transformed_synthetic = X_transformed[len(X_original):]
    similarities = cosine_similarity(X_synthetic, X_original)
    max_similarities = similarities.max(axis=1)
    closest_indices = similarities.argmax(axis=1)
    
    original_df = pd.DataFrame({
        'x': X_transformed_original[:, 0],
        'y': X_transformed_original[:, 1],
        'z': X_transformed_original[:, 2],
        'Protein ID': original_protein_ids,
        'Associated Variable': original_associated_variables,
        'Type': 'Original'
    })
    synthetic_df = pd.DataFrame({
        'x': X_transformed_synthetic[:, 0],
        'y': X_transformed_synthetic[:, 1],
        'z': X_transformed_synthetic[:, 2],
        'Protein ID': synthetic_protein_ids,
        'Associated Variable': synthetic_associated_variables,
        'Similarity': max_similarities,
        'Closest Protein': [original_protein_ids[idx] for idx in closest_indices],
        'Closest Variable': [original_associated_variables[idx] for idx in closest_indices],
        'Type': 'Synthetic'
    })
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(
        x=original_df['x'],
        y=original_df['y'],
        z=original_df['z'],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.7),
        name='Original',
        text=original_df.apply(lambda row: f"Protein ID: {row['Protein ID']}<br>Associated Variable: {row['Associated Variable']}", axis=1),
        hoverinfo='text'
    ))
    figure.add_trace(go.Scatter3d(
        x=synthetic_df['x'],
        y=synthetic_df['y'],
        z=synthetic_df['z'],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.7),
        name='Synthetic',
        text=synthetic_df.apply(lambda row: f"Protein ID: {row['Protein ID']}<br>Associated Variable: {row['Associated Variable']}<br>Similarity: {row['Similarity']:.4f}<br>Closest Protein: {row['Closest Protein']}<br>Closest Variable: {row['Closest Variable']}", axis=1),
        hoverinfo='text'
    ))
    figure.update_layout(
        title="Latent Space Visualization with Similarity (UMAP 3D)",
        scene=dict(xaxis_title="UMAP Dimension 1",
                   yaxis_title="UMAP Dimension 2",
                   zaxis_title="UMAP Dimension 3"),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )
    if output_directory:
        umap_similarity_path = os.path.join(output_directory, "umap_similarity_3D.html")
        figure.write_html(umap_similarity_path)
        logging.info(f"UMAP plot saved at {umap_similarity_path}")
    return figure

def format_and_sum_probabilities(associated_rankings: list) -> tuple:
    """Format rankings for associated variables by summing probabilities across predefined categories."""
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
    for ranking in associated_rankings:
        try:
            probability = float(ranking.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Error processing ranking string: {ranking}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in ranking for pattern in patterns):
                category_sums[category] += probability
    if not category_sums:
        return None, None, None
    top_category, top_sum = max(category_sums.items(), key=lambda item: item[1])
    sorted_categories = sorted(category_sums.items(), key=lambda item: item[1], reverse=True)
    top_two = sorted_categories[:2] if len(sorted_categories) >= 2 else sorted_categories
    top_two_categories = [f"{cat} ({prob:.2f}%)" for cat, prob in top_two]
    top_category_with_confidence = f"{top_category} ({top_sum:.2f}%)"
    return top_category_with_confidence, top_sum, top_two_categories

# --------------------------------------------
# Multi-Label SMOTE Class (Full Definition)

class MultiLabelSMOTE:
    """
    Class to implement the Multi-Label Synthetic Minority Over-sampling Technique (MLSMOTE).
    This implementation performs random oversampling for minority classes and generates synthetic samples via interpolation.
    """
    def __init__(self, num_neighbors: int = 5, random_state: int = None, min_samples: int = 5):
        self.num_neighbors = num_neighbors
        self.random_state = random_state
        self.min_samples = min_samples
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
    def fit_resample(self, X, y, num_samples: int):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            X = X.reset_index(drop=True)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        else:
            y = y.reset_index(drop=True)
        X_original = X.copy()
        y_original = y.copy()
        for column in y.columns:
            column_count = y[column].sum()
            threshold = y.shape[0] / len(y.columns)
            if column_count < threshold:
                needed = self.min_samples - int(column_count)
                if needed > 0:
                    positive_indices = y.index[y[column] == 1].tolist()
                    if len(positive_indices) > 0:
                        oversampled_indices = np.random.choice(positive_indices, size=needed, replace=True)
                        X_oversampled = X.loc[oversampled_indices]
                        y_oversampled = y.loc[oversampled_indices]
                        y_oversampled[column] = 1
                        X = pd.concat([X, X_oversampled], ignore_index=True)
                        y = pd.concat([y, y_oversampled], ignore_index=True)
                        print(f"[DEBUG] Column {column}: Oversampled with {needed} samples.")
        X_sub, y_sub = self.get_minority_instance(X, y)
        print(f"[DEBUG] After identifying minority instances: X_sub={X_sub.shape}, y_sub={y_sub.shape}")
        if X_sub.shape[0] < self.min_samples:
            X_extra, y_extra = self.random_oversample(X_sub, y_sub, self.min_samples)
            X_sub = pd.concat([X_sub, X_extra], axis=0, ignore_index=True)
            y_sub = pd.concat([y_sub, y_extra], axis=0, ignore_index=True)
            print(f"[DEBUG] After random oversampling: X_sub={X_sub.shape}, y_sub={y_sub.shape}")
        neighbors_indices = self.find_nearest_neighbors(X_sub)
        print("[DEBUG] Nearest neighbors found for minority samples.")
        new_X, new_y = self.generate_synthetic_samples(X_sub, y_sub, neighbors_indices, num_samples)
        print(f"[DEBUG] New synthetic samples generated: new_X={new_X.shape}, new_y={new_y.shape}")
        X_original_arr = X_original.to_numpy()
        y_original_arr = y_original.to_numpy()
        X_new_arr = new_X.to_numpy()
        y_new_arr = new_y.to_numpy()
        X_combined = np.vstack([X_original_arr, X_new_arr])
        y_combined = np.vstack([y_original_arr, y_new_arr])
        print(f"[DEBUG] After concatenation: X_combined={X_combined.shape}, y_combined={y_combined.shape}")
        return {"original": (X_original_arr, y_original_arr),
                "synthetic": (X_new_arr, y_new_arr),
                "combined": (X_combined, y_combined)}
    def get_tail_label(self, y):
        columns = y.columns
        counts = np.array([y[column].sum() for column in columns], dtype=float)
        counts[counts == 0] = 1e-9
        ratios = counts.max() / counts
        avg_ratio = ratios.mean()
        tail_labels = [columns[i] for i in range(len(columns)) if ratios[i] > avg_ratio]
        print(f"[DEBUG] Minority classes identified: {tail_labels}")
        return tail_labels
    def get_index(self, y):
        tail_labels = self.get_tail_label(y)
        indices = set()
        for tl in tail_labels:
            indices = indices.union(set(y.index[y[tl] == 1]))
        if len(indices) == 0:
            indices = set(y.index)
        print(f"[DEBUG] Total minority samples: {len(indices)}")
        return list(indices)
    def get_minority_instance(self, X, y):
        indices = self.get_index(y)
        X_sub = X.loc[indices].reset_index(drop=True)
        y_sub = y.loc[indices].reset_index(drop=True)
        return X_sub, y_sub
    def random_oversample(self, X_sub, y_sub, target_samples):
        current = X_sub.shape[0]
        needed = target_samples - current
        if needed <= 0:
            return pd.DataFrame(), pd.DataFrame()
        X_oversampled, y_oversampled = X_sub.sample(n=needed, replace=True, random_state=self.random_state), \
                                        y_sub.sample(n=needed, replace=True, random_state=self.random_state)
        print(f"[DEBUG] Random oversampling: Generated {needed} new samples.")
        return X_oversampled, y_oversampled
    def find_nearest_neighbors(self, X_sub):
        k = min(self.num_neighbors, X_sub.shape[0])
        if k < 2:
            raise ValueError(f"Insufficient samples for neighbors: num_neighbors={self.num_neighbors}, n_samples={X_sub.shape[0]}")
        nearest_model = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='kd_tree').fit(X_sub)
        distances, indices = nearest_model.kneighbors(X_sub)
        return indices
    def generate_synthetic_samples(self, X_sub, y_sub, neighbors, num_samples):
        synthetic_X_list = []
        synthetic_y_list = []
        num_minority = X_sub.shape[0]
        for _ in range(num_samples):
            ref_index = random.randint(0, num_minority - 1)
            neighbor_pool = list(neighbors[ref_index])
            if ref_index in neighbor_pool:
                neighbor_pool.remove(ref_index)
            if not neighbor_pool:
                continue
            neighbor_index = random.choice(neighbor_pool)
            ref_features = X_sub.iloc[ref_index].values
            neighbor_features = X_sub.iloc[neighbor_index].values
            alpha = random.random()
            new_feature = ref_features + alpha * (neighbor_features - ref_features)
            synthetic_X_list.append(new_feature)
            ref_label = y_sub.iloc[ref_index].values.astype(float)
            neighbor_label = y_sub.iloc[neighbor_index].values.astype(float)
            new_label = (ref_label + neighbor_label) / 2.0
            new_label = (new_label > 0.5).astype(int)
            synthetic_y_list.append(new_label)
        if synthetic_X_list:
            new_X = pd.DataFrame(synthetic_X_list, columns=X_sub.columns)
        else:
            new_X = pd.DataFrame(columns=X_sub.columns)
        if synthetic_y_list:
            new_y = pd.DataFrame(synthetic_y_list, columns=y_sub.columns)
        else:
            new_y = pd.DataFrame(columns=y_sub.columns)
        return new_X, new_y

# --------------------------------------------
# Custom Multi-label Scoring Function

def multilabel_f1_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples', zero_division=0)

# --------------------------------------------
# Support Class for Training and Evaluation (Full Definition)

class Support:
    """
    Class to train and evaluate a multi-label RandomForest classifier using oversampling and MLSMOTE.
    """
    def __init__(self, cv_folds: int = 5, random_state: int = GLOBAL_SEED, n_jobs: int = 8):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []
        self.roc_results = []
        self.initial_parameters = {
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
        self.parameter_grid = {
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
    def fit_model(self, X: np.ndarray, y: np.ndarray, protein_ids: list = None, associated_variables: list = None,
                  model_name_prefix: str = 'model', model_directory: str = None, min_kmers: int = None):
        logging.info(f"Starting training for model {model_name_prefix}...")
        if min_kmers is not None:
            logging.info(f"Using provided min_kmers: {min_kmers}")
        else:
            min_kmers = len(X)
        X_dataframe = pd.DataFrame(X)
        y_dataframe = pd.DataFrame(y)
        # Random oversampling for each minority column
        for column in y_dataframe.columns:
            column_count = y_dataframe[column].sum()
            threshold = y_dataframe.shape[0] / len(y_dataframe.columns)
            if column_count < threshold:
                additional = 5 - int(column_count)
                if additional > 0:
                    positive_indices = y_dataframe.index[y_dataframe[column] == 1].tolist()
                    if len(positive_indices) > 0:
                        oversampled_indices = np.random.choice(positive_indices, size=additional, replace=True)
                        X_oversampled = X_dataframe.loc[oversampled_indices]
                        y_oversampled = y_dataframe.loc[oversampled_indices]
                        y_oversampled[column] = 1
                        X_dataframe = pd.concat([X_dataframe, X_oversampled], ignore_index=True)
                        y_dataframe = pd.concat([y_dataframe, y_oversampled], ignore_index=True)
                        print(f"[DEBUG] Column {column}: Oversampled {additional} samples.")
        # Apply MLSMOTE to generate additional synthetic samples
        mlsmote_object = MultiLabelSMOTE(num_neighbors=5, random_state=self.random_state, min_samples=5)
        extra_samples = max(int(len(X_dataframe) * 0.3), 1)
        resampled_results = mlsmote_object.fit_resample(X_dataframe, y_dataframe, extra_samples)
        X_smote, y_smote = resampled_results["combined"]
        logging.info(f"MLSMOTE applied. Final shapes - X: {X_smote.shape}, y: {y_smote.shape}")
        
        # Cross-validation using KFold
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []
        fold_number = 0
        for train_indices, test_indices in kfold.split(X_smote, y_smote):
            fold_number += 1
            X_train = X_smote[train_indices]
            X_test = X_smote[test_indices]
            y_train = y_smote[train_indices]
            y_test = y_smote[test_indices]
            logging.debug(f"Fold {fold_number} - X_train: {X_train.shape}, y_train: {y_train.shape}")
            base_rf_classifier = RandomForestClassifier(**self.initial_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
            ovr_classifier = OneVsRestClassifier(base_rf_classifier)
            ovr_classifier.fit(X_train, y_train)
            train_score = ovr_classifier.score(X_train, y_train)
            test_score = ovr_classifier.score(X_test, y_test)
            y_pred = ovr_classifier.predict(X_test)
            f1_metric = f1_score(y_test, y_pred, average='samples', zero_division=0)
            self.f1_scores.append(f1_metric)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)
            y_pred_proba = ovr_classifier.predict_proba(X_test)
            pr_auc = average_precision_score(y_test, y_pred_proba, average='samples')
            self.pr_auc_scores.append(pr_auc)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Train Score = {train_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test Score = {test_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: F1 Score (samples average) = {f1_metric}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: PR AUC (samples average) = {pr_auc}")
            try:
                roc_values = []
                for i in range(y_test.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
                    roc_values.append(auc(fpr, tpr))
                roc_value = np.mean(roc_values)
                self.roc_results.append(roc_value)
            except ValueError:
                logging.warning(f"Unable to calculate ROC AUC for fold {fold_number}.")
        # Grid search to find the best parameters
        inner_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        custom_scorer = make_scorer(multilabel_f1_scorer)
        base_model = OneVsRestClassifier(RandomForestClassifier(random_state=self.random_state))
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.parameter_grid,
            cv=inner_kfold,
            n_jobs=self.n_jobs,
            scoring=custom_scorer,
            verbose=1
        )
        grid_search.fit(X_smote, y_smote)
        best_model = grid_search.best_estimator_
        best_parameters = grid_search.best_params_
        self.model = best_model
        if model_directory:
            best_model_filename = os.path.join(model_directory, f'model_best_{model_name_prefix}.pkl')
            os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
            joblib.dump(best_model, best_model_filename)
            logging.info(f"Best model saved as {best_model_filename} for {model_name_prefix}")
        else:
            best_model_filename = f'model_best_{model_name_prefix}.pkl'
            joblib.dump(best_model, best_model_filename)
        logging.info(f"Best parameters for {model_name_prefix}: {best_parameters}")
        # After grid search, calibrate the final model (applied once to avoid multi-label issues)
        final_rf_classifier = RandomForestClassifier(
            n_estimators=best_parameters.get('estimator__n_estimators', 100),
            max_depth=best_parameters.get('estimator__max_depth', 10),
            min_samples_split=best_parameters.get('estimator__min_samples_split', 2),
            min_samples_leaf=best_parameters.get('estimator__min_samples_leaf', 4),
            criterion=best_parameters.get('estimator__criterion', 'gini'),
            max_features=best_parameters.get('estimator__max_features', 'log2'),
            class_weight=best_parameters.get('estimator__class_weight', 'balanced'),
            max_leaf_nodes=best_parameters.get('estimator__max_leaf_nodes', 20),
            min_impurity_decrease=best_parameters.get('estimator__min_impurity_decrease', 0.0),
            bootstrap=best_parameters.get('estimator__bootstrap', True),
            ccp_alpha=best_parameters.get('estimator__ccp_alpha', 0.0),
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        final_ovr_model = OneVsRestClassifier(final_rf_classifier)
        final_ovr_model.fit(X_smote, y_smote)
        self.model = final_ovr_model
        if model_directory:
            calibrated_model_filename = os.path.join(model_directory, f'calibrated_model_{model_name_prefix}.pkl')
        else:
            calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
        joblib.dump(final_ovr_model, calibrated_model_filename)
        logging.info(f"Calibrated model saved as {calibrated_model_filename} for {model_name_prefix}")
        return self.model
    def plot_learning_curve(self, output_path: str) -> None:
        plt.figure()
        plt.plot(self.train_scores, label='Train Score')
        plt.plot(self.test_scores, label='Test Score')
        plt.plot(self.f1_scores, label='F1 Score (samples average)')
        plt.plot(self.pr_auc_scores, label='PR AUC (samples average)')
        plt.title("Learning Curve (Multi-label + MLSMOTE)", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')
        plt.close()
    def test_best_RF(self, X: np.ndarray, y: np.ndarray, model_directory: str) -> tuple:
        scaler_path = os.path.join(model_directory, 'scaler_associated.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found at {scaler_path}.")
            sys.exit(1)
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=self.random_state)
        best_rf_classifier = RandomForestClassifier(
            n_estimators=self.parameter_grid.get('estimator__n_estimators', [100])[0],
            max_depth=self.parameter_grid.get('estimator__max_depth', [10])[0],
            min_samples_split=self.parameter_grid.get('estimator__min_samples_split', [2])[0],
            min_samples_leaf=self.parameter_grid.get('estimator__min_samples_leaf', [4])[0],
            criterion=self.parameter_grid.get('estimator__criterion', ['gini'])[0],
            max_features=self.parameter_grid.get('estimator__max_features', ['log2'])[0],
            class_weight=self.parameter_grid.get('estimator__class_weight', ['balanced'])[0],
            max_leaf_nodes=self.parameter_grid.get('estimator__max_leaf_nodes', [20])[0],
            min_impurity_decrease=self.parameter_grid.get('estimator__min_impurity_decrease', [0.0])[0],
            bootstrap=self.parameter_grid.get('estimator__bootstrap', [True])[0],
            ccp_alpha=self.parameter_grid.get('estimator__ccp_alpha', [0.0])[0],
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        ovr_model = OneVsRestClassifier(best_rf_classifier)
        ovr_model.fit(X_train, y_train)
        y_pred_proba = ovr_model.predict_proba(X_test)
        y_pred = ovr_model.predict(X_test)
        f1_value = f1_score(y_test, y_pred, average='samples', zero_division=0)
        pr_auc_value = average_precision_score(y_test, y_pred_proba, average='samples')
        try:
            roc_values = []
            for i in range(y_test.shape[1]):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
                roc_values.append(auc(fpr, tpr))
            roc_value = np.mean(roc_values)
        except ValueError:
            roc_value = float('nan')
        return roc_value, f1_value, pr_auc_value, self.parameter_grid, ovr_model, X_test, y_test

# --------------------------------------------
# Protein Embedding Generator Class Using Word2Vec (Full Definition)

class ProteinEmbeddingGenerator:
    """
    Class to generate protein embeddings using Word2Vec.
    If sequences are not aligned, they are realigned using MAFFT.
    """
    def __init__(self, sequences_path: str, table_data: pd.DataFrame = None, aggregation_method: str = 'none'):
        aligned_path = sequences_path
        if not are_sequences_aligned(sequences_path):
            aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")
            realign_sequences_with_mafft(sequences_path, aligned_path, threads=1)
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
                            model_directory: str = None, min_kmers: int = None,
                            save_min_kmers: bool = False) -> None:
        if model_directory:
            full_word2vec_model_path = os.path.join(model_directory, word2vec_model_path)
        else:
            full_word2vec_model_path = word2vec_model_path
        if os.path.exists(full_word2vec_model_path):
            logging.info(f"Word2Vec model found at {full_word2vec_model_path}. Loading the model.")
            model = joblib.load(full_word2vec_model_path)
            self.models['global'] = model
        else:
            logging.info("Word2Vec model not found. Training a new model.")
            kmer_groups = {}
            all_kmers = []
            kmers_counts = []
            for record in self.alignment:
                sequence = str(record.seq)
                sequence_length = len(sequence)
                protein_accession = record.id.split()[0]
                if self.table_data is not None:
                    matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession
                    matching_info = self.table_data[matching_rows]
                    if matching_info.empty:
                        logging.warning(f"No matching table data for {protein_accession}")
                        continue
                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]
                else:
                    target_variable = None
                    associated_variable = None
                logging.info(f"Processing {protein_accession} with sequence length {sequence_length}")
                if sequence_length < k:
                    logging.warning(f"Sequence too short for {protein_accession}. Length: {sequence_length}")
                    continue
                kmers = [sequence[i:i + k] for i in range(0, sequence_length - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]
                if not kmers:
                    logging.warning(f"No valid k-mers for {protein_accession}")
                    continue
                all_kmers.append(kmers)
                kmers_counts.append(len(kmers))
                kmer_groups[protein_accession] = {
                    'protein_accession': protein_accession,
                    'target_variable': target_variable,
                    'associated_variable': associated_variable,
                    'kmers': kmers
                }
            if not kmers_counts:
                logging.error("No k-mers were collected. Please check your sequences and k-mer parameters.")
                sys.exit(1)
            if min_kmers is not None:
                self.min_kmers = min_kmers
                logging.info(f"Using provided min_kmers: {self.min_kmers}")
            else:
                self.min_kmers = min(kmers_counts)
                logging.info(f"Minimum number of k-mers in any sequence: {self.min_kmers}")
            if save_min_kmers and model_directory:
                min_kmers_path = os.path.join(model_directory, 'min_kmers.txt')
                with open(min_kmers_path, 'w') as file:
                    file.write(str(self.min_kmers))
                logging.info(f"min_kmers saved at {min_kmers_path}")
            from gensim.models import Word2Vec
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=390,
                window=10,
                min_count=1,
                workers=48,
                sg=1,
                hs=1,
                negative=0,
                epochs=2500,
                seed=GLOBAL_SEED
            )
            if model_directory:
                os.makedirs(os.path.dirname(full_word2vec_model_path), exist_ok=True)
            model.save(full_word2vec_model_path)
            self.models['global'] = model
            logging.info(f"Word2Vec model saved at {full_word2vec_model_path}")
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
            kmer_groups[sequence_id] = {
                'protein_accession': sequence_id,
                'target_variable': target_variable,
                'associated_variable': associated_variable,
                'kmers': kmers
            }
        if not kmers_counts:
            logging.error("No k-mers were collected after re-evaluation. Check your sequences and parameters.")
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
            protein_kmers = embedding_info.get('kmers', [])
            if len(protein_kmers) == 0:
                if self.aggregation_method == 'none':
                    concatenated_embedding = np.zeros(self.models['global'].vector_size * self.min_kmers)
                else:
                    concatenated_embedding = np.zeros(self.models['global'].vector_size)
                self.embeddings.append({
                    'protein_accession': sequence_id,
                    'embedding': concatenated_embedding,
                    'target_variable': embedding_info.get('target_variable'),
                    'associated_variable': embedding_info.get('associated_variable')
                })
                continue
            selected_kmers = protein_kmers[:self.min_kmers]
            if len(selected_kmers) < self.min_kmers:
                padding = [np.zeros(self.models['global'].vector_size)] * (self.min_kmers - len(selected_kmers))
                selected_kmers.extend(padding)
            selected_embeddings = [self.models['global'].wv[kmer] if kmer in self.models['global'].wv 
                                     else np.zeros(self.models['global'].vector_size) for kmer in selected_kmers]
            if self.aggregation_method == 'none':
                concatenated_embedding = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                concatenated_embedding = np.mean(selected_embeddings, axis=0)
            else:
                logging.warning(f"Unknown aggregation method '{self.aggregation_method}'. Using concatenation.")
                concatenated_embedding = np.concatenate(selected_embeddings, axis=0)
            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': concatenated_embedding,
                'target_variable': embedding_info.get('target_variable'),
                'associated_variable': embedding_info.get('associated_variable')
            })
            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {concatenated_embedding.shape}")
        train_embeddings_array = np.array([entry['embedding'] for entry in self.embeddings])
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Inconsistent embedding shapes detected: {embedding_shapes}")
            raise ValueError("Embeddings have inconsistent shapes.")
        else:
            unique_shape = embedding_shapes.pop()
            logging.info(f"All embeddings have shape: {unique_shape}")
        scaler_full_path = os.path.join(model_directory, 'scaler_associated.pkl') if model_directory else 'scaler_associated.pkl'
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler found at {scaler_full_path}. Loading scaler.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler not found. Training a new scaler.")
            scaler = StandardScaler().fit(train_embeddings_array)
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler saved at {scaler_full_path}")
    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        embeddings_list = []
        raw_labels_list = []
        for entry in self.embeddings:
            embeddings_list.append(entry['embedding'])
            label_value = entry[label_type]
            if isinstance(label_value, str):
                split_labels = [lbl.strip() for lbl in label_value.split(',')]
                raw_labels_list.append(split_labels)
            else:
                if not label_value:
                    raw_labels_list.append([])
                else:
                    raw_labels_list.append([str(label_value)])
        embeddings_array = np.array(embeddings_list)
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(raw_labels_list)
        self.mlb_classes_ = mlb.classes_
        return embeddings_array, Y

def compute_perplexity(num_samples: int) -> int:
    return max(5, min(50, num_samples // 100))

def plot_dual_umap(train_embeddings: np.ndarray, train_labels: list, train_protein_ids: list,
                   predict_embeddings: np.ndarray, predict_labels: list, predict_protein_ids: list,
                   output_directory: str) -> tuple:
    umap_model_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_result_train = umap_model_train.fit_transform(train_embeddings)
    umap_model_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_result_predict = umap_model_predict.fit_transform(predict_embeddings)
    
    unique_train_labels = sorted(list(set(train_labels)))
    train_color_map = px.colors.qualitative.Dark24
    train_color_dict = {label: train_color_map[i % len(train_color_map)] for i, label in enumerate(unique_train_labels)}
    unique_predict_labels = sorted(list(set(predict_labels)))
    predict_color_map = px.colors.qualitative.Light24
    predict_color_dict = {label: predict_color_map[i % len(predict_color_map)] for i, label in enumerate(unique_predict_labels)}
    
    train_colors = [train_color_dict.get(label, 'gray') for label in train_labels]
    predict_colors = [predict_color_dict.get(label, 'gray') for label in predict_labels]
    
    train_umap_figure = go.Figure()
    train_umap_figure.add_trace(go.Scatter3d(
        x=umap_result_train[:, 0],
        y=umap_result_train[:, 1],
        z=umap_result_train[:, 2],
        mode='markers',
        marker=dict(size=5, color=train_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {label}" for pid, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Training Data'
    ))
    train_umap_figure.update_layout(
        title='UMAP 3D: Training Data',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    
    predict_umap_figure = go.Figure()
    predict_umap_figure.add_trace(go.Scatter3d(
        x=umap_result_predict[:, 0],
        y=umap_result_predict[:, 1],
        z=umap_result_predict[:, 2],
        mode='markers',
        marker=dict(size=5, color=predict_colors, opacity=0.8),
        text=[f"Protein ID: {pid}<br>Label: {label}" for pid, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Predictions'
    ))
    predict_umap_figure.update_layout(
        title='UMAP 3D: Predictions',
        scene=dict(xaxis=dict(title='Component 1'),
                   yaxis=dict(title='Component 2'),
                   zaxis=dict(title='Component 3'))
    )
    
    train_umap_html = os.path.join(output_directory, "umap_train_3d.html")
    predict_umap_html = os.path.join(output_directory, "umap_predict_3d.html")
    pio.write_html(train_umap_figure, file=train_umap_html, auto_open=False)
    pio.write_html(predict_umap_figure, file=predict_umap_html, auto_open=False)
    logging.info(f"UMAP training plot saved as {train_umap_html}")
    logging.info(f"UMAP predictions plot saved as {predict_umap_html}")
    return train_umap_figure, predict_umap_figure

def plot_predictions_scatterplot_custom(results: dict, output_path: str, top_n: int = 1) -> None:
    """Generate a scatterplot of predictions in the same order as in the table.
       The order is maintained according to the insertion order of results."""
    # Do not sort the keys; use the insertion order.
    protein_specificities = {}
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No ranking data for protein {seq_id}. Skipping...")
            continue
        top_category, confidence, top_two_categories = format_and_sum_probabilities(associated_rankings)
        if top_category is None:
            logging.warning(f"Invalid formatting for protein {seq_id}. Skipping...")
            continue
        # Use the associated variable name as it appears (e.g., "C12")
        category = top_category.split(" (")[0]
        protein_specificities[seq_id] = {'top_category': category, 'confidence': confidence}
    if not protein_specificities:
        logging.warning("No data available for scatterplot.")
        return
    # Use the keys in their insertion order
    unique_proteins = list(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))
    x_values = list(range(4, 19))
    for protein, data in protein_specificities.items():
        y_val = protein_order[protein]
        category = data['top_category']
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]
        for specificity in specificities:
            ax.scatter(specificity, y_val, color='#1E3A8A', edgecolors='black', linewidth=0.5, s=100, label='_nolegend_')
        if len(specificities) > 1:
            ax.plot(specificities, [y_val] * len(specificities), color='#1E3A8A', linestyle='-', linewidth=1.0, alpha=0.7)
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

def main(args: argparse.Namespace) -> None:
    model_directory = args.model_dir
    total_steps = 5
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # STEP 1: Training the Model
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Training sequences are not aligned. Realigning with MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Aligned training file found or sequences already aligned: {train_alignment_path}")
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Training table data loaded successfully.")
    current_step += 1
    progress_bar.progress(min(current_step / total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step / total_steps * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    
    protein_embedding_train = ProteinEmbeddingGenerator(train_alignment_path, table_data=train_table_data, aggregation_method=args.aggregation_method)
    protein_embedding_train.generate_embeddings(k=args.kmer_size, step_size=args.step_size,
                                                  word2vec_model_path=args.word2vec_model, model_directory=model_directory,
                                                  save_min_kmers=True)
    logging.info(f"Training embeddings generated: {len(protein_embedding_train.embeddings)}")
    min_kmers = protein_embedding_train.min_kmers
    training_protein_ids = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]
    associated_variables = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"X_associated shape: {X_associated.shape}, y_associated shape: {y_associated.shape}")
    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_directory, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info(f"Scaler for X_associated saved at {scaler_associated_path}")
    X_associated_scaled = scaler_associated.transform(X_associated)
    
    rf_model_associated_full_path = os.path.join(model_directory, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_directory, 'calibrated_model_associated.pkl')
    current_step += 1
    progress_bar.progress(min(current_step / total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step / total_steps * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Loaded existing calibrated model from {calibrated_model_associated_full_path}")
    else:
        support_associated = Support()
        calibrated_model_associated = support_associated.fit_model(X_associated_scaled, y_associated,
                                                                  protein_ids=training_protein_ids,
                                                                  associated_variables=associated_variables,
                                                                  model_name_prefix='associated',
                                                                  model_directory=model_directory,
                                                                  min_kmers=min_kmers)
        support_associated.plot_learning_curve(args.learning_curve_associated)
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated model saved at {calibrated_model_associated_full_path}")
        best_roc, best_f1, best_pr_auc, best_params, best_model, X_test_, y_test_ = support_associated.test_best_RF(X_associated_scaled, y_associated, model_directory)
        logging.info(f"Best ROC (samples average): {best_roc}")
        logging.info(f"Best F1 (samples average): {best_f1}")
        logging.info(f"Best PR AUC (samples average): {best_pr_auc}")
        logging.info(f"Best Parameters: {best_params}")
        joblib.dump(best_model, rf_model_associated_full_path)
        logging.info(f"Random Forest model (multi-label) saved at {rf_model_associated_full_path}")
        y_pred_proba_test = best_model.predict_proba(X_test_)
        mlb_classes = protein_embedding_train.mlb_classes_
        plot_global_roc_curve(y_test_, y_pred_proba_test, title="ROC Curve Multi-label (Associated)",
                              save_as=args.roc_curve_associated, classes=mlb_classes)
    current_step += 1
    progress_bar.progress(min(current_step / total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step / total_steps * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    
    # STEP 2: Classifying New Sequences
    min_kmers_path = os.path.join(model_directory, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as file:
            min_kmers_loaded = int(file.read().strip())
        logging.info(f"Loaded min_kmers: {min_kmers_loaded}")
    else:
        logging.error("min_kmers file not found. Ensure training was completed.")
        sys.exit(1)
    predict_alignment_path = args.predict_fasta
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Prediction sequences are not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Aligned prediction file found: {predict_alignment_path}")
    current_step += 1
    progress_bar.progress(min(current_step / total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step / total_steps * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)
    
    protein_embedding_predict = ProteinEmbeddingGenerator(predict_alignment_path, table_data=None, aggregation_method=args.aggregation_method)
    protein_embedding_predict.generate_embeddings(k=args.kmer_size, step_size=args.step_size,
                                                    word2vec_model_path=args.word2vec_model, model_directory=model_directory,
                                                    min_kmers=min_kmers_loaded)
    logging.info(f"Prediction embeddings generated: {len(protein_embedding_predict.embeddings)}")
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])
    scaler_path_predict = os.path.join(model_directory, 'scaler_associated.pkl')
    if os.path.exists(scaler_path_predict):
        scaler_associated_predict = joblib.load(scaler_path_predict)
        logging.info(f"Scaler loaded for prediction from {scaler_path_predict}")
    else:
        logging.error(f"Scaler not found at {scaler_path_predict}, training incomplete.")
        sys.exit(1)
    X_predict_scaled = scaler_associated_predict.transform(X_predict)
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)
    mlb_classes = protein_embedding_train.mlb_classes_
    rankings_associated = get_global_class_rankings(calibrated_model_associated, X_predict_scaled, mlb_classes=mlb_classes)
    results = {}
    for entry, prediction, ranking in zip(protein_embedding_predict.embeddings, predictions_associated, rankings_associated):
        seq_id = entry['protein_accession']
        results[seq_id] = {"associated_prediction": prediction, "associated_ranking": ranking}
    with open(args.results_file, 'w') as output_file:
        output_file.write("Protein_ID\tAssociated_Prediction\tAssociated_Ranking\n")
        for seq_id, info in results.items():
            output_file.write(f"{seq_id}\t{info['associated_prediction']}\t{'; '.join(info['associated_ranking'])}\n")
            logging.info(f"{seq_id} => {info['associated_prediction']}, {info['associated_ranking']}")
    logging.info("Generating scatter plot for new sequence predictions...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatter plot saved at {args.scatterplot_output}")
    logging.info("Generating dual UMAP plots for training/prediction data...")
    train_labels_str = []
    for row in y_associated:
        active_labels = [mlb_classes[i] for i, value in enumerate(row) if value == 1]
        train_labels_str.append(",".join(active_labels) if active_labels else "NoLabel")
    predict_labels_str = []
    for row in predictions_associated:
        active_labels = [mlb_classes[i] for i, value in enumerate(row) if value == 1]
        predict_labels_str.append(",".join(active_labels) if active_labels else "NoLabel")
    plot_dual_umap(X_associated_scaled, train_labels_str, training_protein_ids,
                   X_predict_scaled, predict_labels_str, [entry['protein_accession'] for entry in protein_embedding_predict.embeddings],
                   model_directory)
    logging.info("Dual UMAP plots completed.")
    tsne_train_fig, tsne_predict_fig = plot_dual_tsne(X_associated_scaled, train_labels_str, training_protein_ids,
                                                      X_predict_scaled, predict_labels_str, [entry['protein_accession'] for entry in protein_embedding_predict.embeddings],
                                                      model_directory)
    logging.info("Dual t-SNE plots completed.")
    current_step += 1
    progress_bar.progress(min(current_step / total_steps, 1.0))
    progress_text.markdown(f"<span style='color:white'>Progress: {int(current_step / total_steps * 100)}%</span>", unsafe_allow_html=True)
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
    def style_table(df):
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
    styled_df = style_table(df_results)
    html_table = styled_df.to_html(index=False, escape=False)
    st.header("Formatted Results")
    st.markdown(f"""<div class="dataframe-container">{html_table}</div>""", unsafe_allow_html=True)
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name='results.csv',
        mime='text/csv',
    )
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Results')
        writer.close()
        excel_data = output_excel.getvalue()
    st.download_button(
        label="Download Results as Excel",
        data=excel_data,
        file_name='results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for folder_name, subfolders, filenames in os.walk(model_directory):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, model_directory))
    zip_buffer.seek(0)
    st.header("Download All Results")
    st.download_button(
        label="Download All Results as results.zip",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )
    df_results.to_excel(args.excel_output, index=False)
    logging.info(f"Results saved to {args.excel_output}")
    with open(args.formatted_results_table, 'w') as final_file:
        final_file.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved at {args.formatted_results_table}")

# --------------------------------------------
# Streamlit UI Setup

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
    </style>
    """,
    unsafe_allow_html=True
)

def load_and_resize_image_with_dpi(image_path: str, base_width: int, dpi: int = 300) -> Image.Image:
    """Load and resize an image with DPI adjustment."""
    try:
        image = Image.open(image_path)
        width_percent = (base_width / float(image.size[0]))
        new_height = int((float(image.size[1]) * width_percent))
        resized_image = image.resize((base_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

def encode_image(image: Image.Image) -> str:
    """Encode an image to a Base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

image_path = "./images/faal.png"
image_base64 = encode_image(load_and_resize_image_with_dpi(image_path, base_width=150, dpi=300))
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
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Input Parameters")

def save_uploaded_file(uploaded_file, save_path: str) -> str:
    with open(save_path, 'wb') as file:
        file.write(uploaded_file.getbuffer())
    return save_path

use_default_training = st.sidebar.checkbox("Use Default Training Data", value=True)
if not use_default_training:
    training_fasta_file = st.sidebar.file_uploader("Upload Training FASTA File", type=["fasta", "fa", "fna"])
    training_table_file = st.sidebar.file_uploader("Upload Training Table File (TSV)", type=["tsv"])
else:
    training_fasta_file = None
    training_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Upload Prediction FASTA File", type=["fasta", "fa", "fna"])
kmer_size_input = st.sidebar.number_input("K-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size_input = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)
aggregation_method_input = st.sidebar.selectbox("Aggregation Method", options=['none', 'mean'], index=0)
st.sidebar.header("Customize Word2Vec Parameters")
customize_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)
if customize_word2vec:
    window_size = st.sidebar.number_input("Window Size", min_value=5, max_value=20, value=10, step=5)
    workers_number = st.sidebar.number_input("Workers", min_value=1, max_value=112, value=8, step=8)
    epochs_number = st.sidebar.number_input("Epochs", min_value=1, max_value=2500, value=2500, step=100)
else:
    window_size = 10
    workers_number = 48
    epochs_number = 2500
model_directory = create_unique_model_directory("results", aggregation_method_input)
output_directory = model_directory
if st.sidebar.button("Run Analysis"):
    internal_training_fasta = "data/train.fasta"
    internal_training_table = "data/train_table.tsv"
    if use_default_training:
        training_fasta_path = internal_training_fasta
        training_table_path = internal_training_table
        st.markdown("<span style='color:white'>Using default training data.</span>", unsafe_allow_html=True)
    else:
        if training_fasta_file is not None and training_table_file is not None:
            training_fasta_path = os.path.join(model_directory, "uploaded_train.fasta")
            training_table_path = os.path.join(model_directory, "uploaded_train_table.tsv")
            save_uploaded_file(training_fasta_file, training_fasta_path)
            save_uploaded_file(training_table_file, training_table_path)
            st.markdown("<span style='color:white'>Uploaded training data will be used.</span>", unsafe_allow_html=True)
        else:
            st.error("Please upload both the training FASTA file and the training TSV table file.")
            st.stop()
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(model_directory, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Please upload a FASTA file for prediction.")
        st.stop()
    arguments = argparse.Namespace(
        train_fasta=training_fasta_path,
        train_table=training_table_path,
        predict_fasta=predict_fasta_path,
        kmer_size=kmer_size_input,
        step_size=step_size_input,
        aggregation_method=aggregation_method_input,
        results_file=os.path.join(model_directory, "predictions.tsv"),
        output_dir=output_directory,
        scatterplot_output=os.path.join(model_directory, "scatterplot_predictions.png"),
        excel_output=os.path.join(model_directory, "results.xlsx"),
        formatted_results_table=os.path.join(model_directory, "formatted_results.txt"),
        roc_curve_associated=os.path.join(model_directory, "roc_curve_associated.png"),
        learning_curve_associated=os.path.join(model_directory, "learning_curve_associated.png"),
        roc_values_associated=os.path.join(model_directory, "roc_values_associated.csv"),
        rf_model_associated="rf_model_associated.pkl",
        word2vec_model="word2vec_model.bin",
        scaler="scaler_associated.pkl",
        model_dir=model_directory,
    )
    if not os.path.exists(arguments.model_dir):
        os.makedirs(arguments.model_dir)
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(arguments)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")

def load_and_resize_image(image_path: str, base_width: int, dpi: int = 300) -> Image.Image:
    """Load and resize an image with DPI adjustment."""
    try:
        img = Image.open(image_path)
        width_percent = (base_width / float(img.size[0]))
        new_height = int((float(img.size[1]) * width_percent))
        resized_img = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
        return resized_img
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

def encode_image_to_base64(img: Image.Image) -> str:
    """Encode an image to a Base64 string."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

image_directory = "images"
image_paths = [
    os.path.join(image_directory, "lab_logo.png"),
    os.path.join(image_directory, "ciimar.png"),
    os.path.join(image_directory, "faal_pred_logo.png"),
    os.path.join(image_directory, "bbf4.png"),
    os.path.join(image_directory, "google.png"),
    os.path.join(image_directory, "uniao.png"),
]
loaded_images = [load_and_resize_image(path, base_width=150, dpi=300) for path in image_paths]
encoded_images = [encode_image_to_base64(img) for img in loaded_images if img is not None]
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
image_tags = "".join(f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images)
st.markdown(footer_html.format(image_tags), unsafe_allow_html=True)
