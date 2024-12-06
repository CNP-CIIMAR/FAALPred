import logging
import os
import sys
from imblearn.over_sampling import ADASYN
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import random
import zipfile
from collections import Counter
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
from sklearn.preprocessing import StandardScaler, label_binarize
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from matplotlib import ticker
from sklearn.manifold import TSNE  # Import para t-SNE
import umap  # Import para UMAP
import base64
from plotly.graph_objs import Figure
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
# ============================================
# Definitions of Functions and Classes
# ============================================

# Fixing seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Alterar para DEBUG para mais verbosidade
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),  # Log em arquivo para registros persistentes
    ],
)
# ============================================
# ConfiguraÃ§Ã£o e Interface do Streamlit
# ============================================


# Ensure st.set_page_config is the very first Streamlit command
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="Ã°Å¸â€Â¬",  # DNA symbol
    layout="wide",
    initial_sidebar_state="expanded",
)



def are_sequences_aligned(fasta_file):
    """
    Checks if all sequences in a FASTA file have the same length.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Returns True if all sequences have the same length


def create_unique_model_directory(base_dir, aggregation_method):
    """
    Cria um diretÃ³rio de modelo Ãºnico baseado no mÃ©todo de agregaÃ§Ã£o.
    
    ParÃ¢metros:
    - base_dir (str): O diretÃ³rio base para os modelos.
    - aggregation_method (str): O mÃ©todo de agregaÃ§Ã£o utilizado.

    Retorna:
    - model_dir (str): Caminho para o diretÃ³rio de modelo exclusivo.
    """
    model_dir = os.path.join(base_dir, f"models_{aggregation_method}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def realign_sequences_with_mafft(input_path, output_path, threads=8):
    """
    Realigns sequences using MAFFT.
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Realigned sequences saved in {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running MAFFT: {e.stderr.decode()}")
        sys.exit(1)

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# FunÃ§Ã£o para realizar o clustering
def perform_clustering(data, method="DBSCAN", eps=0.5, min_samples=5, n_clusters=3):
    """
    Executa clustering nos dados usando DBSCAN ou K-Means.

    ParÃ¢metros:
    - data: np.ndarray com os dados para clustering.
    - method: "DBSCAN" ou "K-Means".
    - eps: ParÃ¢metro para DBSCAN (epsilon).
    - min_samples: ParÃ¢metro para DBSCAN.
    - n_clusters: NÃºmero de clusters para K-Means.

    Retorna:
    - labels: Labels gerados pelo mÃ©todo de clustering.
    """
    if method == "DBSCAN":
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "K-Means":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"MÃ©todo de clustering invÃ¡lido: {method}")

    labels = clustering_model.fit_predict(data)
    return labels
def plot_roc_curve_global(y_true, y_pred_proba, title, save_as=None, classes=None):
    """
    Plots ROC curve for binary or multiclass classifications.
    """
    lw = 2  # Line width

    # Check if it's binary or multiclass classification
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    else:  # Multiclass classification
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
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve of class {class_label} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.title(title, color='white')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')  # Match the background color
    plt.close()


def get_class_rankings_global(model, X):
    """
    Gets class rankings based on the probabilities predicted by the model.
    """
    if model is None:
        raise ValueError("Model not fitted yet. Please fit the model first.")

    # Obtaining probabilities for each class
    y_pred_proba = model.predict_proba(X)

    # Ranking classes based on probabilities
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings


def calculate_roc_values(model, X_test, y_test):
    """
    Calculates ROC AUC values for each class.
    """
    n_classes = len(np.unique(y_test))
    y_pred_proba = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Logging ROC values
        logging.info(f"For class {i}:")
        logging.info(f"FPR: {fpr[i]}")
        logging.info(f"TPR: {tpr[i]}")
        logging.info(f"ROC AUC: {roc_auc[i]}")
        logging.info("--------------------------")

    roc_df = pd.DataFrame(list(roc_auc.items()), columns=['Class', 'ROC AUC'])
    return roc_df


def format_and_sum_probabilities(associated_rankings):
    """
    Formats and sums probabilities for each category.
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

    # Initialize the sums dictionary
    for category in categories:
        category_sums[category] = 0.0

    # Sum probabilities for each category
    for rank in associated_rankings:
        try:
            prob = float(rank.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Error processing ranking string: {rank}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob

    # Sort results and format for output
    sorted_results = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)
    formatted_results = [f"{category} ({sum_prob:.2f}%)" for category, sum_prob in sorted_results if sum_prob > 0]

    return " - ".join(formatted_results)


def calculate_total_probability(associated_rankings, top_n=3):
    """
    Calculates the total probability of the top N predictions.
    """
    total_prob = 0.0
    top_predictions = associated_rankings[:top_n]
    for rank in top_predictions:
        try:
            prob = float(rank.split(": ")[1].replace("%", ""))
            total_prob += prob
        except (IndexError, ValueError):
            logging.error(f"Error processing ranking string: {rank}")
            continue
    return total_prob


def get_top_n_predictions(associated_rankings, top_n=3):
    """
    Retrieves the top N predictions.
    """
    top_predictions = associated_rankings[:top_n]
    predicted_ss = '-'.join([rank.split(": ")[0] for rank in top_predictions])
    return predicted_ss
def analyze_similarity(X_original, X_synthetic):
    similarities = cosine_similarity(X_synthetic, X_original)
    avg_similarity = np.mean(similarities, axis=1)
    logging.info(f"Average similarity between synthetic and original samples: {np.mean(avg_similarity):.4f}")
    return avg_similarity
        

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_latent_space(X_original, X_synthetic, y_original, y_synthetic):
    """
    Visualiza o espaÃ§o latente usando t-SNE para amostras originais e sintÃ©ticas.
    """
    # Combina os dados
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.hstack([y_original, y_synthetic])

    # Mapeia rÃ³tulos para nÃºmeros
    unique_labels = np.unique(y_combined)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_original_mapped = np.array([label_mapping[label] for label in y_original])
    y_synthetic_mapped = np.array([label_mapping[label] for label in y_synthetic])

    # Aplica t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_transformed = tsne.fit_transform(X_combined)

    # GrÃ¡fico t-SNE
    plt.figure(figsize=(12, 8))
    plt.scatter(
        X_transformed[:len(X_original), 0],
        X_transformed[:len(X_original), 1],
        c=y_original_mapped,
        cmap='viridis',
        label="Original",
        alpha=0.7
    )
    plt.scatter(
        X_transformed[len(X_original):, 0],
        X_transformed[len(X_original):, 1],
        c=y_synthetic_mapped,
        cmap='coolwarm',
        marker='x',
        label="Synthetic",
        alpha=0.7
    )

    # Adiciona legenda e tÃ­tulo
    plt.colorbar(label="Classes")
    plt.legend()
    plt.title("t-SNE Visualization of Original and Synthetic Samples")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

class Support:
    """
    Support class for training and evaluating Random Forest models with oversampling techniques.
    """

    def __init__(self, cv=5, seed=SEED, n_jobs=8):
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
            "n_estimators": 500,
            "max_depth": 15,  # Reduced to prevent overfitting
            "min_samples_split": 5,  # Increased to prevent overfitting
            "min_samples_leaf": 2,
            "criterion": "gini",
            "max_features": "sqrt",  # Changed from 'sqrt' to 'log2'
            "class_weight": "balanced_subsample",  # Automatic class balancing
            "max_leaf_nodes": None,  # Adjusted for greater regularization
            "min_impurity_decrease": 0.01,
            "bootstrap": True,
            "ccp_alpha": 0.001,
            "random_state": self.seed  # Added for RandomForest
        }
            

        self.parameters = {
            "n_estimators": [50, 100, 300, 700, 1000],
            "max_depth": [10, 15,20,30],
            "min_samples_split": [2,5,10,20],
            "min_samples_leaf": [1, 2, 4,8],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample", None],
            "max_leaf_nodes": [5, 10, 20, 30, 50, None],
            "min_impurity_decrease": [0.0, 0.01, 0.05],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0, 0.001, 0.01],
        }


    def _oversample_single_sample_classes(self, X, y):
        """
        Customizes oversampling to ensure all classes have at least 50 samples.
        """
        logging.info("Starting oversampling process...")

    # Contar a distribuiÃ§Ã£o inicial das classes
        counter = Counter(y)
        logging.info(f"Initial class distribution: {counter}")

    # Classes com pelo menos 2 amostras sÃ£o selecionadas para oversampling
        classes_to_oversample = {cls: max(50, count) for cls, count in counter.items()}
        logging.info(f"Target sampling strategy: {classes_to_oversample}")

    # RandomOverSampler aplicado para aumentar as classes com menos de 50 amostras
        ros = RandomOverSampler(sampling_strategy=classes_to_oversample, random_state=self.seed)
        X_ros, y_ros = ros.fit_resample(X, y)
        logging.info(f"Class distribution after RandomOverSampler: {Counter(y_ros)}")

    # SMOTE aplicado para sintetizar amostras e equilibrar ainda mais
        smote = SMOTE(random_state=self.seed)
        X_smote, y_smote = smote.fit_resample(X_ros, y_ros)
        logging.info(f"Class distribution after SMOTE: {Counter(y_smote)}")

    # Contagem final das classes
        sample_counts = Counter(y_smote)
        logging.info(f"Final class distribution: {sample_counts}")

    # Salvar contagem das classes no arquivo
        with open("oversampling_counts.txt", "a") as f:
            f.write("Class Distribution after Oversampling:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote

       

        
   # def _oversample_single_sample_classes(self, X, y):
  #      """
  #      Customizes oversampling to avoid oversampling extremely rare classes.
  #      """
  #      counter = Counter(y)
  #      classes_to_oversample = [cls for cls, count in counter.items() if count >= 2]

        # Apply RandomOverSampler only to classes with at least 2 samples
  #      ros = RandomOverSampler(random_state=self.seed)
  #      X_ros, y_ros = ros.fit_resample(X, y)

        # Apply SMOTE to classes that can be synthesized
   #     smote = SMOTE(random_state=self.seed)
#        X_smote, y_smote = smote.fit_resample(X_ros, y_ros)
#
 #       sample_counts = Counter(y_smote)
  #      logging.info(f"Class distribution after oversampling: {sample_counts}")
#
    #    with open("oversampling_counts.txt", "a") as f:
    #        f.write("Class Distribution after Oversampling:\n")
    #        for cls, count in sample_counts.items():
    #            f.write(f"{cls}: {count}\n")

    #    return X_smote, y_smote

    def fit(self, X, y, model_name_prefix='model', model_dir=None, min_kmers=None):
        logging.info(f"Starting fit method for {model_name_prefix}...")

        X = np.array(X)
        y = np.array(y)

        X_smote, y_smote = self._oversample_single_sample_classes(X, y)

        sample_counts = Counter(y_smote)
        logging.info(f"Sample counts after oversampling for {model_name_prefix}: {sample_counts}")

        with open("sample_counts_after_oversampling.txt", "a") as f:
            f.write(f"Sample Counts after Oversampling for {model_name_prefix}:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        if any(count < self.cv for count in sample_counts.values()):
            raise ValueError(f"There are classes with fewer members than the number of folds after oversampling for {model_name_prefix}.")

        min_class_count = min(sample_counts.values())
        self.cv = min(self.cv, min_class_count)

        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []

        fold_number = 1

        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        for train_index, test_index in skf.split(X_smote, y_smote):
            X_train, X_test = X_smote[train_index], X_smote[test_index]
            y_train, y_test = y_smote[train_index], y_smote[test_index]
            
            visualize_latent_space(X_train, X_test, y_train, y_test)

            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test set class distribution: {fold_class_distribution}")

            X_train_resampled, y_train_resampled = self._oversample_single_sample_classes(X_train, y_train)
            
            visualize_latent_space(X_train, X_train_resampled[len(X_train):], y_train, y_train_resampled[len(y_train):])

            train_sample_counts = Counter(y_train_resampled)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Training set class distribution after oversampling: {train_sample_counts}")

            with open("training_sample_counts_after_oversampling.txt", "a") as f:
                f.write(f"Fold {fold_number} Training Sample Counts after Oversampling for {model_name_prefix}:\n")
                for cls, count in train_sample_counts.items():
                    f.write(f"{cls}: {count}\n")

            self.model = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs)
            self.model.fit(X_train_resampled, y_train_resampled)

            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)

            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

            # Calculate F1-score and Precision-Recall AUC
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)

            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)

            if len(np.unique(y_test)) > 1:
                pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')
            else:
                pr_auc = 0.0  # Cannot calculate PR AUC for a single class
            self.pr_auc_scores.append(pr_auc)

            logging.info(f"Fold {fold_number} [{model_name_prefix}]: F1 Score: {f1}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Precision-Recall AUC: {pr_auc}")

            # Calculate ROC AUC
            try:
                if len(np.unique(y_test)) == 2:
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc_score_value = auc(fpr, tpr)
                    self.roc_results.append((fpr, tpr, roc_auc_score_value))
                else:
                    y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                    roc_auc_score_value = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovo', average='macro')
                    self.roc_results.append(roc_auc_score_value)
            except ValueError:
                logging.warning(f"Unable to calculate ROC AUC for fold {fold_number} [{model_name_prefix}] due to insufficient class representation.")

            # Perform grid search and save the best model
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            self.best_params = best_params

            if model_dir:
                best_model_filename = os.path.join(model_dir, f'best_model_{model_name_prefix}.pkl')
                # Ensure the directory exists
                os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved as {best_model_filename} for {model_name_prefix}")
            else:
                best_model_filename = f'best_model_{model_name_prefix}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved as {best_model_filename} for {model_name_prefix}")

            if best_params is not None:
                self.best_params = best_params
                logging.info(f"Best parameters for {model_name_prefix}: {self.best_params}")
            else:
                logging.warning(f"No best parameters found from grid search for {model_name_prefix}.")

            # Integrate Probability Calibration
            calibrator = CalibratedClassifierCV(self.model, method=' isotonic', cv=5, n_jobs=self.n_jobs)
            calibrator.fit(X_train_resampled, y_train_resampled)

            self.model = calibrator

            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'calibrated_model_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Calibrated model saved as {calibrated_model_filename} for {model_name_prefix}")

            fold_number += 1

            # Allow Streamlit to update the UI
            time.sleep(0.1)

        return self.model

    def _perform_grid_search(self, X_train_resampled, y_train_resampled):
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

    def get_best_param(self, param_name, default=None):
        return self.best_params.get(param_name, default)

    def plot_learning_curve(self, output_path):
        plt.figure()
        plt.plot(self.train_scores, label='Training score')
        plt.plot(self.test_scores, label='Cross-validation score')
        plt.plot(self.f1_scores, label='F1 Score')
        plt.plot(self.pr_auc_scores, label='Precision-Recall AUC')
        plt.title("Learning Curve", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')  # Match the background color
        plt.close()

    def get_class_rankings(self, X):
        """
        Gets class rankings for the given data.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        # Obtaining probabilities for each class
        y_pred_proba = self.model.predict_proba(X)

        # Ranking classes based on probabilities
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def test_best_RF(self, X, y, scaler_dir='.'):
        """
        Tests the best Random Forest model with the given data.
        """
        # Load the scaler
        scaler_path = os.path.join(scaler_dir, 'scaler.pkl') if scaler_dir else 'scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found at {scaler_path}")
            sys.exit(1)

        X_scaled = scaler.transform(X)

        # Apply oversampling to the entire dataset before splitting
        X_resampled, y_resampled = self._oversample_single_sample_classes(X_scaled, y)
        
        visualize_latent_space(X_train, X_train_resampled[len(X_train):], y_train, y_train_resampled[len(y_train):])
        

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        # Train RandomForestClassifier with the best parameters
        model = RandomForestClassifier(
            n_estimators=self.best_params.get('n_estimators', 100),
            max_depth=self.best_params.get('max_depth', 5),
            min_samples_split=self.best_params.get('min_samples_split', 4),
            min_samples_leaf=self.best_params.get('min_samples_leaf', 2),
            criterion=self.best_params.get('criterion', 'entropy'),
            max_features=self.best_params.get('max_features', 'log2'),
            class_weight=self.best_params.get('class_weight', 'balanced'),
            max_leaf_nodes=self.best_params.get('max_leaf_nodes', 20),
            min_impurity_decrease=self.best_params.get('min_impurity_decrease', 0.01),
            bootstrap=self.best_params.get('bootstrap', True),
            ccp_alpha=self.best_params.get('ccp_alpha', 0.001),
            random_state=self.seed,
            n_jobs=self.n_jobs
        )
        model.fit(X_train, y_train)  # Fit the model on the training data

        # Integrate Calibration into the Test Model
        calibrator = CalibratedClassifierCV(model, method='method', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)

        calibrated_model = calibrator

        # Make predictions
        y_pred = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred, method='normalize')

        # Calculate the score (e.g., AUC)
        score = self._calculate_score(y_pred_adjusted, y_test)

        # Calculate additional metrics
        y_pred_classes = calibrated_model.predict(X_test)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(y_test, y_pred_adjusted, average='macro')
        else:
            pr_auc = 0.0  # Cannot calculate PR AUC for a single class

        # Return the score, best parameters, trained model, and test sets
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred, y_test):
        """
        Calculates the score (e.g., ROC AUC) based on predictions and actual labels.
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

    def plot_roc_curve(self, y_true, y_pred_proba, title, save_as=None, classes=None):
        """
        Plots ROC curve for binary or multiclass classifications.
        """
        plot_roc_curve_global(y_true, y_pred_proba, title, save_as, classes)


class ProteinEmbeddingGenerator:
    def __init__(self, sequences_path, table_data=None, aggregation_method='none'):
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
        self.aggregation_method = aggregation_method  # Added to choose the aggregation method
        self.min_kmers = None  # Added to store min_kmers

    def generate_embeddings(self, k=3, step_size=1, word2vec_model_path="word2vec_model.bin", model_dir=None, min_kmers=None, save_min_kmers=False):
        """
        Generates embeddings for protein sequences using Word2Vec, standardizing the number of k-mers.
        """
        # Define the full path of the Word2Vec model
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
            # Variable Initialization
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
                        logging.warning(f"No match in data table for {protein_accession_alignment}")
                        continue  # Skip to the next iteration

                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]

                else:
                    # If there's no table, use default values or None
                    target_variable = None
                    associated_variable = None

                logging.info(f"Processing {protein_accession_alignment} with sequence length {seq_len}")

                if seq_len < k:
                    logging.warning(f"Sequence too short for {protein_accession_alignment}. Length: {seq_len}")
                    continue

                # Generate k-mers, allowing k-mers with less than k gaps
                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Allows k-mers with less than k gaps

                if not kmers:
                    logging.warning(f"No valid k-mer for {protein_accession_alignment}")
                    continue

                all_kmers.append(kmers)  # Adds the list of k-mers as a sentence
                kmers_counts.append(len(kmers))  # Stores the count of k-mers

                embedding_info = {
                    'protein_accession': protein_accession_alignment,
                    'target_variable': target_variable,
                    'associated_variable': associated_variable,
                    'kmers': kmers  # Stores the k-mers for later use
                }
                kmer_groups[protein_accession_alignment] = embedding_info

            # Determine the minimum number of k-mers
            if not kmers_counts:
                logging.error("No k-mers were collected. Check your sequences and k-mer parameters.")
                sys.exit(1)

            if min_kmers is not None:
                self.min_kmers = min_kmers
                logging.info(f"Using provided min_kmers: {self.min_kmers}")
            else:
                self.min_kmers = min(kmers_counts)
                logging.info(f"Minimum number of k-mers in any sequence: {self.min_kmers}")

            # Save min_kmers if required
            if save_min_kmers and model_dir:
                min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
                with open(min_kmers_path, 'w') as f:
                    f.write(str(self.min_kmers))
                logging.info(f"min_kmers saved at {min_kmers_path}")

            # Train Word2Vec model using all k-mers
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=125,  # change to 100 if necessary
                window=10,
                min_count=1,
                workers=8,
                sg=1,
                hs=1,  # Hierarchical softmax enabled
                negative=0,  # Negative sampling disabled
                epochs=2500,  # Fix number of epochs for reproducibility
                seed=SEED  # Fix seed for reproducibility
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
            sequence = str(record.seq)
            protein_accession_alignment = record.id.split()[0]

            # If table data is not provided, skip matching
            if self.table_data is not None:
                matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                matching_info = self.table_data[matching_rows]

                if matching_info.empty:
                    logging.warning(f"No match in data table for {protein_accession_alignment}")
                    continue  # Skip to the next iteration

                target_variable = matching_info['Target variable'].values[0]
                associated_variable = matching_info['Associated variable'].values[0]

            else:
                # If there's no table, use default values or None
                target_variable = None
                associated_variable = None

            kmers = [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, step_size)]
            kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Allows k-mers with less than k gaps

            if not kmers:
                logging.warning(f"No valid k-mer for {protein_accession_alignment}")
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

        # Determine the minimum number of k-mers
        if not kmers_counts:
            logging.error("No k-mers were collected. Check your sequences and k-mer parameters.")
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

            # Get embeddings of the selected k-mers
            selected_embeddings = [self.models['global'].wv[kmer] if kmer in self.models['global'].wv else np.zeros(self.models['global'].vector_size) for kmer in selected_kmers]

            if self.aggregation_method == 'none':
    # Concatena os embeddings dos k-mers selecionados
               embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
    # Agrega os embeddings dos k-mers selecionados pela mÃ©dia
               embedding_concatenated = np.mean(selected_embeddings, axis=0)
            else:
    # Se o mÃ©todo nÃ£o for reconhecido, usa a concatenaÃ§Ã£o como padrÃ£o
               logging.warning(f"MÃ©todo de agregaÃ§Ã£o desconhecido '{self.aggregation_method}'. Usando concatenaÃ§Ã£o.")
               embedding_concatenated = np.concatenate(selected_embeddings, axis=0)


            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': embedding_concatenated,
                'target_variable': embedding_info.get('target_variable'),
                'associated_variable': embedding_info.get('associated_variable')
            })

            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {embedding_concatenated.shape}")

        # Adjust StandardScaler with embeddings for training/prediction
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])

        # Check if all embeddings have the same shape
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Inconsistent embedding shapes detected: {embedding_shapes}")
            raise ValueError("Embeddings have inconsistent shapes.")
        else:
            logging.info(f"All embeddings have shape: {embedding_shapes.pop()}")

        # Define the full path of the scaler
        scaler_full_path = os.path.join(model_dir, 'scaler.pkl') if model_dir else 'scaler.pkl'

        # Check if the scaler already exists
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler found at {scaler_full_path}. Loading the scaler.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler not found. Training a new scaler.")
            scaler = StandardScaler().fit(embeddings_array_train)
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler saved at {scaler_full_path}")


    def get_embeddings_and_labels(self, label_type='target_variable'):
        """
        Returns embeddings and associated labels (target_variable or associated_variable).
        """
        embeddings = []
        labels = []

        for embedding_info in self.embeddings:
            embeddings.append(embedding_info['embedding'])
            labels.append(embedding_info[label_type])  # Uses the specified label type

        return np.array(embeddings), np.array(labels)



SEED = 42  # Define a seed for reproducibility

# Ajustar perplexidade dinamicamente
def compute_perplexity(n_samples):
    return max(5, min(50, n_samples // 100))

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.express as px

# FunÃ§Ã£o para plotar os grÃ¡ficos
def plot_dual_tsne_3d(train_embeddings, train_labels, train_protein_ids, 
                      predict_embeddings, predict_labels, predict_protein_ids,output_dir):
    """
    Plota dois grÃ¡ficos t-SNE 3D separados:
    - GrÃ¡fico 1: Dados de Treinamento.
    - GrÃ¡fico 2: PrediÃ§Ãµes.
    
    ParÃ¢metros:
    - train_embeddings (np.ndarray): Embeddings dos dados de treinamento.
    - train_labels (list or array): Labels associados aos dados de treinamento.
    - train_protein_ids (list): IDs de proteÃ­nas nos dados de treinamento.
    - predict_embeddings (np.ndarray): Embeddings das prediÃ§Ãµes.
    - predict_labels (list or array): Labels associados Ã s prediÃ§Ãµes.
    - predict_protein_ids (list): IDs de proteÃ­nas nas prediÃ§Ãµes.
    """
    # Ajustar perplexity dinamicamente
    n_samples_train = train_embeddings.shape[0]
    dynamic_perplexity_train = compute_perplexity(n_samples_train)

    # Inicializar t-SNE com perplexidade ajustada para treinamento
    tsne_train = TSNE(n_components=3, random_state=42, perplexity=dynamic_perplexity_train, n_iter=1000)
    tsne_train_result = tsne_train.fit_transform(train_embeddings)

    # Ajustar perplexity dinamicamente para prediÃ§Ãµes
    n_samples_predict = predict_embeddings.shape[0]
    dynamic_perplexity_predict = compute_perplexity(n_samples_predict)

    # Inicializar t-SNE com perplexidade ajustada para prediÃ§Ãµes
    tsne_predict = TSNE(n_components=3, random_state=42, perplexity=dynamic_perplexity_predict, n_iter=1000)
    tsne_predict_result = tsne_predict.fit_transform(predict_embeddings)

    # Criar mapa de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapa de cores para as prediÃ§Ãµes
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # GrÃ¡fico 1: Dados de treinamento
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter3d(
        x=tsne_train_result[:, 0],
        y=tsne_train_result[:, 1],
        z=tsne_train_result[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=train_colors,
            opacity=0.8
        ),
        # IDs de proteÃ­nas reais adicionados ao campo 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Training Data'
    ))
    fig_train.update_layout(
        title='t-SNE 3D: Training Data',
        scene=dict(
            xaxis=dict(title='Component 1'),
            yaxis=dict(title='Component 2'),
            zaxis=dict(title='Component 3')
        )
    )

    # GrÃ¡fico 2: PrediÃ§Ãµes
    fig_predict = go.Figure()
    fig_predict.add_trace(go.Scatter3d(
        x=tsne_predict_result[:, 0],
        y=tsne_predict_result[:, 1],
        z=tsne_predict_result[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=predict_colors,
            opacity=0.8
        ),
        # IDs de proteÃ­nas adicionados ao campo 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Predictions'
    ))
    fig_predict.update_layout(
        title='t-SNE 3D: Predictions',
        scene=dict(
            xaxis=dict(title='Component 1'),
            yaxis=dict(title='Component 2'),
            zaxis=dict(title='Component 3')
        )
    )
    # Salvar grÃ¡ficos em HTML
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")
    
    pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
    pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)
    
    logging.info(f"t-SNE Training Data saved as {tsne_train_html}")
    logging.info(f"t-SNE Predictions saved as {tsne_predict_html}")

    return fig_train, fig_predict

import umap.umap_ as umap
import plotly.graph_objects as go
import plotly.express as px

def plot_dual_umap(train_embeddings, train_labels, train_protein_ids,
                   predict_embeddings, predict_labels, predict_protein_ids, output_dir):
    """
    Plota dois grÃ¡ficos UMAP 3D separados:
    - GrÃ¡fico 1: Dados de Treinamento.
    - GrÃ¡fico 2: PrediÃ§Ãµes.
    
    ParÃ¢metros:
    - train_embeddings (np.ndarray): Embeddings dos dados de treinamento.
    - train_labels (list or array): Labels associados aos dados de treinamento.
    - train_protein_ids (list): IDs de proteÃ­nas nos dados de treinamento.
    - predict_embeddings (np.ndarray): Embeddings das prediÃ§Ãµes.
    - predict_labels (list or array): Labels associados Ã s prediÃ§Ãµes.
    - predict_protein_ids (list): IDs de proteÃ­nas nas prediÃ§Ãµes.
    """
    # ReduÃ§Ã£o de dimensionalidade para treinamento
    umap_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_train_result = umap_train.fit_transform(train_embeddings)

    # ReduÃ§Ã£o de dimensionalidade para prediÃ§Ãµes
    umap_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_predict_result = umap_predict.fit_transform(predict_embeddings)

    # Criar mapa de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapa de cores para as prediÃ§Ãµes
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # GrÃ¡fico 1: Dados de treinamento
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter3d(
        x=umap_train_result[:, 0],
        y=umap_train_result[:, 1],
        z=umap_train_result[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=train_colors,
            opacity=0.8
        ),
        # IDs de proteÃ­nas reais adicionados ao campo 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Training Data'
    ))
    fig_train.update_layout(
        title='UMAP 3D: Training Data',
        scene=dict(
            xaxis=dict(title='Component 1'),
            yaxis=dict(title='Component 2'),
            zaxis=dict(title='Component 3')
        )
    )

    # GrÃ¡fico 2: PrediÃ§Ãµes
    fig_predict = go.Figure()
    fig_predict.add_trace(go.Scatter3d(
        x=umap_predict_result[:, 0],
        y=umap_predict_result[:, 1],
        z=umap_predict_result[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=predict_colors,
            opacity=0.8
        ),
        # IDs de proteÃ­nas adicionados ao campo 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Predictions'
    ))
    fig_predict.update_layout(
        title='UMAP 3D: Predictions',
        scene=dict(
            xaxis=dict(title='Component 1'),
            yaxis=dict(title='Component 2'),
            zaxis=dict(title='Component 3')
        )
    )

    # Salvar grÃ¡ficos em HTML
    umap_train_html = os.path.join(output_dir, "umap_train_3d.html")
    umap_predict_html = os.path.join(output_dir, "umap_predict_3d.html")
    
    pio.write_html(fig_train, file=umap_train_html, auto_open=False)
    pio.write_html(fig_predict, file=umap_predict_html, auto_open=False)
    
    logging.info(f"UMAP Training Data saved as {umap_train_html}")
    logging.info(f"UMAP Predictions saved as {umap_predict_html}")

    return fig_train, fig_predict


def plot_predictions_scatterplot_custom(results, output_path, top_n=3):
    """
    Generates a scatter plot of the top N predictions for the new sequences.

    Y-axis: Protein accession ID
    X-axis: Specificities from C2 to C18 (fixed scale)
    Each point represents the corresponding specificity for the protein.
    Only the top N predictions are plotted.
    Points are colored in a single uniform color, styled for scientific publication.
    """
    # Prepare data
    protein_specificities = {}
    
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No associated ranking data for protein {seq_id}. Skipping...")
            continue

        specificity_probs = {}
        for ranking in associated_rankings[:top_n]:  # Top N only
            try:
                # Split and extract data
                category, prob = ranking.split(": ")
                prob = float(prob.replace("%", ""))

                # Extract the first number from the category
                if category.startswith('C'):
                    # Extract only the first number before the colon or any other separator
                    spec = int(category.split(':')[0].strip('C'))
                    specificity_probs[spec] = prob
            except ValueError as e:
                logging.error(f"Error processing ranking: {ranking} for protein {seq_id}. Error: {e}")

        if specificity_probs:
            protein_specificities[seq_id] = specificity_probs

    if not protein_specificities:
        logging.warning("No data available to plot the scatterplot.")
        return

    # Sort protein IDs for better visualization
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(unique_proteins) * 0.5))  # Adjust height based on the number of proteins

    # Fixed scale for x-axis from C2 to C18
    x_values = list(range(2, 19))

    for protein, specs in protein_specificities.items():
        y = protein_order[protein]
        
        # Prepare data for plotting (ensure we only plot the specified top N predictions)
        x = []
        probs = []
        for spec in x_values:
            if spec in specs:
                x.append(spec)
                probs.append(specs[spec])

        if not x:
            logging.warning(f"No valid data to plot for protein {protein}. Skipping...")
            continue

        # Plot points for the top N predictions only
        ax.scatter(x, [y] * len(x), color='#1f78b4', edgecolors='black', linewidth=0.5, s=100, label='_nolegend_')

        # Connect points with lines for the top N predictions
        if len(x) > 1:
            ax.plot(x, [y] * len(x), color='#1f78b4', linestyle='-', linewidth=1.0, alpha=0.7)

    # Customize the plot for better publication quality
    ax.set_xlabel('Specificity (C2 to C18)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold')
    ax.set_title('Scatterplot of New Sequences Predictions (Top Rankings)', fontsize=16, fontweight='bold', pad=20)

    # Set fixed x-axis scale and formatting
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12)
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10)

    # Set grid and remove unnecessary spines for a clean look
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Minor ticks on x-axis for improved visibility
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Adjust layout to avoid label cut-off
    plt.tight_layout()

    # Save figure in high quality for publication
    plt.savefig(output_path, facecolor='white', dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Scatterplot saved at {output_path}")



def adjust_predictions_global(predicted_proba, method='normalize', alpha=1.0):
    """
    Adjusts the predicted probabilities from the model.
    """
    if method == 'normalize':
        # Normalize probabilities so they sum to 1 for each sample
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


def main(args):
    model_dir = args.model_dir  # This should be 'results/models'

    """
    Main function coordinating the workflow.
    """
    model_dir = args.model_dir

    # Initialize progress variables
    total_steps = 10
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # =============================
    # STEP 1: Model Training
    # =============================

    # Load training data
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table

    # Check if training sequences are aligned
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Training sequences are not aligned. Realigning with MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)  # Fix threads=1
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Aligned training file found or sequences already aligned: {train_alignment_path}")

    # Load training table data
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Training data table loaded successfully.")

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Initialize and generate embeddings for training
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, 
        train_table_data, 
        aggregation_method=args.aggregation_method  # Passing the aggregation method
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        save_min_kmers=True  # Save min_kmers after training
    )
    logging.info(f"Number of training embeddings generated: {len(protein_embedding_train.embeddings)}")

    # Save min_kmers to ensure consistency
    min_kmers = protein_embedding_train.min_kmers

    # Get embeddings and labels for target_variable
    X_target, y_target = protein_embedding_train.get_embeddings_and_labels(label_type='target_variable')
    logging.info(f"X_target shape: {X_target.shape}")

    # Full paths for target_variable models
    rf_model_target_full_path = os.path.join(model_dir, args.rf_model_target)
    calibrated_model_target_full_path = os.path.join(model_dir, 'calibrated_model_target.pkl')

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Check if calibrated model for target_variable already exists
    if os.path.exists(calibrated_model_target_full_path):
        calibrated_model_target = joblib.load(calibrated_model_target_full_path)
        logging.info(f"Calibrated Random Forest model for target_variable loaded from {calibrated_model_target_full_path}")
    else:
        # Model training for target_variable
        support_model_target = Support()
        calibrated_model_target = support_model_target.fit(X_target, y_target, model_name_prefix='target', model_dir=model_dir, min_kmers=min_kmers)
        logging.info("Training and calibration for target_variable completed.")

        # Save the calibrated model
        joblib.dump(calibrated_model_target, calibrated_model_target_full_path)
        logging.info(f"Calibrated Random Forest model for target_variable saved at {calibrated_model_target_full_path}")

        # Test the model
        best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, scaler_dir=args.model_dir)

        logging.info(f"Best ROC AUC for target_variable: {best_score}")
        logging.info(f"Best F1 Score for target_variable: {best_f1}")
        logging.info(f"Best Precision-Recall AUC for target_variable: {best_pr_auc}")
        logging.info(f"Best Parameters: {best_params}")

        for param, value in best_params.items():
            logging.info(f"{param}: {value}")

        # Get class rankings
        class_rankings = support_model_target.get_class_rankings(X_test_target)

        # Display rankings for the first 5 samples
        logging.info("Top 3 class rankings for the first 5 samples:")
        for i in range(min(5, len(class_rankings))):
            logging.info(f"Sample {i+1}: Class rankings - {class_rankings[i][:3]}")  # Shows top 3 rankings

        # Plot ROC curve
        n_classes_target = len(np.unique(y_test_target))
        if n_classes_target == 2:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)[:, 1]
        else:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)
            unique_classes_target = np.unique(y_test_target).astype(str)
        plot_roc_curve_global(y_test_target, y_pred_proba_target, 'ROC Curve for Target Variable', save_as=args.roc_curve_target, classes=unique_classes_target)

        # Convert y_test_target to integer labels
        unique_labels = sorted(set(y_test_target))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        y_test_target_int = [label_to_int[label.strip()] for label in y_test_target]

        # Calculate and print ROC values for target_variable
        roc_df_target = calculate_roc_values(best_model_target, X_test_target, y_test_target_int)
        logging.info("ROC AUC Scores for target_variable:")
        logging.info(roc_df_target)
        roc_df_target.to_csv(args.roc_values_target, index=False)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Repeat the process for associated_variable
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"X_associated shape: {X_associated.shape}")

    # Full paths for associated_variable models
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Check if calibrated model for associated_variable already exists
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable loaded from {calibrated_model_associated_full_path}")
    else:
        # Model training for associated_variable
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(X_associated, y_associated, model_name_prefix='associated', model_dir=model_dir, min_kmers=min_kmers)
        logging.info("Training and calibration for associated_variable completed.")
        
        # Plot learning curve
        logging.info("Plotting Learning Curve for Associated Variable")
        support_model_associated.plot_learning_curve(args.learning_curve_associated)

        # Save the calibrated model
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable saved at {calibrated_model_associated_full_path}")

        # Test the model
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, scaler_dir=args.model_dir)

        logging.info(f"Best ROC AUC for associated_variable in test_best_RF: {best_score_associated}")
        logging.info(f"Best F1 Score for associated_variable in test_best_RF: {best_f1_associated}")
        logging.info(f"Best Precision-Recall AUC for associated_variable in test_best_RF: {best_pr_auc_associated}")
        logging.info(f"Best Parameters found in test_best_RF: {best_params_associated}")
        logging.info(f"Best model Associated in test_best_RF: {best_model_associated}")

        # Get class rankings for associated_variable
        class_rankings_associated = support_model_associated.get_class_rankings(X_test_associated)
        logging.info("Top 3 class rankings for the first 5 samples in associated data:")
        for i in range(min(5, len(class_rankings_associated))):
            logging.info(f"Sample {i+1}: Class rankings - {class_rankings_associated[i][:3]}")  # Shows top 3 rankings

        # Accessing class_weight from the best_params_associated dictionary
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
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            unique_classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve_global(y_test_associated, y_pred_proba_associated, 'ROC Curve for Associated Variable', save_as=args.roc_curve_associated, classes=unique_classes_associated)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # STEP 2: Classifying New Sequences
    # =============================

    # Load min_kmers
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"Loaded min_kmers: {min_kmers_loaded}")
    else:
        logging.error(f"min_kmers file not found at {min_kmers_path}. Ensure training was completed successfully.")
        sys.exit(1)

    # Load data for prediction
    predict_alignment_path = args.predict_fasta

    # Check if sequences for prediction are aligned
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Sequences for prediction are not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Fix threads=1
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
        aggregation_method=args.aggregation_method  # Passing the aggregation method
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded  # Use the same min_kmers as training
    )
    logging.info(f"Number of embeddings for prediction generated: {len(protein_embedding_predict.embeddings)}")

    # Get embeddings for prediction
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Load the scaler
    scaler_full_path = os.path.join(model_dir, args.scaler)
    if os.path.exists(scaler_full_path):
        scaler = joblib.load(scaler_full_path)
        logging.info(f"Scaler loaded from {scaler_full_path}")
    else:
        logging.error(f"Scaler not found at {scaler_full_path}")
        sys.exit(1)
    X_predict_scaled = scaler.transform(X_predict)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Make predictions on new sequences

    # Verificar o tamanho das features antes da prediÃ§Ã£o
    # Verifique o nÃºmero de caracterÃ­sticas em relaÃ§Ã£o ao estimador original do CalibratedClassifierCV
    if X_predict_scaled.shape[1] > calibrated_model_target.estimator.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_target.estimator.n_features_in_} to match the model input size.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_target.estimator.n_features_in_]


    predictions_target = calibrated_model_target.predict(X_predict_scaled)

    # Verificar e ajustar o tamanho das features para associated_variable
    if X_predict_scaled.shape[1] > calibrated_model_associated.estimator.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_associated.estimator.n_features_in_} to match the model input size for associated_variable.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_associated.estimator.n_features_in_]

    # Realizar a prediÃ§Ã£o para associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)

    # Get class rankings
    rankings_target = get_class_rankings_global(calibrated_model_target, X_predict_scaled)
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled)

    # Process and save the results


# Process and filter results
    results = {}
    for entry, pred_target, pred_associated, ranking_target, ranking_associated in zip(
            protein_embedding_predict.embeddings, 
            predictions_target, 
            predictions_associated, 
            rankings_target, 
            rankings_associated):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "target_prediction": pred_target,
            "associated_prediction": pred_associated,
            "target_ranking": ranking_target,
            "associated_ranking": ranking_associated[:3]  # Only keep top 3 rankings
        }

# Format results for tabulation and export
    formatted_results = []
    for sequence_id, info in results.items():
        associated_rankings = info['associated_ranking']
        top3 = associated_rankings
        predicted_ss = '-'.join([item.split(": ")[0] for item in top3])
        ss_pred_prob = sum([float(item.split(": ")[1].replace("%", "")) for item in top3])
        ranking_completo = ' - '.join(associated_rankings)
        formatted_results.append([sequence_id, predicted_ss, f"{ss_pred_prob:.2f}%", ranking_completo])

# Log formatted results
    logging.info("Formatted Results:")
    for result in formatted_results:
        logging.info(result)

# Define headers for the output table
    headers = ["Query Name", "Predicted SS", "SS Prediction Probability (%)", "Ranking"]

# Save results to Excel
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Results saved in {args.excel_output}")

# Save formatted table as a text file
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved in {args.formatted_results_table}")

# Generate Scatterplot
    logging.info("Generating scatterplot of new sequences predictions...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatterplot saved at {args.scatterplot_output}")

    logging.info("Processing completed.")

# Custom CSS for dark navy blue background and white text
st.markdown(
    """
    <style>
    /* Define the main app background and text color */
    .stApp {
        background-color: #0B3C5D;
        color: white;
    }
    /* Define the sidebar background and text color */
    [data-testid="stSidebar"] {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Ensure all elements inside the sidebar have blue background and white text */
    [data-testid="stSidebar"] * {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Customize input elements inside the sidebar */
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
    /* Customize file uploader drag and drop area */
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] div {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Customize select dropdown options */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Remove borders and shadows */
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
    /* Customize checkbox and radio buttons */
    [data-testid="stSidebar"] .stCheckbox input[type="checkbox"] + div:first-of-type,
    [data-testid="stSidebar"] .stRadio input[type="radio"] + div:first-of-type {
        background-color: #1E3A8A !important;
    }
    /* Customize slider track and thumb */
    [data-testid="stSidebar"] .stSlider > div:first-of-type {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider .st-bo {
        background-color: #1E3A8A !important;
    }
    /* Ensure headers are white */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    /* Ensure alert messages (st.info, st.error, etc.) have white text */
    div[role="alert"] p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from PIL import Image
# FunÃ§Ã£o para converter a imagem em base64
def get_base64_image(image_path):
    """
    Encodes an image file to a base64 string.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - base64 string of the image.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return ""

# Caminho da imagem
image_path = "./images/faal.png"
image_base64 = get_base64_image(image_path)
# Usando HTML com st.markdown para alinhar tÃ­tulo e texto

st.markdown(
    f"""
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 30px; background: linear-gradient(to bottom, #f9f9f9, #ffffff); border-radius: 15px; border: 2px solid #dddddd; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); position: relative;">
        <p style="color: black; font-size: 1.5em; font-weight: bold; margin: 0;">
            FAALPred: Predicting Fatty Acid Specificities of Fatty Acyl-AMP Ligases (FAALs) Using Integrated Approaches of Neural Networks, Bioinformatics, and Machine Learning
        </p>
        <p style="color: #2c3e50; font-size: 1.2em; font-weight: normal; margin-top: 10px;">
            Anne Liong, Leandro de Mattos Pereira, and Pedro LeÃ£o
        </p>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8;">
            <strong>FAALPred</strong> is a comprehensive bioinformatics tool designed to predict 
            the chain length specificity of fatty acid substrates, ranging from C4 to C18.
        </p>
        <h5 style="color: #2c3e50; font-size: 20px; font-weight: bold; margin-top: 25px;">ABSTRACT</h5>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8; text-align: justify;">
            Fatty Acyl-AMP Ligases (FAALs), identified by Zhang et al. (2011), activate fatty acids of varying lengths for natural product biosynthesis. 
            These substrates enable the production of compounds like nocuolin (<em>Nodularia sp.</em>, Martins et al., 2022) 
            and sulfolipid-1 (<em>Mycobacterium tuberculosis</em>, Yan et al., 2023), with applications in cancer and tuberculosis 
            treatment (Kurt et al., 2017; Gilmore et al., 2012). Dr. Pedro LeÃ£o and His Team Identified Several of These Natural Products in Cyanobacteria (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" style="color: #3498db; text-decoration: none;">visit here</a>), 
            and FAALpred classifies FAALs by their substrate specificity.
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{image_base64}" alt="FAAL domain" style="width: auto; height: 120px; object-fit: contain;">
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>FAAL domain from Synechococcus sp. PCC7002, link: <a href="https://www.rcsb.org/structure/7R7F" target="_blank" style="color: #3498db; text-decoration: none;">https://www.rcsb.org/structure/7R7F</a></em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Function to save uploaded files
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Input options
use_default_train = st.sidebar.checkbox("Use default training data", value=True)
if not use_default_train:
    train_fasta_file = st.sidebar.file_uploader("Upload Training FASTA File", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("Upload Training Table File (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Upload Prediction FASTA File", type=["fasta", "fa", "fna"])

kmer_size = st.sidebar.number_input("K-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)
aggregation_method = st.sidebar.selectbox(
    "Aggregation Method",
    options=['none', 'mean'],
    index=0
)

 #Entrada opcional para parÃ¢metros do Word2Vec
st.sidebar.header("Optional Word2Vec Parameters")
custom_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)
if custom_word2vec:
    window = st.sidebar.number_input(
        "Window size", min_value=5, max_value=20, value=5, step=5
    )
    workers = st.sidebar.number_input(
        "Workers", min_value=1, max_value=112, value=8, step=8
    )
    epochs = st.sidebar.number_input(
        "Epochs", min_value=1, max_value=3500, value=2, step=100
    )
else:
    window = 10  # Valor padrÃ£o
    workers = 8  # Valor padrÃ£o
    epochs = 2500  # Valor padrÃ£o
    
# Output directory
#output_dir = "results"
#if not os.path.exists(output_dir):
 #   os.makedirs(output_dir)
# Button to start processing
if st.sidebar.button("Run Analysis"):
    # Paths for internal data
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"
    
    model_dir = create_unique_model_directory("results", aggregation_method)
    output_dir = model_dir
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
            st.error("Please upload both the training FASTA file and the training table TSV file.")
            st.stop()

    # Handling prediction data
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(output_dir, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Please upload a prediction FASTA file.")
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
        roc_curve_target=os.path.join(output_dir, "roc_curve_target.png"),
        roc_curve_associated=os.path.join(output_dir, "roc_curve_associated.png"),
        learning_curve_target=os.path.join(output_dir, "learning_curve_target.png"),
        learning_curve_associated=os.path.join(output_dir, "learning_curve_associated.png"),
        roc_values_target=os.path.join(output_dir, "roc_values_target.csv"),
        rf_model_target="rf_model_target.pkl",
        rf_model_associated="rf_model_associated.pkl",
        word2vec_model="word2vec_model.bin",
        scaler="scaler.pkl",
#       model_dir=os.path.join(output_dir, "models")
        model_dir=model_dir,
    )

    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Run the main analysis function
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(args)

        st.success("Analysis completed successfully!")

        # Display scatterplot
        st.header("Scatterplot of Predictions")
       # st.image(args.scatterplot_output, use_column_width=True)
      #  st.image('results/scatterplot_predictions.png', use_container_width=True)
        scatterplot_path = os.path.join(args.output_dir, "scatterplot_predictions.png")
        st.image(scatterplot_path)



        # Display formatted results table
 #       st.header("Formatted Results Table")
 #       with open(args.formatted_results_table, 'r') as f:
 #           formatted_table = f.read()
 #       st.text(formatted_table)
# Caminho do arquivo formatado
        formatted_table_path = args.formatted_results_table

# Verificar se o arquivo existe e nÃ£o estÃ¡ vazio
        if os.path.exists(formatted_table_path) and os.path.getsize(formatted_table_path) > 0:
            try:
        # Abrir e ler o conteÃºdo do arquivo
                with open(formatted_table_path, 'r') as f:
                    formatted_table = f.read()
        
        # Exibir o conteÃºdo no Streamlit
                st.text(formatted_table)
            except Exception as e:
                st.error(f"An error occurred while reading the formatted results table: {e}")
        else:
            st.error(f"Formatted results table not found or is empty: {formatted_table_path}")
    
        # Prepare results.zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for folder_name, subfolders, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    zip_file.write(file_path, arcname=os.path.relpath(file_path, output_dir))
        zip_buffer.seek(0)

        # Provide download link
        st.header("Download Results")
        st.download_button(
            label="Download All Results as results.zip",
            data=zip_buffer,
            file_name="results.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")

# FunÃ§Ã£o para carregar e redimensionar imagens com ajuste de DPI
# FunÃ§Ã£o para carregar e redimensionar imagens com ajuste de DPI
def load_and_resize_image_with_dpi(image_path, base_width, dpi=300):
    try:
        # Carrega a imagem
        image = Image.open(image_path)
        # Calcula a nova altura proporcional
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Redimensiona a imagem
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

# FunÃ§Ã£o para carregar e redimensionar imagens com ajuste de DPI
def load_and_resize_image_with_dpi(image_path, base_width, dpi=300):
    try:
        # Carrega a imagem
        image = Image.open(image_path)
        # Calcula a nova altura proporcional
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Redimensiona a imagem
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

# DefiniÃ§Ãµes dos caminhos das imagens
image_dir = "images"
image_paths = [
    os.path.join(image_dir, "lab_logo.png"),
    os.path.join(image_dir, "ciimar.png"),
    os.path.join(image_dir, "faal_pred_logo.png"), 
    os.path.join(image_dir, "bbf4.png"),
    os.path.join(image_dir, "google.png"),
    os.path.join(image_dir, "uniao.png"),
]

# Carregar e redimensionar todas as imagens
images = [load_and_resize_image_with_dpi(path, base_width=150, dpi=300) for path in image_paths]

# Codificar imagens como base64
import base64
from io import BytesIO

def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

encoded_images = [encode_image(img) for img in images if img is not None]

# CSS para layout
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

# HTML to display images in the footer
footer_html = """
<div class="support-text">Supported by:</div>
<div class="footer-container">
    {}
</div>
<div class="footer-text" style="font-size: 18px; margin-top: 10px;">
    FAALPred will make predictions for all submitted sequences, even if they are not FAALs. To limit your input only to Fatty Acid Ligase FAALs, first consider extracting from each FAAL the sequences corresponding to the FAAL domain cd05931 (cd05931) using InterProScan.
</div>
<div class="footer-text" style="font-size: 14px; margin-top: 5px;">
    CIIMAR - Pedro LeÃ£o @CNP - 2024 - All rights reserved.
</div>
"""

# Generate <img> tags for each image
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Render the footer
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)
