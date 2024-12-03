import logging
import os
import sys
import subprocess
import random
import zipfile
from collections import Counter
from io import BytesIO
import shutil
import time
import traceback
import argparse 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, average_precision_score, silhouette_score
from sklearn.model_selection import learning_curve, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.express as px
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MafftCommandline
import joblib
import plotly.io as pio
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from matplotlib import ticker
import umap.umap_ as umap
import base64
from io import BytesIO
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score
from scipy.stats import ttest_ind

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
# Configuração e Interface do Streamlit
# ============================================

# Ensure st.set_page_config is the very first Streamlit command
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="🔬",  # DNA symbol
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
    Cria um diretório de modelo único baseado no método de agregação.

    Parâmetros:
    - base_dir (str): O diretório base para os modelos.
    - aggregation_method (str): O método de agregação utilizado.

    Retorna:
    - model_dir (str): Caminho para o diretório de modelo exclusivo.
    """
    model_dir = os.path.join(base_dir, f"models_{aggregation_method}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def realign_sequences_with_mafft(input_path, output_path, threads=8):
    """
    Realinha sequências usando MAFFT.
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Realigned sequences saved in {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running MAFFT: {e.stderr.decode()}")
        sys.exit(1)

def execute_clustering(data, method="DBSCAN", eps=0.5, min_samples=5, n_clusters=3):
    """
    Executa clustering nos dados usando DBSCAN ou K-Means.

    Parâmetros:
    - data: np.ndarray com os dados para clustering.
    - method: "DBSCAN" ou "K-Means".
    - eps: Parâmetro para DBSCAN (epsilon).
    - min_samples: Parâmetro para DBSCAN.
    - n_clusters: Número de clusters para K-Means.

    Retorna:
    - labels: Labels gerados pelo método de clustering.
    """
    if method == "DBSCAN":
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "K-Means":
        if n_clusters is None:
            raise ValueError("n_clusters must be an integer for K-Means clustering.")
        clustering_model = KMeans(n_clusters=n_clusters, random_state=SEED)
    else:
        raise ValueError(f"Método de clustering inválido: {method}")

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
    Gets class rankings for the given data.
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
            "n_estimators": 100,
            "max_depth": 5,  # Reduced to prevent overfitting
            "min_samples_split": 4,  # Increased to prevent overfitting
            "min_samples_leaf": 2,
            "criterion": "entropy",
            "max_features": "log2",  # Changed from 'sqrt' to 'log2'
            "class_weight": "balanced",  # Automatic class balancing
            "max_leaf_nodes": 20,  # Adjusted for greater regularization
            "min_impurity_decrease": 0.01,
            "bootstrap": True,
            "ccp_alpha": 0.001,
            "random_state": self.seed  # Added for RandomForest
        }

        self.parameters = {
            "n_estimators": [50, 100, 150, 250],
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 4, 8, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["entropy"],
            "max_features": ["log2"],
            "class_weight": [None, "balanced"],
            "max_leaf_nodes": [5, 10, 20, 30, None],
            "min_impurity_decrease": [0.0],
            "bootstrap": [True],
            "ccp_alpha": [0.001],
        }

    def _oversample_single_sample_classes(self, X, y, target_min=5):
        """
        Oversamples classes to ensure each has at least target_min samples.
        First uses RandomOverSampler to bring classes with < target_min to target_min.
        Then uses SMOTE to balance all classes to the size of the majority class.

        Parameters:
        - X: Features.
        - y: Labels.
        - target_min: Minimum number of samples per class after RandomOverSampler.

        Returns:
        - X_smote, y_smote: Oversampled features and labels.
        """
        counter = Counter(y)
        logging.info(f"Original class distribution: {counter}")

        # Identificar classes com menos de target_min
        classes_under_min = [cls for cls, count in counter.items() if count < target_min]
        logging.info(f"Classes a serem oversampled para atingir pelo menos {target_min} amostras: {classes_under_min}")

        # Definir a estratégia para RandomOverSampler
        strategy_ros = {cls: target_min for cls in classes_under_min}

        if strategy_ros:
            ros = RandomOverSampler(sampling_strategy=strategy_ros, random_state=self.seed)
            X_ros, y_ros = ros.fit_resample(X, y)
            logging.info(f"Classe após RandomOverSampler: {Counter(y_ros)}")
        else:
            X_ros, y_ros = X, y  # Nenhuma classe precisa de oversampling via ROS

        # Agora, aplicar SMOTE para balancear todas as classes até a maior classe
        counter_ros = Counter(y_ros)
        max_class_count = max(counter_ros.values())
        logging.info(f"Máximo número de amostras em uma classe após RandomOverSampler: {max_class_count}")

        # Definir a estratégia para SMOTE: todas as classes para o tamanho da maior
        strategy_smote = {cls: max_class_count for cls in counter_ros.keys()}

        min_class_size = min(counter_ros.values())
        k_neighbors = min(min_class_size - 1, 5) if min_class_size > 1 else 1
        smote = SMOTE(sampling_strategy=strategy_smote, k_neighbors=k_neighbors, random_state=self.seed)
        logging.info(f"Using k_neighbors={k_neighbors} for SMOTE based on minimum class size={min_class_size}")

        X_smote, y_smote = smote.fit_resample(X_ros, y_ros)
        logging.info(f"Classe após SMOTE: {Counter(y_smote)}")

        # Registrar a distribuição final
        with open("oversampling_counts.txt", "a") as f:
            f.write("Class Distribution after Oversampling:\n")
            for cls, count in Counter(y_smote).items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote

    def fit(self, X, y, model_name_prefix='model', model_dir=None, min_kmers=None):
        logging.info(f"Starting fit method for {model_name_prefix}...")

        X = np.array(X)
        y = np.array(y)

        X_smote, y_smote = self._oversample_single_sample_classes(X, y, target_min=5)

        sample_counts = Counter(y_smote)
        logging.info(f"Sample counts after oversampling for {model_name_prefix}: {sample_counts}")

        with open("sample_counts_after_oversampling.txt", "a") as f:
            f.write(f"Sample Counts after Oversampling for {model_name_prefix}:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        # Ajuste dinâmico de cv
        min_class_count = min(sample_counts.values())
        old_cv = self.cv
        self.cv = min(self.cv, min_class_count)

        # Log a mudança de cv
        if old_cv != self.cv:
            logging.info(f"Adjusted cv from {old_cv} to {self.cv} based on class distribution for {model_name_prefix}.")

        # Assegurar que cv não seja menor que 2
        if self.cv < 2:
            raise ValueError(f"Adjusted cv={self.cv} is less than 2 for {model_name_prefix}. Cannot perform cross-validation.")

        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []

        fold_number = 1

        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        for train_index, test_index in skf.split(X_smote, y_smote):
            X_train, X_test = X_smote[train_index], X_smote[test_index]
            y_train, y_test = y_smote[train_index], y_smote[test_index]

            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test set class distribution: {fold_class_distribution}")

            X_train_resampled, y_train_resampled = self._oversample_single_sample_classes(X_train, y_train, target_min=5)

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
                # Binarize y_test for multiclass average_precision_score
                y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                pr_auc = average_precision_score(y_test_bin, y_pred_proba, average='macro')
            else:
                pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')  # Binary case
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
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=5, n_jobs=self.n_jobs)
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
            scoring='roc_auc_ovo',  # Ensure scoring is compatible with multiclass
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

        # Aplica oversampling em todo o conjunto de dados antes da divisão
        X_resampled, y_resampled = self._oversample_single_sample_classes(X_scaled, y, target_min=5)

        # Divide em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        # Treina o RandomForestClassifier com os melhores parâmetros
        if not self.best_params:
            logging.error("Best parameters not found. Execute fit() before test_best_RF().")
            sys.exit(1)

        model = RandomForestClassifier(**self.best_params, random_state=self.seed, n_jobs=self.n_jobs)
        model.fit(X_train, y_train)  # Treina o modelo nos dados de treinamento

        # Integração da Calibração de Probabilidade
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        # Faz as predições
        y_pred_proba = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred_proba, method='normalize')

        # Calcula o score (e.g., AUC)
        score = self._calculate_score(y_pred_adjusted, y_test)

        # Calcula métricas adicionais
        y_pred_classes = calibrated_model.predict(X_test)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(label_binarize(y_test, classes=model.classes_), y_pred_proba, average='macro')
        else:
            pr_auc = 0.0  # Não pode calcular PR AUC para uma única classe

        # Retorna o score, melhores parâmetros, modelo treinado e conjuntos de teste
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
        if classes is None:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")

        else:
            # Multiclass classification
            plt.figure()
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true == class_label, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {class_label} (area = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")

        if save_as:
            plt.savefig(save_as)
        plt.close()

    def plot_learning_curve_result(self, estimator, X, y, output_path, title='Learning Curve'):
        """
        Plots and saves the learning curve.
        """
        try:
            train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=self.cv, n_jobs=-1)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure()
            plt.title(title)
            plt.xlabel("Training examples")
            plt.ylabel("Score")

            # Plot the learning curve
            plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

            plt.legend(loc="best")

            # Save the plot if output path is specified
            if output_path:
                plt.savefig(output_path)

            plt.close()
        except Exception as e:
            logging.error(f"An error occurred while plotting the learning curve: {e}")
            raise

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
                vector_size=125,  # changed from vector_size to size
                window=10,
                min_count=1,
                workers=8,
                sg=1,
                hs=1,  # Hierarchical softmax enabled
                negative=0,  # Negative sampling disabled
                epochs=2500,  # changed from epochs to iter
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
                # Concatenate embeddings of the selected k-mers
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                # Aggregate embeddings of the selected k-mers by mean
                embedding_concatenated = np.mean(selected_embeddings, axis=0)
            elif self.aggregation_method == 'median':
                # Aggregate embeddings of the selected k-mers by median
                embedding_concatenated = np.median(selected_embeddings, axis=0)
            elif self.aggregation_method == 'sum':
                # Aggregate embeddings of the selected k-mers by sum
                embedding_concatenated = np.sum(selected_embeddings, axis=0)
            elif self.aggregation_method == 'max':
                # Aggregate embeddings of the selected k-mers by maximum
                embedding_concatenated = np.max(selected_embeddings, axis=0)
            else:
                # If method not recognized, use concatenation as default
                logging.warning(f"Unknown aggregation method '{self.aggregation_method}'. Using concatenation.")
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
    return min(max(n_samples // 10, 5), 50)

# Função para calcular a perplexidade dinamicamente
def compute_perplexity_tsne(n_samples):
    return max(5, min(50, n_samples // 100))

# Função para plotar os gráficos
def plot_dual_tsne_3d(train_embeddings, train_labels, train_protein_ids, 
                      predict_embeddings, predict_labels, predict_protein_ids, output_dir, top_n=3, class_rankings=None):
    """
    Plota dois gráficos t-SNE 3D separados:
    - Gráfico 1: Dados de Treinamento.
    - Gráfico 2: Predições.

    Parâmetros:
    - train_embeddings (np.ndarray): Embeddings dos dados de treinamento.
    - train_labels (list or array): Labels associados aos dados de treinamento.
    - train_protein_ids (list): IDs de proteínas nos dados de treinamento.
    - predict_embeddings (np.ndarray): Embeddings das predições.
    - predict_labels (list or array): Labels associados às predições.
    - predict_protein_ids (list): IDs de proteínas nas predições.
    - top_n (int): Número de top predições para exibir labels.
    - class_rankings (list of lists): Top N predições para cada predição.
    """
    # Ajustar perplexity dinamicamente
    n_samples_train = train_embeddings.shape[0]
    dynamic_perplexity_train = compute_perplexity_tsne(n_samples_train)

    # Inicializar t-SNE com perplexidade ajustada para treinamento
    tsne_train = TSNE(n_components=3, random_state=42, perplexity=dynamic_perplexity_train, n_iter=1000)
    tsne_train_result = tsne_train.fit_transform(train_embeddings)

    # Ajustar perplexity dinamicamente para predições
    n_samples_predict = predict_embeddings.shape[0]
    dynamic_perplexity_predict = compute_perplexity_tsne(n_samples_predict)

    # Inicializar t-SNE com perplexidade ajustada para predições
    tsne_predict = TSNE(n_components=3, random_state=42, perplexity=dynamic_perplexity_predict, n_iter=1000)
    tsne_predict_result = tsne_predict.fit_transform(predict_embeddings)

    # Criar mapa de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapa de cores para as predições
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Criar labels para hover (top_n predições)
    if class_rankings:
        # Garantir que class_rankings tenha a mesma quantidade que predict_protein_ids
        if len(class_rankings) != len(predict_protein_ids):
            logging.warning("Length of class_rankings does not match predict_protein_ids. Some hover texts may be incorrect.")
            # Ajustar o tamanho de class_rankings para o mínimo entre os dois
            min_length = min(len(class_rankings), len(predict_protein_ids))
            class_rankings = class_rankings[:min_length]
            predict_protein_ids = predict_protein_ids[:min_length]
        predict_hover_text = [
            f"Protein ID: {protein_id}<br>Top {top_n} Predictions: " + "<br>".join(class_rankings[i][:top_n])
            if class_rankings[i] is not None else f"Protein ID: {protein_id}<br>No Ranking Available"
            for i, protein_id in enumerate(predict_protein_ids)
        ]
    else:
        predict_hover_text = [f"Protein ID: {protein_id}" for protein_id in predict_protein_ids]

    # Gráfico 1: Dados de treinamento
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
        # IDs de proteínas reais adicionados ao campo 'text'
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

    # Gráfico 2: Predições
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
        # IDs de proteínas adicionados ao campo 'text'
        text=predict_hover_text,
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
    # Salvar gráficos em HTML
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")
    
    pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
    pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)
    
    logging.info(f"t-SNE Training Data saved as {tsne_train_html}")
    logging.info(f"t-SNE Predictions saved as {tsne_predict_html}")

    return fig_train, fig_predict

def plot_dual_umap(train_embeddings, train_labels, train_protein_ids,
                   predict_embeddings, predict_labels, predict_protein_ids, output_dir, top_n=3, class_rankings=None):
    """
    Plota dois gráficos UMAP 3D separados:
    - Gráfico 1: Dados de Treinamento.
    - Gráfico 2: Predições.

    Parâmetros:
    - train_embeddings (np.ndarray): Embeddings dos dados de treinamento.
    - train_labels (list or array): Labels associados aos dados de treinamento.
    - train_protein_ids (list): IDs de proteínas nos dados de treinamento.
    - predict_embeddings (np.ndarray): Embeddings das predições.
    - predict_labels (list or array): Labels associados às predições.
    - predict_protein_ids (list): IDs de proteínas nas predições.
    - top_n (int): Número de top predições para exibir labels.
    - class_rankings (list of lists): Top N predições para cada predição.
    """
    # Redução de dimensionalidade para treinamento
    umap_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_train_result = umap_train.fit_transform(train_embeddings)

    # Redução de dimensionalidade para predições
    umap_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_predict_result = umap_predict.fit_transform(predict_embeddings)

    # Criar mapa de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapa de cores para as predições
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Criar labels para hover (top_n predições)
    if class_rankings:
        # Garantir que class_rankings tenha a mesma quantidade que predict_protein_ids
        if len(class_rankings) != len(predict_protein_ids):
            logging.warning("Length of class_rankings does not match predict_protein_ids. Some hover texts may be incorrect.")
            # Ajustar o tamanho de class_rankings para o mínimo entre os dois
            min_length = min(len(class_rankings), len(predict_protein_ids))
            class_rankings = class_rankings[:min_length]
            predict_protein_ids = predict_protein_ids[:min_length]
        predict_hover_text = [
            f"Protein ID: {protein_id}<br>Top {top_n} Predictions: " + "<br>".join(class_rankings[i][:top_n])
            if class_rankings[i] is not None else f"Protein ID: {protein_id}<br>No Ranking Available"
            for i, protein_id in enumerate(predict_protein_ids)
        ]
    else:
        predict_hover_text = [f"Protein ID: {protein_id}" for protein_id in predict_protein_ids]

    # Gráfico 1: Dados de treinamento
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
        # IDs de proteínas reais adicionados ao campo 'text'
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

    # Gráfico 2: Predições
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
        # IDs de proteínas adicionados ao campo 'text'
        text=predict_hover_text,
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

    # Salvar gráficos em HTML
    umap_train_html = os.path.join(output_dir, "umap_train_3d.html")
    umap_predict_html = os.path.join(output_dir, "umap_predict_3d.html")
    
    pio.write_html(fig_train, file=umap_train_html, auto_open=False)
    pio.write_html(fig_predict, file=umap_predict_html, auto_open=False)
    
    logging.info(f"UMAP Training Data saved as {umap_train_html}")
    logging.info(f"UMAP Predictions saved as {umap_predict_html}")

    return fig_train, fig_predict

def generate_accuracy_pie_chart(formatted_results, table_data, output_path):
    """
    Generates a pie chart showing accuracy by category.
    """
    category_counts = Counter()
    correct_counts = Counter()
    pattern_mapping = {
        'C4-C6-C8': ['C4', 'C6', 'C8'],
        'C6-C8-C10': ['C6', 'C8', 'C10'],
        'C8-C10-C12': ['C8', 'C10', 'C12'],
        'C10-C12-C14': ['C10', 'C12', 'C14'],
        'C12-C14-C16': ['C12', 'C14', 'C16'],
        'C14-C16-C18': ['C14', 'C16', 'C18'],
    }

    for result in formatted_results:
        seq_id = result[0]
        corresponding_row = table_data[table_data['Protein.accession'].str.split().str[0] == seq_id]
        if not corresponding_row.empty:
            associated_variable_real = corresponding_row['Associated variable'].values[0]
            for category, patterns in pattern_mapping.items():
                if any(pat in result[1] for pat in patterns):
                    category_counts[category] += 1
                    if any(pat in associated_variable_real for pat in patterns):
                        correct_counts[category] += 1

    # Create pie chart
    accuracy = {category: (correct_counts[category] / category_counts[category] * 100) if category_counts[category] > 0 else 0
                for category in category_counts.keys()}

    # Remove categories with count 0 to avoid NaN in the chart
    accuracy = {k: v for k, v in accuracy.items() if category_counts[k] > 0}

    plt.figure(figsize=(8, 8))
    if accuracy:
        plt.pie(accuracy.values(), labels=[f'{key} ({val:.1f}%)' for key, val in accuracy.items()], autopct='%1.1f%%', textprops={'color': 'white'})
    else:
        logging.warning("No data to plot in the pie chart.")
    plt.title('Accuracy by Category', color='white')
    plt.tight_layout()
    plt.savefig(output_path, facecolor='#0B3C5D')  # Match the background color
    plt.close()

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
        for ranking in associated_rankings[:top_n]:
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

        # Plot points in a fixed color (e.g., dark blue)
        ax.scatter(x, [y] * len(x), color='#1f78b4', edgecolors='black', linewidth=0.5, s=100, label='_nolegend_')

        # Connect points with lines
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
    model_dir = args.model_dir  # This should be 'results/models_none' or similar based on aggregation_method

    """
    Main function coordinating the workflow.
    """
    model_dir = args.model_dir

    # Initialize progress variables
    total_steps = 21  # Adjusted total steps to account for optional clustering and bootstrap
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # =============================
    # STEP 1: Model Training para target_variable e associated_variable
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

    # Get embeddings e labels para target_variable
    X_target, y_target = protein_embedding_train.get_embeddings_and_labels(label_type='target_variable')
    logging.info(f"X_target shape: {X_target.shape}")

    # Get embeddings e labels para associated_variable
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"X_associated shape: {X_associated.shape}")

    # Full paths para modelos target_variable
    rf_model_target_full_path = os.path.join(model_dir, args.rf_model_target)
    calibrated_model_target_full_path = os.path.join(model_dir, 'calibrated_model_target.pkl')

    # Full paths para modelos associated_variable
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)  # Alteração
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')  # Alteração

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # Treinamento e Carregamento do Modelo para target_variable
    # =============================

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
        class_rankings_target = support_model_target.get_class_rankings(X_test_target)

        # Display rankings for the first 5 samples
        logging.info("Top 3 class rankings for the first 5 samples:")
        for i in range(min(5, len(class_rankings_target))):
            logging.info(f"Sample {i+1}: Class rankings - {class_rankings_target[i][:3]}")  # Shows top 3 rankings

        # Plot ROC curve
        n_classes_target = len(np.unique(y_test_target))
        if n_classes_target == 2:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)[:, 1]
        else:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)
            unique_classes_target = np.unique(y_test_target).astype(str)
        plot_roc_curve_global(y_test_target, y_pred_proba_target, 'ROC Curve for Target Variable', save_as=args.roc_curve_target, classes=unique_classes_target)

        # Convert y_test_target to integer labels
        unique_labels_target = sorted(set(y_test_target))
        label_to_int_target = {label: idx for idx, label in enumerate(unique_labels_target)}
        y_test_target_int = [label_to_int_target[label.strip()] for label in y_test_target]

        # Calculate and print ROC values for target_variable
        roc_df_target = calculate_roc_values(best_model_target, X_test_target, y_test_target_int)
        logging.info("ROC AUC Scores for target_variable:")
        logging.info(roc_df_target)
        roc_df_target.to_csv(args.roc_values_target, index=False)

        # Plot Learning Curve for target_variable
        learning_curve_target_path = os.path.join(model_dir, 'learning_curve_target.png')
        # Plot Learning Curve para target_variable
        support_model_target.plot_learning_curve_result(
            estimator=best_model_target,  # Passando o modelo treinado
            X=X_target,
            y=y_target,
            output_path=learning_curve_target_path,
            title='Learning Curve for Target Variable'
        )
        logging.info(f"Learning curve for target_variable saved at {learning_curve_target_path}")

    # =============================
    # Treinamento e Carregamento do Modelo para associated_variable
    # =============================

    # Check if calibrated model for associated_variable already exists
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable loaded from {calibrated_model_associated_full_path}")
    else:
        # Model training for associated_variable
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(X_associated, y_associated, model_name_prefix='associated', model_dir=model_dir, min_kmers=min_kmers)
        logging.info("Training and calibration for associated_variable completed.")

        # Save the calibrated model
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable saved at {calibrated_model_associated_full_path}")

        # Test the model
        best_score_assoc, best_f1_assoc, best_pr_auc_assoc, best_params_assoc, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, scaler_dir=args.model_dir)

        logging.info(f"Best ROC AUC for associated_variable: {best_score_assoc}")
        logging.info(f"Best F1 Score for associated_variable: {best_f1_assoc}")
        logging.info(f"Best Precision-Recall AUC for associated_variable: {best_pr_auc_assoc}")
        logging.info(f"Best Parameters: {best_params_assoc}")

        for param, value in best_params_assoc.items():
            logging.info(f"{param}: {value}")

        # Get class rankings
        class_rankings_associated = support_model_associated.get_class_rankings(X_test_associated)

        # Display rankings for the first 5 samples
        logging.info("Top 3 class rankings for the first 5 samples:")
        for i in range(min(5, len(class_rankings_associated))):
            logging.info(f"Sample {i+1}: Class rankings - {class_rankings_associated[i][:3]}")  # Shows top 3 rankings

        # Plot ROC curve
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            unique_classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve_global(y_test_associated, y_pred_proba_associated, 'ROC Curve for Associated Variable', save_as=args.roc_curve_associated, classes=unique_classes_associated)

        # Convert y_test_associated to integer labels
        unique_labels_associated = sorted(set(y_test_associated))
        label_to_int_associated = {label: idx for idx, label in enumerate(unique_labels_associated)}
        y_test_associated_int = [label_to_int_associated[label.strip()] for label in y_test_associated]

        # Calculate and print ROC values for associated_variable
        roc_df_associated = calculate_roc_values(best_model_associated, X_test_associated, y_test_associated_int)
        logging.info("ROC AUC Scores for associated_variable:")
        logging.info(roc_df_associated)
        roc_df_associated.to_csv(args.roc_curve_associated, index=False)  # Alteração: salvar no arquivo correto

        # Plot Learning Curve for associated_variable
        learning_curve_associated_path = os.path.join(model_dir, 'learning_curve_associated.png')
        # Plot Learning Curve para associated_variable
        support_model_associated.plot_learning_curve_result(
            estimator=best_model_associated,  # Passando o modelo treinado
            X=X_associated,
            y=y_associated,
            output_path=learning_curve_associated_path,
            title='Learning Curve for Associated Variable'
        )
        logging.info(f"Learning curve for associated_variable saved at {learning_curve_associated_path}")

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # STEP 2: Clustering e Análise de Bootstrap (Opcional)
    # =============================

    # Verificar se o usuário deseja realizar clustering
    if args.perform_clustering:
        st.markdown("<span style='color:white'>Performing clustering...</span>", unsafe_allow_html=True)
        # Escolher o método de clustering
        clustering_method = args.clustering_method
        logging.info(f"Selected clustering method: {clustering_method}")

        # Executar clustering nos embeddings de treinamento
        if clustering_method == "DBSCAN":
            labels_train = execute_clustering(
                data=X_target,
                method=clustering_method,
                eps=args.eps,
                min_samples=args.min_samples,
                n_clusters=None  # Not used in DBSCAN
            )
        else:
            labels_train = execute_clustering(
                data=X_target,
                method=clustering_method,
                eps=None,
                min_samples=None,
                n_clusters=args.n_clusters
            )
        logging.info(f"Clustering labels for training data: {labels_train}")

        # Executar clustering nos embeddings de predição
        if clustering_method == "DBSCAN":
            labels_predict = execute_clustering(
                data=X_associated,
                method=clustering_method,
                eps=args.eps,
                min_samples=args.min_samples,
                n_clusters=None  # Not used in DBSCAN
            )
        else:
            labels_predict = execute_clustering(
                data=X_associated,
                method=clustering_method,
                eps=None,
                min_samples=None,
                n_clusters=args.n_clusters
            )
        logging.info(f"Clustering labels for prediction data: {labels_predict}")

        # Atualizar resultados com labels de clustering
        for i, entry in enumerate(protein_embedding_train.embeddings):
            protein_embedding_train.embeddings[i]['clustering_label'] = labels_train[i]

        for i, entry in enumerate(protein_embedding_train.embeddings):  # Correção: agora iterando sobre associated embeddings
            protein_embedding_train.embeddings[i]['clustering_label'] = labels_predict[i]

        # Análise de Bootstrap para Significância dos Clusters
        logging.info("Starting bootstrap analysis for cluster significance...")
        bootstrap_iterations = args.bootstrap_iterations
        ari_scores = []
        silhouette_scores = []

        for i in range(bootstrap_iterations):
            # Resample com reposição
            X_resampled, y_resampled = resample(X_target, labels_train, replace=True, random_state=SEED + i)
            # Reaplicar clustering
            if clustering_method == "DBSCAN":
                labels_resampled = execute_clustering(
                    data=X_resampled,
                    method=clustering_method,
                    eps=args.eps,
                    min_samples=args.min_samples,
                    n_clusters=None
                )
            else:
                labels_resampled = execute_clustering(
                    data=X_resampled,
                    method=clustering_method,
                    eps=None,
                    min_samples=None,
                    n_clusters=args.n_clusters
                )
            # Calcular Adjusted Rand Index (ARI) entre labels originais e resampled
            ari = adjusted_rand_score(y_resampled, labels_resampled)
            ari_scores.append(ari)

            # Calcular Coeficiente de Silhueta
            if len(set(labels_resampled)) > 1:
                sil_score = silhouette_score(X_resampled, labels_resampled)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

        # Calcular média e intervalo de confiança para ARI
        ari_mean = np.mean(ari_scores)
        ari_ci_lower = np.percentile(ari_scores, 2.5)
        ari_ci_upper = np.percentile(ari_scores, 97.5)

        # Calcular média e intervalo de confiança para Coeficiente de Silhueta
        sil_mean = np.mean(silhouette_scores)
        sil_ci_lower = np.percentile(silhouette_scores, 2.5)
        sil_ci_upper = np.percentile(silhouette_scores, 97.5)

        logging.info(f"Bootstrap ARI Mean: {ari_mean:.4f}")
        logging.info(f"Bootstrap ARI 95% CI: [{ari_ci_lower:.4f}, {ari_ci_upper:.4f}]")
        logging.info(f"Bootstrap Silhouette Mean: {sil_mean:.4f}")
        logging.info(f"Bootstrap Silhouette 95% CI: [{sil_ci_lower:.4f}, {sil_ci_upper:.4f}]")

        # Exibir resultados da análise de bootstrap
        st.markdown(f"<span style='color:white'>Bootstrap ARI Mean: {ari_mean:.4f}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:white'>Bootstrap ARI 95% CI: [{ari_ci_lower:.4f}, {ari_ci_upper:.4f}]</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:white'>Bootstrap Silhouette Mean: {sil_mean:.4f}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:white'>Bootstrap Silhouette 95% CI: [{sil_ci_lower:.4f}, {sil_ci_upper:.4f}]</span>", unsafe_allow_html=True)

        # Determinar significância dos clusters
        # Por exemplo, considerar clusters significativos se ARI > 0.5 e Silhouette > 0.3
        ari_significant = ari_mean > 0.5
        sil_significant = sil_mean > 0.3

        if ari_significant and sil_significant:
            significance_message = "Os clusters formados são significativos com base na análise de Bootstrap."
        else:
            significance_message = "Os clusters formados NÃO são significativos com base na análise de Bootstrap."

        st.markdown(f"<span style='color:white'>{significance_message}</span>", unsafe_allow_html=True)

        # Update progress
        current_step += 1
        progress = min(current_step / total_steps, 1.0)
        progress_bar.progress(progress)
        progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
        time.sleep(0.1)
    else:
        logging.info("Clustering not performed as per user selection.")

    # =============================
    # STEP 3: Classifying New Sequences
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

    # Load data para predição
    predict_alignment_path = args.predict_fasta

    # Check if sequences para predição estão alinhadas
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

    # Initialize ProteinEmbedding para predição, sem a tabela
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

    # Get embeddings para predição
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

    # Load the calibrated models from disk
    try:
        calibrated_model_target = joblib.load(calibrated_model_target_full_path)
        logging.info(f"Calibrated model loaded from {calibrated_model_target_full_path}")
    except Exception as e:
        logging.error(f"Error loading calibrated_model_target: {e}")
        sys.exit(1)

    try:
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Calibrated model loaded from {calibrated_model_associated_full_path}")
    except Exception as e:
        logging.error(f"Error loading calibrated_model_associated: {e}")
        sys.exit(1)

    # Check feature size before prediction
    try:
        n_features_target = calibrated_model_target.n_features_in_
        n_features_associated = calibrated_model_associated.n_features_in_
    except AttributeError:
        logging.error("Could not access n_features_in_ from calibrated models.")
        sys.exit(1)

    if n_features_target is not None and X_predict_scaled.shape[1] > n_features_target:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {n_features_target} to match the model's input size.")
        X_predict_scaled = X_predict_scaled[:, :n_features_target]

    # Perform prediction for target_variable
    predictions_target = calibrated_model_target.predict(X_predict_scaled)

    # Check and adjust feature size for associated_variable
    if n_features_associated is not None and X_predict_scaled.shape[1] > n_features_associated:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {n_features_associated} to match the model's input size for associated_variable.")
        X_predict_scaled = X_predict_scaled[:, :n_features_associated]

    # Perform prediction for associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)

    # Get class rankings
    rankings_target = get_class_rankings_global(calibrated_model_target, X_predict_scaled)
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled)

    # Process and save results
    results = {}
    for entry, pred_target, pred_associated, ranking_target, ranking_associated in zip(protein_embedding_predict.embeddings, predictions_target, predictions_associated, rankings_target, rankings_associated):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "target_prediction": pred_target,
            "associated_prediction": pred_associated,
            "target_ranking": ranking_target,
            "associated_ranking": ranking_associated
        }

    # Save results to a file
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tTarget_Prediction\tAssociated_Prediction\tTarget_Ranking\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['target_prediction']}\t{result['associated_prediction']}\t{'; '.join(result['target_ranking'])}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Target Variable: {result['target_prediction']}, Associated Variable: {result['associated_prediction']}, Target Ranking: {'; '.join(result['target_ranking'])}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

    # Format the results
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info['associated_ranking']
        formatted_prob_sums = format_and_sum_probabilities(associated_rankings)
        formatted_results.append([sequence_id, formatted_prob_sums])

    # Log to check the content of formatted_results
    logging.info("Formatted Results:")
    for result in formatted_results:
        logging.info(result)

    # Print the results in a formatted table
    headers = ["Protein Accession", "Associated Prob. Rankings"]
    logging.info(tabulate(formatted_results, headers=headers, tablefmt="grid"))

    # Save the results to an Excel file
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Results saved at {args.excel_output}")

    # Save the table in tabulated format
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved at {args.formatted_results_table}")

    # =============================
    # STEP 5: Implementação das Novas Funcionalidades
    # =============================
    # Implementação 2: Selecionar os kmers consenso e gerar arquivo tabular
    # Extrair feature importances
    logging.info("Extracting feature importances for associated_variable...")
    try:
        # Acessar o estimador base dentro do CalibratedClassifierCV
        base_estimator_associated = calibrated_model_associated.calibrated_classifiers_[0].base_estimator_
        feature_importances = base_estimator_associated.feature_importances_
    except AttributeError:
        logging.error("Could not access feature_importances_ from the base estimator of calibrated_model_associated.")
        sys.exit(1)
    except IndexError:
        logging.error("calibrated_classifiers_ list is empty in calibrated_model_associated.")
        sys.exit(1)

    # Map feature indices to actual kmers
    # Assuming that the order of features corresponds to the order of kmers used in training
    # This requires that the embedding generator maintains a consistent feature ordering
    # For Word2Vec embeddings, each kmer is represented by its vector, so features are concatenated or aggregated
    # Thus, to map back, need to know the aggregation method

    # For simplicity, assuming 'mean' aggregation, so each feature corresponds to one kmer
    if args.aggregation_method == 'none':
        # Concatenated embeddings: features are multiple kmers
        # Need to extract feature importances per kmer by averaging across concatenated parts
        vector_size = calibrated_model_associated.n_features_in_ // min_kmers_loaded
        kmer_importances = {}
        for i in range(min_kmers_loaded):
            start = i * vector_size
            end = (i + 1) * vector_size
            kmer_importance = feature_importances[start:end].mean()
            kmer_importances[f'kmer_{i}'] = kmer_importance
    else:
        # Aggregated embeddings: each feature corresponds to one kmer
        kmer_importances = {f'kmer_{i}': imp for i, imp in enumerate(feature_importances)}

    # Sort kmers by importance
    sorted_kmers = sorted(kmer_importances.items(), key=lambda x: x[1], reverse=True)

    # Select top N kmers as consensus
    top_n_kmers = [kmer for kmer, imp in sorted_kmers[:10]]  # Selecting top 10 kmers

    logging.info(f"Top 10 consensus kmers: {top_n_kmers}")

    # Map kmers to protein IDs
    kmer_to_proteins = {kmer: [] for kmer in top_n_kmers}
    for entry in protein_embedding_train.embeddings:
        protein_id = entry['protein_accession']
        kmers = entry.get('kmers', [])
        for kmer in kmers:
            if kmer in top_n_kmers:
                kmer_to_proteins[kmer].append(protein_id)

    # Prepare tabular data
    consensus_kmers_table = []
    for kmer, proteins in kmer_to_proteins.items():
        consensus_kmers_table.append({
            'Associated Variable': 'associated_variable',
            'Consensus K-mer': kmer,
            'Protein IDs': ', '.join(proteins)
        })

    # Convert to DataFrame
    df_consensus_kmers = pd.DataFrame(consensus_kmers_table)

    # Save the table to a CSV file
    consensus_kmers_output_path = os.path.join(model_dir, 'consensus_kmers_associated_variable.csv')
    df_consensus_kmers.to_csv(consensus_kmers_output_path, index=False)
    logging.info(f"Consensus kmers table saved at {consensus_kmers_output_path}")

    # Exibir tabela no Streamlit
    st.header("Consensus K-mers for Associated Variable")
    st.dataframe(df_consensus_kmers)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # STEP 4: Dimensionality Reduction e Plotagem t-SNE & UMAP
    # =============================
    try:
        logging.info("Generating dual t-SNE and UMAP 3D plots for training data and predictions...")

        # Coletar embeddings e labels para dados de treinamento
        combined_embeddings_train = np.array([entry['embedding'] for entry in protein_embedding_train.embeddings])
        combined_labels_train = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]
        combined_protein_ids_train = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]

        # Coletar embeddings e labels para predições
        combined_embeddings_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])
        combined_labels_predict = predictions_associated  # Usa as predições de associated_variable
        combined_protein_ids_predict = [entry['protein_accession'] for entry in protein_embedding_predict.embeddings]

        # Opções de clustering no Streamlit
        if args.perform_clustering:
            st.sidebar.header("Clustering Analysis")
            st.sidebar.markdown(f"<span style='color:white'>Clustering Method: {args.clustering_method}</span>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<span style='color:white'>Bootstrap ARI Mean: {ari_mean:.4f}</span>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<span style='color:white'>Bootstrap ARI 95% CI: [{ari_ci_lower:.4f}, {ari_ci_upper:.4f}]</span>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<span style='color:white'>Bootstrap Silhouette Mean: {sil_mean:.4f}</span>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<span style='color:white'>Bootstrap Silhouette 95% CI: [{sil_ci_lower:.4f}, {sil_ci_upper:.4f}]</span>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<span style='color:white'>Cluster Significance: {significance_message}</span>", unsafe_allow_html=True)

        generate_tsne_umap = args.generate_tsne_umap

        if generate_tsne_umap:
            # Plotar t-SNE 3D
            st.header("t-SNE 3D Visualization")
            fig_train_tsne, fig_predict_tsne = plot_dual_tsne_3d(
                train_embeddings=combined_embeddings_train,
                train_labels=combined_labels_train,
                train_protein_ids=combined_protein_ids_train,
                predict_embeddings=combined_embeddings_predict,
                predict_labels=combined_labels_predict,
                predict_protein_ids=combined_protein_ids_predict,
                output_dir=args.output_dir,
                top_n=3,
                class_rankings=[results[pid]['associated_ranking'] for pid in combined_protein_ids_predict]
            )
            # Exibir os gráficos separados no Streamlit
            st.plotly_chart(fig_train_tsne, use_container_width=True)  # Gráfico dos dados de treinamento
            st.plotly_chart(fig_predict_tsne, use_container_width=True)  # Gráfico das predições

            # Plotar UMAP 3D
            st.header("UMAP 3D Visualization")
            fig_train_umap, fig_predict_umap = plot_dual_umap(
                train_embeddings=combined_embeddings_train,
                train_labels=combined_labels_train,
                train_protein_ids=combined_protein_ids_train,  # IDs reais das proteínas do treinamento
                predict_embeddings=combined_embeddings_predict,
                predict_labels=combined_labels_predict,
                predict_protein_ids=combined_protein_ids_predict,
                output_dir=args.output_dir,
                top_n=3,
                class_rankings=[results[pid]['associated_ranking'] for pid in combined_protein_ids_predict]
            )

            st.plotly_chart(fig_train_umap, use_container_width=True)
            st.plotly_chart(fig_predict_umap, use_container_width=True)
            logging.info("t-SNE and UMAP 3D plots generated successfully.")
        else:
            logging.info("Dimensionality reduction plots not generated as per user selection.")
    except Exception as e:
        logging.error(f"Failed during dimensionality reduction and plotting: {e}")
        st.error(f"Failed during dimensionality reduction and plotting: {e}")
        sys.exit(1)

    # =============================
    # STEP 6: Conclusão e Download de Resultados
    # =============================

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

    # Update progress to 100%
    progress_bar.progress(1.0)
    progress_text.markdown("<span style='color:black'>Progress: 100%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    return results  # Retornar results para uso posterior

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
    unsafe_allow_html=True,
)

from PIL import Image
# Função para converter a imagem em base64
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
# Usando HTML com st.markdown para alinhar título e texto

st.markdown(
    f"""
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 30px; background: linear-gradient(to bottom, #f9f9f9, #ffffff); border-radius: 15px; border: 2px solid #dddddd; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); position: relative;">
        <p style="color: black; font-size: 1.5em; font-weight: bold; margin: 0;">
            FAALPred: Predicting Fatty Acid Specificities of Fatty Acyl-AMP Ligases (FAALs) Using Integrated Approaches of Neural Networks, Bioinformatics, and Machine Learning
        </p>
        <p style="color: #2c3e50; font-size: 1.2em; font-weight: normal; margin-top: 10px;">
            Anne Liong, Leandro de Mattos Pereira, and Pedro Leão
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
            treatment (Kurt et al., 2017; Gilmore et al., 2012). Dr. Pedro Leão and His Team Identified Several of These Natural Products in Cyanobacteria (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" style="color: #3498db; text-decoration: none;">visit here</a>), 
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
    options=['none', 'mean', 'median', 'sum', 'max'],
    index=0
)

# Entrada opcional para parâmetros do Word2Vec
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
        "Epochs", min_value=1, max_value=3500, value=2500, step=100
    )
else:
    window = 10  # Valor padrão
    workers = 8  # Valor padrão
    epochs = 2500  # Valor padrão

# Opções de Clustering e Bootstrap
st.sidebar.header("Clustering and Bootstrap Options")
perform_clustering = st.sidebar.checkbox("Perform Clustering (DBSCAN/K-Means)", value=False)
if perform_clustering:
    clustering_method = st.sidebar.selectbox("Clustering Method", options=["DBSCAN", "K-Means"], index=0)
    if clustering_method == "DBSCAN":
        eps = st.sidebar.slider("DBSCAN Epsilon (eps)", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
        min_samples = st.sidebar.slider("DBSCAN Min Samples", min_value=1, max_value=20, step=1, value=5)
        n_clusters = None  # Not used in DBSCAN
    else:
        n_clusters = st.sidebar.slider("K-Means Number of Clusters", min_value=2, max_value=15, step=1, value=3)
        eps = None  # Not used in K-Means
        min_samples = None  # Not used in K-Means
    bootstrap_iterations = st.sidebar.number_input("Bootstrap Iterations for Cluster Significance", min_value=10, max_value=1000, value=100, step=10)
else:
    clustering_method = None
    eps = None
    min_samples = None
    n_clusters = None
    bootstrap_iterations = None

# Adicionar input para min_kmers
st.sidebar.header("Additional Parameters")
min_kmers_input = st.sidebar.number_input("Minimum Number of K-mers for Feature Importance Mapping", min_value=1, max_value=10, value=3, step=1)

# Output directory
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
        rf_model_associated="rf_model_associated.pkl",  # Alteração
        word2vec_model="word2vec_model.bin",
        scaler="scaler.pkl",
        model_dir=model_dir,
        perform_clustering=perform_clustering,
        clustering_method=clustering_method,
        eps=eps,
        min_samples=min_samples,
        n_clusters=n_clusters,
        bootstrap_iterations=bootstrap_iterations,
        generate_tsne_umap=True,  # Assuming always generate, can be made configurable
        min_kmers=min_kmers_input  # Correção: garantir que min_kmers seja atribuído corretamente
    )

    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Run the main analysis function
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        results = main(args)

        st.success("Analysis completed successfully!")

        # Display scatterplot
        st.header("Scatterplot of Predictions")
        scatterplot_path = os.path.join(args.output_dir, "scatterplot_predictions.png")
        st.image(scatterplot_path, use_column_width=True)

        # Exibir tabela formatada
        st.header("Formatted Results Table")
        # Verificar se o arquivo existe e não está vazio
        formatted_table_path = args.formatted_results_table

        if os.path.exists(formatted_table_path) and os.path.getsize(formatted_table_path) > 0:
            try:
                # Abrir e ler o conteúdo do arquivo
                with open(formatted_table_path, 'r') as f:
                    formatted_table = f.read()

                # Exibir o conteúdo no Streamlit
                st.text(formatted_table)
            except Exception as e:
                st.error(f"An error occurred while reading the formatted results table: {e}")
        else:
            st.error(f"Formatted results table not found or is empty: {formatted_table_path}")

        # Generate the Scatterplot of Predictions
        logging.info("Generating scatter plot of new sequences predictions...")
        plot_predictions_scatterplot_custom(results, args.scatterplot_output)
        logging.info(f"Scatterplot saved at {args.scatterplot_output}")

        # =============================
        # STEP 4: Dimensionality Reduction e Plotagem t-SNE & UMAP
        # =============================
        if args.perform_clustering:
            st.header("Clustering Results")

            # Carregar labels de clustering para treinamento e predição
            labels_train = [entry.get('clustering_label', None) for entry in protein_embedding_train.embeddings]
            labels_predict = [entry.get('clustering_label', None) for entry in protein_embedding_predict.embeddings]


            # Exibir distribuição dos clusters no treinamento
            cluster_counts_train = Counter(labels_train)
            st.subheader("Cluster Distribution in Training Data")
            st.bar_chart(pd.Series(cluster_counts_train))

            # Exibir distribuição dos clusters nas predições
            cluster_counts_predict = Counter(labels_predict)
            st.subheader("Cluster Distribution in Prediction Data")
            st.bar_chart(pd.Series(cluster_counts_predict))

        # =============================
        # STEP 2: Dimensionality Reduction e Plotagem t-SNE & UMAP
        # =============================
        try:
            logging.info("Generating dual t-SNE and UMAP 3D plots for training data and predictions...")

            # Coletar embeddings e labels para dados de treinamento
            combined_embeddings_train = np.array([entry['embedding'] for entry in protein_embedding_train.embeddings])
            combined_labels_train = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]
            combined_protein_ids_train = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]

            # Coletar embeddings e labels para predições
            combined_embeddings_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])
            combined_labels_predict = predictions_associated  # Usa as predições de associated_variable
            combined_protein_ids_predict = [entry['protein_accession'] for entry in protein_embedding_predict.embeddings]

            # Opções de clustering no Streamlit
            if args.perform_clustering:
                st.sidebar.header("Clustering Analysis")
                st.sidebar.markdown(f"<span style='color:white'>Clustering Method: {args.clustering_method}</span>", unsafe_allow_html=True)
                st.sidebar.markdown(f"<span style='color:white'>Bootstrap ARI Mean: {ari_mean:.4f}</span>", unsafe_allow_html=True)
                st.sidebar.markdown(f"<span style='color:white'>Bootstrap ARI 95% CI: [{ari_ci_lower:.4f}, {ari_ci_upper:.4f}]</span>", unsafe_allow_html=True)
                st.sidebar.markdown(f"<span style='color:white'>Bootstrap Silhouette Mean: {sil_mean:.4f}</span>", unsafe_allow_html=True)
                st.sidebar.markdown(f"<span style='color:white'>Bootstrap Silhouette 95% CI: [{sil_ci_lower:.4f}, {sil_ci_upper:.4f}]</span>", unsafe_allow_html=True)
                st.sidebar.markdown(f"<span style='color:white'>Cluster Significance: {significance_message}</span>", unsafe_allow_html=True)

            generate_tsne_umap = args.generate_tsne_umap

            if generate_tsne_umap:
                # Plotar t-SNE 3D
                st.header("t-SNE 3D Visualization")
                fig_train_tsne, fig_predict_tsne = plot_dual_tsne_3d(
                    train_embeddings=combined_embeddings_train,
                    train_labels=combined_labels_train,
                    train_protein_ids=combined_protein_ids_train,
                    predict_embeddings=combined_embeddings_predict,
                    predict_labels=combined_labels_predict,
                    predict_protein_ids=combined_protein_ids_predict,
                    output_dir=args.output_dir,
                    top_n=3,
                    class_rankings=[results[pid]['associated_ranking'] for pid in combined_protein_ids_predict]
                )
                # Exibir os gráficos separados no Streamlit
                st.plotly_chart(fig_train_tsne, use_container_width=True)  # Gráfico dos dados de treinamento
                st.plotly_chart(fig_predict_tsne, use_container_width=True)  # Gráfico das predições

                # Plotar UMAP 3D
                st.header("UMAP 3D Visualization")
                fig_train_umap, fig_predict_umap = plot_dual_umap(
                    train_embeddings=combined_embeddings_train,
                    train_labels=combined_labels_train,
                    train_protein_ids=combined_protein_ids_train,  # IDs reais das proteínas do treinamento
                    predict_embeddings=combined_embeddings_predict,
                    predict_labels=combined_labels_predict,
                    predict_protein_ids=combined_protein_ids_predict,
                    output_dir=args.output_dir,
                    top_n=3,
                    class_rankings=[results[pid]['associated_ranking'] for pid in combined_protein_ids_predict]
                )

                st.plotly_chart(fig_train_umap, use_container_width=True)
                st.plotly_chart(fig_predict_umap, use_container_width=True)
                logging.info("t-SNE and UMAP 3D plots generated successfully.")
            else:
                logging.info("Dimensionality reduction plots not generated as per user selection.")
        except Exception as e:
            logging.error(f"Failed during dimensionality reduction and plotting: {e}")
            st.error(f"Failed during dimensionality reduction and plotting: {e}")
            sys.exit(1)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())

# Função para carregar e redimensionar imagens com ajuste de DPI
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

# Definições dos caminhos das imagens
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

# HTML para exibição das imagens
footer_html = """
<div class="support-text">Support by:</div>
<div class="footer-container">
    {}
</div>
<div class="footer-text">
    CIIMAR - Pedro Leão @CNP - 2024 - Leandro de Mattos Pereira (developer) - All rights reserved.
</div>
"""

# Gerar tags <img> para cada imagem
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Renderizar o rodapé
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)


