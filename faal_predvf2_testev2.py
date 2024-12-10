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
import catboost
import plotly.io as pio
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, average_precision_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from matplotlib import ticker
import umap.umap_ as umap  # Import for UMAP
import base64
from plotly.graph_objs import Figure
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import optuna  # Import for Bayesian Search
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
import lightgbm as lgb
import xgboost as xgb

# ============================================
# Function and Class Definitions
# ============================================

# Setting seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),  # Log to file for persistent records
    ],
)

# ============================================
# Streamlit Configuration and Interface
# ============================================

# Ensure st.set_page_config is the first Streamlit command
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="ðŸ”¬",  # DNA symbol
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

def realign_sequences_with_mafft(input_path: str, output_path: str, threads: int = 1) -> None:
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Realigned sequences saved in {output_path}")
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

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
    lw = 2
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
    else:  # Multiclass
        y_bin = label_binarize(y_true, classes=unique_classes)
        n_classes = y_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        roc_df = pd.DataFrame(list(roc_auc.items()), columns=['Class', 'ROC AUC'])
        return roc_df

        plt.figure()

        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            class_label = classes[i] if classes is not None else unique_classes[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC Curve for class {class_label} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.title(title, color='white')
    plt.legend(loc="lower right")
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')
    plt.close()

def get_class_rankings(model, X: np.ndarray) -> list:
    if model is None:
        raise ValueError("Model not trained. Please train the model first.")

    y_pred_proba = model.predict_proba(X)
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings

def calculate_roc_values(model, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
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
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.hstack([y_original, y_synthetic])

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
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3"
        ),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )

    if output_dir:
        umap_similarity_path = os.path.join(output_dir, "umap_similarity_3D.html")
        fig.write_html(umap_similarity_path)
        logging.info(f"UMAP plot saved in {umap_similarity_path}")

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
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob

    if not category_sums:
        return None, None, None

    top_category, top_sum = max(category_sums.items(), key=lambda x: x[1])
    sorted_categories = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)
    top_two = sorted_categories[:2] if len(sorted_categories) >= 2 else sorted_categories
    top_two_categories = [f"{cat} ({prob:.2f}%)" for cat, prob in top_two]
    top_category_with_confidence = f"{top_category} ({top_sum:.2f}%)"

    return top_category_with_confidence, top_sum, top_two_categories

class Support:
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
        self.best_model = None

        self.initial_parameters = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "criterion": "entropy",
            "max_features": "sqrt",
            "class_weight": "balanced",
            "ccp_alpha": 0.01,
       }

        self.grid_search_parameters = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample", None],
            "ccp_alpha": [0.0, 0.001, 0.01],
        }

    def _oversample_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        logging.info("Starting oversampling process with ADASYN...")
        try:
            adasyn = ADASYN(sampling_strategy='auto', random_state=self.seed)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            logging.info(f"Class distribution after ADASYN: {Counter(y_resampled)}")
        except ValueError as e:
            logging.error(f"Error during ADASYN: {e}")
            sys.exit(1)

        return X_resampled, y_resampled

    def _perform_random_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray) -> tuple:
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)
        randomized_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.seed),
            param_distributions=self.grid_search_parameters,
            n_iter=50,
            cv=skf,
            scoring='roc_auc_ovo',
            verbose=1,
            random_state=self.seed,
            n_jobs=self.n_jobs
        )

        randomized_search.fit(X_train_resampled, y_train_resampled)
        logging.info(f"Best parameters from randomized search: {randomized_search.best_params_}")
        return randomized_search.best_estimator_, randomized_search.best_params_

    def _perform_bayesian_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray, n_trials: int = 50) -> tuple:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
                "ccp_alpha": trial.suggest_loguniform("ccp_alpha", 1e-4, 1e-1),
            }
            model = RandomForestClassifier(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                **params
            )
            skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)
            roc_auc = cross_val_score(model, X_train_resampled, y_train_resampled, cv=skf, scoring='roc_auc_ovo').mean()
            return roc_auc

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logging.info(f"Best parameters from Bayesian optimization: {best_params}")

        best_model = RandomForestClassifier(
            random_state=self.seed,
            n_jobs=self.n_jobs,
            **best_params
        )
        best_model.fit(X_train_resampled, y_train_resampled)
        return best_model, best_params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protein_ids: list = None,
        var_assoc: list = None,
        model_name_prefix: str = 'model',
        model_dir: str = None,
        min_kmers: int = None
    ) -> dict:
        logging.info(f"Starting fit method for {model_name_prefix}...")
        X = np.array(X)
        y = np.array(y)

        if min_kmers is not None:
            logging.info(f"Using provided min_kmers: {min_kmers}")
        else:
            min_kmers = len(X)
            logging.info(f"min_kmers not provided. Set to the size of X: {min_kmers}")

        X_resampled, y_resampled = self._oversample_data(X, y)

        trained_models = {}
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        model_types = ['random_forest', 'lightgbm', 'xgboost', 'catboost']

        for model_type in model_types:
            logging.info(f"Training model: {model_type}")
            self.train_scores = []
            self.test_scores = []
            self.f1_scores = []
            self.pr_auc_scores = []

            for fold_number, (train_index, test_index) in enumerate(skf.split(X_resampled, y_resampled), start=1):
                X_train, X_test = X_resampled[train_index], X_resampled[test_index]
                y_train, y_test = y_resampled[train_index], y_resampled[test_index]

                X_train_resampled, y_train_resampled = self._oversample_data(X_train, y_train)

                if model_type == 'random_forest':
                    model = RandomForestClassifier(
                        **self.initial_parameters,
                        random_state=self.seed,
                        n_jobs=self.n_jobs
                    )
                elif model_type == 'lightgbm':
                    model = LGBMClassifier(
                        **self.initial_parameters,
                        random_state=self.seed,
                        n_jobs=self.n_jobs
                    )
                elif model_type == 'xgboost':
                    model = XGBClassifier(
                        **self.initial_parameters,
                        use_label_encoder=False,
                        eval_metric='mlogloss',
                        random_state=self.seed,
                        n_jobs=self.n_jobs
                    )
                elif model_type == 'catboost':
                    model = CatBoostClassifier(
                        **self.initial_parameters,
                        verbose=0,
                        random_state=self.seed,
                        thread_count=self.n_jobs
                    )
                else:
                    logging.error(f"Invalid model type: {model_type}")
                    continue

                model.fit(X_train_resampled, y_train_resampled)

                train_score = model.score(X_train_resampled, y_train_resampled)
                test_score = model.score(X_test, y_test)
                y_pred = model.predict(X_test)

                f1 = f1_score(y_test, y_pred, average='weighted')
                self.f1_scores.append(f1)
                self.train_scores.append(train_score)
                self.test_scores.append(test_score)

                if len(np.unique(y_test)) > 1:
                    y_pred_proba = model.predict_proba(X_test)
                    pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')
                else:
                    pr_auc = 0.0
                self.pr_auc_scores.append(pr_auc)

                logging.info(f"Fold {fold_number} [{model_type}]: Training Score: {train_score}")
                logging.info(f"Fold {fold_number} [{model_type}]: Test Score: {test_score}")
                logging.info(f"Fold {fold_number} [{model_type}]: F1 Score: {f1}")
                logging.info(f"Fold {fold_number} [{model_type}]: Precision-Recall AUC: {pr_auc}")

                calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
                calibrator.fit(X_train_resampled, y_train_resampled)
                self.model = calibrator

                if model_dir:
                    best_model_filename = os.path.join(model_dir, f'model_best_{model_type}_fold{fold_number}.pkl')
                    os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                    joblib.dump(calibrator, best_model_filename)
                    logging.info(f"Calibrated model saved as {best_model_filename} for {model_type} Fold {fold_number}")
                else:
                    best_model_filename = f'model_best_{model_type}_fold{fold_number}.pkl'
                    joblib.dump(calibrator, best_model_filename)
                    logging.info(f"Calibrated model saved as {best_model_filename} for {model_type} Fold {fold_number}")

            logging.info(f"Starting Bayesian Optimization for hyperparameter tuning of {model_type} model...")
            best_model, best_params = self._perform_bayesian_search(X_resampled, y_resampled, n_trials=50)
            self.best_params = best_params
            self.model = best_model

            logging.info(f"Best parameters from Bayesian Optimization for {model_type}: {best_params}")

            if model_dir:
                best_model_filename = os.path.join(model_dir, f'model_best_bayesian_{model_type}.pkl')
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best Bayesian model saved as {best_model_filename} for {model_type}")
            else:
                best_model_filename = f'model_best_bayesian_{model_type}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best Bayesian model saved as {best_model_filename} for {model_type}")

            best_score, best_f1, best_pr_auc, best_params, calibrated_model, X_test, y_test = self.test_best_model(
                X_resampled,
                y_resampled,
                model_type=model_type
            )

            logging.info(f"Best ROC AUC for {model_type}: {best_score}")
            logging.info(f"Best F1 Score for {model_type}: {best_f1}")
            logging.info(f"Best Precision-Recall AUC for {model_type}: {best_pr_auc}")
            logging.info(f"Best Parameters: {best_params}")

            for param, value in best_params.items():
                logging.info(f"{param}: {value}")

            class_rankings = self.get_class_rankings(calibrated_model, X_test)
            logging.info(f"Class rankings for {model_type} generated successfully.")

            rf_model_full_path = os.path.join(model_dir, f'rf_model_best_bayesian_{model_type}.pkl')
            joblib.dump(calibrated_model, rf_model_full_path)
            logging.info(f"Calibrated model for {model_type} saved in {rf_model_full_path}")

            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
                classes = None
            else:
                y_pred_proba = calibrated_model.predict_proba(X_test)
                classes = np.unique(y_test).astype(str)
            plot_roc_curve(
                y_test,
                y_pred_proba,
                f'ROC Curve for {model_type}',
                save_as=os.path.join(model_dir, f'roc_curve_{model_type}.png'),
                classes=classes
            )

            trained_models[model_type] = {
                'model': calibrated_model,
                'best_params': best_params,
                'roc_curve_path': os.path.join(model_dir, f'roc_curve_{model_type}.png'),
                'class_rankings': class_rankings
            }

        for model_type in model_types:
            logging.info(f"Plotting Learning Curve for {model_type}")
            learning_curve_path = os.path.join(model_dir, f'learning_curve_{model_type}.png')
            self.plot_learning_curve(learning_curve_path)

        return trained_models

    def test_best_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler_dir: str = '.',
        model_type: str = 'random_forest'
    ) -> tuple:
        scaler_path = os.path.join(scaler_dir, 'scaler_associated.pkl') if scaler_dir else 'scaler_associated.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found in {scaler_path}. It should be scaler_associated.pkl.")
            sys.exit(1)

        X_scaled = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.4, random_state=self.seed, stratify=y
        )

        if model_type == 'random_forest':
            model = RandomForestClassifier(
                **self.best_params,
                random_state=self.seed,
                n_jobs=self.n_jobs
            )
        elif model_type == 'lightgbm':
            model = LGBMClassifier(
                **self.best_params,
                random_state=self.seed,
                n_jobs=self.n_jobs
            )
        elif model_type == 'xgboost':
            model = XGBClassifier(
                **self.best_params,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=self.seed,
                n_jobs=self.n_jobs
            )
        elif model_type == 'catboost':
            model = CatBoostClassifier(
                **self.best_params,
                verbose=0,
                random_state=self.seed,
                thread_count=self.n_jobs
            )
        else:
            logging.error(f"Model type '{model_type}' not supported.")
            sys.exit(1)

        model.fit(X_train, y_train)

        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        y_pred_proba = calibrated_model.predict_proba(X_test)
        y_pred_classes = calibrated_model.predict(X_test)

        score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')

        logging.info(f"ROC AUC Score: {score}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Precision-Recall AUC: {pr_auc}")

        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def plot_learning_curve(self, output_path: str) -> None:
        plt.figure()
        plt.plot(self.train_scores, label='Training Score')
        plt.plot(self.test_scores, label='Cross-Validation Score')
        plt.plot(self.f1_scores, label='F1 Score')
        plt.plot(self.pr_auc_scores, label='Precision-Recall AUC')
        plt.title("Learning Curve", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')
        plt.close()

    def get_class_rankings(self, model, X: np.ndarray) -> list:
        if model is None:
            raise ValueError("Model not trained. Please train the model first.")

        y_pred_proba = model.predict_proba(X)
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def plot_roc_curve_custom(self, y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
        plot_roc_curve(y_true, y_pred_proba, title, save_as, classes)

class ProteinEmbeddingGenerator:
    def __init__(self, fasta_path: str, table_data: pd.DataFrame = None, aggregation_method: str = 'none'):
        self.fasta_path = fasta_path
        self.table_data = table_data
        self.aggregation_method = aggregation_method
        self.embeddings = []
        self.min_kmers = None

    def generate_embeddings(
        self,
        k: int,
        step_size: int,
        word2vec_model_path: str,
        model_dir: str,
        min_kmers: int,
        save_min_kmers: bool = True
    ) -> None:
        logging.info("Starting protein embeddings generation...")
        # Correção: garantir que min_kmers não seja None
        if min_kmers is None:
            min_kmers = 1

        sequences = list(SeqIO.parse(self.fasta_path, "fasta"))
        kmer_sequences = []
        for record in sequences:
            sequence = str(record.seq)
            kmers = [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, step_size)]
            if len(kmers) < min_kmers:
                continue
            kmer_sequences.append(kmers)

        logging.info(f"Total sequences after filtering: {len(kmer_sequences)}")

        if not os.path.exists(word2vec_model_path):
            logging.info("Training Word2Vec model...")
            w2v_model = Word2Vec(sentences=kmer_sequences, vector_size=100, window=5, min_count=1, workers=8, epochs=2500, seed=SEED)
            w2v_model.save(word2vec_model_path)
            logging.info(f"Word2Vec model saved at {word2vec_model_path}")
        else:
            logging.info(f"Loading existing Word2Vec model from {word2vec_model_path}")
            w2v_model = Word2Vec.load(word2vec_model_path)

        for kmers in kmer_sequences:
            embeddings = [w2v_model.wv[kmer] for kmer in kmers if kmer in w2v_model.wv]
            if not embeddings:
                continue
            if self.aggregation_method == 'mean':
                sequence_embedding = np.mean(embeddings, axis=0)
            else:
                sequence_embedding = np.concatenate(embeddings)
            self.embeddings.append({
                'protein_accession': kmers[0],  # Ajuste conforme necessário
                'embedding': sequence_embedding
            })

        logging.info(f"Total embeddings generated: {len(self.embeddings)}")

        if save_min_kmers:
            min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
            with open(min_kmers_path, 'w') as f:
                f.write(str(min_kmers))
            logging.info(f"min_kmers saved at {min_kmers_path}")

        self.min_kmers = min_kmers

    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        embeddings = []
        labels = []
        for entry in self.embeddings:
            embeddings.append(entry['embedding'])
            if label_type == 'associated_variable' and self.table_data is not None:
                label = self.table_data[self.table_data['Protein_ID'] == entry['protein_accession']]['Associated_Variable'].values
                if len(label) > 0:
                    labels.append(label[0])
                else:
                    labels.append('Unknown')
            else:
                labels.append('Unknown')
        return np.array(embeddings), labels

def adjust_predictions(predicted_proba: np.ndarray, method: str = 'normalize', alpha: float = 1.0) -> np.ndarray:
    if method == 'normalize':
        logging.info("Normalizing predicted probabilities.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)

    elif method == 'smoothing':
        logging.info(f"Applying smoothing to predicted probabilities with alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])

    elif method == 'none':
        logging.info("No adjustment applied to predicted probabilities.")
        adjusted_proba = predicted_proba.copy()

    else:
        logging.warning(f"Unknown adjustment method '{method}'. No adjustment will be applied.")
        adjusted_proba = predicted_proba.copy()

    return adjusted_proba

def main(args: argparse.Namespace) -> None:
    model_dir = args.model_dir

    total_steps = 7
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

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
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path,
        table_data=train_table_data,
        aggregation_method=args.aggregation_method
    )
    min_kmers = protein_embedding_train.min_kmers
    args.min_kmers = min_kmers

    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=args.min_kmers,
        save_min_kmers=True
    )
    logging.info(f"Number of training embeddings generated: {len(protein_embedding_train.embeddings)}")

    min_kmers = protein_embedding_train.min_kmers
    args.min_kmers = min_kmers

    protein_ids_associated = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]
    var_assoc_associated = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]

    logging.info(f"Protein IDs for associated_variable extracted: {len(protein_ids_associated)}")
    logging.info(f"Associated variables for associated_variable extracted: {len(var_assoc_associated)}")

    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"Shape of X_associated: {X_associated.shape}")

    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info("Scaler for X_associated created and saved.")

    X_associated_scaled = scaler_associated.transform(X_associated)

    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable loaded from {calibrated_model_associated_full_path}")
    else:
        support_models = Support()
        trained_models = support_models.fit(
            X_associated_scaled,
            y_associated,
            protein_ids=protein_ids_associated,
            var_assoc=var_assoc_associated,
            model_name_prefix='associated',
            model_dir=model_dir,
            min_kmers=min_kmers
        )

        logging.info("Training and calibration for associated_variable completed.")

        for model_type, model_info in trained_models.items():
            logging.info(f"Plotting Learning Curve for {model_type}")
            learning_curve_path = os.path.join(model_dir, f'learning_curve_{model_type}.png')
            support_models.plot_learning_curve(learning_curve_path)

        calibrated_model_associated = trained_models['random_forest']['model']
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable saved at {calibrated_model_associated_full_path}")

        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_models.test_best_model(
            X_associated_scaled,
            y_associated,
            model_type='random_forest'
        )

        logging.info(f"Best ROC AUC for associated_variable: {best_score_associated}")
        logging.info(f"Best F1 Score for associated_variable: {best_f1_associated}")
        logging.info(f"Best Precision-Recall AUC for associated_variable: {best_pr_auc_associated}")
        logging.info(f"Best Parameters: {best_params_associated}")

        for param, value in best_params_associated.items():
            logging.info(f"{param}: {value}")

        class_rankings_associated = support_models.get_class_rankings(best_model_associated, X_test_associated)
        logging.info("Class rankings for associated_variable generated successfully.")

        class_weight = best_params_associated.get('class_weight', None)
        logging.info(f"Class weight used: {class_weight}")

        joblib.dump(best_model_associated, rf_model_associated_full_path)
        logging.info(f"Random Forest model for associated_variable saved at {rf_model_associated_full_path}")

        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
            classes_associated = None
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve(
            y_test_associated,
            y_pred_proba_associated,
            'ROC Curve for associated_variable',
            save_as=args.roc_curve_associated,
            classes=classes_associated
        )

    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers loaded: {min_kmers_loaded}")
    else:
        logging.error(f"min_kmers file not found at {min_kmers_path}. Please ensure training was completed successfully.")
        sys.exit(1)

    predict_alignment_path = args.predict_fasta

    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Prediction sequences are not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Aligned prediction file found or sequences already aligned: {predict_alignment_path}")

    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path,
        table_data=None,
        aggregation_method=args.aggregation_method
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded,
        save_min_kmers=False
    )
    logging.info(f"Number of embeddings generated for prediction: {len(protein_embedding_predict.embeddings)}")

    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')

    if os.path.exists(scaler_associated_path):
        scaler_associated = joblib.load(scaler_associated_path)
        logging.info("Scaler for associated_variable loaded successfully.")
    else:
        logging.error("Scalers not found. Please ensure training was completed successfully.")
        sys.exit(1)

    X_predict_scaled_associated = scaler_associated.transform(X_predict)

    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    logging.info("Making predictions for associated_variable with all trained models...")
    predictions = {}
    rankings = {}
    # Aqui assumimos que trained_models já foi definido acima no bloco do treinamento.
    # Caso a execução não tenha passado pelo bloco de treinamento, essa variável já existe
    # porque ou carregou um modelo calibrado anteriormente.
    for model_type, model_info in trained_models.items():
        model = model_info['model']
        pred = model.predict(X_predict_scaled_associated)
        predictions[model_type] = pred
        rankings[model_type] = Support().get_class_rankings(model, X_predict_scaled_associated)
        logging.info(f"Predictions made for {model_type}.")

    results = {}
    for i, entry in enumerate(protein_embedding_predict.embeddings):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {}
        for model_type in predictions.keys():
            results[sequence_id][f"{model_type}_prediction"] = predictions[model_type][i]
            results[sequence_id][f"{model_type}_ranking"] = rankings[model_type][i]

    with open(args.results_file, 'w') as f:
        headers = ["Protein_ID"]
        for model_type in predictions.keys():
            headers.append(f"{model_type}_Prediction")
            headers.append(f"{model_type}_Ranking")
        f.write("\t".join(headers) + "\n")

        for seq_id, result in results.items():
            row = [seq_id]
            for model_type in predictions.keys():
                row.append(str(result.get(f"{model_type}_prediction", 'Unknown')))
                row.append("; ".join(result.get(f"{model_type}_ranking", ['Unknown'])))
            f.write("\t".join(row) + "\n")
            logging.info(f"{seq_id} - Predictions: {', '.join([str(result.get(f'{model_type}_prediction', 'Unknown')) for model_type in predictions.keys()])}")

    logging.info("Generating scatter plot of predictions for new sequences...")
    if 'random_forest' in predictions:
        scatterplot_path = args.scatterplot_output
        plot_predictions_scatterplot_custom(results, scatterplot_path)
        logging.info(f"Scatter plot saved at {scatterplot_path}")

    logging.info("Generating Dual UMAP and Dual t-SNE plots for training and prediction data...")
    train_labels = y_associated
    predict_labels = predictions['random_forest'] if 'random_forest' in predictions else predictions[next(iter(predictions))]
    train_protein_ids = protein_ids_associated
    predict_protein_ids = [entry['protein_accession'] for entry in protein_embedding_predict.embeddings]

    fig_umap_train, fig_umap_predict = visualize_latent_space_with_similarity(
        X_original=X_associated_scaled,
        X_synthetic=X_predict_scaled_associated,
        y_original=train_labels,
        y_synthetic=predict_labels,
        protein_ids_original=train_protein_ids,
        protein_ids_synthetic=predict_protein_ids,
        var_assoc_original=var_assoc_associated,
        var_assoc_synthetic=[results[seq_id].get(f"random_forest_prediction", 'Unknown') for seq_id in predict_protein_ids],
        output_dir=model_dir
    )

    fig_tsne_train, fig_tsne_predict = None, None
    def plot_dual_tsne(
        train_embeddings: np.ndarray,
        train_labels: list,
        train_protein_ids: list,
        predict_embeddings: np.ndarray,
        predict_labels: list,
        predict_protein_ids: list,
        output_dir: str
    ) -> tuple:
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
            marker=dict(
                size=5,
                color=train_colors,
                opacity=0.8
            ),
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

        tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
        tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")

        pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
        pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)

        logging.info(f"t-SNE Training plot saved as {tsne_train_html}")
        logging.info(f"t-SNE Predictions plot saved as {tsne_predict_html}")

        return fig_train, fig_predict

    fig_tsne_train, fig_tsne_predict = plot_dual_tsne(
        train_embeddings=X_associated_scaled,
        train_labels=train_labels,
        train_protein_ids=train_protein_ids,
        predict_embeddings=X_predict_scaled_associated,
        predict_labels=predict_labels,
        predict_protein_ids=predict_protein_ids,
        output_dir=model_dir
    )

    logging.info("Dual UMAP and Dual t-SNE plots generated successfully.")

    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    st.success("Analysis completed successfully!")

    st.header("Scatter Plot of Predictions")
    scatterplot_path = args.scatterplot_output
    if os.path.exists(scatterplot_path):
        st.image(scatterplot_path, use_column_width=True)
    else:
        st.error(f"Scatter plot not found at {scatterplot_path}")

    formatted_results = []
    for sequence_id, info in results.items():
        for model_type in predictions.keys():
            prediction = info.get(f"{model_type}_prediction", 'Unknown')
            ranking = info.get(f"{model_type}_ranking", ['Unknown'])
            top_specificity = ranking[0] if ranking else 'Unknown'
            confidence = float(top_specificity.split(": ")[1].replace("%", "")) if ranking else 0.0
            top_two_specificities = ranking[:2] if len(ranking) >= 2 else ranking
            formatted_results.append([
                sequence_id,
                f"{model_type.capitalize()} - {top_specificity}",
                f"{confidence:.2f}%",
                "; ".join(top_two_specificities)
            ])

    headers = ["Protein_ID", "SS Prediction Specificity", "Prediction Confidence", "Top 2 Specificities"]
    df_results = pd.DataFrame(formatted_results, columns=headers)

    def highlight_table(df):
        return df.style.set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#1E3A8A'),
                    ('color', 'white'),
                    ('border', '1px solid white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('background-color', '#0B3C5D'),
                    ('color', 'white'),
                    ('border', '1px solid white'),
                    ('text-align', 'center'),
                    ('font-family', 'Arial'),
                    ('font-size', '12px'),
                ]
            },
            {
                'selector': 'tr:nth-child(even) td',
                'props': [
                    ('background-color', '#145B9C')
                ]
            },
            {
                'selector': 'tr:hover td',
                'props': [
                    ('background-color', '#0D4F8B')
                ]
            },
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

    st.markdown(
        f"""
        <div class="dataframe-container">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: #1E3A8A; 
            color: white; 
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .stDownloadButton > button:hover {
            background-color: #145B9C; 
        }
        </style>
    """, unsafe_allow_html=True)

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
        for folder_name, subfolders, filenames in os.walk(model_dir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, model_dir))
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
    logging.info(f"Results saved at {args.excel_output}")

    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved at {args.formatted_results_table}")

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
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 30px; background: linear-gradient(to bottom, #f9f9f9, #ffffff); border-radius: 15px; border: 2px solid #dddddd; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); position: relative;">
        <p style="color: black; font-size: 1.5em; font-weight: bold; margin: 0;">
            FAALPred: Predicting Fatty Acyl Chain Specificities in Fatty Acyl-AMP Ligases (FAALs) Using Integrated Approaches of Neural Networks, Bioinformatics, and Machine Learning
        </p>
        <p style="color: #2c3e50; font-size: 1.2em; font-weight: normal; margin-top: 10px;">
            Anne Liong, Leandro de Mattos Pereira, and Pedro Leão
        </p>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8;">
            <strong>FAALPred</strong> is a comprehensive bioinformatics tool designed to predict the fatty acid chain length specificity of substrates, ranging from C4 to C18.
        </p>
        <h5 style="color: #2c3e50; font-size: 20px; font-weight: bold; margin-top: 25px;">ABSTRACT</h5>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8; text-align: justify;">
            Fatty Acyl-AMP Ligases (FAALs), identified by Zhang et al. (2011), activate fatty acids of varying lengths for the biosynthesis of natural products. 
            These substrates enable the production of compounds such as nocuolin (<em>Nodularia sp.</em>, Martins et al., 2022) 
            and sulfolipid-1 (<em>Mycobacterium tuberculosis</em>, Yan et al., 2023), with applications in cancer and tuberculosis treatment 
            (Kurt et al., 2017; Gilmore et al., 2012). Dr. Pedro Leão and his team identified several of these natural products in cyanobacteria (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" style="color: #3498db; text-decoration: none;">visit here</a>), 
            and FAALPred classifies FAALs by their substrate specificity.
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{image_base64}" alt="FAAL Domain" style="width: auto; height: 120px; object-fit: contain;">
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>FAAL Domain of Synechococcus sp. PCC7002, link: <a href="https://www.rcsb.org/structure/7R7F" target="_blank" style="color: #3498db; text-decoration: none;">https://www.rcsb.org/structure/7R7F</a></em>
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

predict_fasta_file = st.sidebar.file_uploader("Upload FASTA File for Prediction", type=["fasta", "fa", "fna"])

kmer_size = st.sidebar.number_input("k-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)

aggregation_method = st.sidebar.selectbox(
    "Aggregation Method",
    options=['none', 'mean'],
    index=0
)

st.sidebar.header("Customize Word2Vec Parameters")
custom_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)
if custom_word2vec:
    window = st.sidebar.number_input("Window Size", min_value=5, max_value=20, value=10, step=5)
    workers = st.sidebar.number_input("Workers", min_value=1, max_value=112, value=8, step=8)
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=2500, value=2500, step=100)
else:
    window = 10
    workers = 8
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
            st.error("Please upload both the training FASTA file and the TSV table file.")
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
        excel_output=os.path.join(output_dir, "results.xlsx"),
        formatted_results_table=os.path.join(output_dir, "formatted_results.txt"),
        roc_curve_associated=os.path.join(output_dir, "roc_curve_associated.png"),
        learning_curve_associated=os.path.join(output_dir, "learning_curve_associated.png"),
        rf_model_associated="rf_model_associated.pkl",
        word2vec_model="word2vec_model.bin",
        scaler="scaler_associated.pkl",
        model_dir=model_dir,
        scatterplot_output=os.path.join(output_dir, "scatterplot_predictions.png"),
    )

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(args)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")


# ============================================
# End of Functions for Processing Results and Interface
# ============================================

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
        for model_type in ['random_forest', 'lightgbm', 'xgboost', 'catboost']:
            associated_rankings = info.get(f'{model_type}_ranking', [])
            if not associated_rankings:
                logging.warning(f"No associated ranking data for protein {seq_id} with {model_type}. Skipping...")
                continue

            # Use the format_and_sum_probabilities function to get the main category
            top_category_with_confidence, confidence, top_two_categories = format_and_sum_probabilities(associated_rankings)
            if top_category_with_confidence is None:
                logging.warning(f"No valid formatting for protein {seq_id} with {model_type}. Skipping...")
                continue

            # Extract the category without confidence
            category = top_category_with_confidence.split(" (")[0]
            confidence = confidence  # Sum of probabilities for the main category

            protein_specificities[f"{seq_id}_{model_type}"] = {
                'top_category': category,
                'confidence': confidence
            }

    if not protein_specificities:
        logging.warning("No data available to plot the scatter plot.")
        return

    # Sort protein IDs for better visualization
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))  # Adjust height based on the number of proteins

    # Fixed scale for X-Axis from C4 to C18
    x_values = list(range(4, 19))

    # Plot points for all proteins with their main category
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        confidence = data['confidence']

        # Extract specificities from the category string
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]

        for spec in specificities:
            ax.scatter(
                spec, y,
                color='#1f78b4',  # Uniform color
                edgecolors='black',
                linewidth=0.5,
                s=100,
                label='_nolegend_'  # Avoid duplication in the legend
            )

        # Connect points with lines if there are multiple specificities
        if len(specificities) > 1:
            ax.plot(
                specificities,
                [y] * len(specificities),
                color='#1f78b4',
                linestyle='-',
                linewidth=1.0,
                alpha=0.7
            )

    # Customize the plot for better scientific publication quality
    ax.set_xlabel('Specificity (C4 to C18)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Scatter Plot of Predictions for New Sequences (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='white')

    # Define fixed scale and formatting for the X-Axis
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='white')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='white')

    # Define grid and remove unnecessary spines for a clean look
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Minor ticks on the X-Axis for better visibility
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Adjust layout to avoid cutting off labels
    plt.tight_layout()

    # Save the figure in high quality for scientific publication
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=600, bbox_inches='tight')  # Matches background color
    plt.close()
    logging.info(f"Scatter plot saved at {output_path}")

# ============================================
# End of Functions for Processing Results and Interface
# ============================================

