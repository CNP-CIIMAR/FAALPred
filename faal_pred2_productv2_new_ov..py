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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import DBSCAN, KMeans
from tabulate import tabulate
from PIL import Image
from matplotlib import ticker
import base64
import streamlit as st
import plotly.express as px

# Logging configurations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)

# Constants
SEED = 42

# Helper functions
def are_sequences_aligned(fasta_path):
    """
    Checks if the sequences in a FASTA file are aligned (i.e., have the same length).
    """
    try:
        alignment = AlignIO.read(fasta_path, 'fasta')
        seq_length = alignment.get_alignment_length()
        for record in alignment:
            if len(record.seq) != seq_length:
                return False
        return True
    except Exception as e:
        logging.error(f"Error checking alignment: {e}")
        st.error(f"Error checking alignment: {e}")  # Inform the user
        return False

def realign_sequences_with_mafft(input_fasta, output_fasta, threads=1):
    """
    Realigns sequences using MAFFT.
    """
    try:
        mafft_cline = MafftCommandline(input=input_fasta, auto=True, thread=threads)
        stdout, stderr = mafft_cline()
        with open(output_fasta, "w") as handle:
            handle.write(stdout)
        logging.info(f"Realigned sequences saved in {output_fasta}")
        
        # Verificar se o realinhamento foi bem-sucedido
        if are_sequences_aligned(output_fasta):
            logging.info("Realinhamento bem-sucedido.")
        else:
            logging.error("Realinhamento falhou. As sequências ainda não estão alinhadas.")
            st.error("Realinhamento falhou. As sequências ainda não estão alinhadas.")
            st.stop()  # Parar o aplicativo Streamlit graciosamente
    except Exception as e:
        logging.error(f"Error realigning sequences with MAFFT: {e}")
        st.error(f"Error realigning sequences with MAFFT: {e}")  # Inform the user
        st.stop()  # Stop the Streamlit app gracefully

def create_unique_model_directory(base_dir, aggregation_method):
    """
    Creates a unique directory to store models and results.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_dir = os.path.join(base_dir, f"{aggregation_method}_{timestamp}")
    os.makedirs(unique_dir, exist_ok=True)
    logging.info(f"Model directory created: {unique_dir}")
    return unique_dir

def plot_roc_curve_global(y_true, y_pred_proba, title, save_as=None, classes=None):
    """
    Plots the ROC curve for binary or multiclass classifications.
    """
    plt.figure()
    lw = 2

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        # Binarize the classes for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=classes)
        n_classes = y_true_bin.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=lw, label=f'ROC class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    plt.title(title, color='white')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')  # Compatible background color
    plt.close()

def get_class_rankings_global(model, X):
    """
    Retrieves class rankings based on the model's predicted probabilities.
    """
    if model is None:
        raise ValueError("Model not trained. Please train the model first.")

    # Get probabilities for each class
    y_pred_proba = model.predict_proba(X)

    # Class rankings based on probabilities
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
        logging.info(f"False Positive Rate: {fpr[i]}")
        logging.info(f"True Positive Rate: {tpr[i]}")
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

    # Initialize the sum dictionary
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

def adjust_predictions_global(predicted_proba, method='normalize', alpha=1.0):
    """
    Adjusts the model's predicted probabilities.
    """
    if method == 'normalize':
        # Normalize probabilities to sum to 1 for each sample
        logging.info("Normalizing predicted probabilities.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)

    elif method == 'smoothing':
        # Apply smoothing to probabilities to avoid extreme values
        logging.info(f"Applying smoothing to predicted probabilities with alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])

    elif method == 'none':
        # No adjustment
        logging.info("No adjustment applied to predicted probabilities.")
        adjusted_proba = predicted_proba.copy()

    else:
        logging.warning(f"Unknown adjustment method '{method}'. No adjustment will be applied.")
        adjusted_proba = predicted_proba.copy()

    return adjusted_proba

# Support Class with Dynamic Oversampling and Overfitting Monitoring
class Support:
    """
    Support class to train and evaluate Random Forest models with dynamic oversampling techniques.
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
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "criterion": "entropy",
            "max_features": "sqrt",
            "class_weight": None,
            "max_leaf_nodes": 20,
            "min_impurity_decrease": 0.02,
            "bootstrap": True,
            "ccp_alpha": 0.001,
            "random_state": self.seed
        }

        self.parameters = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["entropy", "gini"],
            "max_features": ["sqrt", "log2", None],
            "class_weight": ["balanced", None],
            "max_leaf_nodes": [5, 10, 20, 30, None],
            "min_impurity_decrease": [0.0],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0, 0.001, 0.01],
        }

    def _oversample_with_smote_on_duplicates(self, X, y, target_samples_per_class=None):
        """
        Applies RandomOverSampler to balance classes and then SMOTE to generate synthetic samples.

        Parameters:
        - X: Features (numpy array or pandas DataFrame).
        - y: Labels (numpy array or pandas Series).
        - target_samples_per_class (dict, optional): Dictionary mapping classes to the desired number of samples.

        Returns:
        - X_final: Features after oversampling and SMOTE.
        - y_final: Labels after oversampling and SMOTE.
        """
        # Step 1: Original Data
        original_X, original_y = X.copy(), y.copy()
        original_counts = Counter(original_y)
        logging.info(f"Original class distribution: {original_counts}")

        # Step 2: Define sampling strategy
        if target_samples_per_class:
            # Ensure we are not reducing the number of samples
            sampling_strategy = {cls: min(target_samples_per_class.get(cls, original_counts[cls]), original_counts[cls])
                                 for cls in original_counts}
            # Update to the desired target
            for cls in target_samples_per_class:
                sampling_strategy[cls] = target_samples_per_class[cls]
        else:
            # Complete balancing of classes
            ros = RandomOverSampler(random_state=self.seed)
            X_resampled, y_resampled = ros.fit_resample(original_X, original_y)
            sampling_strategy = None  # Use default strategy

        # Step 3: Apply RandomOverSampler if sampling_strategy is specified
        if sampling_strategy:
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=self.seed)
            X_ros, y_ros = ros.fit_resample(original_X, original_y)
            resampled_counts = Counter(y_ros)
            logging.info(f"Distribution after RandomOverSampler: {resampled_counts}")
        else:
            X_ros, y_ros = X_resampled, y_resampled
            resampled_counts = Counter(y_ros)

        # Step 4: Identify duplicates to apply SMOTE
        duplicates_X = []
        duplicates_y = []

        for cls in resampled_counts:
            n_resampled = resampled_counts[cls]
            n_original = original_counts.get(cls, 0)
            n_duplicates = n_resampled - n_original

            if n_duplicates > 0:
                cls_indices = np.where(original_y == cls)[0]
                duplicated_indices = np.random.choice(cls_indices, size=n_duplicates, replace=True)
                duplicates_X.append(original_X[duplicated_indices])
                duplicates_y.append(original_y[duplicated_indices])

        if duplicates_X:
            duplicates_X = np.vstack(duplicates_X)
            duplicates_y = np.hstack(duplicates_y)
            logging.info(f"Number of duplicated samples: {len(duplicates_X)}")
        else:
            duplicates_X = np.array([])
            duplicates_y = np.array([])
            logging.info("No samples were duplicated.")

        # Step 5: Apply SMOTE to duplicates
        if len(duplicates_X) > 0:
            smote = SMOTE(random_state=self.seed)
            try:
                X_synthetic, y_synthetic = smote.fit_resample(duplicates_X, duplicates_y)
                logging.info(f"Number of synthetic samples generated by SMOTE: {len(X_synthetic)}")
                X_final = np.vstack([original_X, X_synthetic])
                y_final = np.hstack([original_y, y_synthetic])
            except ValueError as e:
                logging.error(f"Error applying SMOTE to duplicates: {e}")
                X_final, y_final = original_X, original_y
        else:
            X_final, y_final = original_X, original_y

        # Step 6: Log final class distribution
        final_counts = Counter(y_final)
        logging.info(f"Final class distribution after RandomOverSampler and SMOTE: {final_counts}")

        # Save class distribution to a log file
        with open("oversampling_counts.txt", "a") as f:
            f.write("Class distribution after RandomOverSampler and SMOTE:\n")
            for cls, count in final_counts.items():
                f.write(f"{cls}: {count}\n")

        return X_final, y_final

    def _oversample_single_sample_classes(self, X, y, target_samples_per_class=None):
        """
        Applies oversampling using RandomOverSampler and SMOTE based on the defined strategy.

        Parameters:
        - X: Features (numpy array or pandas DataFrame).
        - y: Labels (numpy array or pandas Series).
        - target_samples_per_class (dict, optional): Dictionary mapping classes to the desired number of samples.

        Returns:
        - X_final: Features after oversampling and SMOTE.
        - y_final: Labels after oversampling and SMOTE.
        """
        return self._oversample_with_smote_on_duplicates(X, y, target_samples_per_class=target_samples_per_class)

    def _train_and_evaluate(self, X, y, model_name_prefix, model_dir):
        """
        Trains the model with cross-validation and returns the updated Support object.
        """
        self.model = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs)
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        train_scores = []
        test_scores = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply oversampling on the training set
            X_train_resampled, y_train_resampled = self._oversample_single_sample_classes(X_train, y_train)

            # Train the model
            self.model.fit(X_train_resampled, y_train_resampled)

            # Evaluate the model
            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

        # Calculate the average difference between training and validation metrics
        self.avg_train_score = np.mean(train_scores)
        self.avg_test_score = np.mean(test_scores)
        self.overfitting_score_diff = self.avg_train_score - self.avg_test_score

        logging.info(f"Average training score: {self.avg_train_score}")
        logging.info(f"Average validation score: {self.avg_test_score}")
        logging.info(f"Difference (Overfitting Indicator): {self.overfitting_score_diff}")

        return self

    def _check_overfitting(self, support_model, threshold=0.1):
        """
        Checks for overfitting based on the difference between training and validation metrics.

        Parameters:
        - support_model: Support object after training and evaluation.
        - threshold (float): Threshold for the metric difference.

        Returns:
        - Boolean: True if overfitting is detected, False otherwise.
        """
        if support_model.overfitting_score_diff > threshold:
            logging.warning(f"Overfitting detected with a difference of {support_model.overfitting_score_diff}")
            return True
        else:
            logging.info(f"No overfitting detected. Difference of {support_model.overfitting_score_diff}")
            return False

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
        logging.info(f"Best grid search parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def get_best_param(self, param_name, default=None):
        return self.best_params.get(param_name, default)

    def plot_learning_curve(self, output_path):
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
        plt.savefig(output_path, facecolor='#0B3C5D')  # Compatible background color
        plt.close()

    def fit(self, X, y, model_name_prefix='model', model_dir=None, min_kmers=None, max_samples_per_class=100):
        logging.info(f"Starting fit method for {model_name_prefix}...")

        X = np.array(X)
        y = np.array(y)

        # Initialize oversampling
        initial_oversample = True
        target_samples = None

        while True:
            if initial_oversample:
                # Initial oversampling to balance classes
                X_smote, y_smote = self._oversample_single_sample_classes(X, y, target_samples_per_class=None)
                initial_oversample = False
                logging.info("Initial oversampling applied.")
            else:
                if target_samples is None:
                    # Define target_samples_per_class as 100 or the maximum available
                    current_counts = Counter(y_smote)
                    target_samples = {cls: min(max_samples_per_class, count + 10) for cls, count in current_counts.items()}
                    logging.info(f"Increasing oversampling to: {target_samples}")
                else:
                    # Increment until the maximum allowed
                    target_samples = {cls: min(max_samples_per_class, count + 10) for cls, count in target_samples.items()}
                    logging.info(f"Incrementing oversampling to: {target_samples}")

                X_smote, y_smote = self._oversample_single_sample_classes(X, y, target_samples_per_class=target_samples)

            sample_counts = Counter(y_smote)
            logging.info(f"Distribution after oversampling: {sample_counts}")

            if any(count > max_samples_per_class for count in sample_counts.values()):
                logging.info(f"Some class reached the maximum number of {max_samples_per_class} samples.")
                break

            # Train and evaluate the model with cross-validation
            support_model = self._train_and_evaluate(X_smote, y_smote, model_name_prefix, model_dir)

            # Calculate the difference between training and validation metrics to detect overfitting
            overfitting_detected = self._check_overfitting(support_model)

            if overfitting_detected:
                logging.info("Overfitting detected. Stopping the increase of oversampling.")
                break
            else:
                logging.info("No overfitting detected. Proceeding to increase oversampling.")

            # Check if the maximum number of samples per class has been reached
            if all(count >= max_samples_per_class for count in sample_counts.values()):
                logging.info(f"All classes reached the maximum number of {max_samples_per_class} samples.")
                break

        # After the loop, the final model is trained
        return self.model

    def plot_roc_curve(self, y_true, y_pred_proba, title, save_as=None, classes=None):
        """
        Plots the ROC curve using the global function.
        """
        plot_roc_curve_global(y_true, y_pred_proba, title, save_as, classes)

    def get_class_rankings(self, X):
        """
        Retrieves class rankings for the provided data.
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Get probabilities for each class
        y_pred_proba = self.model.predict_proba(X)

        # Class rankings based on probabilities
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def test_best_RF(self, X, y, scaler_dir='.'):
        """
        Tests the best Random Forest model with the provided data.
        """
        # Load the scaler
        scaler_path = os.path.join(scaler_dir, 'scaler.pkl') if scaler_dir else 'scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found at {scaler_path}")
            st.error("Scaler not found.")
            st.stop()  # Gracefully stop the Streamlit app

        X_scaled = scaler.transform(X)

        # Apply oversampling to the entire dataset before splitting
        X_resampled, y_resampled = self._oversample_single_sample_classes(X_scaled, y)
        sample_counts = Counter(y_resampled)
        logging.info(f"Counts after oversampling for testing: {sample_counts}")

        # Save counts to a log file
        with open("sample_counts_after_oversampling.txt", "a") as f:
            f.write("Class distribution after RandomOverSampler and SMOTE:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        # Train the model with the best parameters
        model = RandomForestClassifier(
            n_estimators=self.best_params.get('n_estimators', 100),
            max_depth=self.best_params.get('max_depth', 5),
            min_samples_split=self.best_params.get('min_samples_split', 4),
            min_samples_leaf=self.best_params.get('min_samples_leaf', 2),
            criterion=self.best_params.get('criterion', 'entropy'),
            max_features=self.best_params.get('max_features', 'sqrt'),
            class_weight=self.best_params.get('class_weight', 'balanced'),
            max_leaf_nodes=self.best_params.get('max_leaf_nodes', 20),
            min_impurity_decrease=self.best_params.get('min_impurity_decrease', 0.01),
            bootstrap=self.best_params.get('bootstrap', True),
            ccp_alpha=self.best_params.get('ccp_alpha', 0.001),
            random_state=self.seed,
            n_jobs=self.n_jobs
        )
        model.fit(X_train, y_train)  # Train the model on the training data

        # Integrate Probability Calibration
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        # Make predictions
        y_pred = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred, method='normalize')

        # Calculate the score (e.g., ROC AUC)
        score = self._calculate_score(y_pred_adjusted, y_test)

        # Calculate additional metrics
        y_pred_classes = calibrated_model.predict(X_test)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        if len(np.unique(y_test)) > 1:
            # Binarize the labels for multiclass average_precision_score
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            pr_auc = average_precision_score(y_test_bin, y_pred_adjusted, average='macro')
        else:
            pr_auc = 0.0  # Cannot calculate PR AUC for a single class

        # Return the score, best parameters, trained model, and test sets
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred, y_test):
        """
        Calculates the score (e.g., ROC AUC) based on predictions and true labels.
        """
        n_classes = len(np.unique(y_test))
        if y_pred.ndim == 1 or n_classes == 2:
            return roc_auc_score(y_test, y_pred)
        elif y_pred.ndim == 2 and n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            return roc_auc_score(y_test_bin, y_pred, multi_class='ovo', average='macro')
        else:
            logging.warning(f"Unexpected format or number of classes: y_pred shape: {y_pred.shape}, number of classes: {n_classes}")
            return 0

    def plot_learning_curve(self, output_path):
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
        plt.savefig(output_path, facecolor='#0B3C5D')  # Compatible background color
        plt.close()

# Protein Embedding Generation Class
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
        self.aggregation_method = aggregation_method  # Aggregation method
        self.min_kmers = None  # Stores min_kmers

    def generate_embeddings(self, k=3, step_size=1, word2vec_model_path="word2vec_model.bin", model_dir=None, min_kmers=None, save_min_kmers=False):
        """
        Generates embeddings for protein sequences using Word2Vec, standardizing the number of k-mers.
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
            # Initialize Variables
            kmer_groups = {}
            all_kmers = []
            kmers_counts = []

            # Generate k-mers
            for record in self.alignment:
                sequence = str(record.seq)
                seq_len = len(sequence)
                protein_accession_alignment = record.id.split()[0]

                # If table_data is not provided, skip matching
                if self.table_data is not None:
                    matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                    matching_info = self.table_data[matching_rows]

                    if matching_info.empty:
                        logging.warning(f"No matching data in the training table for {protein_accession_alignment}")
                        continue  # Skip to the next iteration

                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]

                else:
                    # If no table, use default values or None
                    target_variable = None
                    associated_variable = None

                logging.info(f"Processing {protein_accession_alignment} with sequence length {seq_len}")

                if seq_len < k:
                    logging.warning(f"Sequence too short for {protein_accession_alignment}. Length: {seq_len}")
                    continue

                # Generate k-mers, allowing k-mers with fewer than k gaps
                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Allow k-mers with fewer than k gaps

                if not kmers:
                    logging.warning(f"No valid k-mer for {protein_accession_alignment}")
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
                st.error("No k-mers were collected. Please check your sequences and k-mer parameters.")
                st.stop()  # Gracefully stop the Streamlit app

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
                size=125,  # Mantendo 'size' conforme solicitado
                window=window if 'window' in globals() else 10,  # Use the window from Streamlit inputs if available
                min_count=1,
                workers=workers if 'workers' in globals() else 8,
                sg=1,
                hs=1,  # Hierarchical softmax enabled
                negative=0,  # Negative sampling disabled
                iter=iter if 'iter' in globals() else 2500,  # Mantendo 'iter' conforme solicitado
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

            # Get embeddings for the selected k-mers
            selected_embeddings = [self.models['global'].wv[kmer] if kmer in self.models['global'].wv else np.zeros(self.models['global'].vector_size) for kmer in selected_kmers]

            if self.aggregation_method == 'none':
                # Concatenate embeddings of the selected k-mers
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                # Aggregate embeddings of the selected k-mers by mean
                embedding_concatenated = np.mean(selected_embeddings, axis=0)
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

        # Fit the StandardScaler with the embeddings for training/prediction
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])

        # Check if all embeddings have the same format
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Inconsistent embedding formats detected: {embedding_shapes}")
            st.error("Embeddings have inconsistent formats.")
            st.stop()  # Gracefully stop the Streamlit app
        else:
            logging.info(f"All embeddings have format: {embedding_shapes.pop()}")

        # Define the full path for the scaler
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
            labels.append(embedding_info[label_type])  # Use the specified label type

        return np.array(embeddings), np.array(labels)

# Plotting Functions
def plot_dual_tsne_3d(train_embeddings, train_labels, train_protein_ids, 
                      predict_embeddings, predict_labels, predict_protein_ids, output_dir):
    """
    Plots two separate 3D t-SNE graphs:
    - Plot 1: Training Data.
    - Plot 2: Predictions.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.manifold import TSNE

    # Dynamically calculate perplexity
    def compute_perplexity(n_samples):
        return max(5, min(50, n_samples // 100))

    # Adjust perplexity for training data
    n_samples_train = train_embeddings.shape[0]
    dynamic_perplexity_train = compute_perplexity(n_samples_train)

    # Initialize t-SNE with adjusted perplexity for training data
    tsne_train = TSNE(n_components=3, random_state=42, perplexity=dynamic_perplexity_train, n_iter=1000)
    tsne_train_result = tsne_train.fit_transform(train_embeddings)

    # Adjust perplexity for predictions
    n_samples_predict = predict_embeddings.shape[0]
    dynamic_perplexity_predict = compute_perplexity(n_samples_predict)

    # Initialize t-SNE with adjusted perplexity for predictions
    tsne_predict = TSNE(n_components=3, random_state=42, perplexity=dynamic_perplexity_predict, n_iter=1000)
    tsne_predict_result = tsne_predict.fit_transform(predict_embeddings)

    # Create color map for training data
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Create color map for predictions
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Convert labels to colors
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Plot 1: Training Data
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

    # Plot 2: Predictions
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
    # Save the plots as HTML
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")
    
    import plotly.io as pio
    pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
    pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)
    
    logging.info(f"t-SNE Training Data saved at {tsne_train_html}")
    logging.info(f"t-SNE Predictions saved at {tsne_predict_html}")

    return fig_train, fig_predict

def plot_dual_umap(train_embeddings, train_labels, train_protein_ids,
                   predict_embeddings, predict_labels, predict_protein_ids, output_dir):
    """
    Plots two separate 3D UMAP graphs:
    - Plot 1: Training Data.
    - Plot 2: Predictions.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import umap.umap_ as umap

    # Dimensionality reduction for training data
    umap_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_train_result = umap_train.fit_transform(train_embeddings)

    # Dimensionality reduction for predictions
    umap_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_predict_result = umap_predict.fit_transform(predict_embeddings)

    # Create color map for training data
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Create color map for predictions
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Convert labels to colors
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Plot 1: Training Data
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

    # Plot 2: Predictions
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

    # Save the plots as HTML
    umap_train_html = os.path.join(output_dir, "umap_train_3d.html")
    umap_predict_html = os.path.join(output_dir, "umap_predict_3d.html")
    
    import plotly.io as pio
    pio.write_html(fig_train, file=umap_train_html, auto_open=False)
    pio.write_html(fig_predict, file=umap_predict_html, auto_open=False)
    
    logging.info(f"UMAP Training Data saved at {umap_train_html}")
    logging.info(f"UMAP Predictions saved at {umap_predict_html}")

    return fig_train, fig_predict

def plot_predictions_scatterplot_custom(results, output_path, top_n=3):
    """
    Generates a scatter plot of the top N predictions for new sequences.

    Y-Axis: Protein accession ID
    X-Axis: Specificities from C2 to C18 (fixed scale)
    Each point represents the probability of the corresponding specificity for the protein.
    Only the top N predictions are plotted.
    Points are colored uniformly, styled for scientific publication.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Prepare the data
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
                    # Extract only the first number before any colon or other separator
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

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, len(unique_proteins) * 0.5))  # Adjust height based on the number of proteins

    # Fixed scale for the X-axis from C2 to C18
    x_values = list(range(2, 19))

    for protein, specs in protein_specificities.items():
        y = protein_order[protein]
        
        # Prepare data for plotting (ensure only the top N predictions are plotted)
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

        # Connect the points with lines
        if len(x) > 1:
            ax.plot(x, [y] * len(x), color='#1f78b4', linestyle='-', linewidth=1.0, alpha=0.7)

    # Customize the plot for better publication quality
    ax.set_xlabel('Specificity (C2 to C18)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold')
    ax.set_title('Scatterplot of Predictions for New Sequences (Top Rankings)', fontsize=16, fontweight='bold', pad=20)

    # Set fixed scale for the X-axis and formatting
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12)
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10)

    # Set grid and remove unnecessary spines for a clean look
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Minor ticks on the X-axis for better visibility
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Adjust layout to avoid label cutoff
    plt.tight_layout()

    # Save the figure in high quality for publication
    plt.savefig(output_path, facecolor='white', dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Scatterplot saved at {output_path}")

# Main Function
def main(args):
    model_dir = args.model_dir  # Should be 'results/models'

    """
    Main function that coordinates the workflow.
    """
    model_dir = args.model_dir

    # Initialize progress variables
    total_steps = 8  # Updated after removing dimensionality reduction and visualization steps
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
        logging.info(f"Aligned training file found or sequences are already aligned: {train_alignment_path}")

    # Load training data table
    try:
        train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
        logging.info("Training data table loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading training data table: {e}")
        st.error(f"Error loading training data table: {e}")
        st.stop()  # Gracefully stop the Streamlit app

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
        aggregation_method=args.aggregation_method  # Pass aggregation method
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
    logging.info(f"Shape of X_target: {X_target.shape}")

    # Full paths for target_variable models
    calibrated_model_target_full_path = os.path.join(model_dir, 'calibrated_model_target.pkl')

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Check if the calibrated model for target_variable already exists
    if os.path.exists(calibrated_model_target_full_path):
        calibrated_model_target = joblib.load(calibrated_model_target_full_path)
        logging.info(f"Calibrated model for target_variable loaded from {calibrated_model_target_full_path}")
    else:
        # Train the model for target_variable with dynamic oversampling
        support_model_target = Support()
        calibrated_model_target = support_model_target.fit(
            X_target, y_target, 
            model_name_prefix='target', 
            model_dir=model_dir, 
            min_kmers=min_kmers, 
            max_samples_per_class=100  # Limit to 100 samples per class
        )
        logging.info("Training and calibration for target_variable completed.")

        # Save the calibrated model
        joblib.dump(calibrated_model_target, calibrated_model_target_full_path)
        logging.info(f"Calibrated model for target_variable saved at {calibrated_model_target_full_path}")

        # Test the model
        best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(
            X_target, y_target, scaler_dir=args.model_dir
        )

        logging.info(f"Best ROC AUC for target_variable: {best_score}")
        logging.info(f"Best F1 Score for target_variable: {best_f1}")
        logging.info(f"Best Precision-Recall AUC for target_variable: {best_pr_auc}")
        logging.info(f"Best Parameters: {best_params}")

        # Get class rankings
        class_rankings = support_model_target.get_class_rankings(X_test_target)

        # Log rankings for the first 5 samples
        logging.info("Top 3 class rankings for the first 5 samples:")
        for i in range(min(5, len(class_rankings))):
            logging.info(f"Sample {i+1}: Class Rankings - {class_rankings[i][:3]}")  # Shows top 3 rankings

        # Plot the ROC curve for target_variable
        n_classes_target = len(np.unique(y_test_target))
        if n_classes_target == 2:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)[:, 1]
            unique_classes_target = None
        else:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)
            unique_classes_target = np.unique(y_test_target).astype(str)
        plot_roc_curve_global(y_test_target, y_pred_proba_target, 'ROC Curve for Target Variable', save_as=args.roc_curve_target, classes=unique_classes_target)

        # Convert y_test_target to integer labels
        unique_labels = sorted(set(y_test_target))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        y_test_target_int = [label_to_int[label.strip()] for label in y_test_target]

        # Calculate and save ROC values for target_variable
        roc_df_target = calculate_roc_values(best_model_target, X_test_target, y_test_target_int)
        logging.info("ROC AUC values for target_variable:")
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
    logging.info(f"Shape of X_associated: {X_associated.shape}")

    # Full paths for associated_variable models
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
        logging.info(f"Calibrated model for associated_variable loaded from {calibrated_model_associated_full_path}")
    else:
        # Train the model for associated_variable with dynamic oversampling
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(
            X_associated, y_associated, 
            model_name_prefix='associated', 
            model_dir=model_dir, 
            min_kmers=min_kmers, 
            max_samples_per_class=100  # Limit to 100 samples per class
        )
        logging.info("Training and calibration for associated_variable completed.")

        # Plot the learning curve for associated_variable
        logging.info("Plotting Learning Curve for associated_variable")
        support_model_associated.plot_learning_curve(args.learning_curve_associated)

        # Save the calibrated model
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated model for associated_variable saved at {calibrated_model_associated_full_path}")

        # Test the model
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(
            X_associated, y_associated, scaler_dir=args.model_dir
        )

        logging.info(f"Best ROC AUC for associated_variable: {best_score_associated}")
        logging.info(f"Best F1 Score for associated_variable: {best_f1_associated}")
        logging.info(f"Best Precision-Recall AUC for associated_variable: {best_pr_auc_associated}")
        logging.info(f"Best Parameters for associated_variable: {best_params_associated}")
        logging.info(f"Best model for associated_variable: {best_model_associated}")

        # Get class rankings
        class_rankings_associated = support_model_associated.get_class_rankings(X_test_associated)
        logging.info("Top 3 class rankings for the first 5 samples in associated data:")
        for i in range(min(5, len(class_rankings_associated))):
            logging.info(f"Sample {i+1}: Class Rankings - {class_rankings_associated[i][:3]}")  # Shows top 3 rankings

        # Plot the ROC curve for associated_variable
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
            unique_classes_associated = None
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
    # STEP 2: Classification of New Sequences
    # =============================

    # Load min_kmers
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers loaded: {min_kmers_loaded}")
    else:
        logging.error(f"min_kmers file not found at {min_kmers_path}.")
        st.error("min_kmers file not found. Ensure that training was completed successfully.")
        st.stop()  # Gracefully stop the Streamlit app

    # Load data for prediction
    predict_alignment_path = args.predict_fasta

    # Check if prediction sequences are aligned
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Prediction sequences are not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Fix threads=1
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Aligned prediction file found or sequences are already aligned: {predict_alignment_path}")

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Initialize ProteinEmbedding for prediction, without the table
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, 
        table_data=None,
        aggregation_method=args.aggregation_method  # Pass aggregation method
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded  # Use the same min_kmers from training
    )
    logging.info(f"Number of prediction embeddings generated: {len(protein_embedding_predict.embeddings)}")

    # Get embeddings for prediction
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Load scaler
    scaler_full_path = os.path.join(model_dir, args.scaler)
    if os.path.exists(scaler_full_path):
        scaler = joblib.load(scaler_full_path)
        logging.info(f"Scaler loaded from {scaler_full_path}")
    else:
        logging.error(f"Scaler not found at {scaler_full_path}.")
        st.error("Scaler not found. Ensure that training was completed successfully.")
        st.stop()  # Gracefully stop the Streamlit app
    X_predict_scaled = scaler.transform(X_predict)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Make predictions on new sequences

    # Check feature size against the original estimator of CalibratedClassifierCV
    if X_predict_scaled.shape[1] > calibrated_model_target.estimator.n_features_in_:
        logging.info(f"Reducing the number of features from {X_predict_scaled.shape[1]} to {calibrated_model_target.estimator.n_features_in_} to match the model.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_target.estimator.n_features_in_]

    predictions_target = calibrated_model_target.predict(X_predict_scaled)

    # Check feature size for associated_variable
    if X_predict_scaled.shape[1] > calibrated_model_associated.estimator.n_features_in_:
        logging.info(f"Reducing the number of features from {X_predict_scaled.shape[1]} to {calibrated_model_associated.estimator.n_features_in_} to match the associated model.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_associated.estimator.n_features_in_]

    # Make predictions for associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)

    # Get class rankings
    support_model_target = Support()  # Initialize Support object for target_variable
    rankings_target = support_model_target.get_class_rankings(X_predict_scaled)
    support_model_associated = Support()  # Initialize Support object for associated_variable
    rankings_associated = support_model_associated.get_class_rankings(X_predict_scaled)

    # Process and save the results
    results = {}
    for entry, pred_target, pred_associated, ranking_target, ranking_associated in zip(protein_embedding_predict.embeddings, predictions_target, predictions_associated, rankings_target, rankings_associated):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "target_prediction": pred_target,
            "associated_prediction": pred_associated,
            "target_ranking": ranking_target,
            "associated_ranking": ranking_associated
        }

    # Save the results to a file
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tTarget_Prediction\tAssociated_Prediction\tTarget_Ranking\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['target_prediction']}\t{result['associated_prediction']}\t{'; '.join(result['target_ranking'])}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Target Variable: {result['target_prediction']}, Associated Variable: {result['associated_prediction']}, Target Ranking: {'; '.join(result['target_ranking'])}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

    # Format the results for display in Streamlit
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info['associated_ranking']
        if not associated_rankings:
            logging.warning(f"No associated ranking for protein {sequence_id}. Skipping...")
            continue

        # Get the top 3 rankings
        top3_rankings = associated_rankings[:3]
        predicted_ss = '-'.join([rank.split(": ")[0] for rank in top3_rankings])
        # Sum the probabilities
        prob = sum([float(rank.split(": ")[1].replace("%", "")) for rank in top3_rankings])
        complete_ranking = " - ".join(associated_rankings)
        formatted_results.append([sequence_id, predicted_ss, f"{prob:.2f}", complete_ranking])

    # Log to verify the content of formatted_results
    logging.info("Formatted Results:")
    for result in formatted_results:
        logging.info(result)

    # Create DataFrame for Streamlit
    df_results = pd.DataFrame(formatted_results, columns=["Query Name", "Predicted SS", "SS Prediction Probability (%)", "Complete Ranking"])

    # Display the results in Streamlit
    st.header("Prediction Results")
    st.table(df_results)

    # Save the results to an Excel file
    df_results.to_excel(args.excel_output, index=False)
    logging.info(f"Results saved at {args.excel_output}")

    # Save the table in tabular format
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=["Query Name", "Predicted SS", "SS Prediction Probability (%)", "Complete Ranking"], tablefmt="grid"))
    logging.info(f"Formatted table saved at {args.formatted_results_table}")

    # Generate the Scatterplot of Predictions
    logging.info("Generating scatterplot of predictions for new sequences...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatterplot saved at {args.scatterplot_output}")

    logging.info("Processing completed.")

    # ============================================
    # STEP 3: Dimensionality Reduction and Plotting (Removed as per request)
    # ============================================

    # Update progress to 100%
    progress_bar.progress(1.0)
    progress_text.markdown("<span style='color:white'>Progress: 100%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

# Function to load and resize images with DPI adjustment
def load_and_resize_image_with_dpi(image_path, base_width, dpi=300):
    try:
        # Load the image
        image = Image.open(image_path)
        # Calculate the new height proportionally
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Resize the image
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

# Function to convert image to base64
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

# Function to plot dimensionality reduction graphs (already implemented above)
def plot_dual_tsne_3d(train_embeddings, train_labels, train_protein_ids, 
                      predict_embeddings, predict_labels, predict_protein_ids, output_dir):
    """
    Plots two separate 3D t-SNE graphs:
    - Plot 1: Training Data.
    - Plot 2: Predictions.
    """
    # [Implementation as presented above]
    pass

def plot_dual_umap(train_embeddings, train_labels, train_protein_ids,
                   predict_embeddings, predict_labels, predict_protein_ids, output_dir):
    """
    Plots two separate 3D UMAP graphs:
    - Plot 1: Training Data.
    - Plot 2: Predictions.
    """
    # [Implementation as presented above]
    pass

def plot_predictions_scatterplot_custom(results, output_path, top_n=3):
    """
    Generates a scatter plot of the top N predictions for new sequences.

    Y-Axis: Protein accession ID
    X-Axis: Specificities from C2 to C18 (fixed scale)
    Each point represents the probability of the corresponding specificity for the protein.
    Only the top N predictions are plotted.
    Points are colored uniformly, styled for scientific publication.
    """
    # [Implementation as presented above]
    pass

# Streamlit Interface
# Custom CSS for dark navy background and white text
st.markdown(
    """
    <style>
    /* Set the main app background and text color */
    .stApp {
        background-color: #0B3C5D;
        color: white;
    }
    /* Set the sidebar background and text color */
    [data-testid="stSidebar"] {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Ensure all elements within the sidebar have blue background and white text */
    [data-testid="stSidebar"] * {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Customize input elements within the sidebar */
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
    /* Customize the drag-and-drop area of the file uploader */
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] div {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Customize dropdown selection options */
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
    /* Customize checkboxes and radio buttons */
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
# Function to convert image to base64
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

# Image path
image_path = "./images/faal.png"
image_base64 = get_base64_image(image_path)
# Using HTML with st.markdown to align title and text

st.markdown(
    f"""
    <div style="text-align: center; font-family: 'Arial', sans-serif; padding: 30px; background: linear-gradient(to bottom, #f9f9f9, #ffffff); border-radius: 15px; border: 2px solid #dddddd; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); position: relative;">
        <p style="color: black; font-size: 1.5em; font-weight: bold; margin: 0;">
            FAALPred: Predicting Fatty Acid Ligase FAAL (FAALs) Fatty Acid Specificities Using Integrated Neural Networks, Bioinformatics, and Machine Learning Approaches
        </p>
        <p style="color: #2c3e50; font-size: 1.2em; font-weight: normal; margin-top: 10px;">
            Anne Liong, Leandro de Mattos Pereira, and Pedro Leão
        </p>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8;">
            <strong>FAALPred</strong> is a comprehensive bioinformatics tool designed to predict the substrate chain length specificity of Fatty Acid-AMP Ligase FAALs (FAALs), ranging from C4 to C18.
        </p>
        <h5 style="color: #2c3e50; font-size: 20px; font-weight: bold; margin-top: 25px;">ABSTRACT</h5>
        <p style="color: #2c3e50; font-size: 18px; line-height: 1.8; text-align: justify;">
            Fatty Acid-AMP Ligase (FAAL) enzymes, identified by Zhang et al. (2011), activate fatty acids of different lengths for the biosynthesis of natural products. These substrates allow the production of compounds such as nocuolin (<em>Nodularia sp.</em>, Martins et al., 2022) and sulfolipid-1 (<em>Mycobacterium tuberculosis</em>, Yan et al., 2023), with applications in cancer and tuberculosis treatment (Kurt et al., 2017; Gilmore et al., 2012). Dr. Pedro Leão and his team identified several of these natural products in cyanobacteria (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" style="color: #3498db; text-decoration: none;">visit here</a>), and FAALPred classifies FAALs based on their substrate specificity.
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{image_base64}" alt="FAAL Domain" style="width: auto; height: 120px; object-fit: contain;">
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>FAAL Domain from Synechococcus sp. PCC7002, link: <a href="https://www.rcsb.org/structure/7R7F" target="_blank" style="color: #3498db; text-decoration: none;">https://www.rcsb.org/structure/7R7F</a></em>
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
use_default_train = st.sidebar.checkbox("Use Default Training Data", value=True)
if not use_default_train:
    train_fasta_file = st.sidebar.file_uploader("Upload Training FASTA File", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("Upload Training Table (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Upload Prediction FASTA File", type=["fasta", "fa", "fna"])

kmer_size = st.sidebar.number_input("K-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)
aggregation_method = st.sidebar.selectbox(
    "Aggregation Method",
    options=['none', 'mean'],  # Removed 'median', 'sum', 'max'
    index=0
)

# Optional input for Word2Vec parameters
st.sidebar.header("Optional Word2Vec Parameters")
custom_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)
if custom_word2vec:
    window = st.sidebar.number_input(
        "Window Size", min_value=5, max_value=20, value=10, step=5
    )
    workers = st.sidebar.number_input(
        "Workers", min_value=1, max_value=112, value=8, step=1
    )
    iter = st.sidebar.number_input(
        "Number of Iterations", min_value=1, max_value=3500, value=2500, step=100
    )
else:
    window = 10  # Default value
    workers = 8  # Default value
    iter = 2500  # Default value

# Button to start processing
if st.sidebar.button("Run Analysis"):
    # Internal data paths
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"
    
    model_dir = create_unique_model_directory("results/models", aggregation_method)
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
            st.warning("Please upload both the training FASTA file and the TSV table.")
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
        roc_curve_target=os.path.join(output_dir, "roc_curve_target.png"),
        roc_curve_associated=os.path.join(output_dir, "roc_curve_associated.png"),
        learning_curve_target=os.path.join(output_dir, "learning_curve_target.png"),
        learning_curve_associated=os.path.join(output_dir, "learning_curve_associated.png"),
        roc_values_target=os.path.join(output_dir, "roc_values_target.csv"),
        rf_model_target="rf_model_target.pkl",
        rf_model_associated="rf_model_associated.pkl",
        word2vec_model="word2vec_model.bin",
        scaler="scaler.pkl",
        model_dir=model_dir,
    )

    # Create the model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Execute the main function
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(args)

        st.success("Analysis completed successfully!")

        # Display the scatterplot
        st.header("Scatterplot of Predictions")
        scatterplot_path = os.path.join(args.output_dir, "scatterplot_predictions.png")
        if os.path.exists(scatterplot_path):
            st.image(scatterplot_path,  use_column_width=True)
        else:
            st.error(f"Scatterplot not found at {scatterplot_path}.")

        # Display the formatted results table
        formatted_table_path = args.formatted_results_table

        # Check if the file exists and is not empty
        if os.path.exists(formatted_table_path) and os.path.getsize(formatted_table_path) > 0:
            try:
                # Open and read the file content
                with open(formatted_table_path, 'r') as f:
                    formatted_table = f.read()

                # Display the content in Streamlit
                st.text(formatted_table)
            except Exception as e:
                st.error(f"An error occurred while reading the formatted results table: {e}")
        else:
            st.error(f"Formatted results table not found or is empty: {formatted_table_path}")

        # Prepare the results.zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for folder_name, subfolders, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    zip_file.write(file_path, arcname=os.path.relpath(file_path, output_dir))
        zip_buffer.seek(0)

        # Provide the download link
        st.header("Download Results")
        st.download_button(
            label="Download All Results as results.zip",
            data=zip_buffer,
            file_name="results.zip",
            mime="application/zip"
        )

        # ============================================
        # STEP 3: Dimensionality Reduction and Plotting (Removed as per request)
        # ============================================

        # Update progress to 100%
        progress_bar.progress(1.0)
        progress_text.markdown("<span style='color:white'>Progress: 100%</span>", unsafe_allow_html=True)
        time.sleep(0.1)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")

# Function to load and resize images with DPI adjustment (repeated, can be removed if unnecessary)
def load_and_resize_image_with_dpi(image_path, base_width, dpi=300):
    try:
        # Load the image
        image = Image.open(image_path)
        # Calculate the new height proportionally
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Resize the image
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")
        return None

# Image paths
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
encoded_images = [get_base64_image(path) for path in image_paths if os.path.exists(path)]

# CSS for layout
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
    FAALPred will make predictions for all submitted sequences, even if they are not FAALs. To limit your input to only Fatty Acid Ligase FAALs, first consider extracting from each FAAL the sequences corresponding to the FAAL domain cd05931 (cd05931) using InterProScan.
</div>
<div class="footer-text" style="font-size: 14px; margin-top: 5px;">
    CIIMAR - Pedro Leão @CNP - 2024 - All rights reserved.
</div>
"""

# Generate <img> tags for each image
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Render the footer
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)
