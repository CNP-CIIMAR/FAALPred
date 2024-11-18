import argparse
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

import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MafftCommandline
import joblib
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV

import streamlit as st

# ============================================
# Definitions of Functions and Classes
# ============================================

# Fixing seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")  # Log to a file for persistent records
    ]
)


def are_sequences_aligned(fasta_file):
    """
    Checks if all sequences in a FASTA file have the same length.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Returns True if all sequences have the same length


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

    def _oversample_single_sample_classes(self, X, y):
        """
        Customizes oversampling to avoid oversampling extremely rare classes.
        """
        counter = Counter(y)
        classes_to_oversample = [cls for cls, count in counter.items() if count >= 2]

        # Apply RandomOverSampler only to classes with at least 2 samples
        ros = RandomOverSampler(random_state=self.seed)
        X_ros, y_ros = ros.fit_resample(X, y)

        # Apply SMOTE to classes that can be synthesized
        smote = SMOTE(random_state=self.seed)
        X_smote, y_smote = smote.fit_resample(X_ros, y_ros)

        sample_counts = Counter(y_smote)
        logging.info(f"Class distribution after oversampling: {sample_counts}")

        with open("oversampling_counts.txt", "a") as f:
            f.write("Class Distribution after Oversampling:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote

    def fit(self, X, y, model_name_prefix='model', model_dir=None):
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

            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test set class distribution: {fold_class_distribution}")

            X_train_resampled, y_train_resampled = self._oversample_single_sample_classes(X_train, y_train)

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
                    roc_auc = auc(fpr, tpr)
                    self.roc_results.append((fpr, tpr, roc_auc))
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
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
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
            realign_sequences_with_mafft(sequences_path, sequences_path.replace(".fasta", "_aligned.fasta"))
            aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")

        self.alignment = AlignIO.read(aligned_path, 'fasta')
        self.table_data = table_data
        self.embeddings = []
        self.models = {}
        self.aggregation_method = aggregation_method  # Added to choose the aggregation method
        self.min_kmers = None  # Added to store min_kmers

    def generate_embeddings(self, k=3, step_size=1, word2vec_model_path="word2vec_model.bin", model_dir=None, min_kmers=None):
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

            # Train Word2Vec model using all k-mers
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=125,  # change to 100 if necessary
                window=5,
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


def plot_predictions_scatterplot_custom(results, output_path):
    """
    Generates a scatter plot of the predictions for the new sequences.

    Y-axis: Protein accession ID
    X-axis: Specificities from 2 to 18
    Each point represents the corresponding specificity for the protein
    Lines connect the points of each protein
    Points are represented in grayscale, indicating the associated percentage.
    """
    # Prepare data
    protein_specificities = {}

    for seq_id, info in results.items():
        rankings = info['associated_ranking']
        specificity_probs = {}

        for ranking in rankings:
            try:
                category, prob = ranking.split(": ")
                prob = float(prob.replace("%", ""))
                # Extract numbers from categories, assuming they are in the format 'C4-C6-C8'
                specs = [int(s.strip('C')) for s in category.split('-') if s.startswith('C')]
                for spec in specs:
                    if spec in specificity_probs:
                        specificity_probs[spec] += prob  # Sum probabilities if already exists
                    else:
                        specificity_probs[spec] = prob
            except ValueError:
                logging.error(f"Error processing ranking: {ranking} for protein {seq_id}")

        if specificity_probs:
            # Normalize probabilities for each specificity
            total_prob = sum(specificity_probs.values())
            if total_prob > 0:
                for spec in specificity_probs:
                    specificity_probs[spec] = (specificity_probs[spec] / total_prob) * 100
            protein_specificities[seq_id] = specificity_probs

    if not protein_specificities:
        logging.warning("No data available to plot the scatterplot.")
        return

    # Sort protein IDs for better visualization
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    plt.figure(figsize=(20, max(10, len(unique_proteins) * 0.5)))  # Adjust height based on the number of proteins

    for protein, specs in protein_specificities.items():
        y = protein_order[protein]
        x = sorted(specs.keys())
        probs = [specs[spec] for spec in x]

        # Normalize probabilities to [0,1] for grayscale
        probs_normalized = [p / 100.0 for p in probs]

        # Map probabilities to grayscale colors (1 - p so that higher probability is darker)
        colors = [str(1 - p) for p in probs_normalized]

        # Plot points
        plt.scatter(x, [y] * len(x), c=colors, cmap='gray', edgecolors='w', s=100)

        # Connect points with lines
        plt.plot(x, [y] * len(x), color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.xlabel('Specificity', fontsize=12, fontweight='bold', color='white')
    plt.ylabel('Proteins', fontsize=12, fontweight='bold', color='white')
    plt.title('Scatterplot of New Sequences Predictions', fontsize=14, fontweight='bold', color='white')

    plt.yticks(ticks=range(len(unique_proteins)), labels=unique_proteins, fontsize=8, color='white')
    plt.xticks(ticks=range(2, 19), fontsize=10, color='white')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5, color='white')

    plt.gca().set_facecolor('#0B3C5D')  # Match the background color
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=300)  # Match the background color
    plt.close()


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
        model_dir=model_dir
    )
    logging.info(f"Number of training embeddings generated: {len(protein_embedding_train.embeddings)}")

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
        calibrated_model_target = support_model_target.fit(X_target, y_target, model_name_prefix='target', model_dir=model_dir)
        logging.info("Training and calibration for target_variable completed.")

        # Save the calibrated model
        joblib.dump(calibrated_model_target, calibrated_model_target_full_path)
        logging.info(f"Calibrated Random Forest model for target_variable saved at {calibrated_model_target_full_path}")

        # Test the model
        
        #best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, output_dir=args.output_dir)
        #best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, output_dir=args.model_dir)
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
        calibrated_model_associated = support_model_associated.fit(X_associated, y_associated, model_name_prefix='associated', model_dir=model_dir)
        logging.info("Training and calibration for associated_variable completed.")
        
        # Plot learning curve
        logging.info("Plotting Learning Curve for Associated Variable")
        support_model_associated.plot_learning_curve(args.learning_curve_associated)

        # Save the calibrated model
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable saved at {calibrated_model_associated_full_path}")

        # Test the model
#        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, output_dir=args.output_dir)
        #best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, output_dir=args.model_dir)

        #best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, scaler_dir=args.model_dir)

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
        model_dir=model_dir
    )
    logging.info(f"Number of embeddings for prediction generated: {len(protein_embedding_predict.embeddings)}")

    # Get embeddings for prediction
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Load the scaler
    scaler_full_path = os.path.join(model_dir, args.scaler)
    if os.path.exists(scaler_full_path):
        scaler = joblib.load(scaler_full_path)
        logging.info(f"Scaler carregado de {scaler_full_path}")
    else:
        logging.error(f"Scaler nÃ£o encontrado em {scaler_full_path}")
        sys.exit(1)
    X_predict_scaled = scaler.transform(X_predict)

    # Update progress
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Make predictions on new sequences

    # Check feature size before prediction
    if X_predict_scaled.shape[1] > calibrated_model_target.base_estimator_.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_target.base_estimator_.n_features_in_} to match the model input size.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_target.base_estimator_.n_features_in_]

    predictions_target = calibrated_model_target.predict(X_predict_scaled)

    # Check and adjust feature size for associated_variable
    if X_predict_scaled.shape[1] > calibrated_model_associated.base_estimator_.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_associated.base_estimator_.n_features_in_} to match the model input size for associated_variable.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_associated.base_estimator_.n_features_in_]

    # Make prediction for associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)

    # Get class rankings
    rankings_target = get_class_rankings_global(calibrated_model_target, X_predict_scaled)
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled)

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

    # Format results
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info['associated_ranking']
        formatted_prob_sums = format_and_sum_probabilities(associated_rankings)
        formatted_results.append([sequence_id, formatted_prob_sums])

    # Log to check the content of formatted_results
    logging.info("Formatted Results:")
    for result in formatted_results:
        logging.info(result)

    # Print results in a formatted table
    headers = ["Protein Accession", "Associated Prob. Rankings"]
    logging.info(tabulate(formatted_results, headers=headers, tablefmt="grid"))

    # Save the results to an Excel file
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Results saved in {args.excel_output}")

    # Save the table in tabulated format
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Formatted table saved in {args.formatted_results_table}")

    # Generate the Scatterplot of Predictions
    logging.info("Generating scatterplot of new sequences predictions...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatterplot saved at {args.scatterplot_output}")

    logging.info("Processing completed.")

    # Update progress to 100%
    progress_bar.progress(1.0)
    progress_text.markdown("<span style='color:white'>Progress: 100%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

# ============================================
# Streamlit Configuration and Interface
# ============================================

# Streamlit Configuration
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="ðŸ§¬",  # DNA symbol
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Title and description
st.title("FAAL-Pred: Predicting Fatty Acid Activation and Length using Integrated Approaches of Neural Networks, Bioinformatics Methods, and Machine Learning")
st.write("""
**Protein Embedding and Classification Tool** is a comprehensive bioinformatics tool designed to predict the activation and length of fatty acids using advanced machine learning techniques. The tool integrates neural networks, bioinformatics methods, and machine learning algorithms to deliver accurate predictions and insightful visualizations.
""")

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

# Output directory
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        model_dir=os.path.join(output_dir, "models")
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
        st.image(args.scatterplot_output, use_column_width=True)

        # Display formatted results table
        st.header("Formatted Results Table")
        with open(args.formatted_results_table, 'r') as f:
            formatted_table = f.read()
        st.text(formatted_table)

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

        # Credits
        st.markdown("<span style='color:white'>CIIMAR - Pedro LeÃ£o @CNP - 2024 - All rights reserved.</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logging.error(f"An error occurred: {e}")

# ============================================
# End of Code
# ============================================




