#!/usr/bin/env python
import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from Bio import AlignIO, SeqIO
from Bio.Align.Applications import MafftCommandline
from gensim.models import Word2Vec  # Usado se embedding_method for word2vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tabulate import tabulate

# Para uso do PyTorch na nova implementação de seq2vec
import torch
import torch.nn as nn
import torch.optim as optim

# Define uma semente para reprodutibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================
# Classe: Seq2VecPytorch
# Implementação simples de um modelo de embedding usando PyTorch.
# ============================================
class Seq2VecPytorch:
    def __init__(self, sentences, vector_size, window, min_count, epochs, seed):
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.vocab = {}
        self.word2index = {}
        self.index2word = {}
        self.build_vocab()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.vector_size)
        self.optimizer = optim.SGD(self.embedding.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def build_vocab(self):
        counter = Counter()
        for sentence in self.sentences:
            counter.update(sentence)
        # Filtra palavras com frequência menor que min_count
        self.vocab = {word: count for word, count in counter.items() if count >= self.min_count}
        self.word2index = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.index2word = {idx: word for word, idx in self.word2index.items()}

    def train(self):
        # Loop de treinamento dummy – não implementa um skip-gram real.
        for epoch in range(self.epochs):
            total_loss = 0.0
            for sentence in self.sentences:
                for word in sentence:
                    if word in self.word2index:
                        idx = self.word2index[word]
                        input_tensor = torch.tensor([idx])
                        output = self.embedding(input_tensor)
                        # Usamos a própria saída como target (dummy)
                        target = output.detach()
                        loss = self.loss_fn(output, target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()
            if epoch % 100 == 0:
                logging.info(f"Epoch {epoch}: Loss {total_loss}")

    def get_vector(self, word):
        if word in self.word2index:
            idx = self.word2index[word]
            with torch.no_grad():
                return self.embedding(torch.tensor([idx])).squeeze(0).numpy()
        else:
            return np.zeros(self.vector_size)

    def has_vector(self, word):
        return word in self.word2index

    def save(self, path):
        checkpoint = {
            'state_dict': self.embedding.state_dict(),
            'vocab': self.vocab,
            'word2index': self.word2index,
            'index2word': self.index2word,
            'vector_size': self.vector_size
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(sentences=[], vector_size=checkpoint['vector_size'], window=0, min_count=1, epochs=0, seed=42)
        model.vocab = checkpoint['vocab']
        model.word2index = checkpoint['word2index']
        model.index2word = checkpoint['index2word']
        num_embeddings = len(model.vocab)
        model.embedding = nn.Embedding(num_embeddings, model.vector_size)
        model.embedding.load_state_dict(checkpoint['state_dict'])
        return model

# ============================================
# Funções Auxiliares
# ============================================
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

def plot_roc_curve_global(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
    lw = 2
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
    else:
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
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC for class {class_label} (area = {roc_auc[i]:0.2f})')
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

def lighten_color(hex_color, amount=0.5):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02x}{g:02x}{b:02x}'

# ============================================
# Classe: ProteinEmbeddingGenerator
# ============================================
class ProteinEmbeddingGenerator:
    """
    Gera embeddings de proteína usando o método escolhido: "esm2", "seq2vec" ou "word2vec".
    """
    def __init__(self, sequences_path: str, table_data: pd.DataFrame = None, aggregation_method: str = 'none', embedding_method: str = 'word2vec'):
        # Verifica se as sequências estão alinhadas; se não, realinha com MAFFT.
        aligned_path = sequences_path
        if not are_sequences_aligned(sequences_path):
            realign_sequences_with_mafft(sequences_path, sequences_path.replace(".fasta", "_aligned.fasta"), threads=1)
            aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")
        else:
            logging.info(f"Sequences already aligned: {sequences_path}")
        self.alignment = AlignIO.read(aligned_path, 'fasta')
        self.table_data = table_data
        self.embeddings = []
        self.models = {}
        self.aggregation_method = aggregation_method  # 'none' ou 'mean'
        self.min_kmers = None
        self.embedding_method = embedding_method.lower()  # "esm2", "seq2vec" ou "word2vec"

    def generate_embeddings(self, k: int = 3, step_size: int = 1,
                            word2vec_model_path: str = "word2vec_model.bin", 
                            model_dir: str = None, min_kmers: int = None, 
                            save_min_kmers: bool = False, window: int = 5,
                            workers: int = 48, epochs: int = 2500) -> None:
        if self.embedding_method == "esm2":
            import torch
            import esm
            logging.info("Using ESM2 for embeddings.")
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50S()
            model.eval()
            batch_converter = alphabet.get_batch_converter()
            self.embeddings = []
            for record in self.alignment:
                seq_id = record.id.split()[0]
                sequence = str(record.seq)
                data = [(seq_id, sequence)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
                token_representations = results["representations"][model.num_layers]
                embedding = token_representations.mean(1).squeeze(0).cpu().numpy()
                self.embeddings.append({
                    'protein_accession': seq_id,
                    'embedding': embedding,
                    'target_variable': None,
                    'associated_variable': None
                })
            logging.info(f"Total ESM2 embeddings generated: {len(self.embeddings)}")
            return

        # Para os demais métodos, gera k-mers.
        all_kmers = []
        kmers_counts = []
        kmer_groups = {}
        for record in self.alignment:
            sequence_id = record.id.split()[0]
            sequence = str(record.seq)
            if self.table_data is not None:
                matching_rows = self.table_data['Protein.accession'].str.split().str[0] == sequence_id
                matching_info = self.table_data[matching_rows]
                if matching_info.empty:
                    logging.warning(f"No table data for {sequence_id}")
                    target_variable = None
                    associated_variable = None
                else:
                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]
            else:
                target_variable = None
                associated_variable = None
            if len(sequence) < k:
                logging.warning(f"Sequence too short for {sequence_id}")
                continue
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
            logging.error("No k-mers were collected.")
            sys.exit(1)
        if min_kmers is not None:
            self.min_kmers = min_kmers
            logging.info(f"Using provided min_kmers: {self.min_kmers}")
        else:
            self.min_kmers = min(kmers_counts)
            logging.info(f"Minimum k-mers in any sequence: {self.min_kmers}")
        if save_min_kmers and model_dir:
            min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
            with open(min_kmers_path, 'w') as f:
                f.write(str(self.min_kmers))
            logging.info(f"min_kmers saved to {min_kmers_path}")

        # Treina o modelo de embedding usando seq2vec (PyTorch) ou word2vec.
        model_key = "global"
        model_full_path = os.path.join(model_dir, word2vec_model_path) if model_dir else word2vec_model_path
        if self.embedding_method == "seq2vec":
            # Usa a implementação em PyTorch para seq2vec
            if os.path.exists(model_full_path):
                logging.info(f"Seq2VecPytorch model found at {model_full_path}. Loading.")
                model = Seq2VecPytorch.load(model_full_path)
            else:
                logging.info("Seq2VecPytorch model not found. Training new model.")
                model = Seq2VecPytorch(
                    sentences=all_kmers,
                    vector_size=390,
                    window=window,
                    min_count=1,
                    epochs=epochs,
                    seed=SEED
                )
                model.train()
                if model_dir:
                    os.makedirs(os.path.dirname(model_full_path), exist_ok=True)
                model.save(model_full_path)
            self.models[model_key] = model
        else:
            # Default: word2vec
            if os.path.exists(model_full_path):
                logging.info(f"Word2Vec model found at {model_full_path}. Loading.")
                model = Word2Vec.load(model_full_path)
            else:
                logging.info("Word2Vec model not found. Training new model.")
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
                    os.makedirs(os.path.dirname(model_full_path), exist_ok=True)
                model.save(model_full_path)
            self.models[model_key] = model

        # Gera os embeddings padronizados.
        self.embeddings = []
        for record in self.alignment:
            sequence_id = record.id.split()[0]
            embedding_info = kmer_groups.get(sequence_id, {})
            kmers_for_protein = embedding_info.get('kmers', [])
            if len(kmers_for_protein) == 0:
                if self.aggregation_method == 'none':
                    embedding_concatenated = np.zeros(self.models[model_key].vector_size * self.min_kmers)
                else:
                    embedding_concatenated = np.zeros(self.models[model_key].vector_size)
                self.embeddings.append({
                    'protein_accession': sequence_id,
                    'embedding': embedding_concatenated,
                    'target_variable': embedding_info.get('target_variable'),
                    'associated_variable': embedding_info.get('associated_variable')
                })
                continue
            selected_kmers = kmers_for_protein[:self.min_kmers]
            if len(selected_kmers) < self.min_kmers:
                padding = [np.zeros(self.models[model_key].vector_size)] * (self.min_kmers - len(selected_kmers))
                selected_kmers.extend(padding)
            if self.embedding_method == "seq2vec":
                selected_embeddings = [
                    self.models[model_key].get_vector(kmer) if self.models[model_key].has_vector(kmer)
                    else np.zeros(self.models[model_key].vector_size)
                    for kmer in selected_kmers
                ]
            else:
                selected_embeddings = [
                    self.models[model_key].wv[kmer] if kmer in self.models[model_key].wv
                    else np.zeros(self.models[model_key].vector_size)
                    for kmer in selected_kmers
                ]
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
            logging.error(f"Inconsistent embedding shapes: {embedding_shapes}")
            raise ValueError("Embeddings have inconsistent shapes.")
        else:
            logging.info(f"All embeddings have shape: {embedding_shapes.pop()}")
        scaler_full_path = os.path.join(model_dir, 'scaler_associated.pkl') if model_dir else 'scaler_associated.pkl'
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler found at {scaler_full_path}. Loading.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler not found. Training new scaler.")
            scaler = StandardScaler().fit(embeddings_array_train)
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler saved to {scaler_full_path}")

    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        embeddings = []
        labels = []
        for entry in self.embeddings:
            embeddings.append(entry['embedding'])
            labels.append(entry[label_type])
        return np.array(embeddings), np.array(labels)

# ============================================
# Classe: Support (Treinamento de RF)
# ============================================
class Support:
    """
    Classe de suporte para treinamento e avaliação de modelos RandomForest com oversampling.
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
        self.best_params = {}
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
            "n_estimators": [250, 300],
            "max_depth": [10, 20],
            "min_samples_split": [4, 6],
            "min_samples_leaf": [4, 6],
            "criterion": ["gini", "entropy"],
            "max_features": ["log2"],
            "class_weight": ["balanced", None],
            "max_leaf_nodes": [10, 20, None],
            "min_impurity_decrease": [0.0],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0],
        }

    def _oversample_single_sample_classes(self, X: np.ndarray, y: np.ndarray, protein_ids: list = None, var_assoc: list = None) -> tuple:
        logging.info("Starting oversampling...")
        counter = Counter(y)
        logging.info(f"Initial class distribution: {counter}")
        classes_to_oversample = {cls: max(self.cv + 1, count) for cls, count in counter.items()}
        logging.info(f"Oversampling strategy: {classes_to_oversample}")
        try:
            ros = RandomOverSampler(sampling_strategy=classes_to_oversample, random_state=self.seed)
            X_ros, y_ros = ros.fit_resample(X, y)
            logging.info(f"Distribution after RandomOverSampler: {Counter(y_ros)}")
        except ValueError as e:
            logging.error(f"Error during RandomOverSampler: {e}")
            sys.exit(1)
        synthetic_protein_ids = []
        synthetic_var_assoc = []
        if protein_ids and var_assoc:
            for idx in range(len(X), len(X_ros)):
                synthetic_protein_ids.append(f"synthetic_ros_{idx}")
                synthetic_var_assoc.append(var_assoc[idx % len(var_assoc)])
        try:
            smote = SMOTE(random_state=self.seed)
            X_smote, y_smote = smote.fit_resample(X_ros, y_ros)
            logging.info(f"Distribution after SMOTE: {Counter(y_smote)}")
        except ValueError as e:
            logging.error(f"Error during SMOTE: {e}")
            sys.exit(1)
        if protein_ids and var_assoc:
            for idx in range(len(X_ros), len(X_smote)):
                synthetic_protein_ids.append(f"synthetic_smote_{idx}")
                synthetic_var_assoc.append(var_assoc[idx % len(var_assoc)])
        with open("oversampling_counts.txt", "a") as f:
            f.write("Class Distribution after Oversampling:\n")
            for cls, count in Counter(y_smote).items():
                f.write(f"{cls}: {count}\n")
        return X_smote, y_smote, synthetic_protein_ids, synthetic_var_assoc

    def fit(self, X: np.ndarray, y: np.ndarray, protein_ids: list = None, var_assoc: list = None, model_name_prefix: str = 'model', model_dir: str = None, min_kmers: int = None) -> RandomForestClassifier:
        logging.info(f"Starting fit for {model_name_prefix}...")
        X = np.array(X)
        y = np.array(y)
        if min_kmers is not None:
            logging.info(f"Using provided min_kmers: {min_kmers}")
        else:
            min_kmers = len(X)
            logging.info(f"min_kmers not provided. Using size of X: {min_kmers}")
        logging.info("Oversampling...")
        X_smote, y_smote, synthetic_protein_ids, synthetic_var_assoc = self._oversample_single_sample_classes(X, y, protein_ids, var_assoc)
        if protein_ids and var_assoc:
            combined_protein_ids = protein_ids + synthetic_protein_ids
            combined_var_assoc = var_assoc + synthetic_var_assoc
        else:
            combined_protein_ids = None
            combined_var_assoc = None

        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []
        class_counts = Counter(y_smote)
        min_class_count = min(class_counts.values())
        adjusted_n_splits = min(self.cv, min_class_count - 1)
        if adjusted_n_splits < self.cv:
            logging.warning(f"Adjusting n_splits from {self.cv} to {adjusted_n_splits} due to class constraints.")
            skf = StratifiedKFold(n_splits=adjusted_n_splits, random_state=self.seed, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)
        for fold_number, (train_index, test_index) in enumerate(skf.split(X_smote, y_smote), start=1):
            X_train, X_test = X_smote[train_index], X_smote[test_index]
            y_train, y_test = y_smote[train_index], y_smote[test_index]
            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Test distribution: {fold_class_distribution}")
            X_train_resampled, y_train_resampled, _, _ = self._oversample_single_sample_classes(X_train, y_train, protein_ids, var_assoc)
            train_sample_counts = Counter(y_train_resampled)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Training distribution after oversampling: {train_sample_counts}")
            with open("training_sample_counts_after_oversampling.txt", "a") as f:
                f.write(f"Fold {fold_number} for {model_name_prefix}:\n")
                for cls, count in train_sample_counts.items():
                    f.write(f"{cls}: {count}\n")
            self.model = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs)
            self.model.fit(X_train_resampled, y_train_resampled)
            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)
            y_pred = self.model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)
            if len(np.unique(y_test)) > 1:
                pr_auc = average_precision_score(y_test, self.model.predict_proba(X_test), average='macro')
            else:
                pr_auc = 0.0
            self.pr_auc_scores.append(pr_auc)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Training: {train_score}, Test: {test_score}, F1: {f1}, PR AUC: {pr_auc}")
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            self.best_params = best_params
            if model_dir:
                best_model_filename = os.path.join(model_dir, f'model_best_{model_name_prefix}.pkl')
                os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved at {best_model_filename}")
            else:
                best_model_filename = f'model_best_{model_name_prefix}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Best model saved as {best_model_filename}")
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=5, n_jobs=self.n_jobs)
            calibrator.fit(X_train_resampled, y_train_resampled)
            self.model = calibrator
            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'calibrated_model_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Calibrated model saved at {calibrated_model_filename}")
        return self.model

    def _perform_grid_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray) -> tuple:
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
        logging.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def get_best_param(self, param_name: str, default=None):
        return self.best_params.get(param_name, default)

    def plot_learning_curve(self, output_path: str) -> None:
        plt.figure()
        plt.plot(self.train_scores, label='Training')
        plt.plot(self.test_scores, label='CV')
        plt.plot(self.f1_scores, label='F1')
        plt.plot(self.pr_auc_scores, label='PR AUC')
        plt.title("Learning Curve", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')
        plt.close()

    def get_class_rankings(self, X: np.ndarray) -> list:
        if self.model is None:
            raise ValueError("Model not trained.")
        y_pred_proba = self.model.predict_proba(X)
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked]
            class_rankings.append(formatted)
        return class_rankings

    def test_best_RF(self, X: np.ndarray, y: np.ndarray, scaler_dir: str = '.') -> tuple:
        scaler_path = os.path.join(model_dir, 'scaler_associated.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Scaler not found at {scaler_path}.")
            sys.exit(1)
        X_scaled = scaler.transform(X)
        X_resampled, y_resampled, synthetic_protein_ids, synthetic_var_assoc = self._oversample_single_sample_classes(X_scaled, y)
        total_samples = len(y_resampled)
        logging.info(f"Total samples after oversampling: {total_samples}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )
        logging.info(f"Training: {len(y_train)} samples, Test: {len(y_test)} samples")
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
        model.fit(X_train, y_train)
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator
        y_pred = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = y_pred / y_pred.sum(axis=1, keepdims=True)
        score = self._calculate_score(y_pred_adjusted, y_test)
        y_pred_classes = calibrated_model.predict(X_test)
        f1_global = f1_score(y_test, y_pred_classes, average='weighted')
        unique_classes = np.unique(y_test)
        f1_per_class = f1_score(y_test, y_pred_classes, average=None, labels=unique_classes)
        f1_class_dict = dict(zip(unique_classes, f1_per_class))
        logging.info(f"Global F1: {f1_global}")
        logging.info(f"F1 per class: {f1_class_dict}")
        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(y_test, y_pred_adjusted, average='macro')
        else:
            pr_auc = 0.0
        return score, f1_global, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred: np.ndarray, y_test: np.ndarray) -> float:
        n_classes = len(np.unique(y_test))
        if y_pred.ndim == 1 or n_classes == 2:
            return roc_auc_score(y_test, y_pred)
        elif y_pred.ndim == 2 and n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            return roc_auc_score(y_test_bin, y_pred, multi_class='ovo', average='macro')
        else:
            logging.warning("Unexpected y_pred format.")
            return 0

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
        plot_roc_curve_global(y_true, y_pred_proba, title, save_as, classes)

# ============================================
# Função Principal e Execução via CLI
# ============================================
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # ETAPA 1: Treinamento
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Realigning training sequences with MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Training file aligned: {train_alignment_path}")
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Training table data loaded.")
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, table_data=train_table_data, 
        aggregation_method=args.aggregation_method,
        embedding_method=args.embedding_method
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size, step_size=args.step_size,
        word2vec_model_path=args.word2vec_model, model_dir=model_dir,
        save_min_kmers=True, window=args.window, workers=args.workers, epochs=args.epochs
    )
    logging.info(f"Total training embeddings: {len(protein_embedding_train.embeddings)}")
    min_kmers = protein_embedding_train.min_kmers
    protein_ids_associated = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]
    var_assoc_associated = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"X_associated shape: {X_associated.shape}")
    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info("Scaler for X_associated created and saved.")
    X_associated_scaled = scaler_associated.transform(X_associated)
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')
    
    # Instancia support_model_associated
    support_model_associated = Support()
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Calibrated model loaded from {calibrated_model_associated_full_path}")
        support_model_associated.model = calibrated_model_associated
    else:
        calibrated_model_associated = support_model_associated.fit(
            X_associated_scaled, y_associated, protein_ids=protein_ids_associated,
            var_assoc=var_assoc_associated, model_name_prefix='associated', 
            model_dir=model_dir, min_kmers=min_kmers
        )
        logging.info("Training and calibration for associated completed.")
        learning_curve_associated_path = args.learning_curve_associated
        support_model_associated.plot_learning_curve(learning_curve_associated_path)
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated model saved at {calibrated_model_associated_full_path}")
    # ETAPA 2: Previsão
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers loaded: {min_kmers_loaded}")
    else:
        logging.error("min_kmers file not found.")
        sys.exit(1)
    predict_alignment_path = args.predict_fasta
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Realigning prediction sequences with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Prediction file aligned: {predict_alignment_path}")
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, table_data=None, 
        aggregation_method=args.aggregation_method,
        embedding_method=args.embedding_method
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size, step_size=args.step_size,
        word2vec_model_path=args.word2vec_model, model_dir=model_dir,
        min_kmers=min_kmers_loaded, window=args.window, workers=args.workers, epochs=args.epochs
    )
    logging.info(f"Total prediction embeddings: {len(protein_embedding_predict.embeddings)}")
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])
    if os.path.exists(scaler_associated_path):
        scaler_associated = joblib.load(scaler_associated_path)
        logging.info("Scaler for associated loaded.")
    else:
        logging.error("Scaler not found.")
        sys.exit(1)
    X_predict_scaled_associated = scaler_associated.transform(X_predict)
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled_associated)
    predict_labels = [str(label) for label in predictions_associated]
    logging.info(f"Predictions (associated): {predict_labels}")
    rankings_associated = support_model_associated.get_class_rankings(X_predict_scaled_associated)
    results = {}
    for entry, pred_associated, ranking_associated in zip(protein_embedding_predict.embeddings, predictions_associated, rankings_associated):
        seq_id = entry['protein_accession']
        results[seq_id] = {
            "associated_prediction": pred_associated,
            "associated_ranking": ranking_associated
        }
    results_file = args.results_file
    with open(results_file, 'w') as f:
        f.write("Protein_ID\tAssociated_Prediction\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['associated_prediction']}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Prediction: {result['associated_prediction']}")
    st.success("Analysis completed successfully!")
    # Aqui você pode incluir código Streamlit para exibir ou baixar os resultados.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAAL_Pred: Predicting FAAL specificity")
    parser.add_argument("--train_fasta", type=str, required=True, help="Path to training FASTA file")
    parser.add_argument("--train_table", type=str, required=True, help="Path to training table (TSV)")
    parser.add_argument("--predict_fasta", type=str, required=True, help="Path to prediction FASTA file")
    parser.add_argument("--kmer_size", type=int, default=3, help="k-mer size")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--aggregation_method", type=str, default="mean", choices=["none", "mean"], help="Embedding aggregation method")
    parser.add_argument("--embedding_method", type=str, default="word2vec", choices=["esm2", "seq2vec", "word2vec"], help="Embedding method to use")
    parser.add_argument("--word2vec_model", type=str, default="word2vec_model.bin", help="Filename for embedding model")
    parser.add_argument("--rf_model_associated", type=str, default="rf_model_associated.pkl", help="Filename for RF model for associated")
    parser.add_argument("--learning_curve_associated", type=str, default="learning_curve_associated.png", help="Path to save learning curve plot")
    parser.add_argument("--results_file", type=str, default="predictions.tsv", help="Results file")
    parser.add_argument("--model_dir", type=str, default="results", help="Directory to save models and results")
    parser.add_argument("--window", type=int, default=5, help="Window parameter for embedding")
    parser.add_argument("--workers", type=int, default=48, help="Number of workers for embedding")
    parser.add_argument("--epochs", type=int, default=2500, help="Number of epochs for embedding")
    args = parser.parse_args()
    main(args)
