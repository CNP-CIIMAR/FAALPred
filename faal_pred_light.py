#!/usr/bin/env python
import argparse
import base64
import io
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
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize, MultiLabelBinarizer
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap  # or simply: import umap

# ============================================
# Configure Logging and Seed
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================
# PyTorch Implementation for Seq2Vec (Simplified)
# ============================================
class Seq2VecPytorch:
    """
    Uma implementação simples de um modelo de embedding para sequências usando PyTorch.
    (Nesta versão, não há mecanismo de atenção sofisticado; é uma implementação dummy.)
    """
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
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.vector_size)
        self.optimizer = optim.SGD(self.embedding.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def build_vocab(self):
        counter = Counter()
        for sentence in self.sentences:
            counter.update(sentence)
        self.vocab = {word: count for word, count in counter.items() if count >= self.min_count}
        self.word2index = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.index2word = {idx: word for word, idx in self.word2index.items()}

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0.0
            for sentence in self.sentences:
                for word in sentence:
                    if word in self.word2index:
                        idx = self.word2index[word]
                        input_tensor = torch.tensor([idx])
                        output = self.embedding(input_tensor)
                        target = output.detach()
                        loss = self.loss_fn(output, target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()
            if epoch % 100 == 0:
                logging.info(f"Seq2VecPytorch - Epoch {epoch}: Loss {total_loss:.4f}")

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
        model = cls(sentences=[], vector_size=checkpoint['vector_size'], window=0, min_count=1, epochs=0, seed=SEED)
        model.vocab = checkpoint['vocab']
        model.word2index = checkpoint['word2index']
        model.index2word = checkpoint['index2word']
        num_embeddings = len(model.vocab)
        model.embedding = nn.Embedding(num_embeddings, model.vector_size)
        model.embedding.load_state_dict(checkpoint['state_dict'])
        return model

# ============================================
# General Auxiliary Functions
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
        logging.info(f"Sequences realigned and saved to {output_path}")
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
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC Curve (area = {roc_auc:.2f})')
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
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC for class {class_label} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", color='white')
    plt.ylabel("True Positive Rate", color='white')
    plt.title(title, color='white')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')
    plt.close()

# ============================================
# UMAP and Scatter Plot Functions
# ============================================
def plot_umap_3d_combined(X_original: np.ndarray, X_synthetic: np.ndarray, 
                            var_assoc_original: list, var_assoc_synthetic: list,
                            output_path: str = None) -> go.Figure:
    """
    Gera um gráfico 3D interativo com UMAP combinando amostras originais e sintéticas.
    """
    X_combined = np.vstack([X_original, X_synthetic])
    n_orig = X_original.shape[0]
    n_syn = X_synthetic.shape[0]
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.05)
    X_umap = umap_reducer.fit_transform(X_combined)
    df = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1],
        'UMAP3': X_umap[:, 2],
        'Associated Variable': np.concatenate([var_assoc_original, var_assoc_synthetic]),
        'Group': ['Original'] * n_orig + ['Synthetic'] * n_syn
    })
    unique_vars = sorted(set(var_assoc_original).union(set(var_assoc_synthetic)))
    base_colors = {}
    palette = px.colors.qualitative.Plotly
    for i, var in enumerate(unique_vars):
        base_colors[var] = palette[i % len(palette)]
    def assign_color(row):
        base = base_colors.get(row['Associated Variable'], '#808080')
        return base if row['Group'] == 'Original' else lighten_color(base, 0.5)
    df['Color'] = df.apply(assign_color, axis=1)
    fig = go.Figure()
    for var in unique_vars:
        df_orig = df[(df['Group'] == 'Original') & (df['Associated Variable'] == var)]
        if not df_orig.empty:
            fig.add_trace(go.Scatter3d(
                x=df_orig['UMAP1'],
                y=df_orig['UMAP2'],
                z=df_orig['UMAP3'],
                mode='markers',
                marker=dict(size=6, color=df_orig['Color'], opacity=0.8),
                name=f'Original {var} ({len(df_orig)})',
                hoverinfo='text',
                text=df_orig.apply(lambda row: f"Group: {row['Group']}<br>Associated Variable: {row['Associated Variable']}", axis=1)
            ))
        df_syn = df[(df['Group'] == 'Synthetic') & (df['Associated Variable'] == var)]
        if not df_syn.empty:
            fig.add_trace(go.Scatter3d(
                x=df_syn['UMAP1'],
                y=df_syn['UMAP2'],
                z=df_syn['UMAP3'],
                mode='markers',
                marker=dict(size=6, color=df_syn['Color'], opacity=0.8, symbol='diamond'),
                name=f'Synthetic {var} ({len(df_syn)})',
                hoverinfo='text',
                text=df_syn.apply(lambda row: f"Group: {row['Group']}<br>Associated Variable: {row['Associated Variable']}", axis=1)
            ))
    fig.update_layout(
        title=f"3D UMAP (Original: {n_orig}, Synthetic: {n_syn})",
        scene=dict(xaxis=dict(title="UMAP1"), yaxis=dict(title="UMAP2"), zaxis=dict(title="UMAP3")),
        template="plotly_dark"
    )
    if output_path:
        fig.write_html(output_path, auto_open=False)
        logging.info(f"3D UMAP plot saved at {output_path}")
    return fig

def plot_umap_2d_combined(X_original: np.ndarray, X_synthetic: np.ndarray, 
                            var_assoc_original: list, var_assoc_synthetic: list,
                            output_path: str = None) -> go.Figure:
    """
    Gera um gráfico 2D interativo com UMAP combinando amostras originais e sintéticas.
    """
    X_combined = np.vstack([X_original, X_synthetic])
    n_orig = X_original.shape[0]
    n_syn = X_synthetic.shape[0]
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.05)
    X_umap = umap_reducer.fit_transform(X_combined)
    df = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1],
        'Associated Variable': np.concatenate([var_assoc_original, var_assoc_synthetic]),
        'Group': ['Original'] * n_orig + ['Synthetic'] * n_syn
    })
    unique_vars = sorted(set(var_assoc_original).union(set(var_assoc_synthetic)))
    base_colors = {}
    palette = px.colors.qualitative.Plotly
    for i, var in enumerate(unique_vars):
        base_colors[var] = palette[i % len(palette)]
    def assign_color(row):
        base = base_colors.get(row['Associated Variable'], '#808080')
        return base if row['Group'] == 'Original' else lighten_color(base, 0.5)
    df['Color'] = df.apply(assign_color, axis=1)
    fig = go.Figure()
    for var in unique_vars:
        df_orig = df[(df['Group'] == 'Original') & (df['Associated Variable'] == var)]
        if not df_orig.empty:
            fig.add_trace(go.Scatter(
                x=df_orig['UMAP1'],
                y=df_orig['UMAP2'],
                mode='markers',
                marker=dict(size=6, color=df_orig['Color'], opacity=0.8),
                name=f'Original {var} ({len(df_orig)})',
                hoverinfo='text',
                text=df_orig.apply(lambda row: f"Group: {row['Group']}<br>Associated Variable: {row['Associated Variable']}", axis=1)
            ))
        df_syn = df[(df['Group'] == 'Synthetic') & (df['Associated Variable'] == var)]
        if not df_syn.empty:
            fig.add_trace(go.Scatter(
                x=df_syn['UMAP1'],
                y=df_syn['UMAP2'],
                mode='markers',
                marker=dict(size=6, color=df_syn['Color'], opacity=0.8, symbol='diamond'),
                name=f'Synthetic {var} ({len(df_syn)})',
                hoverinfo='text',
                text=df_syn.apply(lambda row: f"Group: {row['Group']}<br>Associated Variable: {row['Associated Variable']}", axis=1)
            ))
    fig.update_layout(
        title=f"2D UMAP (Original: {n_orig}, Synthetic: {n_syn})",
        xaxis=dict(title="UMAP1"),
        yaxis=dict(title="UMAP2"),
        template="plotly_dark"
    )
    if output_path:
        fig.write_html(output_path, auto_open=False)
        logging.info(f"2D UMAP plot saved at {output_path}")
    return fig

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
) -> go.Figure:
    """
    Visualiza o espaço latente em 3D usando UMAP e calcula a similaridade (cosine) entre amostras originais e sintéticas.
    """
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.hstack([y_original, y_synthetic])
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.05)
    X_transformed = umap_reducer.fit_transform(X_combined)
    n_orig = len(X_original)
    df_original = pd.DataFrame({
        'x': X_transformed[:n_orig, 0],
        'y': X_transformed[:n_orig, 1],
        'z': X_transformed[:n_orig, 2],
        'Protein ID': protein_ids_original,
        'Associated Variable': var_assoc_original,
        'Type': 'Original'
    })
    df_synthetic = pd.DataFrame({
        'x': X_transformed[n_orig:, 0],
        'y': X_transformed[n_orig:, 1],
        'z': X_transformed[n_orig:, 2],
        'Protein ID': protein_ids_synthetic,
        'Associated Variable': var_assoc_synthetic,
        'Type': 'Synthetic'
    })
    from sklearn.metrics.pairwise import cosine_similarity
    similarities_list = []
    closest_protein_list = []
    closest_var_list = []
    for i in range(len(df_synthetic)):
        current_class = df_synthetic.loc[i, 'Associated Variable']
        mask = df_original['Associated Variable'] == current_class
        if mask.sum() > 0:
            X_orig_subset = X_original[mask.values]
            cos_sim = cosine_similarity(X_synthetic[i].reshape(1, -1), X_orig_subset)
            max_sim = cos_sim.max()
            best_idx_local = cos_sim.argmax()
            best_global_idx = df_original.index[mask][best_idx_local]
            best_protein = df_original.loc[best_global_idx, 'Protein ID']
            best_var = df_original.loc[best_global_idx, 'Associated Variable']
        else:
            max_sim = np.nan
            best_protein = None
            best_var = None
        similarities_list.append(max_sim)
        closest_protein_list.append(best_protein)
        closest_var_list.append(best_var)
    df_synthetic['Similarity'] = similarities_list
    df_synthetic['Closest Protein'] = closest_protein_list
    df_synthetic['Closest Variable'] = closest_var_list
    unique_classes = sorted(set(var_assoc_original).union(set(var_assoc_synthetic)))
    color_palette = px.colors.qualitative.Plotly
    color_mapping = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(unique_classes)}
    fig = go.Figure()
    for cls in unique_classes:
        df_orig_cls = df_original[df_original['Associated Variable'] == cls]
        df_synth_cls = df_synthetic[df_synthetic['Associated Variable'] == cls]
        count_orig = len(df_orig_cls)
        count_synth = len(df_synth_cls)
        if count_orig > 0:
            fig.add_trace(go.Scatter3d(
                x=df_orig_cls['x'],
                y=df_orig_cls['y'],
                z=df_orig_cls['z'],
                mode='markers',
                marker=dict(size=8, color=color_mapping[cls], opacity=0.7),
                name=f'Original {cls}: {count_orig} samples',
                hoverinfo='text',
                text=df_orig_cls.apply(lambda row: f"Protein ID: {row['Protein ID']}<br>Associated Variable: {row['Associated Variable']}", axis=1)
            ))
        if count_synth > 0:
            def synth_hover_text(row):
                text = f"Protein ID: {row['Protein ID']}<br>Associated Variable: {row['Associated Variable']}"
                if pd.notnull(row['Similarity']):
                    text += f"<br>Similarity: {row['Similarity']:.4f}<br>Closest Protein: {row['Closest Protein']}"
                return text
            fig.add_trace(go.Scatter3d(
                x=df_synth_cls['x'],
                y=df_synth_cls['y'],
                z=df_synth_cls['z'],
                mode='markers',
                marker=dict(size=8, color=color_mapping[cls], opacity=0.7, symbol='diamond'),
                name=f'Synthetic {cls}: {count_synth} samples',
                hoverinfo='text',
                text=df_synth_cls.apply(synth_hover_text, axis=1)
            ))
    fig.update_layout(
        title="3D UMAP with Similarity (Latent Space)",
        scene=dict(xaxis=dict(title='UMAP1'), yaxis=dict(title='UMAP2'), zaxis=dict(title='UMAP3')),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )
    if output_dir:
        umap_similarity_path = os.path.join(output_dir, "umap_similarity_3D.html")
        fig.write_html(umap_similarity_path)
        logging.info(f"UMAP similarity plot saved at {umap_similarity_path}")
    return fig

def plot_predictions_scatterplot_custom(results: dict, output_path: str, top_n: int = 1) -> None:
    """
    Gera um scatter plot mostrando o principal grupo (top 1) para cada proteína.
    """
    protein_specificities = {}
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No ranking for {seq_id}; skipping.")
            continue
        main_category, normalized_confidence = format_and_sum_probabilities(associated_rankings)
        if main_category is None:
            logging.warning(f"No valid formatting for {seq_id}; skipping.")
            continue
        protein_specificities[seq_id] = {
            'top_category': main_category,
            'confidence': normalized_confidence
        }
    if not protein_specificities:
        logging.warning("No data for scatter plot.")
        return
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))
    x_values = list(range(4, 19, 2))
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]
        for spec in specificities:
            ax.scatter(spec, y, color='#1f78b4', edgecolors='black', s=100)
        if len(specificities) > 1:
            ax.plot(specificities, [y]*len(specificities), color='#1f78b4', linestyle='-', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Specificity (C4 to C18)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Scatter Plot of Predictions (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='white')
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='white')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='white')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    from matplotlib import ticker
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Scatter plot saved at {output_path}")

def plot_prediction_confidence_bar(df_results: pd.DataFrame) -> None:
    """
    Cria um gráfico de barras horizontal para visualizar a confiança da predição.
    """
    df_results['Prediction Confidence'] = df_results['Prediction Confidence ( range: 0 - 1 )'].astype(float)
    def get_confidence_level(confidence):
        if confidence < 0.3:
            return "Low Confidence (0 - 0.3)"
        elif confidence < 0.5:
            return "Medium Confidence (0.3 - 0.5)"
        else:
            return "High Confidence (0.5 - 1)"
    df_results['confidence_level'] = df_results['Prediction Confidence'].apply(get_confidence_level)
    color_map = {
        "Low Confidence (0 - 0.3)": "#F4D03F",
        "Medium Confidence (0.3 - 0.5)": "#E67E22",
        "High Confidence (0.5 - 1)": "#3498DB"
    }
    fig_conf = px.bar(
        df_results,
        x='Prediction Confidence',
        y='Query Name',
        orientation='h',
        text=df_results['Prediction Confidence'].apply(lambda x: f"{x:.2f}"),
        color='confidence_level',
        color_discrete_map=color_map
    )
    fig_conf.update_layout(
        title={'text': "Prediction Confidence Bar Chart", 'font': {'color': 'white', 'size': 20}},
        xaxis=dict(range=[0, 1], title={'text': "Prediction Confidence", 'font': {'color': 'white', 'size': 16}}, tickfont={'color': 'white'}),
        yaxis=dict(title={'text': "Query Name", 'font': {'color': 'white', 'size': 16}}, tickfont={'color': 'white'}),
        plot_bgcolor='#0B3C5D',
        paper_bgcolor='#0B3C5D',
        font=dict(color='white'),
        legend_title_text="Confidence Intervals",
        legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    img_bytes = fig_conf.to_image(format="png")
    st.download_button(
        label="Download Bar Chart",
        data=img_bytes,
        file_name="prediction_confidence_bar_chart.png",
        mime="image/png"
    )

# ============================================
# ProteinEmbeddingGenerator and Support Classes
# (Already defined above)
# ============================================

# ============================================
# Main Pipeline Function
# ============================================
def main(args: argparse.Namespace) -> None:
    logging.info("Starting analysis pipeline...")
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # --- Training Phase ---
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
    # --- Prediction Phase ---
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
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled_associated)
    results = {}
    for entry, pred_associated, ranking_associated in zip(protein_embedding_predict.embeddings, predictions_associated, rankings_associated):
        seq_id = entry['protein_accession']
        results[seq_id] = {
            "associated_prediction": pred_associated,
            "associated_ranking": ranking_associated
        }
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tAssociated_Prediction\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['associated_prediction']}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Prediction: {result['associated_prediction']}")
    st.success("Analysis completed successfully!")

    # --- Visualization Section ---
    # Scatter Plot of Predictions
    scatterplot_path = args.scatterplot_output
    plot_predictions_scatterplot_custom(results, scatterplot_path)
    st.header("Scatter Plot of Predictions")
    if os.path.exists(scatterplot_path):
        st.image(scatterplot_path, use_container_width=True)
    else:
        st.error(f"Scatter plot not found at {scatterplot_path}")

    # Example: Plot UMAP for prediction embeddings (3D)
    # Assume we use the same scaler for associated variable and that the variable labels for prediction are not available;
    # here, we simply use the predicted labels as the associated variable for visualization.
    X_umap_predict = X_predict_scaled_associated  # or X_predict directly
    var_assoc_predict = predict_labels  # using predictions as labels
    # For demonstration, we use the same data as "original" and "synthetic" (here, todas as amostras são de previsão)
    fig_umap_3d = plot_umap_3d_combined(X_umap_predict, np.empty((0, X_umap_predict.shape[1])), var_assoc_predict, [])
    st.header("3D UMAP of Prediction Data")
    st.plotly_chart(fig_umap_3d, use_container_width=True)

    # Prediction Confidence Bar Chart
    # Formatar os resultados para gerar a tabela
    formatted_results = []
    for sequence_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            continue
        main_category, normalized_confidence = format_and_sum_probabilities(associated_rankings)
        if main_category is None:
            continue
        formatted_results.append([sequence_id, main_category, f"{normalized_confidence:.2f}"])
    headers = ["Query Name", "SS Prediction Specificity", "Prediction Confidence ( range: 0 - 1 )"]
    df_results = pd.DataFrame(formatted_results, columns=headers)
    st.header("Formatted Results")
    st.dataframe(df_results)
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name='results.csv',
        mime='text/csv',
    )
    # Plot prediction confidence bar chart
    st.header("Prediction Confidence Bar Chart")
    plot_prediction_confidence_bar(df_results)

# ============================================
# Streamlit and CLI Execution
# ============================================
def run_streamlit():
    # Esta parte será executada quando o script for iniciado via Streamlit
    st.title("FAALPred: Fatty Acyl-AMP Ligase Specificity Prediction")
    st.write("Use the sidebar to upload files and configure parameters.")

if __name__ == "__main__":
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        parser = argparse.ArgumentParser(description="FAALPred: Predicting FAAL specificity (CLI Mode)")
        parser.add_argument("--train_fasta", type=str, required=True, help="Path to training FASTA file")
        parser.add_argument("--train_table", type=str, required=True, help="Path to training table (TSV)")
        parser.add_argument("--predict_fasta", type=str, required=True, help="Path to prediction FASTA file")
        parser.add_argument("--kmer_size", type=int, default=3, help="K-mer size")
        parser.add_argument("--step_size", type=int, default=1, help="Step size")
        parser.add_argument("--aggregation_method", type=str, default="mean", choices=["none", "mean"], help="Embedding aggregation method")
        parser.add_argument("--embedding_method", type=str, default="word2vec", choices=["esm2", "seq2vec", "word2vec"], help="Embedding method to use")
        parser.add_argument("--word2vec_model", type=str, default="word2vec_model.bin", help="Filename for embedding model")
        parser.add_argument("--rf_model_associated", type=str, default="rf_model_associated.pkl", help="Filename for RF model for associated")
        parser.add_argument("--learning_curve_associated", type=str, default="learning_curve_associated.png", help="Path to save learning curve plot")
        parser.add_argument("--results_file", type=str, default="predictions.tsv", help="Results file")
        parser.add_argument("--output_dir", type=str, default="results", help="Directory to save models and results")
        parser.add_argument("--scatterplot_output", type=str, default="scatterplot_predictions.png", help="Scatterplot output file")
        parser.add_argument("--excel_output", type=str, default="results.xlsx", help="Excel output file")
        parser.add_argument("--formatted_results_table", type=str, default="formatted_results.txt", help="Formatted results table file")
        parser.add_argument("--roc_curve_associated", type=str, default="roc_curve_associated.png", help="ROC curve for associated")
        parser.add_argument("--window", type=int, default=5, help="Window parameter for embedding")
        parser.add_argument("--workers", type=int, default=48, help="Number of workers for embedding")
        parser.add_argument("--epochs", type=int, default=2500, help="Number of epochs for embedding")
        args = parser.parse_args()
        main(args)
    else:
        run_streamlit()
