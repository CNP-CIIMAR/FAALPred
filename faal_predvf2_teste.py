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
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, average_precision_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV
from PIL import Image
from matplotlib import ticker
import umap.umap_ as umap  # Import para UMAP
import base64
from plotly.graph_objs import Figure
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import optuna  # Import para Busca Bayesiana
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optun
import lightgbm as lgb
import xgboost as xgb
# ============================================
# DefiniÃ§Ãµes de FunÃ§Ãµes e Classes
# ============================================

# Definindo sementes para reprodutibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ConfiguraÃ§Ã£o de Logging
logging.basicConfig(
    level=logging.INFO,  # Mude para DEBUG para mais verbosidade
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),  # Log para arquivo para registros persistentes
    ],
)

# ============================================
# ConfiguraÃ§Ã£o e Interface do Streamlit
# ============================================

# Assegurar que st.set_page_config Ã© o primeiro comando do Streamlit
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="ðŸ”¬",  # SÃ­mbolo de DNA
    layout="wide",
    initial_sidebar_state="expanded",
)

def are_sequences_aligned(fasta_file: str) -> bool:
    """
    Verifica se todas as sequÃªncias em um arquivo FASTA estÃ£o alinhadas, verificando se possuem o mesmo comprimento.
    
    ParÃ¢metros:
    - fasta_file (str): Caminho para o arquivo FASTA.
    
    Retorna:
    - bool: True se todas as sequÃªncias estÃ£o alinhadas (mesmo comprimento), False caso contrÃ¡rio.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Retorna True se todas as sequÃªncias tÃªm o mesmo comprimento

def create_unique_model_directory(base_dir: str, aggregation_method: str) -> str:
    """
    Cria um diretÃ³rio Ãºnico para modelos com base no mÃ©todo de agregaÃ§Ã£o.
    
    ParÃ¢metros:
    - base_dir (str): DiretÃ³rio base para modelos.
    - aggregation_method (str): MÃ©todo de agregaÃ§Ã£o utilizado.
    
    Retorna:
    - str: Caminho para o diretÃ³rio Ãºnico de modelos.
    """
    model_dir = os.path.join(base_dir, f"models_{aggregation_method}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def realign_sequences_with_mafft(input_path: str, output_path: str, threads: int = 8) -> None:
    """
    Realinha sequÃªncias utilizando MAFFT.
    
    ParÃ¢metros:
    - input_path (str): Caminho para o arquivo de entrada.
    - output_path (str): Caminho para salvar o arquivo realinhado.
    - threads (int): NÃºmero de threads para MAFFT.
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"SequÃªncias realinhadas salvas em {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar MAFFT: {e.stderr.decode()}")
        sys.exit(1)

from sklearn.cluster import DBSCAN, KMeans

def perform_clustering(data: np.ndarray, method: str = "DBSCAN", eps: float = 0.5, min_samples: int = 5, n_clusters: int = 3) -> np.ndarray:
    """
    Realiza clustering nos dados usando DBSCAN ou K-Means.
    
    ParÃ¢metros:
    - data (np.ndarray): Dados para clustering.
    - method (str): MÃ©todo de clustering ("DBSCAN" ou "K-Means").
    - eps (float): ParÃ¢metro epsilon para DBSCAN.
    - min_samples (int): NÃºmero mÃ­nimo de amostras para DBSCAN.
    - n_clusters (int): NÃºmero de clusters para K-Means.
    
    Retorna:
    - np.ndarray: Labels gerados pelo mÃ©todo de clustering.
    """
    if method == "DBSCAN":
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "K-Means":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"MÃ©todo de clustering invÃ¡lido: {method}")

    labels = clustering_model.fit_predict(data)
    return labels

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
    """
    Plota a curva ROC para classificaÃ§Ãµes binÃ¡rias ou multiclasse.
    
    ParÃ¢metros:
    - y_true (np.ndarray): Labels verdadeiros.
    - y_pred_proba (np.ndarray): Probabilidades preditas.
    - title (str): TÃ­tulo do plot.
    - save_as (str): Caminho para salvar o plot.
    - classes (list): Lista de classes (para multiclasse).
    """
    lw = 2  # Largura da linha

    # Verifica se Ã© classificaÃ§Ã£o binÃ¡ria ou multiclasse
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # ClassificaÃ§Ã£o binÃ¡ria
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Curva ROC (Ã¡rea = %0.2f)' % roc_auc)
    else:  # ClassificaÃ§Ã£o multiclasse
        y_bin = label_binarize(y_true, classes=unique_classes)
        n_classes = y_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            # Logging dos valores de ROC
            logging.info(f"Para a classe {i}:")
            logging.info(f"FPR: {fpr[i]}")
            logging.info(f"TPR: {tpr[i]}")
            logging.info(f"ROC AUC: {roc_auc[i]}")
            logging.info("--------------------------")

        roc_df = pd.DataFrame(list(roc_auc.items()), columns=['Classe', 'ROC AUC'])
        return roc_df

        plt.figure()

        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            class_label = classes[i] if classes is not None else unique_classes[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'Curva ROC para a classe {class_label} (Ã¡rea = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', color='white')
    plt.ylabel('Taxa de Verdadeiros Positivos', color='white')
    plt.title(title, color='white')
    plt.legend(loc="lower right")
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')  # Combina com a cor de fundo
    plt.close()

def get_class_rankings(model, X: np.ndarray) -> list:
    """
    ObtÃ©m rankings de classes baseados nas probabilidades preditas pelo modelo.
    
    ParÃ¢metros:
    - model: Modelo treinado.
    - X (np.ndarray): Dados para obter previsÃµes.
    
    Retorna:
    - list: Lista de rankings de classes formatados para cada amostra.
    """
    if model is None:
        raise ValueError("Modelo nÃ£o treinado. Por favor, treine o modelo primeiro.")

    # Obter probabilidades para cada classe
    y_pred_proba = model.predict_proba(X)

    # Ranking das classes baseadas nas probabilidades
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings

def calculate_roc_values(model, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Calcula valores de ROC AUC para cada classe.
    
    ParÃ¢metros:
    - model: Modelo treinado.
    - X_test (np.ndarray): Dados de teste.
    - y_test (np.ndarray): Labels verdadeiros de teste.
    
    Retorna:
    - pd.DataFrame: DataFrame contendo valores de ROC AUC por classe.
    """
    n_classes = len(np.unique(y_test))
    y_pred_proba = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Logging dos valores de ROC
        logging.info(f"Para a classe {i}:")
        logging.info(f"FPR: {fpr[i]}")
        logging.info(f"TPR: {tpr[i]}")
        logging.info(f"ROC AUC: {roc_auc[i]}")
        logging.info("--------------------------")

    roc_df = pd.DataFrame(list(roc_auc.items()), columns=['Classe', 'ROC AUC'])
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
    """
    Visualiza o espaÃ§o latente usando UMAP 3D com medidas de similaridade entre amostras originais e sintÃ©ticas.
    
    ParÃ¢metros:
    - X_original (np.ndarray): Embeddings originais.
    - X_synthetic (np.ndarray): Embeddings sintÃ©ticos.
    - y_original (np.ndarray): Labels originais.
    - y_synthetic (np.ndarray): Labels sintÃ©ticas.
    - protein_ids_original (list): IDs das proteÃ­nas originais.
    - protein_ids_synthetic (list): IDs das proteÃ­nas sintÃ©ticas.
    - var_assoc_original (list): VariÃ¡veis associadas dos dados originais.
    - var_assoc_synthetic (list): VariÃ¡veis associadas dos dados sintÃ©ticos.
    - output_dir (str): DiretÃ³rio para salvar o plot.
    
    Retorna:
    - Figure: Objeto da figura Plotly.
    """
    # Combinar dados para UMAP
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.hstack([y_original, y_synthetic])

    # Aplicar UMAP para reduÃ§Ã£o de dimensionalidade
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    X_transformed = umap_reducer.fit_transform(X_combined)

    # Dividir dados transformados
    X_transformed_original = X_transformed[:len(X_original)]
    X_transformed_synthetic = X_transformed[len(X_original):]

    # Calcular similaridades entre amostras sintÃ©ticas e originais
    similarities = cosine_similarity(X_synthetic, X_original)
    max_similarities = similarities.max(axis=1)  # Similaridade mÃ¡xima para cada amostra sintÃ©tica
    closest_indices = similarities.argmax(axis=1)  # Ãndices das amostras originais mais similares

    # Criar DataFrames para facilidade
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

    # Criar plot interativo 3D com Plotly
    fig = go.Figure()

    # Adicionar pontos para amostras originais
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

    # Adicionar pontos para amostras sintÃ©ticas
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

    # Ajustar layout
    fig.update_layout(
        title="Latent Space Visualization with Similarity (UMAP 3D)",
        scene=dict(
            xaxis_title="UMAP DimensÃ£o 1",
            yaxis_title="UMAP DimensÃ£o 2",
            zaxis_title="UMAP DimensÃ£o 3"
        ),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )

    # Salvar o plot se diretÃ³rio de saÃ­da for fornecido
    if output_dir:
        umap_similarity_path = os.path.join(output_dir, "umap_similarity_3D.html")
        fig.write_html(umap_similarity_path)
        logging.info(f"Plot UMAP salvo em {umap_similarity_path}")

    return fig

def format_and_sum_probabilities(associated_rankings: list) -> tuple:
    """
    Formata e soma probabilidades para cada categoria, retornando apenas a categoria principal.
    
    ParÃ¢metros:
    - associated_rankings (list): Lista de rankings associados.
    
    Retorna:
    - tuple: (categoria principal com confianÃ§a, soma das probabilidades, top duas categorias)
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

    # Inicializar o dicionÃ¡rio de somas
    for category in categories:
        category_sums[category] = 0.0

    # Somar probabilidades para cada categoria
    for rank in associated_rankings:
        try:
            prob = float(rank.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Erro ao processar string de ranking: {rank}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob

    if not category_sums:
        return None, None, None  # Sem dados vÃ¡lidos

    # Determinar a categoria principal baseada na soma das probabilidades
    top_category, top_sum = max(category_sums.items(), key=lambda x: x[1])

    # Encontrar as duas principais categorias
    sorted_categories = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)
    top_two = sorted_categories[:2] if len(sorted_categories) >=2 else sorted_categories

    # Extrair as duas principais categorias e suas probabilidades
    top_two_categories = [f"{cat} ({prob:.2f}%)" for cat, prob in top_two]

    # Encontrar a categoria principal com confianÃ§a
    top_category_with_confidence = f"{top_category} ({top_sum:.2f}%)"

    return top_category_with_confidence, top_sum, top_two_categories

class Support:
    """
    Classe de suporte para treinamento e avaliaÃ§Ã£o de modelos com tÃ©cnicas de oversampling.
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
        self.best_model = None

        self.initial_parameters = {
            "n_estimators": 100,  # Mantendo um valor moderado
            "max_depth": 10,  # Reduzindo a profundidade para prevenir overfitting
            "min_samples_split": 5,  # Aumentando para prevenir overfitting
            "min_samples_leaf": 2,  # Mantendo alto para evitar overfitting
            "criterion": "entropy",  # Mantendo como estÃ¡
            "max_features": "sqrt",  # Mantendo uma boa escolha padrÃ£o
            "class_weight": "balanced",  # Usando pesos de classe balanceados
            "ccp_alpha": 0.01,  # Incluindo poda das Ã¡rvores            
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

    def _oversample_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple:
        """
        Aplica tÃ©cnicas de oversampling (ADASYN) para balancear o conjunto de dados.
        
        ParÃ¢metros:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        
        Retorna:
        - tuple: (X_resampled, y_resampled)
        """
        logging.info("Iniciando processo de oversampling com ADASYN...")
        try:
            adasyn = ADASYN(sampling_strategy='auto', random_state=self.seed)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            logging.info(f"DistribuiÃ§Ã£o das classes apÃ³s ADASYN: {Counter(y_resampled)}")
        except ValueError as e:
            logging.error(f"Erro durante ADASYN: {e}")
            sys.exit(1)

        return X_resampled, y_resampled

    def _perform_random_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray) -> tuple:
        """
        Realiza Randomized Search para encontrar os melhores hiperparÃ¢metros.
        
        ParÃ¢metros:
        - X_train_resampled (np.ndarray): Features de treinamento apÃ³s oversampling.
        - y_train_resampled (np.ndarray): Labels de treinamento apÃ³s oversampling.
        
        Retorna:
        - tuple: (Melhor modelo, Melhores parÃ¢metros)
        """
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)
        randomized_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.seed),
            param_distributions=self.grid_search_parameters,
            n_iter=50,  # NÃºmero de combinaÃ§Ãµes aleatÃ³rias a serem avaliadas
            cv=skf,
            scoring='roc_auc_ovo',
            verbose=1,
            random_state=self.seed,
            n_jobs=self.n_jobs
        )
        
        randomized_search.fit(X_train_resampled, y_train_resampled)
        logging.info(f"Melhores parÃ¢metros do randomized search: {randomized_search.best_params_}")
        return randomized_search.best_estimator_, randomized_search.best_params_

    def _perform_bayesian_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray, n_trials: int = 50) -> tuple:
        """
        Realiza OtimizaÃ§Ã£o Bayesiana usando Optuna para encontrar os melhores hiperparÃ¢metros.
        
        ParÃ¢metros:
        - X_train_resampled (np.ndarray): Features de treinamento apÃ³s oversampling.
        - y_train_resampled (np.ndarray): Labels de treinamento apÃ³s oversampling.
        - n_trials (int): NÃºmero de tentativas de otimizaÃ§Ã£o.
        
        Retorna:
        - tuple: (Melhor modelo, Melhores parÃ¢metros)
        """
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
        logging.info(f"Melhores parÃ¢metros da otimizaÃ§Ã£o bayesiana: {best_params}")

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
        """
        Treina modelos com oversampling e cross-validation.
        
        ParÃ¢metros:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        - protein_ids (list): IDs das proteÃ­nas.
        - var_assoc (list): VariÃ¡veis associadas.
        - model_name_prefix (str): Prefixo para salvar o modelo.
        - model_dir (str): DiretÃ³rio para salvar o modelo.
        - min_kmers (int): NÃºmero mÃ­nimo de k-mers.
        
        Retorna:
        - dict: DicionÃ¡rio contendo modelos treinados e seus respectivos parÃ¢metros.
        """
        logging.info(f"Iniciando mÃ©todo fit para {model_name_prefix}...")

        # Converter para arrays numpy
        X = np.array(X)
        y = np.array(y)

        # Determinar min_kmers
        if min_kmers is not None:
            logging.info(f"Usando min_kmers fornecido: {min_kmers}")
        else:
            min_kmers = len(X)
            logging.info(f"min_kmers nÃ£o fornecido. Definido para o tamanho de X: {min_kmers}")

        # Oversampling
        X_resampled, y_resampled = self._oversample_data(X, y)

        # Inicializar dicionÃ¡rio para armazenar modelos
        trained_models = {}

        # Cross-validation com StratifiedKFold
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        # Lista de modelos a serem treinados
        model_types = ['random_forest', 'lightgbm', 'xgboost', 'catboost']

        for model_type in model_types:
            logging.info(f"Treinando modelo: {model_type}")

            # Inicializar listas para mÃ©tricas
            self.train_scores = []
            self.test_scores = []
            self.f1_scores = []
            self.pr_auc_scores = []

            for fold_number, (train_index, test_index) in enumerate(skf.split(X_resampled, y_resampled), start=1):
                X_train, X_test = X_resampled[train_index], X_resampled[test_index]
                y_train, y_test = y_resampled[train_index], y_resampled[test_index]

                # Oversampling para o conjunto de treinamento
                X_train_resampled, y_train_resampled = self._oversample_data(X_train, y_train)

                # SeleÃ§Ã£o e inicializaÃ§Ã£o do modelo
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
                    logging.error(f"Tipo de modelo invÃ¡lido: {model_type}")
                    continue

                # Treinar o modelo
                model.fit(X_train_resampled, y_train_resampled)

                # AvaliaÃ§Ã£o
                train_score = model.score(X_train_resampled, y_train_resampled)
                test_score = model.score(X_test, y_test)
                y_pred = model.predict(X_test)

                # MÃ©tricas
                f1 = f1_score(y_test, y_pred, average='weighted')
                self.f1_scores.append(f1)
                self.train_scores.append(train_score)
                self.test_scores.append(test_score)

                # Precision-Recall AUC
                if len(np.unique(y_test)) > 1:
                    y_pred_proba = model.predict_proba(X_test)
                    pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')
                else:
                    pr_auc = 0.0  # NÃ£o Ã© possÃ­vel calcular PR AUC para uma Ãºnica classe
                self.pr_auc_scores.append(pr_auc)

                logging.info(f"Fold {fold_number} [{model_type}]: Training Score: {train_score}")
                logging.info(f"Fold {fold_number} [{model_type}]: Test Score: {test_score}")
                logging.info(f"Fold {fold_number} [{model_type}]: F1 Score: {f1}")
                logging.info(f"Fold {fold_number} [{model_type}]: Precision-Recall AUC: {pr_auc}")

                # CalibraÃ§Ã£o de Probabilidades
                calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
                calibrator.fit(X_train_resampled, y_train_resampled)
                self.model = calibrator

                # Salvamento do melhor modelo
                if model_dir:
                    best_model_filename = os.path.join(model_dir, f'model_best_{model_type}_fold{fold_number}.pkl')
                    # Assegurar que o diretÃ³rio existe
                    os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                    joblib.dump(calibrator, best_model_filename)
                    logging.info(f"Modelo calibrado salvo como {best_model_filename} para {model_type} Fold {fold_number}")
                else:
                    best_model_filename = f'model_best_{model_type}_fold{fold_number}.pkl'
                    joblib.dump(calibrator, best_model_filename)
                    logging.info(f"Modelo calibrado salvo como {best_model_filename} para {model_type} Fold {fold_number}")

            # ApÃ³s cross-validation, realizar otimizaÃ§Ã£o de hiperparÃ¢metros
            logging.info(f"Iniciando OtimizaÃ§Ã£o Bayesiana para ajuste de hiperparÃ¢metros do modelo {model_type}...")
            best_model, best_params = self._perform_bayesian_search(X_resampled, y_resampled, n_trials=50)
            self.best_params = best_params
            self.model = best_model

            logging.info(f"Melhores parÃ¢metros da OtimizaÃ§Ã£o Bayesiana para {model_type}: {best_params}")

            # Salvar o melhor modelo
            if model_dir:
                best_model_filename = os.path.join(model_dir, f'model_best_bayesian_{model_type}.pkl')
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Melhor modelo Bayesian salvo como {best_model_filename} para {model_type}")
            else:
                best_model_filename = f'model_best_bayesian_{model_type}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Melhor modelo Bayesian salvo como {best_model_filename} para {model_type}")

            # Testar o modelo
            best_score, best_f1, best_pr_auc, best_params, calibrated_model, X_test, y_test = self.test_best_model(
                X_resampled, 
                y_resampled,
                model_type=model_type
            )

            logging.info(f"Melhor ROC AUC para {model_type}: {best_score}")
            logging.info(f"Melhor F1 Score para {model_type}: {best_f1}")
            logging.info(f"Melhor Precision-Recall AUC para {model_type}: {best_pr_auc}")
            logging.info(f"Melhores ParÃ¢metros: {best_params}")

            for param, value in best_params.items():
                logging.info(f"{param}: {value}")

            # Obter rankings de classes
            class_rankings = self.get_class_rankings(calibrated_model, X_test)
            logging.info(f"Rankings de classes para {model_type} gerados com sucesso.")

            # Salvar o modelo treinado
            rf_model_full_path = os.path.join(model_dir, f'rf_model_best_bayesian_{model_type}.pkl')
            joblib.dump(calibrated_model, rf_model_full_path)
            logging.info(f"Modelo calibrado para {model_type} salvo em {rf_model_full_path}")

            # Plotar Curva ROC
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
                classes = None  # ClassificaÃ§Ã£o binÃ¡ria
            else:
                y_pred_proba = calibrated_model.predict_proba(X_test)
                classes = np.unique(y_test).astype(str)
            plot_roc_curve(
                y_test, 
                y_pred_proba, 
                f'Curva ROC para {model_type}', 
                save_as=os.path.join(model_dir, f'roc_curve_{model_type}.png'), 
                classes=classes
            )

            # Salvar mÃ©tricas e modelos no dicionÃ¡rio
            trained_models[model_type] = {
                'model': calibrated_model,
                'best_params': best_params,
                'roc_curve_path': os.path.join(model_dir, f'roc_curve_{model_type}.png'),
                'class_rankings': class_rankings
            }

        # ApÃ³s treinar todos os modelos, plotar a curva de aprendizagem para cada um
        for model_type in model_types:
            logging.info(f"Plotando Curva de Aprendizagem para {model_type}")
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
        """
        Testa o melhor modelo com os dados fornecidos.
        
        ParÃ¢metros:
        - X (np.ndarray): Features para teste.
        - y (np.ndarray): Labels verdadeiros para teste.
        - scaler_dir (str): DiretÃ³rio onde o scaler estÃ¡ salvo.
        - model_type (str): Tipo de modelo ('random_forest', 'lightgbm', etc.)
        
        Retorna:
        - tuple: (Score, F1 Score, Precision-Recall AUC, Melhores parÃ¢metros, Modelo calibrado, X_test, y_test)
        """
        # Carregar o scaler
        scaler_path = os.path.join(scaler_dir, 'scaler_associated.pkl') if scaler_dir else 'scaler_associated.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler carregado de {scaler_path}")
        else:
            logging.error(f"Scaler nÃ£o encontrado em {scaler_path}. Deve ser scaler_associated.pkl.")
            sys.exit(1)

        X_scaled = scaler.transform(X)

        # Dividir em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.4, random_state=self.seed, stratify=y
        )

        # Treinar o modelo com os melhores parÃ¢metros
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
            logging.error(f"Tipo de modelo '{model_type}' nÃ£o suportado.")
            sys.exit(1)

        model.fit(X_train, y_train)

        # CalibraÃ§Ã£o de Probabilidades
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        # Fazer previsÃµes
        y_pred_proba = calibrated_model.predict_proba(X_test)
        y_pred_classes = calibrated_model.predict(X_test)

        # Calcular mÃ©tricas
        score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')

        logging.info(f"ROC AUC Score: {score}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Precision-Recall AUC: {pr_auc}")

        # Retornar as mÃ©tricas e o modelo calibrado
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def plot_learning_curve(self, output_path: str) -> None:
        """
        Plota a curva de aprendizagem do modelo.
        
        ParÃ¢metros:
        - output_path (str): Caminho para salvar o plot.
        """
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
        plt.savefig(output_path, facecolor='#0B3C5D')  # Combina com a cor de fundo
        plt.close()

    def get_class_rankings(self, X: np.ndarray) -> list:
        """
        ObtÃ©m rankings de classes para os dados fornecidos.
        
        ParÃ¢metros:
        - X (np.ndarray): Dados para obter previsÃµes.
        
        Retorna:
        - list: Lista de rankings de classes formatados para cada amostra.
        """
        if self.model is None:
            raise ValueError("Modelo nÃ£o treinado. Por favor, treine o modelo primeiro.")

        # Obter probabilidades para cada classe
        y_pred_proba = self.model.predict_proba(X)

        # Ranking das classes baseadas nas probabilidades
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def plot_roc_curve_custom(self, y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
        """
        Plota a curva ROC para classificaÃ§Ãµes binÃ¡rias ou multiclasse.
        
        ParÃ¢metros:
        - y_true (np.ndarray): Labels verdadeiros.
        - y_pred_proba (np.ndarray): Probabilidades preditas.
        - title (str): TÃ­tulo do plot.
        - save_as (str): Caminho para salvar o plot.
        - classes (list): Lista de classes (para multiclasse).
        """
        plot_roc_curve(y_true, y_pred_proba, title, save_as, classes)

class ProteinEmbeddingGenerator:
    """
    Classe para gerar embeddings de proteÃ­nas a partir de alinhamentos e tabelas.
    """
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
        """
        Gera embeddings para proteÃ­nas usando k-mers e Word2Vec.
        
        ParÃ¢metros:
        - k (int): Tamanho do k-mer.
        - step_size (int): Tamanho do passo.
        - word2vec_model_path (str): Caminho para salvar o modelo Word2Vec.
        - model_dir (str): DiretÃ³rio para salvar o modelo Word2Vec.
        - min_kmers (int): NÃºmero mÃ­nimo de k-mers para considerar uma proteÃ­na.
        - save_min_kmers (bool): Indica se o min_kmers deve ser salvo.
        """
        logging.info("Iniciando geraÃ§Ã£o de embeddings para proteÃ­nas...")

        # Gerar k-mers para cada sequÃªncia
        sequences = list(SeqIO.parse(self.fasta_path, "fasta"))
        kmer_sequences = []
        for record in sequences:
            sequence = str(record.seq)
            kmers = [sequence[i:i+k] for i in range(0, len(sequence) - k +1, step_size)]
            if len(kmers) < min_kmers:
                continue  # Ignorar sequÃªncias com menos de min_kmers
            kmer_sequences.append(kmers)

        logging.info(f"Total de sequÃªncias apÃ³s filtragem: {len(kmer_sequences)}")

        # Treinar Word2Vec
        if not os.path.exists(word2vec_model_path):
            logging.info("Treinando modelo Word2Vec...")
            w2v_model = Word2Vec(sentences=kmer_sequences, vector_size=100, window=5, min_count=1, workers=8, epochs=2500, seed=SEED)
            w2v_model.save(word2vec_model_path)
            logging.info(f"Modelo Word2Vec salvo em {word2vec_model_path}")
        else:
            logging.info(f"Carregando modelo Word2Vec existente de {word2vec_model_path}")
            w2v_model = Word2Vec.load(word2vec_model_path)

        # Gerar embeddings para cada sequÃªncia
        for kmers in kmer_sequences:
            embeddings = [w2v_model.wv[kmer] for kmer in kmers if kmer in w2v_model.wv]
            if not embeddings:
                continue
            if self.aggregation_method == 'mean':
                sequence_embedding = np.mean(embeddings, axis=0)
            else:
                sequence_embedding = np.concatenate(embeddings)
            self.embeddings.append({
                'protein_accession': kmers[0],  # Substitua conforme necessÃ¡rio
                'embedding': sequence_embedding
            })

        logging.info(f"Total de embeddings gerados: {len(self.embeddings)}")

        if save_min_kmers:
            # Salvar min_kmers para consistÃªncia
            min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
            with open(min_kmers_path, 'w') as f:
                f.write(str(min_kmers))
            logging.info(f"min_kmers salvo em {min_kmers_path}")

        self.min_kmers = min_kmers

    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        """
        ObtÃ©m embeddings e labels baseados no tipo de label.
        
        ParÃ¢metros:
        - label_type (str): Tipo de label ('associated_variable').
        
        Retorna:
        - tuple: (embeddings, labels)
        """
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
    """
    Ajusta as probabilidades preditas pelo modelo.
    
    ParÃ¢metros:
    - predicted_proba (np.ndarray): Probabilidades preditas pelo modelo.
    - method (str): MÃ©todo de ajuste ('normalize', 'smoothing', 'none').
    - alpha (float): ParÃ¢metro para smoothing (usado se method='smoothing').
    
    Retorna:
    - np.ndarray: Probabilidades ajustadas.
    """
    if method == 'normalize':
        # Normalizar probabilidades para que somem 1 para cada amostra
        logging.info("Normalizando probabilidades preditas.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)

    elif method == 'smoothing':
        # Aplicar smoothing nas probabilidades para evitar valores extremos
        logging.info(f"Aplicando smoothing nas probabilidades preditas com alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])

    elif method == 'none':
        # NÃ£o aplicar nenhum ajuste
        logging.info("Nenhum ajuste aplicado Ã s probabilidades preditas.")
        adjusted_proba = predicted_proba.copy()

    else:
        logging.warning(f"MÃ©todo de ajuste desconhecido '{method}'. Nenhum ajuste serÃ¡ aplicado.")
        adjusted_proba = predicted_proba.copy()

    return adjusted_proba

def main(args: argparse.Namespace) -> None:
    """
    FunÃ§Ã£o principal coordenando o fluxo de trabalho.
    
    ParÃ¢metros:
    - args (argparse.Namespace): Argumentos de entrada.
    """
    model_dir = args.model_dir

    # Inicializar indicadores de progresso
    total_steps = 7  # Ajustado para incluir dual UMAP e modelos adicionais
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # =============================
    # PASSO 1: Treinamento dos Modelos para associated_variable
    # =============================

    # Carregar dados de treinamento
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table

    # Verificar se as sequÃªncias de treinamento estÃ£o alinhadas
    if not are_sequences_aligned(train_alignment_path):
        logging.info("SequÃªncias de treinamento nÃ£o estÃ£o alinhadas. Realinhando com MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)  # Threads setado para 1
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Arquivo de treinamento alinhado encontrado ou sequÃªncias jÃ¡ estÃ£o alinhadas: {train_alignment_path}")

    # Carregar dados da tabela de treinamento
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Dados da tabela de treinamento carregados com sucesso.")

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Inicializar e gerar embeddings para treinamento
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, 
        table_data=train_table_data, 
        aggregation_method=args.aggregation_method  # Passando o mÃ©todo de agregaÃ§Ã£o ('none' ou 'mean')
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=args.min_kmers,
        save_min_kmers=True  # Salvar min_kmers apÃ³s o treinamento
    )
    logging.info(f"NÃºmero de embeddings de treinamento gerados: {len(protein_embedding_train.embeddings)}")

    # Salvar min_kmers para garantir consistÃªncia
    min_kmers = protein_embedding_train.min_kmers

    # Obter IDs de proteÃ­nas e variÃ¡veis associadas do conjunto de treinamento
    protein_ids_associated = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]
    var_assoc_associated = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]

    logging.info(f"IDs de proteÃ­nas para associated_variable extraÃ­dos: {len(protein_ids_associated)}")
    logging.info(f"VariÃ¡veis associadas para associated_variable extraÃ­das: {len(var_assoc_associated)}")

    # Obter embeddings e labels para associated_variable
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"Shape de X_associated: {X_associated.shape}")

    # Criar scaler para X_associated
    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info("Scaler para X_associated criado e salvo.")

    # Escalar os dados X_associated
    X_associated_scaled = scaler_associated.transform(X_associated)    

    # Caminhos completos para os modelos
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Verificar se o modelo calibrado para associated_variable jÃ¡ existe
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Modelo Random Forest calibrado para associated_variable carregado de {calibrated_model_associated_full_path}")
    else:
        # Treinar os modelos
        support_models = Support()
        trained_models = support_models.fit(
            X_associated_scaled, 
            y_associated, 
            protein_ids=protein_ids_associated,  # Passar IDs de proteÃ­nas para visualizaÃ§Ã£o
            var_assoc=var_assoc_associated, 
            model_name_prefix='associated', 
            model_dir=model_dir, 
            min_kmers=min_kmers
        )

        logging.info("Treinamento e calibraÃ§Ã£o para associated_variable concluÃ­do.")

        # Plotar a curva de aprendizagem para cada modelo
        for model_type, model_info in trained_models.items():
            logging.info(f"Plotando Curva de Aprendizagem para {model_type}")
            learning_curve_path = os.path.join(model_dir, f'learning_curve_{model_type}.png')
            support_models.plot_learning_curve(learning_curve_path)

        # Carregar o modelo calibrado
        calibrated_model_associated = trained_models['random_forest']['model']
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Modelo Random Forest calibrado para associated_variable salvo em {calibrated_model_associated_full_path}")

        # Testar o modelo
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_models.test_best_model(
            X_associated_scaled, 
            y_associated,
            model_type='random_forest'
        )

        logging.info(f"Melhor ROC AUC para associated_variable: {best_score_associated}")
        logging.info(f"Melhor F1 Score para associated_variable: {best_f1_associated}")
        logging.info(f"Melhor Precision-Recall AUC para associated_variable: {best_pr_auc_associated}")
        logging.info(f"Melhores ParÃ¢metros: {best_params_associated}")

        for param, value in best_params_associated.items():
            logging.info(f"{param}: {value}")

        # Obter rankings de classes para associated_variable
        class_rankings_associated = support_models.get_class_rankings(X_test_associated)
        logging.info("Rankings de classes para associated_variable gerados com sucesso.")

        # Acessar class_weight dos melhores parÃ¢metros
        class_weight = best_params_associated.get('class_weight', None)
        logging.info(f"Class weight utilizado: {class_weight}")

        # Salvar o modelo treinado
        joblib.dump(best_model_associated, rf_model_associated_full_path)
        logging.info(f"Modelo Random Forest para associated_variable salvo em {rf_model_associated_full_path}")

        # Plotar Curva ROC para associated_variable
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
            classes_associated = None  # ClassificaÃ§Ã£o binÃ¡ria
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve(
            y_test_associated, 
            y_pred_proba_associated, 
            'Curva ROC para associated_variable', 
            save_as=args.roc_curve_associated, 
            classes=classes_associated
        )

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # PASSO 2: ClassificaÃ§Ã£o de Novas SequÃªncias para associated_variable
    # =============================

    # Carregar min_kmers
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers carregado: {min_kmers_loaded}")
    else:
        logging.error(f"Arquivo min_kmers nÃ£o encontrado em {min_kmers_path}. Por favor, assegure-se de que o treinamento foi concluÃ­do com sucesso.")
        sys.exit(1)

    # Carregar dados de prediÃ§Ã£o
    predict_alignment_path = args.predict_fasta

    # Verificar se as sequÃªncias de prediÃ§Ã£o estÃ£o alinhadas
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("SequÃªncias para prediÃ§Ã£o nÃ£o estÃ£o alinhadas. Realinhando com MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Threads setado para 1
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Arquivo alinhado para prediÃ§Ã£o encontrado ou sequÃªncias jÃ¡ estÃ£o alinhadas: {predict_alignment_path}")

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Inicializar ProteinEmbedding para prediÃ§Ã£o, sem necessidade da tabela
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, 
        table_data=None,
        aggregation_method=args.aggregation_method  # Passando o mÃ©todo de agregaÃ§Ã£o ('none' ou 'mean')
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded,  # Usar o mesmo min_kmers do treinamento
        save_min_kmers=False  # NÃ£o salvar min_kmers novamente
    )
    logging.info(f"NÃºmero de embeddings gerados para prediÃ§Ã£o: {len(protein_embedding_predict.embeddings)}")

    # Obter embeddings para prediÃ§Ã£o
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Carregar scaler para associated_variable
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')

    if os.path.exists(scaler_associated_path):
        scaler_associated = joblib.load(scaler_associated_path)
        logging.info("Scaler para associated_variable carregado com sucesso.")
    else:
        logging.error("Scalers nÃ£o encontrados. Por favor, assegure-se de que o treinamento foi concluÃ­do com sucesso.")
        sys.exit(1)

    # Escalar embeddings de prediÃ§Ã£o usando scaler_associated
    X_predict_scaled_associated = scaler_associated.transform(X_predict)

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Realizar prediÃ§Ãµes para associated_variable com todos os modelos treinados
    logging.info("Realizando prediÃ§Ãµes para associated_variable com todos os modelos treinados...")
    predictions = {}
    rankings = {}
    for model_type, model_info in trained_models.items():
        model = model_info['model']
        pred = model.predict(X_predict_scaled_associated)
        predictions[model_type] = pred
        rankings[model_type] = support_models.get_class_rankings(X_predict_scaled_associated)
        logging.info(f"PrediÃ§Ãµes realizadas para {model_type}.")

    # Processar e salvar resultados
    results = {}
    for i, entry in enumerate(protein_embedding_predict.embeddings):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {}
        for model_type in predictions.keys():
            results[sequence_id][f"{model_type}_prediction"] = predictions[model_type][i]
            results[sequence_id][f"{model_type}_ranking"] = rankings[model_type][i]

    # Salvar resultados em um arquivo
    with open(args.results_file, 'w') as f:
        # CabeÃ§alho
        headers = ["Protein_ID"]
        for model_type in predictions.keys():
            headers.append(f"{model_type}_Prediction")
            headers.append(f"{model_type}_Ranking")
        f.write("\t".join(headers) + "\n")

        # Dados
        for seq_id, result in results.items():
            row = [seq_id]
            for model_type in predictions.keys():
                row.append(str(result.get(f"{model_type}_prediction", 'Unknown')))
                row.append("; ".join(result.get(f"{model_type}_ranking", ['Unknown'])))
            f.write("\t".join(row) + "\n")
            logging.info(f"{seq_id} - PrediÃ§Ãµes: {', '.join([str(result.get(f'{model_type}_prediction', 'Unknown')) for model_type in predictions.keys()])}")

    # Gerar o Scatter Plot das PrediÃ§Ãµes
    logging.info("Gerando scatter plot das prediÃ§Ãµes para novas sequÃªncias...")
    # Apenas para um modelo (por exemplo, Random Forest), ou adaptar conforme necessÃ¡rio
    # Aqui, para simplicidade, vamos gerar o scatter plot para o Random Forest
    if 'random_forest' in predictions:
        scatterplot_path = args.scatterplot_output
        plot_predictions_scatterplot_custom(results, scatterplot_path)
        logging.info(f"Scatter plot salvo em {scatterplot_path}")

    # =============================
    # PASSO 3: Plot Dual UMAP e Dual t-SNE
    # =============================

    # Plot Dual UMAP e Dual t-SNE para dados de treinamento e prediÃ§Ã£o
    logging.info("Gerando plots Dual UMAP e Dual t-SNE para dados de treinamento e prediÃ§Ã£o...")
    train_labels = y_associated
    predict_labels = predictions['random_forest'] if 'random_forest' in predictions else predictions[next(iter(predictions))]
    train_protein_ids = protein_ids_associated
    predict_protein_ids = [entry['protein_accession'] for entry in protein_embedding_predict.embeddings]

    # Gerar Dual UMAP
    fig_umap_train, fig_umap_predict = visualize_latent_space_with_similarity(
        train_embeddings=X_associated_scaled, 
        train_labels=train_labels,
        train_protein_ids=train_protein_ids,
        predict_embeddings=X_predict_scaled_associated, 
        predict_labels=predict_labels, 
        predict_protein_ids=predict_protein_ids, 
        var_assoc_original=var_assoc_associated,
        var_assoc_synthetic=[results[seq_id].get(f"random_forest_prediction", 'Unknown') for seq_id in predict_protein_ids],
        output_dir=model_dir
    )

    # Gerar Dual t-SNE
    fig_tsne_train, fig_tsne_predict = plot_dual_tsne(
        train_embeddings=X_associated_scaled, 
        train_labels=train_labels,
        train_protein_ids=train_protein_ids,
        predict_embeddings=X_predict_scaled_associated, 
        predict_labels=predict_labels, 
        predict_protein_ids=predict_protein_ids, 
        output_dir=model_dir
    )
    logging.info("Plots Dual UMAP e Dual t-SNE gerados com sucesso.")

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    st.success("AnÃ¡lise concluÃ­da com sucesso!")

    # Exibir scatter plot
    st.header("Scatter Plot das PrediÃ§Ãµes")
    scatterplot_path = args.scatterplot_output
    if os.path.exists(scatterplot_path):
        st.image(scatterplot_path, use_column_width=True)
    else:
        st.error(f"Scatter plot nÃ£o encontrado em {scatterplot_path}")

    # Formatar os resultados
    formatted_results = []

    for sequence_id, info in results.items():
        row = [sequence_id]
        for model_type in predictions.keys():
            prediction = info.get(f"{model_type}_prediction", 'Unknown')
            ranking = info.get(f"{model_type}_ranking", ['Unknown'])
            top_specificity = ranking[0] if ranking else 'Unknown'
            confidence = float(ranking[0].split(": ")[1].replace("%", "")) if ranking else 0.0
            top_two_specificities = ranking[:2] if len(ranking) >=2 else ranking
            formatted_results.append([
                sequence_id,
                f"{model_type.capitalize()} - {top_specificity}",
                f"{confidence:.2f}%",
                "; ".join(top_two_specificities)
            ])

    # Converter para pandas DataFrame
    headers = ["Protein_ID", "SS Prediction Specificity", "Prediction Confidence", "Top 2 Specificities"]
    df_results = pd.DataFrame(formatted_results, columns=headers)

    # FunÃ§Ã£o para aplicar estilos personalizados
    def highlight_table(df):
        return df.style.set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#1E3A8A'),  # Azul escuro para cabeÃ§alhos
                    ('color', 'white'),
                    ('border', '1px solid white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('background-color', '#0B3C5D'),  # Azul marinho para linhas Ã­mpares
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
                    ('background-color', '#145B9C')  # Azul um pouco mais claro para linhas pares
                ]
            },
            {
                'selector': 'tr:hover td',
                'props': [
                    ('background-color', '#0D4F8B')  # Azul mais escuro no hover
                ]
            },
        ])

    # Aplicar estilos ao DataFrame
    styled_df = highlight_table(df_results)

    # Renderizar a tabela como HTML
    html = styled_df.to_html(index=False, escape=False)

    # Injetar CSS para botÃµes de download e ajustar estilos adicionais
    st.markdown(
        """
        <style>
        /* Estilo para botÃµes de download */
        .stDownloadButton > button {
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

        /* Efeito hover para botÃµes */
        .stDownloadButton > button:hover {
            background-color: #0B3C5D;
        }

        /* Estilo para a tabela */
        table {
            border-collapse: collapse;
            width: 100%;
        }

        /* Ajustar scroll para a tabela */
        .dataframe-container {
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Exibir a tabela no Streamlit
    st.header("Resultados Formatados")

    st.markdown(
        f"""
        <div class="dataframe-container">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # BotÃµes para download em CSV e Excel
    # EstilizaÃ§Ã£o jÃ¡ coberta pelo CSS acima

    # BotÃ£o para download em CSV
    st.download_button(
        label="Baixar Resultados como CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name='results.csv',
        mime='text/csv',
    )

    # BotÃ£o para download em Excel
    output = BytesIO() 
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Resultados')
        writer.close()

        processed_data = output.getvalue()

    st.download_button(
        label="Baixar Resultados como Excel",
        data=processed_data,
        file_name='results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    # Preparar o arquivo results.zip
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for folder_name, subfolders, filenames in os.walk(model_dir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, model_dir))
    zip_buffer.seek(0)

    # Fornecer link de download para todos os resultados
    st.header("Baixar Todos os Resultados")
    st.download_button(
        label="Baixar Todos os Resultados como results.zip",
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
    logging.info(f"Tabela formatada salva em {args.formatted_results_table}")

    # ============================================
    # Fim do Script
    # ============================================
st.markdown(
    """
    <style>
    /* Define o fundo principal do app e a cor do texto */
    .stApp {
        background-color: #0B3C5D;
        color: white;
    }
    /* Define o fundo da barra lateral e a cor do texto */
    [data-testid="stSidebar"] {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Garante que todos os elementos dentro da barra lateral tenham fundo azul marinho escuro e texto branco */
    [data-testid="stSidebar"] * {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Personaliza elementos de entrada dentro da barra lateral */
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
    /* Personaliza a Ã¡rea de arrastar e soltar do uploader de arquivos */
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] div {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Personaliza as opÃ§Ãµes do dropdown de seleÃ§Ã£o */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Remove bordas e sombras */
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
    /* Personaliza checkboxes e radios */
    [data-testid="stSidebar"] .stCheckbox input[type="checkbox"] + div:first-of-type,
    [data-testid="stSidebar"] .stRadio input[type="radio"] + div:first-of-type {
        background-color: #1E3A8A !important;
    }
    /* Personaliza a barra de sliders */
    [data-testid="stSidebar"] .stSlider > div:first-of-type {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider .st-bo {
        background-color: #1E3A8A !important;
    }
    /* Garante que os cabeÃ§alhos sejam brancos */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    /* Garante que mensagens de alerta (st.info, st.error, etc.) tenham texto branco */
    div[role="alert"] p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def plot_dual_tsne(
    train_embeddings: np.ndarray, 
    train_labels: list,
    train_protein_ids: list,
    predict_embeddings: np.ndarray, 
    predict_labels: list, 
    predict_protein_ids: list, 
    output_dir: str
) -> tuple:
    """
    Plota dois grÃ¡ficos t-SNE 3D e os salva como arquivos HTML:
    - Plot 1: Dados de Treinamento.
    - Plot 2: PrediÃ§Ãµes.
    
    ParÃ¢metros:
    - train_embeddings (np.ndarray): Embeddings de treinamento.
    - train_labels (list): Labels dos dados de treinamento.
    - train_protein_ids (list): IDs das proteÃ­nas de treinamento.
    - predict_embeddings (np.ndarray): Embeddings de prediÃ§Ã£o.
    - predict_labels (list): Labels das prediÃ§Ãµes.
    - predict_protein_ids (list): IDs das proteÃ­nas das prediÃ§Ãµes.
    - output_dir (str): DiretÃ³rio para salvar os plots t-SNE.
    
    Retorna:
    - tuple: (Figura de Treinamento, Figura de PrediÃ§Ã£o)
    """
    from sklearn.manifold import TSNE

    # ReduÃ§Ã£o de dimensionalidade usando t-SNE
    tsne_train = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_train_result = tsne_train.fit_transform(train_embeddings)

    tsne_predict = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_predict_result = tsne_predict.fit_transform(predict_embeddings)

    # Criar mapas de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapas de cores para as prediÃ§Ãµes
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Plot 1: Dados de Treinamento
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
        # IDs de proteÃ­nas reais adicionados ao 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Dados de Treinamento'
    ))
    fig_train.update_layout(
        title='t-SNE 3D: Dados de Treinamento',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Plot 2: PrediÃ§Ãµes
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
        # IDs de proteÃ­nas adicionados ao 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='PrediÃ§Ãµes'
    ))
    fig_predict.update_layout(
        title='t-SNE 3D: PrediÃ§Ãµes',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Salvar os plots como arquivos HTML
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")
    
    pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
    pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)
    
    logging.info(f"Plot t-SNE de Treinamento salvo como {tsne_train_html}")
    logging.info(f"Plot t-SNE de PrediÃ§Ãµes salvo como {tsne_predict_html}")

    return fig_train, fig_predict

def plot_dual_umap(
    train_embeddings: np.ndarray, 
    train_labels: list,
    train_protein_ids: list,
    predict_embeddings: np.ndarray, 
    predict_labels: list, 
    predict_protein_ids: list, 
    var_assoc_original: list,
    var_assoc_synthetic: list,
    output_dir: str
) -> tuple:
    """
    Plota dois grÃ¡ficos UMAP 3D e os salva como arquivos HTML:
    - Plot 1: Dados de Treinamento.
    - Plot 2: PrediÃ§Ãµes.
    
    ParÃ¢metros:
    - train_embeddings (np.ndarray): Embeddings de treinamento.
    - train_labels (list): Labels dos dados de treinamento.
    - train_protein_ids (list): IDs das proteÃ­nas de treinamento.
    - predict_embeddings (np.ndarray): Embeddings de prediÃ§Ã£o.
    - predict_labels (list): Labels das prediÃ§Ãµes.
    - predict_protein_ids (list): IDs das proteÃ­nas das prediÃ§Ãµes.
    - var_assoc_original (list): VariÃ¡veis associadas dos dados originais.
    - var_assoc_synthetic (list): VariÃ¡veis associadas dos dados sintÃ©ticos.
    - output_dir (str): DiretÃ³rio para salvar os plots UMAP.
    
    Retorna:
    - tuple: (Figura de Treinamento, Figura de PrediÃ§Ã£o)
    """
    # ReduÃ§Ã£o de dimensionalidade usando UMAP
    umap_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_train_result = umap_train.fit_transform(train_embeddings)

    umap_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_predict_result = umap_predict.fit_transform(predict_embeddings)

    # Criar mapas de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapas de cores para as prediÃ§Ãµes
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Plot 1: Dados de Treinamento
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
        # IDs de proteÃ­nas reais adicionados ao 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Dados de Treinamento'
    ))
    fig_train.update_layout(
        title='UMAP 3D: Dados de Treinamento',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Plot 2: PrediÃ§Ãµes
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
        # IDs de proteÃ­nas adicionados ao 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='PrediÃ§Ãµes'
    ))
    fig_predict.update_layout(
        title='UMAP 3D: PrediÃ§Ãµes',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Salvar os plots como arquivos HTML
    umap_train_html = os.path.join(output_dir, "umap_train_3d.html")
    umap_predict_html = os.path.join(output_dir, "umap_predict_3d.html")
    
    pio.write_html(fig_train, file=umap_train_html, auto_open=False)
    pio.write_html(fig_predict, file=umap_predict_html, auto_open=False)
    
    logging.info(f"Plot UMAP de Treinamento salvo como {umap_train_html}")
    logging.info(f"Plot UMAP de PrediÃ§Ãµes salvo como {umap_predict_html}")

    return fig_train, fig_predict

def plot_predictions_scatterplot_custom(
    results: dict, 
    output_path: str, 
    top_n: int = 1
) -> None:
    """
    Gera um scatter plot mostrando apenas a categoria principal com a maior soma de probabilidades para cada proteÃ­na.
    
    Eixo Y: ID de acesso da proteÃ­na
    Eixo X: Specificidades de C4 a C18 (escala fixa)
    Cada ponto representa a especificidade correspondente para a proteÃ­na.
    Apenas a categoria principal (top 1) Ã© plotada por proteÃ­na.
    Pontos sÃ£o coloridos em uma Ãºnica cor uniforme, estilizados para publicaÃ§Ã£o cientÃ­fica.
    
    ParÃ¢metros:
    - results (dict): DicionÃ¡rio contendo prediÃ§Ãµes e rankings para proteÃ­nas.
    - output_path (str): Caminho para salvar o scatter plot.
    - top_n (int): NÃºmero de top categorias principais a plotar (padrÃ£o Ã© 1).
    """
    # Preparar dados
    protein_specificities = {}
    
    for seq_id, info in results.items():
        for model_type in ['random_forest', 'lightgbm', 'xgboost', 'catboost']:
            associated_rankings = info.get(f'{model_type}_ranking', [])
            if not associated_rankings:
                logging.warning(f"Sem dados de ranking associados para a proteÃ­na {seq_id} com {model_type}. Pulando...")
                continue

            # Usar a funÃ§Ã£o format_and_sum_probabilities para obter a categoria principal
            top_category_with_confidence, confidence, top_two_categories = format_and_sum_probabilities(associated_rankings)
            if top_category_with_confidence is None:
                logging.warning(f"Sem formataÃ§Ã£o vÃ¡lida para a proteÃ­na {seq_id} com {model_type}. Pulando...")
                continue

            # Extrair a categoria sem confianÃ§a
            category = top_category_with_confidence.split(" (")[0]
            confidence = confidence  # Soma das probabilidades para a categoria principal

            protein_specificities[f"{seq_id}_{model_type}"] = {
                'top_category': category,
                'confidence': confidence
            }

    if not protein_specificities:
        logging.warning("Nenhum dado disponÃ­vel para plotar o scatter plot.")
        return

    # Ordenar IDs de proteÃ­nas para melhor visualizaÃ§Ã£o
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Criar a figura
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))  # Ajustar altura baseada no nÃºmero de proteÃ­nas

    # Escala fixa para o Eixo X de C4 a C18
    x_values = list(range(4, 19))

    # Plotar pontos para todas as proteÃ­nas com sua categoria principal
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        confidence = data['confidence']

        # Extrair specificidades da string da categoria
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]

        for spec in specificities:
            ax.scatter(
                spec, y,
                color='#1f78b4',  # Cor uniforme
                edgecolors='black',
                linewidth=0.5,
                s=100,
                label='_nolegend_'  # Evitar duplicaÃ§Ã£o na legenda
            )

        # Conectar pontos com linhas se houver mÃºltiplas specificidades
        if len(specificities) > 1:
            ax.plot(
                specificities,
                [y] * len(specificities),
                color='#1f78b4',
                linestyle='-',
                linewidth=1.0,
                alpha=0.7
            )

    # Personalizar o plot para melhor qualidade de publicaÃ§Ã£o cientÃ­fica
    ax.set_xlabel('Specificity (C4 to C18)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Scatter Plot of Predictions for New Sequences (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='white')

    # Definir escala fixa e formataÃ§Ã£o para o Eixo X
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='white')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='white')

    # Definir grade e remover spines desnecessÃ¡rios para um visual limpo
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Ticks menores no Eixo X para melhor visibilidade
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Ajustar layout para evitar corte de labels
    plt.tight_layout()

    # Salvar a figura em alta qualidade para publicaÃ§Ã£o
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=600, bbox_inches='tight')  # Combina com a cor de fundo
    plt.close()
    logging.info(f"Scatter plot salvo em {output_path}")

# FunÃ§Ã£o para carregar e redimensionar imagens com ajuste de DPI
def load_and_resize_image_with_dpi(image_path: str, base_width: int, dpi: int = 300) -> Image.Image:
    """
    Carrega e redimensiona uma imagem com ajuste de DPI.
    
    ParÃ¢metros:
    - image_path (str): Caminho para o arquivo de imagem.
    - base_width (int): Largura base para redimensionamento.
    - dpi (int): DPI para a imagem.
    
    Retorna:
    - Image.Image: Objeto de imagem redimensionada.
    """
    try:
        # Carrega a imagem
        image = Image.open(image_path)
        # Calcula a nova altura proporcional Ã  largura base
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Redimensiona a imagem
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Imagem nÃ£o encontrada em {image_path}.")
        return None

# FunÃ§Ã£o para codificar imagens em base64
def encode_image(image: Image.Image) -> str:
    """
    Codifica uma imagem como uma string base64.
    
    ParÃ¢metros:
    - image (Image.Image): Objeto de imagem.
    
    Retorna:
    - str: String base64 da imagem.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

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

# Carrega e redimensiona todas as imagens
images = [load_and_resize_image_with_dpi(path, base_width=150, dpi=300) for path in image_paths]

# Codifica as imagens em base64
encoded_images = [encode_image(img) for img in images if img is not None]

# CSS para layout do rodapÃ©
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

# HTML para exibir imagens no rodapÃ©
footer_html = """
<div class="support-text">Supported by:</div>
<div class="footer-container">
    {}
</div>
<div class="footer-text">
    CIIMAR - Pedro LeÃ£o @CNP - 2024 - All rights reserved.
</div>

"""

# Gera tags <img> para cada imagem
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Renderiza o rodapÃ©
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)

# ============================================
# Fim do Script
# ============================================

# ============================================
# FunÃ§Ãµes Adicionais para t-SNE
# ============================================

def plot_dual_tsne(
    train_embeddings: np.ndarray, 
    train_labels: list,
    train_protein_ids: list,
    predict_embeddings: np.ndarray, 
    predict_labels: list, 
    predict_protein_ids: list, 
    output_dir: str
) -> tuple:
    """
    Plota dois grÃ¡ficos t-SNE 3D e os salva como arquivos HTML:
    - Plot 1: Dados de Treinamento.
    - Plot 2: PrediÃ§Ãµes.
    
    ParÃ¢metros:
    - train_embeddings (np.ndarray): Embeddings de treinamento.
    - train_labels (list): Labels dos dados de treinamento.
    - train_protein_ids (list): IDs das proteÃ­nas de treinamento.
    - predict_embeddings (np.ndarray): Embeddings de prediÃ§Ã£o.
    - predict_labels (list): Labels das prediÃ§Ãµes.
    - predict_protein_ids (list): IDs das proteÃ­nas das prediÃ§Ãµes.
    - output_dir (str): DiretÃ³rio para salvar os plots t-SNE.
    
    Retorna:
    - tuple: (Figura de Treinamento, Figura de PrediÃ§Ã£o)
    """
    from sklearn.manifold import TSNE

    # ReduÃ§Ã£o de dimensionalidade usando t-SNE
    tsne_train = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_train_result = tsne_train.fit_transform(train_embeddings)

    tsne_predict = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    tsne_predict_result = tsne_predict.fit_transform(predict_embeddings)

    # Criar mapas de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Criar mapas de cores para as prediÃ§Ãµes
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converter labels para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Plot 1: Dados de Treinamento
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
        # IDs de proteÃ­nas reais adicionados ao 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(train_protein_ids, train_labels)],
        hoverinfo='text',
        name='Dados de Treinamento'
    ))
    fig_train.update_layout(
        title='t-SNE 3D: Dados de Treinamento',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Plot 2: PrediÃ§Ãµes
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
        # IDs de proteÃ­nas adicionados ao 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='PrediÃ§Ãµes'
    ))
    fig_predict.update_layout(
        title='t-SNE 3D: PrediÃ§Ãµes',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Salvar os plots como arquivos HTML
    tsne_train_html = os.path.join(output_dir, "tsne_train_3d.html")
    tsne_predict_html = os.path.join(output_dir, "tsne_predict_3d.html")
    
    pio.write_html(fig_train, file=tsne_train_html, auto_open=False)
    pio.write_html(fig_predict, file=tsne_predict_html, auto_open=False)
    
    logging.info(f"Plot t-SNE de Treinamento salvo como {tsne_train_html}")
    logging.info(f"Plot t-SNE de PrediÃ§Ãµes salvo como {tsne_predict_html}")

    return fig_train, fig_predict

# ============================================
# Fim das FunÃ§Ãµes Adicionais para t-SNE
# ============================================

# ============================================
# FunÃ§Ãµes de Processamento de Resultados e Interface
# ============================================

def get_base64_image(image_path: str) -> str:
    """
    Codifica um arquivo de imagem para uma string base64.

    ParÃ¢metros:
    - image_path (str): Caminho para o arquivo de imagem.

    Retorna:
    - str: String base64 da imagem.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Imagem nÃ£o encontrada em {image_path}.")
        return ""

# ============================================
# Fim das FunÃ§Ãµes de Processamento de Resultados
# ============================================

# ============================================
# ExecuÃ§Ã£o da Interface do Streamlit
# ============================================

# Barra lateral para parÃ¢metros de entrada
st.sidebar.header("ParÃ¢metros de Entrada")

# FunÃ§Ã£o para salvar arquivos enviados
def save_uploaded_file(uploaded_file, save_path: str) -> str:
    """
    Salva um arquivo enviado pelo usuÃ¡rio.
    
    ParÃ¢metros:
    - uploaded_file: Arquivo enviado pelo usuÃ¡rio.
    - save_path (str): Caminho para salvar o arquivo.
    
    Retorna:
    - str: Caminho para o arquivo salvo.
    """
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# OpÃ§Ãµes de entrada
use_default_train = st.sidebar.checkbox("Usar Dados de Treinamento PadrÃ£o", value=True)
if not use_default_train:
    train_fasta_file = st.sidebar.file_uploader("Upload de Arquivo FASTA de Treinamento", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("Upload de Arquivo de Tabela de Treinamento (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Upload de Arquivo FASTA para PrediÃ§Ã£o", type=["fasta", "fa", "fna"])

kmer_size = st.sidebar.number_input("Tamanho do K-mer", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Tamanho do Passo", min_value=1, max_value=10, value=1, step=1)

aggregation_method = st.sidebar.selectbox(
    "MÃ©todo de AgregaÃ§Ã£o",
    options=['none', 'mean'],  # Apenas 'none' e 'mean' sÃ£o opÃ§Ãµes
    index=0
)

# ParÃ¢metros opcionais do Word2Vec
st.sidebar.header("Personalizar ParÃ¢metros do Word2Vec")
custom_word2vec = st.sidebar.checkbox("Personalizar ParÃ¢metros do Word2Vec", value=False)
if custom_word2vec:
    window = st.sidebar.number_input(
        "Tamanho da Janela", min_value=5, max_value=20, value=10, step=5
    )
    workers = st.sidebar.number_input(
        "Workers", min_value=1, max_value=112, value=8, step=8
    )
    epochs = st.sidebar.number_input(
        "Epochs", min_value=1, max_value=2500, value=2500, step=100
    )
else:
    window = 10  # Valor padrÃ£o
    workers = 8  # Valor padrÃ£o
    epochs = 2500  # Valor padrÃ£o

# DiretÃ³rio de saÃ­da baseado no mÃ©todo de agregaÃ§Ã£o
model_dir = create_unique_model_directory("results", aggregation_method)
output_dir = model_dir

# BotÃ£o para iniciar o processamento
if st.sidebar.button("Executar AnÃ¡lise"):
    # Caminhos para dados internos
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"

    # Tratamento dos dados de treinamento
    if use_default_train:
        train_fasta_path = internal_train_fasta
        train_table_path = internal_train_table
        st.markdown("<span style='color:white'>Usando dados de treinamento padrÃ£o.</span>", unsafe_allow_html=True)

    else:
        if train_fasta_file is not None and train_table_file is not None:
            train_fasta_path = os.path.join(output_dir, "uploaded_train.fasta")
            train_table_path = os.path.join(output_dir, "uploaded_train_table.tsv")
            save_uploaded_file(train_fasta_file, train_fasta_path)
            save_uploaded_file(train_table_file, train_table_path)
            st.markdown("<span style='color:white'>Dados de treinamento enviados serÃ£o usados.</span>", unsafe_allow_html=True)

        else:
            st.error("Por favor, faÃ§a o upload tanto do arquivo FASTA de treinamento quanto do arquivo de tabela TSV.")
            st.stop()

    # Tratamento dos dados de prediÃ§Ã£o
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(output_dir, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Por favor, faÃ§a o upload de um arquivo FASTA para prediÃ§Ã£o.")
        st.stop()
        
    # ParÃ¢metros restantes
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
        scaler="scaler_associated.pkl",  # Nome correto do scaler
        model_dir=model_dir,
        scatterplot_output=os.path.join(output_dir, "scatterplot_predictions.png"),
    )

    # Criar diretÃ³rio de modelo se nÃ£o existir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Executar a funÃ§Ã£o principal de anÃ¡lise
    st.markdown("<span style='color:white'>Processando dados e executando anÃ¡lise...</span>", unsafe_allow_html=True)
    try:
        main(args)

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")
        logging.error(f"Ocorreu um erro: {e}")

# ============================================
# FunÃ§Ãµes para Processamento de Resultados e Interface
# ============================================

def plot_predictions_scatterplot_custom(
    results: dict, 
    output_path: str, 
    top_n: int = 1
) -> None:
    """
    Gera um scatter plot mostrando apenas a categoria principal com a maior soma de probabilidades para cada proteÃ­na.
    
    Eixo Y: ID de acesso da proteÃ­na
    Eixo X: Specificidades de C4 a C18 (escala fixa)
    Cada ponto representa a especificidade correspondente para a proteÃ­na.
    Apenas a categoria principal (top 1) Ã© plotada por proteÃ­na.
    Pontos sÃ£o coloridos em uma Ãºnica cor uniforme, estilizados para publicaÃ§Ã£o cientÃ­fica.
    
    ParÃ¢metros:
    - results (dict): DicionÃ¡rio contendo prediÃ§Ãµes e rankings para proteÃ­nas.
    - output_path (str): Caminho para salvar o scatter plot.
    - top_n (int): NÃºmero de top categorias principais a plotar (padrÃ£o Ã© 1).
    """
    # Preparar dados
    protein_specificities = {}
    
    for seq_id, info in results.items():
        for model_type in ['random_forest', 'lightgbm', 'xgboost', 'catboost']:
            associated_rankings = info.get(f'{model_type}_ranking', [])
            if not associated_rankings:
                logging.warning(f"Sem dados de ranking associados para a proteÃ­na {seq_id} com {model_type}. Pulando...")
                continue

            # Usar a funÃ§Ã£o format_and_sum_probabilities para obter a categoria principal
            top_category_with_confidence, confidence, top_two_categories = format_and_sum_probabilities(associated_rankings)
            if top_category_with_confidence is None:
                logging.warning(f"Sem formataÃ§Ã£o vÃ¡lida para a proteÃ­na {seq_id} com {model_type}. Pulando...")
                continue

            # Extrair a categoria sem confianÃ§a
            category = top_category_with_confidence.split(" (")[0]
            confidence = confidence  # Soma das probabilidades para a categoria principal

            protein_specificities[f"{seq_id}_{model_type}"] = {
                'top_category': category,
                'confidence': confidence
            }

    if not protein_specificities:
        logging.warning("Nenhum dado disponÃ­vel para plotar o scatter plot.")
        return

    # Ordenar IDs de proteÃ­nas para melhor visualizaÃ§Ã£o
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Criar a figura
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))  # Ajustar altura baseada no nÃºmero de proteÃ­nas

    # Escala fixa para o Eixo X de C4 a C18
    x_values = list(range(4, 19))

    # Plotar pontos para todas as proteÃ­nas com sua categoria principal
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        confidence = data['confidence']

        # Extrair specificidades da string da categoria
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]

        for spec in specificities:
            ax.scatter(
                spec, y,
                color='#1f78b4',  # Cor uniforme
                edgecolors='black',
                linewidth=0.5,
                s=100,
                label='_nolegend_'  # Evitar duplicaÃ§Ã£o na legenda
            )

        # Conectar pontos com linhas se houver mÃºltiplas specificidades
        if len(specificities) > 1:
            ax.plot(
                specificities,
                [y] * len(specificities),
                color='#1f78b4',
                linestyle='-',
                linewidth=1.0,
                alpha=0.7
            )

    # Personalizar o plot para melhor qualidade de publicaÃ§Ã£o cientÃ­fica
    ax.set_xlabel('Specificity (C4 to C18)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Proteins', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Scatter Plot of Predictions for New Sequences (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='white')

    # Definir escala fixa e formataÃ§Ã£o para o Eixo X
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='white')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='white')

    # Definir grade e remover spines desnecessÃ¡rios para um visual limpo
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Ticks menores no Eixo X para melhor visibilidade
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Ajustar layout para evitar corte de labels
    plt.tight_layout()

    # Salvar a figura em alta qualidade para publicaÃ§Ã£o
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=600, bbox_inches='tight')  # Combina com a cor de fundo
    plt.close()
    logging.info(f"Scatter plot salvo em {output_path}")

# ============================================
# Fim das FunÃ§Ãµes de Processamento de Resultados e Interface
# ============================================


