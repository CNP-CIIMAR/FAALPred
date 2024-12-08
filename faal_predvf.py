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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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

# ============================================
# Definições de Funções e Classes
# ============================================

# Fixando sementes para reprodutibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,  # Alterar para DEBUG para mais verbosidade
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),  # Log para arquivo para registros persistentes
    ],
)

# ============================================
# Configuração e Interface do Streamlit
# ============================================

# Garantir que st.set_page_config seja o primeiro comando do Streamlit
st.set_page_config(
    page_title="FAAL_Pred",
    page_icon="🔬",  # Símbolo de DNA
    layout="wide",
    initial_sidebar_state="expanded",
)

def are_sequences_aligned(fasta_file: str) -> bool:
    """
    Verifica se todas as sequências em um arquivo FASTA têm o mesmo comprimento.
    
    Parâmetros:
    - fasta_file (str): Caminho para o arquivo FASTA.
    
    Retorna:
    - bool: True se todas as sequências estiverem alinhadas, False caso contrário.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Retorna True se todas as sequências tiverem o mesmo comprimento

def create_unique_model_directory(base_dir: str, aggregation_method: str) -> str:
    """
    Cria um diretório único para modelos baseado no método de agregação.
    
    Parâmetros:
    - base_dir (str): Diretório base para modelos.
    - aggregation_method (str): Método de agregação utilizado.
    
    Retorna:
    - str: Caminho para o diretório único do modelo.
    """
    model_dir = os.path.join(base_dir, f"models_{aggregation_method}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def realign_sequences_with_mafft(input_path: str, output_path: str, threads: int = 8) -> None:
    """
    Realinha sequências usando MAFFT.
    
    Parâmetros:
    - input_path (str): Caminho para o arquivo de entrada.
    - output_path (str): Caminho para salvar o arquivo realinhado.
    - threads (int): Número de threads para MAFFT.
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Sequências realinhadas salvas em {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar MAFFT: {e.stderr.decode()}")
        sys.exit(1)

from sklearn.cluster import DBSCAN, KMeans

def perform_clustering(data: np.ndarray, method: str = "DBSCAN", eps: float = 0.5, min_samples: int = 5, n_clusters: int = 3) -> np.ndarray:
    """
    Realiza clustering nos dados usando DBSCAN ou K-Means.
    
    Parâmetros:
    - data (np.ndarray): Dados para clustering.
    - method (str): Método de clustering ("DBSCAN" ou "K-Means").
    - eps (float): Parâmetro epsilon para DBSCAN.
    - min_samples (int): Número mínimo de amostras para DBSCAN.
    - n_clusters (int): Número de clusters para K-Means.
    
    Retorna:
    - np.ndarray: Labels gerados pelo método de clustering.
    """
    if method == "DBSCAN":
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "K-Means":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Método de clustering inválido: {method}")

    labels = clustering_model.fit_predict(data)
    return labels

def plot_roc_curve_global(y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
    """
    Plota a curva ROC para classificações binárias ou multiclasse.
    
    Parâmetros:
    - y_true (np.ndarray): Rótulos verdadeiros.
    - y_pred_proba (np.ndarray): Probabilidades preditas.
    - title (str): Título do gráfico.
    - save_as (str): Caminho para salvar o gráfico.
    - classes (list): Lista de classes (para multiclasse).
    """
    lw = 2  # Largura da linha

    # Verifica se é classificação binária ou multiclasse
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # Classificação binária
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Curva ROC (área = %0.2f)' % roc_auc)
    else:  # Classificação multiclasse
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
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'Curva ROC da classe {class_label} (área = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', color='white')
    plt.ylabel('Taxa de Verdadeiros Positivos', color='white')
    plt.title(title, color='white')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')  # Combinar com a cor de fundo
    plt.close()

def get_class_rankings_global(model: RandomForestClassifier, X: np.ndarray) -> list:
    """
    Obtém as classificações das classes com base nas probabilidades preditas pelo modelo.
    
    Parâmetros:
    - model (RandomForestClassifier): Modelo treinado.
    - X (np.ndarray): Dados para obter previsões.
    
    Retorna:
    - list: Lista de classificações formatadas para cada amostra.
    """
    if model is None:
        raise ValueError("Modelo não treinado. Por favor, treine o modelo primeiro.")

    # Obtendo probabilidades para cada classe
    y_pred_proba = model.predict_proba(X)

    # Classificando classes com base nas probabilidades
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings

def calculate_roc_values(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Calcula os valores ROC AUC para cada classe.
    
    Parâmetros:
    - model (RandomForestClassifier): Modelo treinado.
    - X_test (np.ndarray): Dados de teste.
    - y_test (np.ndarray): Rótulos verdadeiros de teste.
    
    Retorna:
    - pd.DataFrame: DataFrame contendo as pontuações ROC AUC por classe.
    """
    n_classes = len(np.unique(y_test))
    y_pred_proba = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Logging dos valores ROC
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
    Visualiza o espaço latente usando UMAP 3D com medidas de similaridade entre amostras originais e sintéticas.
    
    Parâmetros:
    - X_original (np.ndarray): Embeddings originais.
    - X_synthetic (np.ndarray): Embeddings sintéticos.
    - y_original (np.ndarray): Rótulos originais.
    - y_synthetic (np.ndarray): Rótulos sintéticos.
    - protein_ids_original (list): IDs das proteínas originais.
    - protein_ids_synthetic (list): IDs das proteínas sintéticas.
    - var_assoc_original (list): Variáveis associadas originais.
    - var_assoc_synthetic (list): Variáveis associadas sintéticas.
    - output_dir (str): Diretório para salvar o gráfico.
    
    Retorna:
    - Figure: Objeto de figura Plotly.
    """
    # Combina dados para UMAP
    X_combined = np.vstack([X_original, X_synthetic])
    y_combined = np.hstack([y_original, y_synthetic])

    # Aplica UMAP para redução de dimensionalidade
    umap_reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    X_transformed = umap_reducer.fit_transform(X_combined)

    # Divide dados transformados
    X_transformed_original = X_transformed[:len(X_original)]
    X_transformed_synthetic = X_transformed[len(X_original):]

    # Calcula similaridades entre sintéticas e originais
    similarities = cosine_similarity(X_synthetic, X_original)
    max_similarities = similarities.max(axis=1)  # Similaridade máxima para cada sintética
    closest_indices = similarities.argmax(axis=1)  # Índices das originais mais próximas

    # Cria DataFrames para facilitar
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

    # Cria o gráfico interativo 3D com Plotly
    fig = go.Figure()

    # Adiciona pontos para as amostras originais
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

    # Adiciona pontos para as amostras sintéticas
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

    # Ajusta layout
    fig.update_layout(
        title="Visualização do Espaço Latente com Similaridade (UMAP 3D)",
        scene=dict(
            xaxis_title="UMAP Dimensão 1",
            yaxis_title="UMAP Dimensão 2",
            zaxis_title="UMAP Dimensão 3"
        ),
        legend=dict(orientation="h", y=-0.1),
        template="plotly_dark"
    )

    # Salva o gráfico, se o diretório de saída for fornecido
    if output_dir:
        umap_similarity_path = os.path.join(output_dir, "umap_similarity_3D.html")
        fig.write_html(umap_similarity_path)
        logging.info(f"Gráfico UMAP salvo em {umap_similarity_path}")

    return fig

def format_and_sum_probabilities(associated_rankings: list) -> tuple:
    """
    Formata e soma probabilidades para cada categoria, retornando apenas a categoria principal.
    
    Parâmetros:
    - associated_rankings (list): Lista de rankings associados.
    
    Retorna:
    - tuple: (categoria principal com confiança, soma das probabilidades, top duas categorias)
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

    # Inicializa o dicionário de somas
    for category in categories:
        category_sums[category] = 0.0

    # Soma probabilidades para cada categoria
    for rank in associated_rankings:
        try:
            prob = float(rank.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Erro ao processar a string de ranking: {rank}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob

    if not category_sums:
        return None, None, None  # Sem dados válidos

    # Determina a categoria principal com base na soma das probabilidades
    top_category, top_sum = max(category_sums.items(), key=lambda x: x[1])

    # Encontra as duas categorias principais
    sorted_categories = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)
    top_two = sorted_categories[:2] if len(sorted_categories) >=2 else sorted_categories

    # Extrai as duas categorias principais e suas probabilidades
    top_two_categories = [f"{cat} ({prob:.2f}%)" for cat, prob in top_two]

    # Encontra a categoria principal com confiança
    top_category_with_confidence = f"{top_category} (1)"

    return top_category_with_confidence, top_sum, top_two_categories

class Support:
    """
    Classe de suporte para treinar e avaliar modelos Random Forest com técnicas de oversampling.
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

        self.init_params = {
            "n_estimators": 500,
            "max_depth": 15,  # Reduzido para prevenir overfitting
            "min_samples_split": 5,  # Aumentado para prevenir overfitting
            "min_samples_leaf": 2,
            "criterion": "gini",
            "max_features": "sqrt",  # Alterado de 'sqrt' para 'log2'
            "class_weight": "balanced_subsample",  # Balanceamento automático das classes
            "max_leaf_nodes": None,  # Ajustado para maior regularização
            "min_impurity_decrease": 0.01,
            "bootstrap": True,
            "ccp_alpha": 0.001,
            "random_state": self.seed  # Adicionado para RandomForest
        }
            

        self.parameters = {
            "n_estimators": [50, 100, 300, 700, 1000],
            "max_depth": [2, 5, 10, 15,20,30],
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


    def _oversample_single_sample_classes(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        protein_ids: list = None, 
        var_assoc: list = None
    ) -> tuple:
        """
        Customiza o oversampling para garantir que todas as classes tenham pelo menos 'self.cv + 1' amostras.
        Também captura informações de dados sintéticos como IDs de proteínas e variáveis associadas.
    
        Parâmetros:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        - protein_ids (list): Lista de IDs de proteínas originais.
        - var_assoc (list): Lista de variáveis associadas originais.
    
        Retorna:
        - tuple: (X após oversampling, y após oversampling, IDs de proteínas sintéticas, variáveis associadas sintéticas)
        """
        logging.info("Iniciando processo de oversampling...")

        # Contar a distribuição inicial das classes
        counter = Counter(y)
        logging.info(f"Distribuição inicial das classes: {counter}")

        # Definir a estratégia de oversampling para garantir pelo menos 'self.cv + 1' amostras por classe
        classes_to_oversample = {cls: max(self.cv + 1, count) for cls, count in counter.items()}
        logging.info(f"Strategia de oversampling para RandomOverSampler: {classes_to_oversample}")

        try:
            # Aplicar RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=classes_to_oversample, random_state=self.seed)
            X_ros, y_ros = ros.fit_resample(X, y)
            logging.info(f"Distribuição das classes após RandomOverSampler: {Counter(y_ros)}")
        except ValueError as e:
            logging.error(f"Erro durante RandomOverSampler: {e}")
            sys.exit(1)

        # Capturar dados sintéticos do RandomOverSampler
        synthetic_protein_ids = []
        synthetic_var_assoc = []
        if protein_ids and var_assoc:
            for idx in range(len(X), len(X_ros)):
                synthetic_protein_ids.append(f"synthetic_ros_{idx}")
                synthetic_var_assoc.append(var_assoc[idx % len(var_assoc)])

        try:
            # Aplicar SMOTE para equilibrar ainda mais as classes
            smote = SMOTE(random_state=self.seed)
            X_smote, y_smote = smote.fit_resample(X_ros, y_ros)
            logging.info(f"Distribuição das classes após SMOTE: {Counter(y_smote)}")
        except ValueError as e:
            logging.error(f"Erro durante SMOTE: {e}")
            sys.exit(1)

        # Capturar dados sintéticos do SMOTE
        if protein_ids and var_assoc:
            for idx in range(len(X_ros), len(X_smote)):
                synthetic_protein_ids.append(f"synthetic_smote_{idx}")
                synthetic_var_assoc.append(var_assoc[idx % len(var_assoc)])

        # Salvar contagem das classes no arquivo
        with open("oversampling_counts.txt", "a") as f:
            f.write("Distribuição das Classes após Oversampling:\n")
            for cls, count in Counter(y_smote).items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote, synthetic_protein_ids, synthetic_var_assoc


    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        protein_ids: list = None, 
        var_assoc: list = None, 
        model_name_prefix: str = 'model', 
        model_dir: str = None, 
        min_kmers: int = None
    ) -> RandomForestClassifier:
        """
        Treina o modelo com oversampling e validação cruzada.
        
        Parâmetros:
        - X (np.ndarray): Features.
        - y (np.ndarray): Labels.
        - protein_ids (list): IDs das proteínas.
        - var_assoc (list): Variáveis associadas.
        - model_name_prefix (str): Prefixo para salvar o modelo.
        - model_dir (str): Diretório para salvar o modelo.
        - min_kmers (int): Número mínimo de k-mers.
        
        Retorna:
        - RandomForestClassifier: Modelo treinado e calibrado.
        """
        logging.info(f"Iniciando método fit para {model_name_prefix}...")

        # Transformar em arrays numpy
        X = np.array(X)
        y = np.array(y)

        # Determinar min_kmers
        if min_kmers is not None:
            logging.info(f"Usando min_kmers fornecido: {min_kmers}")
        else:
            min_kmers = len(X)
            logging.info(f"min_kmers não fornecido. Definindo como o tamanho de X: {min_kmers}")

        # Oversampling inicial
        X_smote, y_smote, synthetic_protein_ids, synthetic_var_assoc = self._oversample_single_sample_classes(
            X, y, protein_ids, var_assoc
        )

        # Combinar dados originais e sintéticos
        if protein_ids and var_assoc:
            combined_protein_ids = protein_ids + synthetic_protein_ids
            combined_var_assoc = var_assoc + synthetic_var_assoc
        else:
            combined_protein_ids = None
            combined_var_assoc = None

        # Visualização do espaço latente após o oversampling
        if protein_ids and var_assoc:
            save_path = f"umap_similarity_visualization_{model_name_prefix}.html"
            logging.info("Visualizando o espaço latente com medidas de similaridade...")
            fig = visualize_latent_space_with_similarity(
                X_original=X, 
                X_synthetic=X_smote[len(y):],  # Ajuste para y
                y_original=y, 
                y_synthetic=y_smote[len(y):],
                protein_ids_original=protein_ids,
                protein_ids_synthetic=synthetic_protein_ids,
                var_assoc_original=var_assoc,
                var_assoc_synthetic=synthetic_var_assoc,
                output_dir=model_dir
            )
            logging.info("Visualização UMAP gerada com sucesso.")

        # Cross-validation com StratifiedKFold
        self.train_scores = []
        self.test_scores = []
        self.f1_scores = []
        self.pr_auc_scores = []

        # Verificar se todas as classes têm pelo menos self.cv +1 amostras
        class_counts = Counter(y_smote)
        min_class_count = min(class_counts.values())
        adjusted_n_splits = min(self.cv, min_class_count - 1)  # Porque SMOTE precisa n_samples > n_neighbors
        if adjusted_n_splits < self.cv:
            logging.warning(f"Ajustando n_splits de {self.cv} para {adjusted_n_splits} devido a restrições de tamanho das classes.")
            skf = StratifiedKFold(n_splits=adjusted_n_splits, random_state=self.seed, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        for fold_number, (train_index, test_index) in enumerate(skf.split(X_smote, y_smote), start=1):
            X_train, X_test = X_smote[train_index], X_smote[test_index]
            y_train, y_test = y_smote[train_index], y_smote[test_index]

            # Verificar a distribuição das classes no conjunto de teste
            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Distribuição das classes no conjunto de teste: {fold_class_distribution}")

            # Oversampling para treinamento
            X_train_resampled, y_train_resampled, synthetic_protein_ids_train, synthetic_var_assoc_train = self._oversample_single_sample_classes(
                X_train, 
                y_train, 
                protein_ids=protein_ids if protein_ids else None, 
                var_assoc=var_assoc if var_assoc else None
            )

            # Verificar a distribuição das classes após o oversampling no treinamento
            train_sample_counts = Counter(y_train_resampled)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Distribuição das classes no conjunto de treinamento após oversampling: {train_sample_counts}")

            with open("training_sample_counts_after_oversampling.txt", "a") as f:
                f.write(f"Fold {fold_number} Contagem de amostras no treinamento após oversampling para {model_name_prefix}:\n")
                for cls, count in train_sample_counts.items():
                    f.write(f"{cls}: {count}\n")

            # Treinamento do modelo
            self.model = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs)
            self.model.fit(X_train_resampled, y_train_resampled)

            # Avaliação
            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)
            y_pred = self.model.predict(X_test)

            # Métricas
            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

            # Precision-Recall AUC
            if len(np.unique(y_test)) > 1:
                pr_auc = average_precision_score(y_test, self.model.predict_proba(X_test), average='macro')
            else:
                pr_auc = 0.0  # Não pode calcular PR AUC para uma única classe
            self.pr_auc_scores.append(pr_auc)

            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Score de Treinamento: {train_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Score de Teste: {test_score}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Score F1: {f1}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Precision-Recall AUC: {pr_auc}")

            # Calcular ROC AUC
            try:
                if len(np.unique(y_test)) == 2:
                    fpr, tpr, thresholds = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
                    roc_auc_score_value = auc(fpr, tpr)
                    self.roc_results.append((fpr, tpr, roc_auc_score_value))
                else:
                    y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                    roc_auc_score_value = roc_auc_score(y_test_bin, self.model.predict_proba(X_test), multi_class='ovo', average='macro')
                    self.roc_results.append(roc_auc_score_value)
            except ValueError:
                logging.warning(f"Não foi possível calcular ROC AUC para o fold {fold_number} [{model_name_prefix}] devido a representação insuficiente das classes.")

            # Realizar Grid Search e salvar o melhor modelo
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            self.best_params = best_params

            if model_dir:
                best_model_filename = os.path.join(model_dir, f'modelo_melhor_{model_name_prefix}.pkl')
                # Garantir que o diretório exista
                os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Melhor modelo salvo como {best_model_filename} para {model_name_prefix}")
            else:
                best_model_filename = f'modelo_melhor_{model_name_prefix}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Melhor modelo salvo como {best_model_filename} para {model_name_prefix}")

            if best_params is not None:
                self.best_params = best_params
                logging.info(f"Melhores parâmetros para {model_name_prefix}: {self.best_params}")
            else:
                logging.warning(f"Nenhum melhor parâmetro encontrado na busca em grade para {model_name_prefix}.")

            # Integrar Calibração de Probabilidades
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=5, n_jobs=self.n_jobs)
            calibrator.fit(X_train_resampled, y_train_resampled)

            self.model = calibrator

            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'modelo_calibrado_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'modelo_calibrado_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Modelo calibrado salvo como {calibrated_model_filename} para {model_name_prefix}")

            fold_number += 1

            # Permitir que o Streamlit atualize a UI
            time.sleep(0.1)

        return self.model

    def _perform_grid_search(self, X_train_resampled: np.ndarray, y_train_resampled: np.ndarray) -> tuple:
        """
        Realiza Grid Search para encontrar os melhores hiperparâmetros.
        
        Parâmetros:
        - X_train_resampled (np.ndarray): Features de treinamento após oversampling.
        - y_train_resampled (np.ndarray): Labels de treinamento após oversampling.
        
        Retorna:
        - tuple: (Melhor modelo, Melhores parâmetros)
        """
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
        logging.info(f"Melhores parâmetros da busca em grade: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def get_best_param(self, param_name: str, default = None):
        """
        Obtém o melhor parâmetro encontrado na busca em grade.
        
        Parâmetros:
        - param_name (str): Nome do parâmetro.
        - default: Valor padrão caso o parâmetro não seja encontrado.
        
        Retorna:
        - Valor do parâmetro ou default.
        """
        return self.best_params.get(param_name, default)

    def plot_learning_curve(self, output_path: str) -> None:
        """
        Plota a curva de aprendizado do modelo.
        
        Parâmetros:
        - output_path (str): Caminho para salvar o gráfico.
        """
        plt.figure()
        plt.plot(self.train_scores, label='Score de Treinamento')
        plt.plot(self.test_scores, label='Score de Validação Cruzada')
        plt.plot(self.f1_scores, label='Score F1')
        plt.plot(self.pr_auc_scores, label='Precision-Recall AUC')
        plt.title("Curva de Aprendizado", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Score", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')  # Combinar com a cor de fundo
        plt.close()

    def get_class_rankings(self, X: np.ndarray) -> list:
        """
        Obtém as classificações das classes para os dados fornecidos.
        
        Parâmetros:
        - X (np.ndarray): Dados para obter previsões.
        
        Retorna:
        - list: Lista de classificações formatadas para cada amostra.
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Por favor, treine o modelo primeiro.")

        # Obtendo probabilidades para cada classe
        y_pred_proba = self.model.predict_proba(X)

        # Classificando classes com base nas probabilidades
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def test_best_RF(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        scaler_dir: str = '.'
    ) -> tuple:
        """
        Testa o melhor modelo Random Forest com os dados fornecidos.
        
        Parâmetros:
        - X (np.ndarray): Features para teste.
        - y (np.ndarray): Labels verdadeiros para teste.
        - scaler_dir (str): Diretório onde o scaler está salvo.
        
        Retorna:
        - tuple: (Score, F1 Score, Precision-Recall AUC, Melhores parâmetros, Modelo calibrado, X_test, y_test)
        """
        # Carregar o scaler
        scaler_path = os.path.join(scaler_dir, 'scaler.pkl') if scaler_dir else 'scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler carregado de {scaler_path}")
        else:
            logging.error(f"Scaler não encontrado em {scaler_path}")
            sys.exit(1)

        X_scaled = scaler.transform(X)

        # Aplicar oversampling no conjunto inteiro antes da divisão
        X_resampled, y_resampled, _, _ = self._oversample_single_sample_classes(X_scaled, y)

        # Dividir em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        # Treinar RandomForestClassifier com os melhores parâmetros
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
        model.fit(X_train, y_train)  # Fit do modelo nos dados de treinamento

        # Integrar Calibração no Modelo de Teste
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        # Fazer previsões
        y_pred = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred, method='normalize')

        # Calcular o score (por exemplo, AUC)
        score = self._calculate_score(y_pred_adjusted, y_test)

        # Calcular métricas adicionais
        y_pred_classes = calibrated_model.predict(X_test)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(y_test, y_pred_adjusted, average='macro')
        else:
            pr_auc = 0.0  # Não pode calcular PR AUC para uma única classe

        # Retornar o score, melhores parâmetros, modelo treinado e conjuntos de teste
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calcula o score (por exemplo, ROC AUC) com base nas previsões e rótulos reais.
        
        Parâmetros:
        - y_pred (np.ndarray): Probabilidades preditas ajustadas.
        - y_test (np.ndarray): Rótulos verdadeiros.
        
        Retorna:
        - float: Valor do score.
        """
        n_classes = len(np.unique(y_test))
        if y_pred.ndim == 1 or n_classes == 2:
            return roc_auc_score(y_test, y_pred)
        elif y_pred.ndim == 2 and n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            return roc_auc_score(y_test_bin, y_pred, multi_class='ovo', average='macro')
        else:
            logging.warning(f"Forma inesperada ou número de classes: forma de y_pred: {y_pred.shape}, número de classes: {n_classes}")
            return 0

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, title: str, save_as: str = None, classes: list = None) -> None:
        """
        Plota a curva ROC para classificações binárias ou multiclasse.
        
        Parâmetros:
        - y_true (np.ndarray): Rótulos verdadeiros.
        - y_pred_proba (np.ndarray): Probabilidades preditas.
        - title (str): Título do gráfico.
        - save_as (str): Caminho para salvar o gráfico.
        - classes (list): Lista de classes (para multiclasse).
        """
        plot_roc_curve_global(y_true, y_pred_proba, title, save_as, classes)

class ProteinEmbeddingGenerator:
    """
    Classe para gerar embeddings de proteínas usando Word2Vec.
    """

    def __init__(self, sequences_path: str, table_data: pd.DataFrame = None, aggregation_method: str = 'none'):
        """
        Inicializa o gerador de embeddings.
        
        Parâmetros:
        - sequences_path (str): Caminho para o arquivo de sequências.
        - table_data (pd.DataFrame): Dados da tabela associada.
        - aggregation_method (str): Método de agregação ('none' ou 'mean').
        """
        aligned_path = sequences_path
        if not are_sequences_aligned(sequences_path):
            realign_sequences_with_mafft(sequences_path, sequences_path.replace(".fasta", "_aligned.fasta"), threads=1)
            aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")
        else:
            logging.info(f"Sequências já estão alinhadas: {sequences_path}")

        self.alignment = AlignIO.read(aligned_path, 'fasta')
        self.table_data = table_data
        self.embeddings = []
        self.models = {}
        self.aggregation_method = aggregation_method  # Método de agregação: 'none' ou 'mean'
        self.min_kmers = None  # Para armazenar min_kmers

    def generate_embeddings(
        self, 
        k: int = 3, 
        step_size: int = 1, 
        word2vec_model_path: str = "word2vec_model.bin", 
        model_dir: str = None, 
        min_kmers: int = None, 
        save_min_kmers: bool = False
    ) -> None:
        """
        Gera embeddings para sequências de proteínas usando Word2Vec, padronizando o número de k-mers.
        
        Parâmetros:
        - k (int): Tamanho do k-mer.
        - step_size (int): Tamanho do passo para gerar k-mers.
        - word2vec_model_path (str): Nome do arquivo do modelo Word2Vec.
        - model_dir (str): Diretório para salvar o modelo Word2Vec.
        - min_kmers (int): Número mínimo de k-mers a ser usado.
        - save_min_kmers (bool): Se True, salva min_kmers em um arquivo.
        """
        # Define o caminho completo do modelo Word2Vec
        if model_dir:
            word2vec_model_full_path = os.path.join(model_dir, word2vec_model_path)
        else:
            word2vec_model_full_path = word2vec_model_path

        # Verifica se o modelo Word2Vec já existe
        if os.path.exists(word2vec_model_full_path):
            logging.info(f"Modelo Word2Vec encontrado em {word2vec_model_full_path}. Carregando o modelo.")
            model = Word2Vec.load(word2vec_model_full_path)
            self.models['global'] = model
        else:
            logging.info("Modelo Word2Vec não encontrado. Treinando um novo modelo.")
            # Inicialização de variáveis
            kmer_groups = {}
            all_kmers = []
            kmers_counts = []

            # Geração de k-mers
            for record in self.alignment:
                sequence = str(record.seq)
                seq_len = len(sequence)
                protein_accession_alignment = record.id.split()[0]

                # Se os dados da tabela não forem fornecidos, pula a correspondência
                if self.table_data is not None:
                    matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                    matching_info = self.table_data[matching_rows]

                    if matching_info.empty:
                        logging.warning(f"Nenhuma correspondência na tabela de dados para {protein_accession_alignment}")
                        continue  # Pula para a próxima iteração

                    target_variable = matching_info['Target variable'].values[0]  # Pode ser removido se não for necessário
                    associated_variable = matching_info['Associated variable'].values[0]
                else:
                    # Se não houver tabela, usa valores padrão ou None
                    target_variable = None  # Pode ser removido
                    associated_variable = None

                logging.info(f"Processando {protein_accession_alignment} com comprimento de sequência {seq_len}")

                if seq_len < k:
                    logging.warning(f"Sequência muito curta para {protein_accession_alignment}. Comprimento: {seq_len}")
                    continue

                # Geração de k-mers, permitindo k-mers com menos de k gaps
                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Permite k-mers com menos de k gaps

                if not kmers:
                    logging.warning(f"Nenhum k-mer válido para {protein_accession_alignment}")
                    continue

                all_kmers.append(kmers)  # Adiciona a lista de k-mers como uma sentença
                kmers_counts.append(len(kmers))  # Armazena a contagem de k-mers

                embedding_info = {
                    'protein_accession': protein_accession_alignment,
                    'target_variable': target_variable,  # Pode ser removido
                    'associated_variable': associated_variable,
                    'kmers': kmers  # Armazena os k-mers para uso posterior
                }
                kmer_groups[protein_accession_alignment] = embedding_info

            # Determina o número mínimo de k-mers
            if not kmers_counts:
                logging.error("Nenhum k-mer foi coletado. Verifique suas sequências e parâmetros de k-mer.")
                sys.exit(1)

            if min_kmers is not None:
                self.min_kmers = min_kmers
                logging.info(f"Usando min_kmers fornecido: {self.min_kmers}")
            else:
                self.min_kmers = min(kmers_counts)
                logging.info(f"Número mínimo de k-mers em qualquer sequência: {self.min_kmers}")

            # Salva min_kmers se necessário
            if save_min_kmers and model_dir:
                min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
                with open(min_kmers_path, 'w') as f:
                    f.write(str(self.min_kmers))
                logging.info(f"min_kmers salvo em {min_kmers_path}")

            # Treina o modelo Word2Vec usando todos os k-mers
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=125,  # Alterar para 100 se necessário
                window=10,
                min_count=1,
                workers=8,
                sg=1,
                hs=1,  # Hierarchical softmax habilitado
                negative=0,  # Negative sampling desabilitado
                epochs=2500,  # Número fixo de épocas para reprodutibilidade
                seed=SEED  # Semente fixa para reprodutibilidade
            )

            # Cria diretório para o modelo Word2Vec se necessário
            if model_dir:
                os.makedirs(os.path.dirname(word2vec_model_full_path), exist_ok=True)

            # Salva o modelo Word2Vec
            model.save(word2vec_model_full_path)
            self.models['global'] = model
            logging.info(f"Modelo Word2Vec salvo em {word2vec_model_full_path}")

        # Gera embeddings padronizados
        kmer_groups = {}
        kmers_counts = []
        all_kmers = []

        for record in self.alignment:
            sequence_id = record.id.split()[0]  # Usa IDs de sequência consistentes
            sequence = str(record.seq)

            # Se os dados da tabela não forem fornecidos, pula a correspondência
            if self.table_data is not None:
                matching_rows = self.table_data['Protein.accession'].str.split().str[0] == sequence_id
                matching_info = self.table_data[matching_rows]

                if matching_info.empty:
                    logging.warning(f"Nenhuma correspondência na tabela de dados para {sequence_id}")
                    continue  # Pula para a próxima iteração

                target_variable = matching_info['Target variable'].values[0]  # Pode ser removido
                associated_variable = matching_info['Associated variable'].values[0]
            else:
                # Se não houver tabela, usa valores padrão ou None
                target_variable = None  # Pode ser removido
                associated_variable = None

            kmers = [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, step_size)]
            kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Permite k-mers com menos de k gaps

            if not kmers:
                logging.warning(f"Nenhum k-mer válido para {sequence_id}")
                continue

            all_kmers.append(kmers)
            kmers_counts.append(len(kmers))

            embedding_info = {
                'protein_accession': sequence_id,
                'target_variable': target_variable,  # Pode ser removido
                'associated_variable': associated_variable,
                'kmers': kmers
            }
            kmer_groups[sequence_id] = embedding_info

        # Determina o número mínimo de k-mers
        if not kmers_counts:
            logging.error("Nenhum k-mer foi coletado. Verifique suas sequências e parâmetros de k-mer.")
            sys.exit(1)

        if min_kmers is not None:
            self.min_kmers = min_kmers
            logging.info(f"Usando min_kmers fornecido: {self.min_kmers}")
        else:
            self.min_kmers = min(kmers_counts)
            logging.info(f"Número mínimo de k-mers em qualquer sequência: {self.min_kmers}")

        # Gera embeddings padronizados
        for record in self.alignment:
            sequence_id = record.id.split()[0]  # Usa IDs de sequência consistentes
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
                    'target_variable': embedding_info.get('target_variable'),  # Pode ser removido
                    'associated_variable': embedding_info.get('associated_variable')
                })
                continue

            # Seleciona os primeiros min_kmers k-mers
            selected_kmers = kmers_for_protein[:self.min_kmers]

            # Preenche com zeros se necessário
            if len(selected_kmers) < self.min_kmers:
                padding = [np.zeros(self.models['global'].vector_size)] * (self.min_kmers - len(selected_kmers))
                selected_kmers.extend(padding)

            # Obtém embeddings dos k-mers selecionados
            selected_embeddings = [self.models['global'].wv[kmer] if kmer in self.models['global'].wv else np.zeros(self.models['global'].vector_size) for kmer in selected_kmers]

            if self.aggregation_method == 'none':
                # Concatena embeddings dos k-mers selecionados
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                # Agrega embeddings dos k-mers selecionados pela média
                embedding_concatenated = np.mean(selected_embeddings, axis=0)
            else:
                # Se o método não for reconhecido, usa concatenação como padrão
                logging.warning(f"Método de agregação desconhecido '{self.aggregation_method}'. Usando concatenação.")
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)

            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': embedding_concatenated,
                'target_variable': embedding_info.get('target_variable'),  # Pode ser removido
                'associated_variable': embedding_info.get('associated_variable')
            })

            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {embedding_concatenated.shape}")

        # Ajustar StandardScaler com embeddings para treinamento/predição
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])

        # Verifica se todos os embeddings têm a mesma forma
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Formas de embedding inconsistentes detectadas: {embedding_shapes}")
            raise ValueError("Embeddings têm formas inconsistentes.")
        else:
            logging.info(f"Todos os embeddings têm a forma: {embedding_shapes.pop()}")

        # Define o caminho completo do scaler
        scaler_full_path = os.path.join(model_dir, 'scaler.pkl') if model_dir else 'scaler.pkl'

        # Verifica se o scaler já existe
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler encontrado em {scaler_full_path}. Carregando o scaler.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler não encontrado. Treinando um novo scaler.")
            scaler = StandardScaler().fit(embeddings_array_train)
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler salvo em {scaler_full_path}")

    def get_embeddings_and_labels(self, label_type: str = 'associated_variable') -> tuple:
        """
        Retorna embeddings e rótulos associados (associated_variable).
        
        Parâmetros:
        - label_type (str): Tipo de rótulo ('associated_variable').
        
        Retorna:
        - tuple: (Embeddings, Rótulos)
        """
        embeddings = []
        labels = []

        for embedding_info in self.embeddings:
            embeddings.append(embedding_info['embedding'])
            labels.append(embedding_info[label_type])  # Usa o tipo de rótulo especificado

        return np.array(embeddings), np.array(labels)

def compute_perplexity(n_samples: int) -> int:
    """
    Calcula a perplexidade dinamicamente (não utilizado, pois t-SNE foi removido).
    
    Parâmetros:
    - n_samples (int): Número de amostras.
    
    Retorna:
    - int: Valor de perplexidade.
    """
    return max(5, min(50, n_samples // 100))

def plot_dual_umap(
    train_embeddings: np.ndarray, 
    train_labels: list,
    train_protein_ids: list,
    predict_embeddings: np.ndarray, 
    predict_labels: list, 
    predict_protein_ids: list, 
    output_dir: str
) -> tuple:
    """
    Plota dois gráficos UMAP 3D e os salva como arquivos HTML:
    - Gráfico 1: Dados de Treinamento.
    - Gráfico 2: Previsões.
    
    Parâmetros:
    - train_embeddings (np.ndarray): Embeddings de treinamento.
    - train_labels (list): Rótulos dos dados de treinamento.
    - train_protein_ids (list): IDs das proteínas de treinamento.
    - predict_embeddings (np.ndarray): Embeddings de previsão.
    - predict_labels (list): Rótulos das previsões.
    - predict_protein_ids (list): IDs das proteínas das previsões.
    - output_dir (str): Diretório para salvar os gráficos UMAP.
    
    Retorna:
    - tuple: (Figura de treinamento, Figura de previsões)
    """
    # Redução de dimensionalidade usando UMAP
    umap_train = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_train_result = umap_train.fit_transform(train_embeddings)

    umap_predict = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_predict_result = umap_predict.fit_transform(predict_embeddings)

    # Cria mapas de cores para os dados de treinamento
    unique_train_labels = sorted(list(set(train_labels)))
    color_map_train = px.colors.qualitative.Dark24
    color_dict_train = {label: color_map_train[i % len(color_map_train)] for i, label in enumerate(unique_train_labels)}

    # Cria mapas de cores para as previsões
    unique_predict_labels = sorted(list(set(predict_labels)))
    color_map_predict = px.colors.qualitative.Light24
    color_dict_predict = {label: color_map_predict[i % len(color_map_predict)] for i, label in enumerate(unique_predict_labels)}

    # Converte rótulos para cores
    train_colors = [color_dict_train.get(label, 'gray') for label in train_labels]
    predict_colors = [color_dict_predict.get(label, 'gray') for label in predict_labels]

    # Gráfico 1: Dados de Treinamento
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
        # IDs reais das proteínas adicionados ao campo 'text'
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

    # Gráfico 2: Previsões
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
        # IDs das proteínas adicionados ao campo 'text'
        text=[f"Protein ID: {protein_id}<br>Label: {label}" for protein_id, label in zip(predict_protein_ids, predict_labels)],
        hoverinfo='text',
        name='Previsões'
    ))
    fig_predict.update_layout(
        title='UMAP 3D: Previsões',
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        )
    )

    # Salva os gráficos como arquivos HTML
    umap_train_html = os.path.join(output_dir, "umap_train_3d.html")
    umap_predict_html = os.path.join(output_dir, "umap_predict_3d.html")
    
    pio.write_html(fig_train, file=umap_train_html, auto_open=False)
    pio.write_html(fig_predict, file=umap_predict_html, auto_open=False)
    
    logging.info(f"UMAP Treinamento salvo como {umap_train_html}")
    logging.info(f"UMAP Previsões salvo como {umap_predict_html}")

    return fig_train, fig_predict

def plot_predictions_scatterplot_custom(
    results: dict, 
    output_path: str, 
    top_n: int = 1
) -> None:
    """
    Gera um gráfico de dispersão mostrando apenas a categoria principal com a maior soma de probabilidades para cada proteína.
    
    Eixo Y: ID de acesso da proteína
    Eixo X: Specificidades de C4 a C18 (escala fixa)
    Cada ponto representa a specificidade correspondente para a proteína.
    Apenas a categoria principal (top 1) é plotada por proteína.
    Pontos são coloridos em uma única cor uniforme, estilizados para publicação científica.
    
    Parâmetros:
    - results (dict): Dicionário contendo previsões e classificações para proteínas.
    - output_path (str): Caminho para salvar o gráfico de dispersão.
    - top_n (int): Número de categorias principais a plotar (padrão é 1).
    """
    # Preparar dados
    protein_specificities = {}
    
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"Sem dados de ranking associado para a proteína {seq_id}. Pulando...")
            continue

        # Utiliza a função format_and_sum_probabilities para obter a categoria principal
        top_category_with_confidence, top_sum, top_two_categories = format_and_sum_probabilities(associated_rankings)
        if top_category_with_confidence is None:
            logging.warning(f"Sem dados válidos de categoria para a proteína {seq_id}. Pulando...")
            continue

        # Extrai a categoria sem a confiança
        category = top_category_with_confidence.split(" (")[0]
        confidence = top_sum  # Soma das probabilidades para a categoria principal

        protein_specificities[seq_id] = {
            'top_category': category,
            'confidence': confidence
        }

    if not protein_specificities:
        logging.warning("Nenhum dado disponível para plotar o gráfico de dispersão.")
        return

    # Ordena os IDs das proteínas para melhor visualização
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Cria a figura
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_proteins) * 0.5)))  # Ajusta a altura com base no número de proteínas

    # Escala fixa para o eixo X de C4 a C18
    x_values = list(range(4, 19))

    # Plota pontos para todas as proteínas com sua categoria principal
    for protein, data in protein_specificities.items():
        y = protein_order[protein]
        category = data['top_category']
        confidence = data['confidence']

        # Extrai specificidades da string da categoria
        specificities = [int(x[1:]) for x in category.split('-') if x.startswith('C')]

        for spec in specificities:
            ax.scatter(
                spec, y,
                color='#1f78b4',  # Cor uniforme
                edgecolors='black',
                linewidth=0.5,
                s=100,
                label='_nolegend_'  # Evita duplicação na legenda
            )

        # Conecta pontos com linhas se houver múltiplas specificidades
        if len(specificities) > 1:
            ax.plot(
                specificities,
                [y] * len(specificities),
                color='#1f78b4',
                linestyle='-',
                linewidth=1.0,
                alpha=0.7
            )

    # Personaliza o gráfico para melhor qualidade de publicação
    ax.set_xlabel('Specificidade (C4 a C18)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Proteínas', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Gráfico de Dispersão das Previsões de Novas Sequências (SS Prediction)', fontsize=16, fontweight='bold', pad=20, color='white')

    # Define escala fixa e formatação do eixo X
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12, color='white')
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10, color='white')

    # Define grade e remove spines desnecessários para um visual limpo
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Ticks menores no eixo X para melhor visibilidade
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Ajusta o layout para evitar corte de rótulos
    plt.tight_layout()

    # Salva a figura em alta qualidade para publicação
    plt.savefig(output_path, facecolor='#0B3C5D', dpi=600, bbox_inches='tight')  # Combina com a cor de fundo
    plt.close()
    logging.info(f"Gráfico de dispersão salvo em {output_path}")

def adjust_predictions_global(
    predicted_proba: np.ndarray, 
    method: str = 'normalize', 
    alpha: float = 1.0
) -> np.ndarray:
    """
    Ajusta as probabilidades preditas pelo modelo.
    
    Parâmetros:
    - predicted_proba (np.ndarray): Probabilidades preditas pelo modelo.
    - method (str): Método de ajuste ('normalize', 'smoothing', 'none').
    - alpha (float): Parâmetro para suavização (usado se method='smoothing').
    
    Retorna:
    - np.ndarray: Probabilidades ajustadas.
    """
    if method == 'normalize':
        # Normaliza as probabilidades para que somem 1 para cada amostra
        logging.info("Normalizando probabilidades preditas.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)

    elif method == 'smoothing':
        # Aplica suavização às probabilidades para evitar valores extremos
        logging.info(f"Aplicando suavização às probabilidades preditas com alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])

    elif method == 'none':
        # Não aplica nenhum ajuste
        logging.info("Nenhum ajuste aplicado às probabilidades preditas.")
        adjusted_proba = predicted_proba.copy()

    else:
        logging.warning(f"Método de ajuste desconhecido '{method}'. Nenhum ajuste será aplicado.")
        adjusted_proba = predicted_proba.copy()

    return adjusted_proba

def main(args: argparse.Namespace) -> None:
    """
    Função principal coordenando o fluxo de trabalho.
    
    Parâmetros:
    - args (argparse.Namespace): Argumentos de entrada.
    """
    model_dir = args.model_dir

    # Inicializar variáveis de progresso
    total_steps = 4  # Ajustado após remover partes de target_variable
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # =============================
    # PASSO 1: Treinamento do Modelo para associated_variable
    # =============================

    # Carregar dados de treinamento
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table

    # Verificar se as sequências de treinamento estão alinhadas
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Sequências de treinamento não estão alinhadas. Realinhando com MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)  # Threads fixadas em 1
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Arquivo de treinamento alinhado encontrado ou sequências já alinhadas: {train_alignment_path}")

    # Carregar dados da tabela de treinamento
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Tabela de dados de treinamento carregada com sucesso.")

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Inicializar e gerar embeddings para treinamento
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, 
        table_data=train_table_data, 
        aggregation_method=args.aggregation_method  # Passando o método de agregação ('none' ou 'mean')
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        save_min_kmers=True  # Salvar min_kmers após o treinamento
    )
    logging.info(f"Número de embeddings de treinamento gerados: {len(protein_embedding_train.embeddings)}")

    # Salvar min_kmers para garantir consistência
    min_kmers = protein_embedding_train.min_kmers

    # Obtendo IDs de proteínas e variáveis associadas do conjunto de treinamento
    protein_ids_associated = [entry['protein_accession'] for entry in protein_embedding_train.embeddings]
    var_assoc_associated = [entry['associated_variable'] for entry in protein_embedding_train.embeddings]

    logging.info(f"IDs de Proteínas para associated_variable extraídos: {len(protein_ids_associated)}")
    logging.info(f"Variáveis associadas para associated_variable extraídas: {len(var_assoc_associated)}")

    # Obter embeddings e rótulos para associated_variable
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"Forma de X_associated: {X_associated.shape}")

    # Criação do scaler para X_associated
    scaler_associated = StandardScaler().fit(X_associated)
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')
    joblib.dump(scaler_associated, scaler_associated_path)
    logging.info("Scaler para X_associated criado e salvo.")

    # Escalar os dados de X_associated
    X_associated_scaled = scaler_associated.transform(X_associated)    

    # Caminhos completos para os modelos de associated_variable
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Verificar se o modelo calibrado para associated_variable já existe
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Modelo Random Forest calibrado para associated_variable carregado de {calibrated_model_associated_full_path}")
    else:
        # Treinamento do modelo para associated_variable
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(
            X_associated_scaled, 
            y_associated, 
            protein_ids=None,  # IDs não são necessários para associated_variable
            var_assoc=None, 
            model_name_prefix='associated', 
            model_dir=model_dir, 
            min_kmers=min_kmers
        )

        logging.info("Treinamento e calibração para associated_variable concluídos.")

        # Plotar a curva de aprendizado
        logging.info("Plotando Curva de Aprendizado para associated_variable")
        learning_curve_associated_path = args.learning_curve_associated
        support_model_associated.plot_learning_curve(learning_curve_associated_path)

        # Salvar o modelo calibrado
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Modelo Random Forest calibrado para associated_variable salvo em {calibrated_model_associated_full_path}")

        # Testar o modelo
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(
            X_associated_scaled, 
            y_associated
        )

        logging.info(f"Melhor ROC AUC para associated_variable: {best_score_associated}")
        logging.info(f"Melhor F1 Score para associated_variable: {best_f1_associated}")
        logging.info(f"Melhor Precision-Recall AUC para associated_variable: {best_pr_auc_associated}")
        logging.info(f"Melhores Parâmetros: {best_params_associated}")

        for param, value in best_params_associated.items():
            logging.info(f"{param}: {value}")

        # Obter rankings das classes para associated_variable
        class_rankings_associated = support_model_associated.get_class_rankings(X_test_associated)
        logging.info("Top 3 class rankings for the first 5 samples in the associated_variable data:")
        for i in range(min(5, len(class_rankings_associated))):
            logging.info(f"Sample {i+1}: Rankings of classes - {class_rankings_associated[i][:3]}")  # Mostra os top 3 rankings

        # Acessando class_weight do dicionário best_params_associated
        class_weight = best_params_associated.get('class_weight', None)
        # Imprimindo resultados
        logging.info(f"Class weight used: {class_weight}")

        # Salvar o modelo treinado para associated_variable
        joblib.dump(best_model_associated, rf_model_associated_full_path)
        logging.info(f"Random Forest model for associated_variable saved at {rf_model_associated_full_path}")

        # Plotar curva ROC para associated_variable
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            unique_classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve_global(
            y_test_associated, 
            y_pred_proba_associated, 
            'Curva ROC para Associated Variable', 
            save_as=args.roc_curve_associated, 
            classes=unique_classes_associated
        )

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # PASSO 2: Classificando Novas Sequências para associated_variable
    # =============================

    # Carregar min_kmers
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers loaded: {min_kmers_loaded}")
    else:
        logging.error(f"min_kmers file not found at {min_kmers_path}. Please make sure the training was completed successfully.")
        sys.exit(1)

    # Carregar dados para previsão
    predict_alignment_path = args.predict_fasta

    # Verificar se as sequências para previsão estão alinhadas
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Sequences for prediction are not aligned. Realigning with MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Threads fixadas em 1
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Aligned file for prediction found or sequences already aligned:{predict_alignment_path}")

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Inicializar ProteinEmbedding para previsão, sem necessidade da tabela
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, 
        table_data=None,
        aggregation_method=args.aggregation_method  # Passando o método de agregação ('none' ou 'mean')
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded  # Usa o mesmo min_kmers do treinamento
    )
    logging.info(f"Number of embeddings generated for prediction: {len(protein_embedding_predict.embeddings)}")

    # Obter embeddings para previsão
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Carregar scalers para associated_variable
    scaler_associated_path = os.path.join(model_dir, 'scaler_associated.pkl')

    if os.path.exists(scaler_associated_path):
        scaler_associated = joblib.load(scaler_associated_path)
        logging.info("Scaler for associated_variable loaded successfully.")
    else:
        logging.error("Scalers not found. Please make sure the training was completed successfully.")
        sys.exit(1)

    # Escalar embeddings para previsão usando scaler_associated
    X_predict_scaled_associated = scaler_associated.transform(X_predict)

    # Atualizar progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progress: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Realizar previsão para associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled_associated)

    # Obter rankings das classes
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled_associated)

    # Processar e salvar os resultados
    results = {}
    for entry, pred_associated, ranking_associated in zip(
        protein_embedding_predict.embeddings,
        predictions_associated,
        rankings_associated
    ):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "associated_prediction": pred_associated,
            "associated_ranking": ranking_associated
        }

    # Salvar os resultados em um arquivo
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tAssociated_Prediction\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['associated_prediction']}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Associated Variable: {result['associated_prediction']}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

    # Gerar o Gráfico de Dispersão das Previsões
    logging.info("Generating scatter plot of predictions for new sequences...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatter plot saved at {args.scatterplot_output}")

    st.success("Analysis completed successfully!")

    # Exibir gráfico de dispersão
    st.header("Scatter Plot of Predictions")
    scatterplot_path = args.scatterplot_output
    if os.path.exists(scatterplot_path):
        st.image(scatterplot_path, use_column_width=True)
    else:
        st.error(f"Scatter plot not found at {scatterplot_path}")

    # Formatar os resultados
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"No ranking data associated with the protein. {sequence_id}. Skiping...")
            continue

        # Utiliza a função format_and_sum_probabilities para obter a categoria principal
        top_specificity, confidence, top_two_specificities = format_and_sum_probabilities(associated_rankings)
        if top_specificity is None:
            logging.warning(f"No valid formatting for protein {sequence_id}. Skipping...")
            continue
        formatted_results.append([
            sequence_id,
            top_specificity,
            f"{confidence:.2f}%",
            "; ".join(top_two_specificities)
        ])

    # Converter para DataFrame do pandas
    headers = ["Query Name", "SS Prediction Specificity", "Prediction Confidence", "Top 2 Specificities"]

    df_results = pd.DataFrame(formatted_results, columns=headers)

    headers = ["Query Name", "SS Prediction Specificity", "Prediction Confidence", "Top 2     Specificities"]
    df_results = pd.DataFrame(formatted_results, columns=headers)

# Função para aplicar estilos personalizados
    def highlight_table(df):
        return df.style.set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#1E3A8A'),  # Dark blue for headers
                    ('color', 'white'),
                    ('border', '1px solid white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('background-color', '#0B3C5D'),  # Navy blue for odd rows
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
                    ('background-color', '#145B9C')  # Slightly lighter blue for even rows
                ]
            },
            {
                'selector': 'tr:hover td',
                'props': [
                    ('background-color', '#0D4F8B')  # Darker blue on hover
                ]
            },
        ])

# Aplicar estilos ao DataFrame
    styled_df = highlight_table(df_results)

# Renderizar a tabela estilizada como HTML
    html = styled_df.to_html(index=False, escape=False)

# Injetar CSS para download buttons e ajustar estilos adicionais
    st.markdown(
        """
        <style>
        /* Estilo para download buttons */
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
    
        /* Hover effect para buttons */
        .stButton > button:hover {
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
    st.header("Formatted Results")

    st.markdown(
        f"""
        <div class="dataframe-container">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

# Opcional: Adicionar botões para download da tabela em CSV e Excel
# Estilização dos botões já está coberta pelo CSS acima
    
# Estilização personalizada com CSS para os botões
    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: #1E3A8A; /* Azul escuro */
            color: white; /* Texto branco */
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
            background-color: #145B9C; /* Azul mais claro no hover */
        }
        </style>
    """, unsafe_allow_html=True)

    
# Estilização personalizada com CSS para os botões
    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: #1E3A8A; /* Azul escuro */
            color: white; /* Texto branco */
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
            background-color: #145B9C; /* Azul mais claro no hover */
        }
        </style>
    """, unsafe_allow_html=True)

# Botão para download em CSV
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name='results.csv',
        mime='text/csv',
    )
    

# Botão para download em Excel
    output = BytesIO() 
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Results')
        # writer.save()  # Linha removida
        writer.close()

        processed_data = output.getvalue()

    st.download_button(
        label="Download Results as Excel",
        data=processed_data,
        file_name='results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

        # Prepare results.zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for folder_name, subfolders, filenames in os.walk(output_dir):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_file.write(file_path, arcname=os.path.relpath(file_path, output_dir))
    zip_buffer.seek(0)

        # Provide download link
    st.header("Download All Results")
    st.download_button(
        label="Download All Results as results.zip",
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
    logging.info(f"Formatted table saved at {args.formatted_results_table}")


    # CSS personalizado para fundo azul marinho escuro e texto branco
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
    /* Personaliza a área de arrastar e soltar do uploader de arquivos */
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] div {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Personaliza as opções do dropdown de seleção */
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
    /* Garante que os cabeçalhos sejam brancos */
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

def get_base64_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.

    
    Parameters:
    - image_path (str): Path to the image file.

    
    Retorna:
    - str: String base64 da imagem.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}.")

        return ""

# Caminho para a imagem
image_path = "./images/faal.png"
image_base64 = get_base64_image(image_path)
# Using HTML with st.markdown to align title and text
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



# Barra lateral para parâmetros de entrada
st.sidebar.header("Input Parameters")

# Função para salvar arquivos enviados
def save_uploaded_file(uploaded_file, save_path: str) -> str:
    """
    Salva um arquivo enviado pelo usuário.
    
    Parâmetros:
    - uploaded_file: File uploaded by the user.
    - save_path (str): Path to save the file.
    
    Retorna:
    - str: Caminho para o arquivo salvo.
    """
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Opções de entrada
use_default_train = st.sidebar.checkbox("Use Default Training Data", value=True)
if not use_default_train:
    train_fasta_file = st.sidebar.file_uploader("Upload Training FASTA File", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("pload Training FASTA File (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Upload Prediction FASTA File", type=["fasta", "fa", "fna"])


kmer_size = st.sidebar.number_input("K-mer Size", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Step Size", min_value=1, max_value=10, value=1, step=1)

aggregation_method = st.sidebar.selectbox(
    "Método de Agregação",
    options=['none', 'mean'],  # Apenas 'none' e 'mean' são opções
    index=0
)

# Parâmetros opcionais do Word2Vec
st.sidebar.header("Customize Word2Vec Parameters")
custom_word2vec = st.sidebar.checkbox("Customize Word2Vec Parameters", value=False)
if custom_word2vec:
    window = st.sidebar.number_input(
        "Window Size", min_value=5, max_value=20, value=10, step=5
    )
    workers = st.sidebar.number_input(
        "Workers", min_value=1, max_value=112, value=8, step=8
    )
    epochs = st.sidebar.number_input(
        "Epochs", min_value=1, max_value=2500, value=2500, step=100
    )
else:
    window = 10  # Valor padrão
    workers = 8  # Valor padrão
    epochs = 2500  # Valor padrão

# Diretório de saída baseado no método de agregação
model_dir = create_unique_model_directory("results", aggregation_method)
output_dir = model_dir

# Button to start processing
if st.sidebar.button("Run Analysis"):
    # Caminhos para dados internos
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"

    # Manipulação dos dados de treinamento
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
            st.markdown("<span style='color:white'>Training data uploaded will be used.</span>", unsafe_allow_html=True)

        else:
            st.error("Please upload both the training FASTA file and the training TSV table file.")

            st.stop()

    # Manipulação dos dados de previsão
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(output_dir, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Please upload a FASTA file for prediction.")
        st.stop()
        
    # Parâmetros restantes
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
        scaler="scaler.pkl",
        model_dir=model_dir,
    )

    # Criar diretório de modelo se não existir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Executar a função principal de análise
    st.markdown("<span style='color:white'>Processing data and running analysis...</span>", unsafe_allow_html=True)
    try:
        main(args)

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")
        logging.error(f"Ocorreu um erro: {e}")

# Função para carregar e redimensionar imagens com ajuste de DPI
def load_and_resize_image_with_dpi(image_path: str, base_width: int, dpi: int = 300) -> Image.Image:
    """
    Carrega e redimensiona uma imagem com ajuste de DPI.
    
    Parâmetros:
    - image_path (str): Caminho para o arquivo de imagem.
    - base_width (int): Largura base para redimensionamento.
    - dpi (int): DPI para a imagem.
    
    Retorna:
    - Image.Image: Objeto de imagem redimensionada.
    """
    try:
        # Carrega a imagem
        image = Image.open(image_path)
        # Calcula a nova altura proporcional à largura base
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        # Redimensiona a imagem
        resized_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return resized_image
    except FileNotFoundError:
        logging.error(f"Imagem não encontrada em {image_path}.")
        return None

# Função para codificar imagens em base64
def encode_image(image: Image.Image) -> str:
    """
    Codifica uma imagem como uma string base64.
    
    Parâmetros:
    - image (Image.Image): Objeto de imagem.
    
    Retorna:
    - str: String base64 da imagem.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

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

# Carrega e redimensiona todas as imagens
images = [load_and_resize_image_with_dpi(path, base_width=150, dpi=300) for path in image_paths]

# Codifica as imagens em base64
encoded_images = [encode_image(img) for img in images if img is not None]

# CSS para layout do rodapé
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

# HTML para exibir imagens no rodapé
footer_html = """
<div class="support-text">Supported by:</div>
<div class="footer-container">
    {}
</div>
<div class="footer-text">
    CIIMAR - Pedro Leão @CNP - 2024 - All rights reserved.
</div>

"""

# Gera tags <img> para cada imagem
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Renderiza o rodapé
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)

# ============================================
# Fim do Script
# ============================================
