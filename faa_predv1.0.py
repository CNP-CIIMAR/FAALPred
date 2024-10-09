# Authors: Leandro de Mattos Pereira, Anne Liong and Pedro Leão
# 10/10/2024
import argparse
import logging
import os
import sys
import subprocess
from collections import Counter, defaultdict
from Bio import SeqIO
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from Bio import AlignIO
from Bio.Align.Applications import MafftCommandline
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, label_binarize
from tabulate import tabulate
import umap


# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)


def generate_accuracy_pie_chart(formatted_results, table_data, output_path):
    """
    Gera um gráfico de pizza mostrando a precisão por categoria.

    Parâmetros:
    - formatted_results (list): Lista contendo os resultados de predição formatados.
    - table_data (DataFrame): DataFrame contendo acessos de proteína e variáveis associadas.
    - output_path (str): Caminho do arquivo para salvar o gráfico de pizza.

    Retorna:
    - None
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

    # Criação do gráfico de pizza
    accuracy = {category: (correct_counts[category] / category_counts[category] * 100) if category_counts[category] > 0 else 0
                for category in category_counts.keys()}

    # Remover categorias com contagem 0 para evitar NaN no gráfico
    accuracy = {k: v for k, v in accuracy.items() if category_counts[k] > 0}

    plt.figure(figsize=(8, 8))
    if accuracy:
        plt.pie(accuracy.values(), labels=[f'{key} ({val:.1f}%)' for key, val in accuracy.items()], autopct='%1.1f%%')
    else:
        logging.warning("Nenhum dado para plotar no gráfico de pizza.")
    plt.title('Precisão por Categoria')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_associated_variable_scatterplot(formatted_results, table_data, output_path):
    """
    Gera um scatter plot mostrando a média das probabilidades por categoria.

    Parâmetros:
    - formatted_results (list): Lista contendo os resultados de predição formatados.
    - table_data (DataFrame): DataFrame contendo acessos de proteína e variáveis associadas.
    - output_path (str): Caminho do arquivo para salvar o scatter plot.

    Retorna:
    - None
    """
    pattern_mapping = {
        'C4-C6-C8': ['C4', 'C6', 'C8'],
        'C6-C8-C10': ['C6', 'C8', 'C10'],
        'C8-C10-C12': ['C8', 'C10', 'C12'],
        'C10-C12-C14': ['C10', 'C12', 'C14'],
        'C12-C14-C16': ['C12', 'C14', 'C16'],
        'C14-C16-C18': ['C14', 'C16', 'C18'],
    }

    scatter_data = {category: [] for category in pattern_mapping.keys()}
    for result in formatted_results:
        seq_id = result[0]
        pred_entries = result[1].split(" - ")

        corresponding_row = table_data[table_data['Protein.accession'].str.split().str[0] == seq_id]
        if not corresponding_row.empty:
            associated_variable_real = corresponding_row['Associated variable'].values[0]
            for pred_entry in pred_entries:
                try:
                    pred_cat, prob = pred_entry.split(" (")
                    prob = float(prob.replace('%)', ''))
                    for category, patterns in pattern_mapping.items():
                        if any(pat in pred_cat for pat in patterns) and any(pat in associated_variable_real for pat in patterns):
                            scatter_data[category].append(prob)
                except ValueError:
                    logging.error(f"Erro ao processar a string: {pred_entry}")

    # Agregando dados para o scatter plot
    aggregated_data = {category: np.mean(probs) if probs else None for category, probs in scatter_data.items()}

    plt.figure(figsize=(12, 8))
    for category, mean_prob in aggregated_data.items():
        if mean_prob is not None:  # Só plotar categorias com dados
            plt.scatter([category], [mean_prob], label=f'{category}')

    plt.xlabel('Categorias')
    plt.ylabel('Média das Probabilidades')
    plt.title('Média das Probabilidades por Categoria')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def is_dark_color(color):
    """Determine if a color is dark based on its luminance."""
    r, g, b, _ = color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return luminance < 0.5


def get_class_rankings_global(model, X):
    """
    Obtém as classificações das classes com base nas probabilidades preditas pelo modelo.

    Parâmetros:
    - model: Modelo treinado que possui o método predict_proba.
    - X (array-like): Dados de entrada para predição.

    Retorna:
    - class_rankings (list): Lista de listas contendo as classes e suas probabilidades ordenadas.
    """
    if model is None:
        raise ValueError("Model not fitted yet. Please fit the model first.")

    # Obtendo probabilidades para cada classe
    y_pred_proba = model.predict_proba(X)

    # Classificando as classes com base nas probabilidades
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings


def calculate_roc_values(model, X_test, y_test):
    """
    Calcula os valores ROC AUC para cada classe.

    Parâmetros:
    - model: Modelo treinado que possui o método predict_proba.
    - X_test (array-like): Dados de teste.
    - y_test (array-like): Rótulos reais de teste.

    Retorna:
    - roc_df (DataFrame): DataFrame contendo a ROC AUC para cada classe.
    """
    n_classes = len(np.unique(y_test))
    y_pred_proba = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Logging valores de ROC
        logging.info(f"For class {i}:")
        logging.info(f"FPR: {fpr[i]}")
        logging.info(f"TPR: {tpr[i]}")
        logging.info(f"ROC AUC: {roc_auc[i]}")
        logging.info("--------------------------")

    roc_df = pd.DataFrame(list(roc_auc.items()), columns=['Class', 'ROC AUC'])
    return roc_df


def plot_roc_curve_global(y_true, y_pred_proba, title, save_as=None, classes=None):
    """
    Plota a curva ROC para classificações binárias ou multiclasse.

    Parâmetros:
    - y_true (array-like): Rótulos reais.
    - y_pred_proba (array-like): Probabilidades preditas.
    - title (str): Título do gráfico.
    - save_as (str): Caminho para salvar o gráfico.
    - classes (list): Lista de classes para legendas.

    Retorna:
    - None
    """
    lw = 2  # Largura da linha

    # Verifica se é classificação binária ou multiclasse
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # Classificação binária
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
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
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve of class {class_label} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.close()


# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)

def are_sequences_aligned(fasta_file):
    """
    Verifica se todas as sequências em um arquivo FASTA têm o mesmo comprimento.
    
    Parâmetros:
    - fasta_file (str): Caminho para o arquivo FASTA a ser verificado.
    
    Retorna:
    - bool: Verdadeiro se todas as sequências estiverem alinhadas, falso caso contrário.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Retorna True se todas as sequências tiverem o mesmo comprimento

def realign_sequences_with_mafft(input_path, output_path, threads=4):
    """
    Realinha sequências usando o MAFFT.

    Parâmetros:
    - input_path (str): Caminho para o arquivo de entrada FASTA.
    - output_path (str): Caminho para salvar o arquivo alinhado FASTA.
    - threads (int): Número de threads para o MAFFT.

    Retorna:
    - None
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Sequências realinhadas salvas em {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar MAFFT: {e.stderr.decode()}")
        sys.exit(1)

def classify_new_sequences(aligned_sequences_path, model_target, model_associated, word2vec_model_path, scaler_path, k=3, step_size=1, results_file="results.tsv"):
    """
    Classifica novas sequências usando modelos treinados e embeddings Word2Vec.

    Parâmetros:
    - aligned_sequences_path (str): Caminho para o arquivo de sequências alinhadas em formato FASTA.
    - model_target: Modelo treinado para a variável alvo.
    - model_associated: Modelo treinado para a variável associada.
    - word2vec_model_path (str): Caminho para o modelo Word2Vec salvo.
    - scaler_path (str): Caminho para o StandardScaler salvo.
    - k (int): Tamanho do k-mer.
    - step_size (int): Tamanho do passo para geração de k-mers.
    - results_file (str): Caminho para salvar os resultados em formato TSV.

    Retorna:
    - results (dict): Dicionário contendo as predições e rankings para cada sequência.
    """
    # Verifica se o modelo Word2Vec existe
    if not os.path.exists(word2vec_model_path):
        logging.error(f"Word2Vec model not found at {word2vec_model_path}")
        sys.exit(1)

    # Verifica se o scaler existe
    if not os.path.exists(scaler_path):
        logging.error(f"Scaler not found at {scaler_path}")
        sys.exit(1)

    # Carrega o modelo Word2Vec e o scaler
    word2vec_model = Word2Vec.load(word2vec_model_path)
    scaler = joblib.load(scaler_path)

    # Verifica se as sequências estão alinhadas
    if not are_sequences_aligned(aligned_sequences_path):
        logging.info("Sequências não estão alinhadas. Realinhando com MAFFT...")
        realign_sequences_with_mafft(aligned_sequences_path, aligned_sequences_path.replace(".fasta", "_aligned.fasta"))
        aligned_sequences_path = aligned_sequences_path.replace(".fasta", "_aligned.fasta")

    embeddings = []
    with open(aligned_sequences_path, 'r') as file:
        alignment = AlignIO.read(file, 'fasta')
    
    for record in alignment:
        sequence = str(record.seq)
        kmers = [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, step_size) if sequence[i:i+k].count('-') != k]
        valid_kmers = [kmer for kmer in kmers if kmer in word2vec_model.wv]
        
        if valid_kmers:
            embedding_average = np.mean([word2vec_model.wv[kmer] for kmer in valid_kmers], axis=0)
            embeddings.append(embedding_average)
        else:
            # Handle sequences with no valid kmers
            embeddings.append(np.zeros(word2vec_model.vector_size))

    # Transformando os embeddings
    embeddings = scaler.transform(embeddings)

    # Realiza previsões
    predictions_target = model_target.predict(embeddings)
    predictions_associated = model_associated.predict(embeddings)

    # Obtém rankings de classe
    rankings_target = get_class_rankings_global(model_target, embeddings)
    rankings_associated = get_class_rankings_global(model_associated, embeddings)

    results = {}
    for record, pred_target, pred_associated, ranking_target, ranking_associated in zip(alignment, predictions_target, predictions_associated, rankings_target, rankings_associated):
        results[record.id] = {
            "target_prediction": pred_target,
            "associated_prediction": pred_associated,
            "target_ranking": ranking_target,
            "associated_ranking": ranking_associated
        }

    # Salvando resultados em um arquivo TSV
    with open(results_file, 'w') as f:
        f.write("Protein_ID\tTarget_Prediction\tAssociated_Prediction\tTarget_Ranking\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['target_prediction']}\t{result['associated_prediction']}\t{'; '.join(result['target_ranking'])}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Target Variable: {result['target_prediction']}, Associated Variable: {result['associated_prediction']}, Target Ranking: {'; '.join(result['target_ranking'])}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

    return results
    
def adjust_predictions_global(y_pred, y_test, all_classes):
    """
    Ajusta as previsões para garantir que todas as classes estejam representadas.

    Parâmetros:
    - y_pred (array-like): Previsões do modelo.
    - y_test (array-like): Rótulos reais de teste.
    - all_classes (list): Lista de todas as classes possíveis.

    Retorna:
    - y_pred_adjusted (array-like): Previsões ajustadas.
    """
    present_classes = np.unique(y_test)
    class_to_index = {cls: index for index, cls in enumerate(all_classes)}
    y_pred_adjusted = np.zeros((y_pred.shape[0], len(all_classes)))

    for i, cls in enumerate(present_classes):
        if cls in class_to_index:
            index = class_to_index[cls]
            y_pred_adjusted[:, index] = y_pred[:, i]

    return y_pred_adjusted


class Suport:
    """
    Classe de suporte para treinamento e avaliação de modelos Random Forest com técnicas de oversampling.
    """

    def __init__(self, cv=5, seed=42, n_jobs=-1):
        """
        Inicializa a classe Suport.

        Parâmetros:
        - cv (int): Número de folds para validação cruzada.
        - seed (int): Semente para reprodutibilidade.
        - n_jobs (int): Número de jobs para paralelização.
        """
        self.cv = cv
        self.model = None
        self.seed = seed
        self.n_jobs = n_jobs
        self.train_scores = []  # Lista para armazenar os scores de treinamento
        self.test_scores = []   # Lista para armazenar os scores de teste
        self.roc_results = []   # Lista para armazenar os resultados ROC
        self.train_sizes = np.linspace(.1, 1.0, 5)

        self.standard = StandardScaler()

        self.init_params = {
            "n_estimators": 100,  # Mantendo um número moderado
            "max_depth": 10,  # Reduzindo a profundidade para prevenir sobreajuste
            "min_samples_split": 2,  # Aumentando para prevenir sobreajuste
            "min_samples_leaf": 2,  # Mantendo alto para evitar sobreajuste
            "criterion": "entropy",  # Mantendo como está
            "max_features": "sqrt",  # Mantendo uma boa escolha padrão
            "class_weight": None,  # Testando sem balanceamento de classes
            "max_leaf_nodes": 10,  # Reduzindo os nós folha para prevenir sobreajuste
            "min_impurity_decrease": 0.02,  # Aumentando o valor para prevenir sobreajuste
            "bootstrap": True,  # Testando sem bootstrap
            "ccp_alpha": 0.01,  # Incluindo poda de árvores            
        }

        self.parameters = {
            "n_estimators": [100],  # Ampliando a gama de opções
            "max_depth": [10],  # Incluindo opções mais baixas
            "min_samples_split": [4],  # Experimentando com valores mais altos
            "min_samples_leaf": [1],  # Variação maior nos limites para folhas
            "criterion": ["entropy"],  # Mantendo uma opção de critério
            "max_features": ["log2"],  # Incluindo uma opção numérica baixa
            "class_weight": [None],  # Tanto balanceado quanto não balanceado
            "max_leaf_nodes": [None],  # Incluindo mais opções para nós folha
            "min_impurity_decrease": [0.0],  # Experimentando com um intervalo maior
            "bootstrap": [True],  # Incluindo a opção de não usar bootstrap
            "ccp_alpha": [0.001],  # Diversas opções para poda            
        }

    def feature_engineering(self, X, y):
        """
        Aplica seleção de features com base na importância das features determinada por um RandomForest.

        Parâmetros:
        - X (array-like): Dados de entrada.
        - y (array-like): Rótulos de entrada.

        Retorna:
        - X_transformed (array-like): Dados de entrada transformados.
        - sfm.get_support() (array-like): Máscara indicando quais features foram selecionadas.
        """
        # Primeiro, treine um modelo para obter a importância das features
        model = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        model.fit(X, y)
        
        # Use SelectFromModel para selecionar as features baseadas na importância
        sfm = SelectFromModel(model, prefit=True)
        X_transformed = sfm.transform(X)
        return X_transformed, sfm.get_support()

    def _oversample_single_sample_classes(self, X, y):
        """
        Aplica RandomOverSampler e SMOTE para lidar com classes desbalanceadas.

        Parâmetros:
        - X (array-like): Dados de entrada.
        - y (array-like): Rótulos de entrada.

        Retorna:
        - X_smote (array-like): Dados após oversampling.
        - y_smote (array-like): Rótulos após oversampling.
        """
        # Aplicar RandomOverSampler
        ros = RandomOverSampler(random_state=self.seed)
        X_ros, y_ros = ros.fit_resample(X, y)

        # Obter o número mínimo de amostras por classe após RandomOverSampler
        _, counts_ros = np.unique(y_ros, return_counts=True)
        min_count_ros = min(counts_ros)

        # Ajustar k_neighbors para SMOTE baseado no min_count_ros
        # Garante que k_neighbors seja pelo menos 2 e menor que min_count_ros
        k_neighbors_smote = max(2, min(min_count_ros - 1, 5))

        # Aplicar SMOTE
        smote = SMOTE(k_neighbors=k_neighbors_smote, random_state=self.seed)
        X_smote, y_smote = smote.fit_resample(X_ros, y_ros)

        # Registro das quantidades de amostras após oversampling
        sample_counts = Counter(y_smote)
        logging.info(f"Class distribution after oversampling: {sample_counts}")

        # Salvar as quantidades em um arquivo
        with open("oversampling_counts.txt", "w") as f:
            f.write("Class Distribution after Oversampling:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote

    def fit(self, X, y):
        """
        Treina o modelo Random Forest utilizando validação cruzada com oversampling.

        Parâmetros:
        - X (array-like): Dados de entrada.
        - y (array-like): Rótulos de entrada.

        Retorna:
        - None
        """
        logging.info("Starting the fit method...")

        X = np.array(X)
        y = np.array(y)

        # Aplicar oversampling
        X_smote, y_smote = self._oversample_single_sample_classes(X, y)

        # Salvando a quantidade de amostras para cada classe
        sample_counts = Counter(y_smote)
        logging.info(f"Sample counts after oversampling: {sample_counts}")

        # Salvar as quantidades em um arquivo
        with open("sample_counts_after_oversampling.txt", "w") as f:
            f.write("Sample Counts after Oversampling:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        # Garantir que todas as classes tenham pelo menos um membro por fold
        if any(count < self.cv for count in sample_counts.values()):
            raise ValueError("Há classes com menos membros que o número de folds após o oversampling.")

        # Reavaliar o número de folds com base na classe com menor contagem
        min_class_count = min(sample_counts.values())
        self.cv = min(self.cv, min_class_count)

        # Resetando scores antes do loop
        self.train_scores = []
        self.test_scores = []

        # Inicializar o contador de folds
        fold_number = 1

        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)

        for train_index, test_index in skf.split(X_smote, y_smote):
            X_train, X_test = X_smote[train_index], X_smote[test_index]
            y_train, y_test = y_smote[train_index], y_smote[test_index]

            # Registro da distribuição de classes no fold atual
            unique, counts_fold = np.unique(y_test, return_counts=True)
            fold_class_distribution = dict(zip(unique, counts_fold))
            logging.info(f"Fold {fold_number}: Test set class distribution: {fold_class_distribution}")

            # Aplicar oversampling apenas no conjunto de treinamento
            X_train_resampled, y_train_resampled = self._oversample_single_sample_classes(X_train, y_train)

            # Registro das quantidades de amostras após oversampling no treinamento
            train_sample_counts = Counter(y_train_resampled)
            logging.info(f"Fold {fold_number}: Training set class distribution after oversampling: {train_sample_counts}")

            # Salvar as quantidades em um arquivo
            with open("training_sample_counts_after_oversampling.txt", "a") as f:
                f.write(f"Fold {fold_number} Training Sample Counts after Oversampling:\n")
                for cls, count in train_sample_counts.items():
                    f.write(f"{cls}: {count}\n")

            # Treinamento do modelo
            self.model = RandomForestClassifier(**self.init_params, random_state=self.seed)
            self.model.fit(X_train_resampled, y_train_resampled)

            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)

            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

            logging.info(f"Fold {fold_number}: Training Score: {train_score}")
            logging.info(f"Fold {fold_number}: Testing Score: {test_score}")

            # Previsões e ajuste das previsões
            y_pred_proba = self.model.predict_proba(X_test)
            y_pred_adjusted = adjust_predictions_global(y_pred_proba, y_test, self.model.classes_)

            # Calculando ROC AUC para cada classe
            try:
                if len(np.unique(y_test)) == 2:
                    # Classificação binária
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_adjusted[:, 1])
                    roc_auc = auc(fpr, tpr)
                    self.roc_results.append((fpr, tpr, roc_auc))
                else:
                    # Classificação multiclasse
                    y_test_bin = label_binarize(y_test, classes=self.model.classes_)
                    roc_auc_score_value = roc_auc_score(y_test_bin, y_pred_adjusted, multi_class='ovr')
                    self.roc_results.append(roc_auc_score_value)
            except ValueError:
                logging.warning(f"Unable to calculate ROC AUC for fold {fold_number} due to insufficient class representation.")

            # Realizando busca em grade e salvando o melhor modelo
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            joblib.dump(best_model, 'best_model.pkl')

            # Armazenando os melhores parâmetros da busca em grade
            if best_params is not None:
                self.best_n_estimators = best_params['n_estimators']
                self.best_max_depth = best_params['max_depth']
                self.best_min_samples_split = best_params['min_samples_split']
                self.best_min_samples_leaf = best_params['min_samples_leaf']
                self.best_criterion = best_params.get('criterion', 'gini')  # Valor padrão
                self.best_max_features = best_params['max_features']
                self.best_class_weight = best_params['class_weight']
                self.best_max_leaf_nodes = best_params.get("max_leaf_nodes", None)
                self.best_min_impurity_decrease = best_params["min_impurity_decrease"]
                self.best_bootstrap = best_params["bootstrap"]
                self.best_ccp_alpha = best_params["ccp_alpha"]
            else:
                logging.warning("No best parameters found from grid search.")

            fold_number += 1  # Incrementar o contador de folds

    def plot_learning_curve(self, output_path):
        """
        Plota a curva de aprendizado.

        Parâmetros:
        - output_path (str): Caminho para salvar o gráfico.

        Retorna:
        - None
        """
        plt.figure()
        plt.plot(self.train_scores, label='Training score')
        plt.plot(self.test_scores, label='Cross-validation score')
        plt.title("Learning Curve")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(output_path)
        plt.close()

    def test_best_RF(self, X, y, output_dir='.'):
        """
        Testa o melhor modelo Random Forest com os dados fornecidos.

        Parâmetros:
        - X (array-like): Dados de entrada.
        - y (array-like): Rótulos de entrada.
        - output_dir (str): Diretório para salvar os arquivos de saída.

        Retorna:
        - score (float): Score calculado (e.g., AUC).
        - best_params (dict): Melhores parâmetros encontrados na busca em grade.
        - model (RandomForestClassifier): Modelo treinado.
        - X_test (array-like): Dados de teste.
        - y_test (array-like): Rótulos de teste.
        """
        if self.model is None:
            raise ValueError("The fit() method must be called before test_best_RF().")

        self.standard = joblib.load('scaler.pkl')  # Carregar o scaler
        X = self.standard.transform(X)  # Aplicar scaling

        # Aplicar oversampling ao conjunto inteiro antes do split
        X_resampled, y_resampled = self._oversample_single_sample_classes(X, y)

        # Dividir em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=42, stratify=y_resampled
        )

        # Treinar o RandomForestClassifier com os melhores parâmetros
        model = RandomForestClassifier(
            n_estimators=self.best_n_estimators,
            max_depth=self.best_max_depth,
            min_samples_split=self.best_min_samples_split,
            min_samples_leaf=self.best_min_samples_leaf,
            criterion=self.best_criterion,
            max_features=self.best_max_features,
            class_weight=self.best_class_weight,
            max_leaf_nodes=self.best_max_leaf_nodes,
            min_impurity_decrease=self.best_min_impurity_decrease,
            bootstrap=self.best_bootstrap,
            ccp_alpha=self.best_ccp_alpha,
            random_state=self.seed
        )
        model.fit(X_train, y_train)  # Fit the model on the training data

        # Fazer previsões
        y_pred = model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred, y_test, model.classes_)

        # Calcular o score (por exemplo, AUC)
        score = self._calculate_score(y_pred_adjusted, y_test)

        # Retornar o score, melhores parâmetros, modelo treinado e conjuntos de teste
        return score, {
            'n_estimators': self.best_n_estimators,
            'max_depth': self.best_max_depth,
            'min_samples_split': self.best_min_samples_split,
            'min_samples_leaf': self.best_min_samples_leaf,
            'criterion': self.best_criterion,
            'max_features': self.best_max_features,
            'class_weight': self.best_class_weight,
            'max_leaf_nodes': self.best_max_leaf_nodes,
            'min_impurity_decrease': self.best_min_impurity_decrease,
            'bootstrap': self.best_bootstrap,
            'ccp_alpha': self.best_ccp_alpha
        }, model, X_test, y_test

    def _perform_grid_search(self, X_train_resampled, y_train_resampled):
        """
        Realiza a busca em grade para encontrar os melhores parâmetros.

        Parâmetros:
        - X_train_resampled (array-like): Dados de treinamento após oversampling.
        - y_train_resampled (array-like): Rótulos de treinamento após oversampling.

        Retorna:
        - best_estimator (RandomForestClassifier): Melhor modelo encontrado.
        - best_params (dict): Melhores parâmetros encontrados.
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
        logging.info(f"Best parameters from grid search: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def _calculate_score(self, y_pred, y_test):
        """
        Calcula o score (e.g., ROC AUC) com base nas previsões e rótulos reais.

        Parâmetros:
        - y_pred (array-like): Previsões ajustadas.
        - y_test (array-like): Rótulos reais.

        Retorna:
        - score (float): Score calculado.
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

    def get_class_rankings(self, X):
        """
        Obtém as classificações das classes para os dados fornecidos.

        Parâmetros:
        - X (array-like): Dados de entrada.

        Retorna:
        - class_rankings (list): Lista de listas contendo as classes e suas probabilidades ordenadas.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        # Obtendo probabilidades para cada classe
        y_pred_proba = self.model.predict_proba(X)

        # Classificando as classes com base nas probabilidades
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def plot_roc_curve(self, y_true, y_pred_proba, title, save_as=None, classes=None):
        """
        Plota a curva ROC para classificações binárias ou multiclasse.

        Parâmetros:
        - y_true (array-like): Rótulos reais.
        - y_pred_proba (array-like): Probabilidades preditas.
        - title (str): Título do gráfico.
        - save_as (str): Caminho para salvar o gráfico.
        - classes (list): Lista de classes para legendas.

        Retorna:
        - None
        """
        plot_roc_curve_global(y_true, y_pred_proba, title, save_as, classes)


class ProteinEmbedding:
    """
    Classe para geração de embeddings de proteínas utilizando Word2Vec e redução de dimensionalidade.
    """

    def __init__(self, sequences_path, table_data):
        """
        Inicializa a classe ProteinEmbedding.

        Parâmetros:
        - sequences_path (str): Caminho para o arquivo de sequências em formato FASTA.
        - table_data (DataFrame): DataFrame contendo dados da tabela associada.
        """
        aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(sequences_path, aligned_path)

        self.alignment = AlignIO.read(aligned_path, 'fasta')
        self.table_data = table_data
        self.embeddings = []
        self.models = {}

    def generate_embeddings(self, k=3, step_size=1, aggregation_method='mean', word2vec_model_path="word2vec_model.bin"):
        """
        Gera embeddings para as sequências de proteínas usando Word2Vec.

        Parâmetros:
        - k (int): Tamanho do k-mer.
        - step_size (int): Tamanho do passo para geração de k-mers.
        - aggregation_method (str): Método de agregação ('mean', 'median', 'sum', 'max').
        - word2vec_model_path (str): Caminho para salvar o modelo Word2Vec.

        Retorna:
        - None
        """
        # Inicialização das Variáveis
        kmer_groups = {}
        unique_embeddings = {}
        all_kmers = []

        # Geração de k-mers
        for record in self.alignment:
            sequence = str(record.seq)
            seq_len = len(sequence)
            protein_accession_alignment = record.id.split()[0]
            matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
            matching_info = self.table_data[matching_rows]

            if not matching_info.empty:
                target_variable = matching_info['Target variable'].values[0]
                associated_variable = matching_info['Associated variable'].values[0]

                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size) if sequence[i:i + k].count('-') != k]
                all_kmers.extend(kmers)

                if target_variable not in unique_embeddings:
                    unique_embeddings[target_variable] = set()
                unique_embeddings[target_variable].update(kmers)

                embedding_info = {
                    'protein_accession': protein_accession_alignment,
                    'target_variable': target_variable,
                    'associated_variable': associated_variable
                }
                kmer_groups[protein_accession_alignment] = (kmers, embedding_info)

        # Treinar modelo Word2Vec usando todos os k-mers
        model = Word2Vec(
            sentences=[all_kmers],
            vector_size=125,
            window=10,
            min_count=1,
            workers=1,
            sg=1,
            hs=1,  # Hierarchical softmax enabled
            negative=0,  # Negative sampling disabled
            epochs=2500,
            seed=42
        )

        # Salvar o modelo Word2Vec
        model.save(word2vec_model_path)
        self.models['global'] = model
        logging.info(f"Word2Vec model salvo em {word2vec_model_path}")

        # Gerar embeddings
        for protein_accession, (kmers_for_protein, embedding_info) in kmer_groups.items():
            embeddings = [model.wv[kmer] for kmer in kmers_for_protein if kmer in model.wv]

            if embeddings:
                if aggregation_method == 'mean':
                    embedding_aggregate = np.mean(embeddings, axis=0)
                elif aggregation_method == 'median':
                    embedding_aggregate = np.median(embeddings, axis=0)
                elif aggregation_method == 'sum':
                    embedding_aggregate = np.sum(embeddings, axis=0)
                elif aggregation_method == 'max':
                    embedding_aggregate = np.max(embeddings, axis=0)
                else:
                    raise ValueError("Invalid aggregation method")

                embedding_info['embedding'] = embedding_aggregate
                self.embeddings.append(embedding_info)
            else:
                # Handle sequences with no valid kmers
                logging.warning(f"No valid k-mers found for protein {protein_accession}. Assigning zero vector.")
                embedding_info['embedding'] = np.zeros(model.vector_size)
                self.embeddings.append(embedding_info)

        embeddings_array = np.array([entry['embedding'] for entry in self.embeddings])

        # Realizar redução de dimensionalidade com t-SNE
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(embeddings_array)

        # Realizar redução de dimensionalidade com UMAP
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
        umap_results = umap_reducer.fit_transform(embeddings_array)

        df_tsne = pd.DataFrame(tsne_results, columns=['Component 1', 'Component 2', 'Component 3'])
        df_tsne['Protein ID'] = [entry['protein_accession'] for entry in self.embeddings]
        df_tsne['Associated Variable'] = [entry['associated_variable'] for entry in self.embeddings]

        df_umap = pd.DataFrame(umap_results, columns=['Component 1', 'Component 2', 'Component 3'])
        df_umap['Protein ID'] = [entry['protein_accession'] for entry in self.embeddings]
        df_umap['Associated Variable'] = [entry['associated_variable'] for entry in self.embeddings]

        # Ajustar o StandardScaler com os embeddings
        scaler = StandardScaler().fit(embeddings_array)

        # Salvar o StandardScaler ajustado
        joblib.dump(scaler, 'scaler.pkl')
        logging.info("StandardScaler salvo em scaler.pkl")

    def get_embeddings_and_labels(self, label_type='target_variable'):
        """
        Retorna os embeddings e os rótulos associados (target_variable ou associated_variable).

        Parâmetros:
        - label_type (str): Tipo de rótulo ('target_variable' ou 'associated_variable').

        Retorna:
        - embeddings (array-like): Embeddings gerados.
        - labels (array-like): Rótulos associados.
        """
        embeddings = []
        labels = []
        
        for embedding_info in self.embeddings:
            embeddings.append(embedding_info['embedding'])
            labels.append(embedding_info[label_type])  # Usa o tipo de rótulo especificado
        
        return np.array(embeddings), np.array(labels)


def format_and_sum_probabilities(associated_rankings):
    """
    Formata e soma as probabilidades para cada categoria.

    Parâmetros:
    - associated_rankings (list): Lista de rankings associados para uma sequência.

    Retorna:
    - formatted_results (str): String formatada com as somas das probabilidades por categoria.
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

    # Inicializar o dicionário de somas
    for category in categories:
        category_sums[category] = 0.0

    # Somar as probabilidades para cada categoria
    for rank in associated_rankings:
        try:
            prob = float(rank.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Erro ao processar a string de ranking: {rank}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob

    # Ordenar os resultados e formatar para saída
    sorted_results = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)
    formatted_results = [f"{category} ({sum_prob:.2f}%)" for category, sum_prob in sorted_results if sum_prob > 0]

    return " - ".join(formatted_results)


def main(args):
    """
    Função principal que coordena o fluxo de trabalho.

    Parâmetros:
    - args (Namespace): Argumentos da linha de comando.

    Retorna:
    - None
    """
    # Passo 1: Verificação de Dados
    # Passo 1: Verificação de Dados
    alignment_path = args.fasta

    # Definir o caminho para o arquivo alinhado
    if args.aligned_fasta:
        aligned_sequences_path = args.aligned_fasta
    else:
        aligned_sequences_path = args.fasta  # Usar o fasta de entrada se o alinhado não for fornecido

    table_data_path = args.table

    # Verifica se as sequências estão alinhadas
    if not are_sequences_aligned(aligned_sequences_path):
        logging.info("Sequências não estão alinhadas. Realinhando com MAFFT...")
        realign_sequences_with_mafft(aligned_sequences_path, aligned_sequences_path.replace(".fasta", "_aligned.fasta"))
        aligned_sequences_path = aligned_sequences_path.replace(".fasta", "_aligned.fasta")
    else:
        logging.info(f"Arquivo alinhado encontrado ou sequências já alinhadas: {aligned_sequences_path}")

    # Carregar dados da tabela
    table_data = pd.read_csv(table_data_path, delimiter="\t")
    logging.info("Tabela de dados carregada com sucesso.")

    # Inicializar e gerar embeddings
    protein_embedding = ProteinEmbedding(aligned_sequences_path, table_data)
    protein_embedding.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        aggregation_method=args.aggregation_method,
        word2vec_model_path=args.word2vec_model
    )
    logging.info(f"Número de embeddings gerados: {len(protein_embedding.embeddings)}")

    # Obter embeddings e rótulos para target_variable
    X_target, y_target = protein_embedding.get_embeddings_and_labels(label_type='target_variable')
    logging.info(f"X_target shape: {X_target.shape}")

    # Treinamento do modelo para target_variable
    support_model_target = Suport()
    support_model_target.fit(X_target, y_target)
    support_model_target.plot_learning_curve(args.learning_curve_target)

    best_score, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, output_dir=args.output_dir)
    logging.info(f"Best AUC for target_variable: {best_score}")
    logging.info(f"Best Parameters: {best_params}")

    for param, value in best_params.items():
        logging.info(f"{param}: {value}")
        
    # Obter rankings de classe
    class_rankings = support_model_target.get_class_rankings(X_test_target)

    # Exibir os rankings para as primeiras 5 amostras
    logging.info("Top 3 class rankings for the first 5 samples:")
    for i in range(min(5, len(class_rankings))):
        logging.info(f"Sample {i+1}: Class rankings - {class_rankings[i][:3]}")  # Mostra as top 3 classificações

    # Salvar o modelo treinado
    joblib.dump(best_model_target, args.rf_model_target)
    logging.info(f"Random Forest model for target_variable saved to {args.rf_model_target}")

    # Plotar a curva ROC para target_variable
    n_classes_target = len(np.unique(y_test_target))
    if n_classes_target == 2:
        y_pred_proba_target = best_model_target.predict_proba(X_test_target)[:, 1]
    else:
        y_pred_proba_target = best_model_target.predict_proba(X_test_target)
        unique_classes_target = np.unique(y_test_target).astype(str)
    plot_roc_curve_global(y_test_target, y_pred_proba_target, 'ROC Curve for target_variable', save_as=args.roc_curve_target, classes=unique_classes_target)

    # Converter y_test_target para rótulos inteiros
    unique_labels = sorted(set(y_test_target))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_test_target_int = [label_to_int[label.strip()] for label in y_test_target]

    # Calcular e imprimir valores ROC para target_variable
    roc_df_target = calculate_roc_values(best_model_target, X_test_target, y_test_target_int)
    logging.info("ROC AUC Scores for target_variable:")
    logging.info(roc_df_target)
    roc_df_target.to_csv(args.roc_values_target, index=False)

    # Repetir o processo para associated_variable
    X_associated, y_associated = protein_embedding.get_embeddings_and_labels(label_type='associated_variable')

    # Treinamento do modelo para associated_variable
    support_model_associated = Suport()
    support_model_associated.fit(X_associated, y_associated)
    logging.info("Printing learning Curve for Associated variable")
    support_model_associated.plot_learning_curve(args.learning_curve_associated)

    best_score_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, output_dir=args.output_dir)
    logging.info(f"Best AUC for associated_variable in teste_RF: {best_score_associated}")
    logging.info(f"Best Parameters found in teste_RF: {best_params_associated}")
    logging.info(f"Best model Associated in teste_RF: {best_model_associated}")

    # Obter rankings de classe para associated_variable
    class_rankings_associated = support_model_associated.get_class_rankings(X_test_associated)
    logging.info("Top 3 class rankings for the first 5 samples in associated data:")
    for i in range(min(5, len(class_rankings_associated))):
        logging.info(f"Sample {i+1}: Class rankings - {class_rankings_associated[i][:3]}")  # Mostra as top 3 classificações

    # Accessing class_weight from the best_params_associated dictionary
    class_weight = best_params_associated.get('class_weight', None)
    # Printing results
    logging.info(f"Class weight used: {class_weight}")

    # Salvar o modelo treinado para associated_variable
    joblib.dump(best_model_associated, args.rf_model_associated)
    logging.info(f"Random Forest model for associated_variable saved to {args.rf_model_associated}")

    # Plotar a curva ROC para associated_variable
    n_classes_associated = len(np.unique(y_test_associated))
    if n_classes_associated == 2:
        y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
    else:
        y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
        unique_classes_associated = np.unique(y_test_associated).astype(str)
    plot_roc_curve_global(y_test_associated, y_pred_proba_associated, 'ROC Curve for associated_variable', save_as=args.roc_curve_associated, classes=unique_classes_associated)

    #########################################

    # Classificar novas sequências
    results = classify_new_sequences(
        aligned_sequences_path, 
        best_model_target, 
        best_model_associated, 
        word2vec_model_path=args.word2vec_model,
        scaler_path='scaler.pkl',
        k=args.kmer_size,
        step_size=args.step_size,
        results_file=args.results_file
    )

    # Formatar resultados
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info['associated_ranking']
        formatted_prob_sums = format_and_sum_probabilities(associated_rankings)
        formatted_results.append([sequence_id, formatted_prob_sums])

    # Registro para verificar o conteúdo de formatted_results
    logging.info("Formatted Results:")
    for result in formatted_results:
        logging.info(result)

    # Imprimir os resultados em uma tabela formatada
    headers = ["Protein Accession", "Associated Prob. Rankings"]
    logging.info(tabulate(formatted_results, headers=headers, tablefmt="grid"))

    # Salvar os resultados em um arquivo Excel
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Resultados salvos em {args.excel_output}")

    # Salvar a tabela no formato tabulado
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Tabela formatada salva em {args.formatted_results_table}")

    # Gerar gráficos
    generate_accuracy_pie_chart(formatted_results, table_data, args.accuracy_pie_chart_png)
    logging.info(f"Gráfico de pizza salvo em {args.accuracy_pie_chart_png}")

    generate_associated_variable_scatterplot(formatted_results, table_data, args.associated_variable_scatterplot_png)
    logging.info(f"Scatterplot salvo em {args.associated_variable_scatterplot_png}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise de Embeddings de Proteínas e Classificação com Random Forest")

    # Argumentos de entrada
    parser.add_argument('-f', '--fasta', required=True, help='Caminho para o arquivo .fasta caracterizado')
    parser.add_argument('-a', '--aligned_fasta', required=False, help='Caminho para salvar/usar o arquivo alinhado .fasta')
    parser.add_argument('-t', '--table', required=True, help='Caminho para a tabela de dados (CSV ou TSV)')

    # Argumentos para k-mer
    parser.add_argument('--kmer_size', type=int, default=3, help='Tamanho do k-mer (default: 3)')
    parser.add_argument('--step_size', type=int, default=1, help='Tamanho do passo para geração de k-mers (default: 1)')
    parser.add_argument('--aggregation_method', type=str, default='mean', choices=['mean', 'median', 'sum', 'max'], help='Método de agregação para embeddings (default: mean)')

    # Argumento de saída para resultados principais
    parser.add_argument('-o', '--results_file', required=True, help='Caminho para salvar os resultados de predição (TSV)')
    parser.add_argument('--output_dir', required=True, help='Caminho para o diretório onde os arquivos de saída serão salvos')

    # Argumentos de saída para gráficos e outros arquivos
    parser.add_argument('--accuracy_pie_chart_png', required=True, help='Caminho para salvar o gráfico de pizza de precisão (PNG)')
    parser.add_argument('--accuracy_pie_chart_svg', required=True, help='Caminho para salvar o gráfico de pizza de precisão (SVG)')
    parser.add_argument('--associated_variable_scatterplot_png', required=True, help='Caminho para salvar o scatterplot de variáveis associadas (PNG)')
    parser.add_argument('--associated_variable_scatterplot_svg', required=True, help='Caminho para salvar o scatterplot de variáveis associadas (SVG)')
    parser.add_argument('--excel_output', required=True, help='Caminho para salvar os resultados em Excel')
    parser.add_argument('--formatted_results_table', required=True, help='Caminho para salvar a tabela de resultados formatados (TXT)')
    parser.add_argument('--roc_curve_target', required=True, help='Caminho para salvar a curva ROC para a variável alvo (PNG)')
    parser.add_argument('--roc_curve_associated', required=True, help='Caminho para salvar a curva ROC para a variável associada (PNG)')
    parser.add_argument('--learning_curve_target', required=True, help='Caminho para salvar a curva de aprendizado para a variável alvo (PNG)')
    parser.add_argument('--learning_curve_associated', required=True, help='Caminho para salvar a curva de aprendizado para a variável associada (PNG)')
    parser.add_argument('--roc_values_target', required=True, help='Caminho para salvar os valores ROC para a variável alvo (CSV)')
    parser.add_argument('--rf_model_target', required=True, help='Caminho para salvar o modelo Random Forest para a variável alvo (PKL)')
    parser.add_argument('--rf_model_associated', required=True, help='Caminho para salvar o modelo Random Forest para a variável associada (PKL)')
    parser.add_argument('--word2vec_model', required=True, help='Caminho para salvar o modelo Word2Vec (BIN)')
    parser.add_argument('--scaler', required=True, help='Caminho para salvar o StandardScaler (PKL)')

    args = parser.parse_args()

    # Criar o diretório de saída para todos os arquivos, se necessário
    output_dirs = set()
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, str) and ('/' in value or '\\' in value):
            dir_path = os.path.dirname(value)
            if dir_path and dir_path not in output_dirs:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                output_dirs.add(dir_path)

    main(args)

