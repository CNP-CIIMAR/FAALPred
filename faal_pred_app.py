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
import base64
from io import BytesIO
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    Verifica se todas as sequências em um arquivo FASTA têm o mesmo comprimento.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Retorna True se todas as sequências tiverem o mesmo comprimento


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
        logging.info(f"Sequências realinhadas salvas em {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar MAFFT: {e.stderr.decode()}")
        sys.exit(1)


def plot_roc_curve_global(y_true, y_pred_proba, title, save_as=None, classes=None):
    """
    Plota curva ROC para classificações binárias ou multiclasses.
    """
    lw = 2  # Espessura da linha

    # Verifica se é classificação binária ou multiclasses
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:  # Classificação binária
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Curva ROC (área = %0.2f)' % roc_auc)
    else:  # Classificação multiclasses
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
        plt.savefig(save_as, bbox_inches='tight', facecolor='#0B3C5D')  # Combina com a cor de fundo
    plt.close()


def get_class_rankings_global(model, X):
    """
    Obtém rankings de classes com base nas probabilidades previstas pelo modelo.
    """
    if model is None:
        raise ValueError("Modelo ainda não treinado. Por favor, treine o modelo primeiro.")

    # Obtém probabilidades para cada classe
    y_pred_proba = model.predict_proba(X)

    # Ordena as classes com base nas probabilidades
    class_rankings = []
    for probabilities in y_pred_proba:
        ranked_classes = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
        class_rankings.append(formatted_rankings)

    return class_rankings


def calculate_roc_values(model, X_test, y_test):
    """
    Calcula valores de ROC AUC para cada classe.
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


def format_and_sum_probabilities(associated_rankings):
    """
    Formata e soma probabilidades para cada categoria, retornando apenas as top 3.
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

    # Soma as probabilidades para cada categoria
    for rank in associated_rankings:
        try:
            prob = float(rank.split(": ")[1].replace("%", ""))
        except (IndexError, ValueError):
            logging.error(f"Erro ao processar string de ranking: {rank}")
            continue
        for category, patterns in pattern_mapping.items():
            if any(pattern in rank for pattern in patterns):
                category_sums[category] += prob

    # Ordena os resultados e formata para saída (top 3)
    sorted_results = sorted(category_sums.items(), key=lambda x: x[1], reverse=True)[:3]
    formatted_results = [f"{category} ({sum_prob:.2f}%)" for category, sum_prob in sorted_results if sum_prob > 0]

    return " - ".join(formatted_results)


class Support:
    """
    Classe de suporte para treinar e avaliar modelos Random Forest com técnicas de oversampling.
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
            "max_depth": 5,  # Reduzido para evitar overfitting
            "min_samples_split": 4,  # Aumentado para evitar overfitting
            "min_samples_leaf": 2,
            "criterion": "entropy",
            "max_features": "log2",  # Alterado de 'sqrt' para 'log2'
            "class_weight": "balanced",  # Balanceamento automático das classes
            "max_leaf_nodes": 20,  # Ajustado para maior regularização
            "min_impurity_decrease": 0.01,
            "bootstrap": True,
            "ccp_alpha": 0.001,
            "random_state": self.seed  # Adicionado para RandomForest
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
        Personaliza o oversampling para evitar oversampling de classes extremamente raras.
        """
        counter = Counter(y)
        classes_to_oversample = [cls for cls, count in counter.items() if count >= 2]

        # Aplica RandomOverSampler apenas para classes com pelo menos 2 amostras
        ros = RandomOverSampler(random_state=self.seed)
        X_ros, y_ros = ros.fit_resample(X, y)

        # Aplica SMOTE para classes que podem ser sintetizadas
        smote = SMOTE(random_state=self.seed)
        X_smote, y_smote = smote.fit_resample(X_ros, y_ros)

        sample_counts = Counter(y_smote)
        logging.info(f"Distribuição das classes após oversampling: {sample_counts}")

        with open("oversampling_counts.txt", "a") as f:
            f.write("Distribuição das Classes após Oversampling:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        return X_smote, y_smote

    def fit(self, X, y, model_name_prefix='model', model_dir=None, min_kmers=None):
        logging.info(f"Iniciando o método fit para {model_name_prefix}...")

        X = np.array(X)
        y = np.array(y)

        X_smote, y_smote = self._oversample_single_sample_classes(X, y)

        sample_counts = Counter(y_smote)
        logging.info(f"Contagens das amostras após oversampling para {model_name_prefix}: {sample_counts}")

        with open("sample_counts_after_oversampling.txt", "a") as f:
            f.write(f"Contagens das Amostras após Oversampling para {model_name_prefix}:\n")
            for cls, count in sample_counts.items():
                f.write(f"{cls}: {count}\n")

        if any(count < self.cv for count in sample_counts.values()):
            raise ValueError(f"Existem classes com menos membros do que o número de folds após oversampling para {model_name_prefix}.")

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
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Distribuição das classes no conjunto de teste: {fold_class_distribution}")

            X_train_resampled, y_train_resampled = self._oversample_single_sample_classes(X_train, y_train)

            train_sample_counts = Counter(y_train_resampled)
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Distribuição das classes no conjunto de treinamento após oversampling: {train_sample_counts}")

            with open("training_sample_counts_after_oversampling.txt", "a") as f:
                f.write(f"Fold {fold_number} Contagens das Amostras de Treinamento após Oversampling para {model_name_prefix}:\n")
                for cls, count in train_sample_counts.items():
                    f.write(f"{cls}: {count}\n")

            self.model = RandomForestClassifier(**self.init_params, n_jobs=self.n_jobs)
            self.model.fit(X_train_resampled, y_train_resampled)

            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)

            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

            # Calcula F1-score e Precision-Recall AUC
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)

            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)

            if len(np.unique(y_test)) > 1:
                pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')
            else:
                pr_auc = 0.0  # Não pode calcular PR AUC para uma única classe
            self.pr_auc_scores.append(pr_auc)

            logging.info(f"Fold {fold_number} [{model_name_prefix}]: F1 Score: {f1}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Precision-Recall AUC: {pr_auc}")

            # Calcula ROC AUC
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
                logging.warning(f"Não foi possível calcular ROC AUC para o fold {fold_number} [{model_name_prefix}] devido à representação insuficiente das classes.")

            # Realiza grid search e salva o melhor modelo
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            self.best_params = best_params

            if model_dir:
                best_model_filename = os.path.join(model_dir, f'best_model_{model_name_prefix}.pkl')
                # Garante que o diretório exista
                os.makedirs(os.path.dirname(best_model_filename), exist_ok=True)
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Melhor modelo salvo como {best_model_filename} para {model_name_prefix}")
            else:
                best_model_filename = f'best_model_{model_name_prefix}.pkl'
                joblib.dump(best_model, best_model_filename)
                logging.info(f"Melhor modelo salvo como {best_model_filename} para {model_name_prefix}")

            if best_params is not None:
                self.best_params = best_params
                logging.info(f"Melhores parâmetros para {model_name_prefix}: {self.best_params}")
            else:
                logging.warning(f"Não foram encontrados melhores parâmetros do grid search para {model_name_prefix}.")

            # Integra Calibração de Probabilidades
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=5, n_jobs=self.n_jobs)
            calibrator.fit(X_train_resampled, y_train_resampled)

            self.model = calibrator

            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'calibrated_model_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Modelo calibrado salvo como {calibrated_model_filename} para {model_name_prefix}")

            fold_number += 1

            # Permite que o Streamlit atualize a UI
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
        logging.info(f"Melhores parâmetros do grid search: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_

    def get_best_param(self, param_name, default=None):
        return self.best_params.get(param_name, default)

    def plot_learning_curve(self, output_path):
        plt.figure()
        plt.plot(self.train_scores, label='Pontuação de Treinamento')
        plt.plot(self.test_scores, label='Pontuação de Validação Cruzada')
        plt.plot(self.f1_scores, label='F1 Score')
        plt.plot(self.pr_auc_scores, label='Precision-Recall AUC')
        plt.title("Curva de Aprendizagem", color='white')
        plt.xlabel("Fold", fontsize=12, fontweight='bold', color='white')
        plt.ylabel("Pontuação", fontsize=12, fontweight='bold', color='white')
        plt.legend(loc="best")
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.savefig(output_path, facecolor='#0B3C5D')  # Combina com a cor de fundo
        plt.close()

    def get_class_rankings(self, X):
        """
        Obtém rankings de classes para os dados fornecidos.
        """
        if self.model is None:
            raise ValueError("Modelo ainda não treinado. Por favor, treine o modelo primeiro.")

        # Obtém probabilidades para cada classe
        y_pred_proba = self.model.predict_proba(X)

        # Ordena as classes com base nas probabilidades
        class_rankings = []
        for probabilities in y_pred_proba:
            ranked_classes = sorted(zip(self.model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            formatted_rankings = [f"{cls}: {prob*100:.2f}%" for cls, prob in ranked_classes]
            class_rankings.append(formatted_rankings)

        return class_rankings

    def test_best_RF(self, X, y, scaler_dir='.'):
        """
        Testa o melhor modelo Random Forest com os dados fornecidos.
        """
        # Carrega o scaler
        scaler_path = os.path.join(scaler_dir, 'scaler.pkl') if scaler_dir else 'scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logging.info(f"Scaler carregado de {scaler_path}")
        else:
            logging.error(f"Scaler não encontrado em {scaler_path}")
            sys.exit(1)

        X_scaled = scaler.transform(X)

        # Aplica oversampling em todo o conjunto de dados antes da divisão
        X_resampled, y_resampled = self._oversample_single_sample_classes(X_scaled, y)

        # Divide em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        # Treina o RandomForestClassifier com os melhores parâmetros
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
        model.fit(X_train, y_train)  # Treina o modelo nos dados de treinamento

        # Integra Calibração no Modelo de Teste
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=self.n_jobs)
        calibrator.fit(X_train, y_train)
        calibrated_model = calibrator

        # Faz predições
        y_pred = calibrated_model.predict_proba(X_test)
        y_pred_adjusted = adjust_predictions_global(y_pred, method='normalize')

        # Calcula a pontuação (ex: AUC)
        score = self._calculate_score(y_pred_adjusted, y_test)

        # Calcula métricas adicionais
        y_pred_classes = calibrated_model.predict(X_test)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(y_test, y_pred_adjusted, average='macro')
        else:
            pr_auc = 0.0  # Não pode calcular PR AUC para uma única classe

        # Retorna a pontuação, melhores parâmetros, modelo treinado e conjuntos de teste
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred, y_test):
        """
        Calcula a pontuação (ex: ROC AUC) com base nas predições e rótulos reais.
        """
        n_classes = len(np.unique(y_test))
        if y_pred.ndim == 1 or n_classes == 2:
            return roc_auc_score(y_test, y_pred)
        elif y_pred.ndim == 2 and n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            return roc_auc_score(y_test_bin, y_pred, multi_class='ovo', average='macro')
        else:
            logging.warning(f"Formato ou número de classes inesperado: forma de y_pred: {y_pred.shape}, número de classes: {n_classes}")
            return 0

    def plot_roc_curve(self, y_true, y_pred_proba, title, save_as=None, classes=None):
        """
        Plota curva ROC para classificações binárias ou multiclasses.
        """
        plot_roc_curve_global(y_true, y_pred_proba, title, save_as, classes)


class ProteinEmbeddingGenerator:
    def __init__(self, sequences_path, table_data=None, aggregation_method='none'):
        aligned_path = sequences_path
        if not are_sequences_aligned(sequences_path):
            realign_sequences_with_mafft(sequences_path, sequences_path.replace(".fasta", "_aligned.fasta"), threads=1)
            aligned_path = sequences_path.replace(".fasta", "_aligned.fasta")
        else:
            logging.info(f"As sequências já estão alinhadas: {sequences_path}")

        self.alignment = AlignIO.read(aligned_path, 'fasta')
        self.table_data = table_data
        self.embeddings = []
        self.models = {}
        self.aggregation_method = aggregation_method  # Método de agregação
        self.min_kmers = None  # Armazena o mínimo de k-mers

    def generate_embeddings(self, k=3, step_size=1, word2vec_model_path="word2vec_model.bin", model_dir=None, min_kmers=None, save_min_kmers=False):
        """
        Gera embeddings para sequências de proteínas usando Word2Vec, padronizando o número de k-mers.
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

                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]

                else:
                    # Se não houver tabela, usa valores padrão ou None
                    target_variable = None
                    associated_variable = None

                logging.info(f"Processando {protein_accession_alignment} com comprimento de sequência {seq_len}")

                if seq_len < k:
                    logging.warning(f"Sequência muito curta para {protein_accession_alignment}. Comprimento: {seq_len}")
                    continue

                # Gera k-mers, permitindo k-mers com menos de k gaps
                kmers = [sequence[i:i + k] for i in range(0, seq_len - k + 1, step_size)]
                kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Permite k-mers com menos de k gaps

                if not kmers:
                    logging.warning(f"Nenhum k-mer válido para {protein_accession_alignment}")
                    continue

                all_kmers.append(kmers)  # Adiciona a lista de k-mers como uma sentença
                kmers_counts.append(len(kmers))  # Armazena a contagem de k-mers

                embedding_info = {
                    'protein_accession': protein_accession_alignment,
                    'target_variable': target_variable,
                    'associated_variable': associated_variable,
                    'kmers': kmers  # Armazena os k-mers para uso posterior
                }
                kmer_groups[protein_accession_alignment] = embedding_info

            # Determina o mínimo número de k-mers
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
                vector_size=125,  # alterado para 125
                window=window if 'window' in locals() else 10,  # Usa o parâmetro personalizado ou padrão
                min_count=1,
                workers=workers if 'workers' in locals() else 8,
                sg=1,
                hs=1,  # Softmax hierárquico habilitado
                negative=0,  # Amostragem negativa desabilitada
                epochs=epochs if 'epochs' in locals() else 2500,  # Fixar número de épocas para reprodutibilidade
                seed=SEED  # Fixar semente para reprodutibilidade
            )

            # Cria o diretório para o modelo Word2Vec se necessário
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
                # Se método não reconhecido, usa concatenação como padrão
                logging.warning(f"Método de agregação desconhecido '{self.aggregation_method}'. Usando concatenação.")
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)

            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': embedding_concatenated,
                'target_variable': embedding_info.get('target_variable'),
                'associated_variable': embedding_info.get('associated_variable')
            })

            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {embedding_concatenated.shape}")

        # Ajusta o StandardScaler com os embeddings para treinamento/predição
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])

        # Verifica se todos os embeddings têm a mesma forma
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Formas inconsistentes de embeddings detectadas: {embedding_shapes}")
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


    def get_embeddings_and_labels(self, label_type='target_variable'):
        """
        Retorna embeddings e rótulos associados (target_variable ou associated_variable).
        """
        embeddings = []
        labels = []

        for embedding_info in self.embeddings:
            embeddings.append(embedding_info['embedding'])
            labels.append(embedding_info[label_type])  # Usa o tipo de rótulo especificado

        return np.array(embeddings), np.array(labels)


# Ajustar perplexidade dinamicamente
def compute_perplexity(n_samples):
    return max(5, min(50, n_samples // 100))


def plot_predictions_scatterplot_custom(results, output_path, top_n=3):
    """
    Gera um gráfico de dispersão das top N predições para as novas sequências.

    Eixo Y: ID de acesso da proteína
    Eixo X: Specificidades de C2 a C18 (escala fixa)
    Cada ponto representa a especificidade correspondente para a proteína.
    Apenas as top N predições são plotadas.
    Pontos são coloridos em uma única cor uniforme, estilizados para publicação científica.
    """
    # Prepara os dados
    protein_specificities = {}
    
    for seq_id, info in results.items():
        associated_rankings = info.get('associated_ranking', [])
        if not associated_rankings:
            logging.warning(f"Nenhum dado de ranking associado para a proteína {seq_id}. Pulando...")
            continue

        specificity_probs = {}
        for rank in associated_rankings[:top_n]:
            try:
                # Divide e extrai os dados
                category, prob = rank.split(": ")
                prob = float(prob.replace("%", ""))

                # Extrai o primeiro número da categoria
                if category.startswith('C'):
                    # Extrai apenas o primeiro número antes dos dois pontos ou qualquer outro separador
                    spec = int(category.split(':')[0].strip('C'))
                    specificity_probs[spec] = prob
            except ValueError as e:
                logging.error(f"Erro ao processar ranking: {rank} para a proteína {seq_id}. Erro: {e}")

        if specificity_probs:
            protein_specificities[seq_id] = specificity_probs

    if not protein_specificities:
        logging.warning("Nenhum dado disponível para plotar o gráfico de dispersão.")
        return

    # Ordena IDs de proteínas para melhor visualização
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    # Cria a figura
    fig, ax = plt.subplots(figsize=(12, len(unique_proteins) * 0.5))  # Ajusta a altura com base no número de proteínas

    # Escala fixa para o eixo X de C2 a C18
    x_values = list(range(2, 19))

    for protein, specs in protein_specificities.items():
        y = protein_order[protein]
        
        # Prepara os dados para plotagem (garante que apenas as top N predições especificadas são plotadas)
        x = []
        probs = []
        for spec in x_values:
            if spec in specs:
                x.append(spec)
                probs.append(specs[spec])

        if not x:
            logging.warning(f"Nenhum dado válido para plotar para a proteína {protein}. Pulando...")
            continue

        # Plota os pontos em uma cor fixa (ex: azul escuro)
        ax.scatter(x, [y] * len(x), color='#1f78b4', edgecolors='black', linewidth=0.5, s=100, label='_nolegend_')

        # Conecta os pontos com linhas
        if len(x) > 1:
            ax.plot(x, [y] * len(x), color='#1f78b4', linestyle='-', linewidth=1.0, alpha=0.7)

    # Personaliza o gráfico para melhor qualidade de publicação
    ax.set_xlabel('Specificidade (C2 a C18)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proteínas', fontsize=14, fontweight='bold')
    ax.set_title('Gráfico de Dispersão das Predições das Novas Sequências (Top 3 Rankings)', fontsize=16, fontweight='bold', pad=20)

    # Define escala fixa e formatação do eixo X
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'C{spec}' for spec in x_values], fontsize=12)
    ax.set_yticks(range(len(unique_proteins)))
    ax.set_yticklabels(unique_proteins, fontsize=10)

    # Define grade e remove bordas desnecessárias para um visual limpo
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ticks menores no eixo X para melhor visibilidade
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.6)

    # Ajusta o layout para evitar corte de rótulos
    plt.tight_layout()

    # Salva a figura em alta qualidade para publicação
    plt.savefig(output_path, facecolor='white', dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Gráfico de dispersão salvo em {output_path}")


def adjust_predictions_global(predicted_proba, method='normalize', alpha=1.0):
    """
    Ajusta as probabilidades previstas pelo modelo.
    """
    if method == 'normalize':
        # Normaliza as probabilidades para que somem 1 para cada amostra
        logging.info("Normalizando probabilidades previstas.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)

    elif method == 'smoothing':
        # Aplica suavização nas probabilidades para evitar valores extremos
        logging.info(f"Aplicando suavização nas probabilidades previstas com alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])

    elif method == 'none':
        # Não aplica nenhum ajuste
        logging.info("Nenhum ajuste aplicado nas probabilidades previstas.")
        adjusted_proba = predicted_proba.copy()

    else:
        logging.warning(f"Método de ajuste desconhecido '{method}'. Nenhum ajuste será aplicado.")
        adjusted_proba = predicted_proba.copy()

    return adjusted_proba


def main(args):
    model_dir = args.model_dir  # Deve ser 'results/models'

    """
    Função principal que coordena o fluxo de trabalho.
    """
    model_dir = args.model_dir

    # Inicializa variáveis de progresso
    total_steps = 8
    current_step = 0
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # =============================
    # STEP 1: Treinamento do Modelo
    # =============================

    # Carrega os dados de treinamento
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table

    # Verifica se as sequências de treinamento estão alinhadas
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Sequências de treinamento não estão alinhadas. Realinhando com MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)  # Threads fixos em 1
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Arquivo de treinamento alinhado encontrado ou sequências já alinhadas: {train_alignment_path}")

    # Carrega os dados da tabela de treinamento
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Tabela de dados de treinamento carregada com sucesso.")

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Inicializa e gera embeddings para o treinamento
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, 
        train_table_data, 
        aggregation_method=args.aggregation_method  # Passa o método de agregação
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        save_min_kmers=True  # Salva min_kmers após o treinamento
    )
    logging.info(f"Número de embeddings de treinamento gerados: {len(protein_embedding_train.embeddings)}")

    # Salva min_kmers para garantir consistência
    min_kmers = protein_embedding_train.min_kmers

    # Obtém embeddings e rótulos para target_variable
    X_target, y_target = protein_embedding_train.get_embeddings_and_labels(label_type='target_variable')
    logging.info(f"Forma de X_target: {X_target.shape}")

    # Caminhos completos para modelos target_variable
    rf_model_target_full_path = os.path.join(model_dir, args.rf_model_target)
    calibrated_model_target_full_path = os.path.join(model_dir, 'calibrated_model_target.pkl')

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Verifica se o modelo calibrado para target_variable já existe
    if os.path.exists(calibrated_model_target_full_path):
        calibrated_model_target = joblib.load(calibrated_model_target_full_path)
        logging.info(f"Modelo Random Forest calibrado para target_variable carregado de {calibrated_model_target_full_path}")
    else:
        # Treinamento do modelo para target_variable
        support_model_target = Support()
        calibrated_model_target = support_model_target.fit(X_target, y_target, model_name_prefix='target', model_dir=model_dir, min_kmers=min_kmers)
        logging.info("Treinamento e calibração para target_variable concluídos.")

        # Salva o modelo calibrado
        joblib.dump(calibrated_model_target, calibrated_model_target_full_path)
        logging.info(f"Modelo Random Forest calibrado para target_variable salvo em {calibrated_model_target_full_path}")

        # Testa o modelo
        best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, scaler_dir=args.model_dir)

        logging.info(f"Melhor ROC AUC para target_variable: {best_score}")
        logging.info(f"Melhor F1 Score para target_variable: {best_f1}")
        logging.info(f"Melhor Precision-Recall AUC para target_variable: {best_pr_auc}")
        logging.info(f"Melhores Parâmetros: {best_params}")

        for param, value in best_params.items():
            logging.info(f"{param}: {value}")

        # Obtém rankings de classe
        class_rankings = support_model_target.get_class_rankings(X_test_target)

        # Exibe rankings para as primeiras 5 amostras
        logging.info("Top 3 rankings de classe para as primeiras 5 amostras:")
        for i in range(min(5, len(class_rankings))):
            logging.info(f"Amostra {i+1}: Rankings de classe - {class_rankings[i][:3]}")  # Mostra os top 3 rankings

        # Plota curva ROC
        n_classes_target = len(np.unique(y_test_target))
        if n_classes_target == 2:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)[:, 1]
        else:
            y_pred_proba_target = best_model_target.predict_proba(X_test_target)
            unique_classes_target = np.unique(y_test_target).astype(str)
        plot_roc_curve_global(y_test_target, y_pred_proba_target, 'Curva ROC para Target Variable', save_as=args.roc_curve_target, classes=unique_classes_target)

        # Converte y_test_target para rótulos inteiros
        unique_labels = sorted(set(y_test_target))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        y_test_target_int = [label_to_int[label.strip()] for label in y_test_target]

        # Calcula e imprime valores de ROC para target_variable
        roc_df_target = calculate_roc_values(best_model_target, X_test_target, y_test_target_int)
        logging.info("Scores de ROC AUC para target_variable:")
        logging.info(roc_df_target)
        roc_df_target.to_csv(args.roc_values_target, index=False)

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Repete o processo para associated_variable
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"Forma de X_associated: {X_associated.shape}")

    # Caminhos completos para modelos associated_variable
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Verifica se o modelo calibrado para associated_variable já existe
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Modelo Random Forest calibrado para associated_variable carregado de {calibrated_model_associated_full_path}")
    else:
        # Treinamento do modelo para associated_variable
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(X_associated, y_associated, model_name_prefix='associated', model_dir=model_dir, min_kmers=min_kmers)
        logging.info("Treinamento e calibração para associated_variable concluídos.")
        
        # Plota curva de aprendizagem
        logging.info("Plotando Curva de Aprendizagem para Associated Variable")
        support_model_associated.plot_learning_curve(args.learning_curve_associated)

        # Salva o modelo calibrado
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Modelo Random Forest calibrado para associated_variable salvo em {calibrated_model_associated_full_path}")

        # Testa o modelo
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, scaler_dir=args.model_dir)

        logging.info(f"Melhor ROC AUC para associated_variable em test_best_RF: {best_score_associated}")
        logging.info(f"Melhor F1 Score para associated_variable em test_best_RF: {best_f1_associated}")
        logging.info(f"Melhor Precision-Recall AUC para associated_variable em test_best_RF: {best_pr_auc_associated}")
        logging.info(f"Melhores Parâmetros encontrados em test_best_RF: {best_params_associated}")
        logging.info(f"Melhor modelo Associated em test_best_RF: {best_model_associated}")

        # Obtém rankings de classe para associated_variable
        class_rankings_associated = support_model_associated.get_class_rankings(X_test_associated)
        logging.info("Top 3 rankings de classe para as primeiras 5 amostras nos dados associados:")
        for i in range(min(5, len(class_rankings_associated))):
            logging.info(f"Amostra {i+1}: Rankings de classe - {class_rankings_associated[i][:3]}")  # Mostra os top 3 rankings

        # Acessa class_weight do dicionário best_params_associated
        class_weight = best_params_associated.get('class_weight', None)
        # Imprime resultados
        logging.info(f"Peso das classes utilizado: {class_weight}")

        # Salva o modelo treinado para associated_variable
        joblib.dump(best_model_associated, rf_model_associated_full_path)
        logging.info(f"Modelo Random Forest para associated_variable salvo em {rf_model_associated_full_path}")

        # Plota curva ROC para associated_variable
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            unique_classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve_global(y_test_associated, y_pred_proba_associated, 'Curva ROC para Associated Variable', save_as=args.roc_curve_associated, classes=unique_classes_associated)

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # =============================
    # STEP 2: Classificação de Novas Sequências
    # =============================

    # Carrega min_kmers
    min_kmers_path = os.path.join(model_dir, 'min_kmers.txt')
    if os.path.exists(min_kmers_path):
        with open(min_kmers_path, 'r') as f:
            min_kmers_loaded = int(f.read().strip())
        logging.info(f"min_kmers carregado: {min_kmers_loaded}")
    else:
        logging.error(f"Arquivo min_kmers não encontrado em {min_kmers_path}. Assegure-se de que o treinamento foi concluído com sucesso.")
        sys.exit(1)

    # Carrega os dados para predição
    predict_alignment_path = args.predict_fasta

    # Verifica se as sequências para predição estão alinhadas
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Sequências para predição não estão alinhadas. Realinhando com MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Threads fixos em 1
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Arquivo alinhado para predição encontrado ou sequências já alinhadas: {predict_alignment_path}")

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Inicializa ProteinEmbedding para predição, sem necessidade da tabela
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, 
        table_data=None,
        aggregation_method=args.aggregation_method  # Passa o método de agregação
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir,
        min_kmers=min_kmers_loaded  # Usa o mesmo min_kmers do treinamento
    )
    logging.info(f"Número de embeddings gerados para predição: {len(protein_embedding_predict.embeddings)}")

    # Obtém embeddings para predição
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Carrega o scaler
    scaler_full_path = os.path.join(model_dir, args.scaler)
    if os.path.exists(scaler_full_path):
        scaler = joblib.load(scaler_full_path)
        logging.info(f"Scaler carregado de {scaler_full_path}")
    else:
        logging.error(f"Scaler não encontrado em {scaler_full_path}")
        sys.exit(1)
    X_predict_scaled = scaler.transform(X_predict)

    # Atualiza o progresso
    current_step += 1
    progress = min(current_step / total_steps, 1.0)
    progress_bar.progress(progress)
    progress_text.markdown(f"<span style='color:white'>Progresso: {int(progress * 100)}%</span>", unsafe_allow_html=True)
    time.sleep(0.1)

    # Faz predições em novas sequências

    # Verifica o tamanho das features antes da predição
# Verifique o número de características em relação ao estimador original do CalibratedClassifierCV
    if X_predict_scaled.shape[1] > calibrated_model_target.estimator.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_target.estimator.n_features_in_} to match the model input size.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_target.estimator.n_features_in_]


    predictions_target = calibrated_model_target.predict(X_predict_scaled)

    # Verificar e ajustar o tamanho das features para associated_variable
    if X_predict_scaled.shape[1] > calibrated_model_associated.estimator.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_associated.base_estimator_.n_features_in_} to match the model input size for associated_variable.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_associated.estimator_.n_features_in_]

    # Realiza a predição para associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)

    # Obtém rankings de classe
    rankings_target = get_class_rankings_global(calibrated_model_target, X_predict_scaled)
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled)

    # Processa e salva os resultados
    results = {}
    for entry, pred_target, pred_associated, ranking_target, ranking_associated in zip(protein_embedding_predict.embeddings, predictions_target, predictions_associated, rankings_target, rankings_associated):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "target_prediction": pred_target,
            "associated_prediction": pred_associated,
            "target_ranking": ranking_target,
            "associated_ranking": ranking_associated
        }

    # Salva os resultados em um arquivo
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tTarget_Prediction\tAssociated_Prediction\tTarget_Ranking\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['target_prediction']}\t{result['associated_prediction']}\t{'; '.join(result['target_ranking'])}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Target Variable: {result['target_prediction']}, Associated Variable: {result['associated_prediction']}, Target Ranking: {'; '.join(result['target_ranking'])}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

    # Formata os resultados
    formatted_results = []

    for sequence_id, info in results.items():
        associated_rankings = info['associated_ranking']
        formatted_prob_sums = format_and_sum_probabilities(associated_rankings)
        formatted_results.append([sequence_id, formatted_prob_sums])

    # Log para verificar o conteúdo de formatted_results
    logging.info("Resultados Formulados:")
    for result in formatted_results:
        logging.info(result)

    # Extrai category_sums para labels_predict
    labels_predict = [result[1] for result in formatted_results]

    # Imprime resultados em uma tabela formatada
    headers = ["Protein Accession", "Associated Prob. Rankings"]
    logging.info(tabulate(formatted_results, headers=headers, tablefmt="grid"))

    # Salva os resultados em um arquivo Excel
    df = pd.DataFrame(formatted_results, columns=headers)
    df.to_excel(args.excel_output, index=False)
    logging.info(f"Resultados salvos em {args.excel_output}")

    # Salva a tabela em formato tabulado
    with open(args.formatted_results_table, 'w') as f:
        f.write(tabulate(formatted_results, headers=headers, tablefmt="grid"))
    logging.info(f"Tabela formatada salva em {args.formatted_results_table}")

    # Gera o gráfico de dispersão das predições
    logging.info("Gerando gráfico de dispersão das predições das novas sequências...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Gráfico de dispersão salvo em {args.scatterplot_output}")

    # ============================================
    # STEP 3: Dimensionality Reduction and Plotting t-SNE & UMAP
    # ============================================
    # Removido conforme solicitado

    # Atualiza o progresso para 100%
    progress_bar.progress(1.0)
    progress_text.markdown("<span style='color:black'>Progresso: 100%</span>", unsafe_allow_html=True)
    time.sleep(0.1)


# Custom CSS para fundo azul marinho escuro e texto branco
st.markdown(
    """
    <style>
    /* Define o fundo principal do app e a cor do texto */
    .stApp {
        background-color: #0B3C5D;
        color: white;
    }
    /* Define o fundo da sidebar e a cor do texto */
    [data-testid="stSidebar"] {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Garante que todos os elementos dentro da sidebar tenham fundo azul e texto branco */
    [data-testid="stSidebar"] * {
        background-color: #0B3C5D !important;
        color: white !important;
    }
    /* Personaliza elementos de input dentro da sidebar */
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
    /* Personaliza a área de drag and drop do file uploader */
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] div {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    /* Personaliza opções do dropdown select */
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
    /* Personaliza caixas de seleção e botões de rádio */
    [data-testid="stSidebar"] .stCheckbox input[type="checkbox"] + div:first-of-type,
    [data-testid="stSidebar"] .stRadio input[type="radio"] + div:first-of-type {
        background-color: #1E3A8A !important;
    }
    /* Personaliza a trilha e o polegar do slider */
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

from PIL import Image
# Função para converter a imagem em base64
def get_base64_image(image_path):
    """
    Codifica um arquivo de imagem para uma string base64.
    
    Parâmetros:
    - image_path (str): Caminho para o arquivo de imagem.
    
    Retorna:
    - string base64 da imagem.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.error(f"Imagem não encontrada em {image_path}.")
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
            treatment (Kurt et al., 2017; Gilmore et al., 2012). Dr. Pedro Leão and His Team Identified Several of These Natural Products in Cyanobacteria (<a href="https://leaolab.wixsite.com/leaolab" target="_blank" style="color: #3498db; text-decoration: none;">visite aqui</a>), 
            e FAALpred classifica FAALs por sua especificidade de substrato.
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{image_base64}" alt="FAAL domain" style="width: auto; height: 120px; object-fit: contain;">
            <p style="text-align: center; color: #2c3e50; font-size: 14px; margin-top: 5px;">
                <em>Domínio FAAL de Synechococcus sp. PCC7002, link: <a href="https://www.rcsb.org/structure/7R7F" target="_blank" style="color: #3498db; text-decoration: none;">https://www.rcsb.org/structure/7R7F</a></em>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar para parâmetros de entrada
st.sidebar.header("Parâmetros de Entrada")

# Função para salvar arquivos carregados
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Opções de entrada
use_default_train = st.sidebar.checkbox("Usar dados de treinamento padrão", value=True)
if not use_default_train:
    train_fasta_file = st.sidebar.file_uploader("Enviar Arquivo FASTA de Treinamento", type=["fasta", "fa", "fna"])
    train_table_file = st.sidebar.file_uploader("Enviar Arquivo de Tabela de Treinamento (TSV)", type=["tsv"])
else:
    train_fasta_file = None
    train_table_file = None

predict_fasta_file = st.sidebar.file_uploader("Enviar Arquivo FASTA para Predição", type=["fasta", "fa", "fna"])

kmer_size = st.sidebar.number_input("Tamanho do K-mer", min_value=1, max_value=10, value=3, step=1)
step_size = st.sidebar.number_input("Tamanho do Passo", min_value=1, max_value=10, value=1, step=1)
aggregation_method = st.sidebar.selectbox(
    "Método de Agregação",
    options=['none', 'mean'],  # Removidas as opções 'sum' e 'max'
    index=0
)

# Entrada opcional para parâmetros do Word2Vec
st.sidebar.header("Parâmetros Opcionais do Word2Vec")
custom_word2vec = st.sidebar.checkbox("Personalizar Parâmetros do Word2Vec", value=False)
if custom_word2vec:
    window = st.sidebar.number_input(
        "Tamanho da Janela", min_value=5, max_value=20, value=5, step=5
    )
    workers = st.sidebar.number_input(
        "Trabalhadores", min_value=1, max_value=112, value=8, step=8
    )
    epochs = st.sidebar.number_input(
        "Épocas", min_value=1, max_value=3500, value=2500, step=100
    )
else:
    window = 10  # Valor padrão
    workers = 8  # Valor padrão
    epochs = 2500  # Valor padrão

# Output directory
#output_dir = "results"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# Botão para iniciar o processamento
if st.sidebar.button("Executar Análise"):
    # Caminhos para dados internos
    internal_train_fasta = "data/train.fasta"
    internal_train_table = "data/train_table.tsv"
    
    model_dir = create_unique_model_directory("results", aggregation_method)
    output_dir = model_dir
    # Tratamento dos dados de treinamento
    if use_default_train:
        train_fasta_path = internal_train_fasta
        train_table_path = internal_train_table
        st.markdown("<span style='color:white'>Usando dados de treinamento padrão.</span>", unsafe_allow_html=True)
    else:
        if train_fasta_file is not None and train_table_file is not None:
            train_fasta_path = os.path.join(output_dir, "uploaded_train.fasta")
            train_table_path = os.path.join(output_dir, "uploaded_train_table.tsv")
            save_uploaded_file(train_fasta_file, train_fasta_path)
            save_uploaded_file(train_table_file, train_table_path)
            st.markdown("<span style='color:white'>Dados de treinamento enviados serão usados.</span>", unsafe_allow_html=True)
        else:
            st.error("Por favor, envie tanto o arquivo FASTA de treinamento quanto o arquivo de tabela TSV de treinamento.")
            st.stop()

    # Tratamento dos dados de predição
    if predict_fasta_file is not None:
        predict_fasta_path = os.path.join(output_dir, "uploaded_predict.fasta")
        save_uploaded_file(predict_fasta_file, predict_fasta_path)
    else:
        st.error("Por favor, envie um arquivo FASTA para predição.")
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
        roc_curve_target=os.path.join(output_dir, "roc_curve_target.png"),
        roc_curve_associated=os.path.join(output_dir, "roc_curve_associated.png"),
        learning_curve_target=os.path.join(output_dir, "learning_curve_target.png"),
        learning_curve_associated=os.path.join(output_dir, "learning_curve_associated.png"),
        roc_values_target=os.path.join(output_dir, "roc_values_target.csv"),
        rf_model_target="rf_model_target.pkl",
        rf_model_associated="rf_model_associated.pkl",
        word2vec_model="word2vec_model.bin",
        scaler="scaler.pkl",
        # model_dir=os.path.join(output_dir, "models")
        model_dir=model_dir,
    )

    # Cria o diretório do modelo se não existir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Executa a função principal de análise
    st.markdown("<span style='color:white'>Processando dados e executando análise...</span>", unsafe_allow_html=True)
    try:
        main(args)

        st.success("Análise concluída com sucesso!")

        # Exibe o gráfico de dispersão
        st.header("Gráfico de Dispersão das Predições")
        scatterplot_path = os.path.join(args.output_dir, "scatterplot_predictions.png")
        st.image(scatterplot_path, use_column_width=True)



        # Exibe a tabela de resultados formatados
        # st.header("Tabela de Resultados Formatados")
        # with open(args.formatted_results_table, 'r') as f:
        #     formatted_table = f.read()
        # st.text(formatted_table)
    # Caminho do arquivo formatado
        formatted_table_path = args.formatted_results_table

    # Verifica se o arquivo existe e não está vazio
        if os.path.exists(formatted_table_path) and os.path.getsize(formatted_table_path) > 0:
            try:
        # Abre e lê o conteúdo do arquivo
                with open(formatted_table_path, 'r') as f:
                    formatted_table = f.read()

        # Exibe o conteúdo no Streamlit
                st.text(formatted_table)
            except Exception as e:
                st.error(f"Ocorreu um erro ao ler a tabela de resultados formatados: {e}")
        else:
            st.error(f"Tabela de resultados formatados não encontrada ou está vazia: {formatted_table_path}")
    
        # Prepara o arquivo results.zip
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for folder_name, subfolders, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    zip_file.write(file_path, arcname=os.path.relpath(file_path, output_dir))
        zip_buffer.seek(0)

        # Fornece o link de download
        st.header("Download dos Resultados")
        st.download_button(
            label="Baixar Todos os Resultados como results.zip",
            data=zip_buffer,
            file_name="results.zip",
            mime="application/zip"
        )

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")
        logging.error(f"Ocorreu um erro: {e}")


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
        logging.error(f"Imagem não encontrada em {image_path}.")
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

# Carrega e redimensiona todas as imagens
images = [load_and_resize_image_with_dpi(path, base_width=150, dpi=300) for path in image_paths]

# Codifica imagens como base64
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
    CIIMAR - Pedro Leão @CNP - 2024 - Leandro de Mattos Pereira (developer) - Todos os direitos reservados.
</div>
"""

# Gera tags <img> para cada imagem
img_tags = "".join(
    f'<img src="data:image/png;base64,{img}" style="width: 100px;">' for img in encoded_images
)

# Renderiza o rodapé
st.markdown(footer_html.format(img_tags), unsafe_allow_html=True)
# ===========



