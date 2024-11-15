# Authors: Leandro de Mattos Pereira, Anne Liong and Pedro Leão
import argparse 
import logging
import os
import sys
import subprocess
import random
from collections import Counter
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MafftCommandline
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tabulate import tabulate
from sklearn.calibration import CalibratedClassifierCV

# Fixando as sementes para reproducibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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

def plot_predictions_scatterplot_custom(results, output_path):
    """
    Gera um scatter plot das previsões das novas sequências.

    Eixo Y: ID de acesso da proteína
    Eixo X: Especificidades de 2 a 18
    Cada ponto representa a especificidade correspondente para a proteína
    Linhas conectam os pontos de cada proteína
    Os pontos são representados em escala de cinza, indicando a porcentagem associada.
    """
    # Preparar os dados
    protein_specificities = {}

    for seq_id, info in results.items():
        rankings = info['associated_ranking']
        specificity_probs = {}

        for ranking in rankings:
            try:
                category, prob = ranking.split(": ")
                prob = float(prob.replace("%", ""))
                # Extrair os números das categorias, assumindo que são no formato 'C4-C6-C8'
                specs = [int(s.strip('C')) for s in category.split('-') if s.startswith('C')]
                for spec in specs:
                    if spec in specificity_probs:
                        specificity_probs[spec] += prob  # Somar probabilidades se já existir
                    else:
                        specificity_probs[spec] = prob
            except ValueError:
                logging.error(f"Erro ao processar o ranking: {ranking} para a proteína {seq_id}")

        if specificity_probs:
            # Normalizar as probabilidades para cada especificidade
            total_prob = sum(specificity_probs.values())
            if total_prob > 0:
                for spec in specificity_probs:
                    specificity_probs[spec] = (specificity_probs[spec] / total_prob) * 100
            protein_specificities[seq_id] = specificity_probs

    if not protein_specificities:
        logging.warning("Nenhum dado disponível para plotar o scatterplot.")
        return

    # Ordenar os IDs das proteínas para melhor visualização
    unique_proteins = sorted(protein_specificities.keys())
    protein_order = {protein: idx for idx, protein in enumerate(unique_proteins)}

    plt.figure(figsize=(20, max(10, len(unique_proteins) * 0.5)))  # Ajustar a altura com base no número de proteínas

    for protein, specs in protein_specificities.items():
        y = protein_order[protein]
        x = sorted(specs.keys())
        probs = [specs[spec] for spec in x]

        # Normalizar as probabilidades para [0,1] para escala de cinza
        probs_normalized = [p / 100.0 for p in probs]

        # Mapear as probabilidades para cores em escala de cinza (1 - p para que maior probabilidade seja mais escura)
        colors = [str(1 - p) for p in probs_normalized]

        # Plotar os pontos
        plt.scatter(x, [y] * len(x), c=colors, cmap='gray', edgecolors='w', s=100)

        # Conectar os pontos com linhas
        plt.plot(x, [y] * len(x), color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.xlabel('Especificidade', fontsize=12, fontweight='bold')
    plt.ylabel('Proteínas', fontsize=12, fontweight='bold')
    plt.title('Scatterplot das Previsões das Novas Sequências', fontsize=14, fontweight='bold')

    plt.yticks(ticks=range(len(unique_proteins)), labels=unique_proteins, fontsize=8)
    plt.xlim(2, 18)
    plt.xticks(ticks=range(2, 19), fontsize=10)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def get_class_rankings_global(model, X):
    """
    Obtém as classificações das classes com base nas probabilidades preditas pelo modelo.
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

def are_sequences_aligned(fasta_file):
    """
    Verifica se todas as sequências em um arquivo FASTA têm o mesmo comprimento.
    """
    lengths = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths.add(len(record.seq))
    return len(lengths) == 1  # Retorna True se todas as sequências tiverem o mesmo comprimento

def realign_sequences_with_mafft(input_path, output_path, threads=1):
    """
    Realinha sequências usando o MAFFT.
    """
    mafft_command = ['mafft', '--thread', str(threads), '--maxiterate', '1000', '--localpair', input_path]
    try:
        with open(output_path, "w") as outfile:
            subprocess.run(mafft_command, stdout=outfile, stderr=subprocess.PIPE, check=True)
        logging.info(f"Sequências realinhadas salvas em {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar MAFFT: {e.stderr.decode()}")
        sys.exit(1)

def adjust_predictions_global(predicted_proba, method='normalize', alpha=1.0):
    """
    Ajusta as probabilidades preditas pelo modelo.
    """
    if method == 'normalize':
        # Normaliza as probabilidades para que somem 1 para cada amostra
        logging.info("Normalizando as probabilidades preditas.")
        adjusted_proba = predicted_proba / predicted_proba.sum(axis=1, keepdims=True)
    
    elif method == 'smoothing':
        # Aplica suavização nas probabilidades para evitar valores extremos
        logging.info(f"Aplicando suavização nas probabilidades preditas com alpha={alpha}.")
        adjusted_proba = (predicted_proba + alpha) / (predicted_proba.sum(axis=1, keepdims=True) + alpha * predicted_proba.shape[1])
    
    elif method == 'none':
        # Não aplica nenhum ajuste
        logging.info("Nenhum ajuste aplicado às probabilidades preditas.")
        adjusted_proba = predicted_proba.copy()
    
    else:
        logging.warning(f"Método de ajuste '{method}' desconhecido. Nenhum ajuste será aplicado.")
        adjusted_proba = predicted_proba.copy()
    
    return adjusted_proba

class Support:
    """
    Classe de suporte para treinamento e avaliação de modelos Random Forest com técnicas de oversampling.
    """

    def __init__(self, cv=5, seed=SEED, n_jobs=1):
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
            "max_depth": 5,  # Reduzido para prevenir overfitting
            "min_samples_split": 4,  # Aumentado para prevenir overfitting
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
        
        # Aplicar RandomOverSampler apenas nas classes que têm pelo menos 2 amostras
        ros = RandomOverSampler(random_state=self.seed)
        X_ros, y_ros = ros.fit_resample(X, y)

        # Aplicar SMOTE nas classes que podem ser sintetizadas
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
        logging.info(f"Iniciando o método fit para {model_name_prefix}...")

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
            raise ValueError(f"Há classes com menos membros que o número de folds após o oversampling para {model_name_prefix}.")

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

            self.model = RandomForestClassifier(**self.init_params, n_jobs=1)  # Fix n_jobs=1
            self.model.fit(X_train_resampled, y_train_resampled)

            train_score = self.model.score(X_train_resampled, y_train_resampled)
            test_score = self.model.score(X_test, y_test)

            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

            # Cálculo de F1-score e Precision-Recall AUC
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)

            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)

            if len(np.unique(y_test)) > 1:
                pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')
            else:
                pr_auc = 0.0  # Não é possível calcular PR AUC para uma única classe
            self.pr_auc_scores.append(pr_auc)

            logging.info(f"Fold {fold_number} [{model_name_prefix}]: F1 Score: {f1}")
            logging.info(f"Fold {fold_number} [{model_name_prefix}]: Precision-Recall AUC: {pr_auc}")

            # Cálculo do ROC AUC
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

            # Realizando busca em grade e salvando o melhor modelo
            best_model, best_params = self._perform_grid_search(X_train_resampled, y_train_resampled)
            self.model = best_model
            self.best_params = best_params

            if model_dir:
                best_model_filename = os.path.join(model_dir, f'best_model_{model_name_prefix}.pkl')
                # Garantir que o diretório exista
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

            # Integração da Calibração de Probabilidades
            calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=5, n_jobs=1)  # Fix n_jobs=1
            calibrator.fit(X_train_resampled, y_train_resampled)

            self.model = calibrator

            if model_dir:
                calibrated_model_filename = os.path.join(model_dir, f'calibrated_model_{model_name_prefix}.pkl')
            else:
                calibrated_model_filename = f'calibrated_model_{model_name_prefix}.pkl'
            joblib.dump(calibrator, calibrated_model_filename)
            logging.info(f"Calibrated model saved as {calibrated_model_filename} for {model_name_prefix}")

            fold_number += 1

        return self.model

    def _perform_grid_search(self, X_train_resampled, y_train_resampled):
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=True)
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.seed),
            self.parameters,
            cv=skf,
            n_jobs=1,  # Fix n_jobs=1 para reproducibilidade
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
        plt.title("Learning Curve")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(output_path)
        plt.close()

    def get_class_rankings(self, X):
        """
        Obtém as classificações das classes para os dados fornecidos.
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
        
    def test_best_RF(self, X, y, scaler_dir='.'):
        """
        Testa o melhor modelo Random Forest com os dados fornecidos.
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

        # Aplicar oversampling ao conjunto inteiro antes do split
        X_resampled, y_resampled = self._oversample_single_sample_classes(X_scaled, y)

        # Dividir em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.4, random_state=self.seed, stratify=y_resampled
        )

        # Treinar o RandomForestClassifier com os melhores parâmetros
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
            n_jobs=1  # Fix n_jobs=1 para reproducibilidade
        )
        model.fit(X_train, y_train)  # Fit the model on the training data

        # Integração da Calibração no Modelo de Teste
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5, n_jobs=1)  # Fix n_jobs=1
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
            pr_auc = 0.0  # Não é possível calcular PR AUC para uma única classe

        # Retornar o score, melhores parâmetros, modelo treinado e conjuntos de teste
        return score, f1, pr_auc, self.best_params, calibrated_model, X_test, y_test

    def _calculate_score(self, y_pred, y_test):
        """
        Calcula o score (e.g., ROC AUC) com base nas previsões e rótulos reais.
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
        Plota a curva ROC para classificações binárias ou multiclasse.
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
        self.aggregation_method = aggregation_method  # Adicionado para escolher o método de agregação

    def generate_embeddings(self, k=3, step_size=1, word2vec_model_path="modelo_word2vec.bin", model_dir=None):
        """
        Gera embeddings para as sequências de proteínas usando Word2Vec, padronizando o número de k-mers.
        """
        # Definir o caminho completo do modelo Word2Vec
        if model_dir:
            word2vec_model_full_path = os.path.join(model_dir, word2vec_model_path)
        else:
            word2vec_model_full_path = word2vec_model_path

        # Verificar se o modelo Word2Vec já existe
        if os.path.exists(word2vec_model_full_path):
            logging.info(f"Modelo Word2Vec encontrado em {word2vec_model_full_path}. Carregando o modelo.")
            model = Word2Vec.load(word2vec_model_full_path)
            self.models['global'] = model
        else:
            logging.info("Modelo Word2Vec não encontrado. Treinando um novo modelo.")
            # Inicialização das Variáveis
            kmer_groups = {}
            all_kmers = []
            kmers_counts = []

            # Geração de k-mers
            for record in self.alignment:
                sequence = str(record.seq)
                seq_len = len(sequence)
                protein_accession_alignment = record.id.split()[0]

                # Se a tabela de dados não for fornecida, skip matching
                if self.table_data is not None:
                    matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                    matching_info = self.table_data[matching_rows]

                    if matching_info.empty:
                        logging.warning(f"Nenhuma correspondência na tabela de dados para {protein_accession_alignment}")
                        continue  # Pula para a próxima iteração

                    target_variable = matching_info['Target variable'].values[0]
                    associated_variable = matching_info['Associated variable'].values[0]

                else:
                    # Se não houver tabela, usamos valores padrão ou None
                    target_variable = None
                    associated_variable = None

                logging.info(f"Processando {protein_accession_alignment} com comprimento de sequência {seq_len}")

                if seq_len < k:
                    logging.warning(f"Sequência muito curta para {protein_accession_alignment}. Tamanho: {seq_len}")
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
                    'target_variable': target_variable,
                    'associated_variable': associated_variable,
                    'kmers': kmers  # Armazena os k-mers para uso posterior
                }
                kmer_groups[protein_accession_alignment] = embedding_info

            # Determinar o número mínimo de k-mers
            if not kmers_counts:
                logging.error("Nenhum k-mer foi coletado. Verifique suas sequências e parâmetros de k-mer.")
                sys.exit(1)

            min_kmers = min(kmers_counts)
            logging.info(f"Número mínimo de k-mers em qualquer sequência: {min_kmers}")

            # Treinar modelo Word2Vec usando todos os k-mers
            model = Word2Vec(
                sentences=all_kmers,
                vector_size=125,  # mudar para 100 se necessário
                window=5,
                min_count=1,
                workers=1,  # Fixar workers=1 para reproducibilidade
                sg=1,
                hs=1,  # Hierarchical softmax enabled
                negative=0,  # Negative sampling disabled
                epochs=2500,  # Fixar número de epochs para reproducibilidade
                seed=SEED  # Fixar seed para reproducibilidade
            )

            # Criar diretório para o modelo Word2Vec, se necessário
            os.makedirs(os.path.dirname(word2vec_model_full_path), exist_ok=True)

            # Salvar o modelo Word2Vec
            model.save(word2vec_model_full_path)
            self.models['global'] = model
            logging.info(f"Modelo Word2Vec salvo em {word2vec_model_full_path}")

        # Gerar embeddings padronizados
        kmer_groups = {}
        kmers_counts = []
        all_kmers = []

        for record in self.alignment:
            sequence = str(record.seq)
            protein_accession_alignment = record.id.split()[0]

            # Se a tabela de dados não for fornecida, skip matching
            if self.table_data is not None:
                matching_rows = self.table_data['Protein.accession'].str.split().str[0] == protein_accession_alignment
                matching_info = self.table_data[matching_rows]

                if matching_info.empty:
                    logging.warning(f"Nenhuma correspondência na tabela de dados para {protein_accession_alignment}")
                    continue  # Pula para a próxima iteração

                target_variable = matching_info['Target variable'].values[0]
                associated_variable = matching_info['Associated variable'].values[0]

            else:
                # Se não houver tabela, usamos valores padrão ou None
                target_variable = None
                associated_variable = None

            kmers = [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, step_size)]
            kmers = [kmer for kmer in kmers if kmer.count('-') < k]  # Permite k-mers com menos de k gaps

            if not kmers:
                logging.warning(f"Nenhum k-mer válido para {protein_accession_alignment}")
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

        # Determinar o número mínimo de k-mers
        if not kmers_counts:
            logging.error("Nenhum k-mer foi coletado. Verifique suas sequências e parâmetros de k-mer.")
            sys.exit(1)

        min_kmers = min(kmers_counts)
        logging.info(f"Número mínimo de k-mers em qualquer sequência: {min_kmers}")

        # Gerar embeddings padronizados
        for record in self.alignment:
            sequence_id = record.id
            embedding_info = kmer_groups.get(sequence_id, {})
            kmers_for_protein = embedding_info.get('kmers', [])

            if len(kmers_for_protein) == 0:
                if self.aggregation_method == 'none':
                    embedding_concatenated = np.zeros(self.models['global'].vector_size * min_kmers)
                else:
                    embedding_concatenated = np.zeros(self.models['global'].vector_size)
                self.embeddings.append({
                    'protein_accession': sequence_id,
                    'embedding': embedding_concatenated,
                    'target_variable': embedding_info.get('target_variable'),
                    'associated_variable': embedding_info.get('associated_variable')
                })
                continue

            # Selecionar os primeiros min_kmers k-mers
            selected_kmers = kmers_for_protein[:min_kmers]

            # Padronizar com vetores de zeros se necessário (não deve ser necessário, mas adicionado por segurança)
            if len(selected_kmers) < min_kmers:
                padding = [np.zeros(self.models['global'].vector_size)] * (min_kmers - len(selected_kmers))
                selected_kmers.extend(padding)

            # Obter os embeddings dos k-mers selecionados
            selected_embeddings = [self.models['global'].wv[kmer] for kmer in selected_kmers if kmer in self.models['global'].wv]
            if len(selected_embeddings) < min_kmers:
                padding = [np.zeros(self.models['global'].vector_size)] * (min_kmers - len(selected_embeddings))
                selected_embeddings.extend(padding)

            if self.aggregation_method == 'none':
                # Concatenar os embeddings dos k-mers selecionados
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)
            elif self.aggregation_method == 'mean':
                # Agregar os embeddings dos k-mers selecionados pela média
                embedding_concatenated = np.mean(selected_embeddings, axis=0)
            elif self.aggregation_method == 'median':
                # Agregar os embeddings dos k-mers selecionados pela mediana
                embedding_concatenated = np.median(selected_embeddings, axis=0)
            elif self.aggregation_method == 'sum':
                # Agregar os embeddings dos k-mers selecionados pela soma
                embedding_concatenated = np.sum(selected_embeddings, axis=0)
            elif self.aggregation_method == 'max':
                # Agregar os embeddings dos k-mers selecionados pelo máximo
                embedding_concatenated = np.max(selected_embeddings, axis=0)
            else:
                # Caso não reconheça o método, usar concatenação como default
                logging.warning(f"Método de agregação desconhecido '{self.aggregation_method}'. Usando concatenação.")
                embedding_concatenated = np.concatenate(selected_embeddings, axis=0)

            self.embeddings.append({
                'protein_accession': sequence_id,
                'embedding': embedding_concatenated,
                'target_variable': embedding_info.get('target_variable'),
                'associated_variable': embedding_info.get('associated_variable')
            })

            logging.debug(f"Protein ID: {sequence_id}, Embedding Shape: {embedding_concatenated.shape}")

        # Verificar se todas as embeddings têm a mesma forma
# Ajustar o StandardScaler com os embeddings para treino/predição
        embeddings_array_train = np.array([entry['embedding'] for entry in self.embeddings])

# Verificar se todas as embeddings têm a mesma forma
        embedding_shapes = set(embedding.shape for embedding in [entry['embedding'] for entry in self.embeddings])
        if len(embedding_shapes) != 1:
            logging.error(f"Inconsistent embedding shapes detected: {embedding_shapes}")
            raise ValueError("Embeddings have inconsistent shapes.")
        else:
            logging.info(f"All embeddings have shape: {embedding_shapes.pop()}")

# Definir o caminho completo do scaler
        scaler_full_path = os.path.join(model_dir, 'scaler.pkl') if model_dir else 'scaler.pkl'

# Verificar se o scaler já existe
        if os.path.exists(scaler_full_path):
            logging.info(f"StandardScaler encontrado em {scaler_full_path}. Carregando o scaler.")
            scaler = joblib.load(scaler_full_path)
        else:
            logging.info("StandardScaler não encontrado. Treinando um novo scaler.")
            scaler = StandardScaler().fit(embeddings_array_train)
            logging.info(f"Salvando StandardScaler em {scaler_full_path}")
            joblib.dump(scaler, scaler_full_path)
            logging.info(f"StandardScaler salvo em {scaler_full_path}")

    def get_embeddings_and_labels(self, label_type='target_variable'):
        """
        Retorna os embeddings e os rótulos associados (target_variable ou associated_variable).
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
    """
    model_dir = args.model_dir

    # =============================
    # ETAPA 1: Treinamento do Modelo
    # =============================

    # Carregar dados de treinamento
    train_alignment_path = args.train_fasta
    train_table_data_path = args.train_table

    # Verifica se as sequências de treinamento estão alinhadas
    if not are_sequences_aligned(train_alignment_path):
        logging.info("Sequências de treinamento não estão alinhadas. Realinhando com MAFFT...")
        aligned_train_path = train_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(train_alignment_path, aligned_train_path, threads=1)  # Fixar threads=1
        train_alignment_path = aligned_train_path
    else:
        logging.info(f"Arquivo alinhado de treinamento encontrado ou sequências já alinhadas: {train_alignment_path}")

    # Carregar dados da tabela de treinamento
    train_table_data = pd.read_csv(train_table_data_path, delimiter="\t")
    logging.info("Tabela de dados de treinamento carregada com sucesso.")

    # Inicializar e gerar embeddings para o treinamento
    protein_embedding_train = ProteinEmbeddingGenerator(
        train_alignment_path, 
        train_table_data, 
        aggregation_method=args.aggregation_method  # Passando o método de agregação
    )
    protein_embedding_train.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir
    )
    logging.info(f"Número de embeddings de treinamento gerados: {len(protein_embedding_train.embeddings)}")

    # Obter embeddings e rótulos para target_variable
    X_target, y_target = protein_embedding_train.get_embeddings_and_labels(label_type='target_variable')
    logging.info(f"X_target shape: {X_target.shape}")

    # Caminhos completos dos modelos para target_variable
    rf_model_target_full_path = os.path.join(model_dir, args.rf_model_target)
    calibrated_model_target_full_path = os.path.join(model_dir, 'calibrated_model_target.pkl')

    # Verificar se o modelo calibrado para target_variable já existe
    if os.path.exists(calibrated_model_target_full_path):
        calibrated_model_target = joblib.load(calibrated_model_target_full_path)
        logging.info(f"Calibrated Random Forest model para target_variable carregado de {calibrated_model_target_full_path}")
    else:
        # Treinamento do modelo para target_variable
        support_model_target = Support()
        calibrated_model_target = support_model_target.fit(X_target, y_target, model_name_prefix='target', model_dir=model_dir)
        logging.info("Treinamento e calibração para target_variable concluídos.")

        # Salvar o modelo calibrado
        joblib.dump(calibrated_model_target, calibrated_model_target_full_path)
        logging.info(f"Calibrated Random Forest model for target_variable salvo em {calibrated_model_target_full_path}")

        # Testar o modelo
# Antes:
        #best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, output_dir=args.output_dir)

# Depois:
        best_score, best_f1, best_pr_auc, best_params, best_model_target, X_test_target, y_test_target = support_model_target.test_best_RF(X_target, y_target, output_dir=model_dir)

        
        logging.info(f"Best ROC AUC for target_variable: {best_score}")
        logging.info(f"Best F1 Score for target_variable: {best_f1}")
        logging.info(f"Best Precision-Recall AUC for target_variable: {best_pr_auc}")
        logging.info(f"Best Parameters: {best_params}")

        for param, value in best_params.items():
            logging.info(f"{param}: {value}")

        # Obter rankings de classe
        class_rankings = support_model_target.get_class_rankings(X_test_target)

        # Exibir os rankings para as primeiras 5 amostras
        logging.info("Top 3 class rankings for the first 5 samples:")
        for i in range(min(5, len(class_rankings))):
            logging.info(f"Sample {i+1}: Class rankings - {class_rankings[i][:3]}")  # Mostra as top 3 classificações

        # Plotar a curva ROC
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
    X_associated, y_associated = protein_embedding_train.get_embeddings_and_labels(label_type='associated_variable')
    logging.info(f"X_associated shape: {X_associated.shape}")

    # Caminhos completos dos modelos para associated_variable
    rf_model_associated_full_path = os.path.join(model_dir, args.rf_model_associated)
    calibrated_model_associated_full_path = os.path.join(model_dir, 'calibrated_model_associated.pkl')

    # Verificar se o modelo calibrado para associated_variable já existe
    if os.path.exists(calibrated_model_associated_full_path):
        calibrated_model_associated = joblib.load(calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model para associated_variable carregado de {calibrated_model_associated_full_path}")
    else:
        # Treinamento do modelo para associated_variable
        support_model_associated = Support()
        calibrated_model_associated = support_model_associated.fit(X_associated, y_associated, model_name_prefix='associated', model_dir=model_dir)
        logging.info("Treinamento e calibração para associated_variable concluídos.")
        
        # Plotar a curva de aprendizado
        logging.info("Plotando Learning Curve para Associated variable")
        support_model_associated.plot_learning_curve(args.learning_curve_associated)

        # Salvar o modelo calibrado
        joblib.dump(calibrated_model_associated, calibrated_model_associated_full_path)
        logging.info(f"Calibrated Random Forest model for associated_variable salvo em {calibrated_model_associated_full_path}")

        # Testar o modelo
#        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, output_dir=args.output_dir)
#Depois
        best_score_associated, best_f1_associated, best_pr_auc_associated, best_params_associated, best_model_associated, X_test_associated, y_test_associated = support_model_associated.test_best_RF(X_associated, y_associated, scaler_dir=model_dir)
        logging.info(f"Best ROC AUC for associated_variable in test_best_RF: {best_score_associated}")
        logging.info(f"Best F1 Score for associated_variable in test_best_RF: {best_f1_associated}")
        logging.info(f"Best Precision-Recall AUC for associated_variable in test_best_RF: {best_pr_auc_associated}")
        logging.info(f"Best Parameters found in test_best_RF: {best_params_associated}")
        logging.info(f"Best model Associated in test_best_RF: {best_model_associated}")

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
        joblib.dump(best_model_associated, rf_model_associated_full_path)
        logging.info(f"Random Forest model for associated_variable salvo em {rf_model_associated_full_path}")

        # Plotar a curva ROC para associated_variable
        n_classes_associated = len(np.unique(y_test_associated))
        if n_classes_associated == 2:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)[:, 1]
        else:
            y_pred_proba_associated = best_model_associated.predict_proba(X_test_associated)
            unique_classes_associated = np.unique(y_test_associated).astype(str)
        plot_roc_curve_global(y_test_associated, y_pred_proba_associated, 'ROC Curve for associated_variable', save_as=args.roc_curve_associated, classes=unique_classes_associated)

    # =============================
    # ETAPA 2: Classificação de Novas Sequências
    # =============================

    # Carregar dados para predição
    predict_alignment_path = args.predict_fasta

    # Verifica se as sequências para predição estão alinhadas
    if not are_sequences_aligned(predict_alignment_path):
        logging.info("Sequências para predição não estão alinhadas. Realinhando com MAFFT...")
        aligned_predict_path = predict_alignment_path.replace(".fasta", "_aligned.fasta")
        realign_sequences_with_mafft(predict_alignment_path, aligned_predict_path, threads=1)  # Fixar threads=1
        predict_alignment_path = aligned_predict_path
    else:
        logging.info(f"Arquivo alinhado para predição encontrado ou sequências já alinhadas: {predict_alignment_path}")

    # Inicializar ProteinEmbedding para predição, sem necessidade da tabela
    protein_embedding_predict = ProteinEmbeddingGenerator(
        predict_alignment_path, 
        table_data=None,
        aggregation_method=args.aggregation_method  # Passando o método de agregação
    )
    protein_embedding_predict.generate_embeddings(
        k=args.kmer_size,
        step_size=args.step_size,
        word2vec_model_path=args.word2vec_model,
        model_dir=model_dir
    )
    logging.info(f"Número de embeddings para predição gerados: {len(protein_embedding_predict.embeddings)}")

    # Obter embeddings para predição
    X_predict = np.array([entry['embedding'] for entry in protein_embedding_predict.embeddings])

    # Carregar o scaler
    scaler_full_path = os.path.join(model_dir, args.scaler)
    if os.path.exists(scaler_full_path):
        scaler = joblib.load(scaler_full_path)
        logging.info(f"Scaler carregado de {scaler_full_path}")
    else:
        logging.error(f"Scaler não encontrado em {scaler_full_path}")
        sys.exit(1)
    X_predict_scaled = scaler.transform(X_predict)

    # Realizar predições nas novas sequências

    # Verificar o tamanho das features antes da predição
    if X_predict_scaled.shape[1] > calibrated_model_target.base_estimator_.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_target.base_estimator_.n_features_in_} to match the model input size.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_target.base_estimator_.n_features_in_]

    predictions_target = calibrated_model_target.predict(X_predict_scaled)

    # Verificar e ajustar o tamanho das features para associated_variable
    if X_predict_scaled.shape[1] > calibrated_model_associated.base_estimator_.n_features_in_:
        logging.info(f"Reducing number of features from {X_predict_scaled.shape[1]} to {calibrated_model_associated.base_estimator_.n_features_in_} to match the model input size for associated_variable.")
        X_predict_scaled = X_predict_scaled[:, :calibrated_model_associated.base_estimator_.n_features_in_]

    # Realizar a predição para associated_variable
    predictions_associated = calibrated_model_associated.predict(X_predict_scaled)

    # Obter rankings de classe
    rankings_target = get_class_rankings_global(calibrated_model_target, X_predict_scaled)
    rankings_associated = get_class_rankings_global(calibrated_model_associated, X_predict_scaled)

    # Processar e salvar os resultados
    results = {}
    for entry, pred_target, pred_associated, ranking_target, ranking_associated in zip(protein_embedding_predict.embeddings, predictions_target, predictions_associated, rankings_target, rankings_associated):
        sequence_id = entry['protein_accession']
        results[sequence_id] = {
            "target_prediction": pred_target,
            "associated_prediction": pred_associated,
            "target_ranking": ranking_target,
            "associated_ranking": ranking_associated
        }

    # Salvar os resultados em um arquivo
    with open(args.results_file, 'w') as f:
        f.write("Protein_ID\tTarget_Prediction\tAssociated_Prediction\tTarget_Ranking\tAssociated_Ranking\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result['target_prediction']}\t{result['associated_prediction']}\t{'; '.join(result['target_ranking'])}\t{'; '.join(result['associated_ranking'])}\n")
            logging.info(f"{seq_id} - Target Variable: {result['target_prediction']}, Associated Variable: {result['associated_prediction']}, Target Ranking: {'; '.join(result['target_ranking'])}, Associated Ranking: {'; '.join(result['associated_ranking'])}")

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

    # Gerar o Scatterplot das Previsões
    logging.info("Gerando scatterplot das previsões das novas sequências...")
    plot_predictions_scatterplot_custom(results, args.scatterplot_output)
    logging.info(f"Scatterplot salvo em {args.scatterplot_output}")

    logging.info("Processamento concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise de Embeddings de Proteínas e Classificação com Random Forest")

    # Argumentos de entrada
    parser.add_argument('--train_fasta', required=True, help='Caminho para o arquivo .fasta de treinamento')
    parser.add_argument('--train_table', required=True, help='Caminho para a tabela de dados de treinamento (CSV ou TSV)')
    parser.add_argument('--predict_fasta', required=True, help='Caminho para o arquivo .fasta para classificação')

    # Argumentos para k-mer
    parser.add_argument('--kmer_size', type=int, default=3, help='Tamanho do k-mer (default: 3)')
    parser.add_argument('--step_size', type=int, default=1, help='Tamanho do passo para geração de k-mers (default: 1)')

    # Argumento para método de agregação
    parser.add_argument('--aggregation_method', type=str, choices=['none', 'mean', 'median', 'sum', 'max'], default='none',
                        help='Método de agregação para os embeddings (default: none)')

    # Argumento de saída para resultados principais
    parser.add_argument('-o', '--results_file', required=True, help='Caminho para salvar os resultados de predição (TSV)')
    parser.add_argument('--output_dir', required=True, help='Caminho para o diretório onde os arquivos de saída serão salvos')

    # Argumentos de saída para gráficos e outros arquivos
    parser.add_argument('--accuracy_pie_chart_png', required=False, help='Caminho para salvar o gráfico de pizza de precisão (PNG)')
    parser.add_argument('--accuracy_pie_chart_svg', required=False, help='Caminho para salvar o gráfico de pizza de precisão (SVG)')
    parser.add_argument('--associated_variable_scatterplot_png', required=False, help='Caminho para salvar o scatterplot de variáveis associadas (PNG)')
    parser.add_argument('--associated_variable_scatterplot_svg', required=False, help='Caminho para salvar o scatterplot de variáveis associadas (SVG)')
    
    # Novo argumento para Scatterplot das Previsões
    parser.add_argument('--scatterplot_output', required=True, help='Caminho para salvar o scatterplot das previsões das novas sequências (PNG)')
    parser.add_argument('--excel_output', required=True, help='Caminho para salvar os resultados em Excel')
    parser.add_argument('--formatted_results_table', required=True, help='Caminho para salvar a tabela de resultados formatados (TXT)')
    parser.add_argument('--roc_curve_target', required=True, help='Caminho para salvar a curva ROC para a variável alvo (PNG)')
    parser.add_argument('--roc_curve_associated', required=True, help='Caminho para salvar a curva ROC para a variável associada (PNG)')
    parser.add_argument('--learning_curve_target', required=True, help='Caminho para salvar a curva de aprendizado para a variável alvo (PNG)')
    parser.add_argument('--learning_curve_associated', required=True, help='Caminho para salvar a curva de aprendizado para a variável associada (PNG)')
    parser.add_argument('--roc_values_target', required=True, help='Caminho para salvar os valores ROC para a variável alvo (CSV)')
    parser.add_argument('--rf_model_target', required=True, help='Nome do arquivo para salvar o modelo Random Forest para a variável alvo (PKL)')
    parser.add_argument('--rf_model_associated', required=True, help='Nome do arquivo para salvar o modelo Random Forest para a variável associada (PKL)')
    parser.add_argument('--word2vec_model', required=True, help='Nome do arquivo para salvar o modelo Word2Vec (BIN)')
    parser.add_argument('--scaler', required=True, help='Nome do arquivo para salvar o StandardScaler (PKL)')
    parser.add_argument('--model_dir', required=True, help='Diretório onde os modelos estão localizados ou serão salvos')

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
                    logging.info(f"Diretório criado para {dir_path}")
                output_dirs.add(dir_path)

    # Criar o diretório de modelos, se necessário
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        logging.info(f"Diretório de modelos criado em {args.model_dir}")
    else:
        logging.info(f"Diretório de modelos encontrado: {args.model_dir}")

    main(args)





