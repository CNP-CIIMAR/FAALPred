# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import matplotlib
import re
## OK so para Phylum
# Configure o backend do matplotlib para TkAgg ou Qt5Agg, dependendo do seu sistema
matplotlib.use('TkAgg')  # Ou 'Qt5Agg', dependendo do seu sistema

import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Carrega dados de um arquivo CSV separado por tabulaÃƒÂ§ÃƒÂµes.

    Args:
        file_path (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame ou None: DataFrame com os dados carregados ou None se ocorrer um erro.
    """
    try:
        dataframe = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        print(f"Arquivo '{file_path}' carregado com sucesso.")
        return dataframe
    except FileNotFoundError:
        print(f"Arquivo '{file_path}' nÃƒÂ£o encontrado.")
        return None
    except pd.errors.ParserError:
        print(f"Erro ao analisar o arquivo '{file_path}'. Verifique o formato do arquivo.")
        return None
    except UnicodeDecodeError:
        print(f"Erro de codificaÃƒÂ§ÃƒÂ£o ao ler o arquivo '{file_path}'. Verifique a codificaÃƒÂ§ÃƒÂ£o do arquivo.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None


def filter_data(dataframe, level, names):
    if level not in dataframe.columns:
        print(f"Coluna '{level}' nÃƒÂ£o encontrada no DataFrame.")
        return pd.DataFrame()
    if level == "Phylum":
        return dataframe[dataframe['Taxonomy'].str.contains('|'.join(names), case=False, na=False)]
    return dataframe[dataframe[level].isin(names)]


def adjust_taxonomic_level(dataframe, level):
    """
    Ajusta o nÃƒÂ­vel taxonÃƒÂ´mico especificado no DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados.
        level (str): NÃƒÂ­vel taxonÃƒÂ´mico a ajustar ('Phylum', 'Order', 'Genus').

    Returns:
        pd.DataFrame: DataFrame com o nÃƒÂ­vel taxonÃƒÂ´mico ajustado.
    """
    dataframe['Taxonomy'] = dataframe['Taxonomy'].fillna("Unknown")
    if level == "Phylum":
        dataframe[level] = dataframe['Taxonomy'].apply(correct_phylum)
    elif level == "Order":
        dataframe[level] = dataframe['Taxonomy'].apply(correct_order)
    elif level == "Genus":
        dataframe[level] = dataframe['Taxonomy'].apply(correct_genus)
    else:
        print(f"NÃƒÂ­vel taxonÃƒÂ´mico '{level}' nÃƒÂ£o reconhecido. Nenhuma alteraÃƒÂ§ÃƒÂ£o serÃƒÂ¡ feita.")
    return dataframe


def correct_phylum(taxonomy):
    """
    Extrai e corrige o nÃƒÂ­vel de Phylum a partir da string de taxonomia.

    Args:
        taxonomy (str): String de taxonomia.

    Returns:
        str: Nome do Phylum ou 'Unknown' se nÃƒÂ£o puder ser determinado.
    """
    if not isinstance(taxonomy, str):
        return "Unknown"
    levels = taxonomy.split(',')
    for lvl in levels:
        if "group" not in lvl.lower():
            return lvl.strip()
    return "Unknown"


def correct_order(taxonomy):
    """
    Identifica e retorna o nÃƒÂ­vel de Order na string de taxonomia com base no sufixo 'ales'.

    Args:
        taxonomy (str): String de taxonomia.

    Returns:
        str: Nome do Order ou 'Unknown' se nÃƒÂ£o puder ser determinado.
    """
    if not isinstance(taxonomy, str):
        return "Unknown"
    
    # Dividir os nÃƒÂ­veis por vÃƒÂ­rgula
    levels = taxonomy.split(',')
    
    # Buscar o nÃƒÂ­vel que termina com 'ales'
    for lvl in levels:
        if lvl.strip().endswith('ales'):
            return lvl.strip()
    
    return "Unknown"



def correct_genus(taxonomy):
    """
    Extrai e corrige o nÃƒÂ­vel de Genus a partir da string de taxonomia.

    Args:
        taxonomy (str): String de taxonomia.

    Returns:
        str: Nome do Genus ou 'Unknown' se nÃƒÂ£o puder ser determinado.
    """
    if not isinstance(taxonomy, str):
        return "Unknown"
    levels = taxonomy.split(',')
    # Supondo que Genus estÃƒÂ¡ no terceiro nÃƒÂ­vel
    if len(levels) >= 3:
        return levels[2].strip()
    return "Unknown"


def select_taxonomic_level():
    """
    Permite ao usuÃƒÂ¡rio selecionar o nÃƒÂ­vel taxonÃƒÂ´mico e inserir os nomes correspondentes.

    Returns:
        tuple: NÃƒÂ­vel taxonÃƒÂ´mico selecionado e lista de nomes inseridos pelo usuÃƒÂ¡rio.
    """
    taxonomic_levels = ['Phylum', 'Order', 'Genus']
    print("Selecione o nÃƒÂ­vel taxonÃƒÂ´mico para filtrar:")
    for index, level in enumerate(taxonomic_levels, 1):
        print(f"{index}. {level}")
    choice = input("Digite o nÃƒÂºmero correspondente ao nÃƒÂ­vel taxonÃƒÂ´mico: ")

    try:
        choice_number = int(choice)
        if choice_number < 1 or choice_number > len(taxonomic_levels):
            raise IndexError
        selected_level = taxonomic_levels[choice_number - 1]
        names = input(f"Digite os nomes de {selected_level} separados por vÃƒÂ­rgulas: ")
        names_list = [name.strip() for name in names.split(',') if name.strip()]
        if not names_list:
            print("Nenhum nome vÃƒÂ¡lido foi inserido.")
            sys.exit(1)
        return selected_level, names_list
    except (ValueError, IndexError):
        print("Entrada invÃƒÂ¡lida. Por favor, selecione um nÃƒÂºmero vÃƒÂ¡lido.")
        sys.exit(1)


#def calculate_proportion_and_genomes(dataframe):
#    dataframe = dataframe.copy()  # Garante que estamos trabalhando com uma cÃƒÂ³pia
#    dataframe['Genome_ID'] = dataframe['BGC'].str.extract(
#        r'(^[\w]+[\w]+|^[\w\.]+\.region\d+|NODE[\w\.]+)', expand=False#
#    )

#    proportion = dataframe.groupby('BiG-SCAPE class').size().reset_index(name='Count')
#    total = proportion['Count'].sum()
#    proportion['Proportion'] = proportion['Count'] / total

#    number_of_genomes = dataframe['Genome_ID'].nunique()

#    return proportion, number_of_genomes
    
def calculate_proportion_and_genomes(dataframe):
    """
    Calcula as proporÃƒÂ§ÃƒÂµes das classes BiG-SCAPE e o nÃƒÂºmero de genomas ÃƒÂºnicos com base nos IDs extraÃƒÂ­dos.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados com colunas 'BGC' e 'BiG-SCAPE class'.
    
    Returns:
        tuple: DataFrame com proporÃƒÂ§ÃƒÂµes e nÃƒÂºmero de genomas ÃƒÂºnicos.
    """
    dataframe = dataframe.copy()  # Garante que estamos trabalhando com uma cÃƒÂ³pia

    # ExpressÃƒÂµes regulares separadas
    regex_default = r'(^[\w]+[\w]+|^[\w\.]+\.region\d+|NODE[\w\.]+)'
    regex_bgc = r'(^BGC[\w]+)'

    # Inicializar a coluna Genome_ID com valores vazios
    dataframe['Genome_ID'] = None

    # Aplicar a primeira expressÃƒÂ£o regular
    mask_default = dataframe['BGC'].str.extract(regex_default, expand=False)
    dataframe.loc[~mask_default.isna(), 'Genome_ID'] = mask_default

    # Aplicar a segunda expressÃƒÂ£o regular (BGC) apenas onde Genome_ID ainda estÃƒÂ¡ vazio
    mask_bgc = dataframe['BGC'].str.extract(regex_bgc, expand=False)
    dataframe.loc[dataframe['Genome_ID'].isna() & ~mask_bgc.isna(), 'Genome_ID'] = mask_bgc

    # Calculando proporÃƒÂ§ÃƒÂµes
    proportion = dataframe.groupby('BiG-SCAPE class').size().reset_index(name='Count')
    total = proportion['Count'].sum()
    proportion['Proportion'] = proportion['Count'] / total

    # Calculando o nÃƒÂºmero de genomas ÃƒÂºnicos
    number_of_genomes = dataframe['Genome_ID'].nunique()

    return proportion, number_of_genomes



def remove_low_proportion_classes(dataframe, taxon_column):
    """
    Remove classes especÃƒÂ­ficas ('Saccharides', 'Ripps', 'Terpenes') do DataFrame
    se sua proporÃƒÂ§ÃƒÂ£o for menor que 0.1%.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados.
        taxon_column (str): Nome da coluna contendo as classes a serem filtradas.

    Returns:
        tuple: DataFrame filtrado e DataFrame removido.
    """
    pattern = r"\b(?:Saccharides|Ripps|Terpenes)\b"
    if 'Proportion' not in dataframe.columns:
        print("Coluna 'Proportion' nÃƒÂ£o encontrada. Pulando filtragem.")
        return dataframe, pd.DataFrame()

    removal_condition = (
        dataframe[taxon_column].str.contains(pattern, case=False, na=False, regex=True) &
        (dataframe['Proportion'] < 0.001)  # Menor que 0.1%
    )
    return dataframe[~removal_condition], dataframe[removal_condition]
def plot_pie_chart(filtered_dataframe, level, taxon_names, num_genomes, color_mapping, ax=None):
    """
    Gera um grÃƒÂ¡fico de pizza baseado no DataFrame filtrado.
    - Fatias Ã¢â€°Â¥5%:
        - Porcentagem exibida dentro da fatia.
        - DescriÃƒÂ§ÃƒÂ£o movida para fora, mas sem caixinhas.
    - Fatias entre 0.1% e 2%:
        - DescriÃƒÂ§ÃƒÂ£o e porcentagem exibidas externamente em caixas conectadas por setas.

    Args:
        filtered_dataframe (pd.DataFrame): DataFrame filtrado para plotagem.
        level (str): NÃƒÂ­vel taxonÃƒÂ´mico usado para filtragem.
        taxon_names (list): Lista de nomes taxonÃƒÂ´micos inseridos pelo usuÃƒÂ¡rio.
        num_genomes (int): NÃƒÂºmero total de genomas.
        color_mapping (dict): Mapeamento de cores para cada classe BiG-SCAPE.
        ax (matplotlib.axes.Axes, optional): Eixo onde o grÃƒÂ¡fico serÃƒÂ¡ plotado.
    """
    proportion = filtered_dataframe.groupby('BiG-SCAPE class').size().reset_index(name='Count')
    proportion['Proportion'] = proportion['Count'] / proportion['Count'].sum()

    if proportion.empty:
        print("Nenhum dado disponÃƒÂ­vel para o grÃƒÂ¡fico de pizza.")
        return

    labels = proportion['BiG-SCAPE class']
    sizes = proportion['Proportion']

    # Atribuir cores com base no mapeamento
    colors = labels.map(color_mapping)

    # Se nenhum eixo for fornecido, criar um novo
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Criar o grÃƒÂ¡fico de pizza
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # Sem labels no pie
        colors=colors,
        startangle=140,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Exibir porcentagens > 5% dentro das fatias
        pctdistance=0.6,  # DistÃƒÂ¢ncia das etiquetas internas do centro
        wedgeprops=dict(edgecolor='w', linewidth=1.2),
        textprops=dict(color="white", fontsize=10),
    )

    # Ajustar porcentagens para que fiquem bem posicionadas e legÃƒÂ­veis
    for autotext in autotexts:
        autotext.set_fontsize(8)
        if autotext.get_text() == '':
            autotext.set_visible(False)  # Ocultar porcentagens nÃƒÂ£o relevantes

    # Adicionar descriÃƒÂ§ÃƒÂµes externas para fatias >=5%
    annotations = []
    for idx, (label, size, wedge) in enumerate(zip(labels, sizes, wedges)):
        if size >= 0.05:
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))

            # Posicionar descriÃƒÂ§ÃƒÂ£o do lado externo, sem caixinha
            annotation = ax.text(
                1.2 * x, 1.2 * y, label,
                horizontalalignment="center" if x > 0 else "right",
                verticalalignment="center",
                fontsize=10, color="black"
            )
            annotations.append(annotation)

    # Adicionar caixas externas para fatias entre 0.1% e 2%
    external_annotations = []
    for idx, (label, size, wedge) in enumerate(zip(labels, sizes, wedges)):
        if 0.001 <= size < 0.05:
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            horizontal_alignment = "left" if x > 0 else "right"

            # Calcular posiÃƒÂ§ÃƒÂ£o para a caixa externa com ajuste adicional de seta
            text_x = 1.4 * np.sign(x)
            text_y = 1.4 * y

            # Ajustar direÃƒÂ§ÃƒÂ£o da seta com base no ÃƒÂ¢ngulo
            arrow_offset = 0.2 if y > 0 else -0.2
            annotation = ax.annotate(
                f"{label} ({size * 100:.1f}%)",
                xy=(x, y),
                xytext=(text_x, text_y + arrow_offset),
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=1.0,
                    connectionstyle="arc3,rad=0.2",  # Adicionar curva suave
                    color='black'
                ),
                horizontalalignment=horizontal_alignment,
                verticalalignment='center',
                fontsize=8,
            )
            external_annotations.append(annotation)

    # Ajustar posiÃƒÂ§ÃƒÂµes das anotaÃƒÂ§ÃƒÂµes externas para evitar sobreposiÃƒÂ§ÃƒÂµes
    adjust_annotations(external_annotations, spacing=0.2)

    # Configurar o tÃƒÂ­tulo personalizado com o nome do taxon e nÃƒÂºmero total de genomas
    taxon_name_str = ', '.join(taxon_names)
    ax.set_title(
        f"{taxon_name_str}\nTotal Genomes: {num_genomes}",
        fontsize=12,
        weight="bold",
        pad=10,        # Ajustar a distÃƒÂ¢ncia do tÃƒÂ­tulo
        loc='center'   # Centralizar o tÃƒÂ­tulo
    )

    # Garantir que o grÃƒÂ¡fico de pizza seja desenhado como um cÃƒÂ­rculo
    ax.axis('equal')

def adjust_annotations(annotations, spacing=0.2):
    """
    Ajusta manualmente as posiÃƒÂ§ÃƒÂµes das anotaÃƒÂ§ÃƒÂµes externas para evitar sobreposiÃƒÂ§ÃƒÂµes.

    Args:
        annotations (list): Lista de objetos de anotaÃƒÂ§ÃƒÂ£o a serem ajustados.
        spacing (float): EspaÃƒÂ§amento mÃƒÂ­nimo entre as linhas das anotaÃƒÂ§ÃƒÂµes.
    """
    if not annotations:
        return

    # Ordenar anotaÃƒÂ§ÃƒÂµes externas por y para ajustar de cima para baixo
    annotations_sorted = sorted(annotations, key=lambda ann: ann.xyann[1], reverse=True)

    for i in range(1, len(annotations_sorted)):
        prev = annotations_sorted[i - 1].xyann[1]
        current = annotations_sorted[i].xyann[1]
        if abs(current - prev) < spacing:
            # Ajustar a posiÃƒÂ§ÃƒÂ£o y da anotaÃƒÂ§ÃƒÂ£o atual
            x, y = annotations_sorted[i].xyann
            if y < 0:
                y_new = y - spacing
            else:
                y_new = y + spacing
            # Limitar a posiÃƒÂ§ÃƒÂ£o y para nÃƒÂ£o sair dos limites do grÃƒÂ¡fico
            y_new = max(min(y_new, 1.5), -1.5)
            annotations_sorted[i].xyann = (x, y_new)

def main():
    """
    FunÃƒÂ§ÃƒÂ£o principal que coordena o fluxo do script.
    """
    if len(sys.argv) < 2:
        print("Uso: python3 script.py <caminho_para_o_arquivo>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataframe = load_data(file_path)

    if dataframe is not None:
        # Verificar se as colunas necessÃƒÂ¡rias existem
        required_columns = ['Taxonomy', 'BiG-SCAPE class', 'BGC']
        for column in required_columns:
            if column not in dataframe.columns:
                print(f"Coluna '{column}' nÃƒÂ£o encontrada no arquivo. Verifique o arquivo de entrada.")
                sys.exit(1)
        level, taxon_names = select_taxonomic_level()
        dataframe = adjust_taxonomic_level(dataframe, level)

        proportion_list = []
        genome_number_list = []

        # Filtrar dados para todos os taxons selecionados primeiro
        filtered_data = []
        genome_number_list = []
        all_classes = set()  # Coletar todas as classes ÃƒÂºnicas

        for taxon_name in taxon_names:
            filtered_df = filter_data(dataframe, level, [taxon_name])

            if filtered_df.empty:
                print(f"Nenhum dado encontrado para {taxon_name}.")
                continue

            proportion, num_genomes = calculate_proportion_and_genomes(filtered_df)

            # Mesclar em uma nova variÃƒÂ¡vel para evitar sobrescrever filtered_df
            filtered_df_with_proportion = filtered_df.merge(
                proportion[['BiG-SCAPE class', 'Proportion']],
                on='BiG-SCAPE class',
                how='left'
            )

            # Aplicar remoÃƒÂ§ÃƒÂ£o na coluna 'BiG-SCAPE class'
            final_filtered_df, removed_df = remove_low_proportion_classes(
                filtered_df_with_proportion,
                'BiG-SCAPE class'
            )

            print("\nClasses Removidas (ProporÃƒÂ§ÃƒÂ£o < 0,1%):")
            if not removed_df.empty:
                print(
                    removed_df[['BiG-SCAPE class', 'Proportion']].drop_duplicates()
                )
            else:
                print("Nenhuma classe foi removida.")

            print("\nDataFrame Filtrado (Apenas Classes VÃƒÂ¡lidas):")
            print(final_filtered_df.head())  # Exibir as primeiras linhas para verificaÃƒÂ§ÃƒÂ£o

            # Armazenar dados para plotagem
            filtered_data.append((final_filtered_df, level, [taxon_name], num_genomes))
            genome_number_list.append(num_genomes)

            # Coletar todas as classes ÃƒÂºnicas
            all_classes.update(proportion['BiG-SCAPE class'].unique())

        if not all_classes:
            print("Nenhuma classe encontrada para mapeamento de cores.")
            sys.exit(1)

        # Criar um mapeamento de cores para cada classe usando a paleta viridis
        sorted_classes = sorted(all_classes)
        num_classes = len(sorted_classes)
        cmap = plt.get_cmap('viridis')  # MantÃƒÂ©m a paleta viridis
        if num_classes > cmap.N:
            # Se houver mais classes do que cores disponÃƒÂ­veis na paleta, distribuir cores igualmente
            colors = [cmap(i / num_classes) for i in range(num_classes)]
        else:
            # Distribuir cores uniformemente pela paleta
            if num_classes > 1:
                colors = [cmap(i / (num_classes - 1)) for i in range(num_classes)]
            else:
                colors = [cmap(0.5)]  # Escolher uma cor central se houver apenas uma classe
        color_mapping = dict(zip(sorted_classes, colors))

        # Determinar o nÃƒÂºmero de taxons vÃƒÂ¡lidos para plotagem
        num_pies = len(filtered_data)
        if num_pies == 0:
            print("Nenhum dado vÃƒÂ¡lido para plotagem.")
            sys.exit(1)

        # Definir o nÃƒÂºmero de colunas como 3
        num_cols = 3
        num_rows = (num_pies + num_cols - 1) // num_cols

        # Ajustar as dimensÃƒÂµes da figura para caber em uma pÃƒÂ¡gina do Word
        # Considerando que uma pÃƒÂ¡gina do Word tem cerca de 6 polegadas de largura utilizÃƒÂ¡vel
        fig_width = 18  # 6 polegadas por subplot
        fig_height = 6 * num_rows  # 6 polegadas por linha

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.5, wspace=0.3)  # Ajuste de margens e espaÃƒÂ§amentos

        # Garantir que 'axes' seja um array 1D
        if num_pies == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()

        # Plotar cada grÃƒÂ¡fico de pizza no subplot correspondente
        for idx, (data, plot_level, taxon_names_plot, num_genomes_plot) in enumerate(filtered_data):
            if idx < len(axes):
                ax = axes[idx]
                plot_pie_chart(data, plot_level, taxon_names_plot, num_genomes_plot, color_mapping, ax=ax)

        # Remover subplots vazios, se houver
        total_subplots = num_rows * num_cols
        if total_subplots > num_pies:
            for idx in range(num_pies, total_subplots):
                fig.delaxes(axes[idx])

        # Ajustar layout antes de salvar
        plt.tight_layout()

        # Salvar a figura nos formatos JPEG, SVG e PNG com 300 DPI
        plt.savefig('pie_charts.png', dpi=300, format='png')
        plt.savefig('pie_charts.jpeg', dpi=300, format='jpeg')
        plt.savefig('pie_charts.svg', dpi=300, format='svg')

        # Exibir a figura
        plt.show()


if __name__ == "__main__":
    main()
