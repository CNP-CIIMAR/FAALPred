# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import matplotlib
import re

## Example: python3 pie_bgc_classe.py merged_Network_Annotations_Full_modificada__clans_0.30_0.70_final.tsv
## Later: Cyanobacteriota, Myxococcota, Actinomycetota, Planctomycetota, Nitrospirota, Deltaproteobacteria, Acidobacteriota, Thermodesulfobacteriota, Chloroflexota, Gemmatimonadota, Pseudomonadota, Verrucomicrobiota, Bacteroidota, Bacillota

# Configura o backend do matplotlib para TkAgg ou Qt5Agg, dependendo do seu sistema
matplotlib.use('TkAgg')  # Ou 'Qt5Agg', dependendo do seu sistema

import matplotlib.pyplot as plt


def carregar_dados(caminho_arquivo):
    """
    Carrega os dados de um arquivo CSV separado por tabulaÃ§Ãµes.

    Args:
        caminho_arquivo (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame or None: DataFrame com os dados carregados ou None se ocorrer um erro.
    """
    try:
        dataframe = pd.read_csv(caminho_arquivo, sep='\t')
        print(f"Arquivo '{caminho_arquivo}' carregado com sucesso.")
        return dataframe
    except FileNotFoundError:
        print(f"Arquivo '{caminho_arquivo}' nÃ£o encontrado.")
        return None
    except pd.errors.ParserError:
        print(f"Erro ao analisar o arquivo '{caminho_arquivo}'. Verifique o formato do arquivo.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None


def filtrar_dados(dataframe, nivel, nomes):
    if nivel not in dataframe.columns:
        print(f"Coluna '{nivel}' nÃ£o encontrada no DataFrame.")
        return pd.DataFrame()
    if nivel == "Phylum":
        return dataframe[dataframe['Taxonomy'].str.contains('|'.join(nomes), case=False, na=False)]
    return dataframe[dataframe[nivel].isin(nomes)]


def ajustar_nivel_taxonomico(dataframe, nivel):
    """
    Ajusta o nÃ­vel taxonÃ´mico especificado no DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados.
        nivel (str): NÃ­vel taxonÃ´mico a ser ajustado ('Phylum', 'Order', 'Genus').

    Returns:
        pd.DataFrame: DataFrame com o nÃ­vel taxonÃ´mico ajustado.
    """
    dataframe['Taxonomy'] = dataframe['Taxonomy'].fillna("Unknown")
    if nivel == "Phylum":
        dataframe[nivel] = dataframe['Taxonomy'].apply(corrigir_phylum)
    elif nivel == "Order":
        dataframe[nivel] = dataframe['Taxonomy'].apply(corrigir_order)
    elif nivel == "Genus":
        dataframe[nivel] = dataframe['Taxonomy'].apply(corrigir_genus)
    else:
        print(f"NÃ­vel taxonÃ´mico '{nivel}' nÃ£o reconhecido. Nenhuma alteraÃ§Ã£o serÃ¡ feita.")
    return dataframe


def corrigir_phylum(taxonomy):
    """
    Extrai e corrige o nÃ­vel de Filo (Phylum) da taxonomia.

    Args:
        taxonomy (str): String de taxonomia.

    Returns:
        str: Nome do Filo ou 'Unknown' se nÃ£o puder ser determinado.
    """
    if not isinstance(taxonomy, str):
        return "Unknown"
    niveis = taxonomy.split(',')
    for nivel in niveis:
        if "group" not in nivel.lower():
            return nivel.strip()
    return "Unknown"


def corrigir_order(taxonomy):
    """
    Extrai e corrige o nÃ­vel de Ordem (Order) da taxonomia.

    Args:
        taxonomy (str): String de taxonomia.

    Returns:
        str: Nome da Ordem ou 'Unknown' se nÃ£o puder ser determinado.
    """
    if not isinstance(taxonomy, str):
        return "Unknown"
    niveis = taxonomy.split(',')
    # Supondo que a Ordem esteja no segundo nÃ­vel
    if len(niveis) >= 2:
        return niveis[1].strip()
    return "Unknown"


def corrigir_genus(taxonomy):
    """
    Extrai e corrige o nÃ­vel de GÃªnero (Genus) da taxonomia.

    Args:
        taxonomy (str): String de taxonomia.

    Returns:
        str: Nome do GÃªnero ou 'Unknown' se nÃ£o puder ser determinado.
    """
    if not isinstance(taxonomy, str):
        return "Unknown"
    niveis = taxonomy.split(',')
    # Supondo que o GÃªnero esteja no terceiro nÃ­vel
    if len(niveis) >= 3:
        return niveis[2].strip()
    return "Unknown"


def selecionar_nivel_taxonomico():
    """
    Permite que o usuÃ¡rio selecione o nÃ­vel taxonÃ´mico e insira os nomes correspondentes.

    Returns:
        tuple: NÃ­vel taxonÃ´mico selecionado e lista de nomes inseridos pelo usuÃ¡rio.
    """
    niveis_taxonomicos = ['Phylum', 'Order', 'Genus']
    print("Selecione o nÃ­vel taxonÃ´mico para filtrar:")
    for indice, nivel in enumerate(niveis_taxonomicos, 1):
        print(f"{indice}. {nivel}")
    escolha = input("Digite o nÃºmero correspondente ao nÃ­vel taxonÃ´mico: ")

    try:
        escolha_numero = int(escolha)
        if escolha_numero < 1 or escolha_numero > len(niveis_taxonomicos):
            raise IndexError
        nivel_selecionado = niveis_taxonomicos[escolha_numero - 1]
        nomes = input(f"Digite os nomes de {nivel_selecionado} separados por vÃ­rgulas: ")
        nomes_lista = [nome.strip() for nome in nomes.split(',') if nome.strip()]
        if not nomes_lista:
            print("Nenhum nome vÃ¡lido foi inserido.")
            sys.exit(1)
        return nivel_selecionado, nomes_lista
    except (ValueError, IndexError):
        print("Entrada invÃ¡lida. Por favor, selecione um nÃºmero vÃ¡lido.")
        sys.exit(1)


def calcular_proporcao_e_genomas(dataframe):
    dataframe = dataframe.copy()  # Garantir que estamos trabalhando com uma cÃ³pia
    dataframe['Genome_ID'] = dataframe['BGC'].str.extract(
        r'(^[\w]+[\w]+|^[\w\.]+\.region\d+|NODE[\w\.]+)', expand=False
    )

    proporcao = dataframe.groupby('BiG-SCAPE class').size().reset_index(name='Count')
    total = proporcao['Count'].sum()
    proporcao['Proportion'] = proporcao['Count'] / total

    numero_genomas = dataframe['Genome_ID'].nunique()

    return proporcao, numero_genomas


def remover_classes_proporcao_zero(dataframe, coluna_taxon):
    """
    Remove classes especÃ­ficas ('Saccharides', 'Ripps', 'Terpenes') do DataFrame
    se a proporÃ§Ã£o associada for menor que 0.1%.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados.
        coluna_taxon (str): Nome da coluna que contÃ©m as classes a serem filtradas.

    Returns:
        tuple: DataFrame filtrado e DataFrame removido.
    """
    padrao = r"\b(?:Saccharides|Ripps|Terpenes)\b"
    if 'Proportion' not in dataframe.columns:
        print("Coluna 'Proportion' nÃ£o encontrada. Pulando filtragem.")
        return dataframe, pd.DataFrame()

    condicao_remocao = (
        dataframe[coluna_taxon].str.contains(padrao, case=False, na=False, regex=True) &
        (dataframe['Proportion'] < 0.001)  # Menor que 0.1%
    )
    return dataframe[~condicao_remocao], dataframe[condicao_remocao]


def plotar_grafico_pizza(dataframe_filtrado, nivel, nomes_taxon, num_genomas, color_mapping, ax=None):
    """
    Gera o grÃ¡fico de pizza com base no DataFrame filtrado.
    Adiciona caixas externas para valores entre 0.1% e 2%.
    Evita sobreposiÃ§Ã£o de rÃ³tulos e duplicidade.

    Args:
        dataframe_filtrado (pd.DataFrame): DataFrame filtrado para plotagem.
        nivel (str): NÃ­vel taxonÃ´mico utilizado para filtragem.
        nomes_taxon (list): Lista de nomes taxonÃ´micos inseridos pelo usuÃ¡rio.
        num_genomas (int): NÃºmero total de genomas.
        color_mapping (dict): Mapeamento de cores para cada classe de BiG-SCAPE.
        ax (matplotlib.axes.Axes, optional): Eixo onde o grÃ¡fico serÃ¡ plotado.
    """
    proporcao = dataframe_filtrado.groupby('BiG-SCAPE class').size().reset_index(name='Count')
    proporcao['Proportion'] = proporcao['Count'] / proporcao['Count'].sum()

    if proporcao.empty:
        print("Nenhum dado disponÃ­vel para o grÃ¡fico de pizza.")
        return

    rotulos = proporcao['BiG-SCAPE class']
    tamanhos = proporcao['Proportion']

    # Atribuir cores com base no mapeamento
    cores = rotulos.map(color_mapping)

    # Se nenhum eixo foi fornecido, crie um novo
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # CriaÃ§Ã£o do grÃ¡fico de pizza
    fatias, textos, autotextos = ax.pie(
        tamanhos,
        labels=[
            rotulo if not (rotulo in ['Ripps', 'Saccharides', 'Terpenes'] and np.isclose(tamanho, 0.0)) and not (0.001 <= tamanho <= 0.02) else ""
            for rotulo, tamanho in zip(rotulos, tamanhos)
        ],
        colors=cores,
        startangle=140,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
        wedgeprops=dict(edgecolor='w', linewidth=1.2),
    )

    # Aumentar o tamanho da fonte dos rÃ³tulos do grÃ¡fico
    for text in textos:
        text.set_fontsize(14)

    # A fonte dentro do pie (autotextos) fica branca e com fonte maior
    for autotext in autotextos:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    deslocamento_vertical = 0  # Controla a posiÃ§Ã£o das caixas para evitar sobreposiÃ§Ã£o

    # Adicionar caixas externas para valores entre 0.1% e 2%
    for idx, (rotulo, tamanho) in enumerate(zip(rotulos, tamanhos)):
        if 0.001 <= tamanho <= 0.02:  # Valores entre 0.1% e 2%
            angulo = (fatias[idx].theta2 - fatias[idx].theta1) / 2. + fatias[idx].theta1
            x = np.cos(np.deg2rad(angulo))
            y = np.sin(np.deg2rad(angulo))
            alinhamento_horizontal = "left" if x > 0 else "right"

            ax.annotate(
                f"{rotulo}: {tamanho * 100:.1f}%",
                xy=(x, y),
                xytext=(1.5 * np.sign(x), 1.5 * y + deslocamento_vertical),
                arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90"),
                horizontalalignment=alinhamento_horizontal,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
                fontsize=9,
            )

            deslocamento_vertical += 0.3  # Incrementa para evitar sobreposiÃ§Ã£o

    # ConfiguraÃ§Ã£o do tÃ­tulo personalizado sem o nÃ­vel taxonÃ´mico, alinhado Ã  direita
    nome_taxon_str = ', '.join(nomes_taxon)
    ax.set_title(
        f"{nome_taxon_str}, Total Genomes: {num_genomas}",
        fontsize=16,  # Aumentado o tamanho do tÃ­tulo
        weight="bold",
        pad=30,        # MantÃ©m a distÃ¢ncia
        loc='right'    # Alinhamento Ã  direita
    )

    # Se nenhum eixo foi fornecido, ajustar o layout e mostrar
    if ax is None:
        plt.tight_layout()
        plt.show()


def main():
    """
    FunÃ§Ã£o principal que coordena o fluxo do script.
    """
    if len(sys.argv) < 2:
        print("Uso: python3 script.py <caminho_para_o_arquivo>")
        sys.exit(1)

    caminho_arquivo = sys.argv[1]
    dataframe = carregar_dados(caminho_arquivo)

    if dataframe is not None:
        # Verificar se as colunas necessÃ¡rias existem
        colunas_necessarias = ['Taxonomy', 'BiG-SCAPE class', 'BGC']
        for coluna in colunas_necessarias:
            if coluna not in dataframe.columns:
                print(f"Coluna '{coluna}' nÃ£o encontrada no arquivo. Por favor, verifique o arquivo de entrada.")
                sys.exit(1)
        nivel, nomes_taxon = selecionar_nivel_taxonomico()
        dataframe = ajustar_nivel_taxonomico(dataframe, nivel)

        lista_proporcoes = []
        lista_numero_genomas = []

        # Filtrar dados para todos os taxons selecionados primeiro
        dados_filtrados = []
        numero_genomas_list = []
        all_classes = set()  # Coletar todas as classes Ãºnicas

        for nome_taxon in nomes_taxon:
            dataframe_filtrado = filtrar_dados(dataframe, nivel, [nome_taxon])

            if dataframe_filtrado.empty:
                print(f"No data found for {nome_taxon}.")
                continue

            proporcao, numero_genomas = calcular_proporcao_e_genomas(dataframe_filtrado)

            # Realizar merge em uma nova variÃ¡vel para evitar sobrescrever dataframe_filtrado
            dataframe_filtrado_com_proporcao = dataframe_filtrado.merge(
                proporcao[['BiG-SCAPE class', 'Proportion']],
                on='BiG-SCAPE class',
                how='left'
            )

            # Aplicar a remoÃ§Ã£o na coluna 'BiG-SCAPE class'
            dataframe_filtrado_final, dataframe_removido = remover_classes_proporcao_zero(
                dataframe_filtrado_com_proporcao,
                'BiG-SCAPE class'
            )

            print("\nClasses Removidas (ProporÃ§Ã£o < 0.1%):")
            if not dataframe_removido.empty:
                print(
                    dataframe_removido[['BiG-SCAPE class', 'Proportion']].drop_duplicates()
                )
            else:
                print("Nenhuma classe foi removida.")

            print("\nDataFrame Filtrado (Apenas Classes VÃ¡lidas):")
            print(dataframe_filtrado_final.head())  # Exibe as primeiras linhas para verificaÃ§Ã£o

            # Armazenar os dados para plotagem
            dados_filtrados.append((dataframe_filtrado_final, nivel, [nome_taxon], numero_genomas))
            numero_genomas_list.append(numero_genomas)

            # Coletar todas as classes Ãºnicas
            all_classes.update(proporcao['BiG-SCAPE class'].unique())

        # Criar um mapeamento de cores para cada classe usando a paleta viridis
        sorted_classes = sorted(all_classes)
        num_classes = len(sorted_classes)
        cmap = plt.get_cmap('viridis')  # Usar a paleta viridis
        if num_classes > cmap.N:
            # Se houver mais classes do que cores disponÃ­veis na paleta, distribuir as cores igualmente
            colors = [cmap(i / num_classes) for i in range(num_classes)]
        else:
            # Distribuir as cores uniformemente pela paleta
            if num_classes > 1:
                colors = [cmap(i / (num_classes - 1)) for i in range(num_classes)]
            else:
                colors = [cmap(0.5)]  # Escolher uma cor central se houver apenas uma classe
        color_mapping = dict(zip(sorted_classes, colors))

        # Determinar o nÃºmero de taxons vÃ¡lidos para plotagem
        num_pies = len(dados_filtrados)
        if num_pies == 0:
            print("Nenhum dado vÃ¡lido para plotagem.")
            sys.exit(1)

        # Calcular o nÃºmero de colunas e linhas para os subplots
        num_cols = min(3, num_pies)  # AtÃ© 3 colunas para evitar subplots muito largos
        num_rows = (num_pies + num_cols - 1) // num_cols

        # Criar a figura e os subplots com espaÃ§amento adicional entre as linhas
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
        plt.subplots_adjust(hspace=0.6)  # Aumentado para maior espaÃ§amento vertical entre os subplots

        # Garantir que 'axes' seja um array 1D
        if num_pies == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()

        # Plotar cada grÃ¡fico de pizza no subplot correspondente
        for idx, (dados, nivel_plot, nomes_taxon_plot, num_genomas_plot) in enumerate(dados_filtrados):
            if idx < len(axes):
                ax = axes[idx]
                plotar_grafico_pizza(dados, nivel_plot, nomes_taxon_plot, num_genomas_plot, color_mapping, ax=ax)

        # Remover subplots vazios, se houver
        total_subplots = num_rows * num_cols
        if total_subplots > num_pies:
            for idx in range(num_pies, total_subplots):
                fig.delaxes(axes[idx])

        # Ajustar layout antes de salvar
        plt.tight_layout()

        # Salvar a figura em JPEG, SVG e PNG com 300 DPI
        plt.savefig('pie_charts.png', dpi=300, format='png')
        plt.savefig('pie_charts.jpeg', dpi=300, format='jpeg')
        plt.savefig('pie_charts.svg', dpi=300, format='svg')

        # Exibir a figura
        plt.show()


if __name__ == "__main__":
    main()

