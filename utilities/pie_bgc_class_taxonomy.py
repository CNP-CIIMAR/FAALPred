# -*- coding: utf-8 -*-
import pandas as pd
import sys
import matplotlib
import re   # Para expressões regulares
from ete3 import NCBITaxa
import matplotlib.pyplot as plt
import numpy as np  # Importação necessária para funções do numpy
import math

#python3 bgc_class_bigscape.py bigscape_update.tsv

# Configurar o backend do matplotlib para TkAgg
matplotlib.use('TkAgg')

# Inicializar NCBITaxa para consulta taxonômica via Ete3
ncbi = NCBITaxa()

def load_data(file_path):
    """
    Carrega o arquivo de dados (TSV) e retorna um DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', low_memory=False)
        print(f"Arquivo '{file_path}' carregado com sucesso.")
        return df
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        sys.exit(1)

def extract_species_from_taxonomy(taxonomy):
    """
    Extrai a espécie a partir do último nível da string 'Taxonomy'.
    Usa vírgulas ou ponto e vírgula como separadores.
    """
    if not isinstance(taxonomy, str):
        return None
    taxonomy = taxonomy.replace(';', ',')
    tokens = [tok.strip() for tok in taxonomy.split(',') if tok.strip()]
    return tokens[-1] if tokens else None

def extract_taxonomic_group_ete3(taxonomy, level):
    """
    Extrai o nível taxonômico (ex.: phylum ou genus) utilizando a biblioteca Ete3.
    A partir da última parte da Taxonomy (espécie), extrai o gênero e consulta o NCBI.
    """
    try:
        species = extract_species_from_taxonomy(taxonomy)
        if not species:
            return None
        tokens = species.split()
        if len(tokens) < 1:
            return None
        genus = tokens[0]
        taxid_dict = ncbi.get_name_translator([genus])
        if genus in taxid_dict:
            taxid = taxid_dict[genus][0]
            return extract_level_from_taxid(taxid, level)
    except Exception as e:
        print(f"Erro ao extrair {level}: {e}")
        return None

def extract_level_from_taxid(taxid, level):
    """
    A partir do taxid, retorna o nome do nível taxonômico especificado.
    """
    try:
        lineage = ncbi.get_lineage(taxid)
        ranks = ncbi.get_rank(lineage)
        names = ncbi.get_taxid_translator(lineage)
        for tid in lineage:
            if ranks[tid] == level.lower():
                return names[tid]
    except Exception as e:
        print(f"Erro ao processar taxid {taxid}: {e}")
    return None

def extract_order(taxonomy, genus):
    """
    Extrai o nível "Order" da coluna Taxonomy seguindo as regras:
      1. Procura por um termo que termine com "ales".
      2. Se não encontrar, utiliza o penúltimo termo (após dividir a string por vírgulas).
      3. Se não houver, utiliza o Genus e consulta o NCBI via Ete3.
    """
    if not isinstance(taxonomy, str):
        return None
    taxonomy = taxonomy.replace(';', ',')
    terms = [term.strip() for term in taxonomy.split(',') if term.strip()]
    for term in reversed(terms):
        if term.lower().endswith("ales"):
            return term
    if len(terms) >= 2:
        return terms[-2]
    if genus:
        try:
            taxid_dict = ncbi.get_name_translator([genus])
            if genus in taxid_dict:
                taxid = taxid_dict[genus][0]
                lineage = ncbi.get_lineage(taxid)
                ranks = ncbi.get_rank(lineage)
                names = ncbi.get_taxid_translator(lineage)
                for tid in lineage:
                    if ranks[tid] == "order":
                        return names[tid]
        except Exception as e:
            print(f"Erro ao extrair Order para o gênero {genus}: {e}")
    return None

def extract_phylum(taxonomy):
    """
    Extrai o Phylum a partir da coluna Taxonomy.
    
    Lógica:
      - A string é separada por vírgulas ou ponto e vírgula.
      - O primeiro token é o domínio (ex.: "Bacteria").
      - A partir do token de índice 1, se um token contiver a palavra "group" 
        (ignorando maiúsculas/minúsculas), pula-o e vai para o próximo.
      - Retorna o primeiro token que não contenha "group". 
      - Caso nenhum token adequado seja encontrado, retorna uma string vazia.
    """
    if not isinstance(taxonomy, str):
        return ""
    taxonomy = taxonomy.replace(';', ',')
    tokens = [token.strip() for token in taxonomy.split(',') if token.strip()]
    if len(tokens) < 2:
        return ""
    # Inicia a partir do primeiro token após o domínio (índice 1)
    i = 1
    while i < len(tokens) and re.search(r'group', tokens[i], flags=re.IGNORECASE):
        i += 1
    result = tokens[i] if i < len(tokens) else ""
    return result.strip()

def extract_genome_id(df):
    """
    Cria a coluna "Genome_ID" a partir da coluna "BGC" utilizando uma lógica similar ao comando grep/awk:
      - Se o valor inicia com "BGC": divide a string por "." e utiliza o primeiro token.
      - Se o valor inicia com "GCA_" ou "GCF_": divide a string por "_" e utiliza a concatenação do primeiro e do segundo token.
      - Se nenhum dos padrões for identificado, retorna o valor original.
    """
    def get_id(val):
        if pd.isna(val):
            return None
        val = val.strip()
        # Se inicia com BGC
        if val.startswith("BGC"):
            parts = val.split(".")
            return parts[0] if parts else val
        # Se inicia com GCA_ ou GCF_
        if val.startswith("GCA_") or val.startswith("GCF_"):
            tokens = val.split("_")
            if len(tokens) >= 2:
                return tokens[0] + "_" + tokens[1]
        # Se nenhum padrão conhecido se aplicar, retorna o valor original
        return val

    df["Genome_ID"] = df["BGC"].apply(get_id)
    return df

def adjust_taxonomic_level(df, level):
    """
    Cria ou ajusta a coluna correspondente ao nível taxonômico selecionado.
      - Para "Phylum":
            utiliza a função extract_phylum para extrair o phylum diretamente da coluna Taxonomy.
      - Para "Order":
            se a coluna "Order" já existir e tiver valores, utiliza-a; caso contrário,
            utiliza a função extract_order.
      - Para "Genus":
            utiliza extração via Ete3.
    """
    if level == "Phylum":
        df[level] = df["Taxonomy"].apply(lambda x: extract_phylum(x))
    elif level == "Order":
        if "Order" in df.columns and df["Order"].notna().sum() > 0:
            df[level] = df["Order"]
        else:
            df[level] = df.apply(lambda row: extract_order(row["Taxonomy"], row.get("Genus")), axis=1)
    elif level == "Genus":
        df[level] = df["Taxonomy"].apply(lambda x: extract_taxonomic_group_ete3(x, "genus"))
    else:
        print(f"Nível taxonômico '{level}' não reconhecido. Encerrando.")
        sys.exit(1)
    return df

def select_taxonomic_level():
    """
    Permite ao usuário selecionar interativamente o nível taxonômico (Phylum, Order ou Genus)
    e informar os nomes para filtragem (separados por vírgulas).
    """
    options = ["Phylum", "Order", "Genus"]
    print("Selecione o nível taxonômico para filtrar:")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    try:
        choice = int(input("Digite o número correspondente: "))
        level = options[choice - 1]
        taxa = input(f"Digite os nomes de {level} separados por vírgula: ")
        taxa_list = [t.strip() for t in taxa.split(",") if t.strip()]
        return level, taxa_list
    except (ValueError, IndexError):
        print("Entrada inválida. Encerrando.")
        sys.exit(1)

def calculate_proportion_and_genomes(df, taxon_value, level):
    """
    Para o taxon_value (do nível selecionado) faz duas coisas:
      1. Para a contagem dos genomas: utiliza os Genome_ID únicos.
         - A filtragem da tabela é feita de acordo com a coluna correspondente ao nível taxonômico.
           Para evitar discrepâncias, os valores são convertidos para minúsculas e espaços são retirados.
         - Para o nível "Phylum", o filtro é feito utilizando 'str.contains' na coluna Taxonomy,
           de forma semelhante ao exemplo fornecido.
         - Para fins de debug, imprime os IDs dos phylums selecionados.
      2. Para calcular as proporções das classes BiG-SCAPE: cada registro (BGC) é contado.
         Assim, agrupa por "BiG-SCAPE class" utilizando a contagem total de linhas (não dos Genome_IDs únicos).
    """
    if level.lower() == "phylum":
        # Para Phylum, filtrar utilizando a coluna Taxonomy
        filtered = df[df['Taxonomy'].str.contains('|'.join([taxon_value]), case=False, na=False)].copy()
    else:
        # Padroniza a coluna para comparação: converte para string, strip e minúsculas
        df[level] = df[level].astype(str).str.strip().str.lower()
        taxon_value_norm = taxon_value.strip().lower()
        filtered = df[df[level] == taxon_value_norm].copy()
    if filtered.empty:
        return pd.DataFrame(), 0

    # Se o nível for phylum, imprime os valores de debug
    if level.lower() == "phylum":
        unique_phylum_ids = set(filtered["Taxonomy"].apply(lambda x: extract_phylum(x).strip().lower() if isinstance(x, str) and extract_phylum(x) else ""))
        print(f"DEBUG: Phylum IDs encontrados para '{taxon_value}':")
        for pid in sorted(unique_phylum_ids):
            print(pid)
    
    # Contagem de genomas (únicos)
    unique_genomes = set(filtered["Genome_ID"].dropna())
    
    # Para as classes BiG-SCAPE, conta cada registro (BGC)
    grp = filtered.groupby("BiG-SCAPE class")["BGC"].count().reset_index(name="Count")
    total_count = grp["Count"].sum()
    grp["Proportion"] = grp["Count"] / total_count
    print(f"\nGenome IDs únicos para {taxon_value} ({level.title()}):")
    for gid in sorted(unique_genomes):
        print(gid)
    print(f"Total de genomas únicos: {len(unique_genomes)}\n")
    
    return grp, len(unique_genomes)

def adjust_annotations(annotations, spacing=0.2):
    """
    Ajusta manualmente as posições das anotações externas para evitar sobreposições.

    Args:
        annotations (list): Lista de objetos de anotação a serem ajustados.
        spacing (float): Espaçamento mínimo entre as linhas das anotações.
    """
    if not annotations:
        return

    # Ordenar as anotações externas por 'y' para ajustar de cima para baixo
    annotations_sorted = sorted(annotations, key=lambda ann: ann.xyann[1], reverse=True)

    for i in range(1, len(annotations_sorted)):
        prev = annotations_sorted[i - 1].xyann[1]
        current = annotations_sorted[i].xyann[1]
        if abs(current - prev) < spacing:
            # Ajusta a posição y da anotação atual
            x, y = annotations_sorted[i].xyann
            if y < 0:
                y_new = y - spacing
            else:
                y_new = y + spacing
            # Limita o valor de y para não sair dos limites do gráfico
            y_new = max(min(y_new, 1.5), -1.5)
            annotations_sorted[i].xyann = (x, y_new)

def plot_pie_chart(prop_df, level, taxon_names, num_genomes, color_mapping, ax=None):
    """
    Gera um gráfico de pizza baseado no DataFrame contendo os dados de proporção já calculados.
    - Fatias ≥5%:
        - Porcentagem exibida dentro da fatia.
        - Descrição movida para fora, mas sem caixinhas.
    - Fatias entre 0.1% e 5%:
        - Descrição e porcentagem exibidas externamente em caixas conectadas por setas.
    
    Args:
        prop_df (pd.DataFrame): DataFrame com as colunas 'BiG-SCAPE class', 'Count' e 'Proportion'.
        level (str): Nível taxonômico usado para filtragem.
        taxon_names (list): Lista de nomes taxonômicos inseridos pelo usuário.
        num_genomes (int): Número total de genomas.
        color_mapping (dict): Mapeamento de cores para cada classe BiG-SCAPE.
        ax (matplotlib.axes.Axes, opcional): Eixo onde o gráfico será plotado.
    """
    if prop_df.empty:
        print("Nenhum dado disponível para o gráfico de pizza.")
        return

    labels = prop_df['BiG-SCAPE class']
    sizes = prop_df['Proportion']

    # Atribui cores com base no mapeamento; se alguma classe não estiver mapeada, usa cor padrão
    colors = [color_mapping.get(lbl, "#333333") for lbl in labels]

    # Se nenhum eixo for fornecido, cria um novo
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Cria o gráfico de pizza
    ### Label das fatias wedges
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # Sem labels diretos
        colors=colors,
        startangle=140,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Exibe porcentagens > 5%
        pctdistance=0.6,
        wedgeprops=dict(edgecolor='w', linewidth=1.2),
        textprops=dict(color="white", fontsize=16),
    )

    # Ajusta a fonte das porcentagens
    for autotext in autotexts:
		### Fonte das porcentagens
        autotext.set_fontsize(14)
        if autotext.get_text() == '':
            autotext.set_visible(False)

    # Adiciona descrições externas para fatias ≥5%
    annotations = []
    for idx, (label, size, wedge) in enumerate(zip(labels, sizes, wedges)):
        if size >= 0.05:
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            annotation = ax.text(
                1.2 * x, 1.2 * y, label,
                horizontalalignment="center" if x > 0 else "right",
                verticalalignment="center",
                fontsize=16, color="black"
            )
            annotations.append(annotation)

    # Adiciona caixas externas para fatias entre 0.1% e 5%
    external_annotations = []
    for idx, (label, size, wedge) in enumerate(zip(labels, sizes, wedges)):
        if 0.001 <= size < 0.05:
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            horizontal_alignment = "left" if x > 0 else "right"
            text_x = 1.4 * np.sign(x)
            text_y = 1.4 * y
            arrow_offset = 0.2 if y > 0 else -0.2
            annotation = ax.annotate(
                f"{label} ({size * 100:.1f}%)",
                xy=(x, y),
                xytext=(text_x, text_y + arrow_offset),
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=1.0,
                    connectionstyle="arc3,rad=0.2",
                    color='black'
                ),
                horizontalalignment=horizontal_alignment,
                verticalalignment='center',
                fontsize=8,
            )
            external_annotations.append(annotation)

    adjust_annotations(external_annotations, spacing=0.2)

    # Configura o título com o nome do taxon e o número total de genomas
    taxon_name_str = ', '.join(taxon_names)
    ax.set_title(
        f"{taxon_name_str}\nTotal Genomes: {num_genomes}",
        fontsize=12,
        weight="bold",
        pad=10,
        loc='center'
    )

    ax.axis('equal')

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 script.py <caminho_para_o_arquivo>")
        sys.exit(1)
    file_path = sys.argv[1]
    df = load_data(file_path)
    if df is None or "BGC" not in df.columns:
        print("Erro: coluna 'BGC' não encontrada no arquivo.")
        sys.exit(1)
    
    # Extrai a coluna Genome_ID utilizando a mesma lógica do grep/awk
    df = extract_genome_id(df)
    if df["Genome_ID"].isna().all():
        print("Erro: a coluna 'Genome_ID' não foi criada corretamente.")
        sys.exit(1)
    print("Coluna 'Genome_ID' criada com sucesso. Primeiros valores:")
    print(df[["BGC", "Genome_ID"]].head())
    
    # Seleciona o nível taxonômico (Phylum, Order ou Genus) e os nomes para filtragem
    level, taxa_list = select_taxonomic_level()
    df = adjust_taxonomic_level(df, level)
    
    # Preparar o layout para 3 gráficos por linha, tamanho para publicação (ex.: página A4)
    num_taxa = len(taxa_list)
    cols = 3
    rows = math.ceil(num_taxa / cols)
    
    fig_width = 18   # Largura em polegadas (aproximadamente A4)
    #fig_height = 11.69 # Altura em polegadas (aproximadamente A4)
    fig_height = 6 * rows  # Altura proporcional ao número de linhas
    # Cria os subplots
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.5, wspace=0.3)
    axes = axes.flatten()
    
    # Remover eixos vazios, se houver
    for idx in range(num_taxa, len(axes)):
        fig.delaxes(axes[idx])
    
    # Mapeamento de cores para as classes BiG-SCAPE
    unique_classes = sorted(df["BiG-SCAPE class"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    color_mapping = {bgc_class: cmap(i / (len(unique_classes) - 1) if len(unique_classes) > 1 else 0.5)
                     for i, bgc_class in enumerate(unique_classes)}
    
    for ax, taxon in zip(axes, taxa_list):
        prop_df, unique_genomes_count = calculate_proportion_and_genomes(df, taxon, level)
        if prop_df.empty:
            print(f"Nenhum dado encontrado para {taxon}.")
            continue
        plot_pie_chart(prop_df, level, [taxon], unique_genomes_count, color_mapping, ax)
    plt.tight_layout()
    # Salvar a figura com as mesmas dimensões e proporções
    output_formats = ['png', 'jpeg', 'svg']
    for fmt in output_formats:
        plt.savefig(f'pie_charts.{fmt}', dpi=300, format=fmt, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()
