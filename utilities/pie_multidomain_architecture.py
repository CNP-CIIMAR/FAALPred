#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ete3 import NCBITaxa

# Inicializa o objeto NCBITaxa
ncbi = NCBITaxa()

# ------------------------------------------------------------
# 1) Combinar descrições de assinatura
# ------------------------------------------------------------
def combine_signature_descriptions(df):
    """
    Gera a coluna 'Combined.description' para cada 'Protein.accession'
    a partir de 'Signature.description'. Substitui 'FAAL' por 'FAAL stand-alone'.
    """
    if 'Signature.description' not in df.columns:
        raise ValueError("A coluna 'Signature.description' não foi encontrada no DataFrame.")

    def simplify_signature(description):
        if 'NRPS' in description:
            return 'NRPS'
        if 'PKS' in description:
            return 'PKS'
        return description

    grouped = (
        df
        .groupby('Protein.accession')['Signature.description']
        .apply(lambda x: '-'.join(sorted(set(simplify_signature(d) for d in x))))
        .reset_index()
    )
    grouped.rename(columns={'Signature.description': 'Combined.description'}, inplace=True)

    # Substitui "FAAL" por "FAAL stand-alone"
    grouped['Combined.description'] = grouped['Combined.description'].apply(
        lambda x: 'FAAL stand-alone' if x == 'FAAL' else x
    )

    # Mesclar de volta
    df = pd.merge(
        df.drop(columns=['Signature.description'], errors='ignore'),
        grouped,
        on='Protein.accession'
    )
    return df

# ------------------------------------------------------------
# 2) Carregar dados
# ------------------------------------------------------------
def load_data(table1_path, table2_path):
    """
    Lê as duas tabelas (TSV) e faz checagens básicas.
    """
    print(f"Carregando dados de: {table1_path} e {table2_path}")
    df1 = pd.read_csv(table1_path, sep='\t')
    df2 = pd.read_csv(table2_path, sep='\t')

    print("Colunas DF1:", df1.columns.tolist())
    print("Colunas DF2:", df2.columns.tolist())
    print("Formato DF1:", df1.shape)
    print("Formato DF2:", df2.shape)

    if 'Assembly' not in df1.columns:
        raise ValueError("A coluna 'Assembly' não foi encontrada em DF1.")

    valid_assemblies = df1['Assembly'].str.startswith(('GCA', 'GCF'))
    if not valid_assemblies.any():
        raise ValueError("Nenhum Assembly ID válido (iniciando com GCA ou GCF) em DF1.")
    return df1, df2

# ------------------------------------------------------------
# 3) Mesclar tabelas
# ------------------------------------------------------------
def merge_tables(df1, df2, on='Protein.accession'):
    """
    Mescla df1 e df2 no campo 'Protein.accession' (inner join).
    Em seguida, gera 'Combined.description'.
    """
    if on not in df1.columns or on not in df2.columns:
        raise ValueError(f"Coluna '{on}' não encontrada em df1 ou df2.")

    merged_df = pd.merge(df1, df2, on=on, how='inner')
    print(f"DataFrame mesclado: {merged_df.shape}")

    if 'Signature.description' not in merged_df.columns:
        raise ValueError("'Signature.description' ausente após mesclagem.")

    merged_df = combine_signature_descriptions(merged_df)
    print("Coluna 'Combined.description' criada com sucesso.")
    return merged_df

# ------------------------------------------------------------
# 4) Extrair níveis taxonômicos (Lineage)
# ------------------------------------------------------------
def extract_taxonomic_levels(lineage):
    """
    Retorna um dicionário com superkingdom, phylum, class, order, family, genus, species
    usando ete3 (NCBITaxa).
    """
    levels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    tax_dict = {lvl: None for lvl in levels}

    if pd.isna(lineage) or not isinstance(lineage, str):
        return tax_dict

    taxa = lineage.split('; ')
    tax_dict['phylum'] = get_phylum(lineage)

    for taxon in taxa:
        taxid = ncbi.get_name_translator([taxon])
        if taxon in taxid:
            rank = ncbi.get_rank([taxid[taxon][0]])
            if rank[taxid[taxon][0]] in levels:
                tax_dict[rank[taxid[taxon][0]]] = taxon
    return tax_dict

def get_phylum(lineage):
    """
    Retorna o phylum diretamente, se encontrado.
    """
    if not lineage or not isinstance(lineage, str):
        return None
    taxa = lineage.split('; ')
    for taxon in taxa:
        taxid = ncbi.get_name_translator([taxon])
        if taxon in taxid:
            rank = ncbi.get_rank([taxid[taxon][0]])
            if rank[taxid[taxon][0]] == 'phylum':
                return taxon
    return None

# ------------------------------------------------------------
# 5) Atualizar linhagem e filtrar por domínio
# ------------------------------------------------------------
def update_lineage(df, domain_name):
    """
    Aplica extract_taxonomic_levels em cada linha.
    Filtra para manter somente o domínio escolhido.
    Remove linhas sem phylum, order, genus.
    """
    print(f"Atualizando linhagem para o domínio: {domain_name}")
    tax_data = df['Lineage'].apply(extract_taxonomic_levels)

    for lvl in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus']:
        df[lvl] = tax_data.apply(lambda x: x.get(lvl, None))

    if domain_name in ['Bacteria', 'Archaea']:
        df['superkingdom'] = df['Lineage'].apply(
            lambda x: x.split(';')[0].strip() if pd.notna(x) else None
        )

    df = df[df['superkingdom'] == domain_name]
    df = df.dropna(subset=['phylum', 'order', 'genus'])

    # Remover espaços extras na coluna 'phylum'
    df['phylum'] = df['phylum'].astype(str).str.strip()

    print("DataFrame filtrado:", df.shape)
    return df

# ------------------------------------------------------------
# 6) Plotar subplots (2 colunas) com regex para
#    "Candidatus Rokuibacteriota" e "Gemmatimonadota"
# ------------------------------------------------------------
def plot_topN_multidomain_in_one_figure(df, taxonomic_level, taxon_list, top_n, dpi):
    """
    Cria subplots em 2 colunas, no máximo 2 "pies" por linha.

    Se o 'taxon_name' for:
      - "Candidatus Rokuibacteriota"
        => aplica regex em "Lineage" c/ padrao:
           (?i).*candidatus\s+rokuibacteriota.*
      - "Gemmatimonadota"
        => aplica regex (?i).*gemmatimonadota.*

    Senão, busca normal em df['phylum'] (case-insensitive).
    """
    level_col = taxonomic_level.lower()
    if level_col not in df.columns:
        print(f"Coluna '{level_col}' não existe no DataFrame.")
        return

    df[level_col] = df[level_col].astype(str).str.strip()

    num_taxons = len(taxon_list)
    if num_taxons == 0:
        print("Nenhum táxon informado. Abortando.")
        return

    n_cols = 2
    n_rows = math.ceil(num_taxons / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10, 6*n_rows)
    )
    if n_rows == 1:
        axes = [axes]

    plt.subplots_adjust(
        left=0.229,
        right=1.0,
        top=0.9,
        bottom=0.1,
        wspace=0.0,
        hspace=0.3
    )

    # Definicao das regex
    #  - "(?i)" => ignore case
    #  - ".*candidatus\s+rokuibacteriota.*" => substring c/ "candidatus <espaços> rokuibacteriota"
    #  - ".*gemmatimonadota.*" => substring gemmatimonadota
    re_roku = r"(?i).*candidatus\s+rokuibacteriota.*"
    re_gemma = r"(?i).*gemmatimonadota.*"

    for i, taxon_name in enumerate(taxon_list):
        row_i = i // n_cols
        col_i = i % n_cols
        ax = axes[row_i][col_i]

        taxon_name_stripped = taxon_name.strip()

        # Comparar ignore-case
        lower_name = taxon_name_stripped.lower()

        # Se for "Candidatus Rokuibacteriota", regex no Lineage
        if lower_name == "candidatus rokuibacteriota":
            sub_df = df[df['Lineage'].str.contains(re_roku, na=False, regex=True)]
        elif lower_name == "gemmatimonadota":
            sub_df = df[df['Lineage'].str.contains(re_gemma, na=False, regex=True)]
        else:
            # Busca normal em df['phylum'], case-insensitive
            sub_df = df[df[level_col].str.lower() == lower_name]

        if sub_df.empty:
            ax.set_title(f"{taxon_name_stripped}\n(sem dados)", fontsize=12)
            ax.axis('off')
            continue

        arch_counts = sub_df['Combined.description'].value_counts()

        # Top N + Others
        top_arch = arch_counts.head(top_n)
        if len(arch_counts) > top_n:
            others_sum = arch_counts[top_n:].sum()
            top_arch = pd.concat([top_arch, pd.Series({'Others': others_sum})])

        labels = top_arch.index.tolist()
        sizes = top_arch.values.tolist()

        # Paleta
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

        # Criar pie sem autopct
        wedges, text_labels = ax.pie(
            sizes,
            labels=None,
            startangle=140,
            colors=colors
        )

        # Doughnut
        centre_circle = plt.Circle((0, 0), 0.6, edgecolor='black', facecolor='white', fill=True)
        ax.add_artist(centre_circle)
        ax.set_title(taxon_name_stripped, fontsize=12, pad=10)
        ax.axis('equal')

        # Montar patches com % na legenda
        patches = []
        total = sum(sizes)
        for lbl, val, c in zip(labels, sizes, colors):
            pct = (val / total)*100
            lbl_pct = f"{lbl} ({pct:.1f}%)"
            patches.append(mpatches.Patch(color=c, label=lbl_pct))

        ax.legend(
            handles=patches,
            loc='center left',
            bbox_to_anchor=(-0.15, 0.5),
            fontsize=9,
            frameon=False
        )

    # Desligar subplots extras, se sobrarem
    total_subplots = n_rows * n_cols
    if total_subplots > num_taxons:
        for idx in range(num_taxons, total_subplots):
            rr = idx // n_cols
            cc = idx % n_cols
            axes[rr][cc].axis('off')

    base_name = f"{taxonomic_level}_top{top_n}_architectures"
    plt.savefig(f"{base_name}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{base_name}.jpeg", dpi=dpi, bbox_inches='tight')
    print(f"Figuras salvas: {base_name}.png, .svg, .jpeg (DPI={dpi})")

    plt.show()

# ------------------------------------------------------------
# 7) Função Principal
# ------------------------------------------------------------
def main():
    """
    Uso:
      python3 pie_chart.py <table1> <table2> <domain_name> <top_n> <taxonomic_level> <taxon_list> <dpi>

    Exemplo:
      python3 pie_chart.py \\
        Genomes_Total_proteinas_taxonomy_FAAL_metadata_nodup.tsv \\
        results_all_lista_proteins.faals_cdd.tsv \\
        Bacteria 6 Phylum "Candidatus Rokuibacteriota,Gemmatimonadota,Myxococcota" 300

    - Se 'taxon_name' == "Candidatus Rokuibacteriota" (ignora maiúsculas):
      => aplica regex: (?i).*candidatus\s+rokuibacteriota.*
    - Se 'taxon_name' == "Gemmatimonadota":
      => aplica regex: (?i).*gemmatimonadota.*
    - Caso contrário, faz busca normal em phylum (case-insensitive).
    """
    if len(sys.argv) < 8:
        print(
            "Uso: python3 pie_chart.py <table1_path> <table2_path> <domain_name> <top_n> "
            "<taxonomic_level> <taxon_list> <dpi>\n"
            "Exemplo:\n"
            "  python3 pie_chart.py \\\n"
            "    Genomes_Total_proteinas_taxonomy_FAAL_metadata_nodup.tsv \\\n"
            "    results_all_lista_proteins.faals_cdd.tsv \\\n"
            "    Bacteria 6 Phylum \"Candidatus Rokuibacteriota,Gemmatimonadota\" 300\n"
        )
        sys.exit(1)

    table1_path = sys.argv[1]
    table2_path = sys.argv[2]
    domain_name = sys.argv[3]
    top_n = int(sys.argv[4])
    taxonomic_level = sys.argv[5]
    taxon_list_str = sys.argv[6]
    dpi = int(sys.argv[7])

    if domain_name not in ['Bacteria', 'Archaea', 'Eukaryota']:
        print("Erro: <domain_name> deve ser 'Bacteria', 'Archaea' ou 'Eukaryota'.")
        sys.exit(1)

    if taxonomic_level not in ['Phylum', 'Order', 'Genus']:
        print("Erro: <taxonomic_level> deve ser 'Phylum', 'Order' ou 'Genus'.")
        sys.exit(1)

    # Permite nomes compostos
    taxon_list = [x.strip() for x in taxon_list_str.split(',') if x.strip()]

    df1, df2 = load_data(table1_path, table2_path)
    merged_df = merge_tables(df1, df2)
    filtered_df = update_lineage(merged_df, domain_name)
    if filtered_df.empty:
        print(f"Nenhum dado encontrado para o domínio {domain_name}. Encerrando.")
        sys.exit(0)

    plot_topN_multidomain_in_one_figure(filtered_df, taxonomic_level, taxon_list, top_n, dpi)

# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
if __name__ == "__main__":
    main()



