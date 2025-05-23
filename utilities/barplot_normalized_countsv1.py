import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import numpy as np
from ete3 import NCBITaxa

# Inicializa o objeto NCBITaxa para correção da linhagem taxonômica
ncbi_taxa = NCBITaxa()

def fix_acetobacteraceae_lineage(lineage):
    """
    Corrige a linhagem quando a família é Acetobacteraceae e
    a ordem está incorretamente indicada como Rhodospirillales.
    Nesse caso, a ordem é substituída por Acetobacterales.
    """
    tokens = [token.strip() for token in lineage.split(';') if token.strip()]
    if len(tokens) > 4:
        if tokens[4].lower() == 'acetobacteraceae' and tokens[3].lower() == 'rhodospirillales':
            tokens[3] = 'Acetobacterales'
        return '; '.join(tokens)
    return lineage

def standardize_lineage_format(lineage_string):
    """
    Normaliza a string da linhagem, garantindo que cada ';' seja seguido de um espaço único,
    removendo espaços extras e garantindo que a string termine com ';'.
    """
    standardized_lineage = re.sub(r'\s*;\s*', '; ', lineage_string)
    standardized_lineage = standardized_lineage.strip()
    if not standardized_lineage.endswith(';'):
        standardized_lineage += ';'
    return standardized_lineage

def get_corrected_lineage_from_species(species_name):
    """
    A partir do nome da espécie (esperado na coluna 'Species'),
    utiliza o ete3 para:
      1. Traduzir o nome para taxid.
      2. Obter a linhagem completa (lista de taxids) e seus nomes oficiais.
      3. Retornar a linhagem formatada e padronizada.
    Em caso de erro, retorna None.
    """
    try:
        name_to_taxid = ncbi_taxa.get_name_translator([species_name])
        if species_name not in name_to_taxid:
            return None
        taxid = name_to_taxid[species_name][0]
        lineage_ids = ncbi_taxa.get_lineage(taxid)
        taxid_to_name = ncbi_taxa.get_taxid_translator(lineage_ids)
        desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        taxonomic_ranks = ncbi_taxa.get_rank(lineage_ids)
        lineage_names = [taxid_to_name[t] for t in lineage_ids if taxonomic_ranks[t] in desired_ranks]
        raw_lineage = '; '.join(lineage_names) + ';'
        return standardize_lineage_format(raw_lineage)
    except Exception as error_message:
        print(f"Error obtaining lineage for species '{species_name}': {error_message}")
        return None

def get_corrected_lineage_from_organism(organism_name):
    """
    Similar à função get_corrected_lineage_from_species, mas utiliza o nome do organismo
    vindo da coluna 'Organism Name' na Tabela 2.
    """
    return get_corrected_lineage_from_species(organism_name)

def update_lineage_for_eukaryotes_table1(df):
    """
    Atualiza a coluna 'Lineage' para as linhas correspondentes a Eukaryota na Tabela 1,
    utilizando a coluna 'Species' para obter a linhagem corrigida via ete3.
    """
    if 'Species' not in df.columns:
        raise KeyError("Coluna 'Species' não encontrada no DataFrame da Tabela 1.")
    
    mask_eukaryotes = df['Lineage'].astype(str).str.startswith("Eukaryota")
    df.loc[mask_eukaryotes, 'Lineage'] = df.loc[mask_eukaryotes, 'Species'].apply(get_corrected_lineage_from_species)
    return df

def update_lineage_for_eukaryotes_table2(df):
    """
    Atualiza a coluna 'Lineage' para as linhas correspondentes a Eukaryota na Tabela 2,
    utilizando a coluna 'Organism Name' para obter a linhagem corrigida via ete3.
    """
    if 'Organism Name' not in df.columns:
        raise KeyError("Coluna 'Organism Name' não encontrada no DataFrame da Tabela 2.")
    
    mask_eukaryotes = df['Lineage'].astype(str).str.startswith("Eukaryota")
    df.loc[mask_eukaryotes, 'Lineage'] = df.loc[mask_eukaryotes, 'Organism Name'].apply(get_corrected_lineage_from_organism)
    return df

def extract_taxonomic_group(lineage, level):
    """
    Extrai o grupo taxonômico da string de linhagem baseado no nível desejado.
    
    Para Genus:
      - Se a classificação possuir 6 ou mais tokens, utiliza o token no índice 5,
        desde que não termine com "ales" ou "eae" e não inicie com "Candidatus".
      - Caso contrário, tenta utilizar o penúltimo token.
    
    Para Family:
      - Itera sobre os tokens e retorna o primeiro token que termina com "eae" (sem considerar maiúsculas/minúsculas).
    
    Para Order:
      - Itera sobre os tokens e retorna o primeiro token que termina com "ales" (sem considerar maiúsculas/minúsculas).
    
    Para outros níveis:
      - Utiliza uma lista fixa de níveis: ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        e retorna o token correspondente ao índice desse nível, se disponível.
    """
    tokens = [token.strip() for token in lineage.split(';') if token.strip()]
    
    if level.lower() == 'genus':
        if len(tokens) >= 6:
            candidate = tokens[5]
            if not (candidate.lower().endswith('ales') or candidate.lower().endswith('eae') or candidate.startswith("Candidatus")):
                return candidate
        if len(tokens) >= 2:
            candidate = tokens[-2]
            if not (candidate.lower().endswith('ales') or candidate.lower().endswith('eae') or candidate.startswith("Candidatus")):
                return candidate
        return None
    elif level.lower() == 'family':
        for token in tokens:
            if token.lower().endswith('eae'):
                return token
        return None
    elif level.lower() == 'order':
        for token in tokens:
            if token.lower().endswith('ales'):
                return token
        return None
    else:
        levels_order = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        try:
            index = levels_order.index(level)
            return tokens[index] if index < len(tokens) else None
        except (ValueError, IndexError):
            return None

def generate_filtered_table_and_graphs(table1_path, table2_path, domain_name, taxonomic_level, top_n, dpi, sub_taxonomic_level=None):
    # Carrega a Tabela 1 (dataset completo)
    df1_all = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Atualiza a linhagem para Eukaryota na Tabela 1, se necessário
    if "Eukaryota" in domain_name:
        df1_all = update_lineage_for_eukaryotes_table1(df1_all)
    
    # Filtra a Tabela 1 pelo domínio e opcionalmente pelo subnível taxonômico
    df1_all = df1_all[df1_all['Lineage'].str.contains(domain_name, na=False)].copy()
    if sub_taxonomic_level:
        df1_all = df1_all[df1_all['Lineage'].str.contains(sub_taxonomic_level, na=False)].copy()
    
    # Para nível Genus, mantém apenas registros com pelo menos 6 tokens na linhagem
    if taxonomic_level.lower() == "genus":
        df1_all = df1_all[df1_all['Lineage'].apply(lambda x: len([t.strip() for t in x.split(';') if t.strip()]) >= 6)]
    
    # Extrai o grupo taxonômico para o nível especificado
    df1_all['Taxonomic_Group'] = df1_all['Lineage'].apply(
        lambda x: extract_taxonomic_group(x, sub_taxonomic_level or taxonomic_level)
    )
    df1_all = df1_all[df1_all['Taxonomic_Group'].notna()]
    df1_all = df1_all[~df1_all['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    # Ajustes específicos para Family e Order (normalização de dados)
    if taxonomic_level.lower() == "family":
        df1_all = df1_all[~df1_all['Taxonomic_Group'].str.lower().eq('cystobacterineae')]
        df1_all = df1_all[~df1_all['Taxonomic_Group'].str.lower().str.endswith('ales')]
        # Filtro adicional para Eukaryota no nível Family
        if domain_name.lower() == "eukaryota":
            grupos_excluir = ['eustigmatophyceae', 'pelagophyceae', 'phaeophyceae', 'vitrellaceae', 'dinophyceae']
            df1_all = df1_all[~df1_all['Taxonomic_Group'].str.lower().isin(grupos_excluir)]
    if taxonomic_level.lower() == "order":
        df1_all = df1_all[df1_all['Taxonomic_Group'].str.lower().str.endswith('ales')]
    
    # Correção para Phylum: padronizando alguns nomes
    if taxonomic_level.lower() == "phylum":
        df1_all.loc[df1_all['Taxonomic_Group'].str.lower().isin(['proteobacteria', 'deltaproteobacteria']), 'Taxonomic_Group'] = 'Pseudomonadota'
    
    if df1_all.empty:
        print("Nenhum dado encontrado para o domínio e nível taxonômico informados na Tabela 1.")
        return
    
    # Agregação dos dados: contagem total de FAALs na Tabela 1
    total_faal_counts_all = df1_all.groupby('Taxonomic_Group')['Protein Accession'].count().reset_index(name='Total FAAL Count')
    
    # Filtra registros com montagem de genoma válidos (Assembly inicia com "GCF_" ou "GCA_")
    df1_filtered = df1_all[df1_all['Assembly'].str.startswith(('GCF_', 'GCA'), na=False)].copy()
    faal_count_series = df1_filtered.groupby('Taxonomic_Group')['Protein Accession'].size().reset_index(name='FAAL_Count')
    unique_genomes = df1_filtered.drop_duplicates(subset=['Assembly', 'Taxonomic_Group'])
    genome_count_series = unique_genomes.groupby('Taxonomic_Group')['Assembly'].count().reset_index(name='Genome_Count')
    
    # Calcula a média de FAAL por genoma
    faal_stats = pd.merge(faal_count_series, genome_count_series, on='Taxonomic_Group', how='left')
    faal_stats['Mean FAAL Count per Genome'] = faal_stats['FAAL_Count'] / faal_stats['Genome_Count']
    
    merged_data = pd.merge(total_faal_counts_all, 
                           faal_stats[['Taxonomic_Group', 'Mean FAAL Count per Genome', 'Genome_Count']],
                           on='Taxonomic_Group')
    
    # Seleciona os Top N grupos taxonômicos com base na contagem total de FAAL
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Total FAAL Count')
    
    # --- Plot A: Contagem Total de FAALs ---
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    order_axis = top_taxonomic_groups.sort_values('Total FAAL Count', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Total FAAL Count', y='Taxonomic_Group', data=top_taxonomic_groups, 
                ax=ax[0], palette='viridis', order=order_axis)
    ax[0].set_xlabel('Fatty Acyl AMP Ligase (FAALs) Counts', fontsize=14)
    ax[0].set_ylabel(f'{taxonomic_level} Level', fontsize=14)
    ax[0].text(-0.1, 1.15, "A", transform=ax[0].transAxes, fontsize=16, fontweight='bold',
               va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    for patch, group in zip(ax[0].patches, order_axis):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        mean_val = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == group]['Mean FAAL Count per Genome'].values[0]
        ax[0].text(x, y, f'{mean_val:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    ax[0].margins(x=0)
    
    # --- Processamento da Tabela 2 ---
    df2 = pd.read_csv(table2_path, sep='\t', low_memory=False)
    
    # Atualiza a linhagem para Eukaryota na Tabela 2, se necessário
    if "Eukaryota" in domain_name:
        df2 = update_lineage_for_eukaryotes_table2(df2)
    
    df2['Lineage'] = df2['Lineage'].apply(fix_acetobacteraceae_lineage)
    df2 = df2[df2['Assembly Accession'].str.startswith(('GCF_', 'GCA'), na=False)].copy()
    
    # Remove registros duplicados com base no sufixo do accession
    df2['accession_suffix'] = df2['Assembly Accession'].str.replace(r'^(GCF_|GCA_)', '', regex=True)
    df2 = df2.drop_duplicates(subset=['accession_suffix'])
    
    if taxonomic_level.lower() == "genus":
        df2 = df2[df2['Lineage'].apply(lambda x: len([t.strip() for t in x.split(';') if t.strip()]) >= 6)]
    
    df2['Taxonomic_Group'] = df2['Lineage'].apply(
        lambda x: extract_taxonomic_group(x, sub_taxonomic_level or taxonomic_level)
    )
    df2 = df2[df2['Taxonomic_Group'].notna()]
    
    # Padroniza os nomes dos grupos taxonômicos com base nos dados da Tabela 1
    unique_tax_groups = df1_all['Taxonomic_Group'].dropna().unique()
    mapping = {tg.lower(): tg for tg in unique_tax_groups}
    df2['Taxonomic_Group'] = df2['Taxonomic_Group'].str.lower().map(mapping)
    df2_filtered = df2[df2['Taxonomic_Group'].notna()].copy()
    
    if taxonomic_level.lower() in ["phylum", "family", "order", "genus"]:
        df2_filtered = df2_filtered[~df2_filtered['Taxonomic_Group'].str.lower().eq('proteobacteria')]
        df2_filtered.loc[df2_filtered['Taxonomic_Group'].str.lower() == 'deltaproteobacteria', 'Taxonomic_Group'] = 'Pseudomonadota'
    
    if taxonomic_level.lower() == "family":
        df2_filtered = df2_filtered[~df2_filtered['Taxonomic_Group'].str.lower().eq('cystobacterineae')]
        df2_filtered = df2_filtered[~df2_filtered['Taxonomic_Group'].str.lower().str.endswith('ales')]
        # Filtro adicional para Eukaryota no nível Family
        if domain_name.lower() == "eukaryota":
            grupos_excluir = ['eustigmatophyceae', 'pelagophyceae', 'phaeophyceae', 'vitrellaceae', 'dinophyceae']
            df2_filtered = df2_filtered[~df2_filtered['Taxonomic_Group'].str.lower().isin(grupos_excluir)]
    if taxonomic_level.lower() == "order":
        df2_filtered = df2_filtered[df2_filtered['Taxonomic_Group'].str.lower().str.endswith('ales')]
    
    df2_filtered = df2_filtered[~df2_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    if df2_filtered.empty:
        print("Nenhum dado encontrado para os grupos taxonômicos informados na Tabela 2.")
        return
    
    output_filtered_table = 'Taxonomic_groups_with_FAAL.tsv'
    if not os.path.exists(output_filtered_table):
        df2_filtered.to_csv(output_filtered_table, sep='\t', index=False)
    
    # Obtém a contagem total de genomas da Tabela 2 (usando Assembly Accession único)
    genome_counts_total = df2_filtered.groupby('Taxonomic_Group')['Assembly Accession'].nunique().reset_index(name='Total Genome Count')
    
    normalized_data = pd.merge(top_taxonomic_groups, genome_counts_total, on='Taxonomic_Group', how='left')
    normalized_data['Total Genome Count'] = normalized_data['Total Genome Count'].fillna(0)
    
    if taxonomic_level.lower() in ['phylum', 'family', 'order', 'genus']:
        normalized_data['Normalized'] = np.where(
            normalized_data['Total Genome Count'] == 0,
            0,
            (normalized_data['Genome_Count'] / normalized_data['Total Genome Count']) * 100
        )
        annotation_inside_label = 'Genome_Count'
    else:
        normalized_data['Normalized'] = np.where(
            normalized_data['Total Genome Count'] == 0,
            0,
            (normalized_data['Total FAAL Count'] / normalized_data['Total Genome Count']) * 100
        )
        annotation_inside_label = 'Total FAAL Count'
    
    # --- Correção Genérica para Contagem Não-Redundante de Genomas ---
    # Para cada grupo taxonômico, se a contagem de genomas da Tabela 1 (Genome_Count)
    # for menor que a contagem total de genomas da Tabela 2, recalcula a união dos IDs de genomas de ambas as tabelas.
    updated_total_ids = {}
    for taxon in normalized_data['Taxonomic_Group']:
        ids_table1 = set(df1_filtered[df1_filtered['Taxonomic_Group'] == taxon]['Assembly'].unique())
        ids_table2 = set(df2_filtered[df2_filtered['Taxonomic_Group'] == taxon]['Assembly Accession'].unique())
        union_ids = ids_table1.union(ids_table2)
        union_count = len(union_ids)
        row = normalized_data[normalized_data['Taxonomic_Group'] == taxon]
        table1_count = row['Genome_Count'].values[0] if not row.empty else 0
        if union_count > table1_count:
            normalized_data.loc[normalized_data['Taxonomic_Group'].str.lower() == taxon.lower(), 'Total Genome Count'] = union_count
            if taxonomic_level.lower() in ['phylum', 'family', 'order', 'genus']:
                norm_val = (table1_count / union_count) * 100 if union_count > 0 else 0
                normalized_data.loc[normalized_data['Taxonomic_Group'].str.lower() == taxon.lower(), 'Normalized'] = norm_val
            else:
                total_faal = row['Total FAAL Count'].values[0]
                norm_val = (total_faal / union_count) * 100 if union_count > 0 else 0
                normalized_data.loc[normalized_data['Taxonomic_Group'].str.lower() == taxon.lower(), 'Normalized'] = norm_val
        updated_total_ids[taxon] = ", ".join(sorted(union_ids))
    
    # --- Geração de Listas Agregadas de IDs para Tabela de Verificação ---
    protein_ids_by_group = df1_filtered.groupby('Taxonomic_Group')['Protein Accession']\
                                       .apply(lambda x: ', '.join(x.dropna().unique())).reset_index(name='Protein IDs')
    genomes_with_faal_ids = df1_filtered.groupby('Taxonomic_Group')['Assembly']\
                                        .apply(lambda x: ', '.join(x.dropna().unique())).reset_index(name='Genomes with FAAL IDs')
    total_genome_ids = df2_filtered.groupby('Taxonomic_Group')['Assembly Accession']\
                                  .apply(lambda x: ', '.join(x.dropna().unique())).reset_index(name='Total Genome IDs')
    
    # Para cada grupo taxonômico, atualiza os IDs dos genomas usando a união gerada
    for taxon, union_ids_str in updated_total_ids.items():
        condition = total_genome_ids['Taxonomic_Group'].str.lower() == taxon.lower()
        total_genome_ids.loc[condition, 'Total Genome IDs'] = union_ids_str
    
    # Mescla as listas de IDs agregados com os dados normalizados
    verification_table = normalized_data.merge(protein_ids_by_group, on='Taxonomic_Group', how='left')\
                                          .merge(genomes_with_faal_ids, on='Taxonomic_Group', how='left')\
                                          .merge(total_genome_ids, on='Taxonomic_Group', how='left')
    verification_table = verification_table.rename(columns={
        'Taxonomic_Group': 'Taxonomy',
        'Normalized': 'Percentage of Genomes with FAAL'
    })
    verification_table = verification_table[['Taxonomy', 'Protein IDs', 'Genomes with FAAL IDs', 'Total Genome IDs', 'Percentage of Genomes with FAAL']]
    
    verification_table_file = 'verification_table.tsv'
    verification_table.to_csv(verification_table_file, sep='\t', index=False)
    print(f"Tabela de verificação gerada com sucesso: {verification_table_file}")
    
    # --- Plot B: Normalização ---
    order_normalized = normalized_data.sort_values('Normalized', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Normalized', y='Taxonomic_Group', data=normalized_data,
                ax=ax[1], palette='viridis', order=order_normalized)
    ax[1].set_xlabel('Percentage (%) of Deposited Genomes Containing FAALs', fontsize=14)
    ax[1].set_ylabel(f'{taxonomic_level} Level', fontsize=14)
    ax[1].text(-0.1, 1.15, "B", transform=ax[1].transAxes, fontsize=16, fontweight='bold',
               va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    normalized_data_sorted = normalized_data.set_index('Taxonomic_Group').loc[order_normalized].reset_index()
    for patch, (_, row) in zip(ax[1].patches, normalized_data_sorted.iterrows()):
        bar_width = patch.get_width()
        bar_y = patch.get_y() + patch.get_height() / 2
        ax[1].text(bar_width + 0.5, bar_y, f'{int(row["Total Genome Count"])}', 
                   color='black', ha='left', va='center', fontsize=10, fontweight='bold')
        ax[1].text(bar_width / 2, bar_y, f'{int(row[annotation_inside_label])}', 
                   color='white', ha='center', va='center', fontsize=10, fontweight='bold')
    ax[1].margins(x=0)
    
    if taxonomic_level.lower() in ['genus', 'species']:
        ax[0].set_yticklabels(ax[0].get_yticklabels(), style='italic')
        ax[1].set_yticklabels(ax[1].get_yticklabels(), style='italic')
    
    plt.tight_layout()
    plt.savefig('ranking_FAAL_combined.png', dpi=dpi)
    plt.savefig('ranking_FAAL_combined.svg', dpi=dpi)
    plt.savefig('ranking_FAAL_combined.jpeg', dpi=dpi)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) not in [7, 8]:
        print("Usage: python3 barplot_normalized_counts_faal.py <table1.tsv> <table2.tsv> <Domain> <Taxonomic Level> <Top N> <DPI> [<Sub Taxonomic Level>]")
        sys.exit(1)
    table1_path = sys.argv[1]
    table2_path = sys.argv[2]
    domain_name = sys.argv[3]
    taxonomic_level = sys.argv[4]
    top_n = int(sys.argv[5])
    dpi = int(sys.argv[6])
    sub_taxonomic_level = sys.argv[7] if len(sys.argv) == 8 else None
    generate_filtered_table_and_graphs(table1_path, table2_path, domain_name, taxonomic_level, top_n, dpi, sub_taxonomic_level)

