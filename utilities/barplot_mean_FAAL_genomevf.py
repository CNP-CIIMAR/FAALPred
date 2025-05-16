import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ete3 import NCBITaxa

# Inicializa o NCBI
ncbi = NCBITaxa()

def extract_taxonomic_group_by_ete3(species_name, target_rank):
    """
    Dado um nome de espécie (ou gênero), retorna o nome do taxon
    no nível target_rank (domain, phylum, class, order, family, genus)
    usando exclusivamente ETE3.
    """
    try:
        # Tenta obter o taxid pelo nome completo; se falhar, usa apenas o gênero
        tx = ncbi.get_name_translator([species_name])
        if not tx and ' ' in species_name:
            genus = species_name.split()[0]
            tx = ncbi.get_name_translator([genus])
        if not tx:
            return None

        taxid = list(tx.values())[0][0]
        lineage = ncbi.get_lineage(taxid)
        ranks   = ncbi.get_rank(lineage)
        names   = ncbi.get_taxid_translator(lineage)

        for tid in lineage:
            if ranks.get(tid, '').lower() == target_rank.lower():
                return names.get(tid)
    except Exception:
        return None
    return None


def extract_taxonomic_group(lineage, level):
    levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    lineage_split = lineage.split(';')
    if level in levels:
        try:
            index = levels.index(level)
            return lineage_split[index].strip()
        except IndexError:
            return None
    return None


def filter_by_criteria_ete3(name, level):
    """
    Critérios de filtragem por sufixo:
    - Order deve terminar em 'ales'
    - Family deve terminar em 'eae'
    - Genus não deve terminar em 'ales' nem 'eae'
    - Phylum e Domain sem critérios especiais
    """
    if name is None:
        return False
    lvl = level.lower()
    if lvl == 'order'  and name.endswith('ales'):
        return True
    if lvl == 'family' and name.endswith('eae'):
        return True
    if lvl == 'genus'  and not (name.endswith('ales') or name.endswith('eae')):
        return True
    if lvl in ['phylum', 'domain']:
        return True
    return False


def generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi):
    # Carrega a tabela
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Fluxo Eukaryota vs outros domínios
    if domain_name == 'Eukaryota':
        df1 = df1.copy()
        df1['Lineage'] = df1['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Domain') or '') + '; ' + \
                       df1['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Phylum') or '') + '; ' + \
                       df1['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Class') or '') + '; ' + \
                       df1['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Order') or '') + '; ' + \
                       df1['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Family') or '') + '; ' + \
                       df1['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Genus') or '')
        df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
        df1 = df1.dropna(subset=['Assembly'])
        df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
        df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF','GCA'), na=False)]

        # Extração por nível taxonômico
        level = taxonomic_level.lower()
        if level in ['phylum', 'order', 'family', 'genus']:
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(
                lambda s: extract_taxonomic_group_by_ete3(s, taxonomic_level)
            )
        else:
            df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(
                lambda x: extract_taxonomic_group(x, taxonomic_level)
            )

        df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(
            lambda x: filter_by_criteria_ete3(x, taxonomic_level)
        )]
    else:
        df1 = df11 = df1.copy()
        df1['Lineage'] = df1['Lineage']  # assume campo já preenchido
        df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
        df1 = df1.dropna(subset=['Assembly'])
        df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
        df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF','GCA'), na=False)]

        # Extração por nível taxonômico
        level = taxonomic_level.lower()
        if level == 'genus':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(
                lambda s: extract_taxonomic_group_by_ete3(s, 'genus')
            )
        else:
            df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(
                lambda x: extract_taxonomic_group(x, taxonomic_level)
            )

        df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(
            lambda x: filter_by_criteria_ete3(x, taxonomic_level)
        )]

    # Contagem e merge
    faal_counts = df1_filtered.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
    genome_counts = df1_filtered.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
    merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')

    # Filtra por número de genomas
    min_genomes = 0 if domain_name == 'Eukaryota' else 5
    merged_data = merged_data[merged_data['Genome Count'] >= min_genomes]
    if merged_data.empty:
        print("No taxonomic groups with sufficient genomes found.")
        return

    # Calcula a média de FAAL por genoma
    merged_data['Mean FAAL Count per Genome'] = \
        merged_data['Total FAAL Count'] / merged_data['Genome Count']
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Mean FAAL Count per Genome')

    # Salva resultado completo
    merged_data.to_csv('merged_data.tsv', sep='\t', index=False)

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    order_vals = top_taxonomic_groups.sort_values(
        'Mean FAAL Count per Genome', ascending=False
    )['Taxonomic_Group']
    barplot = sns.barplot(
        x='Taxonomic_Group',
        y='Mean FAAL Count per Genome',
        data=top_taxonomic_groups,
        palette='viridis',
        order=order_vals
    )
    barplot.set_xlabel(f'{taxonomic_level} Level', fontsize=14, fontweight='bold')
    barplot.set_ylabel('Mean FAAL Count per Genome', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)

    for i, grp in enumerate(order_vals):
        mean_val = top_taxonomic_groups.loc[
            top_taxonomic_groups['Taxonomic_Group']==grp,
            'Mean FAAL Count per Genome'
        ].iloc[0]
        gen_cnt = top_taxonomic_groups.loc[
            top_taxonomic_groups['Taxonomic_Group']==grp,
            'Genome Count'
        ].iloc[0]
        faal_tot = top_taxonomic_groups.loc[
            top_taxonomic_groups['Taxonomic_Group']==grp,
            'Total FAAL Count'
        ].iloc[0]
        barplot.text(i, mean_val/2, str(gen_cnt), ha="center", va="center", fontsize=10, fontweight='bold')
        barplot.text(i, mean_val, str(faal_tot), ha="center", va="bottom", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('mean_faal_per_genome.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('mean_faal_per_genome.svg', dpi=dpi, bbox_inches='tight')
    plt.savefig('mean_faal_per_genome.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 script_name.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    taxonomic_level = sys.argv[3]
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi)













