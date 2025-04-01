import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ete3 import NCBITaxa

# Inicializa NCBITaxa
ncbi = NCBITaxa()

def extract_taxonomic_group_ete3(taxid, level):
    """
    A partir de um taxid e nível taxonômico (em minúsculas, ex: 'phylum', 'order'),
    percorre a linhagem e retorna o nome do grupo no nível desejado.
    """
    lineage = ncbi.get_lineage(taxid)
    lineage_ranks = ncbi.get_rank(lineage)
    rank_names = ncbi.get_taxid_translator(lineage)
    
    level = level.lower()
    for tid in lineage:
        if lineage_ranks[tid].lower() == level:
            return rank_names[tid]
    return None

def filter_by_criteria_ete3(name, level):
    """
    Filtra os nomes dos grupos taxonômicos com base em critérios específicos
    para cada nível (Order, Family, Genus e para Phylum não há critério específico).
    O parâmetro 'level' é esperado com a primeira letra maiúscula.
    """
    if name is None:
        return False
    if level == 'Order' and name.endswith('ales'):
        return True
    if level == 'Family' and name.endswith('eae'):
        return True
    if level == 'Genus' and not (name.endswith('ales') or name.endswith('eae')):
        return True
    if level == 'Phylum':
        return True
    return False

def get_lineage_from_ncbi(species_name):
    """
    Obtém a linhagem completa para uma espécie (baseada no gênero) a partir do NCBI.
    """
    try:
        genus = species_name.split()[0]  # Extrai o gênero
        taxid_dict = ncbi.get_name_translator([genus])
        if genus in taxid_dict:
            taxid = taxid_dict[genus][0]
            lineage = ncbi.get_lineage(taxid)
            lineage_names = ncbi.get_taxid_translator(lineage)
            lineage_str = "; ".join([lineage_names[tid] for tid in lineage])
            return lineage_str
    except Exception as e:
        return None

def update_lineage(df):
    """
    Atualiza a coluna 'Lineage' do DataFrame com as informações obtidas via NCBI.
    """
    df['Lineage'] = df['Species'].apply(get_lineage_from_ncbi)
    return df

def extract_phylum_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid_dict = ncbi.get_name_translator([genus])
        if genus in taxid_dict:
            taxid = taxid_dict[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'phylum')
    except Exception as e:
        return None

def extract_order_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid_dict = ncbi.get_name_translator([genus])
        if genus in taxid_dict:
            taxid = taxid_dict[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'order')
    except Exception as e:
        return None

def extract_family_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid_dict = ncbi.get_name_translator([genus])
        if genus in taxid_dict:
            taxid = taxid_dict[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'family')
    except Exception as e:
        return None

def extract_genus_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid_dict = ncbi.get_name_translator([genus])
        if genus in taxid_dict:
            taxid = taxid_dict[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'genus')
    except Exception as e:
        return None

def extract_taxonomic_group(lineage, level):
    """
    Extrai o grupo taxonômico de um nível específico a partir de uma string de linhagem.
    Níveis esperados: Domain, Phylum, Class, Order, Family, Genus, Species.
    """
    levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    lineage_split = lineage.split(';')
    if level in levels:
        try:
            index = levels.index(level)
            return lineage_split[index].strip()
        except IndexError:
            return None
    return None

def filter_by_criteria(name, level, domain_name):
    """
    Aplica critérios de filtragem para grupos taxonômicos, dependendo do nível e do domínio.
    """
    if name is None:
        return False
    if level == 'Order' and name.endswith('ales'):
        return True
    if level == 'Family' and name.endswith('eae'):
        return True
    if level == 'Genus' and not (name.endswith('ales') or name.endswith('eae')):
        return True
    if level in ['Phylum', 'Domain']:
        return True
    return False

def generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi):
    # Carrega a tabela
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Para Eukaryota, utiliza a atualização da linhagem via ETE3
    if domain_name == 'Eukaryota':
        df1 = update_lineage(df1)
        df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
        df1 = df1.dropna(subset=['Assembly'])
        df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
        if df1_filtered.empty:
            print("Nenhum dado encontrado para o domínio fornecido na tabela.")
            return
        
        df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
        if df1_filtered.empty:
            print("Nenhum valor de Assembly iniciando com 'GCF' ou 'GCA' foi encontrado nos dados filtrados.")
            return

        # Extrai o grupo taxonômico conforme o nível especificado
        if taxonomic_level == 'Phylum':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_phylum_ete3)
        elif taxonomic_level == 'Order':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_order_ete3)
        elif taxonomic_level == 'Family':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_family_ete3)
        elif taxonomic_level == 'Genus':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_genus_ete3)
        else:
            df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
        
        df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(lambda x: filter_by_criteria_ete3(x, taxonomic_level))]
        if df1_filtered.empty:
            print("Nenhum grupo taxonômico encontrado após a filtragem.")
            return
    
    else:
        # Lógica para outros domínios (ex: Bacteria)
        df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
        df1 = df1.dropna(subset=['Assembly'])
        df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
        if df1_filtered.empty:
            print("Nenhum dado encontrado para o domínio fornecido na tabela.")
            return
        
        df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
        if df1_filtered.empty:
            print("Nenhum valor de Assembly iniciando com 'GCF' ou 'GCA' foi encontrado nos dados filtrados.")
            return
        
        df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
        df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(lambda x: filter_by_criteria(x, taxonomic_level, domain_name))]
        if df1_filtered.empty:
            print("Nenhum grupo taxonômico encontrado após a filtragem.")
            return
    
    # Agrupa e calcula as contagens de FAAL e número de genomas
    faal_counts = df1_filtered.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
    genome_counts = df1_filtered.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
    
    # Junta os dados
    merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')
    
    # Define os critérios de corte para manter grupos com genomas suficientes
    if domain_name == 'Eukaryota':
        merged_data = merged_data[merged_data['Genome Count'] > 0]
    else:
        merged_data = merged_data[merged_data['Genome Count'] > 4]
    
    if merged_data.empty:
        print("Nenhum grupo taxonômico com genomas suficientes foi encontrado.")
        return
    
    # Calcula a média de FAAL por genoma
    merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
    
    # Ordena e filtra os top N grupos com maior média
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Mean FAAL Count per Genome')
    
    # Salva a tabela com os dados de contagem
    table_filename = 'taxonomic_group_counts.tsv'
    merged_data.to_csv(table_filename, sep='\t', index=False)
    print(f"Tabela de contagens salva em '{table_filename}':")
    print(merged_data)
    
    # Plota o gráfico de barras para a média de FAAL por genoma
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    order = top_taxonomic_groups.sort_values('Mean FAAL Count per Genome', ascending=False)['Taxonomic_Group']
    barplot = sns.barplot(x='Taxonomic_Group', y='Mean FAAL Count per Genome', data=top_taxonomic_groups, palette='viridis', order=order)
    barplot.set_xlabel(f'{taxonomic_level} Level', fontsize=14, fontweight='bold')
    barplot.set_ylabel('Mean FAAL Count per Genome', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Adiciona sobre as barras o número de genomas e o total de FAALs
    for i, tax_group in enumerate(order):
        mean_faal = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == tax_group]['Mean FAAL Count per Genome'].values[0]
        genome_count = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == tax_group]['Genome Count'].values[0]
        total_faal = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == tax_group]['Total FAAL Count'].values[0]
        barplot.text(i, mean_faal / 2, f'{genome_count}', color='white', ha="center", va="center", fontsize=10, fontweight='bold')
        barplot.text(i, mean_faal, f'{total_faal}', color='black', ha="center", va="bottom", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mean_faal_per_genome.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('mean_faal_per_genome.svg', dpi=dpi, bbox_inches='tight')
    plt.savefig('mean_faal_per_genome.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Uso: python3 script_name.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    taxonomic_level = sys.argv[3]
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    
    generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi)


