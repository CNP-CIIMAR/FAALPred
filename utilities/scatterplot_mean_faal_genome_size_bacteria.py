import pandas as pd
import matplotlib.pyplot as plt
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
        genus = species_name.split()[0]
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

def generate_panel(table1_path, domain_name, taxonomic_levels, top_n, dpi):
    """
    Gera um painel com scatterplots lado a lado para os níveis taxonômicos especificados.
    Cada ponto representa a média do Genome Size (em MB) e o Mean FAAL Count per Genome para o grupo.
    taxonomic_levels deve ser uma lista de strings (ex: ['Phylum', 'Order', 'Genus']).
    """
    # Carrega a tabela
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Se o domínio for Eukaryota, atualiza a coluna 'Lineage'
    if domain_name == 'Eukaryota':
        df1 = update_lineage(df1)
    
    # Remove registros indesejados e sem Assembly
    df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
    df1 = df1.dropna(subset=['Assembly'])
    
    # Filtra por domínio
    df1 = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
    if df1.empty:
        print("Nenhum dado encontrado para o domínio fornecido na tabela.")
        return
    
    # Filtra Assembly que iniciam com 'GCF' ou 'GCA'
    df1 = df1[df1['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
    if df1.empty:
        print("Nenhum valor de Assembly iniciando com 'GCF' ou 'GCA' foi encontrado nos dados filtrados.")
        return
    
    # Prepara o painel com subplots (um para cada nível taxonômico)
    num_plots = len(taxonomic_levels)
    fig, axes = plt.subplots(ncols=num_plots, nrows=1, figsize=(6 * num_plots, 8), sharex=False, sharey=False)
    
    # Se só houver um subplot, encapsula-o em uma lista para iteração
    if num_plots == 1:
        axes = [axes]
    
    # Define os ticks fixos
    ticks = np.arange(0, 12, 2)
    
    # Para cada nível taxonômico, processa os dados e gera o scatterplot
    for ax, level in zip(axes, taxonomic_levels):
        df_level = df1.copy()
        
        # Extrai o grupo taxonômico conforme o nível e domínio
        if domain_name == 'Eukaryota':
            if level == 'Phylum':
                df_level['Taxonomic_Group'] = df_level['Species'].apply(extract_phylum_ete3)
            elif level == 'Order':
                df_level['Taxonomic_Group'] = df_level['Species'].apply(extract_order_ete3)
            elif level == 'Family':
                df_level['Taxonomic_Group'] = df_level['Species'].apply(extract_family_ete3)
            elif level == 'Genus':
                df_level['Taxonomic_Group'] = df_level['Species'].apply(extract_genus_ete3)
            else:
                df_level['Taxonomic_Group'] = df_level['Lineage'].apply(lambda x: extract_taxonomic_group(x, level))
            
            df_level = df_level[df_level['Taxonomic_Group'].apply(lambda x: filter_by_criteria_ete3(x, level))]
        else:
            df_level['Taxonomic_Group'] = df_level['Lineage'].apply(lambda x: extract_taxonomic_group(x, level))
            df_level = df_level[df_level['Taxonomic_Group'].apply(lambda x: filter_by_criteria(x, level, domain_name))]
        
        if df_level.empty:
            ax.text(0.5, 0.5, f"Nenhum grupo encontrado para {level}", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
        
        # Agrupa e calcula contagens de FAAL e genomas
        faal_counts = df_level.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
        genome_counts = df_level.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
        merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')
        
        # Critério de corte para genomas
        if domain_name == 'Eukaryota':
            merged_data = merged_data[merged_data['Genome Count'] > 0]
        else:
            merged_data = merged_data[merged_data['Genome Count'] > 4]
        
        if merged_data.empty:
            ax.text(0.5, 0.5, f"Nenhum grupo com genomas suficientes para {level}", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
        
        merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
        
        # Converte o Genome Size para numérico e calcula a média por grupo
        df_level['Genome Size'] = pd.to_numeric(
            df_level['Assembly Stats Total Sequence Length MB'].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip(),
            errors='coerce'
        )
        df_level = df_level.dropna(subset=['Genome Size'])
        genome_size = df_level.groupby('Taxonomic_Group')['Genome Size'].mean().reset_index()
        
        # Junta os dados e seleciona os top N grupos
        top_taxonomic_groups = pd.merge(merged_data, genome_size, on='Taxonomic_Group', how='left')
        top_taxonomic_groups = top_taxonomic_groups.nlargest(top_n, 'Mean FAAL Count per Genome')
        
        # Salva a tabela para este nível (opcional)
        table_filename = f'taxonomic_mean_faal_and_genome_size_{level}.tsv'
        top_taxonomic_groups.to_csv(table_filename, sep='\t', index=False)
        print(f"Tabela de médias para {level} salva em {table_filename}")
        
        # Cada ponto será plotado na interseção da média do Genome Size (em MB) e do Mean FAAL Count per Genome
        for idx, (_, row) in enumerate(top_taxonomic_groups.iterrows()):
            x_val = row['Genome Size']    # Valor em MB
            y_val = row['Mean FAAL Count per Genome']
            ax.scatter(x_val, y_val,
                       s=150, facecolor=(0.7, 0.7, 0.7), edgecolor='black', marker='o', zorder=3)
            # Anota os três primeiros grupos
            if idx < 3:
                ax.annotate(row['Taxonomic_Group'], (x_val, y_val),
                            textcoords="offset points", xytext=(5, 5),
                            color='black', fontsize=12, fontweight='bold', zorder=4)
        
        # Configura os rótulos, ticks e limites dos eixos
        ax.set_xlabel('Average Genome Size (MB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean FAAL Count per Genome', fontsize=14, fontweight='bold')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        # Define os tick labels, ocultando o valor 0
        new_xticks = ["" if tick == 0 else tick for tick in ticks]
        new_yticks = ["" if tick == 0 else tick for tick in ticks]
        ax.set_xticklabels(new_xticks)
        ax.set_yticklabels(new_yticks)
        
        # Define os limites dos eixos conforme o nível taxonômico
        if level == 'Phylum':
            ax.set_xlim(0, 10.5)
            ax.set_ylim(0, 10.5)
        elif level == 'Order':
            ax.set_xlim(0, 11)
            ax.set_ylim(0, 11)
        elif level == 'Genus':
            ax.set_xlim(0, 13)
            ax.set_ylim(0, 13)
        else:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
        
        ax.grid(False)
    
    plt.tight_layout()
    # Salva o painel nos formatos desejados
    plt.savefig('scatterplot_panel_faal_vs_genome.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('scatterplot_panel_faal_vs_genome.svg', dpi=dpi, bbox_inches='tight')
    plt.savefig('scatterplot_panel_faal_vs_genome.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Uso: python3 script_name.py <table1.tsv> <Domain> <Taxonomic Levels (comma-separated)> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    # O argumento de níveis taxonômicos deve ser uma string separada por vírgula, ex: "Phylum,Order,Genus"
    taxonomic_levels = [level.strip() for level in sys.argv[3].split(',')]
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    
    generate_panel(table1_path, domain_name, taxonomic_levels, top_n, dpi)





