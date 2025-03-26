import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ete3 import NCBITaxa
import sys

ncbi = NCBITaxa()

def extract_taxonomic_group(lineage, level):
    levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    lineage_split = lineage.split(';')
    if level in levels:
        try:
            index = levels.index(level)
            return lineage_split[index].strip()
        except IndexError:
            return 'Unknown'
    return 'Unknown'

def filter_by_criteria(name, level, domain_name):
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

def get_corrected_lineage(species, use_ete3=False):
    if use_ete3:
        try:
            genus = species.split()[0]
            taxid = ncbi.get_name_translator([genus])
            if not taxid:
                return 'Unknown'
            taxid = list(taxid.values())[0][0]
            lineage = ncbi.get_lineage(taxid)
            names = ncbi.get_taxid_translator(lineage)
            ranks = ncbi.get_rank(lineage)
            levels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            lineage_names = [names[t] for t in lineage if ranks[t] in levels]
            return '; '.join(lineage_names)
        except Exception as e:
            print(f"Error obtaining lineage with ete3 for {species}: {e}")
            return 'Unknown'
    else:
        return 'Unknown'

def update_lineage(df, use_ete3=False):
    if use_ete3:
        df['Lineage'] = df['Species'].apply(lambda species: get_corrected_lineage(species, use_ete3))
    return df

def jitter(values, jitter_amount):
    return values + np.random.uniform(-jitter_amount, jitter_amount, len(values))

def generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi):
    use_ete3 = (domain_name == 'Eukaryota')
    
    # Carrega a tabela e remove espaços extras dos nomes das colunas
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    df1.columns = df1.columns.str.strip()
    
    df1 = update_lineage(df1, use_ete3)
    df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
    df1 = df1.dropna(subset=['Assembly'])
    
    df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
    if df1_filtered.empty:
        print("No data found for the given domain.")
        return
    df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
    if df1_filtered.empty:
        print("No Assembly values starting with 'GCF' or 'GCA' found in the filtered data.")
        return

    df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
    df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(lambda x: filter_by_criteria(x, taxonomic_level, domain_name))]
    if df1_filtered.empty:
        print("No taxonomic groups found after filtering.")
        return
    
    # Verifica a coluna para o tamanho do genoma
    col_genome = "Assembly Stats Total Sequence Length MB"
    if col_genome not in df1_filtered.columns:
        print("Coluna para tamanho do genoma não encontrada. Colunas disponíveis:")
        print(df1_filtered.columns.tolist())
        return
    
    # Converte os valores para numérico removendo vírgulas, se houver
    df1_filtered[col_genome] = df1_filtered[col_genome].astype(str).str.replace(',', '')
    df1_filtered['Genome Size'] = pd.to_numeric(df1_filtered[col_genome], errors='coerce')
    df1_filtered = df1_filtered.dropna(subset=['Genome Size'])
    
    # Calcula a contagem total de FAALs e o número de genomas por grupo taxonômico
    faal_counts = df1_filtered.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
    genome_counts = df1_filtered.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
    merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')
    
    if domain_name == 'Eukaryota':
        merged_data = merged_data[merged_data['Genome Count'] > 0]
    else:
        merged_data = merged_data[merged_data['Genome Count'] > 4]
    
    if merged_data.empty:
        print("No taxonomic groups with more than one genome found.")
        return
    
    # Calcula a média de FAALs por genoma para cada grupo
    merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
    # Calcula o tamanho médio do genoma para cada grupo
    genome_size = df1_filtered.groupby('Taxonomic_Group')['Genome Size'].mean().reset_index()
    
    # Seleciona os top N grupos com maior média de FAALs por genoma
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Mean FAAL Count per Genome')
    top_taxonomic_groups = pd.merge(top_taxonomic_groups, genome_size, on='Taxonomic_Group', how='left')
    
    # Salva a tabela com os resultados
    top_taxonomic_groups.to_csv("results_table.tsv", sep="\t", index=False)
    print("Tabela gerada e salva como 'results_table.tsv'.")
    
    # Configura o estilo do gráfico para publicação (fundo branco, sem grid)
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(figsize=(14, 8) if domain_name == 'Eukaryota' else (8, 8))
    ax.set_facecolor('white')

    jitter_amount = 0.8 if taxonomic_level in ['Order', 'Genus'] else 0.2
    all_genome_sizes = []
    all_faal_counts = []
    
    # Define cor de preenchimento para os pontos (50% cinza)
    gray_fill = '0.5'
    
    # Loop para plotar cada grupo taxonômico com jitter para melhor visualização
    for idx, taxonomic_group in enumerate(top_taxonomic_groups['Taxonomic_Group']):
        group_data = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == taxonomic_group]
        jittered_genome_size = jitter(group_data['Genome Size'], jitter_amount)
        jittered_faal_count = jitter(group_data['Mean FAAL Count per Genome'], jitter_amount)
        
        # Conversão do tamanho do genoma para exibição:
        if domain_name == 'Eukaryota':
            # Convertendo de MB para Gb
            jittered_genome_size = np.round(jittered_genome_size / 1000, 2)
        else:
            # Convertendo bases para Mb
            jittered_genome_size = np.round(jittered_genome_size / 1e6, 2)
        
        all_genome_sizes.extend(jittered_genome_size)
        all_faal_counts.extend(jittered_faal_count)
        
        ax.scatter(jittered_genome_size, jittered_faal_count,
                   s=150, facecolor=gray_fill, edgecolor='black', marker='o', zorder=3)
        
        # Anota os três primeiros grupos taxonômicos dentro do gráfico
        if idx < 3:
            x_mean = np.mean(jittered_genome_size)
            y_mean = np.mean(jittered_faal_count)
            ax.annotate(taxonomic_group, (x_mean, y_mean),
                        textcoords="offset points", xytext=(5, 5),
                        color='black', fontsize=14, fontweight='bold', zorder=4)
    
    # Configura os limites e os ticks dos eixos
    # Para o eixo x (tamanho médio do genoma)
    if domain_name == 'Eukaryota':
        ax.set_xlim(2, 10)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_xlabel('Average Genome Size (Gb)', fontsize=16, fontweight='bold')
    else:
        ax.set_xlim(2, 10)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_xlabel('Average Genome Size (Mb)', fontsize=16, fontweight='bold')
    
    # Para o eixo y (média de FAALs por genoma)
    if domain_name == 'Eukaryota':
        ax.set_ylim(2, 10)
    else:
        ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylabel('Average FAAL Counts', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    fig.tight_layout()
    fig.savefig('taxonomic_analysis_plot.png', dpi=dpi, bbox_inches='tight')
    fig.savefig('taxonomic_analysis_plot.svg', dpi=dpi, bbox_inches='tight')
    fig.savefig('taxonomic_analysis_plot.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 Scatterplot_S2C.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    taxonomic_level = sys.argv[3]
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    
    generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi)



