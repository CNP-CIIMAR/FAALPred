import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ete3 import NCBITaxa
import sys
import string

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
            lineage_names = [names[taxid] for taxid in lineage if ranks[taxid] in levels]
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

def generate_subplot_for_taxonomic_level(ax, df_filtered, domain_name, taxonomic_level, top_n, dpi, panel_label):
    # Cria dataframe para o nível taxonômico corrente
    df_level = df_filtered.copy()
    df_level['Taxonomic_Group'] = df_level['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
    df_level = df_level[df_level['Taxonomic_Group'].apply(lambda x: filter_by_criteria(x, taxonomic_level, domain_name))]
    
    if df_level.empty:
        print(f"Não foram encontrados grupos taxonômicos para o nível {taxonomic_level}.")
        return
    
    # Contagens e médias
    faal_counts = df_level.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
    genome_counts = df_level.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
    merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')
    
    if domain_name == 'Eukaryota':
        merged_data = merged_data[merged_data['Genome Count'] > 0]
    else:
        merged_data = merged_data[merged_data['Genome Count'] > 4]
    
    if merged_data.empty:
        print(f"Nenhum grupo taxonômico com genomas suficientes para o nível {taxonomic_level}.")
        return
    
    merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
    
    df_level['Genome Size'] = pd.to_numeric(df_level['Assembly Stats Total Sequence Length MB'], errors='coerce')
    df_level = df_level.dropna(subset=['Genome Size'])
    genome_size = df_level.groupby('Taxonomic_Group')['Genome Size'].mean().reset_index()
    
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Mean FAAL Count per Genome')
    top_taxonomic_groups = pd.merge(top_taxonomic_groups, genome_size, on='Taxonomic_Group', how='left')
    
    # Configuração de cores
    num_groups = len(top_taxonomic_groups['Taxonomic_Group'])
    colors = (sns.color_palette("tab20", 20) + sns.color_palette("Set1", 9) +
              sns.color_palette("Set2", 8) + sns.color_palette("Dark2", 8))
    excluded_colors = [
        (1.0, 0.498, 0.498), (1.0, 0.6, 0.6), (0.86, 0.196, 0.184), (0.839, 0.153, 0.157),
        (0.2, 0.8, 0.2), (0.2, 1.0, 0.2), (0.2, 0.6, 0.2)
    ]
    colors = [color for color in colors if color not in excluded_colors]
    additional_colors = [(0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.6, 0.6, 0.6), (0.8, 0.8, 0.8)]
    colors.extend(additional_colors)
    colors = colors[:num_groups]
    
    jitter_amount = 0.8 if taxonomic_level in ['Order', 'Genus'] else 0.2
    
    # Plot dos dados com jitter
    for taxonomic_group, color in zip(top_taxonomic_groups['Taxonomic_Group'], colors):
        group_data = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == taxonomic_group]
        jittered_genome_size = jitter(group_data['Genome Size'], jitter_amount)
        jittered_faal_count = jitter(group_data['Mean FAAL Count per Genome'], jitter_amount)
        if domain_name == 'Eukaryota':
            jittered_genome_size = np.round(jittered_genome_size / 1000, 2)  # Converter Mb para Gb
        ax.scatter(jittered_genome_size, jittered_faal_count,
                   s=100, color=color,
                   label=f'{taxonomic_group} (N={group_data["Genome Count"].values[0]})', 
                   edgecolor='black', marker='o')
    
    # Configura os eixos
    if domain_name == 'Eukaryota':
        ax.set_xlabel('Average Genome Size (Gb)', fontsize=16, fontweight='bold')
        min_genome_size = np.round(top_taxonomic_groups['Genome Size'].min() / 1000, 2)
        max_genome_size = np.round(top_taxonomic_groups['Genome Size'].max() / 1000, 2)
        ax.set_xlim(min_genome_size * 0.95, max_genome_size * 1.05)
        ax.set_xticks(np.round(np.linspace(min_genome_size * 0.95, max_genome_size * 1.05, 10), 2))
        ax.tick_params(axis='x', labelsize=14)
    else:
        ax.set_xlabel('Average Genome Size (Mb)', fontsize=16, fontweight='bold')
        max_genome_size = top_taxonomic_groups['Genome Size'].max()
        ax.set_xlim(0, max_genome_size + 0.5)
        ax.set_xticks(np.arange(0, max_genome_size + 1, 2))
        ax.tick_params(axis='x', labelsize=14)
        
    ax.set_ylabel('Average FAAL Counts', fontsize=16, fontweight='bold')
    
    # Configura a legenda: título ajustado e duas colunas
    if domain_name == 'Eukaryota' and taxonomic_level == 'Phylum':
        legend_title = 'Clade/Phylum'
        legend_bbox = (1.1, 0.5)  # desloca a legenda mais para a direita
    else:
        legend_title = taxonomic_level
        legend_bbox = (1.05, 0.5)
    ax.legend(title=legend_title, loc='center left', bbox_to_anchor=legend_bbox,
              fontsize=14, title_fontsize=16, ncol=2)
    
    if domain_name != 'Eukaryota':
        ax.set_title(f'FAAL Counts by {taxonomic_level} and Genome Size', fontsize=18, fontweight='bold')
    
    # Adiciona a label do painel fora da área do eixo, no canto superior esquerdo (lado externo)
    ax.text(-0.20, 1.05, panel_label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')
    
    ax.grid(True)

def generate_plots_for_taxonomic_levels(table1_path, domain_name, taxonomic_levels, top_n, dpi):
    use_ete3 = (domain_name == 'Eukaryota')
    df = pd.read_csv(table1_path, sep='\t', low_memory=False)
    df = update_lineage(df, use_ete3)
    df = df[~df['Lineage'].str.contains('environmental samples', na=False)]
    df = df.dropna(subset=['Assembly'])
    df_filtered = df[df['Lineage'].str.contains(domain_name, na=False)].copy()
    if df_filtered.empty:
        print("Nenhum dado encontrado para o domínio informado.")
        sys.exit(1)
    df_filtered = df_filtered[df_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
    if df_filtered.empty:
        print("Nenhum valor de Assembly iniciando com 'GCF' ou 'GCA' encontrado nos dados filtrados.")
        sys.exit(1)
    
    n_levels = len(taxonomic_levels)
    # Define o tamanho total da figura com base no domínio e número de níveis
    if domain_name == 'Eukaryota':
        fig, axes = plt.subplots(n_levels, 1, figsize=(14, 8 * n_levels))
    else:
        fig, axes = plt.subplots(n_levels, 1, figsize=(8, 8 * n_levels))
    
    # Se for apenas um subplot, converte axes para lista
    if n_levels == 1:
        axes = [axes]
    
    panel_labels = list(string.ascii_uppercase)
    
    for i, taxonomic_level in enumerate(taxonomic_levels):
        print(f"Gerando figura para o nível taxonômico: {taxonomic_level}")
        generate_subplot_for_taxonomic_level(axes[i], df_filtered, domain_name, taxonomic_level, top_n, dpi, panel_labels[i])
    
    # Ajusta o layout para dar espaço à esquerda (para as labels dos painéis)
    plt.subplots_adjust(left=0.25, hspace=0.4)
    
    # Salva a figura combinada
    plt.savefig('taxonomic_analysis_combined_plot.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('taxonomic_analysis_combined_plot.svg', dpi=dpi, bbox_inches='tight')
    plt.savefig('taxonomic_analysis_combined_plot.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # A quantidade mínima de argumentos agora é 6 (com possibilidade de mais níveis taxonômicos)
    if len(sys.argv) < 6:
        print("Usage: python3 scatterplot_counts_faal.py <table1.tsv> <Domain> <Taxonomic Level(s)> <Top N> <DPI>")
        sys.exit(1)
    
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    # Todos os argumentos entre o domínio e os dois últimos são níveis taxonômicos
    taxonomic_levels = sys.argv[3:-2]
    top_n = int(sys.argv[-2])
    dpi = int(sys.argv[-1])
    
    generate_plots_for_taxonomic_levels(table1_path, domain_name, taxonomic_levels, top_n, dpi)
