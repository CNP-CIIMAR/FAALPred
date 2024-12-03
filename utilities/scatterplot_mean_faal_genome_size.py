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

def generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi):
    use_ete3 = (domain_name == 'Eukaryota')
    
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
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
    
    merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
    
    df1_filtered['Genome Size'] = pd.to_numeric(df1_filtered['Assembly Stats Total Sequence Length MB'], errors='coerce')
    
    df1_filtered = df1_filtered.dropna(subset=['Genome Size'])
    
    genome_size = df1_filtered.groupby('Taxonomic_Group')['Genome Size'].mean().reset_index()
    
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Mean FAAL Count per Genome')
    
    top_taxonomic_groups = pd.merge(top_taxonomic_groups, genome_size, on='Taxonomic_Group', how='left')
    
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(14, 8) if domain_name == 'Eukaryota' else (8, 8))

    num_groups = len(top_taxonomic_groups['Taxonomic_Group'])
    colors = sns.color_palette("tab20", 20) + sns.color_palette("Set1", 9) + sns.color_palette("Set2", 8) + sns.color_palette("Dark2", 8)
    
    excluded_colors = [
        (1.0, 0.498, 0.498), (1.0, 0.6, 0.6), (0.86, 0.196, 0.184), (0.839, 0.153, 0.157), # Similar pink/rose
        (0.2, 0.8, 0.2), (0.2, 1.0, 0.2), (0.2, 0.6, 0.2)  # Similar greens
    ]
    colors = [color for color in colors if color not in excluded_colors]
    
    additional_colors = [(0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.6, 0.6, 0.6), (0.8, 0.8, 0.8)]  # Grayscale colors
    colors.extend(additional_colors)
    
    colors = colors[:num_groups]

    jitter_amount = 0.8 if taxonomic_level in ['Order', 'Genus'] else 0.2

    for taxonomic_group, color in zip(top_taxonomic_groups['Taxonomic_Group'], colors):
        group_data = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == taxonomic_group]
        jittered_genome_size = jitter(group_data['Genome Size'], jitter_amount)
        jittered_faal_count = jitter(group_data['Mean FAAL Count per Genome'], jitter_amount)
        if domain_name == 'Eukaryota':
            jittered_genome_size = np.round(jittered_genome_size / 1000, 2)  # Convert Mb to Gb and round to 2 decimals
        plt.scatter(jittered_genome_size, jittered_faal_count,
                    s=100, color=color, label=f'{taxonomic_group} (N={group_data["Genome Count"].values[0]})', 
                    edgecolor='black', marker='o')

    if domain_name == 'Eukaryota':
        plt.xlabel('Average Genome Size (Gb)', fontsize=14, fontweight='bold')
        min_genome_size = np.round(top_taxonomic_groups['Genome Size'].min() / 1000, 2)
        max_genome_size = np.round(top_taxonomic_groups['Genome Size'].max() / 1000, 2)
        plt.xlim(min_genome_size * 0.95, max_genome_size * 1.05)  # Ajuste automÃ¡tico da escala do eixo x
        plt.xticks(np.round(np.linspace(min_genome_size * 0.95, max_genome_size * 1.05, 10), 2), fontsize=14, fontweight='bold')
    else:
        plt.xlabel('Average Genome Size (Mb)', fontsize=14, fontweight='bold')
        max_genome_size = top_taxonomic_groups['Genome Size'].max()
        plt.xlim(0, max_genome_size + 0.5)  # Default x-axis limit for Bacteria
        plt.xticks(np.arange(0, max_genome_size + 1, 2), fontsize=14, fontweight='bold')
    
    plt.ylabel('Average FAAL Counts', fontsize=14, fontweight='bold')

    def format_taxonomic_group_name(name, level):
        return name

    legend_labels = [format_taxonomic_group_name(taxonomic_group, taxonomic_level) for taxonomic_group in top_taxonomic_groups['Taxonomic_Group']]
    
    if domain_name == 'Eukaryota' and taxonomic_level in ['Order', 'Genus']:
        plt.legend(title=taxonomic_level, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=12, title_fontsize='13', ncol=2)
    else:
        plt.legend(title=taxonomic_level, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, title_fontsize='13')

    if domain_name != 'Eukaryota':
        plt.title(f'FAAL Counts by {taxonomic_level} and Genome Size', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0.305, 0.321, 0.579, 0.788])
    plt.subplots_adjust(left=0.305, bottom=0.321, right=0.579, top=0.788, wspace=0.762, hspace=0.643)
    plt.grid(True)
    plt.savefig('taxonomic_analysis_plot.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('taxonomic_analysis_plot.svg', dpi=dpi, bbox_inches='tight')
    plt.savefig('taxonomic_analysis_plot.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 scatterplot_counts_faal.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    taxonomic_level = sys.argv[3]
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    
    generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi)

