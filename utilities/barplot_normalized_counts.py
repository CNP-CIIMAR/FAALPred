import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from ete3 import NCBITaxa
import numpy as np

ncbi = NCBITaxa()

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

def filter_by_criteria(name, level):
    if name is None:
        return False
    if level == 'Order' and name.endswith('ales'):
        return True
    if level == 'Family' and name.endswith('eae'):
        return True
    if level == 'Genus' and not (name.endswith('ales') or name.endswith('eae')):
        return True
    if level in ['Phylum', 'Domain']:
        return True  # No specific filtering criteria for Phylum or Domain
    return False

def get_corrected_lineage(taxid):
    try:
        lineage = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage)
        ranks = ncbi.get_rank(lineage)
        levels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        lineage_names = [names[taxid] for taxid in lineage if ranks[taxid] in levels]
        return '; '.join(lineage_names)
    except:
        return None

def update_lineage(df):
    df['Lineage'] = df['Organism Taxonomic ID'].apply(get_corrected_lineage)
    return df

def generate_filtered_table_and_graphs(table1_path, table2_path, domain_name, taxonomic_level, top_n, dpi):
    # Load the tables
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    df2 = pd.read_csv(table2_path, sep='\t', low_memory=False)
    
    print("Tabela 1 carregada:")
    print(df1.head())
    
    print("Tabela 2 carregada:")
    print(df2.head())
    
    # Update the lineage in table 1 using ETE3
    df1 = update_lineage(df1)
    
    print("Tabela 1 com a coluna 'Lineage' atualizada:")
    print(df1[['Organism Name', 'Organism Taxonomic ID', 'Lineage']].head())
    
    # Filter table 1 by domain
    df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
    
    print(f"Dados filtrados da Tabela 1 para o domÃ­nio {domain_name}:")
    print(df1_filtered.head())
    
    if df1_filtered.empty:
        print("No data found for the provided domain in table 1.")
        return
    
    # Filter 'Assembly' values that start with 'GCF' or 'GCA'
    df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
    
    print("Dados filtrados da Tabela 1 com valores de 'Assembly' comeÃ§ando com 'GCF' ou 'GCA':")
    print(df1_filtered.head())
    
    if df1_filtered.empty:
        print("No Assembly values starting with 'GCF' or 'GCA' found in the filtered data.")
        return

    # Extract taxonomic groups from Corrected Lineages
    df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
    
    # Apply filtering criteria based on taxonomic level
    df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(lambda x: filter_by_criteria(x, taxonomic_level))]
    
    print("Dados da Tabela 1 apÃ³s aplicaÃ§Ã£o dos critÃ©rios de filtragem:")
    print(df1_filtered.head())
    
    if df1_filtered.empty:
        print("No taxonomic groups found after filtering.")
        return
    
    # Count FAALs per Taxonomic Group
    faal_counts = df1_filtered.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
    
    # Calculate the number of unique genomes per taxonomic group
    genome_counts = df1_filtered.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
    
    # Merge FAAL counts and genome counts
    merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')
    
    # Calculate the mean FAAL count per genome
    merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
    
    # Filter for the top N most abundant taxonomic groups
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Total FAAL Count')
    
    # Debugging output
    print(f"Top {top_n} taxonomic groups based on FAAL count:")
    print(top_taxonomic_groups)
    
    # Filter table 2 based on taxonomic groups where FAAL was found
    taxonomic_groups = top_taxonomic_groups['Taxonomic_Group'].tolist()
    
    def match_taxonomic_group(lineage, groups):
        for group in groups:
            if group in lineage:
                return group
        return None

    df2_filtered = df2[df2['Lineage'].apply(lambda x: match_taxonomic_group(x, taxonomic_groups)).notnull()].copy()
    df2_filtered['Taxonomic_Group'] = df2_filtered['Lineage'].apply(lambda x: match_taxonomic_group(x, taxonomic_groups))
    
    print("Dados filtrados da Tabela 2 baseados nos grupos taxonÃ´micos da Tabela 1:")
    print(df2_filtered.head(10))  # Mostrar mais linhas para depuraÃ§Ã£o
    
    if df2_filtered.empty:
        print("No data found for the provided taxonomic groups in table 2.")
        return
    
    # Save the filtered table
    output_filtered_table = 'Taxonomic_groups_with_FAAL.tsv'
    df2_filtered.to_csv(output_filtered_table, sep='\t', index=False)
    
    # Get total number of sequenced genomes per taxonomic group (including those without FAAL)
    genome_counts_total = df2_filtered.groupby('Taxonomic_Group')['Assembly Accession'].nunique().reset_index(name='Total Genome Count')
    
    # Debugging output
    print("Total genome counts for each taxonomic group in table 2:")
    print(genome_counts_total.head())
    
    # Merge the data with total genome counts
    normalized_data = pd.merge(top_taxonomic_groups, genome_counts_total, on='Taxonomic_Group', how='left').fillna(0)
    
    # Calculate the normalized FAAL count per genome (as a percentage)
    normalized_data['Normalized FAAL Count per Genome (%)'] = (normalized_data['Genome Count'] / normalized_data['Total Genome Count'].replace({0: np.nan})) * 100
    
    # Replace infinite values with zeros
    normalized_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Ensure all values are finite
    normalized_data = normalized_data[np.isfinite(normalized_data['Normalized FAAL Count per Genome (%)'])]
    
    # Save the results to .tsv files
    merged_data.to_csv('merged_data.tsv', sep='\t', index=False)
    normalized_data.to_csv('normalized_data.tsv', sep='\t', index=False)
    
    # Plot raw FAAL counts
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # Order by Total FAAL Count
    order = top_taxonomic_groups.sort_values('Total FAAL Count', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Total FAAL Count', y='Taxonomic_Group', data=top_taxonomic_groups, ax=ax[0], palette='viridis', order=order)
    ax[0].set_xlabel('Fatty Acyl AMP Ligase (FAALs) Counts', fontsize=14)
    ax[0].set_ylabel(f'{taxonomic_level} Level', fontsize=14)
    
    # Set x-axis ticks for the first plot based on the domain
    if domain_name.lower() == 'eukaryota':
        max_faal_count = top_taxonomic_groups['Total FAAL Count'].max()
        interval = max(1, (max_faal_count // 10))
        ax[0].set_xticks(range(0, max_faal_count + interval, interval))
    else:
        max_faal_count = top_taxonomic_groups['Total FAAL Count'].max()
        ax[0].set_xticks(range(0, int(max_faal_count) + 2000, 2000))
    
    # Add mean FAAL count per genome to the right of the bars
    for i in range(len(top_taxonomic_groups)):
        ax[0].text(top_taxonomic_groups['Total FAAL Count'].iloc[i] + top_taxonomic_groups['Total FAAL Count'].max() * 0.01, i, 
                   f'{top_taxonomic_groups["Mean FAAL Count per Genome"].iloc[i]:.2f}', 
                   color='black', ha="left", va="center", fontsize=10, fontweight='bold')

    # Add 'A)' to the left of the first plot
    ax[0].annotate('A)', xy=(-0.15, 1.02), xycoords='axes fraction', fontsize=16, fontweight='bold')
    
    # Order by Normalized FAAL Count per Genome
    order_normalized = normalized_data.sort_values('Normalized FAAL Count per Genome (%)', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Normalized FAAL Count per Genome (%)', y='Taxonomic_Group', data=normalized_data, ax=ax[1], palette='viridis', order=order_normalized)
    ax[1].set_xlabel('Normalized (%) Fatty Acyl AMP Ligase (FAALs) Counts', fontsize=14)
    ax[1].set_ylabel(f'{taxonomic_level} Level', fontsize=14)
    
    # Set x-axis ticks for the second plot (based on the percentage values)
    max_normalized_count = normalized_data['Normalized FAAL Count per Genome (%)'].max()
    if np.isfinite(max_normalized_count) and max_normalized_count > 0:
        ax[1].set_xticks(range(0, int(max_normalized_count) + 10, 10))
    else:
        ax[1].set_xticks([0])
    
    # Add genome counts inside and above the bars in the second plot
    for i in range(len(order_normalized)):
        taxonomic_group = order_normalized.iloc[i]
        total_genome_count = normalized_data[normalized_data['Taxonomic_Group'] == taxonomic_group]['Total Genome Count'].values[0]
        genome_count_with_faal = normalized_data[normalized_data['Taxonomic_Group'] == taxonomic_group]['Genome Count'].values[0]
        
        ax[1].text(normalized_data[normalized_data['Taxonomic_Group'] == taxonomic_group]['Normalized FAAL Count per Genome (%)'].values[0] / 2, i, 
                   f'{genome_count_with_faal:,}', 
                   color='white', ha="center", va="center", fontsize=10, fontweight='bold')
        
        ax[1].text(normalized_data[normalized_data['Taxonomic_Group'] == taxonomic_group]['Normalized FAAL Count per Genome (%)'].values[0] + normalized_data['Normalized FAAL Count per Genome (%)'].max() * 0.01, i, 
                   f'{total_genome_count:,}', 
                   color='black', ha="left", va="center", fontsize=10, fontweight='bold')

    # Add 'B)' to the left of the second plot
    ax[1].annotate('B)', xy=(-0.15, 1.02), xycoords='axes fraction', fontsize=16, fontweight='bold')

    # Apply italics to y-axis labels if taxonomic level is Genus or Species
    if taxonomic_level in ['Genus', 'Species']:
        ax[0].set_yticklabels(ax[0].get_yticklabels(), style='italic')
        ax[1].set_yticklabels(ax[1].get_yticklabels(), style='italic')
    
    plt.tight_layout()
    plt.savefig('ranking_FAAL_combined.png', dpi=dpi)
    plt.savefig('ranking_FAAL_combined.svg', dpi=dpi)
    plt.savefig('ranking_FAAL_combined.jpeg', dpi=dpi)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 barplot_normalized_counts_faal.py <table1.tsv> <table2.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    table2_path = sys.argv[2]
    domain_name = sys.argv[3]
    taxonomic_level = sys.argv[4]
    top_n = int(sys.argv[5])
    dpi = int(sys.argv[6])
    generate_filtered_table_and_graphs(table1_path, table2_path, domain_name, taxonomic_level, top_n, dpi)

