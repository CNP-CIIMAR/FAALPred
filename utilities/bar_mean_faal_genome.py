import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ete3 import NCBITaxa

# Initialize NCBITaxa
ncbi = NCBITaxa()

def extract_taxonomic_group_ete3(taxid, level):
    lineage = ncbi.get_lineage(taxid)
    lineage_ranks = ncbi.get_rank(lineage)
    rank_names = ncbi.get_taxid_translator(lineage)
    
    for taxid in lineage:
        if lineage_ranks[taxid] == level:
            return rank_names[taxid]
    return None

def filter_by_criteria_ete3(name, level):
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
    try:
        genus = species_name.split()[0]  # Extract the genus part of the species name
        taxid = ncbi.get_name_translator([genus])
        if genus in taxid:
            taxid = taxid[genus][0]
            lineage = ncbi.get_lineage(taxid)
            lineage_names = ncbi.get_taxid_translator(lineage)
            lineage_str = "; ".join([lineage_names[taxid] for taxid in lineage])
            return lineage_str
    except:
        return None

def update_lineage(df):
    df['Lineage'] = df['Species'].apply(get_lineage_from_ncbi)
    return df

def extract_phylum_ete3(species_name):
    try:
        genus = species_name.split()[0]  # Extract the genus part of the species name
        taxid = ncbi.get_name_translator([genus])
        if genus in taxid:
            taxid = taxid[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'phylum')
    except:
        return None

def extract_order_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid = ncbi.get_name_translator([genus])
        if genus in taxid:
            taxid = taxid[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'order')
    except:
        return None

def extract_family_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid = ncbi.get_name_translator([genus])
        if genus in taxid:
            taxid = taxid[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'family')
    except:
        return None

def extract_genus_ete3(species_name):
    try:
        genus = species_name.split()[0]
        taxid = ncbi.get_name_translator([genus])
        if genus in taxid:
            taxid = taxid[genus][0]
            return extract_taxonomic_group_ete3(taxid, 'genus')
    except:
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
        return True  # Sem critérios específicos para Phylum ou Domain
    return False

def generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi):
    # Load the table
    df1 = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Use ETE3 logic for Eukaryota
    if domain_name == 'Eukaryota':
        df1 = update_lineage(df1)
        
        # Remove rows where 'Lineage' contains 'environmental samples'
        df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
        
        # Remove rows where 'Assembly' is NaN
        df1 = df1.dropna(subset=['Assembly'])
        
        # Filter table by domain
        df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
        
        if df1_filtered.empty:
            print("No data found for the provided domain in table 1.")
            return
        
        # Filter 'Assembly' values that start with 'GCF' or 'GCA'
        df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
        
        if df1_filtered.empty:
            print("No Assembly values starting with 'GCF' or 'GCA' found in the filtered data.")
            return

        # Extract taxonomic groups based on provided criteria
        if taxonomic_level == 'Phylum':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_phylum_ete3)
        elif taxonomic_level == 'Order':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_order_ete3)
        elif taxonomic_level == 'Family':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_family_ete3)
        elif taxonomic_level == 'Genus':
            df1_filtered['Taxonomic_Group'] = df1_filtered['Species'].apply(extract_genus_ete3)
        else:
            df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(lambda x: extract_taxonomic_group_ete3(x, taxonomic_level))
        
        # Apply filtering criteria based on taxonomic level
        df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(lambda x: filter_by_criteria_ete3(x, taxonomic_level))]
        
        if df1_filtered.empty:
            print("No taxonomic groups found after filtering.")
            return
    
    else:
        # Logic for other domains (e.g., Bacteria)
        # Remove rows where 'Phylum' is 'environmental samples'
        df1 = df1[~df1['Lineage'].str.contains('environmental samples', na=False)]
        
        # Remove rows where 'Assembly' is NaN
        df1 = df1.dropna(subset=['Assembly'])
        
        # Filter by domain
        df1_filtered = df1[df1['Lineage'].str.contains(domain_name, na=False)].copy()
        
        if df1_filtered.empty:
            print("No data found for the provided domain in table 1.")
            return
        
        # Filter 'Assembly' values that start with 'GCF' or 'GCA'
        df1_filtered = df1_filtered[df1_filtered['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]
        
        if df1_filtered.empty:
            print("No Assembly values starting with 'GCF' or 'GCA' found in the filtered data.")
            return

        # Extract taxonomic groups from lineages
        df1_filtered['Taxonomic_Group'] = df1_filtered['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
        
        # Apply filtering criteria based on taxonomic level and domain
        df1_filtered = df1_filtered[df1_filtered['Taxonomic_Group'].apply(lambda x: filter_by_criteria(x, taxonomic_level, domain_name))]
        
        if df1_filtered.empty:
            print("No taxonomic groups found after filtering.")
            return
    
    # Count FAALs per Taxonomic Group
    faal_counts = df1_filtered.groupby('Taxonomic_Group').size().reset_index(name='Total FAAL Count')
    
    # Calculate the number of unique genomes per taxonomic group
    genome_counts = df1_filtered.groupby('Taxonomic_Group')['Assembly'].nunique().reset_index(name='Genome Count')
    
    # Merge FAAL counts and genome counts
    merged_data = pd.merge(faal_counts, genome_counts, on='Taxonomic_Group')
    
    # Remove groups with less than 5 genomes deposited
    if domain_name == 'Eukaryota':
        merged_data = merged_data[merged_data['Genome Count'] > 0]  # Less restrictive for Eukaryota
    else:
        merged_data = merged_data[merged_data['Genome Count'] > 4]  # More restrictive for other domains
    
    if merged_data.empty:
        print("No taxonomic groups with more than one genome found.")
        return
    
    # Calculate the mean FAAL count per genome
    merged_data['Mean FAAL Count per Genome'] = merged_data['Total FAAL Count'] / merged_data['Genome Count']
    
    # Filter for the top N most abundant taxonomic groups
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Mean FAAL Count per Genome')
    
    # Save the results to .tsv files
    merged_data.to_csv('merged_data.tsv', sep='\t', index=False)
    
    # Plot Mean FAAL counts per genome
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Order by Mean FAAL Count per Genome
    order = top_taxonomic_groups.sort_values('Mean FAAL Count per Genome', ascending=False)['Taxonomic_Group']
    barplot = sns.barplot(x='Taxonomic_Group', y='Mean FAAL Count per Genome', data=top_taxonomic_groups, palette='viridis', order=order)
    barplot.set_xlabel(f'{taxonomic_level} Level', fontsize=14, fontweight='bold')
    barplot.set_ylabel('Mean FAAL Count per Genome', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add genome counts and total FAAL counts inside the bars and above the bars
    for i in range(len(order)):
        taxonomic_group = order.iloc[i]
        mean_faal_count = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == taxonomic_group]['Mean FAAL Count per Genome'].values[0]
        genome_count_with_faal = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == taxonomic_group]['Genome Count'].values[0]
        total_faal_count = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == taxonomic_group]['Total FAAL Count'].values[0]
        
        barplot.text(i, mean_faal_count / 2, f'{genome_count_with_faal}', color='white', ha="center", va="center", fontsize=10, fontweight='bold')
        barplot.text(i, mean_faal_count, f'{total_faal_count}', color='black', ha="center", va="bottom", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('mean_faal_per_genome.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('mean_faal_per_genome.svg', dpi=dpi, bbox_inches='tight')
    plt.savefig('mean_faal_per_genome.jpeg', dpi=dpi, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Usage: python3 script_name.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    taxonomic_level = sys.argv[3]
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    
    generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi)
