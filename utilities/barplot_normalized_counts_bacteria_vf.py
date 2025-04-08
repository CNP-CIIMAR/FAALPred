import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from ete3 import NCBITaxa

# Initialize the NCBITaxa object (local cache may be downloaded or updated on the first run)
ncbi = NCBITaxa()

def correct_lineage_all(lineage_str):
    """
    Receives a taxonomic lineage string and attempts to correct all levels
    (Domain, Phylum, Class, Order, Family, Genus, Species) using the official NCBI taxonomy database via ete3.
    If correction is possible, returns a new string with the official levels separated by "; ".
    Otherwise, returns the original lineage.
    
    Additionally, applies extra corrections:
      - For Acetobacteraceae: if any token contains "Acetobacteraceae" and there are at least 5 tokens,
        the Order level (index 3) is forced to "Acetobacterales" if necessary.
      - For Polyangiales/Polyangiaceae: if the Order (index 3) is "polyangiales" but the Family (index 4)
        is not "polyangiaceae" (or vice versa), the tokens are adjusted to maintain consistency.
    """
    # Split the string into tokens, removing extra spaces
    tokens = [token.strip() for token in lineage_str.split(";") if token.strip()]
    if not tokens:
        return lineage_str  # If no tokens, return the original lineage
    # Assume the last token is the species; use it to obtain the TaxID
    species_name = tokens[-1]
    try:
        taxid_map = ncbi.get_name_translator([species_name])
        if not taxid_map:
            return lineage_str
        taxid = list(taxid_map.values())[0][0]
        # Get the full lineage based on the TaxID
        lineage_ids = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage_ids)
        ranks = ncbi.get_rank(lineage_ids)
        
        # Create an official dictionary mapping desired levels to names
        # We use "superkingdom" as Domain
        official_lineage = {}
        for tid in lineage_ids:
            rank = ranks.get(tid)
            if rank in ["superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species"]:
                if rank == "superkingdom":
                    key = "Domain"
                else:
                    key = rank.capitalize()
                official_lineage[key] = names.get(tid)
        
        # Define the desired order of levels
        desired_order = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
        corrected_levels = []
        for level in desired_order:
            if level in official_lineage:
                corrected_levels.append(official_lineage[level])
        
        # If corrected levels exist, apply additional corrections
        if corrected_levels:
            # For Acetobacteraceae: if any token contains "acetobacteraceae" and there are at least 5 tokens,
            # force the Order level (index 3) to be "Acetobacterales"
            if any("acetobacteraceae" in token.lower() for token in corrected_levels) and len(corrected_levels) >= 5:
                if corrected_levels[3].lower() != "acetobacterales":
                    corrected_levels[3] = "Acetobacterales"
            
            # For Polyangiales/Polyangiaceae:
            # If Order (index 3) is "polyangiales" but Family (index 4) is not "polyangiaceae", correct to "Polyangiaceae"
            if len(corrected_levels) >= 5:
                if corrected_levels[3].lower() == "polyangiales" and corrected_levels[4].lower() != "polyangiaceae":
                    corrected_levels[4] = "Polyangiaceae"
                # If Family is "polyangiaceae" but Order is not "polyangiales", correct to "Polyangiales"
                if corrected_levels[4].lower() == "polyangiaceae" and corrected_levels[3].lower() != "polyangiales":
                    corrected_levels[3] = "Polyangiales"
            
            return "; ".join(corrected_levels)
        else:
            return lineage_str
    except Exception as error:
        print(f"Error correcting lineage '{lineage_str}': {error}")
        return lineage_str

def extract_taxonomic_group(lineage, level):
    """
    Extracts the taxonomic group from the lineage string based on the desired level.
    For the Genus level, a specific logic is applied:
      - If the classification has 6 or more tokens, uses the 6th token (index 5)
        provided it does not end with "ales" or "eae" and does not start with "Candidatus".
      - Otherwise, it attempts to use the penultimate token if available.
    For the Order and Family levels, this function attempts to identify the correct token
    using suffix rules. For Order, tokens ending with "ales" are used; for Family, tokens
    ending with "eae" or "aceae" are used. If none are found, it falls back to the fixed index.
    For all other levels, a fixed index is used based on the standard order.
    """
    tokens = [token.strip() for token in lineage.split(';') if token.strip()]
    lower_level = level.lower()
    
    if lower_level == 'genus':
        if len(tokens) >= 6:
            candidate = tokens[5]
            if not (candidate.lower().endswith('ales') or candidate.lower().endswith('eae') or candidate.startswith("Candidatus")):
                return candidate
        if len(tokens) >= 2:
            candidate = tokens[-2]
            if not (candidate.lower().endswith('ales') or candidate.lower().endswith('eae') or candidate.startswith("Candidatus")):
                return candidate
        return None
    elif lower_level == 'order':
        # Attempt to identify Order by the "ales" suffix
        for token in tokens:
            if token.lower().endswith("ales"):
                return token
        # Fallback to fixed index: expected Order is the 4th token (index 3)
        return tokens[3] if len(tokens) > 3 else None
    elif lower_level == 'family':
        # Attempt to identify Family by the "eae" or "aceae" suffix
        for token in tokens:
            if token.lower().endswith("eae") or token.lower().endswith("aceae"):
                return token
        # Fallback to fixed index: expected Family is the 5th token (index 4)
        return tokens[4] if len(tokens) > 4 else None
    else:
        # For Domain, Phylum, Class, and Species, use fixed positions based on standard order
        standard_order = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        try:
            index = standard_order.index(level.capitalize())
        except ValueError:
            return None
        return tokens[index] if index < len(tokens) else None

def generate_filtered_table_and_graphs(table1_path, table2_path, domain_name, taxonomic_level, top_n, dpi, sub_taxonomic_level=None):
    # Load the complete Table 1 (all records)
    table1_df = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Filter Table 1 by domain and, if provided, by sub-taxonomic level
    table1_df = table1_df[table1_df['Lineage'].str.contains(domain_name, na=False)].copy()
    if sub_taxonomic_level:
        table1_df = table1_df[table1_df['Lineage'].str.contains(sub_taxonomic_level, na=False)].copy()
    
    # For the Genus level, consider only records with at least 6 tokens in the Lineage
    if taxonomic_level.lower() == "genus":
        table1_df = table1_df[table1_df['Lineage'].apply(lambda x: len([token.strip() for token in x.split(';') if token.strip()]) >= 6)]
    
    # Extract the Taxonomic Group column using the desired level or sub-level
    table1_df['Taxonomic_Group'] = table1_df['Lineage'].apply(
        lambda line: extract_taxonomic_group(line, sub_taxonomic_level or taxonomic_level))
    
    # Remove records with null Taxonomic_Group
    table1_df = table1_df[table1_df['Taxonomic_Group'].notna()]
    
    # Remove records that contain "environmental" to maintain consistency across graphs
    table1_df = table1_df[~table1_df['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    # For Family level: remove groups ending with "ales" and those that equal "cystobacterineae"
    if taxonomic_level.lower() == "family":
        table1_df = table1_df[~table1_df['Taxonomic_Group'].str.lower().str.endswith('ales')]
        table1_df = table1_df[~table1_df['Taxonomic_Group'].str.lower().eq('cystobacterineae')]
    
    # For Order level: keep only groups that end with "ales"
    if taxonomic_level.lower() == "order":
        table1_df = table1_df[table1_df['Taxonomic_Group'].str.lower().str.endswith('ales')]
    
    # For Graph A: if the level is Phylum, correct specific group names
    if taxonomic_level.lower() == "phylum":
        table1_df.loc[table1_df['Taxonomic_Group'].str.lower().isin(['proteobacteria', 'deltaproteobacteria']), 'Taxonomic_Group'] = 'Pseudomonadota'
    
    if table1_df.empty:
        print("No data found for the specified domain and taxonomic level in Table 1.")
        return
    
    # Calculate the total FAAL Count per taxonomic group (using all records from Table 1)
    total_faal_count = table1_df.groupby('Taxonomic_Group')['Protein Accession'].count().reset_index(name='Total FAAL Count')
    
    # Filter records with a valid genome (Assembly starting with "GCF_" or "GCA")
    table1_filtered = table1_df[table1_df['Assembly'].str.startswith(('GCF_', 'GCA'), na=False)].copy()
    
    # Calculate the FAAL count (there may be more than one FAAL protein per genome)
    faal_count = table1_filtered.groupby('Taxonomic_Group')['Protein Accession'].size().reset_index(name='FAAL_Count')
    # For genome count, remove duplicates (the same genome should not be counted more than once)
    unique_genomes = table1_filtered.drop_duplicates(subset=['Assembly', 'Taxonomic_Group'])
    genome_count = unique_genomes.groupby('Taxonomic_Group')['Assembly'].count().reset_index(name='Genome_Count')
    
    # Merge counts and calculate the mean FAAL per genome
    faal_stats = pd.merge(faal_count, genome_count, on='Taxonomic_Group', how='left')
    faal_stats['Mean FAAL Count per Genome'] = faal_stats['FAAL_Count'] / faal_stats['Genome_Count']
    
    # Merge Table 1 data
    merged_data = pd.merge(total_faal_count, 
                           faal_stats[['Taxonomic_Group', 'Mean FAAL Count per Genome', 'Genome_Count']],
                           on='Taxonomic_Group')
    
    # Select the top N groups (based on the total FAAL count from Table 1)
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Total FAAL Count')
    
    # Configure the plot with 2 subplots (Graph A and Graph B)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # --- Graph A ---
    group_order = top_taxonomic_groups.sort_values('Total FAAL Count', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Total FAAL Count', y='Taxonomic_Group', data=top_taxonomic_groups, 
                ax=axes[0], palette='viridis', order=group_order)
    axes[0].set_xlabel('Fatty Acyl AMP Ligase (FAALs) Counts', fontsize=14)
    axes[0].set_ylabel(f'{taxonomic_level.capitalize()} Level', fontsize=14)
    axes[0].text(-0.1, 1.15, "A", transform=axes[0].transAxes, fontsize=16, fontweight='bold',
                 va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # Annotate the "Mean FAAL Count per Genome" values centered vertically on each bar
    for rect, group in zip(axes[0].patches, group_order):
        width = rect.get_width()
        y_position = rect.get_y() + rect.get_height() / 2
        mean_value = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == group]['Mean FAAL Count per Genome'].values[0]
        axes[0].text(width, y_position, f'{mean_value:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    
    axes[0].margins(x=0)
    
    # --- Graph B ---
    table2_df = pd.read_csv(table2_path, sep='\t', low_memory=False)
    table2_df = table2_df[table2_df['Assembly Accession'].str.startswith(('GCF_', 'GCA'), na=False)].copy()
    
    # Correct the full taxonomic lineage (all levels) using ete3, if possible
    table2_df["Lineage"] = table2_df["Lineage"].apply(correct_lineage_all)
    
    # For Genus level, consider only records with at least 6 tokens in the Lineage
    if taxonomic_level.lower() == "genus":
        table2_df = table2_df[table2_df['Lineage'].apply(lambda x: len([token.strip() for token in x.split(';') if token.strip()]) >= 6)]
    
    table2_df['Taxonomic_Group'] = table2_df['Lineage'].apply(
        lambda line: extract_taxonomic_group(line, sub_taxonomic_level or taxonomic_level))
    
    # Remove null values in Taxonomic_Group from Table 2
    table2_df = table2_df[table2_df['Taxonomic_Group'].notna()]
    
    # Standardize group names using the filtered Table 1 data
    unique_taxonomic_groups = table1_df['Taxonomic_Group'].dropna().unique()
    mapping = {group.lower(): group for group in unique_taxonomic_groups}
    table2_df['Taxonomic_Group'] = table2_df['Taxonomic_Group'].str.lower().map(mapping)
    
    # Filter records with a valid Taxonomic_Group that are present in the top N from Table 1
    table2_filtered = table2_df[table2_df['Taxonomic_Group'].notna()].copy()
    valid_groups = top_taxonomic_groups['Taxonomic_Group'].unique()
    table2_filtered = table2_filtered[table2_filtered['Taxonomic_Group'].isin(valid_groups)].copy()
    
    # Apply additional corrections if applicable (for Phylum, Family, Order, and Genus)
    if taxonomic_level.lower() in ["phylum", "family", "order", "genus"]:
        table2_filtered = table2_filtered[~table2_filtered['Taxonomic_Group'].str.lower().eq('proteobacteria')]
        table2_filtered.loc[table2_filtered['Taxonomic_Group'].str.lower() == 'deltaproteobacteria', 'Taxonomic_Group'] = 'Pseudomonadota'
    
    # For Family level, remove records with groups ending with "ales" and those equal to "cystobacterineae"
    if taxonomic_level.lower() == "family":
        table2_filtered = table2_filtered[~table2_filtered['Taxonomic_Group'].str.lower().str.endswith('ales')]
        table2_filtered = table2_filtered[~table2_filtered['Taxonomic_Group'].str.lower().eq('cystobacterineae')]
    
    # For Order level, keep only records with groups ending with "ales"
    if taxonomic_level.lower() == "order":
        table2_filtered = table2_filtered[table2_filtered['Taxonomic_Group'].str.lower().str.endswith('ales')]
    
    # Remove records that contain "environmental" to maintain consistency
    table2_filtered = table2_filtered[~table2_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    if table2_filtered.empty:
        print("No data found for the specified taxonomic groups in Table 2.")
        return
    
    output_filtered_file = 'Taxonomic_groups_with_FAAL.tsv'
    if not os.path.exists(output_filtered_file):
        table2_filtered.to_csv(output_filtered_file, sep='\t', index=False)
    
    # Add a 'Common Genome ID' column to remove duplicates between GCF_ and GCA_
    table2_filtered['Common Genome ID'] = table2_filtered['Assembly Accession'].str.replace(r'^(GCF_|GCA_)', '', regex=True)
    
    # Calculate the total number of unique genomes per taxonomic group using the 'Common Genome ID'
    total_genome_count = table2_filtered.groupby('Taxonomic_Group')['Common Genome ID'].nunique().reset_index(name='Total Genome Count')
    
    # Merge to keep all groups from the top N of Table 1
    normalized_data = pd.merge(top_taxonomic_groups, total_genome_count, on='Taxonomic_Group', how='left')
    normalized_data['Total Genome Count'] = normalized_data['Total Genome Count'].fillna(0)
    
    # Calculate the appropriate normalization or percentage
    if taxonomic_level.lower() in ['phylum', 'family', 'order', 'genus']:
        normalized_data['Normalized'] = np.where(
            normalized_data['Total Genome Count'] == 0,
            0,
            (normalized_data['Genome_Count'] / normalized_data['Total Genome Count']) * 100)
        normalized_data['Normalized'] = normalized_data['Normalized'].clip(upper=100)
        internal_annotation = 'Genome_Count'
    else:
        normalized_data['Normalized'] = np.where(
            normalized_data['Total Genome Count'] == 0,
            0,
            (normalized_data['Total FAAL Count'] / normalized_data['Total Genome Count']) * 100)
        internal_annotation = 'Total FAAL Count'
    
    norm_order = normalized_data.sort_values('Normalized', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Normalized', y='Taxonomic_Group', data=normalized_data,
                ax=axes[1], palette='viridis', order=norm_order)
    axes[1].set_xlabel('Percentage (%) of Deposited Genomes Containing FAAL', fontsize=14)
    axes[1].set_ylabel(f'{taxonomic_level.capitalize()} Level', fontsize=14)
    axes[1].text(-0.1, 1.15, "B", transform=axes[1].transAxes, fontsize=16, fontweight='bold',
                 va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    normalized_ordered = normalized_data.set_index('Taxonomic_Group').loc[norm_order].reset_index()
    for rect, (_, row) in zip(axes[1].patches, normalized_ordered.iterrows()):
        bar_width = rect.get_width()
        bar_y = rect.get_y() + rect.get_height() / 2
        axes[1].text(bar_width + 0.5, bar_y, f'{int(row["Total Genome Count"])}', 
                     color='black', ha='left', va='center', fontsize=10, fontweight='bold')
        axes[1].text(bar_width / 2, bar_y, f'{int(row[internal_annotation])}', 
                     color='white', ha='center', va='center', fontsize=10, fontweight='bold')
    
    axes[1].margins(x=0)
    
    # For Genus or Species, set the y-axis tick labels in italic
    if taxonomic_level.lower() in ['genus', 'species']:
        axes[0].set_yticklabels(axes[0].get_yticklabels(), style='italic')
        axes[1].set_yticklabels(axes[1].get_yticklabels(), style='italic')
    
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

