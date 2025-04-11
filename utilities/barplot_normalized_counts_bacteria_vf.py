import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

def fix_acetobacteraceae_lineage(lineage):
    """
    Corrects the lineage when the family is Acetobacteraceae and
    the order is incorrectly listed as Rhodospirillales.
    In such cases, the order is replaced with Acetobacterales.
    """
    tokens = [token.strip() for token in lineage.split(';') if token.strip()]
    if len(tokens) > 4:
        if tokens[4].lower() == 'acetobacteraceae' and tokens[3].lower() == 'rhodospirillales':
            tokens[3] = 'Acetobacterales'
        return '; '.join(tokens)
    return lineage

def extract_taxonomic_group(lineage, level):
    """
    Extracts the taxonomic group from the lineage string based on the desired level.
    
    For Genus:
      - If the classification has 6 or more tokens, it uses the token at index 5,
        provided it does not end with "ales" or "eae" and does not start with "Candidatus".
      - Otherwise, it attempts to use the second-to-last token.
    
    For Family:
      - Iterates over tokens and returns the first token that ends with "eae" (case insensitive).
    
    For Order:
      - Iterates over tokens and returns the first token that ends with "ales" (case insensitive).
    
    For other levels:
      - Uses a fixed list of levels: ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        and returns the token corresponding to that level's index if available.
    """
    levels_order = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
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
        if level in levels_order:
            try:
                index = levels_order.index(level)
                return tokens[index] if index < len(tokens) else None
            except IndexError:
                return None
        return None

def generate_filtered_table_and_graphs(table1_path, table2_path, domain_name, taxonomic_level, top_n, dpi, sub_taxonomic_level=None):
    # Load Table 1 (full dataset)
    df1_all = pd.read_csv(table1_path, sep='\t', low_memory=False)
    
    # Filter Table 1 by domain and optionally by sub-taxonomic level
    df1_all = df1_all[df1_all['Lineage'].str.contains(domain_name, na=False)].copy()
    if sub_taxonomic_level:
        df1_all = df1_all[df1_all['Lineage'].str.contains(sub_taxonomic_level, na=False)].copy()
    
    # For Genus level, keep only records with at least 6 tokens in the lineage
    if taxonomic_level.lower() == "genus":
        df1_all = df1_all[df1_all['Lineage'].apply(lambda x: len([t.strip() for t in x.split(';') if t.strip()]) >= 6)]
    
    # Extract the taxonomic group for the specified level
    df1_all['Taxonomic_Group'] = df1_all['Lineage'].apply(
        lambda x: extract_taxonomic_group(x, sub_taxonomic_level or taxonomic_level)
    )
    df1_all = df1_all[df1_all['Taxonomic_Group'].notna()]
    df1_all = df1_all[~df1_all['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    # Specific adjustments for Family and Order levels (data normalization)
    if taxonomic_level.lower() == "family":
        df1_all = df1_all[~df1_all['Taxonomic_Group'].str.lower().eq('cystobacterineae')]
        df1_all = df1_all[~df1_all['Taxonomic_Group'].str.lower().str.endswith('ales')]
    if taxonomic_level.lower() == "order":
        df1_all = df1_all[df1_all['Taxonomic_Group'].str.lower().str.endswith('ales')]
    
    # Correction for Phylum: standardizing some names
    if taxonomic_level.lower() == "phylum":
        df1_all.loc[df1_all['Taxonomic_Group'].str.lower().isin(['proteobacteria', 'deltaproteobacteria']), 'Taxonomic_Group'] = 'Pseudomonadota'
    
    if df1_all.empty:
        print("No data found for the provided domain and taxonomic level in Table 1.")
        return
    
    # Aggregate data: count total FAALs from Table 1
    total_faal_counts_all = df1_all.groupby('Taxonomic_Group')['Protein Accession'].count().reset_index(name='Total FAAL Count')
    
    # Filter records with valid genome assemblies (Assembly starts with "GCF_" or "GCA_")
    df1_filtered = df1_all[df1_all['Assembly'].str.startswith(('GCF_', 'GCA'), na=False)].copy()
    faal_count_series = df1_filtered.groupby('Taxonomic_Group')['Protein Accession'].size().reset_index(name='FAAL_Count')
    unique_genomes = df1_filtered.drop_duplicates(subset=['Assembly', 'Taxonomic_Group'])
    genome_count_series = unique_genomes.groupby('Taxonomic_Group')['Assembly'].count().reset_index(name='Genome_Count')
    
    # Calculate mean FAAL count per genome
    faal_stats = pd.merge(faal_count_series, genome_count_series, on='Taxonomic_Group', how='left')
    faal_stats['Mean FAAL Count per Genome'] = faal_stats['FAAL_Count'] / faal_stats['Genome_Count']
    
    merged_data = pd.merge(total_faal_counts_all, 
                           faal_stats[['Taxonomic_Group', 'Mean FAAL Count per Genome', 'Genome_Count']],
                           on='Taxonomic_Group')
    
    # Select the Top N taxonomic groups based on total FAAL count
    top_taxonomic_groups = merged_data.nlargest(top_n, 'Total FAAL Count')
    
    # --- Plot A: Total FAAL Counts ---
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    order_axis = top_taxonomic_groups.sort_values('Total FAAL Count', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Total FAAL Count', y='Taxonomic_Group', data=top_taxonomic_groups, 
                ax=ax[0], palette='viridis', order=order_axis)
    ax[0].set_xlabel('FAAL Total Counts', fontsize=14)
    ax[0].set_ylabel(f'{taxonomic_level} Level', fontsize=14)
    ax[0].text(-0.1, 1.15, "A", transform=ax[0].transAxes, fontsize=16, fontweight='bold',
               va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    for patch, group in zip(ax[0].patches, order_axis):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        mean_val = top_taxonomic_groups[top_taxonomic_groups['Taxonomic_Group'] == group]['Mean FAAL Count per Genome'].values[0]
        ax[0].text(x, y, f'{mean_val:.2f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    ax[0].margins(x=0)
    
    # --- Process Table 2 ---
    df2 = pd.read_csv(table2_path, sep='\t', low_memory=False)
    df2['Lineage'] = df2['Lineage'].apply(fix_acetobacteraceae_lineage)
    df2 = df2[df2['Assembly Accession'].str.startswith(('GCF_', 'GCA'), na=False)].copy()
    
    # Remove duplicate records based on the accession suffix
    df2['accession_suffix'] = df2['Assembly Accession'].str.replace(r'^(GCF_|GCA_)', '', regex=True)
    df2 = df2.drop_duplicates(subset=['accession_suffix'])
    
    if taxonomic_level.lower() == "genus":
        df2 = df2[df2['Lineage'].apply(lambda x: len([t.strip() for t in x.split(';') if t.strip()]) >= 6)]
    
    df2['Taxonomic_Group'] = df2['Lineage'].apply(
        lambda x: extract_taxonomic_group(x, sub_taxonomic_level or taxonomic_level)
    )
    df2 = df2[df2['Taxonomic_Group'].notna()]
    
    # Standardize taxonomic group names based on Table 1 data
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
    if taxonomic_level.lower() == "order":
        df2_filtered = df2_filtered[df2_filtered['Taxonomic_Group'].str.lower().str.endswith('ales')]
    
    df2_filtered = df2_filtered[~df2_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    if df2_filtered.empty:
        print("No data found for the provided taxonomic groups in Table 2.")
        return
    
    output_filtered_table = 'Taxonomic_groups_with_FAAL.tsv'
    if not os.path.exists(output_filtered_table):
        df2_filtered.to_csv(output_filtered_table, sep='\t', index=False)
    
    # Get total genome counts from Table 2 (using unique Assembly Accession)
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
    
    # --- Generic Correction for Non-Redundant Genome Count ---
    # For each taxonomic group, if the genome count from Table 1 (Genome_Count)
    # is less than the total genome count from Table 2,
    # recalc the union of genome IDs from both tables.
    updated_total_ids = {}
    for taxon in normalized_data['Taxonomic_Group']:
        # IDs from Table 1 (FAAL genomes; column 'Assembly')
        ids_table1 = set(df1_filtered[df1_filtered['Taxonomic_Group'] == taxon]['Assembly'].unique())
        # IDs from Table 2 (total genomes; column 'Assembly Accession')
        ids_table2 = set(df2_filtered[df2_filtered['Taxonomic_Group'] == taxon]['Assembly Accession'].unique())
        union_ids = ids_table1.union(ids_table2)
        union_count = len(union_ids)
        # Se a união (não redundante) for maior que o valor obtido individualmente,
        # atualize a contagem para esse taxon.
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
    
    # --- Generate Aggregated Lists of IDs for Verification Table ---
    protein_ids_by_group = df1_filtered.groupby('Taxonomic_Group')['Protein Accession']\
                                       .apply(lambda x: ', '.join(x.dropna().unique())).reset_index(name='Protein IDs')
    genomes_with_faal_ids = df1_filtered.groupby('Taxonomic_Group')['Assembly']\
                                        .apply(lambda x: ', '.join(x.dropna().unique())).reset_index(name='Genomes with FAAL IDs')
    total_genome_ids = df2_filtered.groupby('Taxonomic_Group')['Assembly Accession']\
                                  .apply(lambda x: ', '.join(x.dropna().unique())).reset_index(name='Total Genome IDs')
    
    # For each taxon, update the Total Genome IDs using the union (non-redundant) generated above
    for taxon, union_ids_str in updated_total_ids.items():
        condition = total_genome_ids['Taxonomic_Group'].str.lower() == taxon.lower()
        total_genome_ids.loc[condition, 'Total Genome IDs'] = union_ids_str
    
    # Merge the aggregated ID lists with the normalized data
    verification_table = normalized_data.merge(protein_ids_by_group, on='Taxonomic_Group', how='left')\
                                          .merge(genomes_with_faal_ids, on='Taxonomic_Group', how='left')\
                                          .merge(total_genome_ids, on='Taxonomic_Group', how='left')
    verification_table = verification_table.rename(columns={
        'Taxonomic_Group': 'Taxonomy',
        'Normalized': 'Percentage of Genomes with FAAL'
    })
    
    # Rearrange the columns in the final verification table
    verification_table = verification_table[['Taxonomy', 'Protein IDs', 'Genomes with FAAL IDs', 'Total Genome IDs', 'Percentage of Genomes with FAAL']]
    
    verification_table_file = 'verification_table.tsv'
    verification_table.to_csv(verification_table_file, sep='\t', index=False)
    print(f"Verification table successfully generated: {verification_table_file}")
    
    # --- Plot B: Normalization ---
    order_normalized = normalized_data.sort_values('Normalized', ascending=False)['Taxonomic_Group']
    sns.barplot(x='Normalized', y='Taxonomic_Group', data=normalized_data,
                ax=ax[1], palette='viridis', order=order_normalized)
    ax[1].set_xlabel('Percentage (%) of Deposited Genomes Containing FAAL', fontsize=14)
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


