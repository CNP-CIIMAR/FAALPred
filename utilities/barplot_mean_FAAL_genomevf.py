import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
from ete3 import NCBITaxa
import numpy as np

ncbi_taxa = NCBITaxa()

def standardize_lineage_format(lineage_string):
    standardized_lineage = re.sub(r'\s*;\s*', '; ', lineage_string)
    standardized_lineage = standardized_lineage.strip()
    if not standardized_lineage.endswith(';'):
        standardized_lineage += ';'
    return standardized_lineage

def extract_taxonomic_group(lineage_string, desired_level):
    tokens = [token.strip() for token in lineage_string.split(';') if token.strip()]
    if desired_level == 'Order':
        for token in tokens:
            if token.lower().endswith('ales'):
                return token
    elif desired_level == 'Family':
        for token in tokens:
            if token.lower().endswith('eae'):
                return token
    elif desired_level == 'Genus':
        if len(tokens) >= 6:
            candidate = tokens[5]
            if not (candidate.lower().endswith('ales')
                    or candidate.lower().endswith('eae')):
                return candidate
        if len(tokens) >= 2:
            candidate = tokens[-2]
            if not (candidate.lower().endswith('ales')
                    or candidate.lower().endswith('eae')):
                return candidate
        return None
    elif desired_level == 'Phylum':
        if len(tokens) > 1:
            return tokens[1]
    else:
        levels_order = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        try:
            index = levels_order.index(desired_level)
            return tokens[index]
        except (ValueError, IndexError):
            return None

def extract_taxonomic_group_by_ete3(species_name, target_rank, ete3_cache=None):
    if ete3_cache is not None and (species_name, target_rank) in ete3_cache:
        return ete3_cache[(species_name, target_rank)]
    try:
        tx = ncbi_taxa.get_name_translator([species_name])
        if not tx:
            genus = species_name.split()[0]
            tx = ncbi_taxa.get_name_translator([genus])
        if not tx:
            result = None
        else:
            taxid = list(tx.values())[0][0]
            lineage = ncbi_taxa.get_lineage(taxid)
            ranks = ncbi_taxa.get_rank(lineage)
            names = ncbi_taxa.get_taxid_translator(lineage)
            result = None
            for tid in lineage:
                if ranks.get(tid, '').lower() == target_rank.lower():
                    nome = names[tid]
                    if target_rank.lower() == "genus":
                        if nome.lower().endswith('ales') or nome.lower().endswith('eae'):
                            continue
                    result = nome
                    break
        if ete3_cache is not None:
            ete3_cache[(species_name, target_rank)] = result
        return result
    except Exception as e:
        print(f"Erro ao extrair {target_rank} de '{species_name}': {e}")
        if ete3_cache is not None:
            ete3_cache[(species_name, target_rank)] = None
        return None

def get_corrected_lineage_from_species(species_name):
    try:
        name_to_taxid = ncbi_taxa.get_name_translator([species_name])
        if species_name not in name_to_taxid:
            return None
        taxid = name_to_taxid[species_name][0]
        lineage_ids = ncbi_taxa.get_lineage(taxid)
        taxid_to_name = ncbi_taxa.get_taxid_translator(lineage_ids)
        desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        taxonomic_ranks = ncbi_taxa.get_rank(lineage_ids)
        lineage_names = [
            taxid_to_name[t]
            for t in lineage_ids
            if taxonomic_ranks[t] in desired_ranks
        ]
        raw_lineage = '; '.join([name.strip() for name in lineage_names]) + ';'
        return standardize_lineage_format(raw_lineage)
    except Exception as error_message:
        print(f"Error obtaining the lineage for species '{species_name}': {error_message}")
        return None

def update_lineage_for_eukaryotes(data_frame):
    if 'Species' not in data_frame.columns:
        raise KeyError("Column 'Species' not found in the DataFrame.")
    mask_eukaryotes = data_frame['Lineage'].astype(str).str.startswith("Eukaryota")
    data_frame.loc[mask_eukaryotes, 'Lineage'] = data_frame.loc[mask_eukaryotes, 'Species'].apply(get_corrected_lineage_from_species)
    return data_frame

def get_group_ancestry(df, taxonomic_level):
    mapping = {}
    for idx, row in df.iterrows():
        lineage = row['Lineage']
        tax_group = row['Taxonomic_Group']
        if not pd.isnull(lineage) and not pd.isnull(tax_group):
            phylum = extract_taxonomic_group(lineage, 'Phylum')
            order = extract_taxonomic_group(lineage, 'Order')
            if tax_group not in mapping:
                mapping[tax_group] = (str(phylum) if phylum is not None else '', str(order) if order is not None else '')
    return mapping

def assign_custom_colors(groups, group_ancestry, taxonomic_level, darker_black=False):
    """
    - Phylum Actinomycetota, suas Orders e seus Genus: Marrom #b5865a
    - Phylum Cyanobacteriota, suas Orders e seus Genus: Verde #49a87b
    - Phylum Myxococcota, Order Myxococcales, Genus de Myxococcales: Roxo #a34aad
    - Outros: Preto #222222 ou #111111 se darker_black
    """
    colors = []
    normal_black = '#222222'
    strong_black = '#111111'
    for group in groups:
        group_lower = str(group).lower()
        phylum, order = group_ancestry.get(group, ('',''))
        phylum = str(phylum).lower()
        order = str(order).lower()
        # Myxococcota (phylum, order Myxococcales, genus de Myxococcales)
        if (phylum == 'myxococcota' or order == 'myxococcales' or group_lower == 'myxococcota' or group_lower == 'myxococcales'):
            colors.append('#a34aad')
        # Actinomycetota (phylum, orders, genus)
        elif (phylum == 'actinomycetota' or group_lower == 'actinomycetota'):
            colors.append('#b5865a')
        # Cyanobacteriota (phylum, orders, genus)
        elif (phylum == 'cyanobacteriota' or group_lower == 'cyanobacteriota'):
            colors.append('#49a87b')
        else:
            colors.append(strong_black if darker_black else normal_black)
    return colors

def generate_barplot(table1_file_path, domain_argument, taxonomic_level, top_n_groups, plot_dpi, show_inside_values=True):
    data_frame = pd.read_csv(table1_file_path, sep='\t', low_memory=False)
    print("Table 1 loaded:", len(data_frame))
    print(data_frame.head())

    taxonomic_id_column = 'Organism Taxonomic ID' if 'Organism Taxonomic ID' in data_frame.columns else 'Organism Tax ID'
    print("Table 1 with original 'Lineage':")
    print(data_frame[[taxonomic_id_column, 'Lineage']].head())

    # Remove environmental samples if present
    if "Sample" in data_frame.columns:
        data_frame = data_frame[~data_frame["Sample"].str.contains("environmental", case=False, na=False)]

    # Clean up Assembly column and filter valid IDs
    data_frame["Assembly"] = data_frame["Assembly"].astype(str).str.strip()
    assembly_valid_mask = data_frame["Assembly"].str.lower().apply(
        lambda x: x not in ["", "none", "na", "null", "not available"]
    )
    data_frame = data_frame[assembly_valid_mask]
    data_frame = data_frame[data_frame["Assembly"].str.match(r"^(GCA_|GCF_)")]
    print("Rows after filtering invalid Assembly:", len(data_frame))

    # Correct eukaryotic lineages
    data_frame = update_lineage_for_eukaryotes(data_frame)

    # Determine which domains to include
    domain_list = []
    if "Bacteria" in domain_argument:
        domain_list.append("Bacteria")
    if "Eukaryota" in domain_argument:
        domain_list.append("Eukaryota")
    if not domain_list:
        domain_list = [d.strip() for d in domain_argument.split(",")]

    pattern = f"^({'|'.join(domain_list)});\\s*"
    data_frame_filtered = data_frame[
        data_frame['Lineage'].notnull() &
        data_frame['Lineage'].str.match(pattern, case=False)
    ].copy()
    print("Rows after filtering Lineage by domain:", len(data_frame_filtered))

    # Extra cache para acelerar as consultas
    ete3_cache = {}

    # Extract the desired taxonomic group
    if taxonomic_level == 'Genus':
        data_frame_filtered['Taxonomic_Group'] = data_frame_filtered['Species'].apply(
            lambda s: extract_taxonomic_group_by_ete3(s, taxonomic_level, ete3_cache=ete3_cache)
        )
    else:
        data_frame_filtered['Taxonomic_Group'] = data_frame_filtered['Lineage'].apply(
            lambda lineage: extract_taxonomic_group(lineage, taxonomic_level)
        )

    # Further phylum-specific adjustments
    if taxonomic_level == "Phylum":
        data_frame_filtered = data_frame_filtered[~data_frame_filtered['Taxonomic_Group'].str.lower().eq('proteobacteria')]
        data_frame_filtered.loc[
            data_frame_filtered['Taxonomic_Group'].str.lower() == 'deltaproteobacteria',
            'Taxonomic_Group'
        ] = 'Pseudomonadota'
        data_frame_filtered = data_frame_filtered[
            ~data_frame_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)
        ]

    print("After extraction and adjustments, rows with Taxonomic_Group:", len(data_frame_filtered))
    print(data_frame_filtered[['Taxonomic_Group', 'Lineage']].head())

    if data_frame_filtered.empty:
        print("No taxonomic group found after adjustments.")
        return

    # Aggregate and compute metrics
    grouped_data = data_frame_filtered.groupby('Taxonomic_Group').agg(
        Total_FAAL_Count=('Taxonomic_Group', 'size'),
        Genome_Count=('Assembly', 'nunique'),
        Assembly_List=('Assembly', lambda assemblies: ';'.join(sorted(assemblies.unique()))),
        Protein_IDs=('Protein Accession', lambda pids: ';'.join(sorted(pd.Series(pids).dropna().unique())))
    ).reset_index()

    # Keep only groups with at least 5 genomes
    grouped_data = grouped_data[grouped_data['Genome_Count'] >= 5]
    grouped_data['Mean_FAALs_per_Genome'] = (
        grouped_data['Total_FAAL_Count'] / grouped_data['Genome_Count']
    )

    # Select top N groups
    top_groups = grouped_data.sort_values(
        by='Mean_FAALs_per_Genome', ascending=False
    ).head(top_n_groups)

    # Salva a tabela completa ordenada por total de FAAL
    output_table_path = 'top_taxonomic_groups_FAAL.tsv'
    top_sorted = grouped_data.sort_values(by='Total_FAAL_Count', ascending=False)
    with open(output_table_path, 'w') as f:
        f.write("# Top Taxonomic Groups (FAAL)\n")
        top_sorted.to_csv(f, sep='\t', index=False)

    # Para cada grupo, encontra seu phylum e ordem
    group_ancestry = get_group_ancestry(
        data_frame_filtered[data_frame_filtered['Taxonomic_Group'].isin(top_groups['Taxonomic_Group'])],
        taxonomic_level
    )

    # Custom color palette (preto mais forte se não mostrar valores dentro)
    bar_colors = assign_custom_colors(
        top_groups['Taxonomic_Group'], group_ancestry, taxonomic_level,
        darker_black=not show_inside_values
    )

    # Abreviação "Candidatus"
    def abbreviate_candidatus(name):
        if isinstance(name, str) and name.strip().lower().startswith('candidatus '):
            parts = name.strip().split(' ', 1)
            if len(parts) == 2:
                return f'C. {parts[1]}'
        return name

    if taxonomic_level in ['Phylum', 'Order', 'Genus']:
        xtick_labels = [abbreviate_candidatus(x) for x in top_groups['Taxonomic_Group']]
    else:
        xtick_labels = list(top_groups['Taxonomic_Group'])

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.bar(
        xtick_labels,
        top_groups['Mean_FAALs_per_Genome'],
        color=bar_colors,
        edgecolor='black',
        alpha=0.85
    )

    # Eixo Y: fonte maior e negrito
    ax.set_ylabel('Mean FAAL Count / genome', fontsize=22, fontweight='bold')
    ax.set_yticks(ax.get_yticks())  # Garante ticks atualizados
    ax.set_yticklabels([str(int(tick)) if tick == int(tick) else str(tick) for tick in ax.get_yticks()],
                       fontsize=18, fontweight='bold')

    # Eixo X: negrito só para Phylum
    if taxonomic_level == "Phylum":
        plt.xticks(rotation=45, ha='right', fontsize=18, fontweight='bold')
    else:
        plt.xticks(rotation=45, ha='right', fontsize=18, fontweight='normal')

    # Nome do taxonomic_level no canto superior direito (negrito)
    ax.text(
        0.99, 0.93, taxonomic_level.lower(),
        ha='right', va='top',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold'
    )

    # Adiciona os números acima das barras, na vertical, em negrito
    for i, bar in enumerate(bars):
        height = bar.get_height()
        total = top_groups.iloc[i]['Total_FAAL_Count']
        genomes = top_groups.iloc[i]['Genome_Count']
        # Número acima da barra (Total_FAAL_Count) -- fonte menor que o eixo Y
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.03 * height,
            f'{total}',
            ha='center', va='bottom',
            fontsize=16, fontweight='bold',
            rotation=90
        )
        # Número dentro da barra (Genome_Count), só se permitido
        if show_inside_values:
            label_color = 'black' if genomes > 1000 else 'white'
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height/2,
                f'{genomes}',
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color=label_color
            )

    # Ajuste dos eixos y e ticks
    max_mean = top_groups['Mean_FAALs_per_Genome'].max()
    upper = max_mean * 1.1
    ax.set_ylim(0, upper)
    top_tick = int(np.ceil(upper))
    ticks = np.arange(0, top_tick + 1, 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(int(tick)) for tick in ticks], fontsize=18, fontweight='bold')
    ax.margins(x=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)

    # Deixa os spines em negrito, preto escuro
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('#111111')

    # Salva as figuras
    plt.savefig('barplot_mean_faal_per_genome.png', dpi=plot_dpi)
    plt.savefig('barplot_mean_faal_per_genome.svg', dpi=plot_dpi)
    plt.show()

    print("Results table saved to:", output_table_path)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 bar_faal_all_countsv2.py <table1.tsv> <Domain(s)> <Taxonomic Level> <Top N> <DPI> <ShowInsideValues>")
        print("Example: python3 bar_faal_all_countsv2.py table1.tsv 'Bacteria' Phylum 10 300 True")
        sys.exit(1)
    table1_file_path = sys.argv[1]
    domain_argument = sys.argv[2]      # e.g., "Bacteria", "Eukaryota" or "Bacteria,Eukaryota"
    taxonomic_level = sys.argv[3]      # e.g., "Order", "Phylum", "Genus", etc.
    top_n_groups = int(sys.argv[4])
    plot_dpi = int(sys.argv[5])
    show_inside_values = sys.argv[6].lower() in ["true", "1", "yes", "sim"]
    generate_barplot(table1_file_path, domain_argument, taxonomic_level, top_n_groups, plot_dpi, show_inside_values)















