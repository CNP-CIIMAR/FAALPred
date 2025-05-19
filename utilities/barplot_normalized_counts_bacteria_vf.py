import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
from ete3 import NCBITaxa
import numpy as np  # For tick adjustment

# Initialize the NCBITaxa object for correcting taxonomic lineage
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
                    or candidate.lower().endswith('eae')
                    or candidate.startswith("Candidatus")):
                return candidate
        if len(tokens) >= 2:
            candidate = tokens[-2]
            if not (candidate.lower().endswith('ales')
                    or candidate.lower().endswith('eae')
                    or candidate.startswith("Candidatus")):
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

def extract_taxonomic_group_by_ete3(species_name, target_rank):
    """
    Usa ETE3 para buscar o taxid do nome da espécie (ou gênero, se não achar a espécie)
    e retorna o nome associado ao rank taxonômico desejado (aplicado apenas para 'Genus').
    Adiciona os filtros para genus: não termina com "ales" nem "eae", e não começa com "Candidatus".
    """
    try:
        tx = ncbi_taxa.get_name_translator([species_name])
        if not tx:
            # Se não encontrou taxid da espécie, tenta o gênero (primeira palavra)
            genus = species_name.split()[0]
            tx = ncbi_taxa.get_name_translator([genus])
        if not tx:
            return None
        taxid = list(tx.values())[0][0]
        lineage = ncbi_taxa.get_lineage(taxid)
        ranks   = ncbi_taxa.get_rank(lineage)
        names   = ncbi_taxa.get_taxid_translator(lineage)
        for tid in lineage:
            if ranks.get(tid, '').lower() == target_rank.lower():
                nome = names[tid]
                # Para genus, aplica os filtros pedidos
                if target_rank.lower() == "genus":
                    if (nome.lower().endswith('ales') or 
                        nome.lower().endswith('eae') or 
                        nome.startswith("Candidatus")):
                        continue  # ignora e segue procurando
                return nome
        return None
    except Exception as e:
        print(f"Erro ao extrair {target_rank} de '{species_name}': {e}")
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
        lineage_names = [taxid_to_name[taxid_item].strip() for taxid_item in lineage_ids if taxonomic_ranks[taxid_item] in desired_ranks]
        raw_lineage = '; '.join(lineage_names) + ';'
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

def generate_barplot(table1_file_path, domain_argument, taxonomic_level, top_n_groups, plot_dpi):
    data_frame = pd.read_csv(table1_file_path, sep='\t', low_memory=False)
    print("Table 1 loaded:", len(data_frame))
    print(data_frame.head())
    taxonomic_id_column = 'Organism Taxonomic ID' if 'Organism Taxonomic ID' in data_frame.columns else 'Organism Tax ID'
    print("Table 1 with original 'Lineage':")
    print(data_frame[[taxonomic_id_column, 'Lineage']].head())
    if "Sample" in data_frame.columns:
        data_frame = data_frame[~data_frame["Sample"].str.contains("environmental", case=False, na=False)]
    data_frame["Assembly"] = data_frame["Assembly"].astype(str).str.strip()
    assembly_valid_mask = data_frame["Assembly"].str.lower().apply(
        lambda valor: valor not in ["", "none", "na", "null", "not available"]
    )
    data_frame = data_frame[assembly_valid_mask]
    print("Rows after filtering invalid Assembly:", len(data_frame))
    data_frame = data_frame[data_frame["Assembly"].str.match(r"^(GCA_|GCF_)")]
    print("Rows after keeping only IDs starting with GCA_/GCF_:", len(data_frame))
    data_frame = update_lineage_for_eukaryotes(data_frame)
    domain_list = []
    if "Bacteria" in domain_argument:
        domain_list.append("Bacteria")
    if "Eukaryota" in domain_argument:
        domain_list.append("Eukaryota")
    if not domain_list:
        domain_list = [dominio.strip() for dominio in domain_argument.split(",")]
    pattern = f"^({'|'.join(domain_list)});\\s*"
    data_frame_filtered = data_frame[data_frame['Lineage'].notnull() & data_frame['Lineage'].str.match(pattern, case=False)].copy()
    print("Rows after filtering Lineage by domain:", len(data_frame_filtered))

    # >>>>>>>>>>>> NOVO TRECHO SÓ PARA GENUS <<<<<<<<<<<<
    if taxonomic_level == 'Genus':
        data_frame_filtered['Taxonomic_Group'] = data_frame_filtered['Species'].apply(
            lambda s: extract_taxonomic_group_by_ete3(s, taxonomic_level)
        )
    else:
        data_frame_filtered['Taxonomic_Group'] = data_frame_filtered['Lineage'].apply(
            lambda lineage: extract_taxonomic_group(lineage, taxonomic_level)
        )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    data_frame_filtered = data_frame_filtered[data_frame_filtered['Taxonomic_Group'] != 'Candidatus Entotheonellales']
    if taxonomic_level == "Phylum":
        data_frame_filtered = data_frame_filtered[~data_frame_filtered['Taxonomic_Group'].str.lower().eq('proteobacteria')]
        data_frame_filtered.loc[data_frame_filtered['Taxonomic_Group'].str.lower() == 'deltaproteobacteria', 'Taxonomic_Group'] = 'Pseudomonadota'
        data_frame_filtered = data_frame_filtered[~data_frame_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    print("After extraction and adjustments, rows with Taxonomic_Group:", len(data_frame_filtered))
    print(data_frame_filtered[['Taxonomic_Group', 'Lineage']].head())
    if data_frame_filtered.empty:
        print("No taxonomic group found after adjustments.")
        return
    grouped_data = data_frame_filtered.groupby('Taxonomic_Group').agg(
        Total_FAAL_Count=('Taxonomic_Group', 'size'),
        Genome_Count=('Assembly', 'nunique'),
        Assembly_List=('Assembly', lambda assemblies: ';'.join(sorted(assemblies.unique()))),
        Protein_IDs=('Protein Accession', lambda protein_ids: ';'.join(sorted(protein_ids.unique())))
    ).reset_index()
    grouped_data = grouped_data[grouped_data['Genome_Count'] >= 5]
    grouped_data['Mean_FAALs_per_Genome'] = grouped_data['Total_FAAL_Count'] / grouped_data['Genome_Count']
    top_groups = grouped_data.sort_values(by='Mean_FAALs_per_Genome', ascending=False).head(top_n_groups)
    print(f"Top {top_n_groups} taxonomic groups with highest mean (Mean FAALs per Genome):")
    print(top_groups)
    output_table_path = 'top_taxonomic_groups_FAAL.tsv'
    top_groups_sorted = grouped_data.sort_values(by='Total_FAAL_Count', ascending=False)
    with open(output_table_path, 'w') as output_file:
        output_file.write("== Top Taxonomic Groups (FAAL) ==\n")
        output_file.write(top_groups_sorted.to_csv(sep='\t', index=False))
    color_palette = sns.color_palette("viridis", top_n_groups)
    figura, eixo = plt.subplots(figsize=(14, 10))
    barras = eixo.bar(
        top_groups['Taxonomic_Group'],
        top_groups['Mean_FAALs_per_Genome'],
        color=color_palette,
        edgecolor='black',
        alpha=0.85
    )
    eixo.set_xlabel("" + taxonomic_level, fontsize=20, fontweight='bold')
    eixo.set_ylabel('Mean FAAL Count per Genome', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=18)
    for indice, barra in enumerate(barras):
        altura_barra = barra.get_height()
        total_faal = top_groups.iloc[indice]['Total_FAAL_Count']
        numero_genomas = top_groups.iloc[indice]['Genome_Count']
        eixo.text(
            barra.get_x() + barra.get_width()/2,
            altura_barra + 0.03 * altura_barra,
            f'{total_faal}',
            ha='center', va='bottom',
            fontsize=16,
            fontweight='bold',
            color='black',
            clip_on=False
        )
        cor_label = 'black' if numero_genomas > 1000 else 'white'
        eixo.text(
            barra.get_x() + barra.get_width()/2,
            altura_barra/2,
            f'{numero_genomas}',
            ha='center', va='center',
            fontsize=16,
            fontweight='bold',
            color=cor_label,
            clip_on=False
        )
    valor_maximo_medio = top_groups['Mean_FAALs_per_Genome'].max()
    limite_superior = valor_maximo_medio * 1.1
    eixo.set_ylim(bottom=0, top=limite_superior)
    if taxonomic_level == "Phylum":
        limite_tick_superior = np.ceil(limite_superior / 0.5) * 0.5
        ticks_eixo_y = np.arange(0, limite_tick_superior + 0.5, 0.5)
    else:
        limite_tick_superior = int(np.ceil(limite_superior))
        ticks_eixo_y = np.arange(0, limite_tick_superior + 1, 1)
    eixo.set_yticks(ticks_eixo_y)
    eixo.margins(x=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
    plt.savefig('barplot_mean_faal_per_genome.png', dpi=plot_dpi)
    plt.savefig('barplot_mean_faal_per_genome.svg', dpi=plot_dpi)
    plt.show()
    print("Results table saved to:", output_table_path)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 bar_faal_all_countsv2.py <table1.tsv> <Domain(s)> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    table1_file_path = sys.argv[1]
    domain_argument = sys.argv[2]      # e.g., "Bacteria", "Eukaryota" or "Bacteria,Eukaryota"
    taxonomic_level = sys.argv[3]        # e.g., "Order", "Phylum", "Genus", etc.
    top_n_groups = int(sys.argv[4])
    plot_dpi = int(sys.argv[5])
    generate_barplot(table1_file_path, domain_argument, taxonomic_level, top_n_groups, plot_dpi)








