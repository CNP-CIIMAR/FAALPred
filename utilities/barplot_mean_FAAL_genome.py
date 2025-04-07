import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
from ete3 import NCBITaxa
import matplotlib
import numpy as np  # Importado para ajuste dos ticks

# Inicializa o objeto NCBITaxa para correção da linhagem taxonômica
ncbi_taxa = NCBITaxa()

def standardize_lineage_format(lineage_string):
    """
    Normaliza a string da linhagem para garantir que cada ';' seja seguido por um espaço único,
    removendo espaços extras e garantindo que a string termine com ';'.
    """
    standardized_lineage = re.sub(r'\s*;\s*', '; ', lineage_string)
    standardized_lineage = standardized_lineage.strip()
    if not standardized_lineage.endswith(';'):
        standardized_lineage += ';'
    return standardized_lineage

def extract_taxonomic_group(lineage_string, desired_level):
    """
    Extrai o grupo taxonômico correspondente ao nível desejado a partir da string de linhagem.
    
    Critérios:
      - Para Order: retorna o primeiro token que termina com "ales" (ignorando maiúsculas).
      - Para Family: retorna o primeiro token que termina com "eae" (ignorando maiúsculas).
      - Para Genus:
          * Se a classificação estiver completa (>= 6 tokens), utiliza o token na posição 6 (índice 5)
            desde que este não termine com "ales", nem com "eae", e não comece com "Candidatus".
          * Caso contrário, tenta usar o penúltimo token se disponível.
      - Para Phylum: retorna o segundo token, se disponível.
      - Para outros níveis: utiliza a posição fixa baseada na ordem 
        ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'].
    """
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

def get_corrected_lineage_from_species(species_name):
    """
    A partir do nome da espécie (coluna Species), utiliza ete3 para:
      1. Traduzir o nome para taxid.
      2. Obter a linhagem completa (lista de taxids) e os seus nomes oficiais.
      3. Retornar a linhagem formatada e padronizada.
    Caso ocorra erro, retorna None.
    """
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
        print(f"Erro ao obter a linhagem para a espécie '{species_name}': {error_message}")
        return None

def update_lineage_for_eukaryotes(data_frame):
    """
    Atualiza a coluna 'Lineage' apenas para as linhas de Eukaryota.
    Em vez de usar o taxid, utiliza o nome da espécie (coluna 'Species')
    para obter a linhagem corrigida via ete3.
    """
    if 'Species' not in data_frame.columns:
        raise KeyError("Coluna 'Species' não encontrada no DataFrame.")
    
    mask_eukaryotes = data_frame['Lineage'].astype(str).str.startswith("Eukaryota")
    data_frame.loc[mask_eukaryotes, 'Lineage'] = data_frame.loc[mask_eukaryotes, 'Species'].apply(get_corrected_lineage_from_species)
    return data_frame

def generate_barplot(table1_file_path, domain_argument, taxonomic_level, top_n_groups, plot_dpi):
    # Carrega a tabela
    data_frame = pd.read_csv(table1_file_path, sep='\t', low_memory=False)
    print("Tabela 1 carregada:", len(data_frame))
    print(data_frame.head())
    
    # Exibe a coluna 'Lineage' conforme está na tabela original
    taxonomic_id_column = 'Organism Taxonomic ID' if 'Organism Taxonomic ID' in data_frame.columns else 'Organism Tax ID'
    print("Tabela 1 com 'Lineage' original:")
    print(data_frame[[taxonomic_id_column, 'Lineage']].head())
    
    # Filtra amostras ambientais, se existir a coluna "Sample"
    if "Sample" in data_frame.columns:
        data_frame = data_frame[~data_frame["Sample"].str.contains("environmental", case=False, na=False)]
    
    # Converte a coluna Assembly em string e remove espaços
    data_frame["Assembly"] = data_frame["Assembly"].astype(str).str.strip()
    
    # Filtra (remove) linhas onde a coluna Assembly é vazia ou possui valores inválidos
    assembly_valid_mask = data_frame["Assembly"].str.lower().apply(
        lambda valor: valor not in ["", "none", "na", "null", "not available"]
    )
    data_frame = data_frame[assembly_valid_mask]
    print("Linhas após filtrar Assembly inválido:", len(data_frame))
    
    # Mantém apenas assemblies que iniciam com GCA_ ou GCF_
    data_frame = data_frame[data_frame["Assembly"].str.match(r"^(GCA_|GCF_)")]
    print("Linhas após manter somente IDs iniciando com GCA_/GCF_:", len(data_frame))
    
    # Atualiza a linhagem para Eukaryota usando a coluna Species
    data_frame = update_lineage_for_eukaryotes(data_frame)
    
    # Filtra as linhas cujo Lineage inicia com os domínios especificados (ex.: Bacteria, Eukaryota)
    domain_list = []
    if "Bacteria" in domain_argument:
        domain_list.append("Bacteria")
    if "Eukaryota" in domain_argument:
        domain_list.append("Eukaryota")
    if not domain_list:
        # Se o usuário passou algo como "Archaea", "Bacteria,Eukaryota" etc.
        domain_list = [dominio.strip() for dominio in domain_argument.split(",")]
    
    pattern = f"^({'|'.join(domain_list)});\\s*"
    data_frame_filtered = data_frame[data_frame['Lineage'].notnull() & data_frame['Lineage'].str.match(pattern, case=False)].copy()
    print("Linhas após filtrar Lineage por domínio:", len(data_frame_filtered))
    
    # Extrai o grupo taxonômico conforme o nível desejado
    data_frame_filtered['Taxonomic_Group'] = data_frame_filtered['Lineage'].apply(
        lambda lineage: extract_taxonomic_group(lineage, taxonomic_level)
    )
    
    # Remove a ordem "Candidatus Entotheonellales"
    data_frame_filtered = data_frame_filtered[data_frame_filtered['Taxonomic_Group'] != 'Candidatus Entotheonellales']
    
    # Ajustes extras para o nível Phylum
    if taxonomic_level == "Phylum":
        data_frame_filtered = data_frame_filtered[~data_frame_filtered['Taxonomic_Group'].str.lower().eq('proteobacteria')]
        data_frame_filtered.loc[data_frame_filtered['Taxonomic_Group'].str.lower() == 'deltaproteobacteria', 'Taxonomic_Group'] = 'Pseudomonadota'
        data_frame_filtered = data_frame_filtered[~data_frame_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    print("Após extração e ajustes, linhas com Taxonomic_Group:", len(data_frame_filtered))
    print(data_frame_filtered[['Taxonomic_Group', 'Lineage']].head())
    if data_frame_filtered.empty:
        print("Nenhum grupo taxonômico encontrado após os ajustes.")
        return
    
    # Agrupa os dados por Taxonomic_Group
    grouped_data = data_frame_filtered.groupby('Taxonomic_Group').agg(
        Total_FAAL_Count=('Taxonomic_Group', 'size'),
        Genome_Count=('Assembly', 'nunique'),
        Assembly_List=('Assembly', lambda assemblies: ';'.join(sorted(assemblies.unique()))),
        Protein_IDs=('Protein Accession', lambda protein_ids: ';'.join(sorted(protein_ids.unique())))
    ).reset_index()
    
    # Filtra grupos com menos de 5 genomas depositados
    grouped_data = grouped_data[grouped_data['Genome_Count'] >= 5]
    
    # Calcula o valor médio de FAALs por genoma
    grouped_data['Mean_FAALs_per_Genome'] = grouped_data['Total_FAAL_Count'] / grouped_data['Genome_Count']
    
    # Seleciona os top N grupos com maior média de FAALs por genoma
    top_groups = grouped_data.sort_values(by='Mean_FAALs_per_Genome', ascending=False).head(top_n_groups)
    
    print(f"Top {top_n_groups} grupos taxonômicos com maior média (Mean FAALs per Genome):")
    print(top_groups)
    
    # Salva uma tabela com os resultados de todos os grupos com >= 5 genomas
    output_table_path = 'top_taxonomic_groups_FAAL.tsv'
    top_groups_sorted = grouped_data.sort_values(by='Total_FAAL_Count', ascending=False)
    with open(output_table_path, 'w') as arquivo_saida:
        arquivo_saida.write("== Top Taxonomic Groups (FAAL) ==\n")
        arquivo_saida.write(top_groups_sorted.to_csv(sep='\t', index=False))
    
    # Cria o gráfico
    color_palette = sns.color_palette("viridis", top_n_groups)
    figura, eixo = plt.subplots(figsize=(14, 10))
    
    barras = eixo.bar(
        top_groups['Taxonomic_Group'],
        top_groups['Mean_FAALs_per_Genome'],
        color=color_palette,
        edgecolor='black',
        alpha=0.85
    )
    
    # Define os rótulos dos eixos com fonte em negrito para os títulos
    eixo.set_xlabel("Nível " + taxonomic_level, fontsize=20, fontweight='bold')
    eixo.set_ylabel('Média de FAALs por Genoma', fontsize=20, fontweight='bold')
    
    # Configura os rótulos do eixo x sem negrito e com rotação
    plt.xticks(rotation=45, ha='right', fontsize=18)
    
    # Adiciona o Total de FAALs acima de cada barra e o número de genomas centralizado dentro da barra (em negrito)
    for indice, barra in enumerate(barras):
        altura_barra = barra.get_height()
        total_faal = top_groups.iloc[indice]['Total_FAAL_Count']
        numero_genomas = top_groups.iloc[indice]['Genome_Count']
        
        # Texto acima da barra (Total_FAAL_Count) em negrito
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
        
        # Texto centralizado dentro da barra (Genome_Count) em negrito
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
    
    # Define o limite superior do eixo y e ajusta os ticks
    valor_maximo_medio = top_groups['Mean_FAALs_per_Genome'].max()
    limite_superior = valor_maximo_medio * 1.1
    eixo.set_ylim(bottom=0, top=limite_superior)
    
    # Gera os ticks: se o nível for Phylum, utiliza intervalo de 0.5; caso contrário, utiliza 1.0
    if taxonomic_level == "Phylum":
        limite_tick_superior = np.ceil(limite_superior / 0.5) * 0.5
        ticks_eixo_y = np.arange(0, limite_tick_superior + 0.5, 0.5)
    else:
        limite_tick_superior = int(np.ceil(limite_superior))
        ticks_eixo_y = np.arange(0, limite_tick_superior + 1, 1)
    eixo.set_yticks(ticks_eixo_y)
    
    # Elimina o espaço entre o eixo y e a primeira barra
    eixo.margins(x=0)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
    
    plt.savefig('barplot_mean_faal_per_genome.png', dpi=plot_dpi)
    plt.savefig('barplot_mean_faal_per_genome.svg', dpi=plot_dpi)
    plt.show()
    
    print("Tabela de resultados salva em:", output_table_path)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 bar_faal_all_countsv2.py <table1.tsv> <Domain(s)> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    
    table1_file_path = sys.argv[1]
    domain_argument = sys.argv[2]      # Ex.: "Bacteria", "Eukaryota" ou "Bacteria,Eukaryota"
    taxonomic_level = sys.argv[3]        # Ex.: "Order", "Phylum", "Genus", etc.
    top_n_groups = int(sys.argv[4])
    plot_dpi = int(sys.argv[5])
    
    generate_barplot(table1_file_path, domain_argument, taxonomic_level, top_n_groups, plot_dpi)










