import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
from ete3 import NCBITaxa
#matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')  # Definir backend antes de importar pyplot
import matplotlib.pyplot as plt


# Inicializa o NCBITaxa para correção da linhagem taxonômica
ncbi = NCBITaxa()

def standardize_lineage_format(lineage):
    """
    Normaliza a string da linhagem para garantir que cada ';' seja seguido por um espaço único,
    removendo espaços extras e garantindo que a string termine com ';'.
    """
    lineage = re.sub(r'\s*;\s*', '; ', lineage)
    lineage = lineage.strip()
    if not lineage.endswith(';'):
        lineage += ';'
    return lineage

def extract_taxonomic_group(lineage, level):
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
    tokens = [token.strip() for token in lineage.split(';') if token.strip()]
    if level == 'Order':
        for token in tokens:
            if token.lower().endswith('ales'):
                return token
    elif level == 'Family':
        for token in tokens:
            if token.lower().endswith('eae'):
                return token
    elif level == 'Genus':
        if len(tokens) >= 6:
            candidate = tokens[5]
            if not (candidate.lower().endswith('ales') or candidate.lower().endswith('eae') or candidate.startswith("Candidatus")):
                return candidate
        if len(tokens) >= 2:
            candidate = tokens[-2]
            if not (candidate.lower().endswith('ales') or candidate.lower().endswith('eae') or candidate.startswith("Candidatus")):
                return candidate
        return None
    elif level == 'Phylum':
        if len(tokens) > 1:
            return tokens[1]
    else:
        levels_order = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        try:
            index = levels_order.index(level)
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
        name2taxid = ncbi.get_name_translator([species_name])
        if species_name not in name2taxid:
            return None
        taxid = name2taxid[species_name][0]
        lineage_ids = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage_ids)
        desired_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        ranks = ncbi.get_rank(lineage_ids)
        lineage_names = [names[t].strip() for t in lineage_ids if ranks[t] in desired_ranks]
        raw_lineage = '; '.join(lineage_names) + ';'
        return standardize_lineage_format(raw_lineage)
    except Exception as e:
        print(f"Erro ao obter a linhagem para a espécie '{species_name}': {e}")
        return None

def update_lineage_eukaryotes(df):
    """
    Atualiza a coluna 'Lineage' apenas para as linhas de Eukaryota.
    Em vez de usar o taxid, utiliza o nome da espécie (coluna 'Species')
    para obter a linhagem corrigida via ete3.
    """
    if 'Species' not in df.columns:
        raise KeyError("Coluna 'Species' não encontrada no DataFrame.")
    
    mask = df['Lineage'].astype(str).str.startswith("Eukaryota")
    df.loc[mask, 'Lineage'] = df.loc[mask, 'Species'].apply(get_corrected_lineage_from_species)
    return df

def generate_barplot(table1_path, domain_arg, taxonomic_level, top_n, dpi):
    # Carrega a tabela
    df = pd.read_csv(table1_path, sep='\t', low_memory=False)
    print("Tabela 1 carregada:", len(df))
    print(df.head())
    
    # Exibe a coluna 'Lineage' conforme está na tabela original
    taxid_col = 'Organism Taxonomic ID' if 'Organism Taxonomic ID' in df.columns else 'Organism Tax ID'
    print("Tabela 1 com 'Lineage' original:")
    print(df[[taxid_col, 'Lineage']].head())
    
    # Filtra amostras ambientais, se existir a coluna "Sample"
    if "Sample" in df.columns:
        df = df[~df["Sample"].str.contains("environmental", case=False, na=False)]
    
    # Filtro para Assembly: converte para string, remove espaços, converte para minúsculas e descarta valores inválidos
    df["Assembly"] = df["Assembly"].astype(str)
    mask_assembly = df["Assembly"].str.strip().str.lower().apply(lambda x: x not in ["", "none", "na", "null", "not available"])
    df = df[mask_assembly]
    print("Linhas após filtrar Assembly válido:", len(df))
    
    # Atualiza a linhagem para Eukaryota usando a coluna Species
    df = update_lineage_eukaryotes(df)
    
    # Filtra as linhas cujo Lineage inicia com os domínios especificados
    domain_list = []
    if "Bacteria" in domain_arg:
        domain_list.append("Bacteria")
    if "Eukaryota" in domain_arg:
        domain_list.append("Eukaryota")
    if not domain_list:
        domain_list = [dom.strip() for dom in domain_arg.split(",")]
    
    pattern = f"^({'|'.join(domain_list)});\\s*"
    df_filtered = df[df['Lineage'].notnull() & df['Lineage'].str.match(pattern, case=False)].copy()
    print("Linhas após filtrar Lineage por domínio:", len(df_filtered))
    
    # Extrai o grupo taxonômico conforme o nível desejado
    df_filtered['Taxonomic_Group'] = df_filtered['Lineage'].apply(lambda x: extract_taxonomic_group(x, taxonomic_level))
    
    # Para nível Phylum, realiza ajustes adicionais
    if taxonomic_level == "Phylum":
        df_filtered = df_filtered[~df_filtered['Taxonomic_Group'].str.lower().eq('proteobacteria')]
        df_filtered.loc[df_filtered['Taxonomic_Group'].str.lower() == 'deltaproteobacteria', 'Taxonomic_Group'] = 'Pseudomonadota'
        df_filtered = df_filtered[~df_filtered['Taxonomic_Group'].str.contains("environmental", case=False, na=False)]
    
    print("Após extração e ajustes, linhas com Taxonomic_Group:", len(df_filtered))
    print(df_filtered[['Taxonomic_Group', 'Lineage']].head())
    if df_filtered.empty:
        print("Nenhum grupo taxonômico encontrado após os ajustes.")
        return
    
    # Agrupa os dados por Taxonomic_Group
    group_counts = df_filtered.groupby('Taxonomic_Group').agg(
        Total_FAAL_Count=('Taxonomic_Group', 'size'),
        Genome_Count=('Assembly', 'nunique')
    ).reset_index()
    
    # Filtra grupos com menos de 5 genomas depositados
    group_counts = group_counts[group_counts['Genome_Count'] >= 5]
    
    # Calcula o Mean FAAL Count per Genome
    group_counts['Mean_FAALs_per_Genome'] = group_counts['Total_FAAL_Count'] / group_counts['Genome_Count']
    
    # Seleciona os top N grupos com maior média
    top_groups = group_counts.sort_values(by='Mean_FAALs_per_Genome', ascending=False).head(top_n)
    
    print(f"Top {top_n} grupos taxonômicos com maior média (Mean FAALs per Genome):")
    print(top_groups)
    
    # Gera e salva uma tabela com os resultados
    output_table_path = 'top_taxonomic_groups_FAAL.tsv'
    top_groups_sorted = group_counts.sort_values(by='Total_FAAL_Count', ascending=False)
    with open(output_table_path, 'w') as f:
        f.write("== Top Taxonomic Groups (FAAL) ==\n")
        f.write(top_groups_sorted.to_csv(sep='\t', index=False))
    
    # Cria o gráfico de barras
    colors = sns.color_palette("viridis", top_n)
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.bar(top_groups['Taxonomic_Group'], top_groups['Mean_FAALs_per_Genome'],
                  color=colors, edgecolor='black', alpha=0.85)
    
    xlabel = f"{taxonomic_level} Level" if taxonomic_level in ["Phylum", "Order", "Genus"] else taxonomic_level
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold')
    ax.set_ylabel('Mean FAAL Count per Genome', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=18, fontweight='bold')
    
    # Faz com que o primeiro e o último bar toquem as bordas do gráfico
    ax.set_xlim(left=-0.5, right=len(top_groups)-0.5)
    
    # Adiciona o Total FAAL Count acima de cada barra e o Genome Count centralizado dentro da barra.
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        total_faal = top_groups.iloc[idx]['Total_FAAL_Count']
        genome_count = top_groups.iloc[idx]['Genome_Count']
        # Total FAAL Count acima da barra
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03 * height,
                f'{total_faal}', ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', clip_on=False)
        # Genome Count centralizado na barra; se genome_count > 1000, texto em preto
        label_color = 'black' if genome_count > 1000 else 'white'
        ax.text(bar.get_x() + bar.get_width()/2, height/2,
                f'{genome_count}', ha='center', va='center', fontsize=16, fontweight='bold', color=label_color, clip_on=False)
    
    max_mean = top_groups['Mean_FAALs_per_Genome'].max()
    ax.set_ylim(bottom=0, top=max_mean * 1.1)
    ax.margins(y=0)
    
    # Ajusta as margens para que os rótulos do eixo X fiquem visíveis e o gráfico ocupe toda a largura
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
    
    plt.savefig('barplot_mean_faal_per_genome.png', dpi=dpi)
    plt.savefig('barplot_mean_faal_per_genome.svg', dpi=dpi)
    plt.show()
    
    print("Tabela de resultados salva em:", output_table_path)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 bar_faal_all_countsv2.py <table1.tsv> <Domain(s)> <Taxonomic Level> <Top N> <DPI>")
        sys.exit(1)
    
    table1_path = sys.argv[1]
    domain_arg = sys.argv[2]      # Ex.: "Bacteria", "Eukaryota" ou "Bacteria,Eukaryota"
    taxonomic_level = sys.argv[3] # Ex.: "Order", "Phylum", "Genus", etc.
    top_n = int(sys.argv[4])
    dpi = int(sys.argv[5])
    
    generate_barplot(table1_path, domain_arg, taxonomic_level, top_n, dpi)



