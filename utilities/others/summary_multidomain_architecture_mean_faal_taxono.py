import pandas as pd
from ete3 import NCBITaxa
import random
import sys
from collections import defaultdict
#python3 summary_multidomain_architecture_mean_faal_taxono.py Genomes_total_15_thousand.tsv Genomes_total_15_thousand_interproscan.tsv Eukaryota 10 40 saida
# Initialize the NCBITaxa object
ncbi = NCBITaxa()

# Function to simplify and combine signature descriptions
def combine_signature_descriptions(df):
    # Verificar se a coluna 'Signature.description' está presente
    if 'Signature.description' not in df.columns:
        raise ValueError("Erro: A coluna 'Signature.description' não foi encontrada no DataFrame.")

    # Simplificar e combinar as descrições de assinatura
    def simplify_signature(description):
        if 'NRPS' in description:
            return 'NRPS'
        if 'PKS' in description:
            return 'PKS'
        return description
    
    combined = df.groupby('Protein.accession')['Signature.description'].apply(
        lambda x: '-'.join(sorted(set(simplify_signature(desc) for desc in x)))
    ).reset_index()
    
    # Renomear a coluna para Combined.description
    combined = combined.rename(columns={'Signature.description': 'Combined.description'})
    
    # Substituir "FAAL" por "FAAL stand-alone"
    combined['Combined.description'] = combined['Combined.description'].apply(
        lambda x: 'FAAL stand-alone' if x == 'FAAL' else x
    )
    
    # Mesclar de volta ao DataFrame original
    df = pd.merge(df.drop(columns=['Signature.description'], errors='ignore'), combined, on='Protein.accession')
    return df

# Function to load data from files
def load_data(table1_path, table2_path):
    print(f"Carregando dados de {table1_path} e {table2_path}...")
    df1 = pd.read_csv(table1_path, sep='\t')
    df2 = pd.read_csv(table2_path, sep='\t')
    print("Colunas do DataFrame 1:", df1.columns.tolist())
    print("Colunas do DataFrame 2:", df2.columns.tolist())
    print("Formato do DataFrame 1:", df1.shape)
    print("Formato do DataFrame 2:", df2.shape)
    if 'Assembly' not in df1.columns:
        raise ValueError("'Assembly' coluna não encontrada na primeira tabela")
    valid_assemblies = df1['Assembly'].str.startswith(('GCA', 'GCF'))
    if not valid_assemblies.any():
        raise ValueError("Nenhum Assembly ID válido (iniciando com GCA ou GCF) encontrado nos dados")
    print(f"Número de Assembly IDs válidos: {valid_assemblies.sum()}")
    print("Primeiras linhas do DataFrame 1:")
    print(df1.head())
    print("\nPrimeiras linhas do DataFrame 2:")
    print(df2.head())
    return df1, df2

# Function to merge tables
def merge_tables(df1, df2, merge_on='Protein.accession'):
    print("Mesclando tabelas...")
    if merge_on not in df1.columns or merge_on not in df2.columns:
        print(f"Erro: coluna '{merge_on}' não encontrada em um ou ambos os DataFrames.")
        print("Colunas disponíveis no DataFrame 1:", df1.columns.tolist())
        print("Colunas disponíveis no DataFrame 2:", df2.columns.tolist())
        raise ValueError(f"Coluna '{merge_on}' não encontrada para mesclagem")

    # Realizar a mesclagem
    merged_df = pd.merge(df1, df2, on=merge_on, how='inner')
    print(f"Formato do DataFrame mesclado: {merged_df.shape}")

    # Verificar se a coluna 'Signature.description' está presente após a mesclagem
    if 'Signature.description' not in merged_df.columns:
        raise ValueError("Erro: A coluna 'Signature.description' não foi encontrada após a mesclagem.")
    else:
        print("Coluna 'Signature.description' encontrada com sucesso.")

    # Gerar 'Combined.description' imediatamente após a mesclagem
    merged_df = combine_signature_descriptions(merged_df)

    # Imprimir algumas linhas para depuração
    print("Primeiras linhas do DataFrame mesclado com 'Combined.description':")
    print(merged_df.head())
    
    return merged_df

# Function to update the 'Lineage' column in the DataFrame
def update_lineage(df, domain_name):
    print(f"Extraindo informações taxonômicas para {domain_name}...")
    taxonomic_data = df['Lineage'].apply(extract_taxonomic_levels)
    for level in ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus']:
        df[level] = taxonomic_data.apply(lambda x: x.get(level, None))
    if domain_name in ['Bacteria', 'Archaea']:
        df['superkingdom'] = df['Lineage'].apply(lambda x: x.split(';')[0].strip() if pd.notna(x) else None)
    df = df[df['superkingdom'] == domain_name]
    df = df.dropna(subset=['phylum', 'order', 'genus'])
    print(f"Forma do DataFrame após filtragem por domínio: {df.shape}")
    return df

# Function to extract taxonomic levels
def extract_taxonomic_levels(lineage):
    levels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomic_dict = {level: None for level in levels}
    if pd.isna(lineage) or not isinstance(lineage, str):
        return taxonomic_dict
    taxa = lineage.split('; ')
    taxonomic_dict['phylum'] = get_phylum(lineage)
    for taxon in taxa:
        taxid = ncbi.get_name_translator([taxon])
        if taxon in taxid:
            rank = ncbi.get_rank([taxid[taxon][0]])
            if rank[taxid[taxon][0]] in levels:
                taxonomic_dict[rank[taxid[taxon][0]]] = taxon
    return taxonomic_dict

# Function to get the phylum from the lineage
def get_phylum(lineage):
    if not lineage or not isinstance(lineage, str):
        return None
    taxa = lineage.split('; ')
    for taxon in taxa:
        taxid = ncbi.get_name_translator([taxon])
        if taxon in taxid:
            rank = ncbi.get_rank([taxid[taxon][0]])
            if rank[taxid[taxon][0]] == 'phylum':
                return taxon
    return None

# Function to calculate the mean FAAL stand-alone and mean multidomain per taxon
def calculate_mean_faal_multidomain_per_genome(df):
    faal_df = df[df['Combined.description'] == 'FAAL stand-alone']
    multidomain_df = df[df['Combined.description'] != 'FAAL stand-alone']

    faal_count_per_genome = faal_df.groupby('Assembly')['Protein.accession'].nunique()
    multidomain_count_per_genome = multidomain_df.groupby('Assembly')['Protein.accession'].nunique()

    mean_faal_per_taxon = faal_df.groupby(['phylum', 'order', 'genus']).apply(
        lambda group: faal_count_per_genome.reindex(group['Assembly']).mean()
    ).reset_index(name='Mean_FAAL_per_Genome')

    mean_multidomain_per_taxon = multidomain_df.groupby(['phylum', 'order', 'genus']).apply(
        lambda group: multidomain_count_per_genome.reindex(group['Assembly']).mean()
    ).reset_index(name='Mean_Multidomain_per_Genome')

    return mean_faal_per_taxon, mean_multidomain_per_taxon

# Function to create the final sorted table with domain names and counts
def create_final_sorted_table(mean_faal_per_taxon, mean_multidomain_per_taxon, df, output_file):
    print("Criando tabela final com phylum, order, genus, Mean_FAAL_per_Genome, Mean_Multidomain_per_Genome e Nome do domain counts...")

    # Mescla as médias de FAAL e multidomínios em uma única tabela
    combined_df = pd.merge(mean_faal_per_taxon, mean_multidomain_per_taxon, on=['phylum', 'order', 'genus'], how='outer')

    # Preencher valores NaN com 0 para contagens
    combined_df['Mean_FAAL_per_Genome'] = combined_df['Mean_FAAL_per_Genome'].fillna(0)
    combined_df['Mean_Multidomain_per_Genome'] = combined_df['Mean_Multidomain_per_Genome'].fillna(0)

    # Função para obter os domínios mais comuns em ordem decrescente
    def get_domain_counts(group):
        domain_counts = group['Combined.description'].value_counts()
        return ', '.join([f"{domain}:{count}" for domain, count in domain_counts.items()])

    # Adicionar a coluna "Nome do domain counts"
    combined_df['Nome do domain counts'] = combined_df.apply(lambda row: get_domain_counts(df[(df['phylum'] == row['phylum']) & 
                                                                                                (df['order'] == row['order']) & 
                                                                                                (df['genus'] == row['genus'])]), axis=1)

    # Ordena por contagens decrescentes de FAAL
    final_sorted_df = combined_df.sort_values(by=['Mean_FAAL_per_Genome', 'Mean_Multidomain_per_Genome'], ascending=[False, False])

    # Salvar a tabela final como CSV
    final_sorted_df.to_csv(f"{output_file}_final_sorted.csv", index=False)

    print(f"Tabela final combinada criada e salva como {output_file}_final_sorted.csv")

# Function to create representative selection with balanced FAAL and multidomain
def create_representative_selection(df, total_genomes, top_n):
    print("Colunas disponíveis no DataFrame antes de combinar assinaturas:")
    print(df.columns.tolist())

    print("Criando seleção representativa com base na média de FAAL e combinações de múltiplos domínios...")

    mean_faal_per_taxon, mean_multidomain_per_taxon = calculate_mean_faal_multidomain_per_genome(df)

    top_faal_taxa = mean_faal_per_taxon.nlargest(top_n, 'Mean_FAAL_per_Genome')
    top_multidomain_taxa = mean_multidomain_per_taxon.nlargest(top_n, 'Mean_Multidomain_per_Genome')

    selection_data = defaultdict(list)
    selected_genomes = set()

    def select_genomes(taxon_df, n):
        available_genomes = set(taxon_df['Assembly']) - selected_genomes
        selected = random.sample(list(available_genomes), min(n, len(available_genomes)))
        selected_genomes.update(selected)
        return selected

    for _, row in top_faal_taxa.iterrows():
        taxon_df = df[(df['phylum'] == row['phylum']) & (df['order'] == row['order']) & (df['genus'] == row['genus'])]

        faal_taxon_df = taxon_df[taxon_df['Combined.description'] == 'FAAL stand-alone']
        taxon_genomes = select_genomes(faal_taxon_df, total_genomes // 2)
        if taxon_genomes:
            selection_data['Taxonomic Level'].append('Genus/Order/Phylum')
            selection_data['Taxon'].append(f"{row['phylum']} / {row['order']} / {row['genus']}")
            selection_data['Multidomain Architecture'].append('FAAL stand-alone')
            selection_data['Architecture Count'].append(len(taxon_genomes))
            # Converte todos os valores para string
            selection_data['Selected Genomes'].append(', '.join(map(str, taxon_genomes)))

    for _, row in top_multidomain_taxa.iterrows():
        taxon_df = df[(df['phylum'] == row['phylum']) & (df['order'] == row['order']) & (df['genus'] == row['genus'])]

        multidomain_taxon_df = taxon_df[taxon_df['Combined.description'] != 'FAAL stand-alone']
        domain_combinations = multidomain_taxon_df.groupby('Protein.accession')['Combined.description'].first()
        top_architectures = domain_combinations.value_counts().nlargest(top_n)
        for arch in top_architectures.index:
            arch_df = multidomain_taxon_df[multidomain_taxon_df['Combined.description'] == arch]
            taxon_genomes = select_genomes(arch_df, total_genomes // 2)
            if taxon_genomes:
                selection_data['Taxonomic Level'].append('Genus/Order/Phylum')
                selection_data['Taxon'].append(f"{row['phylum']} / {row['order']} / {row['genus']}")
                selection_data['Multidomain Architecture'].append(arch)
                selection_data['Architecture Count'].append(top_architectures[arch])
                # Converte todos os valores para string
                selection_data['Selected Genomes'].append(', '.join(map(str, taxon_genomes)))

    selection_df = pd.DataFrame(selection_data)

    print(f"Total de genomas únicos selecionados: {len(selected_genomes)}")
    print(f"FAAL stand-alone genomas selecionados: {len(selection_df[selection_df['Multidomain Architecture'] == 'FAAL stand-alone'])}")
    print(f"Multidomain genomas selecionados: {len(selection_df[selection_df['Multidomain Architecture'] != 'FAAL stand-alone'])}")

    return selection_df

# Function to create simplified summary
def create_simplified_summary(df, top_n):
    print("Criando resumo simplificado...")

    if 'Combined.description' not in df.columns:
        print("Erro: A coluna 'Combined.description' não foi encontrada no DataFrame.")
        return None

    summary_data = defaultdict(list)

    for level in ['phylum', 'order', 'genus']:
        top_taxa = df[level].value_counts().nlargest(top_n).index

        for taxon in top_taxa:
            taxon_df = df[df[level] == taxon]

            domain_combinations = taxon_df.groupby('Protein.accession').apply(lambda x: x['Combined.description'].iloc[0])
            top_architectures = domain_combinations.value_counts().nlargest(top_n)

            for arch, count in top_architectures.items():
                summary_data['Taxonomic Level'].append(level.capitalize())
                summary_data['Taxon'].append(taxon)
                summary_data['Rank'].append(df[level].value_counts()[taxon])
                summary_data['Multidomain Architecture'].append(arch)
                summary_data['Architecture Count'].append(count)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Taxonomic Level', 'Rank', 'Architecture Count'], ascending=[True, False, False])
    print("Resumo simplificado completo.")
    return summary_df

# Function for debugging taxonomic classification
def debug_taxonomic_classification(df):
    print("\nDepurando classificação taxonômica:")
    print(f"Total de linhas: {len(df)}")
    for level in ['phylum', 'order', 'genus']:
        if level in df.columns:
            print(f"\nÚnicos {level}s: {df[level].nunique()}")
            print(f"\nTop 10 {level}s mais comuns:")
            print(df[level].value_counts().head(10))
        else:
            print(f"Coluna {level} não encontrada no DataFrame.")
    print("\nAmostra de linhas:")
    cols_to_display = [col for col in ['Species', 'phylum', 'order', 'genus'] if col in df.columns]
    print(df[cols_to_display].head())

# Main function to generate summary and representative selection
def generate_summary_and_representative_selection(table1_path, table2_path, domain_name, total_genomes, top_n, output_file):
    df1, df2 = load_data(table1_path, table2_path)
    merged_df = merge_tables(df1, df2)
    
    # A coluna 'Combined.description' já está gerada após a mesclagem das tabelas
    filtered_df = update_lineage(merged_df, domain_name)
    debug_taxonomic_classification(filtered_df)
    if filtered_df.empty:
        print(f"Nenhum dado encontrado para o domínio: {domain_name}")
        return
    
    mean_faal_per_taxon, mean_multidomain_per_taxon = calculate_mean_faal_multidomain_per_genome(filtered_df)
    
    create_final_sorted_table(mean_faal_per_taxon, mean_multidomain_per_taxon, filtered_df, output_file)
    
    selection_df = create_representative_selection(filtered_df, total_genomes, top_n)
    simplified_df = create_simplified_summary(filtered_df, top_n)
    
    selection_df.to_csv(f"{output_file}_representative_selection.csv", index=False)
    if simplified_df is not None:
        simplified_df.to_csv(f"{output_file}_simplified_summary.csv", index=False)
        print(f"Resumo simplificado salvo em {output_file}_simplified_summary.csv")
    print(f"Seleção representativa salva em {output_file}_representative_selection.csv")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Uso: python script.py <table1_path> <table2_path> <domain_name> <total_genomes> <top_n> <output_file>")
        sys.exit(1)
    table1_path = sys.argv[1]
    table2_path = sys.argv[2]
    domain_name = sys.argv[3]
    try:
        total_genomes = int(sys.argv[4])
        top_n = int(sys.argv[5])
    except ValueError:
        print("Erro: <total_genomes> e <top_n> devem ser inteiros")
        sys.exit(1)
    output_file = sys.argv[6]
    if domain_name not in ['Bacteria', 'Archaea', 'Eukaryota']:
        print("Erro: <domain_name> deve ser um dos 'Bacteria', 'Archaea', 'Eukaryota'")
        sys.exit(1)
    generate_summary_and_representative_selection(table1_path, table2_path, domain_name, total_genomes, top_n, output_file)
