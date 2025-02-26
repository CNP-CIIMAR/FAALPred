#!/usr/bin/env python3
# organize_big_slice.py

"""
Script para organizar diretórios antiSMASH em datasets para BiG-SLiCE,
baseados em um nível taxonômico escolhido pelo usuário (Phylum, Order ou Genus).
Autor: Leandro de Mattos Pereira (novembro de 2024)
Revisado: Abril de 2025
"""

import os
import sys
import pandas as pd
import shutil
import argparse
from collections import defaultdict
import re
import logging
from logging.handlers import RotatingFileHandler

def parse_arguments():
    """
    Analisa os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Organiza diretórios antiSMASH em datasets para BiG-SLiCE, "
                    "baseados em um nível taxonômico escolhido (Phylum, Order ou Genus)."
    )
    parser.add_argument('--bigslice_dir', type=str, required=True,
                        help='Caminho para o diretório de entrada do BiG-SLiCE (onde os datasets serão criados).')
    parser.add_argument('--antismash_dir', type=str, required=True,
                        help='Caminho para o diretório de resultados do antiSMASH.')
    parser.add_argument('--taxonomy_table', type=str, required=True,
                        help='Caminho para a tabela de taxonomia (TSV).')
    parser.add_argument('--assembly_column', type=str, default='Assembly Accession',
                        help='Nome da coluna na tabela de taxonomia que contém o Assembly Accession (default: "Assembly Accession").')
    parser.add_argument('--lineage_column', type=str, default='Lineage',
                        help='Nome da coluna na tabela de taxonomia que contém a Lineage (default: "Lineage").')
    parser.add_argument('--log_file', type=str, default='organize_big_slice.log',
                        help='Caminho para o arquivo de log (default: "organize_big_slice.log").')
    parser.add_argument('--taxonomic_level', type=str, default='Genus',
                        choices=['Phylum', 'Order', 'Genus'],
                        help='Escolha o nível taxonômico a ser usado para agrupar os resultados (Phylum, Order ou Genus). '
                             'O padrão é "Genus".')
    parser.add_argument('--verbose', action='store_true',
                        help='Ativa a verbosidade detalhada do logging.')
    return parser.parse_args()

def setup_logging(log_file, verbose=False):
    """
    Configura o sistema de logging com rotação.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(console_formatter)
    logger.addHandler(console)

def sanitize_dataset_name(taxon):
    """
    Sanitiza o nome do dataset baseado no nome do taxon.
    Remove ou substitui caracteres inválidos para nomes de diretórios.
    """
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '_', taxon)
    return f"dataset_{sanitized}"

def extract_assembly_accession(directory_name):
    """
    Extrai o Assembly Accession do nome do diretório usando regex.
    Exemplos:
        'Result_GCA_000007865.1.fasta_genomic.fna' -> 'GCA_000007865.1'
        'Result_GCA_910592015.1_Diversispora_eburnea_AZ414A_genomic.fna' -> 'GCA_910592015.1'
        'Result_GCA_027600645.1_ASM2760064v1_genomic.fna' -> 'GCA_027600645.1'
    """
    pattern = r'Result_(GCA|GCF)_(\d+\.\d+)'
    match = re.search(pattern, directory_name, re.IGNORECASE)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    else:
        return None

def get_taxonomic_level(lineage, level_name):
    """
    Extrai o nível taxonômico (Phylum, Order ou Genus/Família) a partir de uma string de lineage.
    - Para 'Order': encontra o primeiro nível que termina com 'ales'.
    - Para 'Phylum': usa o 4º nível (index=3).
    - Para 'Genus': extrai o Gênero do nível imediatamente após a Família que termina com 'aceae'.
      Se o Gênero não puder ser identificado ou for inválido (termina com 'aceae'), faz fallback para a Família.
    Caso não encontre, retorna 'Unknown'.
    """
    if not lineage:
        return 'Unknown'

    # Divide a linhagem em níveis, removendo espaços extras
    levels = [lvl.strip() for lvl in lineage.split(';')]

    if level_name.lower() == 'order':
        # Encontrar o primeiro nível que termina com 'ales'
        for lvl in levels:
            if lvl.endswith('ales'):
                logging.debug(f"Order extraído: {lvl}")
                return lvl
        logging.debug("Order não encontrado, retornando 'Unknown'")
        return 'Unknown'

    elif level_name.lower() == 'phylum':
        if len(levels) >=4:
            phylum = levels[3]
            logging.debug(f"Phylum extraído: {phylum}")
            return phylum
        else:
            logging.debug("Phylum não encontrado, retornando 'Unknown'")
            return 'Unknown'

    elif level_name.lower() == 'genus':
        # Encontrar o primeiro nível que termina com 'aceae' (Família)
        family_index = None
        for idx, lvl in enumerate(levels):
            if lvl.endswith('aceae'):
                family_index = idx
                break

        if family_index is not None and family_index +1 < len(levels):
            genus = levels[family_index +1]
            if not genus.endswith('aceae'):
                # Validar se a espécie começa com o Gênero
                if family_index +2 < len(levels):
                    species = levels[family_index +2]
                    species_first_word = species.split()[0].lower()
                    genus_lower = genus.lower()
                    if species_first_word == genus_lower:
                        logging.debug(f"Gênero '{genus}' confirmado pela espécie '{species}'")
                        return genus
                    else:
                        logging.debug(f"Espécie '{species}' não confirma o Gênero '{genus}', usando Família '{levels[family_index]}'")
                        return levels[family_index]
                else:
                    logging.debug(f"Gênero '{genus}' extraído sem validação pela espécie")
                    return genus
            else:
                # Genus inválido, usa Família
                logging.debug(f"Gênero extraído '{genus}' termina com 'aceae', usando Família '{levels[family_index]}'")
                return levels[family_index]
        elif family_index is not None:
            # Não há Genus após Família, usa Família
            logging.debug(f"Genus não disponível após Família, usando Família '{levels[family_index]}'")
            return levels[family_index]
        else:
            # Não encontrou Família, retorna 'Unknown'
            logging.debug("Família não encontrada, retornando 'Unknown'")
            return 'Unknown'

    logging.debug("Nível taxonômico inválido, retornando 'Unknown'")
    return 'Unknown'

def load_taxonomy(taxonomy_table_path, assembly_column, lineage_column, taxonomic_level):
    """
    Carrega e processa a tabela de taxonomia.
    Retorna um DataFrame com uma nova coluna 'SelectedTaxon' (Phylum, Order ou Genus/Família).
    """
    logging.info("Carregando a tabela de taxonomia...")
    try:
        df_taxonomy = pd.read_csv(taxonomy_table_path, sep='\t')
    except FileNotFoundError:
        logging.error(f"A tabela de taxonomia '{taxonomy_table_path}' não foi encontrada.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logging.error(f"Erro ao analisar a tabela de taxonomia: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro inesperado ao ler a tabela de taxonomia: {e}")
        sys.exit(1)

    if assembly_column not in df_taxonomy.columns or lineage_column not in df_taxonomy.columns:
        logging.error(f"As colunas '{assembly_column}' e/ou '{lineage_column}' não foram encontradas na tabela de taxonomia.")
        logging.error(f"Colunas disponíveis: {df_taxonomy.columns.tolist()}")
        sys.exit(1)

    df_taxonomy[lineage_column] = df_taxonomy[lineage_column].fillna('').astype(str)

    logging.info(f"Extraindo o nível taxonômico selecionado: {taxonomic_level}")
    df_taxonomy['SelectedTaxon'] = df_taxonomy[lineage_column].apply(lambda x: get_taxonomic_level(x, taxonomic_level))

    unknown_count = df_taxonomy['SelectedTaxon'].value_counts().get('Unknown', 0)
    logging.info(f"Total de 'SelectedTaxon' como 'Unknown': {unknown_count}")

    # Logar algumas amostras para verificar
    sample_taxa = df_taxonomy['SelectedTaxon'].drop_duplicates().sample(n=min(5, df_taxonomy['SelectedTaxon'].nunique()), random_state=1)
    for taxon in sample_taxa:
        logging.debug(f"Exemplo de taxa: {taxon}")

    return df_taxonomy

def create_dataset_mapping(df_taxonomy, assembly_column):
    """
    Cria um mapeamento do nível taxonômico selecionado para nomes de datasets.
    Cada 'SelectedTaxon' único recebe um nome de dataset único no formato 'dataset_{Genus}'.
    """
    unique_taxa = sorted(df_taxonomy['SelectedTaxon'].unique())
    taxon_to_dataset = {}
    
    for taxon in unique_taxa:
        if taxon == 'Unknown':
            taxon_to_dataset[taxon] = 'dataset_unknown'
        else:
            sanitized_taxon = sanitize_dataset_name(taxon)
            dataset_name = sanitized_taxon
            taxon_to_dataset[taxon] = dataset_name

    accession_to_taxon = pd.Series(df_taxonomy.SelectedTaxon.values,
                                   index=df_taxonomy[assembly_column]).to_dict()

    total_taxa = len(taxon_to_dataset) - (1 if 'Unknown' in taxon_to_dataset else 0)
    logging.info(f"Total de taxa únicos (excluindo 'Unknown'): {total_taxa}")
    if 'Unknown' in taxon_to_dataset:
        logging.info("Taxa 'Unknown' mapeada para 'dataset_unknown'")

    # Logar todos os mapeamentos para verificar
    for taxon, dataset in taxon_to_dataset.items():
        logging.debug(f"Mapeamento de taxa: {taxon} -> {dataset}")

    return taxon_to_dataset, accession_to_taxon

def initialize_datasets_tsv(bigslice_dir):
    """
    Inicializa (ou sobrescreve) o arquivo datasets.tsv com os cabeçalhos.
    """
    datasets_tsv_path = os.path.join(bigslice_dir, "datasets.tsv")
    try:
        with open(datasets_tsv_path, 'w') as f:
            f.write("#Dataset name\tPath to dataset folder\tPath to taxonomy file\tDescription of the dataset\n")
    except Exception as e:
        logging.error(f"Erro ao criar o arquivo 'datasets.tsv': {e}")
        sys.exit(1)
    return datasets_tsv_path

def process_antismash_directories(antismash_dir, bigslice_dir,
                                  taxon_to_dataset, accession_to_taxon,
                                  df_taxonomy, assembly_column, lineage_column):
    """
    Processa os diretórios de resultados do antiSMASH e organiza-os em datasets do BiG-SLiCE.
    """
    taxonomy_data = defaultdict(list)
    taxonomy_folder = os.path.join(bigslice_dir, "taxonomy")
    os.makedirs(taxonomy_folder, exist_ok=True)

    all_result_dirs = [d for d in os.listdir(antismash_dir)
                       if os.path.isdir(os.path.join(antismash_dir, d))]

    total_dirs = len(all_result_dirs)
    logging.info(f"Total de diretórios de resultado antiSMASH encontrados: {total_dirs}")

    for idx, result_dir in enumerate(all_result_dirs, 1):
        if idx % 1000 == 0 or idx == total_dirs:
            logging.info(f"Processando {idx}/{total_dirs} diretórios...")

        accession = extract_assembly_accession(result_dir)
        if not accession:
            logging.warning(f"O nome do diretório '{result_dir}' não corresponde ao padrão esperado. Pulando...")
            continue

        taxon = accession_to_taxon.get(accession, 'Unknown')
        dataset_name = taxon_to_dataset.get(taxon, 'dataset_unknown')

        result_dir_path = os.path.join(antismash_dir, result_dir)
        dataset_dir_path = os.path.join(bigslice_dir, dataset_name)
        genome_folder_name = f"genome_{accession}"
        genome_dir_path = os.path.join(dataset_dir_path, genome_folder_name)

        os.makedirs(genome_dir_path, exist_ok=True)

        gbk_files_copied = False
        for file in os.listdir(result_dir_path):
            if file.endswith(".gbk"):
                src_file = os.path.join(result_dir_path, file)
                dst_file = os.path.join(genome_dir_path, file)
                try:
                    shutil.copy2(src_file, dst_file)
                    if not gbk_files_copied:
                        logging.info(f"Arquivo copiado: {src_file} -> {dst_file}")
                        gbk_files_copied = True
                except Exception as e:
                    logging.error(f"Erro ao copiar o arquivo '{src_file}' para '{dst_file}': {e}")

        if not gbk_files_copied:
            logging.warning(f"Nenhum arquivo '.gbk' encontrado no diretório '{result_dir_path}'.")

        # Recuperar informações de taxonomia para registro
        tax_info = df_taxonomy[df_taxonomy[assembly_column] == accession]
        if not tax_info.empty:
            tax_info = tax_info.iloc[0]
            # Dividir lineage em níveis
            lineage_levels = [lvl.strip() for lvl in tax_info[lineage_column].split(';')]

            # Estrutura esperada da linhagem do NCBI:
            # 0: root
            # 1: cellular organisms
            # 2: Domain
            # 3: Phylum
            # 4: Class
            # 5: Order
            # 6: Family
            # 7: Genus
            # 8: Species
            # 9: Organism/Strain

            def safe_get(lst, index):
                return lst[index] if index < len(lst) else 'Unknown'

            # Montar um dicionário
            genome_folder_entry = {
                'Genome folder name': f"{genome_folder_name}/",
                'Domain': safe_get(lineage_levels, 2),
                'Phylum': safe_get(lineage_levels, 3),
                'Class': safe_get(lineage_levels, 4),
                'Order': safe_get(lineage_levels, 5),
                'Family': safe_get(lineage_levels, 6),
                'Genus': safe_get(lineage_levels, 7),
                'Species': safe_get(lineage_levels, 8),
                'Organism/Strain': safe_get(lineage_levels, 9),
            }
            taxonomy_data[dataset_name].append(genome_folder_entry)

            # Logar o mapeamento
            logging.debug(f"Genome '{accession}' mapeado para taxa '{taxon}' no dataset '{dataset_name}'")
        else:
            logging.warning(f"Informações de taxonomia para '{accession}' não encontradas.")

    return taxonomy_data

def generate_taxonomy_files(taxonomy_data, taxonomy_folder, bigslice_dir, selected_level):
    """
    Gera arquivos TSV de taxonomia para cada dataset e atualiza o arquivo datasets.tsv.
    """
    datasets_tsv_path = os.path.join(bigslice_dir, "datasets.tsv")
    for dataset, entries in taxonomy_data.items():
        taxonomy_file_path = os.path.join(taxonomy_folder, f"taxonomy_{dataset}.tsv")
        df_taxonomy_dataset = pd.DataFrame(entries)
        try:
            df_taxonomy_dataset.to_csv(taxonomy_file_path, sep='\t', index=False)
            logging.info(f"Arquivo de taxonomia criado: {taxonomy_file_path}")
        except Exception as e:
            logging.error(f"Erro ao criar o arquivo de taxonomia '{taxonomy_file_path}': {e}")
            continue

        try:
            with open(datasets_tsv_path, 'a') as f:
                description = f"Dataset agrupado por {selected_level}: {dataset}"
                f.write(f"{dataset}\t{dataset}\ttaxonomy/taxonomy_{dataset}.tsv\t{description}\n")
        except Exception as e:
            logging.error(f"Erro ao atualizar 'datasets.tsv' com o dataset '{dataset}': {e}")

def main():
    # Parse dos argumentos
    args = parse_arguments()
    setup_logging(args.log_file, verbose=args.verbose)

    bigslice_dir = args.bigslice_dir
    antismash_dir = args.antismash_dir
    taxonomy_table_path = args.taxonomy_table
    assembly_column = args.assembly_column
    lineage_column = args.lineage_column
    taxonomic_level = args.taxonomic_level  # Phylum, Order ou Genus

    logging.info("Iniciando o processo de organização dos datasets para BiG-SLiCE.")
    logging.info(f"Diretório BiG-SLiCE: {bigslice_dir}")
    logging.info(f"Diretório antiSMASH: {antismash_dir}")
    logging.info(f"Tabela de Taxonomia: {taxonomy_table_path}")
    logging.info(f"Coluna de Assembly: {assembly_column}")
    logging.info(f"Coluna de Lineage: {lineage_column}")
    logging.info(f"Nível taxonômico selecionado: {taxonomic_level}")

    if not os.path.isdir(antismash_dir):
        logging.error(f"O diretório antiSMASH '{antismash_dir}' não existe.")
        sys.exit(1)
    if not os.path.isfile(taxonomy_table_path):
        logging.error(f"A tabela de taxonomia '{taxonomy_table_path}' não existe.")
        sys.exit(1)
    os.makedirs(bigslice_dir, exist_ok=True)

    # Carregar e processar dados de taxonomia
    df_taxonomy = load_taxonomy(taxonomy_table_path, assembly_column, lineage_column, taxonomic_level)

    # Criar mapeamentos
    taxon_to_dataset, accession_to_taxon = create_dataset_mapping(df_taxonomy, assembly_column)

    # Inicializar datasets.tsv
    datasets_tsv_path = initialize_datasets_tsv(bigslice_dir)

    # Processar diretórios do antiSMASH
    taxonomy_data = process_antismash_directories(
        antismash_dir=antismash_dir,
        bigslice_dir=bigslice_dir,
        taxon_to_dataset=taxon_to_dataset,
        accession_to_taxon=accession_to_taxon,
        df_taxonomy=df_taxonomy,
        assembly_column=assembly_column,
        lineage_column=lineage_column
    )

    # Gerar arquivos de taxonomia e atualizar datasets.tsv
    generate_taxonomy_files(
        taxonomy_data=taxonomy_data,
        taxonomy_folder=os.path.join(bigslice_dir, "taxonomy"),
        bigslice_dir=bigslice_dir,
        selected_level=taxonomic_level
    )

    logging.info("Organização de datasets concluída com sucesso!")
    logging.info(f"Arquivo 'datasets.tsv' criado em: {datasets_tsv_path}")

if __name__ == "__main__":
    main()

