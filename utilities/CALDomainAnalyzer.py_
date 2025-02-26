import os
import argparse
from pathlib import Path
from Bio import SeqIO
import csv
import shutil
import logging
from datetime import datetime

def contains_domain(feature, domains):
    """
    Verifica se uma feature contém um dos domínios especificados em uma variedade de qualificadores.

    Args:
        feature (SeqFeature): Uma feature do registro GenBank.
        domains (list): Lista de domínios a procurar.

    Returns:
        dict: Contagem dos domínios encontrados.
    """
    domain_count = {domain: 0 for domain in domains}
    relevant_qualifiers = [
        "detection_rule", "detection_rules", "NRPS_PKS", "sec_met_domain",
        "domains", "aSDomain", "description", "domain_id", "label"
    ]
    for qual in relevant_qualifiers:
        if qual in feature.qualifiers:
            for value in feature.qualifiers[qual]:
                value_lower = value.lower()
                for domain in domains:
                    if domain.lower() in value_lower:
                        domain_count[domain] += 1
    return domain_count

def extract_genome_id(subdir_name):
    """
    Extrai o ID do genoma do nome do subdiretório.
    Pode lidar com dois padrões:
    1. Entre "Result_" e o segundo "_" após "GCA_" ou "GCF_".
    2. Entre "Result_" e ".fna".

    Args:
        subdir_name (str): Nome do subdiretório.

    Returns:
        str: ID do genoma extraído ou vazio se não encontrado.
    """
    genome_id = ""
    if subdir_name.startswith("Result_"):
        # Primeiro padrão: Result_GCA_xxxxx ou Result_GCF_xxxxx
        parts = subdir_name.split("_")
        if len(parts) > 2 and parts[1] in {"GCA", "GCF"}:
            genome_id = f"{parts[1]}_{parts[2]}"
        # Segundo padrão: Result_BGCalfanumericocode.fna
        elif subdir_name.endswith(".fna"):
            start = len("Result_")
            end = subdir_name.rfind(".fna")
            if end > start:
                genome_id = subdir_name[start:end]
    return genome_id

def format_size(size_mb):
    """
    Formata o tamanho em Megabases e Gigabases.

    Args:
        size_mb (float): Tamanho em Megabases.

    Returns:
        tuple: Tamanho formatado em Megabases e Gigabases.
    """
    size_gb = size_mb / 1000
    return f"{size_mb:.2f} Mb", f"{size_gb:.2f} Gb"

def setup_logging(log_file):
    """
    Configura o módulo de logging para registrar informações no arquivo de log.

    Args:
        log_file (Path): Caminho para o arquivo de log.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Também loga no console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def process_gbk_files(input_dir, log_file, search_amp_binding=False):
    """
    Processa arquivos .gbk em um diretório de entrada, verifica a presença de domínios
    e calcula o tamanho total em megabases e gigabases de cada subdiretório.
    Além disso, copia os arquivos .gbk identificados com 'CAL_domain' para um diretório filtrado
    com prefixo do Genome ID.

    Args:
        input_dir (Path): Diretório contendo subdiretórios com arquivos .gbk.
        log_file (Path): Arquivo de log para salvar informações sobre o processamento.
        search_amp_binding (bool): Se True, também procura por 'AMP-binding' na análise.
    """
    input_dir = Path(input_dir)
    log_file = Path(log_file)

    # Configura o logging
    setup_logging(log_file)
    logging.info("Iniciando o processamento dos arquivos .gbk")

    if not input_dir.is_dir():
        logging.error(f"O diretório de entrada '{input_dir}' não existe ou não é um diretório válido.")
        return

    summary_data = []
    total_size_mb = 0.0
    subdirs_accessed = 0
    subdirs_with_cal = 0

    # Diretório principal para armazenar os subdiretórios filtrados com sufixo _CAL
    filtered_main_dir = input_dir.parent / "filtrados_subdir_CAL"
    filtered_main_dir.mkdir(exist_ok=True)
    logging.info(f"Diretório de saída para arquivos filtrados: {filtered_main_dir}")

    # Verifica o uso de disco no diretório pai onde filtered_main_dir será criado
    try:
        total, used, free = shutil.disk_usage(input_dir.parent)
        free_mb = free / 1_000_000  # Convertendo bytes para megabases
        free_mb_str, free_gb_str = format_size(free_mb)
        logging.info(f"Espaço disponível no disco: {free_mb_str} / {free_gb_str}")
    except Exception as e:
        logging.error(f"Erro ao verificar o uso de disco no diretório '{input_dir.parent}': {e}")
        return

    for subdir in input_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("Result_"):
            genome_id = extract_genome_id(subdir.name)
            subdirs_accessed += 1
            if genome_id:
                logging.info(f"Processando subdiretório: {subdir.name} com Genome ID: {genome_id}")
            else:
                logging.warning(f"Subdiretório {subdir.name} sem Genome ID válido.")

            if not genome_id:
                # Inclui no resumo com contagens zeradas
                summary_entry = {
                    "Assembly": "N/A",
                    "CAL_domain": 0,
                    "AMP-binding": 0 if search_amp_binding else "N/A",
                    "Total_size_MB": "0.00 Mb",
                    "Total_size_GB": "0.00 Gb"
                }
                summary_data.append(summary_entry)
                continue  # Pula para o próximo subdiretório

            gbk_files = list(subdir.glob("*.gbk"))
            if not gbk_files:
                logging.warning(f"Subdiretório {subdir.name} ignorado (nenhum arquivo .gbk encontrado)")
                # Inclui no resumo com contagens zeradas
                summary_entry = {
                    "Assembly": genome_id,
                    "CAL_domain": 0,
                    "AMP-binding": 0 if search_amp_binding else "N/A",
                    "Total_size_MB": "0.00 Mb",
                    "Total_size_GB": "0.00 Gb"
                }
                summary_data.append(summary_entry)
                continue  # Pula para o próximo subdiretório

            # Preferir arquivos com ".region" no nome
            region_gbk_files = list(subdir.glob("*.region*.gbk"))
            if region_gbk_files:
                gbk_files = region_gbk_files
                logging.info(f"Preferindo arquivos com '.region' no nome para o subdiretório {subdir.name}")

            cal_count = 0
            amp_count = 0
            subdir_size_mb = 0.0  # em megabases

            # Lista para armazenar os arquivos que contêm CAL_domain
            gbk_files_to_copy = []

            for gbk_file in gbk_files:
                try:
                    logging.info(f"  Processando arquivo: {gbk_file.name}")
                    file_has_cal = False
                    file_cal_count = 0
                    file_amp_count = 0
                    file_size_bp = 0

                    # Corrigindo o erro: passar o caminho como string
                    for record in SeqIO.parse(str(gbk_file), "genbank"):
                        file_size_bp += len(record.seq)
                        for feature in record.features:
                            domains_to_search = ["CAL_domain"]
                            if search_amp_binding:
                                domains_to_search.append("AMP-binding")
                            domain_counts = contains_domain(feature, domains_to_search)
                            if domain_counts["CAL_domain"] > 0:
                                file_cal_count += domain_counts["CAL_domain"]
                                file_has_cal = True
                            if search_amp_binding and domain_counts.get("AMP-binding", 0) > 0:
                                file_amp_count += domain_counts["AMP-binding"]

                    if file_has_cal:
                        cal_count += file_cal_count
                        if search_amp_binding:
                            amp_count += file_amp_count
                        subdir_size_mb += file_size_bp / 1_000_000  # Convertendo para megabases
                        size_mb_str, size_gb_str = format_size(file_size_bp / 1_000_000)
                        logging.info(f"    Encontrado 'CAL_domain' no arquivo: {gbk_file.name} (Tamanho: {size_mb_str} / {size_gb_str})")

                        # Adiciona o arquivo à lista de arquivos a serem copiados
                        gbk_files_to_copy.append(gbk_file)
                except Exception as e:
                    logging.error(f"  Erro ao processar o arquivo {gbk_file}: {e}")

            # Verifica se há pelo menos um arquivo com CAL_domain
            if cal_count > 0:
                subdirs_with_cal += 1
                size_mb_str, size_gb_str = format_size(subdir_size_mb)
                summary_entry = {
                    "Assembly": genome_id,
                    "CAL_domain": cal_count,
                    "AMP-binding": amp_count if search_amp_binding else "N/A",
                    "Total_size_MB": size_mb_str,
                    "Total_size_GB": size_gb_str
                }
                summary_data.append(summary_entry)
                total_size_mb += subdir_size_mb
                logging.info(f"  Subdiretório {subdir.name} processado: CAL_domain={cal_count}, AMP-binding={amp_count if search_amp_binding else 'N/A'}, Total_size_MB={size_mb_str} / Total_size_GB={size_gb_str}\n")

                # Cria o subdiretório filtrado e copia apenas os arquivos identificados com prefixo Genome ID
                filtered_subdir = filtered_main_dir / f"{subdir.name}_CAL"
                filtered_subdir.mkdir(exist_ok=True)
                for gbk_file in gbk_files_to_copy:
                    try:
                        # Novo nome do arquivo com prefixo Genome ID
                        new_filename = f"{genome_id}_{gbk_file.name}"
                        destination_file = filtered_subdir / new_filename
                        shutil.copy2(gbk_file, destination_file)
                        logging.info(f"    Copiado para: {destination_file}")
                    except Exception as e:
                        logging.error(f"    Erro ao copiar o arquivo {gbk_file} para {filtered_subdir}: {e}")
            else:
                # Inclui no resumo com contagens zeradas
                summary_entry = {
                    "Assembly": genome_id,
                    "CAL_domain": 0,
                    "AMP-binding": amp_count if search_amp_binding else "N/A",
                    "Total_size_MB": "0.00 Mb",
                    "Total_size_GB": "0.00 Gb"
                }
                summary_data.append(summary_entry)
                logging.info(f"  Subdiretório {subdir.name} não contém arquivos com 'CAL_domain'\n")

    # Resumo final no log
    total_size_mb_str, total_size_gb_str = format_size(total_size_mb)
    logging.info(f"Total de subdiretórios acessados: {subdirs_accessed}")
    logging.info(f"Total de subdiretórios com 'CAL_domain': {subdirs_with_cal}")
    logging.info(f"Tamanho total de todos os arquivos .gbk encontrados: {total_size_mb_str} / {total_size_gb_str}")
    logging.info(f"Espaço disponível no disco para cópia: {free_mb_str} / {free_gb_str}")
    if total_size_mb <= free_mb:
        logging.info("Há espaço suficiente no disco para copiar os arquivos.")
    else:
        logging.warning("Não há espaço suficiente no disco para copiar os arquivos.")

    # Escreve a tabela de resumo em um arquivo CSV
    summary_file = input_dir / "summary.csv"
    try:
        with open(summary_file, "w", newline="") as csvfile:
            fieldnames = ["Assembly", "CAL_domain", "AMP-binding", "Total_size_MB", "Total_size_GB"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        logging.info(f"Resumo salvo em: {summary_file}")
    except Exception as e:
        logging.error(f"Erro ao escrever o arquivo de resumo CSV: {e}")

    logging.info("Processamento concluído.")
    logging.info(f"Total de subdiretórios acessados: {subdirs_accessed}")
    logging.info(f"Total de subdiretórios com 'CAL_domain': {subdirs_with_cal}")
    logging.info(f"Tamanho total de todos os arquivos .gbk encontrados: {total_size_mb_str} / {total_size_gb_str}")
    logging.info(f"Resumo salvo em: {summary_file}")
    logging.info(f"Arquivos filtrados copiados para o diretório: {filtered_main_dir}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analisa subdiretórios com domínios CAL_domain e opcionalmente AMP-binding, "
            "calculando o tamanho total dos arquivos .gbk encontrados para verificar espaço em disco disponível. "
            "Além disso, copia os arquivos identificados para um diretório filtrado com prefixo Genome ID."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Diretório de entrada contendo subdiretórios com arquivos .gbk."
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Arquivo de log para salvar informações sobre o processamento. Exemplo: log_processamento.log"
    )
    parser.add_argument(
        "--search-amp-binding",
        action="store_true",
        help="Se especificado, também procura por 'AMP-binding'."
    )

    args = parser.parse_args()

    # Verifica se o log_file já existe e não é um diretório
    log_file_path = Path(args.log_file)
    if log_file_path.exists() and log_file_path.is_dir():
        print(f"O caminho do log file '{log_file_path}' é um diretório. Por favor, especifique um arquivo.")
        exit(1)

    # Inicia o processamento
    process_gbk_files(args.input_dir, args.log_file, args.search_amp_binding)

if __name__ == "__main__":
    main()


