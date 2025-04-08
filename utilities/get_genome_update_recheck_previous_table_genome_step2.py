import os
import sys
import logging
import pandas as pd
from Bio import Entrez
import subprocess
import time
import xml.etree.ElementTree as ET
from http.client import IncompleteRead  # para capturar erros de leitura incompleta

# Cria o diretório de logs, se não existir
os.makedirs("./log", exist_ok=True)
logging.basicConfig(
    filename="./log/process.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Verifica os argumentos de entrada
if len(sys.argv) < 3:
    logging.error("Uso: python script.py <input_table> <final_output_filename>")
    print("Usage: python script.py <input_table> <final_output_filename>")
    sys.exit(1)

input_table = sys.argv[1]
final_output_filename = sys.argv[2]

# Configura o Entrez com seu e-mail e chave de API
Entrez.email = 'your_email@example.com'
Entrez.api_key = '36893de1f94eda657a8428ae5f1a6dd7f409'

def remove_kingdom_from_lineage(lineage):
    """
    Remove o reino (segundo nível) da linhagem se este terminar com "ati".
    Exemplo:
      Entrada: "Bacteria; Bacillati; Actinomycetota; unclassified Actinomycetota"
      Saída:  "Bacteria; Actinomycetota; unclassified Actinomycetota"
    Se a linhagem for "Not Available" ou não tiver o formato esperado, retorna-a sem alterações.
    """
    if lineage == "Not Available":
        return lineage
    parts = [p.strip() for p in lineage.split(';')]
    if len(parts) >= 2 and parts[1].lower().endswith("ati"):
        new_parts = [parts[0]] + parts[2:]
        return "; ".join(new_parts)
    return lineage

def get_taxonomic_rank(protein_accession):
    """
    Recupera a espécie e a linhagem taxonômica usando efetch (db=protein).
    """
    try:
        handle = Entrez.efetch(db='protein', id=protein_accession, retmode='xml')
        records = Entrez.read(handle)
        handle.close()
        time.sleep(0.1)
        species = records[0].get('GBSeq_organism', "Not Available")
        lineage = records[0].get('GBSeq_taxonomy', "Not Available")
        return species, lineage
    except (Exception, IncompleteRead) as e:
        logging.error(f"Error retrieving data for Protein Accession {protein_accession}: {e}")
        return "Not Available", "Not Available"

def get_genome_accession(protein_accession):
    """
    Tenta obter o genome (assembly) accession via efetch no formato ipg.
    Implementa retry com backoff exponencial:
      - Tentativas: 10
      - Delay inicial: 10 segundos, dobrando até 90 segundos.
    Procura nos rótulos "RefSeq", "INSDC" e "GenBank".
    """
    attempts = 10
    delay = 10
    max_delay = 90
    for i in range(attempts):
        try:
            ipg_result = subprocess.check_output(
                ['efetch', '-db', 'protein', '-id', protein_accession, '-format', 'ipg'],
                universal_newlines=True
            ).strip()
            refseq_info = None
            insdc_info = None
            genbank_info = None
            for line in ipg_result.split('\n'):
                if 'RefSeq' in line:
                    refseq_info = line.split()[-1]
                elif 'INSDC' in line:
                    insdc_info = line.split()[-1]
                elif 'GenBank' in line:
                    genbank_info = line.split()[-1]
            genome_accession = refseq_info or insdc_info or genbank_info
            time.sleep(5)
            return genome_accession if genome_accession else "Not Available"
        except subprocess.CalledProcessError as e:
            logging.error(f"Protein {protein_accession} - Tentativa {i+1}: Falha ao buscar genome ({e}).")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
    return "Not Available"

def fetch_assembly_data(assembly_accession):
    """
    Recupera os metadados do assembly (formato TSV) usando o comando datasets.
    Retorna uma linha de dados com os campos separados por tabulação.
    Os 13 campos retornados são:
      0: accession
      1: organism-name
      2: organism-common-name
      3: organism-tax-id
      4: assminfo-level
      5: assminfo-bioproject
      6: assminfo-biosample-accession
      7: assmstats-gc-percent
      8: assmstats-total-sequence-len
      9: assminfo-sequencing-tech
      10: assminfo-release-date
      11: assminfo-biosample-collection-date
      12: assminfo-biosample-description-title
    """
    command = (
        f"datasets summary genome accession {assembly_accession} --as-json-lines | "
        f"dataformat tsv genome --fields accession,organism-name,organism-common-name,organism-tax-id,"
        f"assminfo-level,assminfo-bioproject,assminfo-biosample-accession,assmstats-gc-percent,"
        f"assmstats-total-sequence-len,assminfo-sequencing-tech,assminfo-release-date,"
        f"assminfo-biosample-collection-date,assminfo-biosample-description-title"
    )
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            output_lines = result.stdout.decode('utf-8').splitlines()
            if output_lines and output_lines[0].startswith("Assembly Accession"):
                output_lines = output_lines[1:]
            if output_lines:
                time.sleep(5)
                return output_lines[0]
        else:
            logging.error(f"Assembly {assembly_accession}: {result.stderr.decode('utf-8')}")
    except Exception as e:
        logging.error(f"Erro ao executar comando para Assembly {assembly_accession}: {e}")
    return "Not Available"

def fetch_biosample_metadata(assembly_accession):
    """
    Recupera os metadados do BioSample associados ao assembly.
    Extrai os atributos:
      - geo_loc_name   --> location
      - isolation_source
      - environmental_sample
    """
    try:
        search_handle = Entrez.esearch(db='assembly', term=f'{assembly_accession}[Assembly Accession]')
        search_record = Entrez.read(search_handle)
        search_handle.close()
        time.sleep(0.1)
        if search_record['IdList']:
            assembly_uid = search_record['IdList'][0]
            link_handle = Entrez.elink(dbfrom='assembly', id=assembly_uid, db='biosample')
            link_records = Entrez.read(link_handle)
            link_handle.close()
            time.sleep(0.1)
            if link_records[0]['LinkSetDb']:
                biosample_uid = link_records[0]['LinkSetDb'][0]['Link'][0]['Id']
                fetch_handle = Entrez.efetch(db='biosample', id=biosample_uid, rettype='xml')
                xml_data = fetch_handle.read()
                fetch_handle.close()
                time.sleep(0.1)
                root = ET.fromstring(xml_data)
                biosample_attributes = {}
                for attribute in root.findall('.//Attribute'):
                    attr_name = attribute.get('attribute_name', '').strip().lower()
                    attr_value = attribute.text.strip() if attribute.text else "Not Available"
                    biosample_attributes[attr_name] = attr_value
                location = biosample_attributes.get('geo_loc_name', "Not Available")
                isolation_source = biosample_attributes.get('isolation_source', "Not Available")
                environmental_sample = biosample_attributes.get('environmental_sample', "Not Available")
                return {
                    "location": location,
                    "isolation_source": isolation_source,
                    "environmental_sample": environmental_sample
                }
        return {"location": "Not Available", "isolation_source": "Not Available", "environmental_sample": "Not Available"}
    except Exception as e:
        logging.error(f"Erro ao buscar biosample para Assembly {assembly_accession}: {e}")
        return {"location": "Not Available", "isolation_source": "Not Available", "environmental_sample": "Not Available"}

def get_lineage_from_taxid(taxid):
    """
    Tenta recuperar a linhagem (lineage) usando o Tax ID pelo banco de dados taxonomy.
    """
    try:
        handle = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        time.sleep(0.1)
        if records and isinstance(records, list):
            lineage = records[0].get("Lineage", "Not Available")
            return lineage
        else:
            return "Not Available"
    except Exception as e:
        logging.error(f"Erro ao recuperar lineage para TaxID {taxid}: {e}")
        return "Not Available"

# Tenta obter a opção de atualizar a coluna "Lineage"
try:
    if sys.stdin.isatty():
        update_lineage_option = input("Deseja atualizar a coluna 'Lineage'? (s/n): ").strip().lower()
    else:
        update_lineage_option = "y"
        logging.info("STDIN não é interativo. Usando valor padrão para update_lineage_option: y")
except OSError as e:
    logging.error(f"Erro de input: {e}. Usando valor padrão 'y'.")
    update_lineage_option = "y"

# Leitura da tabela de entrada (deve conter as colunas "Protein.accession" e "Assembly")
df_in = pd.read_csv(input_table, sep="\t", header=0)

# Atualiza o campo Assembly se estiver vazio, "Not Available" ou se não iniciar com "GCF_" ou "GCA_"
genome_ids = []
for index, row in df_in.iterrows():
    protein_id = str(row["Protein.accession"]).strip()
    assembly_id = str(row["Assembly"]).strip()
    if (assembly_id == "Not Available" or assembly_id == "" or
       (not assembly_id.startswith("GCF_") and not assembly_id.startswith("GCA_"))):
        logging.info(f"Linha {index+1}: Assembly '{assembly_id}' inválido para Protein {protein_id}. Atualizando...")
        assembly_id = get_genome_accession(protein_id)
        if assembly_id == "Not Available":
            assembly_id = protein_id
    genome_ids.append(assembly_id)
df_in["Assembly"] = genome_ids

# Define as colunas finais (27 colunas)
final_columns = [
    "Protein.accession",
    "Assembly",
    "Species",
    "Lineage",
    "Assembly BioProject Lineage Title",
    "Assembly BioSample Attribute Name",
    "Assembly BioSample Attribute Value",
    "Assembly BioSample Description Title",
    "Assembly BioSample Sample Identifiers Database",
    "Assembly BioSample Models",
    "Assembly BioSample Owner Name",
    "Assembly BioSample Package",
    "Assembly BioSample Publication date",
    "Assembly Level",
    "Assembly Notes",
    "Assembly Sequencing Tech",
    "Assembly Stats GC Count",
    "Assembly Stats GC Percent",
    "Assembly Stats Genome Coverage",
    "Assembly Stats Total Sequence Length",
    "Current Accession",
    "Organism Name",
    "Organism Taxonomic ID",
    "Assembly Stats Total Sequence Length MB",
    "Location",
    "Isolation Source",
    "Environmental Sample"
]

data = []
for index, row in df_in.iterrows():
    protein_id = row["Protein.accession"]
    assembly_id = row["Assembly"]
    logging.info(f"Processando linha {index+1}: Protein.accession: {protein_id}, Assembly: {assembly_id}")
    
    # Recupera dados taxonômicos usando o Protein ID
    species, lineage = get_taxonomic_rank(protein_id)
    
    # Recupera dados do assembly via datasets
    dataset_line = fetch_assembly_data(assembly_id)
    if dataset_line != "Not Available":
        fields = dataset_line.split('\t')
    else:
        fields = ["Not Available"] * 13

    # Mapeia os campos retornados do datasets
    assembly_bioproject_lineage_title = fields[5] if len(fields) > 5 else "Not Available"
    assembly_level = fields[4] if len(fields) > 4 else "Not Available"
    assembly_sequencing_tech = fields[9] if len(fields) > 9 else "Not Available"
    assembly_stats_gc_percent = fields[7] if len(fields) > 7 else "Not Available"
    assembly_stats_total_seq_length = fields[8] if len(fields) > 8 else "Not Available"
    organism_name_field = fields[1] if len(fields) > 1 else "Not Available"
    organism_tax_id_field = fields[3] if len(fields) > 3 else "Not Available"
    assembly_biosample_description_title = fields[12] if len(fields) > 12 else "Not Available"
    assembly_biosample_pub_date = fields[10] if len(fields) > 10 else "Not Available"
    
    # Campos não obtidos via datasets são preenchidos como "Not Available"
    assembly_biosample_attr_name = "Not Available"
    assembly_biosample_attr_value = "Not Available"
    assembly_biosample_sample_ids_db = "Not Available"
    assembly_biosample_models = "Not Available"
    assembly_biosample_owner_name = "Not Available"
    assembly_biosample_package = "Not Available"
    assembly_notes = "Not Available"
    assembly_stats_gc_count = "Not Available"
    assembly_stats_genome_coverage = "Not Available"
    
    # Cálculo do Total Sequence Length em MB
    try:
        total_seq_length_mb = float(assembly_stats_total_seq_length) / 1e6 if assembly_stats_total_seq_length != "Not Available" else "Not Available"
    except Exception as e:
        logging.error(f"Linha {index+1} - Erro ao calcular Total Sequence Length MB: {e}")
        total_seq_length_mb = "Not Available"
    
    # Se o usuário optou por atualizar a coluna "Lineage" e o valor atual estiver vazio ou "Not Available",
    # tenta atualizar utilizando o Tax ID obtido do dataset.
    if update_lineage_option == "s" and (lineage == "Not Available" or lineage.strip() == ""):
        if organism_tax_id_field != "Not Available":
            lineage = get_lineage_from_taxid(organism_tax_id_field)
    
    # Remove o reino (segundo nível) da linhagem, se aplicável
    lineage = remove_kingdom_from_lineage(lineage)
    
    # Recupera os metadados do BioSample (location, isolation_source, environmental_sample)
    biosample_data = fetch_biosample_metadata(assembly_id)
    location = biosample_data.get("location", "Not Available")
    isolation_source = biosample_data.get("isolation_source", "Not Available")
    environmental_sample = biosample_data.get("environmental_sample", "Not Available")
    
    # Monta a linha final (27 colunas)
    row_data = [
        protein_id,                                  # Protein.accession
        assembly_id,                                 # Assembly
        species,                                     # Species
        lineage,                                     # Lineage (atualizada e com reino removido)
        assembly_bioproject_lineage_title,           # Assembly BioProject Lineage Title
        assembly_biosample_attr_name,                # Assembly BioSample Attribute Name
        assembly_biosample_attr_value,               # Assembly BioSample Attribute Value
        assembly_biosample_description_title,        # Assembly BioSample Description Title
        assembly_biosample_sample_ids_db,            # Assembly BioSample Sample Identifiers Database
        assembly_biosample_models,                   # Assembly BioSample Models
        assembly_biosample_owner_name,               # Assembly BioSample Owner Name
        assembly_biosample_package,                  # Assembly BioSample Package
        assembly_biosample_pub_date,                 # Assembly BioSample Publication date
        assembly_level,                              # Assembly Level
        assembly_notes,                              # Assembly Notes
        assembly_sequencing_tech,                    # Assembly Sequencing Tech
        assembly_stats_gc_count,                     # Assembly Stats GC Count
        assembly_stats_gc_percent,                   # Assembly Stats GC Percent
        assembly_stats_genome_coverage,              # Assembly Stats Genome Coverage
        assembly_stats_total_seq_length,             # Assembly Stats Total Sequence Length
        assembly_id,                                 # Current Accession
        organism_name_field,                         # Organism Name
        organism_tax_id_field,                       # Organism Taxonomic ID
        total_seq_length_mb,                         # Assembly Stats Total Sequence Length MB
        location,                                    # Location (geo_loc_name)
        isolation_source,                            # Isolation Source
        environmental_sample                         # Environmental Sample
    ]
    data.append(row_data)

df_out = pd.DataFrame(data, columns=final_columns)
df_out.to_csv(final_output_filename, sep='\t', index=False)
logging.info(f"Tabela final com todos os metadados salva em {final_output_filename}")
print(f"Tabela final com todos os metadados salva em {final_output_filename}")
