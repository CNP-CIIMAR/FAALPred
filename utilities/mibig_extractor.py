import pandas as pd
import os
import json
import sys

#python3 mibig_data_extractorv2.py bigscape_copia.tsv bigscape_copia_update.tsv /home/mattoslmp/bigscape/mibig_json_files/mibig_json_4.0

def extract_bgc_data(json_dir):
    """
    Percorre todos os arquivos .json no diretório `json_dir`.
    Para cada arquivo, lê o 'accession' e extrai informações de 'compounds', 'substrates' e 'legacy_references'.
    Retorna um dicionário {bgc_id: {campos_interessantes}}.
    """
    bgc_data = {}
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

                # Verifique se o arquivo JSON tem o campo "accession"
                bgc_id = data.get("accession", None)
                if not bgc_id:
                    continue  # Ignorar arquivos sem 'accession'

                # Lista de compostos
                compounds = data.get("compounds", [])

                # Extrair dados dos compostos
                compounds_name = "; ".join([c.get("name", "N/A") for c in compounds])
                bioactivities_name = "; ".join(
                    f"{b.get('name', 'N/A')} (observed: {b.get('observed', 'N/A')})"
                    for c in compounds
                    for b in c.get("bioactivities", [])
                    if isinstance(b, dict)
                )
                structures = "; ".join([c.get("structure", "N/A") for c in compounds])
                database_ids = "; ".join(
                    [", ".join(c.get("databaseIds", [])) for c in compounds]
                )
                substrates = "; ".join(
                    [", ".join(c.get("substrates", [])) for c in compounds]
                )
                mass = "; ".join([str(c.get("mass", "N/A")) for c in compounds])
                formula = "; ".join([c.get("formula", "N/A") for c in compounds])

                # Extrair referências
                legacy_references = "; ".join(data.get("legacy_references", ["N/A"]))

                # Adiciona ao dicionário
                bgc_data[bgc_id] = {
                    "compounds_name": compounds_name,
                    "substrates": substrates,
                    "bioactivities_name": bioactivities_name,
                    "structure": structures,
                    "databaseIds": database_ids,
                    "mass": mass,
                    "formula": formula,
                    "legacy_references": legacy_references,
                }
    return bgc_data

def process_table(input_file, output_file, json_dir):
    """
    Lê a tabela de entrada (TSV), adiciona colunas para as informações de compostos,
    e preenche os valores para BGCs que forem encontrados nos arquivos JSON extraídos.
    """
    df = pd.read_csv(input_file, sep="\t")

    # Cria as novas colunas
    df["compounds_name"] = "N/A"
    df["substrates"] = "N/A"
    df["bioactivities_name"] = "N/A"
    df["structure"] = "N/A"
    df["databaseIds"] = "N/A"
    df["mass"] = "N/A"
    df["formula"] = "N/A"
    df["legacy_references"] = "N/A"

    # Carrega os dados de BGC a partir dos arquivos JSON
    bgc_data = extract_bgc_data(json_dir)

    # Preenche a tabela para cada linha cujo 'BGC' comece com "BGC"
    for idx, row in df.iterrows():
        bgc_id = str(row["BGC"])
        if bgc_id.startswith("BGC") and bgc_id in bgc_data:
            print(f"Preenchendo dados para BGC ID: {bgc_id}")
            df.at[idx, "compounds_name"] = bgc_data[bgc_id]["compounds_name"]
            df.at[idx, "substrates"] = bgc_data[bgc_id]["substrates"]
            df.at[idx, "bioactivities_name"] = bgc_data[bgc_id]["bioactivities_name"]
            df.at[idx, "structure"] = bgc_data[bgc_id]["structure"]
            df.at[idx, "databaseIds"] = bgc_data[bgc_id]["databaseIds"]
            df.at[idx, "mass"] = bgc_data[bgc_id]["mass"]
            df.at[idx, "formula"] = bgc_data[bgc_id]["formula"]
            df.at[idx, "legacy_references"] = bgc_data[bgc_id]["legacy_references"]

    df.to_csv(output_file, sep="\t", index=False)
    print(f"Tabela de saída gerada em: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python script_name.py <arquivo_de_entrada> <arquivo_de_saida> <diretorio_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    json_dir = sys.argv[3]

    process_table(input_file, output_file, json_dir)
