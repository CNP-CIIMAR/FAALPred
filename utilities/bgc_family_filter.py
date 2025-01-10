import pandas as pd
import argparse

##python3 filter_bgcs_characterized_family.py bigscape_copia_update.tsv bigscape_charact_not.tsv
def filter_bgc(input_file, output_file):
    try:
        # Carregar o arquivo, ajustando possíveis problemas de separação e cabeçalhos
        df = pd.read_csv(input_file, delimiter='\t', low_memory=False)

        # Remover espaços dos nomes das colunas
        df.columns = df.columns.str.strip()

        # Confirmar se a coluna 'BGC Accession ID' existe
        if 'Accession ID' not in df.columns:
            print("Erro: A coluna 'Accession ID' não foi encontrada no arquivo.")
            print("Colunas disponíveis:", df.columns.tolist())
            return

        # Filtrar linhas com valores na coluna 'Family Number'
        df_filtered = df[df['Family Number'].notna()]

        # Identificar os BGCs caracterizados no MIBiG (IDs que começam com "BGC")
        bgc_mibig = df_filtered[df_filtered['Accession ID'].str.startswith('BGC')]

        # Obter os números de família dos BGCs caracterizados no MIBiG
        mibig_family_numbers = bgc_mibig['Family Number'].unique()

        # Filtrar os BGCs não caracterizados no MIBiG, mas com o mesmo Family Number
        non_mibig_bgc = df_filtered[
            ~df_filtered['Accession ID'].str.startswith('BGC') &
            df_filtered['Family Number'].isin(mibig_family_numbers)
        ]

        # Combinar os dois DataFrames
        result = pd.concat([bgc_mibig, non_mibig_bgc])

        # Ordenar por 'Family Number'
        result = result.sort_values(by='Family Number')

        # Salvar a nova tabela
        result.to_csv(output_file, index=False, sep='\t')
        print(f"Tabela filtrada e ordenada salva em: {output_file}")

    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Filtra BGCs com base no Family Number relacionado ao MIBiG.")
    parser.add_argument("input_file", help="Caminho para o arquivo de entrada (TSV ou CSV).")
    parser.add_argument("output_file", help="Caminho para o arquivo de saída (TSV ou CSV).")
    args = parser.parse_args()

    # Executar a função
    filter_bgc(args.input_file, args.output_file)
