import pandas as pd
import argparse

####python3 merge_bigscape_step1.py Network_Annotations_Full.tsv ./mix/mix_clans_0.30_0.70.tsv merged_Network_Annotations_Full_modificada__clans_0.30_0.70_update.tsv

## OBS: alterar a coluna a tabela 2 de BGC name para BGC
def merge_tables(table1_path, table2_path, output_path):
    # Ler as tabelas
    try:
        df1 = pd.read_csv(table1_path, sep='\t', on_bad_lines='skip')  # Ajustado para TSV
        df2 = pd.read_csv(table2_path, sep='\t', on_bad_lines='skip')  # Ajustado para TSV
    except Exception as e:
        print(f"Error reading the files: {e}")
        return

    # Realizar o merge
    merged_df = pd.merge(df1, df2, on="BGC", how="left")  # Merge com base na coluna "BGC"

    # Salvar o resultado em um novo arquivo
    merged_df.to_csv(output_path, sep='\t', index=False)
    print(f"Merged table saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two tables based on BGC column")
    parser.add_argument("table1", help="Path to the first table (TSV format)")
    parser.add_argument("table2", help="Path to the second table (TSV format)")
    parser.add_argument("output", help="Path to save the merged table (TSV format)")
    
    args = parser.parse_args()
    
    merge_tables(args.table1, args.table2, args.output)
