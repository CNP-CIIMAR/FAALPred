import pandas as pd
import argparse

def combine_tables(table1, table2):
    # Converter as tabelas para DataFrames
    df1 = pd.DataFrame(table1)
    df2 = pd.DataFrame(table2)

    # Garantir que as colunas necessárias estão presentes e tratar valores ausentes
    if 'Signature.accession' not in df2.columns or 'Signature.description' not in df2.columns:
        raise ValueError("Tabela 2 deve conter as colunas 'Signature.accession' e 'Signature.description'")

    df2['Signature.accession'] = df2['Signature.accession'].fillna('')
    df2['Signature.description'] = df2['Signature.description'].fillna('')

    # Agrupar a segunda tabela por 'Protein.accession' e contar as assinaturas e combinar descriptions
    grouped_df2 = df2.groupby('Protein.accession').agg({
        'Signature.accession': lambda x: '-'.join(sorted(set(x))),
        'Signature.description': lambda x: '-'.join(sorted(set(x)))
    }).reset_index()

    grouped_df2.rename(columns={'Signature.description': 'Combined Signature description', 'Protein.accession': 'Protein Accession'}, inplace=True)
    grouped_df2['Total Signature Description'] = grouped_df2['Combined Signature description'].apply(lambda x: len(x.split('-')))

    # Adicionar coluna de cor
    def assign_color(description):
        if 'FAAL' in description and len(description.split('-')) == 1:
            return '#FFFFFF'
        else:
            return '#000000'

    grouped_df2['color three'] = grouped_df2['Combined Signature description'].apply(assign_color)

    # Combinar a tabela 1 com as informações agrupadas da tabela 2
    merged_df = pd.merge(df1, grouped_df2, on='Protein Accession', how='left')

    return merged_df

def main(table1_path, table2_path, output_path):
    try:
        table1_data = pd.read_csv(table1_path, delimiter="\t", on_bad_lines='skip')
        table2_data = pd.read_csv(table2_path, delimiter="\t", on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading TSV files: {e}")
        return

    combined_df = combine_tables(table1_data, table2_data)

    combined_df.to_csv(output_path, index=False, sep="\t")
    print(f"Combined table saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two tables based on Protein Accession")
    parser.add_argument("table1_path", help="Path to the first input TSV file")
    parser.add_argument("table2_path", help="Path to the second input TSV file")
    parser.add_argument("output_path", help="Path to the output TSV file")

    args = parser.parse_args()
    main(args.table1_path, args.table2_path, args.output_path)

