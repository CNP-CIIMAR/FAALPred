import pandas as pd

def extract_taxonomy_info(row):
    taxonomy = row['Taxonomy']
    
    # Tratar valores nulos ou NaN
    if pd.isna(taxonomy):
        return pd.Series([None, None, None])

    levels = [lvl.strip() for lvl in taxonomy.split(';')]

    # Garantir que nÃ­veis existam
    phylum = levels[2] if len(levels) > 2 and levels[1].endswith('group') else (levels[1] if len(levels) > 1 else None)
    order = next((lvl for lvl in levels if lvl.endswith('ales')), None)
    genus = levels[-2] if len(levels) > 1 else None

    return pd.Series([phylum, order, genus])

def process_table(input_file, output_file):
    # Ler a tabela
    df = pd.read_csv(input_file, sep='\t')

    # Verificar se a coluna Taxonomy existe
    if 'Taxonomy' not in df.columns:
        raise ValueError("A coluna 'Taxonomy' nÃ£o foi encontrada no arquivo de entrada.")

    # Criar as colunas extras
    df[['Phylum', 'Order', 'Genus']] = df.apply(extract_taxonomy_info, axis=1)

    # Salvar o resultado
    df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    import argparse

    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Processar tabela para adicionar colunas Phylum, Order e Genus.")
    parser.add_argument("input_file", help="Arquivo de entrada no formato TSV")
    parser.add_argument("output_file", help="Arquivo de saÃ­da no formato TSV")

    args = parser.parse_args()

    # Processar a tabela com os arquivos fornecidos
    process_table(args.input_file, args.output_file)
    print(f"Tabela processada salva em {args.output_file}")

