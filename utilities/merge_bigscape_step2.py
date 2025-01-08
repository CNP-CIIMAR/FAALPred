import pandas as pd
import argparse
import re
import csv
import logging
import sys

def extract_assembly_key(bgc_name):
    """
    Extracts the assembly key from the BGC name.
    - For names starting with 'BGC', returns the name as is.
    - For names starting with 'GCA_', 'GCF_', etc., extracts the first two parts separated by '_'.
    """
    if pd.isna(bgc_name):
        return None
    bgc_name = bgc_name.strip()
    if bgc_name.startswith('BGC'):
        return bgc_name.split('_')[0]  # e.g., 'BGC0000001'
    elif bgc_name.startswith(('GCA_', 'GCF_', 'GCG_', 'GCI_')):
        parts = bgc_name.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:2])  # e.g., 'GCF_001570505.1'
        else:
            return bgc_name
    else:
        return bgc_name  # Return as is for unknown patterns

def extract_taxonomy(lineage):
    """
    Extracts taxonomy information starting from 'domain' onwards, i.e., after 'cellular organisms;'.
    If 'cellular organisms;' is not found, returns the entire lineage.
    """
    if pd.isna(lineage):
        return ''
    # Split by ';' and strip whitespace
    parts = [part.strip() for part in lineage.split(';')]
    try:
        # Find the index of 'cellular organisms'
        index = parts.index('cellular organisms') + 1
        # Return the taxonomy starting from the next element
        taxonomy = ','.join(parts[index:])  # Use ',' to match the existing separator
        return taxonomy
    except ValueError:
        # 'cellular organisms' not found; return entire lineage with ','
        return ','.join(parts)

def extract_order(lineage):
    """
    Extracts the term ending with 'ales' from the Lineage.
    Returns the first matching term or 'Unclassified' if not found.
    """
    if pd.isna(lineage):
        return 'Unclassified'
    # Split the lineage by ';' and strip whitespace
    parts = [part.strip() for part in lineage.split(';')]
    # Find the first part that ends with 'ales'
    for part in parts:
        if part.lower().endswith('ales'):
            return part
    return 'Unclassified'

def detect_separator(taxonomy_str):
    """
    Detecta o separador utilizado na string de taxonomia.
    Retorna ';' se presente, caso contrÃ¡rio, retorna ','.
    """
    if ';' in taxonomy_str:
        return ';'
    elif ',' in taxonomy_str:
        return ','
    else:
        return ','  # PadrÃ£o se nenhum for encontrado

def extract_genus(taxonomy_list):
    """
    Extrai o Genus da lista taxonÃ´mica.
    Ignora termos como 'group', 'complex', 'species', etc.
    """
    ignore_keywords = ['group', 'complex', 'species', 'subsp.', 'strain']
    for tax in reversed(taxonomy_list):
        if not any(keyword in tax.lower() for keyword in ignore_keywords):
            return tax
    return taxonomy_list[-1] if taxonomy_list else None

def extract_taxonomy_columns(taxonomy_str):
    """
    Extrai Phylum, Order, Family e Genus de uma string de taxonomia.
    
    Args:
        taxonomy_str (str): String com nÃ­veis taxonÃ´micos separados por vÃ­rgula ou ponto e vÃ­rgula.
    
    Returns:
        pd.Series: SÃ©rie com os campos 'Phylum', 'Order', 'Family' e 'Genus'.
    """
    if not isinstance(taxonomy_str, str):
        return pd.Series({'Phylum': None, 'Order': None, 'Family': None, 'Genus': None})
    
    # Detectar o separador
    separator = detect_separator(taxonomy_str)
    
    # Dividir a string de taxonomia
    taxonomy_list = [tax.strip() for tax in taxonomy_str.split(separator)]
    
    # Inicializar as variÃ¡veis
    phylum = None
    order = None
    family = None
    genus = None
    
    # ExtraÃ§Ã£o de Phylum
    phylum_found = False
    for idx, tax in enumerate(taxonomy_list):
        if 'group' in tax.lower():
            if idx + 1 < len(taxonomy_list):
                phylum = taxonomy_list[idx + 1]
                phylum_found = True
                logging.debug(f"Encontrado 'group' em '{tax}'. Phylum extraÃ­do: '{phylum}'")
                break
    if not phylum_found:
        if len(taxonomy_list) >= 2:
            phylum = taxonomy_list[1]
            logging.debug(f"'group' nÃ£o encontrado. Phylum extraÃ­do do segundo nÃ­vel: '{phylum}'")
    
    # ExtraÃ§Ã£o de Order
    for tax in taxonomy_list:
        if tax.lower().endswith('ales'):
            order = tax
            logging.debug(f"Order encontrado: '{order}'")
            break
    
    # ExtraÃ§Ã£o de Family
    for tax in taxonomy_list:
        if tax.lower().endswith('aceae'):
            family = tax
            logging.debug(f"Family encontrado: '{family}'")
            break
    
    # ExtraÃ§Ã£o de Genus
    genus = extract_genus(taxonomy_list)
    if genus:
        logging.debug(f"Genus extraÃ­do: '{genus}'")
    
    return pd.Series({'Phylum': phylum, 'Order': order, 'Family': family, 'Genus': genus})

def update_missing_info(main_table_path, secondary_table_path, output_path):
    # Configure logging
    logging.basicConfig(
        filename='update_table.log',
        filemode='w',
        level=logging.DEBUG,  # Alterado para DEBUG para capturar mais detalhes
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting the update process.")
    
    # Read the main and supplementary tables
    try:
        main_df = pd.read_csv(
            main_table_path,
            sep='\t',
            dtype=str,
            on_bad_lines='skip',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            engine='python'
        )
        secondary_df = pd.read_csv(
            secondary_table_path,
            sep='\t',
            dtype=str,
            on_bad_lines='skip',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            engine='python'
        )
        logging.info("Files successfully read.")
    except Exception as e:
        logging.error(f"Error reading the files: {e}")
        print(f"Erro ao ler os arquivos: {e}", file=sys.stderr)
        return
    
    # Ensure required columns exist in the main table
    required_main_columns = ['BGC', 'Organism', 'Taxonomy', 'Order']
    for col in required_main_columns:
        if col not in main_df.columns:
            logging.error(f"Coluna '{col}' ausente na tabela principal.")
            raise ValueError(f"Coluna '{col}' ausente na tabela principal.")
    
    # Ensure required columns exist in the supplementary table
    required_secondary_columns = ['Assembly Accession', 'Organism Name', 'Lineage']
    for col in required_secondary_columns:
        if col not in secondary_df.columns:
            logging.error(f"Coluna '{col}' ausente na tabela suplementar.")
            raise ValueError(f"Coluna '{col}' ausente na tabela suplementar.")
    
    # Create 'Assembly_key' in main_df by extracting from 'BGC'
    main_df['Assembly_key'] = main_df['BGC'].apply(extract_assembly_key)
    logging.info("Assembly_key extracted from 'BGC' column.")
    
    # Identify rows with missing 'Organism', 'Taxonomy', or 'Order'
    # Consider missing if ANY of the columns is empty, NaN, or 'Unclassified'
    missing_mask = (
        main_df[['Organism', 'Taxonomy', 'Order']]
        .apply(lambda x: x.isnull() | x.str.strip().eq('') | x.str.strip().eq('Unclassified'), axis=1)
        .any(axis=1)
    )
    
    missing_df = main_df[missing_mask].copy()
    
    if missing_df.empty:
        print("No missing entries found. No updates necessary.")
        logging.info("No missing entries found. No updates necessary.")
        # Proceed to extract taxonomy columns for all entries
    else:
        logging.info(f"Found {len(missing_df)} missing entries to update.")
        
        # Create a mapping from 'Assembly Accession' to 'Organism Name' and 'Lineage'
        mapping_org = secondary_df.set_index('Assembly Accession')['Organism Name'].to_dict()
        mapping_lineage = secondary_df.set_index('Assembly Accession')['Lineage'].to_dict()
        
        # Apply mappings to fill missing 'Organism' and 'Taxonomy'
        main_df.loc[missing_mask, 'Organism'] = main_df.loc[missing_mask, 'Assembly_key'].map(mapping_org).combine_first(main_df.loc[missing_mask, 'Organism'])
        main_df.loc[missing_mask, 'Taxonomy'] = main_df.loc[missing_mask, 'Assembly_key'].map(mapping_lineage).apply(extract_taxonomy).combine_first(main_df.loc[missing_mask, 'Taxonomy'])
        
        # Extract 'Order' from 'Lineage'
        main_df.loc[missing_mask, 'Order'] = main_df.loc[missing_mask, 'Assembly_key'].map(mapping_lineage).apply(extract_order).combine_first(main_df.loc[missing_mask, 'Order'])
        
        # Handle cases where 'Order' is still missing after extraction
        order_missing_mask = main_df['Order'].isnull() | main_df['Order'].eq('')
        main_df.loc[order_missing_mask, 'Order'] = 'Unclassified'
        logging.info("Handled missing 'Order' entries by setting to 'Unclassified'.")
        
        # Identify and log unmatched keys (where 'Organism Name' was not found)
        unmatched_keys = main_df.loc[missing_mask & main_df['Organism'].isnull(), 'Assembly_key'].unique()
        if len(unmatched_keys) > 0:
            logging.warning("Unmatched Assembly Keys found:")
            print("\nUnmatched Assembly Keys:")
            with open("unmatched_keys.txt", "w") as f:
                for key in unmatched_keys:
                    logging.warning(f"- {key}")
                    f.write(f"{key}\n")
                    print(f"- {key}")
            print("\nUnmatched keys have been saved to 'unmatched_keys.txt' for further review.")
    
    # Proceed to extract Phylum, Order, Family, and Genus for all entries
    logging.info("Starting taxonomy extraction for all entries.")
    print("Starting taxonomy extraction for all entries...", file=sys.stderr)
    taxonomy_extracted = main_df['Taxonomy'].apply(extract_taxonomy_columns)
    
    # Add the new taxonomy columns to the main DataFrame
    main_df['Phylum'] = taxonomy_extracted['Phylum']
    main_df['Order_extracted'] = taxonomy_extracted['Order']  # Temporarily store extracted Order
    main_df['Family'] = taxonomy_extracted['Family']
    main_df['Genus'] = taxonomy_extracted['Genus']
    
    # Decide how to handle 'Order': prefer existing 'Order' if present, else use 'Order_extracted'
    # Update 'Order' only if it was previously missing or 'Unclassified'
    main_df['Order'] = main_df.apply(
        lambda row: row['Order_extracted'] if pd.isna(row['Order']) or row['Order'] == 'Unclassified' else row['Order'],
        axis=1
    )
    
    # Drop the temporary 'Order_extracted' column
    main_df.drop(columns=['Order_extracted'], inplace=True)
    
    # Optional: Reorganize columns for better readability
    try:
        taxonomy_cols = ['Phylum', 'Order', 'Family', 'Genus']
        if 'Taxonomy' in main_df.columns:
            taxonomy_idx = main_df.columns.get_loc('Taxonomy')
            before = main_df.columns[:taxonomy_idx + 1].tolist()
            after = main_df.columns[taxonomy_idx + 1:].tolist()
            # Remove taxonomy_cols from 'after' to avoid duplication
            after = [col for col in after if col not in taxonomy_cols]
            new_order = before + taxonomy_cols + after
            main_df = main_df[new_order]
            logging.info("Reorganized columns to place taxonomy fields after 'Taxonomy'.")
        else:
            logging.warning("Column 'Taxonomy' not found. Skipping column reorganization.")
    except Exception as e:
        logging.error(f"Erro ao reorganizar as colunas: {e}")
        print(f"Erro ao reorganizar as colunas: {e}", file=sys.stderr)
        # Continue without reorganizing
    
    # Drop the temporary 'Assembly_key' column
    if 'Assembly_key' in main_df.columns:
        main_df.drop(columns=['Assembly_key'], inplace=True)
        logging.info("Dropped temporary 'Assembly_key' column.")
    
    # Save the updated main table
    try:
        main_df.to_csv(output_path, sep='\t', index=False, na_rep='')
        logging.info(f"Updated table has been saved to: {output_path}")
        print(f"\nUpdated table has been saved to: {output_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo de saÃ­da: {e}")
        print(f"Erro ao salvar o arquivo de saÃ­da: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update missing Organism, Taxonomy, and Order information in the main table using a supplementary table, and extract Phylum, Order, Family, and Genus from Taxonomy.")
    parser.add_argument("main_table", help="Path to the main table (TSV format)")
    parser.add_argument("secondary_table", help="Path to the supplementary table (TSV format)")
    parser.add_argument("output_table", help="Path to save the updated main table (TSV format)")
    
    args = parser.parse_args()
    
    update_missing_info(args.main_table, args.secondary_table, args.output_table)
