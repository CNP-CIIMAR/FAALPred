import pandas as pd
import argparse
import re
import csv
import logging

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
        taxonomy = '; '.join(parts[index:])
        return taxonomy
    except ValueError:
        # 'cellular organisms' not found; return entire lineage
        return lineage

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
        if part.endswith('ales'):
            return part
    return 'Unclassified'

def update_missing_info(main_table_path, secondary_table_path, output_path):
    # Configure logging
    logging.basicConfig(
        filename='update_table.log',
        filemode='w',
        level=logging.INFO,
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
        print(f"Erro ao ler os arquivos: {e}")
        return

    # Ensure 'Order' column exists in the main table; if not, create it
    if 'Order' not in main_df.columns:
        main_df['Order'] = ''
        logging.info("Column 'Order' not found in main table. Created 'Order' column.")
    
    # Ensure required columns exist in the main table
    required_main_columns = ['BGC', 'Organism', 'Taxonomy', 'Order']
    for col in required_main_columns:
        if col not in main_df.columns:
            raise ValueError(f"Coluna '{col}' ausente na tabela principal.")
    
    # Ensure required columns exist in the supplementary table
    required_secondary_columns = ['Assembly Accession', 'Organism Name', 'Lineage']
    for col in required_secondary_columns:
        if col not in secondary_df.columns:
            raise ValueError(f"Coluna '{col}' ausente na tabela suplementar.")
    
    # Create 'Assembly_key' in main_df by extracting from 'BGC'
    main_df['Assembly_key'] = main_df['BGC'].apply(extract_assembly_key)
    
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
        main_df.drop(columns=['Assembly_key'], inplace=True)
        main_df.to_csv(output_path, sep='\t', index=False)
        return
    
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
    
    # Drop the temporary 'Assembly_key' column
    main_df.drop(columns=['Assembly_key'], inplace=True)
    
    # Save the updated main table
    main_df.to_csv(output_path, sep='\t', index=False)
    logging.info(f"Updated table has been saved to: {output_path}")
    print(f"\nUpdated table has been saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update missing Organism, Taxonomy, and Order information in the main table using a supplementary table.")
    parser.add_argument("main_table", help="Path to the main table (TSV format)")
    parser.add_argument("secondary_table", help="Path to the supplementary table (TSV format)")
    parser.add_argument("output_table", help="Path to save the updated main table (TSV format)")
    
    args = parser.parse_args()
    
    update_missing_info(args.main_table, args.secondary_table, args.output_table)

