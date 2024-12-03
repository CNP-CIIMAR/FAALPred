# FAALs Taxonomic Analysis: barplot_normalized_counts.py

This repository contains a Python script for analyzing Fatty Acyl AMP Ligases (FAALs) across different taxonomic groups. The script processes input data tables, filters and aggregates FAAL counts, normalizes the data, and generates informative visualizations to help understand the distribution and prevalence of FAALs in various taxonomic levels.
 Load and Update Taxonomic Data: Reads two input TSV files containing organism and assembly information, updates taxonomic lineages using the NCBI taxonomy database via the ete3 library.

- Filter Data: Filters organisms based on specified domain and taxonomic level criteria.
- Aggregate Counts: Calculates the total number of FAALs and unique genomes per taxonomic group.
- Normalize Data: Normalizes FAAL counts per genome to account for varying genome sizes across taxonomic groups.
- Generate Visualizations: Creates bar plots to visualize raw and normalized FAAL counts across the top N taxonomic groups.
- Export Results: Saves filtered data and aggregated counts to TSV files and generates plots in multiple formats.

# Features
- Taxonomic Filtering: Allows filtering by Domain (e.g., Eukaryota) and specific taxonomic levels (e.g., Order, Family, Genus).
- Data Normalization: Normalizes FAAL counts to account for the number of genomes in each taxonomic group.
- Visualization: Produces clear and informative bar plots showing both raw and normalized FAAL counts.
- Flexibility: Easily adjustable parameters for different domains, taxonomic levels, and top N groups.
- Installation
Clone the Repository

bash
git clone https://github.com/yourusername/faals-taxonomic-analysis.git

cd faals-taxonomic-analysis

# Create a Virtual Environment (Optional but Recommended)

bash

python3 -m venv venv
source venv/bin/activate
Install Dependencies

bash
pip install -r requirements.txt

If requirements.txt is not provided, install the necessary packages manually:

bash

pip install pandas matplotlib seaborn ete3 numpy
Additionally, you may need to install the NCBI taxonomy database for ete3:


python -c "from ete3 import NCBITaxa; NCBITaxa().update_taxonomy_database()"

# Usage
The script is executed via the command line and requires specific arguments to function correctly.

# Command-Line Arguments

bash

python3 barplot_normalized_counts_faal.py <table1.tsv> <table2.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>

- <table1.tsv>: Path to the first input TSV file containing organism data.
- <table2.tsv>: Path to the second input TSV file containing assembly data.
- <Domain>: Taxonomic domain to filter by (e.g., Eukaryota, Bacteria).
- <Taxonomic Level>: Specific taxonomic level for analysis (e.g., Order, Family, Genus, Phylum, Domain).
- <Top N>: Number of top taxonomic groups to display based on FAAL counts.
<- DPI>: Resolution for the output plots (e.g., 300).

# Example:

bash
python3 barplot_normalized_counts_faal.py organisms.tsv assemblies.tsv Eukaryota Genus 10 300

# Input Files
1. table1.tsv
Contains organism-related data with at least the following columns:

Organism Name: Name of the organism.
Organism Taxonomic ID: NCBI taxonomic ID for the organism.
Assembly: Assembly accession numbers starting with GCF or GCA.

2. table2.tsv
Contains assembly-related data with at least the following columns:

Lineage: Semicolon-separated taxonomic lineage.

Assembly Accession: Assembly accession numbers.

Ensure that both TSV files are properly formatted and contain the necessary columns for the script to function correctly.

Output Files

Upon successful execution, the script generates the following output files:

Taxonomic_groups_with_FAAL.tsv: Filtered data from table2.tsv based on top taxonomic groups with FAALs.
merged_data.tsv: Aggregated FAAL counts and genome counts per taxonomic group.
normalized_data.tsv: Normalized FAAL counts per genome as percentages.
ranking_FAAL_combined.png: Combined bar plots showing raw and normalized FAAL counts.
ranking_FAAL_combined.svg: Scalable Vector Graphics version of the plots.
ranking_FAAL_combined.jpeg: JPEG version of the plots.
# Example
A
ssuming you have organisms.tsv and assemblies.tsv in your working directory and want to analyze the top 10 genera within the Eukaryota domain with a plot resolution of 300 DPI:

bash

python3 barplot_normalized_counts_faal.py organisms.tsv assemblies.tsv Eukaryota Genus 10 300

This command will process the data, perform the analysis, and generate the output files as described above.

# Dependencies
- Python 3.6 or higher
Libraries:
- pandas
- matplotlib
- seaborn
- ete3
- numpy
# Ensure all dependencies are installed before running the script.

# License

This project is licensed under the MIT License.

# Acknowledgements
- ETE Toolkit: For providing tools to work with phylogenetic trees and taxonomy.
- Pandas, Matplotlib, Seaborn: For data manipulation and visualization capabilities.
For any questions or issues, please open an issue in the repository.
