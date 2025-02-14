
# Script 1: title: "ProteinHMMSearch Tool": protein_hmm_search.py

# description: 
  A Python-based tool designed to execute HMMER's `hmmsearch` on multiple FASTA (.faa) files using a set of HMM (.hmm) models.
  The tool automates the search process, parses the results, and consolidates them into organized TSV files for further analysis.

# features:
  - Executes `hmmsearch` for multiple HMM models against multiple FASTA files.
  - Parses HMMER output (`tblout`) and extracts relevant information into structured TSV files.
  - Supports specifying custom E-value thresholds for domain searches.
  - Automatically organizes output files based on models and genome accessions.
  - Handles errors gracefully and provides informative messages for troubleshooting.

# dependencies:
  - Anaconda 3.x
  - Python 3.8 or higher
  - pandas
  - HMMER 3.3 or higher

# installation:
  prerequisites:
    - Download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.
    - Ensure that the HMMER suite is installed. You can install it via conda:
      ```bash
      conda install -c bioconda hmmer
      ```

  # steps:
    - Clone the repository:
      ```bash
      git clone https://github.com/yourusername/proteinHMM.git
      cd proteinHMM
      ```

    - Create the Anaconda environment named `proteinHMM`:
      ```bash
      conda create -n proteinHMM python=3.8
      ```

    - Activate the environment:
      ```bash
      conda activate proteinHMM
      ```

    - Install the required Python packages:
      ```bash
      pip install pandas
      ```

    - (Optional) If additional dependencies are required, install them as needed:
      ```bash
      pip install [package_name]
      ```

usage:
  description: 
    The script processes multiple FASTA files containing protein sequences (.faa) against a set of HMM models (.hmm).
    It performs domain searches using `hmmsearch`, parses the results, and compiles them into consolidated TSV files.

  command:
    ```bash
    python3 protein_hmm_search.py <models_dir> <fastas_dir> <output_dir>
    ```

  arguments:
    - `<models_dir>`: 
        description: "Path to the directory containing HMM model files (.hmm)."
        example: "models/"
    - `<fastas_dir>`: 
        description: "Path to the directory containing FASTA files (.faa) to be searched."
        example: "fastas/"
    - `<output_dir>`: 
        description: "Path to the directory where output results will be saved."
        example: "results/"

  example:
    ```bash
    python3 protein_hmm_search.py models/ fastas/ results/
    ```
  arguments_details:
    models_dir:
      - Ensure that the directory contains valid HMM model files with a `.hmm` extension.
      - Example files: `model1.hmm`, `model2.hmm`, etc.
    fastas_dir:
      - Ensure that the directory contains valid FASTA files with a `.faa` extension.
      - Example files: `genome1.faa`, `genome2.faa`, etc.
    output_dir:
      - If the directory does not exist, it will be created automatically.
      - Results will be saved as `modelName_genomeAccession_tblout.txt` and consolidated into `modelName_all_results.tsv`.

model_hmm_query:
  description: >
    The HMM models used as queries should represent the protein families or domains of interest.
    These models can be custom-built or sourced from established databases.

  creating_hmm_profiles:
    steps:
      - Collect a multiple sequence alignment (MSA) for the protein family of interest.
      - Use HMMER's `hmmbuild` to create an HMM profile from the MSA.
        ```bash
        hmmbuild modelName.hmm alignment.sto
        ```
      - Validate the created HMM model to ensure its accuracy.

#   download_hmmer_profile_databases:
    description: >
      Pre-built HMM profiles can be downloaded from various sources for commonly studied protein families.

    sources:
      - NCBI CDD (Conserved Domain Database):
          url: "https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml"
          description: "Provides a comprehensive collection of well-annotated multiple sequence alignment models for ancient domains and full-length proteins."
      - Pfam:
          url: "https://pfam.xfam.org/"
          description: "A large collection of protein families, each represented by multiple sequence alignments and profile hidden Markov models."
      - TIGRFAMs:
          url: "http://www.jcvi.org/cgi-bin/tigrfams/main"
          description: "Protein families developed by the Joint Genome Institute."

    example_download_command:
      ```bash
      wget -O pfam-A.hmm.gz ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
      gunzip pfam-A.hmm.gz
      ```
# environment_setup:
  description: 
    Instructions to create and activate the Anaconda environment named `proteinHMM`, and install all necessary dependencies.

  steps:
    - Open your terminal or command prompt.
    - Create the Anaconda environment:
      ```bash
      conda create -n proteinHMM python=3.8
      ```
    - Activate the environment:
      ```bash
      conda activate proteinHMM
      ```
    - Install HMMER via conda:
      ```bash
      conda install -c bioconda hmmer
      ```
    - Install required Python packages:
      ```bash
      pip install pandas
      ```
    - Verify installations:
      ```bash
      python --version
      hmmsearch -h
      ```

output:
  description: >
    The script generates output files containing the results of the HMM searches. These include detailed search results for each model and a consolidated summary.

  files_generated:
    - `<modelName>_<genomeAccession>_tblout.txt`:
        description: "Detailed HMMER `tblout` results for each model and genome accession, including all hits with E-values below the specified threshold."
    - `<modelName>_all_results.tsv`:
        description: "Consolidated TSV file aggregating all hits across different genome accessions for a specific HMM model."

# contributing:
  guidelines: >
    Contributions are welcome! Please follow these steps to contribute:
      1. Fork the repository.
      2. Create a new branch for your feature or bugfix.
      3. Commit your changes with clear and descriptive messages.
      4. Push to your forked repository.
      5. Open a pull request detailing your changes and the rationale behind them.

    Ensure that your code follows the existing style and includes appropriate documentation and tests.


# troubleshooting:
  issue: "HMMER commands not found"
  solution: >
    Ensure that HMMER is installed and properly added to your system's PATH. If installed via conda, activate the `proteinHMM` environment:
      ```bash
      conda activate proteinHMM
      ```
    Verify installation by running:
      ```bash
      hmmsearch -h
      ```

  issue: "No data found for the given parameters"
  solution: >
    - Verify that the input directories (`models_dir` and `fastas_dir`) contain the correct file types (.hmm and .faa respectively).
    - Check that the FASTA files are properly formatted.
    - Ensure that the HMM models are compatible with the FASTA sequences.
    - Adjust the E-value threshold if necessary.

# Script 2: title: "NCBI Assembly Metadata Enrichment Tool":  get_genome_metadata.py

# description: 
  A Python script designed to process NCBI assembly IDs, retrieve comprehensive taxonomic lineages,
  and fetch additional metadata from the NCBI BioSample database. The tool enriches assembly data with
  geographic and biome distribution information, outputting both comprehensive and filtered datasets
  for further analysis.

# features:
  - Retrieves taxonomic lineage information using NCBITaxa from the ete3 library.
  - Fetches additional metadata from the NCBI BioSample database using Biopython's Entrez module.
  - Parses XML responses to extract geographic locations, biome distributions, and latitude/longitude coordinates.
  - Categorizes biome descriptions based on GOLD standards.
  - Generates both comprehensive and filtered TSV output files containing enriched assembly data.
  - Provides summary statistics on the number of assemblies with specific metadata fields populated.

# dependencies:
  - Anaconda 3.x
  - Python 3.8 or higher
  - pandas
  - ete3
  - biopython

# installation:
  prerequisites:
    - Download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.
    - Ensure you have access to the internet for downloading NCBI databases and fetching metadata.

  steps:
    - Clone the repository:
      ```bash
      git clone https://github.com/yourusername/yourrepository.git
      cd yourrepository
      ```
    
    - Create the Anaconda environment named `ncbi_metadata`:
      ```bash
      conda create -n ncbi_metadata python=3.8
      ```
    
    - Activate the environment:
      ```bash
      conda activate ncbi_metadata
      ```
    
    - Install the required Python packages:
      ```bash
      pip install pandas ete3 biopython
      ```
    
    - Initialize the NCBI taxonomy database for ete3:
      ```bash
      python -m ete3 ncbiupdate
      ```
# usage:
  description: >
    The script processes a tab-separated input file containing NCBI assembly IDs, retrieves taxonomic lineages,
    fetches additional metadata from the BioSample database, and outputs enriched data files. An additional
    filtered output file includes only assemblies with specific biome distributions and geographic coordinates.

  command:
    ```bash
    python3 script.py <input_file_with_assembly_ids> <output_file>
    ```

  arguments:
    - `<input_file_with_assembly_ids>`:
        description: "Path to the input TSV file containing assembly IDs. The file should have a header and assembly IDs in the first column."
        example: "data/assembly_ids.tsv"
    - `<output_file>`:
        description: "Path to the output TSV file where enriched assembly data will be saved."
        example: "results/enriched_assemblies.tsv"

  example:
    ```bash
    python3 script.py data/assembly_ids.tsv results/enriched_assemblies.tsv
    ```

  arguments_details:
    input_file_with_assembly_ids:
      - Ensure the input file is a TSV with a header row.
      - Assembly IDs should be listed in the first column.
      - Example:
        ```
        Assembly_ID	Other_Column
        GCF_000001405.39	...
        GCF_000002035.6	...
        ```
    output_file:
      - The script will generate two output files:
        - `<output_file>`: Comprehensive enriched data.
        - `filtered_<output_file>`: Filtered data with specific biome and geographic information.
      - If the specified output directory does not exist, it will be created automatically.

environment_setup:
  description: >
    Instructions to create and activate the Anaconda environment named `ncbi_metadata`, and install all necessary dependencies.

  steps:
    - Open your terminal or command prompt.
    - Create the Anaconda environment:
      ```bash
      conda create -n ncbi_metadata python=3.8
      ```
    - Activate the environment:
      ```bash
      conda activate ncbi_metadata
      ```
    - Install required Python packages:
      ```bash
      pip install pandas ete3 biopython
      ```
    - Initialize ete3's NCBI taxonomy database:
      ```bash
      python -m ete3 ncbiupdate
      ```
    - Verify installations:
      ```bash
      python --version
      ```
      ```bash
      hmmsearch -h  # If HMMER is also used
      ```
script_details:
  description: >
    The script performs the following operations:
      1. Reads assembly IDs from the input TSV file.
      2. For each assembly ID:
          - Executes external commands to retrieve assembly summary information.
          - Parses the output to extract relevant fields.
          - Retrieves the taxonomic lineage using NCBITaxa.
          - Fetches additional metadata from the BioSample database, including geographic location and biome distribution.
      3. Enriches the assembly data with the retrieved metadata.
      4. Writes the enriched data to the specified output file.
      5. Generates a filtered output file containing only assemblies with known biome distributions and geographic coordinates.
      6. Prints summary statistics on the number of assemblies processed and metadata retrieved.

# troubleshooting:
  issue: "Script exits with usage error."
  solution: >
    Ensure you are providing the required arguments when running the script. The correct usage is:
    ```bash
    python3 script.py <input_file_with_assembly_ids> <output_file>
    ```
    Example:
    ```bash
    python3 script.py data/assembly_ids.tsv results/enriched_assemblies.tsv
    ```

  issue: "Error retrieving lineage information."
  solution: >
    - Verify that the Tax ID is valid and exists in the NCBI taxonomy database.
    - Ensure that the ete3 taxonomy database is properly initialized and updated.
    - Check your internet connection, as the script requires access to NCBI servers.

  issue: "Entrez fetch fails or returns no data."
  solution: >
    - Ensure that the assembly accession IDs are correct and exist in the NCBI Assembly database.
    - Verify that your email is correctly set in the script.
    - Respect NCBI's rate limits to avoid being temporarily blocked.

  issue: "Latitude and Longitude not parsed correctly."
  solution: >
    - Check the format of the `lat_lon` attribute in the BioSample data.
    - Ensure that the script's `parse_latitude_longitude` function matches the format of the latitude and longitude data.
    - Modify the parsing logic if the data format differs.

# support:
  description: >
    If you encounter any issues or have questions, feel free to reach out via the contact information provided above or open an issue in the repository. Contributions and feedback are highly appreciated!

example_input:
  description: >
    An example of the input TSV file (`assembly_ids.tsv`) structure:
    ```
    Assembly_ID	Other_Info
    GCF_000001405.39	...
    GCF_000002035.6	...
    ```

# output:
  description: >
    The script generates two output files:
      - `<output_file>`: Contains enriched assembly data with taxonomic lineages and additional metadata.
      - `filtered_<output_file>`: Contains only assemblies with known biome distributions and geographic coordinates.

  files_generated:
    - `<output_file>`:
        description: "Comprehensive TSV file with enriched assembly data, including taxonomic lineage, location, biome distribution, latitude, and longitude."
    - `filtered_<output_file>`:
        description: "Filtered TSV file containing assemblies with specific biome and geographic information for targeted analysis."

additional_resources:
  - name: "NCBI Taxonomy Database"
    url: "https://www.ncbi.nlm.nih.gov/taxonomy"
    description: "Provides authoritative taxonomic information for all organisms recognized by the National Center for Biotechnology Information."
  
  - name: "NCBI BioSample Database"
    url: "https://www.ncbi.nlm.nih.gov/biosample/"
    description: "Stores descriptive metadata about biological samples used in various NCBI databases."

  - name: "ETE Toolkit Documentation"
    url: "http://etetoolkit.org/docs/latest/index.html"
    description: "Comprehensive documentation for the ete3 library, including usage of NCBITaxa."

  - name: "Biopython Documentation"
    url: "https://biopython.org/wiki/Documentation"
    description: "Documentation for Biopython, a set of tools for biological computation in Python."
    
# Script 3: title: "FASTA Sequence Filter Tool" : fasta_sequence_filter.py

# description: 
  A Python script designed to filter sequences in FASTA files based on a list of sequence IDs.
  Whether you need to keep or exclude specific sequences, this tool provides a straightforward
  command-line interface to efficiently process large FASTA datasets for bioinformatics analyses.

# features:
  - Filters FASTA files to keep or exclude sequences based on a provided list of IDs.
  - Utilizes Biopython's SeqIO for efficient parsing and writing of FASTA files.
  - Supports mutually exclusive options to either keep or exclude specified sequences.
  - Provides informative output on the number of sequences processed and filtered.
  - Easy integration into bioinformatics pipelines for preprocessing sequence data.

# dependencies:
  - Anaconda 3.x
  - Python 3.8 or higher
  - Biopython

# installation:
  prerequisites:
    - Download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.
  
  steps:
    - Clone the repository:
      ```bash
      git clone https://github.com/yourusername/yourrepository.git
      cd yourrepository
      ```
    
    - Create the Anaconda environment named `filter_sequences`:
      ```bash
      conda create -n filter_sequences python=3.8
      ```
    
    - Activate the environment:
      ```bash
      conda activate filter_sequences
      ```
    
    - Install the required Python packages:
      ```bash
      pip install biopython
      ```
    
    - (Optional) If additional dependencies are needed, install them as required:
      ```bash
      pip install [package_name]
      ```
# usage:
  description: >
    The script filters sequences in an input FASTA file based on a list of sequence IDs provided.
    You can choose to either keep only the specified sequences or exclude them from the output.

  command:
    ```bash
    python3 filter_fasta.py <input_fasta> <output_fasta> <ids_file> (--keep | --exclude)
    ```

  arguments:
    - `<input_fasta>`:
        description: "Path to the input FASTA file containing sequences to be filtered."
        example: "data/input_sequences.fasta"
    - `<output_fasta>`:
        description: "Path to the output FASTA file where filtered sequences will be saved."
        example: "results/filtered_sequences.fasta"
    - `<ids_file>`:
        description: "Path to the file containing sequence IDs to keep or exclude. Each ID should be on a separate line."
        example: "data/sequence_ids.txt"
    - `--keep`:
        description: "Keep only the sequences with IDs specified in the IDs file."
    - `--exclude`:
        description: "Exclude the sequences with IDs specified in the IDs file."

#  example:
    ```bash
    python3 filter_fasta.py data/input_sequences.fasta results/kept_sequences.fasta data/sequence_ids.txt --keep
    ```
    ```bash
    python3 filter_fasta.py data/input_sequences.fasta results/excluded_sequences.fasta data/sequence_ids.txt --exclude
    ```
# arguments_details:
  input_fasta:
    - Ensure the input file is in proper FASTA format.
    - Supports large FASTA files efficiently using Biopython's SeqIO.
  output_fasta:
    - The script will create this file if it does not exist.
    - If the file exists, it will be overwritten with the filtered sequences.
  ids_file:
    - Should be a plain text file with one sequence ID per line.
    - No headers or additional formatting required.
  --keep:
    - When this option is used, only sequences with IDs present in the `ids_file` will be written to the `output_fasta`.
  --exclude:
    - When this option is used, sequences with IDs present in the `ids_file` will be excluded from the `output_fasta`.

# environment_setup:
  description: >
    Instructions to create and activate the Anaconda environment named `filter_sequences`, and install all necessary dependencies.

  steps:
    - Open your terminal or command prompt.
    - Create the Anaconda environment:
      ```bash
      conda create -n filter_sequences python=3.8
      ```
    - Activate the environment:
      ```bash
      conda activate filter_sequences
      ```
    - Install required Python packages:
      ```bash
      pip install biopython
      ```
    - Verify installations:
      ```bash
      python --version
      ```
      ```bash
      python -c "import Bio; print(Bio.__version__)"
      ```
# script_details:
  description: >
    The script performs the following operations:
      1. Parses command-line arguments to determine input and output files, IDs file, and filter mode.
      2. Reads the list of sequence IDs from the provided `ids_file`.
      3. Iterates through the input FASTA file, filtering sequences based on the specified mode:
          - **Keep Mode (`--keep`)**: Retains only sequences whose IDs are in the `ids_file`.
          - **Exclude Mode (`--exclude`)**: Removes sequences whose IDs are in the `ids_file`.
      4. Writes the filtered sequences to the `output_fasta` file.
      5. Outputs summary information on the number of sequences processed and filtered.

# faq:
  - question: "What is the purpose of this script?"
    answer: >
      The script is designed to filter sequences in FASTA files based on a list of sequence IDs.
      It allows users to either keep only the specified sequences or exclude them from the output,
      facilitating targeted analyses in bioinformatics workflows.

  - question: "How do I prepare the IDs file?"
    answer: >
      The IDs file should be a plain text file with one sequence ID per line. Ensure there are no
      headers or additional formatting. For example:
      ```
      seq1
      seq2
      seq3
      ```

  - question: "Can I use this script with large FASTA files?"
    answer: >
      Yes, the script utilizes Biopython's SeqIO for efficient parsing, allowing it to handle large
      FASTA files effectively. However, ensure that your system has sufficient memory and storage
      resources for processing large datasets.

  - question: "What happens if a sequence ID in the IDs file is not found in the FASTA file?"
    answer: >
      The script will process all sequences in the FASTA file and apply the filtering criteria.
      Sequence IDs in the IDs file that do not match any sequences in the FASTA file will have no
      effect on the output.

  - question: "Can I use this script to filter sequences based on partial IDs or patterns?"
    answer: >
      Currently, the script filters sequences based on exact matches of sequence IDs. For partial
      matches or pattern-based filtering, you would need to modify the script to incorporate regular
      expressions or other matching criteria.

# troubleshooting:
  issue: "Script exits with usage error."
  solution: >
    Ensure you are providing the required arguments when running the script. The correct usage is:
    ```bash
    python3 filter_fasta.py <input_fasta> <output_fasta> <ids_file> (--keep | --exclude)
    ```
    Example:
    ```bash
    python3 filter_fasta.py data/input_sequences.fasta results/filtered_sequences.fasta data/sequence_ids.txt --keep
    ```

  issue: "No sequences are being written to the output file."
  solution: >
    - Verify that the IDs in the `ids_file` match the sequence IDs in the `input_fasta`.
    - Ensure that you are using the correct mode (`--keep` or `--exclude`) based on your filtering needs.
    - Check for any leading/trailing whitespaces in the `ids_file` that might prevent matching.

  issue: "Biopython is not installed or not found."
  solution: >
    Ensure that you have activated the correct Anaconda environment (`filter_sequences`) and installed Biopython:
    ```bash
    conda activate filter_sequences
    pip install biopython
    ```
    Verify installation by running:
    ```bash
    python -c "import Bio; print(Bio.__version__)"
    ```

  issue: "Permission denied when creating or writing to the output file."
  solution: >
    Ensure that you have the necessary write permissions for the directory where you are trying to save the `output_fasta`.
    You can change the directory permissions or choose a different directory with appropriate permissions.

support:
  description: >
    If you encounter any issues or have questions, feel free to reach out via the contact information provided above or open an issue in the repository. Contributions and feedback are highly appreciated!

example_input:
  description: >
    An example of the input IDs file (`sequence_ids.txt`) structure:
    ```
    seq1
    seq2
    seq3
    ```

output:
  description: >
    The script generates an output FASTA file containing the filtered sequences based on the provided IDs and selected mode (keep or exclude).

  files_generated:
    - `<output_fasta>`:
        description: "FASTA file containing the filtered sequences. Depending on the selected mode, it either includes only the specified sequences or excludes them from the original set."

additional_resources:
  - name: "Biopython Documentation"
    url: "https://biopython.org/wiki/Documentation"
    description: "Comprehensive documentation for Biopython, including the SeqIO module used for parsing and writing FASTA files."
  
  - name: "FASTA Format Specification"
    url: "https://en.wikipedia.org/wiki/FASTA_format"
    description: "Detailed information about the FASTA file format used for representing nucleotide or peptide sequences."

  - name: "Anaconda Documentation"
    url: "https://docs.anaconda.com/anaconda/"
    description: "Official documentation for Anaconda, a distribution of Python and R for scientific computing."

# Script 4: FAALs Taxonomic Analysis: barplot_normalized_counts.py

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

```bash
    git clone https://github.com/yourusername/faals-taxonomic-analysis.git
```


cd faals-taxonomic-analysis

# Create a Virtual Environment (Optional but Recommended)

```bash

python3 -m venv venv
source venv/bin/activate
```

# Install Dependencies

bash```bash
pip install -r requirements.txt
```

If requirements.txt is not provided, install the necessary packages manually:

```bash
pip install pandas matplotlib seaborn ete3 numpy
```
Additionally, you may need to install the NCBI taxonomy database for ete3:

```bash
python -c "from ete3 import NCBITaxa; NCBITaxa().update_taxonomy_database()"
```

# Usage
The script is executed via the command line and requires specific arguments to function correctly.

# Command-Line Arguments

```bash
python3 barplot_normalized_counts_faal.py <table1.tsv> <table2.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>
```
- <table1.tsv>: Path to the first input TSV file containing organism data.
- <table2.tsv>: Path to the second input TSV file containing assembly data.
- <Domain>: Taxonomic domain to filter by (e.g., Eukaryota, Bacteria).
- <Taxonomic Level>: Specific taxonomic level for analysis (e.g., Order, Family, Genus, Phylum, Domain).
- <Top N>: Number of top taxonomic groups to display based on FAAL counts.
<- DPI>: Resolution for the output plots (e.g., 300).

# Example:

```bash
python3 barplot_normalized_counts_faal.py organisms.tsv assemblies.tsv Eukaryota Genus 10 300
```
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

# Output Files
Upon successful execution, the script generates the following output files:

- Taxonomic_groups_with_FAAL.tsv: Filtered data from table2.tsv based on top taxonomic groups with FAALs.
- merged_data.tsv: Aggregated FAAL counts and genome counts per taxonomic group.
- normalized_data.tsv: Normalized FAAL counts per genome as percentages.
- ranking_FAAL_combined.png: Combined bar plots showing raw and normalized FAAL counts.
- ranking_FAAL_combined.svg: Scalable Vector Graphics version of the plots.
- ranking_FAAL_combined.jpeg: JPEG version of the plots.

# Example
Assuming you have organisms.tsv and assemblies.tsv in your working directory and want to analyze the top 10 genera within the Eukaryota domain with a plot resolution of 300 DPI:

```bash
python3 barplot_normalized_counts_faal.py organisms.tsv assemblies.tsv Eukaryota Genus 10 300
```
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

# Scriot 5: "Taxonomic Analysis and Visualization Tool" : scatterplot_mean_faal_genome.py 

  A Python script designed to process genomic assembly data, perform taxonomic classification,
  filter based on specified criteria, and generate insightful visualizations of FAAL counts
  relative to genome sizes across different taxonomic groups.

# features:
  - Extracts and updates taxonomic lineage information using the NCBI taxonomy database.
  - Filters data based on domain, taxonomic levels, and specific naming conventions.
  - Calculates FAAL counts and genome statistics for selected taxonomic groups.
  - Generates scatter plots visualizing the relationship between genome size and FAAL counts.
  - Supports customization of top N taxonomic groups and output resolution.

# dependencies:
  - pandas
  - matplotlib
  - seaborn
  - numpy
  - ete3

# installation:
  steps:
    - Ensure Python 3.6 or higher is installed.
    - Install required Python packages using pip:
      ```bash
      pip install pandas matplotlib seaborn numpy ete3
      ```
    - Download and set up the NCBI taxonomy database for ete3:
      ```bash
      python -m ete3 ncbiupdate
      ```

# usage:
  description: 
    The script processes a TSV file containing genomic assembly data, filters the data based on
    specified domain and taxonomic level, and generates scatter plots showing the average
    FAAL counts per genome against genome sizes for the top N taxonomic groups.

  command:
    ```bash
    python3 scatterplot_counts_faal.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>
    ```

  parameters:
    - `<table1.tsv>`: Path to the input TSV file containing genomic assembly data.
    - `<Domain>`: Taxonomic domain to filter (e.g., Eukaryota, Bacteria).
    - `<Taxonomic Level>`: Taxonomic rank for grouping (e.g., Phylum, Class, Order, Family, Genus, Species).
    - `<Top N>`: Number of top taxonomic groups to visualize based on mean FAAL counts.
    - `<DPI>`: Resolution of the output plot images.

  example:
    ```bash
    python3 scatterplot_counts_faal.py data/assemblies.tsv Eukaryota Genus 10 300
    ```

# output:
  - `taxonomic_analysis_plot.png`: PNG image of the generated scatter plot.
  - `taxonomic_analysis_plot.svg`: SVG vector image of the generated scatter plot.
  - `taxonomic_analysis_plot.jpeg`: JPEG image of the generated scatter plot.

# script_details:
  description: 
    The script performs the following steps:
      1. Reads the input TSV file containing genomic assembly data.
      2. Updates the taxonomic lineage information using ete3 if the domain is Eukaryota.
      3. Filters out entries related to environmental samples and ensures assembly IDs start with 'GCF' or 'GCA'.
      4. Extracts the specified taxonomic group from the lineage and applies filtering criteria.
      5. Aggregates FAAL counts and genome counts per taxonomic group.
      6. Calculates the mean FAAL count per genome and average genome size.
      7. Selects the top N taxonomic groups based on mean FAAL counts.
      8. Generates and saves scatter plots visualizing the data.

# Script 6: name: CAL Domain and GenBank Analyzer
# description: 
  A Python script to analyze subdirectories containing `.gbk` files for the presence of `CAL_domain` and optionally `AMP-binding` domains. The script calculates the total size of `.gbk` files, checks disk space availability, and copies identified files to a filtered directory with the genome ID as a prefix.
usage: |
  python filter_count_CAA_AMP.py <input_dir> <log_file> [--search-amp-binding]
  
  - `input_dir`: Path to the input directory containing subdirectories with `.gbk` files.
  - `log_file`: Path to the log file where processing information will be stored.
  - `--search-amp-binding`: Optional flag to search for `AMP-binding` domains in addition to `CAL_domain`.
# features:
  - Analyzes `.gbk` files in subdirectories for specified domains.
  - Extracts genome IDs from subdirectory names.
  - Calculates the total size of `.gbk` files in megabases and gigabases.
  - Checks for available disk space before copying files.
  - Copies `.gbk` files containing `CAL_domain` to a filtered directory with a genome ID prefix.
  - Generates a summary CSV file with detailed results.
  - Provides a detailed log file for troubleshooting and analysis.
dependencies:
  - Python >= 3.7
  - Biopython
arguments:
  input_dir:
    description: Path to the input directory containing subdirectories with `.gbk` files.
    required: true
  log_file:
    description: Path to the log file where processing information will be stored.
    required: true
  search_amp_binding:
    description: Optional flag to include `AMP-binding` domain in the search.
    required: false
output:
  - Summary CSV file: A report summarizing the analysis for each subdirectory.
  - Log file: A detailed log of the processing steps and outcomes.
  - Filtered files: `.gbk` files containing `CAL_domain`, copied to a filtered directory.
example_usage: |
  # Analyze `.gbk` files for `CAL_domain` only:
  python script.py /path/to/input_dir /path/to/log_file.log
  
  # Analyze `.gbk` files for both `CAL_domain` and `AMP-binding`:
  python filter_count_CAL_AMP.py /path/to/input_dir /path/to/log_file.log --search-amp-binding

# Overview# Script 7: BGC Statistics Bar Chart

This repository contains a Python script that processes a TSV table of biosynthetic gene cluster (BGC) annotations, calculates several statistics, and generates a high-resolution bar chart suitable for publication (e.g., for NAR).

#  Figure S20 with results obtained from BiG-SCAPE for 12,214 bacterial genomes 

The script reads an input file named `Network_Annotations_Full_annotation_mibig_ref_taxo` and performs the following tasks:

- **Data Processing:**
  - Reads the input TSV file using Pandas.
  - Processes columns (notably, `BGC` and `Family Number`).
  - Separates entries into:
    - **MIBIG BGCs:** Rows where the `BGC` column starts with "BGC".
    - **Identified BGCs:** Rows that do not start with "BGC" and have a defined `Family Number`.

- **Statistics Computation:**
  - **New BGCs in MIBIG Families:** Count of identified BGCs whose `Family Number` is among those in the MIBIG set.
  - **New BGCs outside MIBIG Families:** Count of identified BGCs whose `Family Number` is not in the MIBIG set.
  - **Total BGCs Families:** The number of unique `Family Number` values among identified BGCs.
  - **Singleton BGCs Families:** Count of families that appear only once among the identified BGCs.
  - **Identified BGCs:** Total count of identified BGCs.
  
- **Fixed Data:**
  - Two additional fixed bars are added:
    - **MIBIG BGCs:** 333
    - **MIBIG BGCs with FAAL:** 122

- **Chart Generation:**
  - Generates a bar chart that combines the two fixed data bars with the computed statistics (sorted in ascending order).
  - Configures the chart with larger font sizes and appropriate spacing (labels are wrapped in two lines if needed) to meet NAR publication standards.
  - Saves the chart in both PNG and SVG formats with 900 dpi resolution.

## Dependencies

- Python 3.x
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

You can install the Python dependencies using pip:

```bash
pip install pandas numpy matplotlib
    
# contact:
  name: "Leandro de Mattos Pereira"
  email: "lmattos@ciimar.up.pt"
  url: "[https://github.com/yourusername/repository](https://github.com/CNP-CIIMAR/FAALPred)"

# License
  name: "MIT License"
  url: "https://opensource.org/licenses/MIT"

# Acknowledgements
- ETE Toolkit: For providing tools to work with phylogenetic trees and taxonomy.
- Pandas, Matplotlib, Seaborn: For data manipulation and visualization capabilities.
For any questions or issues, please open an issue in the repository.
