
# Script 1: ProteinHMMSearch Tool: protein_hmm_search.py

# Description: 
A Python-based tool designed to execute HMMER's `hmmsearch` on multiple FASTA (.faa) files using a set of HMM (.hmm) models. The tool automates the search process, parses the results, and consolidates them into organized TSV files for further analysis.

# Features:
  - Executes `hmmsearch` for multiple HMM models against multiple FASTA files.
  - Parses HMMER output (`tblout`) and extracts relevant information into structured TSV files.
  - Supports specifying custom E-value thresholds for domain searches.
  - Automatically organizes output files based on models and genome accessions.
  - Handles errors gracefully and provides informative messages for troubleshooting.

# Dependencies:
  - Anaconda 3.x
  - Python 3.8 or higher
  - pandas
  - HMMER 3.3 or higher

# Installation:
  Prerequisites:
    - Download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.
    - Ensure that the HMMER suite is installed. You can install it via conda:
      ```bash
      conda install -c bioconda hmmer
      ```

  Steps:
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

# Usage:
  Description: 
    The script processes multiple FASTA files containing protein sequences (.faa) against a set of HMM models (.hmm). It performs domain searches using `hmmsearch`, parses the results, and compiles them into consolidated TSV files.

  Command:
    ```bash
    python3 protein_hmm_search.py <models_dir> <fastas_dir> <output_dir>
    ```

  Arguments:
    - `<models_dir>`: 
        Description: "Path to the directory containing HMM model files (.hmm)."
        Example: "models/"
    - `<fastas_dir>`: 
        Description: "Path to the directory containing FASTA files (.faa) to be searched."
        Example: "fastas/"
    - `<output_dir>`: 
        Description: "Path to the directory where output results will be saved."
        Example: "results/"

  Example:
    ```bash
    python3 protein_hmm_search.py models/ fastas/ results/
    ```

  Arguments Details:
    models_dir:
      - Ensure that the directory contains valid HMM model files with a `.hmm` extension.
      - Example files: `model1.hmm`, `model2.hmm`, etc.
    fastas_dir:
      - Ensure that the directory contains valid FASTA files with a `.faa` extension.
      - Example files: `genome1.faa`, `genome2.faa`, etc.
    output_dir:
      - If the directory does not exist, it will be created automatically.
      - Results will be saved as `modelName_genomeAccession_tblout.txt` and consolidated into `modelName_all_results.tsv`.

# Model HMM Query:
  Description: >
    The HMM models used as queries should represent the protein families or domains of interest. These models can be custom-built or sourced from established databases.

  Creating HMM Profiles:
    Steps:
      - Collect a multiple sequence alignment (MSA) for the protein family of interest.
      - Use HMMER's `hmmbuild` to create an HMM profile from the MSA.
        ```bash
        hmmbuild modelName.hmm alignment.sto
        ```
      - Validate the created HMM model to ensure its accuracy.

# Download HMMER Profile Databases:
    Description: >
      Pre-built HMM profiles can be downloaded from various sources for commonly studied protein families.

    Sources:
      - NCBI CDD (Conserved Domain Database):
          URL: "https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd.shtml"
          Description: "Provides a comprehensive collection of well-annotated multiple sequence alignment models for ancient domains and full-length proteins."
      - Pfam:
          URL: "https://pfam.xfam.org/"
          Description: "A large collection of protein families, each represented by multiple sequence alignments and profile hidden Markov models."
      - TIGRFAMs:
          URL: "http://www.jcvi.org/cgi-bin/tigrfams/main"
          Description: "Protein families developed by the Joint Genome Institute."

    Example Download Command:
      ```bash
      wget -O pfam-A.hmm.gz ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
      gunzip pfam-A.hmm.gz
      ```

# Environment Setup:
  Description: 
    Instructions to create and activate the Anaconda environment named `proteinHMM`, and install all necessary dependencies.

  Steps:
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

# Output:
  Description: >
    The script generates output files containing the results of the HMM searches. These include detailed search results for each model and a consolidated summary.

  Files Generated:
    - `<modelName>_<genomeAccession>_tblout.txt`:
        Description: "Detailed HMMER `tblout` results for each model and genome accession, including all hits with E-values below the specified threshold."
    - `<modelName>_all_results.tsv`:
        Description: "Consolidated TSV file aggregating all hits across different genome accessions for a specific HMM model."

# Contributing:
  Guidelines: >
    Contributions are welcome! Please follow these steps to contribute:
      1. Fork the repository.
      2. Create a new branch for your feature or bugfix.
      3. Commit your changes with clear and descriptive messages.
      4. Push to your forked repository.
      5. Open a pull request detailing your changes and the rationale behind them.

    Ensure that your code follows the existing style and includes appropriate documentation and tests.

# Troubleshooting:
  Issue: "HMMER commands not found"
  Solution: >
    Ensure that HMMER is installed and properly added to your system's PATH. If installed via conda, activate the `proteinHMM` environment:
      ```bash
      conda activate proteinHMM
      ```
    Verify installation by running:
      ```bash
      hmmsearch -h
      ```

  Issue: "No data found for the given parameters"
  Solution: >
    - Verify that the input directories (`models_dir` and `fastas_dir`) contain the correct file types (.hmm and .faa respectively).
    - Check that the FASTA files are properly formatted.
    - Ensure that the HMM models are compatible with the FASTA sequences.
    - Adjust the E-value threshold if necessary.

---

# Script 2: NCBI Assembly Metadata Enrichment Tool: get_genome_metadata.py

# Description: 
A Python script designed to process NCBI assembly IDs, retrieve comprehensive taxonomic lineages, and fetch additional metadata from the NCBI BioSample database. The tool enriches assembly data with geographic and biome distribution information, outputting both comprehensive and filtered datasets for further analysis.

# Features:
  - Retrieves taxonomic lineage information using NCBITaxa from the ete3 library.
  - Fetches additional metadata from the NCBI BioSample database using Biopython's Entrez module.
  - Parses XML responses to extract geographic locations, biome distributions, and latitude/longitude coordinates.
  - Categorizes biome descriptions based on GOLD standards.
  - Generates both comprehensive and filtered TSV output files containing enriched assembly data.
  - Provides summary statistics on the number of assemblies with specific metadata fields populated.

# Dependencies:
  - Anaconda 3.x
  - Python 3.8 or higher
  - pandas
  - ete3
  - biopython

# Installation:
  Prerequisites:
    - Download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.
    - Ensure you have access to the internet for downloading NCBI databases and fetching metadata.

  Steps:
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

# Usage:
  Description: >
    The script processes a tab-separated input file containing NCBI assembly IDs, retrieves taxonomic lineages, fetches additional metadata from the BioSample database, and outputs enriched data files. An additional filtered output file includes only assemblies with specific biome distributions and geographic coordinates.

  Command:
    ```bash
    python3 script.py <input_file_with_assembly_ids> <output_file>
    ```

  Arguments:
    - `<input_file_with_assembly_ids>`:
        Description: "Path to the input TSV file containing assembly IDs. The file should have a header and assembly IDs in the first column."
        Example: "data/assembly_ids.tsv"
    - `<output_file>`:
        Description: "Path to the output TSV file where enriched assembly data will be saved."
        Example: "results/enriched_assemblies.tsv"

  Example:
    ```bash
    python3 script.py data/assembly_ids.tsv results/enriched_assemblies.tsv
    ```

  Arguments Details:
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

# Environment Setup:
  Description: >
    Instructions to create and activate the Anaconda environment named `ncbi_metadata`, and install all necessary dependencies.

  Steps:
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

# Script Details:
  Description: >
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

# Troubleshooting:
  Issue: "Script exits with usage error."
  Solution: >
    Ensure you are providing the required arguments when running the script. The correct usage is:
    ```bash
    python3 script.py <input_file_with_assembly_ids> <output_file>
    ```
    Example:
    ```bash
    python3 script.py data/assembly_ids.tsv results/enriched_assemblies.tsv
    ```

  Issue: "Error retrieving lineage information."
  Solution: >
    - Verify that the Tax ID is valid and exists in the NCBI taxonomy database.
    - Ensure that the ete3 taxonomy database is properly initialized and updated.
    - Check your internet connection, as the script requires access to NCBI servers.

  Issue: "Entrez fetch fails or returns no data."
  Solution: >
    - Ensure that the assembly accession IDs are correct and exist in the NCBI Assembly database.
    - Verify that your email is correctly set in the script.
    - Respect NCBI's rate limits to avoid being temporarily blocked.

  Issue: "Latitude and Longitude not parsed correctly."
  Solution: >
    - Check the format of the `lat_lon` attribute in the BioSample data.
    - Ensure that the script's `parse_latitude_longitude` function matches the format of the latitude and longitude data.
    - Modify the parsing logic if the data format differs.

# Support:
  Description: >
    If you encounter any issues or have questions, feel free to reach out via the contact information provided above or open an issue in the repository. Contributions and feedback are highly appreciated!

Example Input:
  Description: >
    An example of the input TSV file (`assembly_ids.tsv`) structure:
    ```
    Assembly_ID	Other_Info
    GCF_000001405.39	...
    GCF_000002035.6	...
    ```

# Output:
  Description: >
    The script generates two output files:
      - `<output_file>`: Contains enriched assembly data with taxonomic lineages and additional metadata.
      - `filtered_<output_file>`: Contains only assemblies with known biome distributions and geographic coordinates.

  Files Generated:
    - `<output_file>`:
        Description: "Comprehensive TSV file with enriched assembly data, including taxonomic lineage, location, biome distribution, latitude, and longitude."
    - `filtered_<output_file>`:
        Description: "Filtered TSV file containing assemblies with specific biome and geographic information for targeted analysis."

Additional Resources:
  - Name: "NCBI Taxonomy Database"
    URL: "https://www.ncbi.nlm.nih.gov/taxonomy"
    Description: "Provides authoritative taxonomic information for all organisms recognized by the National Center for Biotechnology Information."
  
  - Name: "NCBI BioSample Database"
    URL: "https://www.ncbi.nlm.nih.gov/biosample/"
    Description: "Stores descriptive metadata about biological samples used in various NCBI databases."

  - Name: "ETE Toolkit Documentation"
    URL: "http://etetoolkit.org/docs/latest/index.html"
    Description: "Comprehensive documentation for the ete3 library, including usage of NCBITaxa."

  - Name: "Biopython Documentation"
    URL: "https://biopython.org/wiki/Documentation"
    Description: "Documentation for Biopython, a set of tools for biological computation in Python."

---

# Script 3: FASTA Sequence Filter Tool: fasta_sequence_filter.py

# Description: 
A Python script designed to filter sequences in FASTA files based on a list of sequence IDs. Whether you need to keep or exclude specific sequences, this tool provides a straightforward command-line interface to efficiently process large FASTA datasets for bioinformatics analyses.

# Features:
  - Filters FASTA files to keep or exclude sequences based on a provided list of IDs.
  - Utilizes Biopython's SeqIO for efficient parsing and writing of FASTA files.
  - Supports mutually exclusive options to either keep or exclude specified sequences.
  - Provides informative output on the number of sequences processed and filtered.
  - Easy integration into bioinformatics pipelines for preprocessing sequence data.

# Dependencies:
  - Anaconda 3.x
  - Python 3.8 or higher
  - Biopython

# Installation:
  Prerequisites:
    - Download and install [Anaconda](https://www.anaconda.com/products/distribution) for your operating system.
  
  Steps:
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

# Usage:
  Description: >
    The script filters sequences in an input FASTA file based on a list of sequence IDs provided. You can choose to either keep only the specified sequences or exclude them from the output.

  Command:
    ```bash
    python3 filter_fasta.py <input_fasta> <output_fasta> <ids_file> (--keep | --exclude)
    ```

  Arguments:
    - `<input_fasta>`:
        Description: "Path to the input FASTA file containing sequences to be filtered."
        Example: "data/input_sequences.fasta"
    - `<output_fasta>`:
        Description: "Path to the output FASTA file where filtered sequences will be saved."
        Example: "results/filtered_sequences.fasta"
    - `<ids_file>`:
        Description: "Path to the file containing sequence IDs to keep or exclude. Each ID should be on a separate line."
        Example: "data/sequence_ids.txt"
    - `--keep`:
        Description: "Keep only the sequences with IDs specified in the IDs file."
    - `--exclude`:
        Description: "Exclude the sequences with IDs specified in the IDs file."

# Example:
    ```bash
    python3 filter_fasta.py data/input_sequences.fasta results/kept_sequences.fasta data/sequence_ids.txt --keep
    ```
    ```bash
    python3 filter_fasta.py data/input_sequences.fasta results/excluded_sequences.fasta data/sequence_ids.txt --exclude
    ```

# Arguments Details:
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

# Environment Setup:
  Description: >
    Instructions to create and activate the Anaconda environment named `filter_sequences`, and install all necessary dependencies.

  Steps:
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

# Script Details:
  Description: >
    The script performs the following operations:
      1. Parses command-line arguments to determine input and output files, IDs file, and filter mode.
      2. Reads the list of sequence IDs from the provided `ids_file`.
      3. Iterates through the input FASTA file, filtering sequences based on the specified mode:
          - **Keep Mode (`--keep`)**: Retains only sequences whose IDs are in the `ids_file`.
          - **Exclude Mode (`--exclude`)**: Removes sequences whose IDs are in the `ids_file`.
      4. Writes the filtered sequences to the `output_fasta` file.
      5. Outputs summary information on the number of sequences processed and filtered.

# FAQ:
  - Question: "What is the purpose of this script?"
    Answer: >
      The script is designed to filter sequences in FASTA files based on a list of sequence IDs. It allows users to either keep only the specified sequences or exclude them from the output, facilitating targeted analyses in bioinformatics workflows.

  - Question: "How do I prepare the IDs file?"
    Answer: >
      The IDs file should be a plain text file with one sequence ID per line. Ensure there are no headers or additional formatting. For example:
      ```
      seq1
      seq2
      seq3
      ```

  - Question: "Can I use this script with large FASTA files?"
    Answer: >
      Yes, the script utilizes Biopython's SeqIO for efficient parsing, allowing it to handle large FASTA files effectively. However, ensure that your system has sufficient memory and storage resources for processing large datasets.

  - Question: "What happens if a sequence ID in the IDs file is not found in the FASTA file?"
    Answer: >
      The script will process all sequences in the FASTA file and apply the filtering criteria. Sequence IDs in the IDs file that do not match any sequences in the FASTA file will have no effect on the output.

  - Question: "Can I use this script to filter sequences based on partial IDs or patterns?"
    Answer: >
      Currently, the script filters sequences based on exact matches of sequence IDs. For partial matches or pattern-based filtering, you would need to modify the script to incorporate regular expressions or other matching criteria.

# Troubleshooting:
  Issue: "Script exits with usage error."
  Solution: >
    Ensure you are providing the required arguments when running the script. The correct usage is:
    ```bash
    python3 filter_fasta.py <input_fasta> <output_fasta> <ids_file> (--keep | --exclude)
    ```
    Example:
    ```bash
    python3 filter_fasta.py data/input_sequences.fasta results/filtered_sequences.fasta data/sequence_ids.txt --keep
    ```

  Issue: "No sequences are being written to the output file."
  Solution: >
    - Verify that the IDs in the `ids_file` match the sequence IDs in the `input_fasta`.
    - Ensure that you are using the correct mode (`--keep` or `--exclude`) based on your filtering needs.
    - Check for any leading/trailing whitespaces in the `ids_file` that might prevent matching.

  Issue: "Biopython is not installed or not found."
  Solution: >
    Ensure that you have activated the correct Anaconda environment (`filter_sequences`) and installed Biopython:
    ```bash
    conda activate filter_sequences
    pip install biopython
    ```
    Verify installation by running:
    ```bash
    python -c "import Bio; print(Bio.__version__)"
    ```

  Issue: "Permission denied when creating or writing to the output file."
  Solution: >
    Ensure that you have the necessary write permissions for the directory where you are trying to save the `output_fasta`.
    You can change the directory permissions or choose a different directory with appropriate permissions.

# Support:
  Description: >
    If you encounter any issues or have questions, feel free to reach out via the contact information provided above or open an issue in the repository. Contributions and feedback are highly appreciated!

Example Input:
  Description: >
    An example of the input IDs file (`sequence_ids.txt`) structure:
    ```
    seq1
    seq2
    seq3
    ```

# Output:
  Description: >
    The script generates an output FASTA file containing the filtered sequences based on the provided IDs and selected mode (keep or exclude).

  Files Generated:
    - `<output_fasta>`:
        Description: "FASTA file containing the filtered sequences. Depending on the selected mode, it either includes only the specified sequences or excludes them from the original set."

Additional Resources:
  - Name: "Biopython Documentation"
    URL: "https://biopython.org/wiki/Documentation"
    Description: "Comprehensive documentation for Biopython, including the SeqIO module used for parsing and writing FASTA files."
  
  - Name: "FASTA Format Specification"
    URL: "https://en.wikipedia.org/wiki/FASTA_format"
    Description: "Detailed information about the FASTA file format used for representing nucleotide or peptide sequences."

  - Name: "Anaconda Documentation"
    URL: "https://docs.anaconda.com/anaconda/"
    Description: "Official documentation for Anaconda, a distribution of Python and R for scientific computing."

---

# Script 4: pie_multidomain_architecture.py, Taxonomic Pie Chart Generator

This repository contains a Python script that processes two TSV files containing genomic and protein annotation data, merges the data, extracts taxonomic lineages, and generates pie charts displaying the distribution of protein signature architectures. The script is designed to work with biosynthetic data and is particularly useful for visualizing the prevalence of different signature types (e.g., FAAL, NRPS, PKS) across taxonomic groups.

## Features

- **Signature Description Combination:**  
  Combines multiple "Signature.description" entries per protein accession into a single field (`Combined.description`). Special handling is included to replace "FAAL" with "FAAL stand-alone".

- **Data Loading and Validation:**  
  Reads two TSV files, prints column information and shapes, and performs basic checks (e.g., verifying the presence of an `Assembly` column with valid IDs).

- **Data Merging:**  
  Merges the two datasets using the `Protein.accession` field (inner join) and creates the combined signature description.

- **Taxonomic Extraction:**  
  Extracts taxonomic levels (superkingdom, phylum, class, order, family, genus, species) from a `Lineage` column using the [ete3](http://etetoolkit.org/) NCBITaxa module. Includes functions to directly obtain the phylum.

- **Lineage Update and Domain Filtering:**  
  Updates the DataFrame with extracted taxonomic levels and filters the data to keep only the rows that match the chosen domain (e.g., Bacteria, Archaea, Eukaryota). It also removes rows with missing values for key taxonomic levels.

- **Pie Chart Plotting:**  
  Generates pie charts as subplots (2 columns per row) for specified taxonomic groups.  
  - For taxa such as "Candidatus Rokuibacteriota" and "Gemmatimonadota", the script uses regular expressions to match the `Lineage` data.
  - Displays the top N most frequent architectures and groups the remaining counts into an "Others" category.
  - Creates a doughnut chart style with a central white circle.
  - Automatically saves the plots in PNG, SVG, and JPEG formats with a specified DPI.

## Requirements

- Python 3.x
- Libraries:
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [numpy](https://numpy.org/)
  - [ete3](http://etetoolkit.org/)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
    ```
  ```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
 ```

```bash
pip install -r requirements.txt
 ```
```bash
pip install pandas matplotlib numpy ete3

# Example 
 ``
python3 pie_multidomain_architecture.py <table1_path> <table2_path> <domain_name> <top_n> <taxonomic_level> <taxon_list> <dpi>
 ```

# Parameters:

<table1_path>: Path to the first TSV file containing genomic/protein taxonomy data.
<table2_path>: Path to the second TSV file containing protein signature descriptions.
<domain_name>: Domain to filter (e.g., Bacteria, Archaea, Eukaryota).
<top_n>: Number of top signature architectures to display in the pie charts.
<taxonomic_level>: Taxonomic level to use for filtering (Phylum, Order, or Genus).
<taxon_list>: Comma-separated list of taxon names.
Note: For "Candidatus Rokuibacteriota" and "Gemmatimonadota", the script applies specific regex matching.
<dpi>: Resolution (dots per inch) for saving the figures.

# Script 6: Taxonomic FAAL Analyzer: bar_mean_faal_genome.py

# Taxonomic FAAL Analyzer

This repository contains a Python script that processes a TSV table of genomic and taxonomic data, extracts taxonomic lineages using the ETE3 library, and generates filtered statistics and visualizations of FAAL counts per genome across different taxonomic groups.

## Features

- **Taxonomic Extraction:** Uses the ETE3 library to fetch taxonomic lineage information from NCBI.
- **Data Filtering:** Filters out unwanted rows (e.g., those with 'environmental samples' or missing assembly information) and applies specific criteria based on taxonomic levels.
- **Aggregation:** Counts total FAAL occurrences and unique genome assemblies for each taxonomic group.
- **Visualization:** Generates bar plots that show the mean FAAL count per genome for the top N taxonomic groups, with annotations for genome counts and total FAAL counts.
- **Output Files:** Saves the merged data in a TSV file and exports plots in PNG, SVG, and JPEG formats.

## Requirements

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [numpy](https://numpy.org/)
- [ete3](http://etetoolkit.org/)

## Installation
You can install the required packages using `pip`:

# Usage
Run the script from the command line with the following arguments:

```bash
python script_name.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>
```

<table1.tsv>: Path to the input TSV file containing genomic and taxonomic data. The file should include columns such as Species, Assembly, and Lineage.
<Domain>: The domain to filter the data (e.g., Eukaryota or Bacteria).
<Taxonomic Level>: The taxonomic level for analysis (e.g., Phylum, Order, Family, or Genus).
<Top N>: The number of top taxonomic groups to visualize, ranked by the mean FAAL count per genome.
<DPI>: The dots-per-inch resolution for the output images.


# Example

```bash
python3 barplot_mean_FAAL_genomev1.py Genomes_Total_proteinas_taxonomy_FAAL_metadata_nodup.tsv Eukaryota Genus 30 300
```
# How It Works
# Data Loading & Preprocessing:

The script loads the TSV file using pandas.
It updates the Lineage column by retrieving taxonomic information via the ETE3 toolkit.
Rows with environmental samples in their lineage or missing Assembly information are removed.
Taxonomic Group Extraction & Filtering:

For Eukaryota, the lineage is updated and filtered; for other domains, alternative logic is applied.
Taxonomic groups are extracted from species names based on the specified taxonomic level.
Specific criteria (e.g., names ending with "ales" for orders or "eae" for families) are applied to filter the results.
Data Aggregation:

The script calculates the total FAAL count per taxonomic group and the number of unique genomes (assemblies) per group.
It computes the mean FAAL count per genome and filters out groups with fewer than a minimum number of genomes.
Visualization:

A bar plot is generated using seaborn, with bars ordered by the mean FAAL count per genome.
Genome counts and total FAAL counts are annotated on the bars.
The plot is saved in PNG, SVG, and JPEG formats.

# Output:
- The merged data is saved as merged_data.tsv.
- Visualizations are saved as mean_faal_per_genome.png, mean_faal_per_genome.svg, and mean_faal_per_genome.jpeg.

```bash
pip install pandas matplotlib seaborn numpy ete3
```

# Script 7: FAALs Taxonomic Analysis: barplot_normalized_counts.py

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
python3 barplot_normalized_counts_faal.py Genomes_Total_proteinas_taxonomy_FAAL_metadata_nodup.tsv ncbi_dataset_data_28_january_taxonomy_nodup.tsv Eukaryota Genus 30 300
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

# Scriot 8: "Taxonomic Analysis and Visualization Tool" : scatterplot_mean_faal_genome.py 

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

# Script 9: name: CAL Domain and GenBank Analyzer
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

# Overview# Script 8: organize_bigslice.py
# Organize Big Slice

A script to organize antiSMASH directories into BiG-SLiCE datasets, grouped by a user-selected taxonomic level (Phylum, Order, or Genus).

## Description

This script is designed to help organize antiSMASH results into a directory structure compatible with BiG-SLiCE. It performs the following tasks:

- **Taxonomic Grouping:** Uses a TSV taxonomy table to map each genome (identified by its Assembly Accession) to a specific taxonomic level (Phylum, Order, or Genus).
- **File Copying:** Searches for `.gbk` files in antiSMASH result directories and copies them into a structured dataset directory.
- **Taxonomy Files Generation:** Creates TSV taxonomy files for each dataset and updates a master `datasets.tsv` file containing information on all datasets.
- **Detailed Logging:** Provides detailed logs (with a verbose mode option) and supports log file rotation.
- 
# Example comand line: 

python3 organize_bigslice.py --bigslice_dir bigslice_dir --antismash_dir filtrados_subdir_CAL/ --taxonomy_table Fungi_supplementar2.tsv

# Arguments
--bigslice_dir: Path to the directory where BiG-SLiCE datasets will be created.
--antismash_dir: Path to the directory containing antiSMASH results.
--taxonomy_table: Path to the taxonomy table (TSV format).
--assembly_column: Column name in the taxonomy table that contains the Assembly Accession (default: "Assembly Accession").
--lineage_column: Column name in the taxonomy table that contains the Lineage (default: "Lineage").
--log_file: Path to the log file (default: organize_big_slice.log).
--taxonomic_level: Taxonomic level to group results by. Options: Phylum, Order, or Genus (default: Genus).
--verbose: Enables detailed logging output.

# This command will:

Process antiSMASH result directories located at /data/antismash_results.
Use the taxonomy information in /data/taxonomy.tsv.
Group the results based on the "Order" taxonomic level.
Enable verbose logging to display detailed processing information.

bigslice_dir/
├── dataset_<taxon_name>/            # A directory for each generated dataset
│   └── genome_<Assembly>/           # Subdirectories containing the .gbk files for each genome
├── taxonomy/                        # TSV taxonomy files for each dataset
│   └── taxonomy_dataset_<taxon_name>.tsv
└── datasets.tsv                     # A master file with dataset information

# Logging
Log File: By default, the script creates a log file named organize_big_slice.log to record progress, warnings, and errors.
Verbose Mode: Use the --verbose flag to enable more detailed debug output.

## Prerequisites

- **Python 3.6+**
- **Python Libraries:**
  - `pandas` (for table manipulation)
  - Other libraries used in the script are part of the Python standard library.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/organize_big_slice.git
   cd organize_big_slice


# Script 10: Combine Tables Script

This Python script is designed to combine two TSV tables based on the "Protein Accession" column. It processes and groups information from the second table by aggregating signature accessions and descriptions, then merges the results with the first table.

---

## Features

- **File Reading and Validation:**  
  Reads two TSV files and ensures that the second table contains the required columns: `Signature.accession` and `Signature.description`.

- **Processing the Second Table:**  
  - Groups data by `Protein.accession`.  
  - Aggregates the signature accessions and descriptions into unique strings separated by hyphens.  
  - Creates a new column (`Total Signature Description`) that counts the number of aggregated descriptions.  
  - Adds a new column, `color three`, which is assigned as follows:
    - `#FFFFFF` if the description contains "FAAL" and has only one signature.
    - `#000000` for all other cases.

- **Merging Tables:**  
  Combines the first table with the processed data from the second table using the `Protein Accession` key.

- **Output Generation:**  
  Saves the combined table as a TSV file for further analysis and visualization.

---

## Requirements

- **Python 3.x**
- **Pandas**
- **Argparse**

Install the primary dependency using:

```bash
pip install pandas
```
# How to Use
# Prepare your files:
Ensure that your input files are in TSV format and that the second table includes the Signature.accession and Signature.description columns.

# Run the script:
Open your terminal and run the following command, replacing the paths as needed:

```
python combine_tables_script.py <path_to_table1.tsv> <path_to_table2.tsv> <output_path.tsv>
```
bash
python combine_tables_script.py data/table1.tsv data/table2.tsv output/combined_table.tsv

# Output:
The script will generate the combined table at the specified output path and print a confirmation message.

# Code Structure
Function combine_tables:

Converts the input tables into pandas DataFrames.
Validates the presence of required columns in the second table.
Groups and aggregates data from the second table.
Adds the color three column based on defined conditions.
Merges the first DataFrame with the processed DataFrame from the second table.
Function main:

Reads the TSV files.
Calls the combine_tables function to process the data.
Saves the merged DataFrame as a new TSV file.
Main Block:

Uses argparse to capture command-line arguments and trigger the script execution.
Error Handling
The script includes basic error handling for reading the TSV files. If an error occurs while reading the files, an error message will be displayed and the execution will be halted.

# Overview# Script 10: Merge Bigscape Step 1

A Python script to merge two TSV tables based on the "BGC" column.

## Description

This script reads two TSV (tab-separated values) files and performs a left merge on the "BGC" column using pandas. The resulting merged table is then saved as a new TSV file.

> **Note:**  
> Ensure that in the second table the column originally named "BGC name" is renamed to "BGC" before running the script.

## Prerequisites

- **Python 3.6+**
- **Python Packages:**
  - `pandas` (Install via pip: `pip install pandas`)

## Usage

Run the script from the command line by providing three arguments:

1. Path to the first table (TSV format).
2. Path to the second table (TSV format).
3. Path for the output merged table (TSV format).

### Command Line Example

```bash
python3 merge_bigscape_step1.py Network_Annotations_Full.tsv ./mix/mix_clans_0.30_0.70.tsv merged_Network_Annotations_Full_modificada__clans_0.30_0.70_update.tsv
```
OBS: Before run the script merge_bigscape_step1.py rename collum 1 of the table mix_clans_0.30_0.70.tsv for BGC

# Arguments
table1: Path to the first TSV file.
table2: Path to the second TSV file (make sure the column "BGC name" is changed to "BGC").

output: Path where the merged TSV file will be saved.

# How It Works

# Reading the Tables:

The script reads the two input TSV files using pandas.read_csv() with a tab separator (\t). The parameter on_bad_lines='skip' is used to ignore any problematic lines.

# Merging the Tables:
The two tables are merged using a left join on the "BGC" column.

# Saving the Output:
The merged DataFrame is saved to the specified output file in TSV format.

# Error Handling:
If an error occurs while reading the input files, an error message is printed and the script exits.

# Overview# Script 9: pie_bgc_classe_domain_S16.py

# BGC Statistics Bar Chart

This repository contains a Python script that processes a TSV table of biosynthetic gene cluster (BGC) annotations, calculates several statistics, and generates a high-resolution bar chart suitable for publication (e.g., for NAR).

#  Figure S16 with results obtained from BiG-SCAPE for 12,214 bacterial genomes 

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
```
# Script 12: # BGC Bigscape Parser & Visualizer

This project contains a Python script that processes TSV files with biosynthetic gene cluster (BGC) data, extracts taxonomic information, and generates pie charts to visualize the distribution of BiG-SCAPE classes. The script performs the following tasks:

- **Data Loading:** Reads a TSV file containing structured data.
- **Taxonomic Information Extraction:** Extracts species, genus, order, and phylum from the `Taxonomy` column using custom functions. It leverages the [ete3](http://etetoolkit.org/) library and the NCBITaxa API.
- **Genome ID Creation:** Generates the `Genome_ID` column from the `BGC` data.
- **Proportion Calculation:** Groups and calculates the proportions of BiG-SCAPE classes present in the dataset.
- **Visualization:** Creates pie charts and automatically saves the figures in SVG, PNG, and JPEG formats at high resolution (300 dpi).

## Requirements

- Python 3.x
- Libraries:
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [numpy](https://numpy.org/)
  - [ete3](http://etetoolkit.org/)
- Standard modules: `sys`, `math`, `re`

> **Tip:** It is recommended to use a virtual environment to manage the dependencies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
```
Create and activate a virtual environment:

   ```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

# Install the dependencies: If you have a requirements.txt file:

```bash
pip install -r requirements.txt
```
Or, install the libraries manually:

```bash
pip install pandas matplotlib numpy ete3
```
# Usage

Run the script by passing the path to the TSV file as an argument:

```bash
python3 bgc_class_bigscape.py path_to_file.tsv
```

# During execution, the script will prompt you to:

Select the taxonomic level (Phylum, Order, or Genus) interactively.
Enter the taxon names (separated by commas) for filtering.
After processing, the script will display the pie charts and automatically save the images in SVG, PNG, and JPEG formats at 300 dpi.

# GeneClusterMineX v 2.0 :rocket:
_____________________________________________________________________________________________________________________________________________________

 - **v 2.0.0: automation of processing of several Genomes fasta/fna files by antismash**

GeneClusterMineXv2.0.0.py

## Automated Secondary Metabolite Analysis with antiSMASH
**run_antismash.py is a Python script designed to streamline the execution of antiSMASH on multiple sequence files (.fna or .fasta) within a directory. The script automates the creation of output directories for each input file, manages detailed logs, and offers flexibility to customize analyses according to user needs.**

- Batch Processing: Executes antiSMASH on multiple sequence files simultaneously.
## Organized Results: Creates specific output directories for each input file, named as Result_input_filename.
Comprehensive Logging: Records detailed logs for each processing step, including timestamps and error messages.
## Flexible Analyses: Allows activating all available antiSMASH analysis tools or selecting specific analyses.
## GlimmerHMM Support: Integrates the GlimmerHMM gene prediction tool, automatically adjusting taxonomy to fungi.


## Requirements

- Operating System: Linux or Unix-based
- Python: Version 3.6 or higher
- antiSMASH: Properly installed and configured
- Appropriate Permissions: Read permissions for input files and write permissions for output and log directories

1. Clone the repository:

 ```git clone https://github.com/mattoslmp/CNP-Ciimar.git ```

2. Navigate to the cloned directory.

3. Create a Conda environment and install antiSMASH:

- conda create -n antiSMASH-env
- conda activate antiSMASH-env
- install dependencies:
- conda install hmmer2 hmmer diamond fasttree prodigal blast muscle glimmerhmm
- install python (versions 3.9, 3.10, and 3.11 tested, any version >= 3.9.0 should work):
- conda install -c conda-forge python
- wget https://dl.secondarymetabolites.org/releases/7.0.0/antismash-7.0.0.tar.gz
- tar -zxf antismash-7.0.0.tar.gz
- pip install ./antismash-7.0.0
- download-antismash-databases


## Usage
- Directory Structure
Input Directory (input_dir): Contains all .fna or .fasta files you wish to analyze.

```bash
./fna_inputs/
├── sample1.fna
├── sample2.fasta
└── ...

```

## Output Directory (output_dir): Where the results will be stored. Each input file will have its own results subdirectory.

---

## Output and Logs

After running the script, the output directory will be organized as follows:

```bash
./output_antismash/
├── log.txt
├── Result_sample1/
│   ├── index.html
│   ├── ... (other output files)
├── Result_sample2/
│   ├── index.html
│   ├── ... (other output files)
└── ...



```shell
 ./run_antismash.py <input_dir> <output_dir> [options]
```

## How to run GeneClusterMinerX for all available tools:

```shell
nohup python3 GeneClusterMineXv2.0.py --genefinding-tool prodigal ./ output_antismash  --parallel-processes 138 --all --cpus 111 &
```


- <input_dir>: Path to the directory containing .fna or .fasta files.
- <output_dir>: Path to the directory where results will be saved.
- --all: All available tools will be used.
- --parallel-processes 80 fasta files will be processed at same time 
- --cpus 8: 8 CPUs will be used.
- --genefinding-tool glimmerhmm: The GlimmerHMM tool will be used.
  


Available Options
Positional Arguments:

### Available Options

#### Positional Arguments:

- **input_dir**: Input directory with `.fna` or `.fasta` files.
- **output_dir**: Output directory to store results.

#### General Options:

- `--antismash-help`: Displays antiSMASH help and exits.
- `-t`, `--taxon`: Taxonomic classification of the input sequence. Options: `bacteria` (default), `fungi`.
- `-c`, `--cpus`: Number of CPUs to use in parallel (default: 4).
- `--databases`: Root directory of databases used by antiSMASH.

#### Additional Output Options:

- `--output-basename`: Base name for output files within the output directory.
- `--html-title`: Custom title for the HTML output page.
- `--html-description`: Custom description to add to the output.
- `--html-start-compact`: Uses compact view by default on the overview page.
- `--html-ncbi-context`: Shows links to NCBI genomic context of genes.
- `--no-html-ncbi-context`: Does not show links to NCBI genomic context of genes.

#### Additional Analyses:

- `--all`: Activates all available antiSMASH analyses.
- `--fullhmmer`: Executes HMMer analysis across the entire genome using Pfam profiles.
- `--cassis`: Prediction based on motifs of SM gene cluster regions.
- `--clusterhmmer`: Executes HMMer analysis limited to clusters using Pfam profiles.
- `--tigrfam`: Annotates clusters using TIGRFam profiles.
- `--asf`: Executes active site analysis.
- `--cc-mibig`: Compares identified clusters with the MIBiG database.
- `--cb-general`: Compares identified clusters with a database of antiSMASH-predicted clusters.
- `--cb-subclusters`: Compares identified clusters with known subclusters synthesizing precursors.
- `--cb-knownclusters`: Compares clusters with known clusters from the MIBiG database.
- `--pfam2go`: Maps Pfam to Gene Ontology.
- `--rre`: Executes RREFinder in precision mode on all RiPP clusters.
- `--smcog-trees`: Generates phylogenetic trees of orthologous groups of secondary metabolite clusters.
- `--tfbs`: Executes transcription factor binding site locator on all clusters.
- `--tta-threshold`: Minimum GC content to annotate TTA codons (default: 0.65).

#### Gene Prediction Options:

- `--genefinding-tool`: Gene prediction tool to use. Options: `glimmerhmm`, `prodigal`, `prodigal-m`, `none`, `error` (default).
- `--genefinding-gff3`: Specifies a GFF3 file to extract features.

#### Logging Options:

- `--log-file`: Log file to save messages (default: `log.txt` in the output directory).


---

## Monitoring and Management

### Checking Running Processes

You can verify if the script is running using `ps` or `pgrep`. For example:

```bash
ps aux | grep run_antismash.py

Or:

```shell
pgrep -fl run_antismash.py
```

## Monitoring Logs in Real-Time
Internal Script Logs (output_log.txt):

```shell
tail -f ./output_antismash/output_log.txt
```

```shell
tail -f run_output.log
```

## Stopping the Process
If you need to terminate the background process, follow these steps:

## Identify the PID (Process ID):

```shell
pgrep -f run_antismash.py
```

##Kill the Process:

```shell
kill -9 PID
```

# Replace PID with the actual process number.

## Final Considerations

- File Extensions Compatibility: The script is configured to process files with .fna and .fasta extensions. Ensure your input files have one of these extensions to be recognized by the script.

- System Resources: Specifying a high number of CPUs with the --cpus option can speed up processing but ensure your system has sufficient resources to handle the load without overloading.

Detailed Logs: Utilize the log files to monitor progress and identify any issues during processing.

Future Updates: If antiSMASH introduces new analysis options in the future, you will need to update the script to include these new options, both in argparse and in the logic that activates options with --all.

File Extensions: If you have compressed files (like .fasta.gz), you will need to decompress them before processing, as the current script does not support compressed files.

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
