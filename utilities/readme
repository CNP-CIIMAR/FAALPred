# FAALpred/utilities

The FAALpred/utilities repository provides a comprehensive suite of Python scripts for the identification and phylogenetic analysis of FAAL (Fatty Acyl-AMP Ligases) homologs across diverse genomes and metagenomes. The pipeline includes tools for FAAL homolog detection, taxonomic classification, sequence filtering, and phylogenetic tree construction using HMM-based searches, clustering, and alignment methods. Additionally, it features genome metadata retrieval, visualization of FAAL distributions across taxonomic groups, and secondary metabolite gene cluster analysis using antiSMASH and BiG-SCAPE. The repository integrates custom filtering scripts, statistical analysis tools, and visualization methods, making it a valuable resource for exploring the global diversity and functional role of FAAL proteins.


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
      git clone https://github.com/CNP-CIIMAR/FAALPred/blob/main/utilities/protein_hmm.py
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
# Script 2: FASTA Sequence Filter Tool: fasta_sequence_filter.py

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
**Prerequisites:**
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

- **`input_fasta`**:  
  - Ensure the input file is in proper FASTA format.  
  - Supports large FASTA files efficiently using Biopython's SeqIO.

- **`output_fasta`**:  
  - The script will create this file if it does not exist.  
  - If the file exists, it will be overwritten with the filtered sequences.

- **`ids_file`**:  
  - Should be a plain text file with one sequence ID per line.  
  - No headers or additional formatting required.

- **`--keep`**:  
  - When this option is used, only sequences with IDs present in the `ids_file` will be written to the `output_fasta`.

- **`--exclude`**:  
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
# Script 3: Protein to Taxonomic Data Pipeline: faal_genotax.py, 

This script retrieves taxonomic information for protein accessions by querying the NCBI databases and processing the results. It first fetches genome accession IDs via a subprocess call to `efetch` and then retrieves the corresponding species and lineage information using Biopython's Entrez module. Finally, the script outputs two files: one containing the mapping between protein and genome accessions, and another with taxonomic details.

---

## Features

- **Genome Accession Retrieval:**  
  Uses a subprocess call to `efetch` to retrieve IPG information and extract either the RefSeq or INSDC genome accession.

- **Taxonomic Data Retrieval:**  
  Queries the NCBI Protein database using Biopython's Entrez module to obtain species and taxonomic lineage for each protein accession.

- **Output Files:**  
  - A genome mapping file containing protein accessions and their corresponding genome accessions.
  - A taxonomic information file (in TSV format) with columns for Protein Accession, Genome Accession, Species, and Lineage.

---

## Prerequisites

- Python 3.x
- [pandas](https://pandas.pydata.org/)
- [Biopython](https://biopython.org/)
- NCBI Entrez access (set your email in the script)
- Command-line tool `efetch` installed and available in your system's PATH

---

## Installation

1. **Clone the repository** (if applicable) or download the script file.

2. **Install dependencies** using pip:
   ```bash
pip install pandas biopython
```

# Ensure efetch is Installed:
Follow instructions from the [NCBI Entrez Direct documentation](https://www.ncbi.nlm.nih.gov/books/NBK179288/) to install and configure **efetch**.

---

# Usage

Run the script from the command line with the following arguments:

```bash
python faal_genotax.py <input_filename> <genome_output_filename> <taxonomic_output_filename>
```
________________________________________
# Script Overview

- **Argument Check**:  
  The script verifies that at least three arguments are provided (input filename, genome output filename, taxonomic output filename). If not, it prints usage instructions and exits.

- **Entrez Email Setup**:  
  You must set your email address in the script (replace `'your_email@example.com'` with your actual email).

- **Function `get_taxonomic_rank`**:  
  This function takes a protein accession, queries the NCBI Protein database via Entrez, and extracts the species and lineage information.

- **Genome Mapping Generation**:  
  For each protein accession in the input file, the script calls `efetch` (via `subprocess`) to retrieve IPG information. It extracts the genome accession from lines containing `"RefSeq"` or `"INSDC"` and writes the mapping to the genome output file.

- **Taxonomic Table Creation**:  
  The genome mapping file is then read, and for each protein accession, the taxonomic data (species and lineage) is retrieved using the `get_taxonomic_rank` function. A `pandas` DataFrame is created with the results and saved as a TSV file.

- **Output Messages**:  
  Upon completion, the script prints the locations of the output files.


# Script 4: NCBI Assembly Metadata Enrichment Tool - `get_genome_metadata.py`

## Description:
A Python script designed to process NCBI assembly IDs, retrieve comprehensive taxonomic lineages, and fetch additional metadata from the NCBI BioSample database. The tool enriches assembly data with geographic and biome distribution information, outputting both comprehensive and filtered datasets for further analysis.

## Features:
- Retrieves taxonomic lineage information using `NCBITaxa` from the `ete3` library.
- Fetches additional metadata from the NCBI BioSample database using Biopython's `Entrez` module.
- Parses XML responses to extract geographic locations, biome distributions, and latitude/longitude coordinates.
- Categorizes biome descriptions based on GOLD standards.
- Generates both comprehensive and filtered TSV output files containing enriched assembly data.
- Provides summary statistics on the number of assemblies with specific metadata fields populated.


# Dependencies:
- Anaconda 3.x
- Python 3.8 or higher
- `pandas`
- `ete3`
- `biopython`

# Installation:
## Prerequisites:
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


# Script 5: FAALs Taxonomic Analysis: barplot_normalized_counts.py

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

# Script 6: Taxonomic Analysis and Graph Generation: scatterplot_mean_faal_genome_size.py
This Python script processes a tab-separated values (TSV) file containing genomic and taxonomic data to perform taxonomic filtering, analysis, and visualization. It is designed to analyze the relationship between genome size and FAAL (fatty acyl-AMP ligase) counts across different taxonomic groups.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Script Details](#script-details)
  - [Taxonomic Lineage Extraction](#taxonomic-lineage-extraction)
  - [Filtering Criteria](#filtering-criteria)
  - [Lineage Correction using ete3](#lineage-correction-using-ete3)
  - [Data Visualization](#data-visualization)
- [Output Files](#output-files)
- [License](#license)

---

# Script 6: Mean FAAL Counts: scatterplot_counts_faal.py

## Description:
`scatterplot_counts_faal.py` is a comprehensive tool for:

- Extracting specific taxonomic groups from a lineage string.
- Filtering taxonomic data based on custom criteria (e.g., ending patterns such as `ales` for orders, `eae` for families).
- Optionally correcting taxonomic lineages using the `ete3` package when analyzing *Eukaryota*.
- Calculating average FAAL counts per genome and average genome sizes.
- Visualizing the results in a scatterplot with customizable aesthetics.

The script is modular and configurable to handle different domains (e.g., *Eukaryota*, *Bacteria*) and taxonomic levels (e.g., Order, Family, Genus).

---

## Requirements

- **Python 3.x**
- **pandas** – for data manipulation  
- **matplotlib** – for plotting  
- **seaborn** – for enhanced visualization  
- **numpy** – for numerical operations  
- **ete3** – for taxonomic data processing (optional, used when analyzing *Eukaryota*)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CNP-CIIMAR/FAALPred/blob/main/utilities/scatterplot_mean_faal_genome_size.py
cd your-repository
   ```
# Install the required Python packages:
 ```bash
pip install pandas matplotlib seaborn numpy ete3
   ```

________________________________________
# Usage

Run the script from the command line by providing the following arguments:

```bash
python3 scatterplot_counts_faal.py <table1.tsv> <Domain> <Taxonomic Level> <Top N> <DPI>
```
- `<table1.tsv>`: Path to the input TSV file.
- `<Domain>`: The taxonomic domain to filter (e.g., `"Eukaryota"` or `"Bacteria"`).
- `<Taxonomic Level>`: The taxonomic level to analyze (e.g., `"Order"`, `"Family"`, `"Genus"`).
- `<Top N>`: Number of top taxonomic groups (by mean FAAL count per genome) to visualize.
- `<DPI>`: Dots per inch (DPI) setting for the output image resolution.

### Example:

```bash
python3 scatterplot_counts_faal.py data/table1.tsv Eukaryota Order 10 300
```
________________________________________
# Script Details

## Taxonomic Lineage Extraction
- **Function:** `extract_taxonomic_group(lineage, level)`
- **Purpose:** Parses a semicolon-separated lineage string and extracts the taxonomic group at the specified level (e.g., Domain, Phylum, Class, Order, Family, Genus, Species).
- **Error Handling:** Returns `'Unknown'` if the desired level is not present in the lineage.

## Filtering Criteria
- **Function:** `filter_by_criteria(name, level, domain_name)`
- **Purpose:** Implements custom filtering:
  - **Order:** Accepts names ending with `ales`.
  - **Family:** Accepts names ending with `eae`.
  - **Genus:** Excludes names ending with `ales` or `eae`.
  - **Domain and Phylum:** Always accepted.
- **Benefit:** Allows for refined selection of taxonomic groups based on naming conventions.

## Lineage Correction using `ete3`
- **Function:** `get_corrected_lineage(species, use_ete3)` and `update_lineage(df, use_ete3)`
- **Purpose:** Optionally corrects and updates the lineage information for a species using the `ete3` package.
- **Usage:** Activated when the specified domain is *Eukaryota*. If an error occurs during lineage retrieval, the function safely returns `'Unknown'`.

## Data Visualization
- **Function:** `generate_filtered_table_and_graphs(table1_path, domain_name, taxonomic_level, top_n, dpi)`
- **Process:**
  1. **Data Import and Cleaning:** Reads the TSV file, updates lineage data (if applicable), and filters out environmental samples and missing assembly entries.
  2. **Filtering:** Filters the dataset based on the specified domain and taxonomic criteria.
  3. **Aggregation:** Computes FAAL counts and unique genome counts for each taxonomic group.
  4. **Calculation:** Determines the mean FAAL count per genome and average genome size.
  5. **Visualization:** Generates a scatterplot:
     - Customizes plot size based on the domain.
     - Applies jitter to avoid overlapping points.
     - Uses a tailored color palette with excluded colors for clarity.
     - Adjusts axis labels, scales, and legend based on the domain and taxonomic level.
  6. **Output:** Saves the plot in PNG, SVG, and JPEG formats.

________________________________________
# Output Files

After successful execution, the following files are generated:

- `taxonomic_analysis_plot.png`
- `taxonomic_analysis_plot.svg`
- `taxonomic_analysis_plot.jpeg`

These files contain the scatterplot visualization of average genome size versus average FAAL count per genome for the top taxonomic groups.


# Script 7: GeneClusterMineX v 2.0 :rocket:
_____________________________________________________________________________________________________________________________________________________

 - **v 2.0.0: automation of processing of several Genomes fasta/fna files by antismash**

# GeneClusterMineX v2.0.0

## Automated Secondary Metabolite Analysis with antiSMASH

**`run_antismash.py` is a Python script designed to streamline the execution of antiSMASH on multiple sequence files (`.fna` or `.fasta`) within a directory. The script automates the creation of output directories for each input file, manages detailed logs, and offers flexibility to customize analyses according to user needs.**

### Features:

- **Batch Processing:** Executes antiSMASH on multiple sequence files simultaneously.
- **Organized Results:** Creates specific output directories for each input file, named as `Result_input_filename`.
- **Comprehensive Logging:** Records detailed logs for each processing step, including timestamps and error messages.
- **Flexible Analyses:** Allows activating all available antiSMASH analysis tools or selecting specific analyses.
- **GlimmerHMM Support:** Integrates the `GlimmerHMM` gene prediction tool, automatically adjusting taxonomy to fungi.



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


# Acknowledgements
- ETE Toolkit: For providing tools to work with phylogenetic trees and taxonomy.
- Pandas, Matplotlib, Seaborn: For data manipulation and visualization capabilities.
For any questions or issues, please open an issue in the repository.


# Script 8: GBK Files Domain Analysis and Filtering Script: CALDomainAnalyzer.py.

This Python script processes GenBank (`.gbk`) files contained within subdirectories of a specified input directory. It searches for features containing specific domains—primarily the "CAL_domain" and optionally "AMP-binding"—calculates the total file sizes, and copies the files that contain the target domains to a filtered output directory. A summary CSV file is generated, and detailed processing logs are saved to a log file.

---
## Features

- **Domain Analysis:**  
  - Scans each GenBank file for features containing "CAL_domain" (and "AMP-binding" if requested) by checking various qualifiers.
  - Counts the number of occurrences of these domains per file.

- **Genome ID Extraction:**  
  - Extracts the Genome ID from the subdirectory name using two possible naming patterns.

- **Size Calculation:**  
  - Calculates the total size (in megabases and gigabases) of all `.gbk` files that contain the target domain(s) in each subdirectory.
  - Formats size values for readability.

- **File Filtering and Copying:**  
  - Creates a new output directory (`filtrados_subdir_CAL`) in the parent directory of the input.
  - Copies files containing the "CAL_domain" (and optionally "AMP-binding") to subdirectories named with the corresponding Genome ID prefix.

- **Logging:**  
  - Uses the Python `logging` module to log detailed processing information, warnings, and errors.
  - Logs are written both to a specified log file and the console.

- **Summary CSV:**  
  - Generates a summary CSV file (`summary.csv`) in the input directory containing:
    - Genome Assembly ID
    - Count of "CAL_domain" occurrences
    - Count of "AMP-binding" occurrences (if enabled)
    - Total file size in Mb and Gb for the subdirectory

- **Disk Space Check:**  
  - Checks available disk space in the parent directory to ensure there is sufficient space to copy the filtered files.


---

## Requirements

- **Python 3.x**
- **Biopython** (for parsing GenBank files)  
- **Argparse**, **Pathlib**, **CSV**, **Shutil**, **Logging**, **Datetime** (standard Python libraries)

Install Biopython using pip if not already installed:

```bash
pip install biopython
```
# How to Use
1.	Prepare your input directory:
Organize your GenBank files (.gbk) in subdirectories. Each subdirectory should be named using one of the following patterns to allow extraction of a Genome ID:
o	Pattern 1: Starts with Result_ followed by GCA_ or GCF_ and additional information.
o	Pattern 2: Ends with .fna (for alphanumeric codes).
2.	Run the script:
Open your terminal and execute the script with the required arguments. For example:
``bash

python CALDomainAnalyzer.py <input_directory> <log_file> [--search-amp-binding]

```
Example:
```bash
python CALDomainAnalyzer.py data/gbk_files log_processamento.log --search-amp-binding
```

- **`<input_directory>`**:  
  Path to the directory containing subdirectories with .gbk files.

- **`<log_file>`**:  
  Path to the log file where processing information will be saved (e.g., log_processamento.log).

- **`--search-amp-binding`** (optional):  
  If specified, the script will also search for the "AMP-binding" domain.

---

## Output Files

- **Log File:**  
  Contains detailed logs of the processing steps.

- **Summary CSV:**  
  A file named `summary.csv` will be created in the input directory with the analysis summary.

- **Filtered Files:**  
  A directory named `filtrados_subdir_CAL` will be created in the parent directory of the input. Inside, subdirectories with filtered GenBank files (prefixed by the Genome ID) will be available.

---

________________________________________

# Code Structure

- **Function `contains_domain`:**  
  Checks if a GenBank feature contains any of the specified domains by examining multiple qualifiers.

- **Function `extract_genome_id`:**  
  Extracts the Genome ID from the subdirectory name based on defined patterns.

- **Function `format_size`:**  
  Converts and formats file sizes from megabases to gigabases.

- **Function `setup_logging`:**  
  Configures logging to output messages to both a file and the console.

- **Function `process_gbk_files`:**  
  - Iterates over each subdirectory in the input directory.
  - Processes .gbk files to detect target domains.
  - Calculates total sizes, logs processing details, and copies files containing the target domains.
  - Checks available disk space and logs warnings if insufficient.
  - Generates a summary CSV file.

- **Main Function:**  
  Uses argparse to handle command-line arguments and triggers the processing workflow.

________________________________________

# Error Handling

- The script checks for the existence and validity of the input directory.
- Exceptions during file reading, processing, or copying are logged with error messages.
- The log file is created (or overwritten) to capture all processing details.


Script 9: Merge Bigscape Step 1. A Python script to merge two TSV tables based on the "BGC" column.
merge_bigscape_step1.py

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

Script 10.1: Merge BiG-SCAPE Step 2
•	Código: merge_bigscape_step2.py

Script 11: 

# BGC Class Bigscape Analysis Script

This repository contains a Python script for processing a tab-separated values (TSV) file with genomic and biosynthetic gene cluster (BGC) data. The script is designed to perform taxonomic extraction, filtering, and visualization of BiG-SCAPE classes in a user-defined taxonomic context. It generates pie charts representing the distribution of BiG-SCAPE classes and prints out unique genome identifiers and debug information. The script is set up for interactive taxonomic level selection and is optimized for publication-quality figures (e.g., for an A4 page).

## Overview

This Python script is used to:
- **Load and process data:** It reads a TSV file with genomic and BGC data.
- **Extract taxonomy:** The script extracts taxonomic information (Phylum, Order, or Genus) from a taxonomy string using simple string manipulation as well as the [Ete3](http://etetoolkit.org/) library for NCBI taxonomy queries.
- **Identify genome IDs:** It creates a new column (`Genome_ID`) based on patterns in the `BGC` field.
- **Calculate proportions:** For a selected taxonomic level, the script calculates the proportion of each BiG-SCAPE class and counts unique genomes.
- **Generate pie charts:** It creates pie charts with external annotations for classes representing low proportions and internal labels for larger slices.
- **Interactive filtering:** The user is prompted to select the taxonomic level (Phylum, Order, or Genus) and input names (comma-separated) for filtering.

---

## Features

- **Data Import:** Reads a TSV file with UTF-8 encoding.
- **Flexible Taxonomy Extraction:** Uses regular expressions and string manipulation to extract species, phylum, and order. For Genus extraction, the script uses Ete3 to query the NCBI database.
- **Custom Genome ID Creation:** Creates a unique identifier for genomes based on the content of the `BGC` field.
- **Interactive Taxonomic Selection:** Allows the user to choose the taxonomic level and filtering criteria at runtime.
- **Proportional Analysis:** Computes proportions for BiG-SCAPE classes and counts unique genome IDs.
- **Publication-Quality Visualization:** Generates pie charts with manual adjustment of annotation positions to avoid overlap.
- **Multiple Output Formats:** Saves the resulting figure in PNG, JPEG, and SVG formats.

---

## Requirements

- **Python 3.x**
- **pandas** – for data manipulation
- **matplotlib** – for plotting (configured to use the TkAgg backend)
- **numpy** – for numerical operations
- **math** – for mathematical calculations
- **re** – for regular expressions
- **ete3** – for accessing NCBI taxonomy information

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CNP-CIIMAR/FAALPred/blob/main/utilities/pie_bgc_class_taxonomy.py
   cd your-repository

## Install the required packages:

```bash
pip install pandas matplotlib numpy ete3
   ```

Usage
Run the script from the command line by passing the path to your TSV file as an argument. For example:

```bash
python3 bgc_class_bigscape.py bigscape_update.tsv

```
# Interactive Steps
•	Taxonomic Level Selection:
The script will prompt you to select the taxonomic level for filtering (Phylum, Order, or Genus) by entering the corresponding number.
•	Input Taxon Names:
After selecting the level, you will be asked to enter the taxon names separated by commas. The script uses these names to filter the dataset and generate pie charts.
________________________________________
# Script Details

## Data Loading
- **Function:** `load_data(file_path)`  
  **Purpose:**  
  - Reads a TSV file using `pandas.read_csv` with UTF-8 encoding.
  - If the file cannot be loaded, an error message is printed and the script terminates.

## Taxonomy Extraction
- **Function:** `extract_species_from_taxonomy(taxonomy)`  
  **Purpose:**  
  - Extracts the species from the taxonomy string by splitting on commas or semicolons and returning the last token.

- **Function:** `extract_taxonomic_group_ete3(taxonomy, level)`  
  **Purpose:**  
  - Uses the Ete3 library to query the NCBI database for a given genus and extract the desired taxonomic level (e.g., genus).

- **Function:** `extract_level_from_taxid(taxid, level)`  
  **Purpose:**  
  - Given an NCBI taxid, it returns the name of the specified taxonomic level from the lineage.

- **Function:** `extract_order(taxonomy, genus)`  
  **Purpose:**  
  - Attempts to extract the "Order" from the taxonomy string.
  - It first searches for tokens ending with "ales" and, if none are found, defaults to the penultimate token or uses an Ete3 lookup if a genus is provided.

- **Function:** `extract_phylum(taxonomy)`  
  **Purpose:**  
  - Extracts the phylum by tokenizing the taxonomy string and skipping tokens that include the term "group".

## Genome ID Extraction
- **Function:** `extract_genome_id(df)`  
  **Purpose:**  
  - Creates a new column (`Genome_ID`) in the DataFrame by parsing the `BGC` column.
  - **Logic includes:**
    - If the value starts with "BGC": splits by a period and takes the first token.
    - If the value starts with "GCA_" or "GCF_": splits by underscore and combines the first two tokens.
    - Otherwise, returns the original value.

## Taxonomic Level Adjustment
- **Function:** `adjust_taxonomic_level(df, level)`  
  **Purpose:**  
  - Depending on the selected taxonomic level (Phylum, Order, or Genus), the function creates or adjusts the corresponding column using the appropriate extraction function.
  - If an unrecognized level is passed, the script exits.

- **Function:** `select_taxonomic_level()`  
  **Purpose:**  
  - Provides an interactive prompt allowing the user to select the taxonomic level and input names (comma-separated) for filtering.

## Proportion Calculation
- **Function:** `calculate_proportion_and_genomes(df, taxon_value, level)`  
  **Purpose:**  
  - Filters the DataFrame based on the provided taxon name and taxonomic level.
    - For Phylum-level filtering, the function searches within the `Taxonomy` column.
  - Computes the count of unique genome IDs and the proportion of each BiG-SCAPE class.
  - Prints debug information including unique genome IDs and total genome counts.

## Visualization
- **Function:** `plot_pie_chart(prop_df, level, taxon_names, num_genomes, color_mapping, ax=None)`  
  **Purpose:**  
  - Generates a pie chart using `matplotlib` to visualize the proportion of BiG-SCAPE classes.
  
  **Large Slices (≥5%):**
  - Displays the percentage inside the slice.
  - Labels are positioned outside without bounding boxes.

  **Small Slices (0.1%–5%):**
  - Annotations (description and percentage) are placed externally with connecting arrows.

  **Annotation Adjustment:**
  - Uses the `adjust_annotations` function to ensure that external annotation labels do not overlap.
  - Configures the title with the taxon name and total number of genomes.

## Main Function
- **Function:** `main()`  
  **Purpose:**  
  - Validates command-line arguments.
  - Loads the data and checks for necessary columns.
  - Calls `extract_genome_id` to generate unique genome identifiers.
  - Invokes interactive selection for the taxonomic level and filtering criteria.
  - Sets up a multi-panel layout for generating pie charts (three charts per row, adjusted for publication quality).
  - Applies a color mapping for BiG-SCAPE classes using a colormap.
  - Generates and saves the pie charts in PNG, JPEG, and SVG formats.
  - Displays the resulting plots.

## Output
The script produces:
- **Pie Charts:**  
  Visual representations of the distribution of BiG-SCAPE classes for each selected taxon.
- **Saved Figures:**  
  The final figures are saved as `pie_charts.png`, `pie_charts.jpeg`, and `pie_charts.svg` with a resolution of 300 DPI.
- **Console Output:**  
  Debug information including confirmation of file loading, creation of the `Genome_ID` column, and lists of unique genome IDs for each taxon.



Script 12: A script to organize antiSMASH directories into BiG-SLiCE datasets, grouped by a user-selected taxonomic level (Phylum, Order, or Genus). organize_bigslice.py
# Organize Big Slice
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


# Script 13:  Pie Chart for Multidomain Architectures.
pie_multidomain_architecture.py

This Python script processes two TSV input tables containing genome and protein signature data to create pie (or doughnut) charts that summarize the distribution of combined signature descriptions across taxonomic groups. It also extracts taxonomic lineage information using the ete3 NCBITaxa module and applies specific regex searches for certain taxon names.

---

## Features

1. **Combine Signature Descriptions:**  
   - Groups signature descriptions for each protein accession.
   - Simplifies descriptions by converting values such as "NRPS" or "PKS" to their abbreviated forms.
   - Replaces "FAAL" with "FAAL stand-alone" when appropriate.
   
2. **Data Loading and Validation:**  
   - Loads two TSV files and prints basic information (column names and shapes).
   - Validates that required columns (e.g., `Assembly` in the first table and `Protein.accession` in both) are present.
   
3. **Merging Tables:**  
   - Merges the two tables on the `Protein.accession` column using an inner join.
   - Generates a new column `Combined.description` in the merged dataframe.
   
4. **Taxonomic Lineage Extraction:**  
   - Uses the ete3 NCBITaxa module to extract taxonomic levels (superkingdom, phylum, class, order, family, genus, species) from a semicolon-separated lineage string.
   - Provides functions to extract the phylum directly.
   
5. **Lineage Update and Filtering:**  
   - Adds taxonomic level columns to the merged dataframe.
   - Filters data by the specified domain (Bacteria, Archaea, or Eukaryota) and removes entries missing key taxonomic levels (phylum, order, genus).
   
6. **Plotting Multidomain Architectures:**  
   - Creates pie charts (formatted as doughnut charts) for each taxon in a given list.
   - Supports special regex-based searches for "Candidatus Rokuibacteriota" and "Gemmatimonadota" within the full lineage.
   - Displays the top N architectures plus an "Others" category for remaining counts.
   - Saves the resulting figures in PNG, SVG, and JPEG formats.
   
7. **Command-Line Interface:**  
   - The script is executed from the command line and requires several parameters (paths to input tables, domain name, top N, taxonomic level, comma-separated taxon list, and DPI for plots).

---

## Requirements

- **Python 3.x**
- **Pandas**
- **Matplotlib**
- **NumPy**
- **ete3** (for NCBITaxa; requires a local NCBI taxonomy database)
  
Install required packages using pip:

```bash
pip install pandas matplotlib numpy ete3
```

```bash

python3 pie_multidomain_architecture.py<table1_path> <table2_path> <domain_name> <top_n> <taxonomic_level> <taxon_list> <dpi>
```

- **`<table1_path>`**: Path to the first TSV file (genome/protein metadata).
- **`<table2_path>`**: Path to the second TSV file (protein signature data).
- **`<domain_name>`**: Must be one of Bacteria, Archaea, or Eukaryota.
- **`<top_n>`**: Number of top signature architectures to display in each chart.
- **`<taxonomic_level>`**: One of Phylum, Order, or Genus.
- **`<taxon_list>`**: A comma-separated list of taxa names to plot.
  - **Special cases:**
    - "Candidatus Rokuibacteriota" applies a regex search pattern `(?i).*candidatus\s+rokuibacteriota.*`
    - "Gemmatimonadota" applies a regex search pattern `(?i).*gemmatimonadota.*`
- **`<dpi>`**: DPI (dots per inch) setting for the output figures.

**Example**

```bash
python3 pie_multidomain_architecture.py \
  Genomes_Total_proteinas_taxonomy_FAAL_metadata_nodup.tsv \
  results_all_lista_proteins.faals_cdd.tsv \
  Bacteria 6 Phylum "Candidatus Rokuibacteriota,Gemmatimonadota,Myxococcota" 300
```
________________________________________

# Code Structure

- **combine_signature_descriptions(df)**
  - Groups and simplifies signature descriptions for each protein accession, creating a `Combined.description` column.

- **load_data(table1_path, table2_path)**
  - Loads the input TSV files into pandas DataFrames, prints column and shape information, and checks for required columns.

- **merge_tables(df1, df2, on='Protein.accession')**
  - Merges the two DataFrames on `Protein.accession` and applies the signature description combination function.

- **Taxonomic Extraction Functions:**
  - **extract_taxonomic_levels(lineage):**  
    Parses a semicolon-separated lineage string and extracts various taxonomic levels.
  - **get_phylum(lineage):**  
    Returns the phylum directly if available.

- **update_lineage(df, domain_name)**
  - Updates the DataFrame with taxonomic levels, filters entries based on the specified domain, and removes rows missing essential taxonomic data.

- **plot_topN_multidomain_in_one_figure(df, taxonomic_level, taxon_list, top_n, dpi)**
  - Generates pie charts (doughnut style) for the specified taxa and saves the figures in multiple formats.

- **main()**
  - Parses command-line arguments.


# License

All codes in this project is licensed under the MIT License. 

Contacts about the codes described above : - mattoslmp@gmail.com
                                           - aliong@ciimar.up.pt
                                           - pleao@ciimar.up.pt

