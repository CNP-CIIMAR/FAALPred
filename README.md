
title: "Run InterProScan Automation Script to get FAAL -domain of FAAL fasta file sequences - FAALpred_prepare.py"
- version: "1.0.0"
- date: "2024-10-10"
# description: 
  - This Python script automates the setup and execution of InterProScan for analyzing FASTA files.
  - It ensures that all necessary dependencies are installed, configures environment variables,
    and processes the results to extract FAAL domains.

## features:
  - "Automatically checks and installs dependencies: Perl, Python 3, Java (11+), and bedtools."
  - "Detects existing InterProScan installation to avoid redundant installations."
  - "Configures JAVA_HOME and updates PATH environment variables."
  - "Executes InterProScan on a provided FASTA file and processes results."
  - "Extracts FAAL domains and generates BED and FASTA files for further analysis."

## requirements:
  - "Operating System: Linux (64-bit)"
  - "Python 3.6 or higher"
  - "Internet connection for downloading InterProScan and dependencies."
  - "Sudo privileges for installing system packages."

## installation:
  - steps:
    - "Clone the repository or download the `run_interproscan.py` script."
    - "Ensure the script has execute permissions (optional):"
      ```bash
      chmod +x run_interproscan.py
      ```
    - "Install necessary Python packages (if any) using pip (optional):"
    - |
      ```bash
      pip3 install -r requirements.txt
      ```

# usage:
  - description: 
  - Run the script by providing the path to your FASTA file. Optionally, specify the InterProScan installation directory.
# commands:
    - "Basic usage with default InterProScan directory (`./interproscan`):"
      ```bash
      python3 run_interproscan.py your_file.fasta
      ```
    - "Specify a custom InterProScan installation directory:"
      ```bash
      python3 run_interproscan.py your_file.fasta -d /path/to/interproscan
      ```
  notes:
    - "The script may prompt for your sudo password to install missing dependencies."
    - "Ensure that `interproscan.sh` is present in the specified directory or the current directory."

# configuration:
  - arguments:
     fasta:
      - description: "Path to the input FASTA file."
      - required: true
      - type: "string"
    -d, --interproscan_dir:
      - description: "Directory for InterProScan installation."
      - required: false
      - default: "./interproscan"
      - type: "string"

# output:
  - description: 
    - After successful execution, the script generates the following:
  files:
    - "interproscan_output/results.tsv: The raw InterProScan results in TSV format."
    - "interproscan_output/FAAL.interpro.tsv: Filtered results containing FAAL domains."
    - "interproscan_output/FAAL.interpro.bed: BED file with FAAL domain coordinates."
    - "interproscan_output/faal_FAAL.fasta: FASTA file containing extracted FAAL domain sequences."

## troubleshooting:
  # common_issues:
    - issue: "UnboundLocalError: local variable 'interproscan_version' referenced before assignment"
    -solution: 
        - Ensure that the `interproscan_version` variable is defined before it's used in the script.
        - Update the script to define `interproscan_version` early in the main function.
    - issue: "InterProScan not found in PATH."
    - solution: 
        - Verify that `interproscan.sh` is located in the specified InterProScan directory or the current directory.
        - Ensure the script has added the correct directory to the PATH and reload the shell configuration:
        ```bash
        source ~/.bashrc
        ```
    - issue: "Java version not detected correctly."
      # solution: 
        # Ensure that Java 11 or higher is installed and accessible. You can check the Java version with:
        ```bash
        java -version
        ```
        - If a newer version is installed, the script should recognize it. If not, reinstall Java 11.

# FAALPred

## contributing:
  # description: 
    # Contributions are welcome! Please follow these steps to contribute:
  steps:
    - "Fork the repository."
    - "Create a new branch for your feature or bugfix."
    - "Commit your changes with clear messages."
    - "Push your branch to your forked repository."
    - "Submit a pull request detailing your changes."

license:
  name: "MIT License"
  url: "https://opensource.org/licenses/MIT"

contact:
  name: "Leandro de Mattos Pereira"
  email: "mattoslmp@gmail.com"
  github: "https://github.com/CNP-CIIMAR/FAALPredClassifier/"
---
