#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import shutil
import urllib.request
import tarfile
import hashlib
import platform
import re
from Bio import SeqIO
from pathlib import Path

def echo_info(message):
    print(f"\033[92m[INFO]\033[0m {message}")

def echo_warn(message):
    print(f"\033[93m[WARNING]\033[0m {message}")

def echo_error(message):
    print(f"\033[91m[ERROR]\033[0m {message}")

def command_exists(command):
    return shutil.which(command) is not None

def detect_package_manager():
    if command_exists("apt-get"):
        return "apt-get"
    elif command_exists("yum"):
        return "yum"
    else:
        return None

def install_packages(packages, package_manager):
    if package_manager == "apt-get":
        cmd_update = ["sudo", "apt-get", "update"]
        cmd_install = ["sudo", "apt-get", "install", "-y"] + packages
    elif package_manager == "yum":
        cmd_update = ["sudo", "yum", "makecache"]
        cmd_install = ["sudo", "yum", "install", "-y"] + packages
    else:
        echo_error("Gerenciador de pacotes não suportado. Por favor, instale as dependências manualmente.")
        sys.exit(1)
    
    try:
        echo_info(f"Atualizando lista de pacotes usando {package_manager}...")
        subprocess.run(cmd_update, check=True)
        echo_info(f"Instalando pacotes: {' '.join(packages)}...")
        subprocess.run(cmd_install, check=True)
    except subprocess.CalledProcessError:
        echo_error("Falha ao instalar pacotes. Verifique as mensagens de erro acima.")
        sys.exit(1)

def get_java_home():
    try:
        java_path = shutil.which("java")
        if not java_path:
            return None
        java_real_path = os.path.realpath(java_path)
        java_home = os.path.dirname(os.path.dirname(java_real_path))
        return java_home
    except Exception:
        return None

def append_to_bashrc(line):
    bashrc = Path.home() / ".bashrc"
    with open(bashrc, "r") as file:
        content = file.read()
    if line not in content:
        with open(bashrc, "a") as file:
            file.write(f"\n{line}\n")
        echo_info(f"Linha adicionada ao {bashrc}")
    else:
        echo_info(f"Linha já presente no {bashrc}")

def download_file(url, dest):
    try:
        echo_info(f"Baixando {url}...")
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        echo_error(f"Falha ao baixar {url}: {e}")
        sys.exit(1)

def verify_md5(file_path, md5_file_path):
    try:
        with open(md5_file_path, 'r') as f:
            md5_expected = f.read().split()[0]
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        md5_calculated = md5_hash.hexdigest()
        if md5_calculated == md5_expected:
            echo_info("Verificação MD5 bem-sucedida.")
        else:
            echo_error("Verificação MD5 falhou. O arquivo pode estar corrompido.")
            sys.exit(1)
    except Exception as e:
        echo_error(f"Erro ao verificar MD5: {e}")
        sys.exit(1)

def extract_tar_gz(file_path, extract_to):
    try:
        echo_info(f"Extraindo {file_path} para {extract_to}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    except Exception as e:
        echo_error(f"Falha ao extrair {file_path}: {e}")
        sys.exit(1)

def run_command(command, cwd=None, capture_output=False):
    try:
        if capture_output:
            result = subprocess.run(command, check=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout, result.stderr
        else:
            subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        echo_error(f"Falha ao executar comando: {' '.join(command)}")
        if capture_output:
            echo_error(f"stdout: {e.stdout}")
            echo_error(f"stderr: {e.stderr}")
        sys.exit(1)

def parse_java_version(java_version_output):
    """
    Extrai a versão do Java a partir da saída de 'java -version'
    Retorna o número da versão principal como inteiro.
    """
    # Exemplos de saídas:
    # openjdk version "11.0.4" 2019-07-16
    # openjdk version "17.0.1" 2021-10-19
    # java version "1.8.0_275"
    match = re.search(r'version "(.*?)"', java_version_output)
    if match:
        version_str = match.group(1)
        # Remove possíveis sufixos, como +1, _build, etc.
        version_clean = re.split(r'[+\-]', version_str)[0]
        version_parts = version_clean.split('.')
        try:
            major = int(version_parts[0])
            # Para versões como "1.8.0_275"
            if major == 1 and len(version_parts) > 1:
                major = int(version_parts[1])
            return major
        except ValueError:
            return 0
    return 0

def map_fasta_headers(fasta_path):
    """Lê o arquivo FASTA de entrada e cria um dicionário com IDs e seus respectivos cabeçalhos completos."""
    headers_map = {}
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            headers_map[record.id] = record.description
    return headers_map

def replace_fasta_headers(input_fasta, output_fasta, headers_map):
    """Substitui os cabeçalhos das sequências de saída pelos cabeçalhos originais do FASTA de entrada."""
    updated_records = []
    with open(output_fasta, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            original_header = headers_map.get(record.id)
            if original_header:
                record.description = original_header
            updated_records.append(record)
    
    # Escrever as sequências atualizadas em um novo arquivo FASTA
    with open(output_fasta, "w") as output_file:
        SeqIO.write(updated_records, output_file, "fasta")
    echo_info(f"Cabeçalhos do FASTA de entrada aplicados ao arquivo de saída: {output_fasta}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Script para configurar e executar o InterProScan.")
    parser.add_argument("fasta", help="Arquivo FASTA de entrada.")
    parser.add_argument("-d", "--interproscan_dir", default="interproscan", help="Diretório de instalação do InterProScan (padrão: ./interproscan).")
    args = parser.parse_args()

    input_fasta = args.fasta
    interproscan_dir = Path(args.interproscan_dir).resolve()

    if not os.path.isfile(input_fasta):
        echo_error(f"O arquivo FASTA '{input_fasta}' não existe.")
        sys.exit(1)
    # Ler cabeçalhos do FASTA de entrada
    headers_map = map_fasta_headers(input_fasta)
    echo_info("Cabeçalhos do FASTA de entrada mapeados.")
    
    # Verificar arquitetura do sistema
    echo_info("Verificando a arquitetura do sistema...")
    arch = platform.machine()
    if arch != "x86_64":
        echo_error(f"InterProScan requer um sistema de 64-bit. Arquitetura detectada: {arch}")
        sys.exit(1)
    else:
        echo_info(f"Arquitetura verificada: {arch} (64-bit)")

    # Detectar gerenciador de pacotes
    package_manager = detect_package_manager()
    if not package_manager:
        echo_error("Gerenciador de pacotes não suportado. Por favor, instale as dependências manualmente.")
        sys.exit(1)

    # Verificar e instalar Perl 5
    echo_info("Verificando instalação do Perl...")
    if command_exists("perl"):
        perl_version_output = subprocess.getoutput("perl -v | grep 'This is perl'")
        perl_version_match = re.search(r'version\s+(\S+)', perl_version_output)
        if perl_version_match:
            perl_version = perl_version_match.group(1)
            echo_info(f"Perl encontrado, versão {perl_version}")
        else:
            echo_warn("Perl encontrado, mas não foi possível determinar a versão.")
    else:
        echo_warn("Perl não encontrado. Instalando Perl...")
        install_packages(["perl"], package_manager)

    # Verificar e instalar Python 3
    echo_info("Verificando instalação do Python 3...")
    if command_exists("python3"):
        python_version = subprocess.getoutput("python3 --version")
        echo_info(f"{python_version} encontrado")
    else:
        echo_warn("Python 3 não encontrado. Instalando Python 3...")
        install_packages(["python3"], package_manager)

    # Verificar e instalar Java 11 ou superior
    echo_info("Verificando instalação do Java...")
    java_installed = False
    java_major_version = 0
    if command_exists("java"):
        java_version_output = subprocess.getoutput("java -version 2>&1")
        java_major_version = parse_java_version(java_version_output)
        echo_info(f"Versão do Java encontrada: {java_major_version}")
        if java_major_version >= 11:
            echo_info(f"Java {java_major_version} já está instalado.")
            java_installed = True
        else:
            echo_warn("Java instalado não é a versão 11 ou superior. Instalando OpenJDK 11...")
    else:
        echo_warn("Java não encontrado. Instalando OpenJDK 11...")

    if not java_installed:
        if package_manager == "apt-get":
            install_packages(["openjdk-11-jdk"], package_manager)
        elif package_manager == "yum":
            install_packages(["java-11-openjdk-devel"], package_manager)
        
        # Verificar novamente
        if command_exists("java"):
            java_version_output = subprocess.getoutput("java -version 2>&1")
            java_major_version = parse_java_version(java_version_output)
            if java_major_version >= 11:
                echo_info(f"Java {java_major_version} instalado com sucesso.")
                java_installed = True
            else:
                echo_error("Falha ao instalar Java 11 ou superior.")
                sys.exit(1)
        else:
            echo_error("Falha ao instalar Java 11 ou superior.")
            sys.exit(1)

    # Configurar JAVA_HOME e PATH
    echo_info("Configurando JAVA_HOME e atualizando PATH...")
    java_home = get_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"
        # Adicionar ao ~/.bashrc se ainda não estiver
        bashrc_line = f"export JAVA_HOME={java_home}"
        path_line = "export PATH=$JAVA_HOME/bin:$PATH"
        append_to_bashrc(bashrc_line)
        append_to_bashrc(path_line)
    else:
        echo_warn("Não foi possível determinar JAVA_HOME automaticamente.")

    # Verificar e instalar bedtools
    echo_info("Verificando instalação do bedtools...")
    if command_exists("bedtools"):
        bedtools_version = subprocess.getoutput("bedtools --version")
        echo_info(f"bedtools encontrado, versão {bedtools_version}")
    else:
        echo_warn("bedtools não encontrado. Instalando bedtools...")
        install_packages(["bedtools"], package_manager)

    # Definir a versão do InterProScan
    interproscan_version = "5.70-102.0"
    interproscan_archive = f"interproscan-{interproscan_version}-64-bit.tar.gz"
    interproscan_url = f"https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/{interproscan_version}/{interproscan_archive}"
    interproscan_md5_url = f"{interproscan_url}.md5"

    # Definir o caminho esperado do InterProScan
    interproscan_path = interproscan_dir / f"interproscan-{interproscan_version}"

    # Verificar se InterProScan já está instalado no diretório especificado
    if interproscan_path.exists():
        echo_info(f"InterProScan já está instalado em {interproscan_path}")
    else:
        # Verificar se interproscan.sh está presente no diretório atual
        current_dir = Path.cwd()
        expected_sh = current_dir / "interproscan.sh"
        if expected_sh.exists():
            echo_info("InterProScan encontrado no diretório atual. Usando a instalação existente.")
            interproscan_path = current_dir
        else:
            echo_info("InterProScan não encontrado. Instalando InterProScan...")
            interproscan_dir.mkdir(parents=True, exist_ok=True)
            
            # Baixar InterProScan e o arquivo MD5
            archive_path = interproscan_dir / interproscan_archive
            md5_path = interproscan_dir / f"{interproscan_archive}.md5"
            
            download_file(interproscan_url, archive_path)
            download_file(interproscan_md5_url, md5_path)
            
            # Verificar MD5
            verify_md5(archive_path, md5_path)
            
            # Extrair o tar.gz
            extract_tar_gz(archive_path, interproscan_dir)
            
            # Remover arquivos baixados
            archive_path.unlink()
            md5_path.unlink()
            
            echo_info(f"InterProScan instalado em {interproscan_path}")

    # Adicionar InterProScan ao PATH
    # Verificar onde está 'interproscan.sh'
    interproscan_sh = interproscan_path / "interproscan.sh"
    if interproscan_sh.exists():
        interproscan_bin = interproscan_path
    else:
        # Se 'interproscan.sh' não estiver diretamente no interproscan_path, procurar em 'bin/'
        interproscan_bin = interproscan_path / "bin"
        if not (interproscan_bin / "interproscan.sh").exists():
            echo_error(f"'interproscan.sh' não encontrado em {interproscan_path} ou {interproscan_bin}.")
            sys.exit(1)
    
    os.environ["PATH"] = f"{interproscan_bin}:{os.environ['PATH']}"
    
    # Adicionar ao ~/.bashrc se ainda não estiver
    interproscan_bin_str = str(interproscan_bin)
    bashrc_line_ipr = f"export PATH={interproscan_bin_str}:$PATH"
    append_to_bashrc(bashrc_line_ipr)

    # Verificar se InterProScan está acessível
    if not command_exists("interproscan.sh"):
        echo_error("InterProScan não está no PATH. Verifique a instalação.")
        sys.exit(1)

    # Executar InterProScan
    echo_info(f"Executando InterProScan no arquivo {input_fasta}...")
    output_dir = interproscan_path / "interproscan_output"
    output_dir.mkdir(exist_ok=True)

    # Definir comando
    interproscan_cmd = [
        "interproscan.sh",
        "-i", str(Path(input_fasta).resolve()),
        "-f", "tsv",
        "-dp",
        "-appl", "CDD",
        "-o", str(output_dir / "results.tsv")
    ]

    run_command(interproscan_cmd, cwd=interproscan_path)

    echo_info("InterProScan concluído. Processando resultados...")

    results_tsv = output_dir / "results.tsv"
    faal_tsv = output_dir / "FAAL.interpro.tsv"

    # Extrair linhas contendo "FAAL"
    try:
        with open(results_tsv, 'r') as infile, open(faal_tsv, 'w') as outfile:
            for line in infile:
                if "FAAL" in line:
                    outfile.write(line)
        echo_info(f"Linhas contendo 'FAAL' extraídas para {faal_tsv}")
    except Exception as e:
        echo_error(f"Erro ao processar resultados: {e}")
        sys.exit(1)

    # Verificar se o grep encontrou resultados
    if not faal_tsv.exists() or os.path.getsize(faal_tsv) == 0:
        echo_warn("Nenhum domínio FAAL encontrado no arquivo de resultados.")
        sys.exit(0)

    # Criar arquivo BED a partir das colunas 1,7,8
    faal_bed = output_dir / "FAAL.interpro.bed"
    try:
        with open(faal_tsv, 'r') as infile, open(faal_bed, 'w') as outfile:
            for line in infile:
                parts = line.strip().split('\t')
                if len(parts) >= 8:
                    chrom = parts[0]
                    start = parts[6]
                    end = parts[7]
                    outfile.write(f"{chrom}\t{start}\t{end}\n")
        echo_info(f"Arquivo BED criado em {faal_bed}")
    except Exception as e:
        echo_error(f"Erro ao criar arquivo BED: {e}")
        sys.exit(1)

    # Extrair as sequências FAAL usando bedtools
    output_fasta = output_dir / f"{Path(input_fasta).stem}_FAAL.fasta"
    try:
        bedtools_cmd = [
            "bedtools",
            "getfasta",
            "-fi", str(Path(input_fasta).resolve()),
            "-bed", str(faal_bed),
            "-fo", str(output_fasta)
        ]
        stdout, stderr = run_command(bedtools_cmd, capture_output=True)
        if stdout:
            echo_info(f"bedtools output:\n{stdout}")
        if stderr:
            echo_warn(f"bedtools warnings/errors:\n{stderr}")
        echo_info(f"Sequências FAAL extraídas para {output_fasta}")
    except Exception as e:
        echo_error(f"Erro ao extrair sequências FAAL: {e}")
        sys.exit(1)
    # Substituir os cabeçalhos das sequências de saída
    
    replace_fasta_headers(input_fasta, output_fasta, headers_map)
    
    echo_info("Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
