#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib
import textwrap

# Tenta usar o backend TkAgg para exibição interativa; se não conseguir, usa Agg.
try:
    import tkinter as tk
    matplotlib.use("TkAgg")
except Exception as e:
    print("TkAgg não disponível, usando Agg. A figura não será exibida interativamente.")
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Processa a tabela de BGCs e gera um gráfico com estatísticas."
    )
    parser.add_argument("input_file", 
                        help="Caminho para a tabela de entrada (formato TSV)")
    parser.add_argument("output_base", 
                        help="Nome base para os arquivos de saída (ex.: 'figuraA' gerará figuraA.png e figuraA.svg)")
    args = parser.parse_args()

    # Lê a tabela (formato TSV)
    df = pd.read_csv(args.input_file, delimiter="\t")
    # Substitui strings vazias por NA na coluna "Family Number"
    df["Family Number"] = df["Family Number"].replace("", pd.NA)

    # --- Definição dos grupos ---
    # MIBIG BGCs: registros cuja coluna "BGC" inicia com "BGC"
    df_mibig = df[df["BGC"].str.startswith("BGC", na=False)]
    # Identified BGCs: registros que não iniciam com "BGC" e que possuem Family Number
    df_identified = df[~df["BGC"].str.startswith("BGC", na=False)].copy()
    df_identified = df_identified[df_identified["Family Number"].notna()]
    mibig_families = set(df_mibig["Family Number"].dropna().unique())

    # --- Cálculos para os dados computados ---
    df_new_in_mibig = df_identified[df_identified["Family Number"].isin(mibig_families)]
    new_in_mibig = df_new_in_mibig.shape[0]
    df_new_outside = df_identified[~df_identified["Family Number"].isin(mibig_families)]
    new_outside_mibig = df_new_outside.shape[0]
    total_families = df_identified["Family Number"].nunique()
    singleton_families = (df_identified.groupby("Family Number").size() == 1).sum()
    identified_bgcs = df_identified.shape[0]

    print("New BGCs in MIBIG Families:", new_in_mibig)
    print("New BGCs outside MIBIG Families:", new_outside_mibig)
    print("Total BGCs Families:", total_families)
    print("Singleton BGCs Families:", singleton_families)
    print("Identified BGCs:", identified_bgcs)

    # --- Dados para as barras ---
    # Dados fixos a serem inseridos como primeiras barras
    fixed_pairs = [("MIBIG BGCs", 333), ("MIBIG BGCs with FAAL", 122)]
    # Dados computados
    computed_pairs = list(zip(
        ["New BGCs in MIBIG Families",
         "New BGCs outside MIBIG Families",
         "Total BGCs Families",
         "Singleton BGCs Families",
         "Identified BGCs"],
        [new_in_mibig, new_outside_mibig, total_families, singleton_families, identified_bgcs]
    ))
    # Ordena os dados computados do menor para o maior
    computed_pairs.sort(key=lambda x: x[1])
    # Junta os dados: fixos primeiro e depois os computados
    all_pairs = fixed_pairs + computed_pairs
    all_labels, all_counts = zip(*all_pairs)
    x_positions = np.arange(len(all_labels))
    
    # Quebra os rótulos em duas linhas se necessário (limite de 20 caracteres por linha)
    wrapped_labels = [textwrap.fill(label, width=20) for label in all_labels]

    # --- Criação do gráfico ---
    # Tamanho ideal para publicação: 7 x 5 polegadas
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(all_labels) - 1)) for i in range(len(all_labels))]
    bars = ax.bar(x_positions, all_counts, color=colors)
    ax.set_ylabel("Counts", fontsize=16)
    
    # Define os ticks no centro de cada barra
    centers = [bar.get_x() + bar.get_width()/2 for bar in bars]
    ax.set_xticks(centers)
    ax.set_xticklabels(wrapped_labels, rotation=45, ha="center", rotation_mode="anchor", fontsize=14)
    
    # Ajusta o espaçamento vertical dos rótulos para que fiquem abaixo do gráfico
    ax.tick_params(axis='x', pad=50, labelsize=14)
    # Ajusta a margem inferior para garantir que os rótulos fiquem fora da área do gráfico
    fig.subplots_adjust(bottom=0.85)
    
    # Adiciona os valores sobre cada barra, centralizados com fonte maior
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, str(height),
                ha="center", va="bottom", fontsize=14)
    
    # Salva a figura com 900 dpi, em PNG e SVG, com alta resolução
    png_file = args.output_base + ".png"
    svg_file = args.output_base + ".svg"
    plt.savefig(png_file, dpi=900, bbox_inches="tight")
    plt.savefig(svg_file, bbox_inches="tight")
    print("Gráfico salvo em:", png_file, "e", svg_file)
    plt.show()

if __name__ == "__main__":
    main()
