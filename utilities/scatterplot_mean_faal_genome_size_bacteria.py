import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ete3 import NCBITaxa
import re
import sys

ncbi = NCBITaxa()

# --- Genus extraction using ete3, replicating the reference logic ---
def extract_taxonomic_group_by_ete3(species_name, target_rank):
    try:
        # Try species, then fallback to genus
        tx = ncbi.get_name_translator([species_name])
        if not tx:
            tx = ncbi.get_name_translator([species_name.split()[0]])
        if not tx:
            return None
        taxid = list(tx.values())[0][0]
        lineage = ncbi.get_lineage(taxid)
        ranks   = ncbi.get_rank(lineage)
        names   = ncbi.get_taxid_translator(lineage)
        for tid in lineage:
            if ranks.get(tid, '').lower() == target_rank.lower():
                nome = names[tid]
                if target_rank.lower() == 'genus' and (nome.lower().endswith('ales') or nome.lower().endswith('eae')):
                    continue
                return nome
        return None
    except Exception as e:
        print(f"Erro ao extrair {target_rank} de '{species_name}': {e}")
        return None

# --- Original helper functions ---
def extract_taxonomic_group_ete3(taxid, level):
    lineage = ncbi.get_lineage(taxid)
    ranks   = ncbi.get_rank(lineage)
    names   = ncbi.get_taxid_translator(lineage)
    target  = level.lower()
    for tid in lineage:
        if ranks.get(tid, '').lower() == target:
            name = names.get(tid)
            if target == 'genus':
                if not name or name.endswith('ales') or name.endswith('eae') or name.startswith('Candidatus'):
                    continue
            return name
    return None

def extract_phylum_ete3(species_name):
    genus = species_name.split()[0]
    taxid = ncbi.get_name_translator([genus]).get(genus, [None])[0]
    return extract_taxonomic_group_ete3(taxid, 'phylum') if taxid else None

def extract_order_ete3(species_name):
    genus = species_name.split()[0]
    taxid = ncbi.get_name_translator([genus]).get(genus, [None])[0]
    return extract_taxonomic_group_ete3(taxid, 'order') if taxid else None

def extract_family_ete3(species_name):
    genus = species_name.split()[0]
    taxid = ncbi.get_name_translator([genus]).get(genus, [None])[0]
    return extract_taxonomic_group_ete3(taxid, 'family') if taxid else None

def extract_taxonomic_group(lineage, level):
    levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    tokens = [tok.strip() for tok in lineage.split(';') if tok.strip()]
    if level in levels:
        idx = levels.index(level)
        return tokens[idx] if idx < len(tokens) else None
    return None

def filter_by_criteria_ete3(name, level):
    if not name: return False
    if level == 'Order':  return name.endswith('ales')
    if level == 'Family': return name.endswith('eae')
    if level == 'Genus':  return not (name.endswith('ales') or name.endswith('eae'))
    if level == 'Phylum': return True
    return False

def filter_by_criteria(name, level, domain):
    if not name: return False
    if level == 'Order':  return name.endswith('ales')
    if level == 'Family': return name.endswith('eae')
    if level == 'Genus':  return not (name.endswith('ales') or name.endswith('eae'))
    if level in ['Phylum', 'Domain']: return True
    return False

def get_lineage_from_ncbi(species_name):
    genus = species_name.split()[0]
    taxid = ncbi.get_name_translator([genus]).get(genus, [None])[0]
    if not taxid:
        return None
    lineage_ids = ncbi.get_lineage(taxid)
    names = ncbi.get_taxid_translator(lineage_ids)
    return '; '.join(names[tid] for tid in lineage_ids)

def update_lineage(df):
    df['Lineage'] = df['Species'].apply(get_lineage_from_ncbi)
    return df

def generate_panel(table1_path, domain_name, taxonomic_levels, top_n, dpi):
    # Load
    df = pd.read_csv(table1_path, sep='\t', low_memory=False)

    # Required columns
    required_columns = [
        'Species',
        'Lineage',
        'Assembly',
        'Assembly Stats Total Sequence Length MB'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    # Correct lineage for Eukaryota
    if domain_name == 'Eukaryota':
        df = update_lineage(df)

    # Base filters
    df = df[~df['Lineage'].str.contains('environmental', na=False)]
    df = df.dropna(subset=['Assembly'])
    df = df[df['Lineage'].str.contains(domain_name, na=False)]
    df = df[df['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]

    # Plot setup
    fig, axes = plt.subplots(
        1,
        len(taxonomic_levels),
        figsize=(6 * len(taxonomic_levels), 8)
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    ticks = np.arange(0, 12, 2)

    for ax, level in zip(axes, taxonomic_levels):
        df_lvl = df.copy()

        # Genus uses new extraction
        if level == 'Genus':
            df_lvl['Taxonomic_Group'] = (
                df_lvl['Species']
                .apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Genus'))
            )
            df_lvl = df_lvl[df_lvl['Taxonomic_Group'].apply(
                lambda x: filter_by_criteria_ete3(x, 'Genus')
            )]

        # Eukaryota phylum/order/family via ete3
        elif domain_name == 'Eukaryota' and level in ['Phylum', 'Order', 'Family']:
            func = {
                'Phylum': extract_phylum_ete3,
                'Order':  extract_order_ete3,
                'Family': extract_family_ete3
            }[level]
            df_lvl['Taxonomic_Group'] = df_lvl['Species'].apply(func)
            df_lvl = df_lvl[df_lvl['Taxonomic_Group'].apply(
                lambda x: filter_by_criteria_ete3(x, level)
            )]

        # Other ranks from existing lineage
        else:
            df_lvl['Taxonomic_Group'] = (
                df_lvl['Lineage']
                .apply(lambda x: extract_taxonomic_group(x, level))
            )
            df_lvl = df_lvl[df_lvl['Taxonomic_Group'].apply(
                lambda x: filter_by_criteria(x, level, domain_name)
            )]

        # Drop NaN groups
        df_lvl = df_lvl[df_lvl['Taxonomic_Group'].notna()]
        if df_lvl.empty:
            ax.text(
                0.5, 0.5,
                f"Nenhum grupo encontrado para {level}",
                ha='center', va='center', transform=ax.transAxes
            )
            continue

        # Compute FAAL counts and genome counts
        faal = (
            df_lvl.groupby('Taxonomic_Group')
            .size()
            .reset_index(name='Total FAAL Count')
        )
        genomes = (
            df_lvl.groupby('Taxonomic_Group')['Assembly']
            .nunique()
            .reset_index(name='Genome Count')
        )
        data = pd.merge(faal, genomes, on='Taxonomic_Group')
        cutoff = 0 if domain_name == 'Eukaryota' else 5
        data = data[data['Genome Count'] >= cutoff]
        if data.empty:
            ax.text(
                0.5, 0.5,
                f"Sem grupos suficientes para {level}",
                ha='center', va='center', transform=ax.transAxes
            )
            continue
        data['Mean FAAL Count per Genome'] = (
            data['Total FAAL Count'] / data['Genome Count']
        )

        # Average genome size
        df_lvl['Genome Size'] = pd.to_numeric(
            df_lvl['Assembly Stats Total Sequence Length MB']
            .astype(str)
            .str.replace(',', ''),
            errors='coerce'
        )
        size = (
            df_lvl.groupby('Taxonomic_Group')['Genome Size']
            .mean()
            .reset_index()
        )

        # Select top-N by mean FAAL/genome
        top = (
            pd.merge(data, size, on='Taxonomic_Group')
            .nlargest(top_n, 'Mean FAAL Count per Genome')
        )
        # --- Padroniza nomes de colunas removendo espaços ---
        top = top.rename(columns=lambda x: x.replace(' ', '_'))

        # Save per-level table
        table_filename = (
            f'taxonomic_mean_faal_and_genome_size_{level}.tsv'
        )
        top.to_csv(table_filename, sep='\t', index=False)
        print(f"Tabela '{table_filename}' salva.")

        # Scatter plot
        ax.scatter(
            top['Genome_Size'],
            top['Mean_FAAL_Count_per_Genome'],
            s=150,
            color=(0.7, 0.7, 0.7),
            edgecolor='black',
            zorder=3
        )

        # — Top 3 labels acima das esferas (agora sempre correto) —
        top3 = top.nlargest(3, 'Mean_FAAL_Count_per_Genome')
        for r in top3.itertuples():
            gx = r.Genome_Size
            gy = r.Mean_FAAL_Count_per_Genome
            if pd.notna(gx) and pd.notna(gy):
                ax.annotate(
                    str(r.Taxonomic_Group).strip(),
                    (gx, gy),
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontweight='bold',
                    ha='center',
                    va='bottom'
                )

        # Axes formatting
        xmax = top['Genome_Size'].max()
        ymax = top['Mean_FAAL_Count_per_Genome'].max()
        # Ajuste dinâmico para Genus: aumenta um pouco o range se for Genus
        if level == 'Genus':
            ax.set_xlim(0, max(xmax * 1.2, 10))
            ax.set_ylim(0, max(ymax * 1.2, 10))
        else:
            ax.set_xlim(0, max(xmax * 1.1, 10))
            ax.set_ylim(0, max(ymax * 1.1, 10))
        ax.set_xlabel(
            'Average Genome Size (MB)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_ylabel(
            'Mean FAAL Count per Genome',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(['' if t == 0 else t for t in ticks])
        ax.set_yticklabels(['' if t == 0 else t for t in ticks])
        ax.grid(False)

    plt.tight_layout()
    out_prefix = (
        f'scatterplot_panel_faal_vs_genome_{domain_name}_{"-".join(taxonomic_levels)}'
    )
    for ext in ['png', 'svg', 'jpeg']:
        plt.savefig(
            f'{out_prefix}.{ext}',
            dpi=dpi,
            bbox_inches='tight'
        )
    plt.show()

# --- Script entry point with proper argument parsing ---
if __name__ == '__main__':
    if len(sys.argv) != 6:
        print(
            "Uso: python3 scatterplot_counts_faalvf_codev2.py "
            "<table1.tsv> <Domain> <Levels(comma)> <Top N> <DPI>"
        )
        sys.exit(1)
    table1_path     = sys.argv[1]
    domain_name     = sys.argv[2]
    taxonomic_levels = [lvl.strip() for lvl in sys.argv[3].split(',')]
    top_n           = int(sys.argv[4])
    dpi             = int(sys.argv[5])
    generate_panel(
        table1_path,
        domain_name,
        taxonomic_levels,
        top_n,
        dpi
    )


