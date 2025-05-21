import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ete3 import NCBITaxa
import sys

ncbi = NCBITaxa()

FIG_HEIGHT = 4.3

def extract_taxonomic_group_by_ete3(species_name, target_rank):
    try:
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

def clean_int_labels(vals):
    return [str(int(v)) if float(v).is_integer() else str(v) for v in vals]

def prepare_plot_data(df, domain_name, configs):
    all_tops = []
    xlims = []
    for conf in configs:
        level, top_n = conf['level'], conf['top_n']
        df_lvl = df.copy()

        if level == 'Genus':
            df_lvl['Taxonomic_Group'] = (
                df_lvl['Species'].apply(lambda s: extract_taxonomic_group_by_ete3(s, 'Genus'))
            )
            df_lvl = df_lvl[df_lvl['Taxonomic_Group'].apply(
                lambda x: filter_by_criteria_ete3(x, 'Genus')
            )]
        elif domain_name == 'Eukaryota' and level in ['Phylum', 'Order', 'Family']:
            func = {'Phylum': extract_phylum_ete3, 'Order': extract_order_ete3, 'Family': extract_family_ete3}[level]
            df_lvl['Taxonomic_Group'] = df_lvl['Species'].apply(func)
            df_lvl = df_lvl[df_lvl['Taxonomic_Group'].apply(
                lambda x: filter_by_criteria_ete3(x, level)
            )]
        else:
            df_lvl['Taxonomic_Group'] = (
                df_lvl['Lineage'].apply(lambda x: extract_taxonomic_group(x, level))
            )
            df_lvl = df_lvl[df_lvl['Taxonomic_Group'].apply(
                lambda x: filter_by_criteria(x, level, domain_name)
            )]

        df_lvl = df_lvl[df_lvl['Taxonomic_Group'].notna()]
        faal = (
            df_lvl.groupby('Taxonomic_Group')
            .size().reset_index(name='Total FAAL Count')
        )
        genomes = (
            df_lvl.groupby('Taxonomic_Group')['Assembly']
            .nunique().reset_index(name='Genome Count')
        )
        data = pd.merge(faal, genomes, on='Taxonomic_Group')
        cutoff = 0 if domain_name == 'Eukaryota' else 5
        data = data[data['Genome Count'] >= cutoff]
        if data.empty:
            all_tops.append(None)
            xlims.append((0,10))
            continue
        data['Average FAALs counts'] = data['Total FAAL Count'] / data['Genome Count']
        df_lvl['Genome Size'] = pd.to_numeric(
            df_lvl['Assembly Stats Total Sequence Length MB'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        size = (
            df_lvl.groupby('Taxonomic_Group')['Genome Size']
            .mean().reset_index()
        )
        top = (
            pd.merge(data, size, on='Taxonomic_Group')
            .nlargest(top_n, 'Average FAALs counts')
        )
        top = top.rename(columns=lambda x: x.replace(' ', '_'))
        all_tops.append(top)
        this_xmax = top['Genome_Size'].max() if not top.empty else 10
        xlims.append((0, max(this_xmax*1.2, 10)))
    return all_tops, xlims

def plot_single(
    top, level, top_n, dpi, output_prefix,
    xlim, ylim,
    show_panel=False, ax=None,
    jitter_strength=0.05
):
    if top is None or top.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, FIG_HEIGHT))
        ax.text(0.5, 0.5, f"Nenhum grupo encontrado para {level}",
                ha='center', va='center', transform=ax.transAxes,
                fontweight='bold', color='black', fontsize=14)
        if not show_panel:
            plt.tight_layout()
            for ext in ['png', 'svg', 'jpeg']:
                plt.savefig(f"{output_prefix}.{ext}", dpi=dpi, bbox_inches='tight')
            plt.close()
        return ax

    table_filename = f'{output_prefix}_table.tsv'
    top.to_csv(table_filename, sep='\t', index=False)
    print(f"Tabela '{table_filename}' salva.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, FIG_HEIGHT))

    np.random.seed(42)
    jitter_x = np.random.uniform(-jitter_strength, jitter_strength, size=len(top))
    jitter_y = np.random.uniform(-jitter_strength, jitter_strength, size=len(top))
    plot_x = top['Genome_Size'].values + jitter_x
    plot_y = top['Average_FAALs_counts'].values + jitter_y

    ax.scatter(
        plot_x,
        plot_y,
        s=120, color=(0.7, 0.7, 0.7), edgecolor='black', zorder=3,
        alpha=0.75, linewidth=1.5
    )

    top3 = top.nlargest(3, 'Average_FAALs_counts')
    for idx, r in enumerate(top3.itertuples()):
        gx = r.Genome_Size + jitter_x[idx]
        gy = r.Average_FAALs_counts + jitter_y[idx]
        if pd.notna(gx) and pd.notna(gy):
            ax.annotate(
                str(r.Taxonomic_Group).strip(),
                (gx, gy),
                xytext=(0, 5),
                textcoords='offset points',
                fontweight='bold',
                color='black',
                ha='center',
                va='bottom',
                fontsize=12
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ticks_x = np.arange(xlim[0], xlim[1]+1, 2)
    ticks_y = np.arange(ylim[0], ylim[1]+1, 2)
    ax.set_xlabel('Average Genome Size (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average FAALs counts', fontsize=14, fontweight='bold', color='black')
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(clean_int_labels(ticks_x), fontweight='bold', color='black')
    ax.set_yticklabels(clean_int_labels(ticks_y), fontweight='bold', color='black')
    ax.tick_params(axis='x', colors='black', labelsize=12)
    ax.tick_params(axis='y', colors='black', labelsize=12)
    ax.grid(False)
    if not show_panel:
        plt.tight_layout()
        for ext in ['png', 'svg', 'jpeg']:
            plt.savefig(f"{output_prefix}.{ext}", dpi=dpi, bbox_inches='tight')
        plt.close()
    return ax

def generate_panel(table1_path, domain_name, configs):
    df = pd.read_csv(table1_path, sep='\t', low_memory=False)

    required_columns = [
        'Species',
        'Lineage',
        'Assembly',
        'Assembly Stats Total Sequence Length MB'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    if domain_name == 'Eukaryota':
        df = update_lineage(df)

    df = df[~df['Lineage'].str.contains('environmental', na=False)]
    df = df.dropna(subset=['Assembly'])
    df = df[df['Lineage'].str.contains(domain_name, na=False)]
    df = df[df['Assembly'].str.startswith(('GCF', 'GCA'), na=False)]

    # Prepara dados e limites individuais
    tops, xlims = prepare_plot_data(df, domain_name, configs)
    # Y máximo global para todos os gráficos
    all_ymax = [t['Average_FAALs_counts'].max() for t in tops if t is not None and not t.empty]
    ymax = max(all_ymax) if all_ymax else 10
    ylim = (0, max(ymax * 1.2, 10))

    # Painel: soma das larguras individuais
    figwidth_panel = sum([max(8, 8*(x[1]/10)) for x in xlims])
    fig, axes = plt.subplots(
        1, len(configs),
        figsize=(figwidth_panel, FIG_HEIGHT),
        squeeze=False
    )
    axes = axes[0]

    for idx, (conf, top, xlim) in enumerate(zip(configs, tops, xlims)):
        level, top_n, dpi = conf['level'], conf['top_n'], conf['dpi']
        ax = axes[idx]
        output_prefix = f'scatterplot_{level}_top{top_n}'
        plot_single(
            top, level, top_n, dpi, output_prefix,
            xlim, ylim,
            show_panel=False, ax=None
        )
        plot_single(
            top, level, top_n, dpi, output_prefix,
            xlim, ylim,
            show_panel=True, ax=ax
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)  # <- espaçamento horizontal extra
    if len(configs) > 1:
        out_prefix = (
            f'scatterplot_panel_faal_vs_genome_{domain_name}_{"-".join([c["level"] for c in configs])}'
        )
    else:
        out_prefix = (
            f'scatterplot_{domain_name}_{configs[0]["level"]}_top{configs[0]["top_n"]}'
        )
    for ext in ['png', 'svg', 'jpeg']:
        plt.savefig(
            f'{out_prefix}.{ext}',
            dpi=configs[0]["dpi"],
            bbox_inches='tight'
        )
    plt.show()

# ---- Argumentos flexíveis ----
if __name__ == '__main__':
    if len(sys.argv) < 5 or (len(sys.argv)-3) % 3 != 0:
        print(
            "Uso:\n"
            "  python3 scatterplot_counts_faalvf_codev4.py <table1.tsv> <Domain> <Level1> <TopN1> <DPI1> [<Level2> <TopN2> <DPI2> ...]\n"
            "Exemplo:\n"
            "  python3 scatterplot_counts_faalvf_codev4.py table1.tsv Bacteria Phylum 14 300 Order 30 300 Genus 30 300\n"
        )
        sys.exit(1)

    table1_path = sys.argv[1]
    domain_name = sys.argv[2]
    configs = []
    args = sys.argv[3:]
    for i in range(0, len(args), 3):
        level = args[i]
        top_n = int(args[i+1])
        dpi   = int(args[i+2])
        configs.append({'level': level, 'top_n': top_n, 'dpi': dpi})
    generate_panel(
        table1_path,
        domain_name,
        configs
    )


