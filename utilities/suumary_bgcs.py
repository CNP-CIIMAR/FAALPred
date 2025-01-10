import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
#python3 summary_bgc.py bigscape_copia_update.tsv summary.tsv res

# Configuração para gráficos e fontes de publicação
rcParams.update({
    'font.size': 14,  # Aumentando o tamanho base da fonte
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

def process_and_plot(input_file, output_file, graph_prefix):
    try:
        # Carregar o arquivo
        df = pd.read_csv(input_file, delimiter='\t', low_memory=False)

        # Remover espaços nos nomes das colunas
        df.columns = df.columns.str.strip()

        # Confirmar se as colunas críticas existem
        required_columns = ['Accession ID', 'Family Number']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Erro: As seguintes colunas estão ausentes no arquivo: {missing_columns}")
            print("Colunas disponíveis:", df.columns.tolist())
            return

        # Filtrar apenas linhas com Family Number preenchido
        df_filtered = df[df['Family Number'].notna()].copy()

        # Adicionar prefixo 'FAM_' aos números de famílias
        df_filtered.loc[:, 'Family Number'] = df_filtered['Family Number'].apply(lambda x: f"FAM_{x}")

        # Identificar os BGCs caracterizados no MIBiG (IDs que começam com "BGC")
        bgc_mibig = df_filtered[df_filtered['Accession ID'].str.startswith('BGC')]

        # Obter os números de família dos BGCs caracterizados no MIBiG
        mibig_family_numbers = bgc_mibig['Family Number'].unique()

        # Filtrar BGCs novos na mesma família dos BGCs do MIBiG
        bgcs_new_in_same_family = df_filtered[
            ~df_filtered['Accession ID'].str.startswith('BGC') &
            df_filtered['Family Number'].isin(mibig_family_numbers)
        ]

        # Filtrar BGCs novos que não estão na mesma família dos BGCs do MIBiG
        bgcs_new_not_in_mibig_family = df_filtered[
            ~df_filtered['Family Number'].isin(mibig_family_numbers)
        ]

        # Identificar famílias com apenas um BGC
        family_counts = df_filtered['Family Number'].value_counts()
        singleton_families = family_counts[family_counts == 1].index
        singleton_bgcs = df_filtered[df_filtered['Family Number'].isin(singleton_families)]

        # Total de famílias únicas
        total_families = len(df_filtered['Family Number'].unique())

        # Identificar BGCs identificados (ID não começa com "BGC")
        bgcs_identified = df_filtered[~df_filtered['Accession ID'].str.startswith('BGC')]

        # Criar tabela intermediária
        df_summary = pd.DataFrame({
            'Metric': [
                'MIBiG BGCs',
                'MIBiG BGCs with FAAL',
                'New BGCs in MIBiG Families',
                'New BGCs outside MIBiG Families',
                'Singleton BGC Families',
                'Total BGC Families',
                'Identified BGCs'
            ],
            'Count': [
                len(bgc_mibig),
                129,  # Valor fixo para "MIBiG BGCs with FAAL"
                len(bgcs_new_in_same_family),
                len(bgcs_new_not_in_mibig_family),
                len(singleton_families),
                total_families,
                len(bgcs_identified)
            ]
        })

        # Salvar a tabela intermediária
        df_summary.to_csv(f"{output_file}_summary.tsv", index=False, sep='\t')
        print(f"Tabela intermediária salva em: {output_file}_summary.tsv")

        # Criar a tabela final combinando os filtros
        result = pd.concat([bgc_mibig, bgcs_new_in_same_family])
        result.to_csv(output_file, index=False, sep='\t')
        print(f"Tabela filtrada salva em: {output_file}")

        # Dados para o gráfico
        counts = dict(zip(df_summary['Metric'], df_summary['Count']))

        # Criar o gráfico
        num_bars = len(counts)
        colors = plt.cm.viridis([i / num_bars for i in range(num_bars)])
        plt.figure(figsize=(14, 8))  # Aumentado o tamanho da figura
        bars = plt.bar(counts.keys(), counts.values(), color=colors, edgecolor='black')

        # Adicionar rótulo ao eixo Y
        plt.ylabel('Counts', fontsize=16)

        # Rótulos do eixo X
        plt.xticks(rotation=15, ha='right', fontsize=14)

        # Ajustar margens para publicação
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3)

        # Adicionar valores acima das barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 500, str(height),
                     ha='center', va='bottom', fontsize=14)

        plt.tight_layout()

        # Salvar o gráfico em múltiplos formatos
        for ext in ['png', 'svg', 'jpeg']:
            plt.savefig(f"{graph_prefix}.{ext}", format=ext, dpi=300)
        print(f"Gráficos salvos com os prefixos: {graph_prefix}.png, .svg, .jpeg")

        # Exibir o gráfico se o sistema suportar
        try:
            plt.show()
        except Exception as e:
            print(f"Não foi possível exibir o gráfico: {e}")

    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtra BGCs e gera gráficos.")
    parser.add_argument("input_file", help="Caminho para o arquivo de entrada (TSV).")
    parser.add_argument("output_file", help="Caminho para o arquivo de saída (TSV).")
    parser.add_argument("graph_prefix", help="Prefixo para salvar os gráficos.")
    args = parser.parse_args()

    process_and_plot(args.input_file, args.output_file, args.graph_prefix)
