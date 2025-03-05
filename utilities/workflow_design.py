# -*- coding: utf-8 -*-
import os
from graphviz import Digraph
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor

# ============================================================
# Funções Auxiliares para cores (Graphviz e PPTX)
# ============================================================
def hex_to_rgb(hex_color: str) -> tuple:
    """Converte cor hexadecimal para uma tupla RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def is_dark_color(hex_color: str, threshold: int = 130) -> bool:
    """Determina se uma cor é considerada escura com base na luminância."""
    r, g, b = hex_to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < threshold

# ============================================================
# PARTE 1 – FLUXOGRAMA COM GRAPHVIZ (PNG e SVG com DPI 900)
# ============================================================
def create_flowchart():
    # Usando orientação vertical (TB)
    dot = Digraph(format='png')
    dot.attr(rankdir='TB', size='11.69,8.27!', dpi='900')
    # Aumenta fonte e dimensões dos nós
    dot.attr('node', fontname='Arial', fontsize='20', width='2.5', height='1.5')
    dot.attr('edge', fontname='Arial', fontsize='16')
    
    # Definindo um dicionário de cores em escala de cinza suaves
    gray_colors = {
        'A': '#F2F2F2',
        'B': '#E6E6E6',
        'C': '#D9D9D9',
        'D': '#CCCCCC',
        'E': '#C0C0C0',
        'F1': '#C0C0C0',  # mesma cor de E
        'F': '#B3B3B3',
        'G': '#A6A6A6',
        'H': '#999999',
        'I': '#8C8C8C',
        'J': '#808080',
        'K': '#737373',
        'L': '#666666',
        'M': '#5A5A5A',
        'N': '#4D4D4D',
        'O': '#404040',
        'P': '#333333',
        'Q': '#262626'
    }
    
    # Função para definir atributos do nó, incluindo fontcolor se a cor for escura
    def add_node(node_id, label, shape, fillcolor):
        attrs = {'shape': shape, 'style': 'filled', 'fillcolor': fillcolor}
        if is_dark_color(fillcolor):
            attrs['fontcolor'] = 'white'
        dot.node(node_id, label, **attrs)
    
    # Usando etiquetas HTML para negrito; escapando "&" como "&amp;"
    add_node('A', '<<B>Start: Configure Streamlit &amp; Environment</B>>', 'ellipse', gray_colors['A'])
    add_node('B', '<<B>Load Training Data:\n- FASTA\n- Table</B>>', 'parallelogram', gray_colors['B'])
    add_node('C', '<<B>Check Training Sequence Alignment</B>>', 'box', gray_colors['C'])
    add_node('D', '<<B>Realign with MAFFT</B>>', 'diamond', gray_colors['D'])
    add_node('E', '<<B>Use Aligned Training Data</B>>', 'box', gray_colors['E'])
    # Novo nó: Breaking alignment into k-mers
    add_node('F1', '<<B>Breaking alignment into k-mers</B>>', 'box', gray_colors['F1'])
    # Nó F com texto alterado:
    add_node('F', '<<B>Generate Training\nEmbeddings\n(Word2Vec and saving global word2vec model)</B>>', 'box', gray_colors['F'])
    add_node('G', '<<B>Standardize Training Embeddings\n(StandardScaler)</B>>', 'box', gray_colors['G'])
    add_node('H', '<<B>Train Model with Oversampling\n(RandomForest, Grid Search)</B>>', 'box', gray_colors['H'])
    add_node('I', '<<B>Calibrate Model\n(Isotonic Calibration)</B>>', 'box', gray_colors['I'])
    add_node('J', '<<B>Evaluate Model\n(ROC, F1, PR AUC)</B>>', 'box', gray_colors['J'])
    add_node('K', '<<B>Load Prediction Data (FASTA)</B>>', 'parallelogram', gray_colors['K'])
    add_node('L', '<<B>Check Prediction Sequence Alignment</B>>', 'box', gray_colors['L'])
    # Nó M com texto alterado:
    add_node('M', '<<B>Generate New Sequence\nEmbeddings\n(Word2Vec, k-mers) using global ww model - transfer learning</B>>', 'box', gray_colors['M'])
    add_node('N', '<<B>Standardize New Sequence Embeddings\n(Using Training Scaler)</B>>', 'box', gray_colors['N'])
    add_node('O', '<<B>Predict on New Sequences &amp;\nGet Class Rankings</B>>', 'box', gray_colors['O'])
    add_node('P', '<<B>Visualize Results\n(Scatter Plot, UMAP,\nLearning Curve)</B>>', 'box', gray_colors['P'])
    add_node('Q', '<<B>Output Results\n(Save Files, Download)</B>>', 'ellipse', gray_colors['Q'])
    
    # Dividindo o fluxo em 4 clusters (painéis)
    with dot.subgraph(name='cluster_A') as cA:
        cA.attr(label='A) Neural Network: Protein Embedding Phase', fontsize='24', fontname='Arial Bold', fontcolor='black', style='dashed', color='black')
        for n in ['A', 'B', 'C', 'D', 'E', 'F1', 'F', 'G']:
            cA.node(n)
    with dot.subgraph(name='cluster_B') as cB:
        cB.attr(label='B) Oversampling Phase', fontsize='24', fontname='Arial Bold', fontcolor='black', style='dashed', color='black')
        cB.node('H')
    with dot.subgraph(name='cluster_C') as cC:
        cC.attr(label='C) Training and Optimization of Random Forest Phase', fontsize='24', fontname='Arial Bold', fontcolor='black', style='dashed', color='black')
        for n in ['I', 'J']:
            cC.node(n)
    with dot.subgraph(name='cluster_D') as cD:
        cD.attr(label='D) Classification of New Sequences Phase – Transfer Learning', fontsize='24', fontname='Arial Bold', fontcolor='black', style='dashed', color='black')
        for n in ['K', 'L', 'M', 'N', 'O', 'P', 'Q']:
            cD.node(n)
    
    # Define as arestas:
    # Agora, a seta de saída do nó A é mantida (A -> B)
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D', label='Not Aligned')
    dot.edge('C', 'E', label='Aligned')
    dot.edge('D', 'E')
    dot.edge('E', 'F1')
    dot.edge('F1', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H', ltail='cluster_A', lhead='cluster_B')
    dot.edge('H', 'I', ltail='cluster_B', lhead='cluster_C')
    dot.edge('I', 'J', ltail='cluster_C', lhead='cluster_C')
    dot.edge('J', 'K', ltail='cluster_C', lhead='cluster_D')
    dot.edge('K', 'L', lhead='cluster_D')
    # Alterando a sequência: 
    # (L -> M já conecta o nó L a M com texto alterado para M)
    dot.edge('L', 'M', lhead='cluster_D')
    dot.edge('M', 'N', lhead='cluster_D')
    dot.edge('N', 'O', lhead='cluster_D')
    dot.edge('O', 'P', lhead='cluster_D')
    dot.edge('P', 'Q', lhead='cluster_D')
    
    return dot

def generate_flowchart_images(filename_base="workflow"):
    flowchart = create_flowchart()
    png_path = flowchart.render(filename_base, format='png', view=False, cleanup=True)
    svg_path = flowchart.render(filename_base, format='svg', view=False, cleanup=True)
    print(f"PNG gerado em: {png_path}")
    print(f"SVG gerado em: {svg_path}")

# ============================================================
# PARTE 2 – FLUXOGRAMA COMO PPTX (FORMAS NATIVAS – ORGANIZADOS VERTICALMENTE)
# ============================================================
def map_shape(shape_str: str):
    """
    Mapeia o tipo de forma para a forma nativa do PowerPoint usando MSO_AUTO_SHAPE_TYPE.
    """
    if shape_str == 'ellipse':
        return MSO_AUTO_SHAPE_TYPE.OVAL
    elif shape_str == 'parallelogram':
        return MSO_AUTO_SHAPE_TYPE.PARALLELOGRAM
    elif shape_str == 'diamond':
        return MSO_AUTO_SHAPE_TYPE.DIAMOND
    elif shape_str == 'box':
        return MSO_AUTO_SHAPE_TYPE.RECTANGLE
    else:
        return MSO_AUTO_SHAPE_TYPE.RECTANGLE

def ppt_font_color(fill_hex: str) -> str:
    """
    Retorna 'FFFFFF' se a cor de preenchimento for escura, senão '000000'.
    """
    return "FFFFFF" if is_dark_color(fill_hex) else "000000"

# Atualizando os nós para incluir o novo passo F1 e ajustando posições (em polegadas)
ppt_nodes = {
    # Painel A: Neural Network: Protein Embedding Phase (nós A a G, com F1)
    'A': {'text': 'Start: Configure\nStreamlit &amp; Environment', 'shape': 'ellipse',       'fillcolor': '#F2F2F2', 'x': 1,   'y': 0.5,    'w': 7, 'h': 2.5},
    'B': {'text': 'Load Training Data:\n- FASTA\n- Table',             'shape': 'parallelogram', 'fillcolor': '#E6E6E6', 'x': 1,   'y': 3.5,    'w': 7, 'h': 2.5},
    'C': {'text': 'Check Training\nSequence Alignment',                'shape': 'box',           'fillcolor': '#D9D9D9', 'x': 1,   'y': 6.5,    'w': 7, 'h': 2.5},
    'D': {'text': 'Realign with MAFFT',                                   'shape': 'diamond',       'fillcolor': '#CCCCCC', 'x': 1,   'y': 9.0,    'w': 7, 'h': 2.5},
    'E': {'text': 'Use Aligned\nTraining Data',                          'shape': 'box',           'fillcolor': '#C0C0C0', 'x': 1,   'y': 11.5,   'w': 7, 'h': 2.5},
    # Novo nó F1: Breaking alignment into k-mers
    'F1': {'text': 'Breaking alignment into\nk-mers',                    'shape': 'box',           'fillcolor': '#C0C0C0', 'x': 1,   'y': 14.0,   'w': 7, 'h': 2.5},
    # Nó F com texto alterado:
    'F': {'text': 'Generate Training\nEmbeddings\n(Word2Vec and saving global word2vec model)', 'shape': 'box', 'fillcolor': '#B3B3B3', 'x': 1, 'y': 16.5, 'w': 7, 'h': 2.5},
    'G': {'text': 'Standardize Training\nEmbeddings\n(StandardScaler)',   'shape': 'box',           'fillcolor': '#A6A6A6', 'x': 1,   'y': 19.0,   'w': 7, 'h': 2.5},

    # Painel B: Oversampling Phase (nó H)
    'H': {'text': 'Train Model with\nOversampling\n(RandomForest, Grid Search)', 'shape': 'box', 'fillcolor': '#999999', 'x': 1,   'y': 22.5,   'w': 7, 'h': 2.5},

    # Painel C: Training and Optimization of Random Forest Phase (nós I e J)
    'I': {'text': 'Calibrate Model\n(Isotonic Calibration)', 'shape': 'box', 'fillcolor': '#8C8C8C', 'x': 1,   'y': 25.5,   'w': 7, 'h': 2.5},
    'J': {'text': 'Evaluate Model\n(ROC, F1, PR AUC)', 'shape': 'box', 'fillcolor': '#808080', 'x': 1,   'y': 28.5,   'w': 7, 'h': 2.5},

    # Painel D: Classification of New Sequences Phase – Transfer Learning (nós K a Q)
    'K': {'text': 'Load Prediction\nData (FASTA)', 'shape': 'parallelogram', 'fillcolor': '#737373', 'x': 1,   'y': 31.5,   'w': 7, 'h': 2.5},
    'L': {'text': 'Check Prediction\nSequence Alignment', 'shape': 'box', 'fillcolor': '#666666', 'x': 1,   'y': 34.5,   'w': 7, 'h': 2.5},
    # Nó M com texto alterado:
    'M': {'text': 'Generate New Sequence\nEmbeddings\n(Word2Vec, k-mers) using global ww model - transfer learning', 'shape': 'box', 'fillcolor': '#5A5A5A', 'x': 1,   'y': 37.5,   'w': 7, 'h': 2.5},
    'N': {'text': 'Standardize New Sequence\nEmbeddings\n(Using Training Scaler)', 'shape': 'box', 'fillcolor': '#4D4D4D', 'x': 1,   'y': 40.5,   'w': 7, 'h': 2.5},
    'O': {'text': 'Predict on New Sequences &amp;\nGet Class Rankings', 'shape': 'box', 'fillcolor': '#404040', 'x': 1,   'y': 43.5,   'w': 7, 'h': 2.5},
    'P': {'text': 'Visualize Results\n(Scatter Plot, UMAP,\nLearning Curve)', 'shape': 'box', 'fillcolor': '#333333', 'x': 1,   'y': 46.5,   'w': 7, 'h': 2.5},
    'Q': {'text': 'Output Results\n(Save Files, Download)', 'shape': 'ellipse', 'fillcolor': '#262626', 'x': 1,   'y': 49.5,   'w': 7, 'h': 2.5}
}

# Conexões entre os nós (fluxo completo)
# Agora, conectamos A a B, garantindo a seta de saída do nó A.
ppt_edges = [
    ('A', 'B', ''),
    ('B', 'C', ''),
    ('C', 'D', 'Not Aligned'),
    ('C', 'E', 'Aligned'),
    ('D', 'E', ''),
    # Incluindo F1 entre E e F
    ('E', 'F1', ''),
    ('F1', 'F', ''),
    ('F', 'G', ''),
    # Conexão de Painel A para B
    ('G', 'H', ''),
    # Conexão de Painel B para C
    ('H', 'I', ''),
    # Painel C
    ('I', 'J', ''),
    # Conexão de Painel C para D
    ('J', 'K', ''),
    # Dentro do Painel D
    ('K', 'L', ''),
    ('L', 'M', ''),
    ('M', 'N', ''),
    ('N', 'O', ''),
    ('O', 'P', ''),
    ('P', 'Q', '')
]

def create_ppt_flowchart(ppt_filename="workflow_native.pptx"):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Slide em branco
    shapes_dict = {}

    # Adiciona cada nó com as propriedades definidas
    for node_id, props in ppt_nodes.items():
        shape_type = map_shape(props['shape'])
        left = Inches(props['x'])
        top = Inches(props['y'])
        width = Inches(props['w'])
        height = Inches(props['h'])
        shape = slide.shapes.add_shape(shape_type, left, top, width, height)
        shape.fill.solid()
        # Define a cor de preenchimento (RGB)
        shape.fill.fore_color.rgb = RGBColor.from_string(props['fillcolor'].lstrip('#'))
        # Define o texto e formata com fonte Arial, tamanho 20 pt e alinhamento centralizado.
        # Se a cor de preenchimento for escura, define a cor da fonte como branca.
        shape.text = props['text']
        for paragraph in shape.text_frame.paragraphs:
            paragraph.font.size = Pt(20)
            paragraph.font.name = 'Arial'
            paragraph.font.bold = True
            paragraph.alignment = 1  # Centralizado
            paragraph.font.color.rgb = RGBColor.from_string(ppt_font_color(props['fillcolor']))
        shapes_dict[node_id] = shape

    # Adiciona os títulos dos painéis como caixas de texto
    panel_titles = {
        'A': {'text': 'A) Neural Network: Protein Embedding Phase', 'x': Inches(1), 'y': Inches(0),   'w': Inches(7), 'h': Inches(0.8)},
        'B': {'text': 'B) Oversampling Phase',                     'x': Inches(1), 'y': Inches(20.0), 'w': Inches(7), 'h': Inches(0.8)},
        'C': {'text': 'C) Training and Optimization of Random Forest Phase', 'x': Inches(1), 'y': Inches(24.0), 'w': Inches(7), 'h': Inches(0.8)},
        'D': {'text': 'D) Classification of New Sequences Phase – Transfer Learning', 'x': Inches(1), 'y': Inches(30.0), 'w': Inches(7), 'h': Inches(0.8)}
    }
    for p in panel_titles.values():
        title_box = slide.shapes.add_textbox(p['x'], p['y'], p['w'], p['h'])
        title_tf = title_box.text_frame
        title_tf.text = p['text']
        for para in title_tf.paragraphs:
            para.font.size = Pt(24)
            para.font.name = 'Arial'
            para.font.bold = True
            para.alignment = 1

    # Adiciona os conectores entre os nós
    for start_id, end_id, label in ppt_edges:
        start_shape = shapes_dict[start_id]
        end_shape = shapes_dict[end_id]
        # Calcula os pontos de conexão
        start_x = int(start_shape.left + start_shape.width)
        start_y = int(start_shape.top + start_shape.height / 2)
        end_x = int(end_shape.left)
        end_y = int(end_shape.top + end_shape.height / 2)
        connector = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            start_x, start_y,
            end_x - start_x,
            end_y - start_y
        )
        connector.line.width = Pt(2)
        connector.begin_connect(start_shape, 1)
        connector.end_connect(end_shape, 3)
        if label:
            mid_x = int((start_x + end_x) / 2)
            mid_y = int((start_y + end_y) / 2)
            textbox = slide.shapes.add_textbox(Inches(mid_x/72), Inches(mid_y/72), Inches(1), Inches(0.5))
            textbox.text_frame.text = label
            for paragraph in textbox.text_frame.paragraphs:
                paragraph.font.size = Pt(16)
                paragraph.font.name = 'Arial'
                paragraph.font.bold = True

    prs.save(ppt_filename)
    print(f"PPT salvo como {ppt_filename}")

# ============================================================
# EXECUÇÃO: GERANDO OS FORMATOS DO FLUXOGRAMA
# ============================================================
if __name__ == "__main__":
    # Gera imagens PNG e SVG com DPI 900
    generate_flowchart_images("workflow")
    # Gera arquivo PPTX com os 4 painéis organizados verticalmente, com títulos aumentados, textos em negrito e caixas em escala de cinza suaves;
    # A seta de saída do nó A (A -> B) foi adicionada conforme solicitado.
    create_ppt_flowchart("workflow_native.pptx")
    
# ====================================================================
# (O restante do código, que inclui funções de processamento, treinamento, visualizações e interface Streamlit,
# permanece inalterado, exceto pelas modificações realizadas na parte do fluxograma.)
# ====================================================================

# IMPORTANTE: Este código inclui as alterações solicitadas para as etapas do fluxo.
