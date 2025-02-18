#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import matplotlib
# Use TkAgg for interactive display; if issues arise, try 'Agg'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

##############################
# FIGURE A: Pie Chart Functions
##############################
def load_data(file_path):
    """
    Loads the TSV file and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', low_memory=False)
        print(f"File '{file_path}' loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def calculate_proportions(df):
    """
    Filters records that:
      - Contain "Bacteria" in the 'Taxonomy' column (domain level Bacteria)
      - Have a 'BGC' value that does NOT start with "BGC"
    Then groups by 'BiG-SCAPE class', counts them, calculates proportions,
    and removes classes with proportions < 0.1% (< 0.001).
    """
    df_bac = df[
        df['Taxonomy'].str.contains('Bacteria', case=False, na=False) &
        ~df['BGC'].str.startswith("BGC", na=False)
    ].copy()
    
    if df_bac.empty:
        print("No Bacteria records with a BGC value not starting with 'BGC' were found.")
        sys.exit(1)
    
    grp = df_bac.groupby("BiG-SCAPE class")["BGC"].count().reset_index(name="Count")
    total = grp["Count"].sum()
    grp["Proportion"] = grp["Count"] / total
    grp = grp[grp["Proportion"] >= 0.001]  # Remove classes with proportions < 0.1%
    return grp

def adjust_annotations(annotations, spacing=0.05):
    """
    Adjusts positions of external annotations to minimize overlap.
    """
    if not annotations:
        return
    annotations_sorted = sorted(annotations, key=lambda ann: ann.xyann[1], reverse=True)
    for i in range(1, len(annotations_sorted)):
        prev_y = annotations_sorted[i-1].xyann[1]
        curr_x, curr_y = annotations_sorted[i].xyann
        if abs(curr_y - prev_y) < spacing:
            if curr_y > 0:
                curr_y = prev_y - spacing
            else:
                curr_y = prev_y + spacing
            annotations_sorted[i].xyann = (curr_x, curr_y)

def draw_pie_chart(ax, tsv_file):
    """
    Draws the pie chart (Figure A) on the provided axis using data from the TSV file.
    Slices with ≥2% display the percentage inside; slices <2% get an external annotation with arrow.
    """
    df = load_data(tsv_file)
    prop_df = calculate_proportions(df)
    
    # Create a color mapping using Viridis
    unique_classes = sorted(prop_df["BiG-SCAPE class"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    color_mapping = {cls: cmap(i / (len(unique_classes) - 1) if len(unique_classes) > 1 else 0.5)
                     for i, cls in enumerate(unique_classes)}
    
    labels = prop_df['BiG-SCAPE class']
    sizes = prop_df['Proportion']
    colors = [color_mapping.get(lbl, "#333333") for lbl in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
        pctdistance=0.6,
        wedgeprops=dict(edgecolor='w', linewidth=1.2),
        textprops=dict(color="white", fontsize=12)
    )
    
    for autotext in autotexts:
        if autotext.get_text() != '':
            autotext.set_fontsize(12)
    
    small_annotations = []
    for idx, (lbl, prop, wedge) in enumerate(zip(labels, sizes, wedges)):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        angle_rad = np.deg2rad(angle)
        x, y = np.cos(angle_rad), np.sin(angle_rad)
        if prop >= 0.02:
            x_text = 1.15 * np.cos(angle_rad)
            y_text = 1.15 * np.sin(angle_rad)
            ax.text(
                x_text, y_text, f"{lbl}",
                horizontalalignment="left" if x_text >= 0 else "right",
                verticalalignment="center",
                fontsize=12,
                color="black"
            )
        else:
            x_text = 1.5 * np.cos(angle_rad)
            y_text = 1.5 * np.sin(angle_rad)
            ann = ax.annotate(
                f"{lbl}: {prop*100:.1f}%",
                xy=(x, y),
                xytext=(x_text, y_text),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.3",
                    color="gray",
                    lw=0.8
                ),
                horizontalalignment="left" if x_text >= 0 else "right",
                verticalalignment="center",
                fontsize=12,
                color="black"
            )
            small_annotations.append(ann)
    adjust_annotations(small_annotations, spacing=0.05)
    
    ax.set_title("")  # No title for the pie chart
    ax.axis('equal')

##############################
# FIGURE B: Bar Chart Function
##############################
def draw_bar_chart(ax):
    """
    Draws the grouped bar chart (Figure B) on the provided axis.
    "Complete" (blue) bars appear on the left; "Fragmented (in_contig_edge)" (orange) bars on the right.
    No value labels are added above the bars.
    """
    categories = [
        "0-10 KB", "10-20 KB", "20-30 KB", "30-40 KB", "40-50 KB", 
        "50-60 KB", "60-70 KB", "70-80 KB", "80-90 KB", "90-100 KB", ">100 KB"
    ]
    complete = [
        0,      # 0-10 KB
        0,      # 10-20 KB
        75,     # 20-30 KB
        79,     # 30-40 KB
        10502,  # 40-50 KB
        2281,   # 50-60 KB
        1140,   # 60-70 KB
        1030,   # 70-80 KB
        1670,   # 80-90 KB
        1579,   # 90-100 KB
        1053    # >100 KB
    ]
    fragmented = [
        1190,   # 0-10 KB
        1200,   # 10-20 KB
        2267,   # 20-30 KB
        2090,   # 30-40 KB
        1933,   # 40-50 KB
        683,    # 50-60 KB
        1833,   # 60-70 KB
        543,    # 70-80 KB
        1035,   # 80-90 KB
        126,    # 90-100 KB
        181     # >100 KB
    ]
    
    N = len(categories)
    ind = np.arange(N)
    width = 0.35
    
    # Use Viridis colormap for two distinct colors.
    cmap = plt.get_cmap('viridis')
    color_complete = cmap(0.7)   # darker for Complete
    color_fragmented = cmap(0.3) # lighter for Fragmented
    
    ax.bar(ind - width/2, complete, width, label='Complete', color=color_complete)
    ax.bar(ind + width/2, fragmented, width, label='Fragmented (in_contig_edge)', color=color_fragmented)
    
    ax.set_ylim(0, 12000)
    ax.set_yticks(np.arange(0, 12001, 2000))
    ax.set_ylabel('Count', fontsize=14)
    
    ax.set_xticks(ind)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
    ax.set_xlabel('KB', fontsize=14)
    
    ax.legend(fontsize=12)
    plt.tight_layout()

##############################
# MAIN: Composite Figure
##############################
def main():
    parser = argparse.ArgumentParser(
        description="Generate a composite figure with Figure A (Pie Chart) above and Figure B (Bar Chart) below, for publication."
    )
    parser.add_argument("--tsv", required=True, help="Path to the TSV file for Figure A (Pie Chart)")
    parser.add_argument("--outfile", required=True, help="Base output filename for the composite figure (without extension)")
    args = parser.parse_args()
    
    # Create a composite figure with two subplots arranged vertically (2 rows x 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), dpi=900)
    
    # Draw Figure A (Pie Chart) in the upper subplot
    draw_pie_chart(ax1, args.tsv)
    # Label this subplot "A" in the upper-left corner
    ax1.text(0.02, 0.98, "A", transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    
    # Draw Figure B (Bar Chart) in the lower subplot (no bar value labels)
    draw_bar_chart(ax2)
    # Label this subplot "B" in the upper-left corner
    ax2.text(0.02, 0.98, "B", transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    
    # Save composite figure as SVG and PNG at 900 dpi.
    svg_filename = f"{args.outfile}.svg"
    png_filename = f"{args.outfile}.png"
    fig.savefig(svg_filename, format='svg', dpi=900, bbox_inches='tight')
    fig.savefig(png_filename, format='png', dpi=900, bbox_inches='tight')
    print(f"Composite figure saved as '{svg_filename}' and '{png_filename}'.")
    
    # Display the composite figure interactively
    plt.show()

if __name__ == "__main__":
    main()

