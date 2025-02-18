#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# Configure matplotlib backend to TkAgg
matplotlib.use('TkAgg')

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
      - Have a value in the 'BGC' column that does NOT start with "BGC"
    
    Then, groups the records by 'BiG-SCAPE class', counts them, and calculates proportions.
    Classes with proportions less than 0.1% (< 0.001) are removed.
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
    # Remove classes with proportions less than 0.1%
    grp = grp[grp["Proportion"] >= 0.001]
    return grp

def adjust_annotations(annotations, spacing=0.05):
    """
    Adjusts the positions of external annotations to minimize overlap.
    The annotations are sorted by their y-value (from highest to lowest) and, if the difference
    between two annotations is less than 'spacing', the current annotation's y-position is adjusted.
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

def plot_pie_chart(prop_df, color_mapping, ax=None):
    """
    Generates the pie chart with the following features:
      - For slices with proportion ≥ 2%: the percentage is shown inside the slice (via autopct)
        and an external label displays only the class name, positioned at 1.15× the radius.
      - For slices with proportion < 2%: no internal percentage is shown, and an external annotation
        (with arrow) displays the class name and percentage, positioned at 1.5× the radius.
      - Annotations for slices <2% are adjusted to minimize overlap.
      - No title is added.
    """
    if prop_df.empty:
        print("No data available for the pie chart.")
        return

    labels = prop_df['BiG-SCAPE class']
    sizes = prop_df['Proportion']

    # Define colors using the mapping; default to gray if not found
    colors = [color_mapping.get(lbl, "#333333") for lbl in labels]

    if ax is None:
        # Figure size suitable for Figure A in an NAR paper (7 x 7 inches)
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Create the pie chart with autopct for slices ≥2%
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

    # Set font size for internal percentages (for slices ≥2%)
    for autotext in autotexts:
        if autotext.get_text() != '':
            autotext.set_fontsize(12)

    # List to store annotations for slices <2% (to adjust later)
    small_annotations = []

    # Add external annotations for each slice
    for idx, (lbl, prop, wedge) in enumerate(zip(labels, sizes, wedges)):
        # Calculate the central angle of the slice
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        angle_rad = np.deg2rad(angle)
        # Point on the boundary of the slice (radius = 1)
        x, y = np.cos(angle_rad), np.sin(angle_rad)
        
        if prop >= 0.02:
            # For slices with ≥2%: position text at 1.15× the radius
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
            # For slices with <2%: position text at 1.5× the radius
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
    
    # Adjust annotations for slices with <2% to minimize overlapping
    adjust_annotations(small_annotations, spacing=0.05)

    ax.set_title("")  # Remove title
    ax.axis('equal')
    return ax

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <path_to_file.tsv>")
        sys.exit(1)
    file_path = sys.argv[1]
    df = load_data(file_path)
    # Check that the necessary columns exist
    required_cols = {"BGC", "Taxonomy", "BiG-SCAPE class"}
    if not required_cols.issubset(df.columns):
        print("Error: The columns 'BGC', 'Taxonomy' and 'BiG-SCAPE class' must exist in the file.")
        sys.exit(1)
    
    # Calculate proportions for Bacteria records (keeping records whose 'BGC' does NOT start with "BGC")
    prop_df = calculate_proportions(df)
    
    # Create a color mapping using the viridis colormap
    unique_classes = sorted(prop_df["BiG-SCAPE class"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    color_mapping = {cls: cmap(i / (len(unique_classes) - 1) if len(unique_classes) > 1 else 0.5)
                     for i, cls in enumerate(unique_classes)}

    # Create the pie chart with dimensions suitable for Figure A in NAR
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    plot_pie_chart(prop_df, color_mapping, ax)

    plt.tight_layout()
    # Save the figure:
    # - SVG (vector) at 300 dpi
    # - PNG at 900 dpi
    fig.savefig('bigscape_class_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
    fig.savefig('bigscape_class_distribution.png', format='png', dpi=900, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
