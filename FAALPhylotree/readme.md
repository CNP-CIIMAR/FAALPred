
# FAALPhylotree: Circular FAAL Phylogenetic Tree with Fatty Acid Structures

This repository contains **FAALPhylotree**, an R workflow that builds a **circular phylogenetic tree of FAAL proteins** and decorates it with:

- A tip–colored phylogeny by **phylum**
- Two **rectangular annotation rings** (MIBIG vs non-MIBIG, and multi-domain vs single-domain)
- **Fatty acid (FA) structures** drawn from SMILES via **rcdk**, standardized in size and plotted in solid red on a transparent background
- Automatic export of high-resolution figures (SVG, PNG, TIFF, PDF) suitable for publication (e.g. in *Protein Science*)

The code is fully data-driven: once you provide the Newick tree and three metadata tables, FAALPhylotree reconstructs the figure end-to-end.

> **Note on biome data**  
> Biome and geographic metadata from NCBI are **not available for all genomes**.  
> For this reason, **biome information is loaded and merged but not plotted** in the current version of FAALPhylotree.  
> We keep the reference to the biome table (`metadata2.xlsx`) in the code for completeness and future extensions, but no biome ring or biome-based color scale is drawn.

---

## 1. Overview of the pipeline

1. **Input files**
   - `tree_newick.txt` — rooted phylogenetic tree of FAAL proteins in Newick format.
   - `metadata1.xlsx` — protein-level and taxonomic metadata.
   - `metadata2.xlsx` — biome and geographic metadata at the assembly level.  
     *(Loaded and merged, but **not plotted** because NCBI metadata are incomplete for many genomes.)*
   - `metadata3.xlsx` — fatty-acid specificity metadata (e.g. C16:0, C18:1).

2. **Tree and metadata integration**
   - Protein accessions in the tree are normalized and matched to the metadata tables.
   - Taxonomic information is used to color tips by **collapsed phylum** (top N phyla + “Other Phylum” and “Outgroup”).
   - Each protein is classified as **MIBIG vs Non-MIBIG** and **Multidomain vs Single**.
   - Biome and geographic metadata are attached to each protein (when available), but they are not visualized.

3. **Fatty acid structures**
   - For each protein with a mapped FA (e.g. `C16:0`), a SMILES string is obtained (PubChem + fallback).
   - The molecule is drawn with **rcdk** and converted to a red-only structure on transparent background.
   - All FA images are rescaled to a **standardized width**, ensuring a uniform look around the tree.
   - A short colored segment connects each tip to the corresponding FA image, and the FA label (e.g. `C16:0`) is printed next to it.

4. **Output**
   - Circular tree with:
     - Tip points colored by phylum.
     - Inner ring: MIBIG vs Non-MIBIG.
     - Outer ring: Multi-architecture vs single FAAL.
     - External FA structures + labels.
   - Figures saved as:
     - `figure_tree_publication_circular_full_FAred_version_simple_nobarplot_redonly_stdsize.svg`
     - `figure_tree_publication_circular_full_FAred_version_simple_nobarplot_redonly_stdsize.png`
     - `figure_tree_publication_circular_full_FAred_version_simple_nobarplot_redonly_stdsize.tiff` (optional, safer size)
     - `figure_tree_publication_circular_full_FAred_version_simple_nobarplot_redonly_stdsize.pdf`

---

## 2. Folder structure and main script (FAALPhylotree)

A typical repository layout looks like this:

```text
.
├── FAALPhylotree.R                   # Main R script (FAALPhylotree code)
├── tree_newick.txt                   # Newick tree of FAAL proteins
├── metadata1.xlsx                    # Protein-level & taxonomic metadata
├── metadata2.xlsx                    # Biome & geographic metadata (not plotted)
├── metadata3.xlsx                    # Fatty-acid specificity metadata
└── README.md                         # This file
```

The script assumes that all input files are in the **working directory** set at the beginning:

```r
setwd("C:/Users/Leandro/Desktop/itol_tree")
```

In your own setup, change this `setwd()` call to the directory where you keep the tree and metadata files.

---

## 3. R version and dependencies

FAALPhylotree is written in **R** and uses several packages from CRAN and Bioconductor.

### 3.1. R packages

The following packages are used:

- `readxl`, `dplyr`, `tidyr`, `stringr`
- `ape`
- `ggplot2`, `ggnewscale`, `scales`, `patchwork`
- `ggtree`, `ggtreeExtra`
- `ggimage`, `png`, `magick`
- `RColorBrewer`
- `rJava`, `rcdklibs`, `rcdk`

The script automatically installs any missing packages:

```r
pkgs <- c(
  "readxl","dplyr","tidyr","stringr","ape","ggplot2","ggnewscale",
  "scales","patchwork","ggimage","png","magick","ggtree",
  "ggtreeExtra","RColorBrewer","rJava","rcdklibs","rcdk"
)

inst <- rownames(installed.packages())
to_install <- pkgs[!(pkgs %in% inst)]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
```

### 3.2. System requirements

- **R ≥ 4.x** (recommended)
- A working **Java** installation (needed for `rJava`, `rcdklibs`, `rcdk`)
- Internet connection (optional but recommended) for retrieving SMILES from **PubChem**

Without internet, the script still works for **saturated** fatty acids, for which a simple linear SMILES is generated as a fallback. Unsaturated FAs without PubChem SMILES may be omitted from the plot.

---

## 4. Input files and metadata tables

The figure is driven by four input files:

1. `tree_newick.txt` — the phylogenetic tree
2. `metadata1.xlsx` — protein-level/taxonomic metadata
3. `metadata2.xlsx` — biome and geographic metadata (not plotted)
4. `metadata3.xlsx` — fatty-acid specificity metadata

Below is a description of each file and how it is associated with each protein ID.

### 4.1. `tree_newick.txt` — phylogenetic tree

- **Format:** Newick
- **Tip labels:** protein accessions (e.g. `WP_123456789.1`, `BGC0001234` etc.)
- **Outgroup:** the script expects a specific outgroup protein:
  - `NNJ93123.1`

The tree is read and rooted as:

```r
tree_raw <- read.tree(FILE_TREE)
tree     <- root(tree_raw, outgroup = OUTGROUP_PROTEIN, resolve.root = TRUE)
```

Each tip label is then normalized into an internal key `ProtNorm` by:

- Trimming whitespace
- Removing spaces and underscores

This normalization is applied consistently to tree tips and metadata tables, ensuring robust matching even if there are stray spaces or underscores.

### 4.2. `metadata1.xlsx` — protein-level and taxonomic metadata

This table provides per-protein annotation used to color tips, define multi-domain architectures, and collapse phyla.

The script assumes, at minimum, the following columns (names must match exactly):

- **`Protein Accession`** (required)  
  - Unique identifier of the protein in the tree.
  - Must correspond to the tip labels in `tree_newick.txt` (after trimming and removing spaces/underscores).
  - Used as the **primary key** to connect the tree to metadata.
- **`Assembly`**  
  - Assembly or genome accession.
  - Used to connect to `metadata2.xlsx` (biome and location).
- **`Species`**  
  - Species name (e.g. *Escherichia coli*).
- **`Lineage`**  
  - Full taxonomic lineage as a **semicolon-separated string** (e.g. `Bacteria; Proteobacteria; Gammaproteobacteria; ...`).
  - Internally split into:
    - `Domain`, `Phylum`, `Class`, `Order`, `Family`, `Extra1`, `Extra2`.
- **`Combined Signature description`**  
  - Description of domain architecture / functional signature.
  - Used to classify proteins as **Multidomain** vs **Single**:
    - If it contains `"FAAL-"` → `Multidomain`
    - If it contains `"FAAL"` (without `"FAAL-"`) → `Single`
    - Else → `Single` (default)
- **`color three`** (optional)  
  - Additional color information; not essential for the current plotting workflow.

Internally, the table is processed as:

```r
meta1 <- m1 %>%
  rename(
    ProteinRaw = `Protein Accession`,
    Assembly   = Assembly,
    Species    = Species,
    Lineage    = Lineage,
    Signature  = `Combined Signature description`,
    ColorThree = `color three`
  ) %>%
  mutate(
    Protein   = ProteinRaw,
    ProtNorm  = normalize_id(ProteinRaw),
    MultiDomain = case_when(
      !is.na(Signature) & str_detect(Signature,"FAAL-") ~ "Multidomain",
      !is.na(Signature) & str_detect(Signature,"FAAL")  ~ "Single",
      TRUE ~ "Single"
    )
  )
```

**Association to each protein:**

- For each tip in the tree:
  - `ProtNorm` is computed from its label.
  - The script finds the corresponding row in `metadata1.xlsx` using the same `ProtNorm`.
  - Taxonomic fields such as `Phylum` are used to color the tree.
  - The `MultiDomain` classification is used to build the outer ring.

### 4.3. `metadata2.xlsx` — biome and geographic metadata *(not plotted)*

This table provides biome and location information at the **assembly** level.

Because different datasets may use slightly different column names, the script tries several options for each field using a helper `pick_col()`.

The internally used fields are:

- **Assembly (one of):**
  - `Assembly Accession`, `Assembly`, or `Assembly_Accession`
- **Biome (one of):**
  - `BiomeDistribution`, `Biome Distribution`, `Biome`, or `Biome_Distribution`
- **Location (one of):**
  - `Location` or `Locality`
- **Latitude (one of):**
  - `Latitude` or `Lat`
- **Longitude (one of):**
  - `Longitude`, `Long`, or `Lon`

In the code:

```r
meta2 <- tibble(
  Assembly = pick_col(m2, c("Assembly Accession","Assembly","Assembly_Accession")),
  Biome    = pick_col(m2, c("BiomeDistribution","Biome Distribution","Biome","Biome_Distribution")),
  Location = pick_col(m2, c("Location","Locality")),
  Latitude = suppressWarnings(as.numeric(pick_col(m2, c("Latitude","Lat")))),
  Longitude= suppressWarnings(as.numeric(pick_col(m2, c("Longitude","Long","Lon"))))
) %>% distinct()
```

Typical biological meaning:

- **`Biome`** — ecological/biome category (e.g. marine, freshwater, soil, host-associated).
- **`Location`** — human-readable geographic description (e.g. sampling site, country).
- **`Latitude`, `Longitude`** — coordinates of sampling location.

**Association to each protein:**

- Each protein has an `Assembly` field defined in `metadata1.xlsx`.
- `metadata2.xlsx` is joined by the `Assembly` column:
  - all environmental and geographic information is attached to the corresponding proteins.

> **Important:** Biome and geographic metadata from NCBI are **not available for all genomes**, which would lead to uneven coverage and potential biases in the visualization. Therefore, in FAALPhylotree:
>
> - `metadata2.xlsx` is **loaded and merged** with the main annotation table.
> - However, **no biome ring or biome-based color scale is plotted**.
> - Keeping this table in the code maintains compatibility with future versions or other plots that may use biome information.

### 4.4. `metadata3.xlsx` — fatty-acid specificity metadata

This table provides the fatty-acid substrate specificity for each protein and (optionally) a custom color for that substrate.

Expected columns:

- **`Protein Accession`**  
  - Same protein accession used in `metadata1.xlsx` and in the tree tips.
  - Used (after normalization) to link to each protein in the tree.
- **`Specificity`**  
  - A string describing the fatty-acid substrate, typically in the format `C<number>:<double-bonds>`.
  - Examples: `C16:0`, `C18:1`, `C18:2`, `C16.1`.
  - The script normalizes this field to a standard form:
    - Removes spaces
    - Converts e.g. `C16.1` to `C16:1`
    - Converts `C16` to `C16:0`
- **`Label color`**  
  - Optional; hex color used to draw the small line segment connecting the tip to the FA image.
  - If missing, a color is automatically assigned.

Processing in the script:

```r
meta3 <- m3 %>%
  rename(
    ProteinRaw     = `Protein Accession`,
    SubstrateLabel = Specificity,
    SubstrateColor = `Label color`
  ) %>%
  mutate(
    ProteinRaw     = str_replace_all(ProteinRaw,"\s+",""),
    SubstrateLabel = normalize_fa_key(na_if(trimws(SubstrateLabel),"NA")),
    ProtNorm       = normalize_id(ProteinRaw)
  ) %>%
  select(ProtNorm,SubstrateLabel,SubstrateColor) %>%
  distinct(ProtNorm,.keep_all = TRUE)
```

**Association to each protein:**

- `ProtNorm` is again computed from `Protein Accession` and used to join with the tree tips.
- Once linked, each protein knows:
  - `SubstrateLabel` (e.g. `C16:0`) → determines which fatty acid is drawn.
  - `SubstrateColor` → determines the color of the connecting segment.

---

## 5. How metadata and tree are merged

The central table `ann` (annotations) is built by merging the tree tips with the three metadata tables:

1. For each tree tip label:
   - Compute `ProtNorm = normalize_id(tip_label)`.
2. Join with `meta1` on `ProtNorm` to add taxonomic and assembly information.
3. Join with `meta2` on `Assembly` to add biome and location (not plotted).
4. Join with `meta3` on `ProtNorm` to add FA specificity.

In code:

```r
tips <- tibble(
  Protein  = tree$tip.label,
  ProtNorm = normalize_id(tree$tip.label)
)

meta_tree <- tips %>%
  left_join(meta1 %>% select(-Protein,-ProteinRaw), by = "ProtNorm") %>%
  left_join(meta2, by = "Assembly") %>%
  left_join(meta3, by = "ProtNorm")
```

The final annotation table `ann` is ordered to match the exact tip order of the tree:

```r
ann <- meta_tree %>%
  transmute(
    Protein, Assembly, Lineage, Phylum, Class, Order, Family, Genus, Species,
    Signature, MultiDomain, Biome, Location, Latitude, Longitude,
    SubstrateLabel, SubstrateColor,
    MIBIG = if_else(str_detect(Protein,"^BGC"), "MIBIG", "Non-MIBIG"),
    PhylumCollapsed = if_else(
      !is.na(Phylum) & Phylum %in% top_phyla,
      Phylum,
      "Other Phylum"
    )
  ) %>%
  distinct(Protein,.keep_all = TRUE) %>%
  slice(match(tree$tip.label, Protein))
```

Additional derived fields:

- **`MIBIG`**
  - `MIBIG` if protein ID starts with `"BGC"` (MIBIG identifiers).
  - `Non-MIBIG` otherwise.
- **`PhylumCollapsed`**
  - Top `TOP_N_PHYLA` phyla kept as is.
  - All others → `"Other Phylum"`.
  - Outgroup protein is assigned `"Outgroup"`.

These fields drive:

- Tip colors (by `PhylumCollapsed`)
- Inner rectangular ring (`MIBIG` vs `Non-MIBIG`)
- Outer rectangular ring (`MultiDomain` vs `Single`)

Biome/Location columns are stored in `ann` but not used in the plotting layers.

---

## 6. Fatty-acid structures: SMILES, rcdk and plotting

For each protein with a valid `SubstrateLabel` (e.g. `C16:0`), FAALPhylotree:

1. Parses the label to obtain:
   - **`nC`** — number of carbons
   - **`nDB`** — number of double bonds
2. Attempts to find a **PubChem CID and SMILES** using known synonyms (systematic and trivial names).
3. If PubChem lookup fails and `nDB == 0`, a simple linear saturated SMILES is generated as a fallback.
4. Uses **rcdk** to:
   - Parse the SMILES.
   - Generate 2D coordinates.
   - Render a high-resolution molecule image.
5. Uses **magick** to:
   - Convert the molecule to a **red-only** structure on transparent background.
   - Place it on a **square canvas**.
   - Resize all FA images to the same final size, ensuring standardized width in the tree.

The resulting PNG files are stored under:

```text
fa_fatty_acids_png/red/FA_CXX_Y.png
```

where `XX` is the number of carbons (zero-padded) and `Y` is the number of double bonds, e.g.:

- `FA_C16_0.png`
- `FA_C18_1.png`

On the tree:

- A short colored segment connects each tip to the FA image.
- The FA image is placed at a fixed radial fraction between the tip and the first ring.
- The FA label (e.g. `C16:0`) is written next to the image.

All FAs therefore appear **uniform in size and style** around the circular tree.

---

## 7. Legend and visual encoding

The plot includes a multi-part legend with three logical sections:

1. **Phylum**
   - Entry label: `" Phylum"` (with a leading space; no legend box, text only).
   - One line per phylum in `PhylumCollapsed`.
   - “Outgroup” listed separately.

2. **MIBIG**
   - Entry label: `" MIBIG"` (no legend box).
   - Two entries: `MIBIG` and `Non-MIBIG`, corresponding to colors in the inner ring.

3. **Multi-Architecture**
   - Entry label: `" Multi-Architecture"` (no legend box).
   - Two entries: `Multidomain` and `Single`, corresponding to colors in the outer ring.

The titles themselves are drawn without colored boxes by mapping their internal keys (`TITLE_Phylum`, `TITLE_MIBIG`, `TITLE_Multi_Architect`) to `NA` in the `scale_fill_manual()` values and using neutral `guide_legend()` settings.

---

## 8. Running FAALPhylotree

1. **Clone or download** this repository.
2. Place your input files in the same directory as `FAALPhylotree.R`:
   - `tree_newick.txt`
   - `metadata1.xlsx`
   - `metadata2.xlsx`
   - `metadata3.xlsx`
3. Edit the following lines at the top of `FAALPhylotree.R` if needed:
   - `setwd("...")` — working directory.
   - `FILE_META1`, `FILE_META2`, `FILE_META3`, `FILE_TREE` — input file names, if you use different names.
4. Open R (or RStudio) and run:

```r
source("FAALPhylotree.R")
```

or from a terminal:

```bash
Rscript FAALPhylotree.R
```

5. After the script finishes, you should see the figure files in the working directory:
   - `*.svg`, `*.png`, `*.tiff` (if successful), `*.pdf`

---

## 8.1 - Note:

**The final figure (svg format) of the article was edited after generating the figure using the code (FAALPhylotree.R) provided in this repository**.

## 9. Important parameters you may want to adjust

At the top of FAALPhylotree, several parameters control the appearance of the figure. Some of the most useful ones are:

- **Phylogeny & figure size**
  - `TOP_N_PHYLA` — number of phyla to highlight individually (others collapsed).
  - `FIG_W`, `FIG_H` — figure width and height for `ggsave()` (in inches).
  - `TREE_SIZE` — line width of the tree branche
