############################################################
# Autor: Leandro de Mattos Pereira - 21/11/2025
# Circular FAAL tree with fatty acid structures (FAs)
# - Circular phylogenetic tree (ggtree)
# - Rectangular rings (MIBIG + Multi-domain)
# - Fatty acids drawn from SMILES using rcdk
# - FAs displayed in solid red lines on transparent background
# - All FAs have standardized visual size (same width) in the tree
# - No barplots, no biome ring
# - Outputs: SVG, PNG (high-res), TIFF (safer size), PDF
############################################################

########## 0) Working directory ##########

# Set your working directory (folder where metadata/tree files live)
setwd("C:/Users/Leandro/Desktop/itol_tree")


########## 1) Packages ##########

# List of required packages
pkgs <- c(
  "readxl","dplyr","tidyr","stringr","ape","ggplot2","ggnewscale",
  "scales","patchwork","ggimage","png","magick","ggtree",
  "ggtreeExtra","RColorBrewer","rJava","rcdklibs","rcdk"
)

# Install any missing packages
inst <- rownames(installed.packages())
to_install <- pkgs[!(pkgs %in% inst)]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

# Load packages quietly
suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(stringr); library(ape)
  library(ggplot2); library(ggnewscale); library(scales); library(patchwork)
  library(ggtree); library(ggtreeExtra); library(ggimage); library(png); library(magick)
  library(RColorBrewer); library(rcdk); library(rcdklibs)
})


########## 2) Global parameters ##########

# Number of top phyla to highlight (others collapsed into "Other phyla")
TOP_N_PHYLA      <- 10

# Input file names
FILE_META1       <- "metadata1.xlsx"
FILE_META2       <- "metadata2.xlsx"
FILE_META3       <- "metadata3.xlsx"
FILE_TREE        <- "tree_newick.txt"

# Outgroup protein label in the tree
OUTGROUP_PROTEIN <- "NNJ93123.1"

# Figure size (inches) for main ggsave (SVG/PNG/PDF)
FIG_W <- 20
FIG_H <- 22

# Tree line size
TREE_SIZE <- 0.35

# Ring geometry (MIBIG + Multi-domain)
RING_WIDTH   <- 0.14     # radial ring thickness in tree units
FIRST_GAP_CM <- 4.5      # gap from tips in cm (converted later to tree units)

# Fatty acid image placement
IMG_TARGET_H     <- 0.12 # nominal FA height in tree coordinates (not critical now)
IMG_MAX_W        <- 0.12 # nominal max FA width in tree coordinates
ANGLE_FIXED_DEG  <- 90   # rotate FA images (degrees)
FA_POS_FRAC      <- 0.55 # fraction of distance from tip to ring where FA is placed

# Fixed FA width in tree units (all FAs will use exactly this size)
FA_CONST_SIZE <- 0.5    # tweak this if you want larger/smaller FAs

# FA rendering parameters (internal resolution)
FA_SRC_SIZE_PX   <- 3000    # internal image size used for processing
FA_FINAL_PX      <- "1100x" # final size passed to magick::image_resize
FA_RED_COLOR     <- "#D70000" # red color for FA lines

# Overwrite FA PNGs each run?
ALWAYS_OVERWRITE_FA <- TRUE

# Debug flag (prints [DEBUG] messages)
DEBUG_FA <- TRUE

# Ring colors (MIBIG / Non-MIBIG / Multi-domain)
MIBIG_COLOR       <- "#2E006A"
NON_MIBIG_COLOR   <- "#FFFFFF"
MULTI_BLACK       <- "#000000"
MULTI_SINGLE_GRAY <- "#6F6F6F"

# Safer TIFF export parameters (smaller size to avoid memory errors)
TIFF_W_IN  <- 12      # width in inches (smaller than FIG_W)
TIFF_H_IN  <- 13.2    # height in inches (scaled to keep ~same aspect)
TIFF_DPI   <- 600     # dpi (still high enough for publication)

if (DEBUG_FA) {
  message("[DEBUG] Script started: using SMILES + rcdk for FA rendering (no PubChem PNG/SVG).")
}


########## 3) General helper functions ##########

# Normalize protein IDs: trim and remove spaces/underscores
normalize_id <- function(x){
  x %>%
    stringr::str_trim() %>%
    stringr::str_replace_all("[ _]","")
}

# Pick first existing column from a list of names
pick_col <- function(df, cs){
  for (nm in cs){
    if (nm %in% names(df)) return(df[[nm]])
  }
  rep(NA_character_, nrow(df))
}

# Normalize FA labels to format like "C16:0"
normalize_fa_key <- function(x){
  s <- toupper(trimws(as.character(x)))
  s[is.na(s)] <- NA_character_
  s <- gsub("\\s+","",s)
  s <- sub("^C(\\d+)$","C\\1:0",s)      # C16 -> C16:0
  s <- gsub("\\.",":",s)               # C16.1 -> C16:1
  s <- sub("^C0+([1-9]\\d*)(:|$)","C\\1\\2",s) # C016 -> C16
  s[nchar(s)==0] <- NA_character_
  s
}

# Parse FA string "C16:1" to numeric (nC=16, nDB=1)
parse_fa <- function(x){
  s <- normalize_fa_key(x)
  m <- stringr::str_match(as.character(s),"^C(\\d+)(?::(\\d+))?$")
  if (is.na(m[1,1])) return(c(NA_integer_,0L))
  nC  <- suppressWarnings(as.integer(m[1,2]))
  nDB <- suppressWarnings(as.integer(m[1,3]))
  if (is.na(nDB)) nDB <- 0L
  c(nC,nDB)
}

# Close all magick devices (safety)
close_magick_devices <- function(){
  dl <- dev.list()
  if (is.null(dl)) return(invisible())
  for (i in seq_along(dl)){
    if (grepl("magick", names(dl)[i], TRUE)){
      try(grDevices::dev.off(which = dl[[i]]), silent = TRUE)
    }
  }
}

# Safe wrapper for magick::image_resize
safe_resize <- function(img, geom){
  tryCatch(
    image_resize(img, geom, filter = "Point"),
    error = function(e) image_resize(img, geom)
  )
}

# Convert a gap size in cm at outer radius r_ref to tree coordinate offset
cm_to_offset <- function(cm, r_ref, fig_width_in = FIG_W){
  A <- (cm/2.54)/fig_width_in
  (A * r_ref) / (1 - A)
}


########## 4) PubChem helpers (used only to retrieve SMILES) ##########

# Systematic alkanoic acid names (C4–C18), Title Case
alkanoic_name <- function(nC){
  lut <- c(
    `4`  = "Butanoic Acid",
    `5`  = "Pentanoic Acid",
    `6`  = "Hexanoic Acid",
    `7`  = "Heptanoic Acid",
    `8`  = "Octanoic Acid",
    `9`  = "Nonanoic Acid",
    `10` = "Decanoic Acid",
    `11` = "Undecanoic Acid",
    `12` = "Dodecanoic Acid",
    `13` = "Tridecanoic Acid",
    `14` = "Tetradecanoic Acid",
    `15` = "Pentadecanoic Acid",
    `16` = "Hexadecanoic Acid",
    `17` = "Heptadecanoic Acid",
    `18` = "Octadecanoic Acid"
  )
  lut[as.character(nC)]
}

# Additional trivial FA names
fa_name_map_extra <- list(
  "C4:0"  = c("Butyric Acid"),
  "C6:0"  = c("Caproic Acid"),
  "C8:0"  = c("Caprylic Acid"),
  "C10:0" = c("Capric Acid"),
  "C12:0" = c("Lauric Acid"),
  "C14:0" = c("Myristic Acid"),
  "C16:0" = c("Palmitic Acid"),
  "C18:0" = c("Stearic Acid"),
  "C16:1" = c("Palmitoleic Acid"),
  "C18:1" = c("Oleic Acid"),
  "C18:2" = c("Linoleic Acid"),
  "C18:3" = c("Alpha-Linolenic Acid")
)

# Build a vector of synonyms to query PubChem for a given FA
fa_synonyms <- function(nC, nDB){
  key <- sprintf("C%d:%d", nC, nDB)
  extra <- fa_name_map_extra[[key]]
  extra_vec <- if (!is.null(extra)) extra else character(0)
  
  if (nDB == 0){
    nm0 <- alkanoic_name(nC)
    all_syns <- c(extra_vec, tolower(extra_vec), nm0, tolower(nm0))
    syns <- unique(all_syns[!is.na(all_syns) & nzchar(all_syns)])
    syns
  } else {
    if (length(extra_vec)){
      all_syns <- c(extra_vec, tolower(extra_vec))
      unique(all_syns[!is.na(all_syns) & nzchar(all_syns)])
    } else {
      character(0)
    }
  }
}

# Get PubChem CID from compound name
pubchem_get_cid <- function(name){
  u <- paste0(
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/",
    utils::URLencode(name, reserved=TRUE),
    "/cids/TXT"
  )
  cid <- tryCatch(readLines(u, warn = FALSE), error = function(e) NA_character_)
  if (length(cid) == 0) return(NA_character_)
  gsub("[^0-9].*$","", cid[1])
}

# Get canonical SMILES from CID
pubchem_get_smiles <- function(cid){
  u <- paste0(
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/",
    cid,
    "/property/CanonicalSMILES/TXT"
  )
  smi <- tryCatch(readLines(u, warn = FALSE), error = function(e) character(0))
  if (length(smi) == 0) return(NA_character_)
  trimws(smi[1])
}

# Cache for FA PubChem queries
fa_pubchem_cache <- new.env(parent = emptyenv())

# Get PubChem record (CID + SMILES) for FA (nC, nDB)
get_pubchem_record <- function(nC, nDB){
  key <- sprintf("C%02d:%d", nC, nDB)
  if (exists(key, envir = fa_pubchem_cache, inherits = FALSE)){
    return(get(key, envir = fa_pubchem_cache, inherits = FALSE))
  }
  
  syns <- fa_synonyms(nC, nDB)
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][get_pubchem_record] nC=%d, nDB=%d, synonyms=%s",
                    nC, nDB, if (length(syns)) paste(syns, collapse=", ") else "<none>"))
  }
  
  if (!length(syns)){
    res <- list(cid = NA_character_, smiles = NA_character_)
    assign(key, res, envir = fa_pubchem_cache)
    return(res)
  }
  
  cid_used <- NA_character_
  smiles   <- NA_character_
  
  for (nm in syns){
    if (DEBUG_FA) {
      message(sprintf("[DEBUG][get_pubchem_record]   trying name='%s'", nm))
    }
    cid_try <- suppressWarnings(pubchem_get_cid(nm))
    if (!is.na(cid_try) && nzchar(cid_try)){
      if (DEBUG_FA) {
        message(sprintf("[DEBUG][get_pubchem_record]   found CID='%s' for name='%s'", cid_try, nm))
      }
      smi <- pubchem_get_smiles(cid_try)
      if (!is.na(smi) && nzchar(smi)){
        cid_used <- cid_try
        smiles   <- smi
        if (DEBUG_FA) {
          message(sprintf("[DEBUG][get_pubchem_record]   got SMILES='%s' for CID='%s'", smiles, cid_used))
        }
        break
      } else if (DEBUG_FA) {
        message(sprintf("[DEBUG][get_pubchem_record]   no SMILES for CID='%s'", cid_try))
      }
    }
  }
  
  if (is.na(smiles) || !nzchar(smiles)){
    if (DEBUG_FA) {
      message(sprintf(
        "[DEBUG][get_pubchem_record]   WARNING: no SMILES in PubChem for C%d:%d",
        nC, nDB
      ))
    }
  }
  
  res <- list(cid = cid_used, smiles = smiles)
  assign(key, res, envir = fa_pubchem_cache)
  res
}

# Fallback: simple saturated linear chain SMILES if PubChem fails (only for nDB=0)
generate_fa_smiles <- function(nC, nDB){
  if (is.na(nC) || nC < 4L) return(NA_character_)
  if (is.na(nDB) || nDB == 0L){
    smi <- paste0(strrep("C", nC - 1L), "C(=O)O")
    if (DEBUG_FA) {
      message(sprintf("[DEBUG][generate_fa_smiles] Using generated saturated SMILES for C%d:%d: %s",
                      nC, nDB, smi))
    }
    return(smi)
  }
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][generate_fa_smiles] No fallback SMILES for unsaturated FA C%d:%d", nC, nDB))
  }
  NA_character_
}


########## 4.1) Mask extraction for magick ##########

extract_fa_mask <- function(img){
  img <- safe_resize(img, paste0(FA_SRC_SIZE_PX, "x", FA_SRC_SIZE_PX))
  g   <- image_convert(img, colorspace = "gray")
  bw  <- image_threshold(g, type = "white", threshold = "90%")
  image_negate(bw)
}


########## 4.2) Apply red-only style on square canvas (no black outline) ##########

fa_red_nobg <- function(img){
  img_big <- image_convert(img, "PNG")
  img_big <- safe_resize(img_big, paste0(FA_SRC_SIZE_PX, "x", FA_SRC_SIZE_PX))
  
  mask <- extract_fa_mask(img_big)
  red_lines <- image_colorize(img_big, opacity = 100, color = FA_RED_COLOR)
  red_mol <- image_composite(red_lines, mask, operator = "copyopacity")
  
  info <- image_info(red_mol)
  side <- max(info$width, info$height)
  red_square <- image_extent(
    red_mol,
    geometry = paste0(side, "x", side),
    gravity  = "center",
    color    = "transparent"
  )
  
  final <- safe_resize(red_square, FA_FINAL_PX)
  final
}


########## 4.3) Draw FA from SMILES using rcdk ##########

draw_fa_from_smiles <- function(smiles,
                                width  = FA_SRC_SIZE_PX,
                                height = FA_SRC_SIZE_PX){
  if (is.na(smiles) || !nzchar(smiles)) {
    if (DEBUG_FA) {
      message("[DEBUG][draw_fa_from_smiles] Empty or NA SMILES, returning NULL.")
    }
    return(NULL)
  }
  
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][draw_fa_from_smiles] Parsing SMILES: %s", smiles))
  }
  
  mols <- tryCatch(rcdk::parse.smiles(smiles), error = function(e) {
    if (DEBUG_FA) {
      message(sprintf("[DEBUG][draw_fa_from_smiles] ERROR in parse.smiles: %s", e$message))
    }
    NULL
  })
  if (is.null(mols) || length(mols) == 0L || is.null(mols[[1]])) {
    if (DEBUG_FA) {
      message("[DEBUG][draw_fa_from_smiles] Failed to parse SMILES, returning NULL.")
    }
    return(NULL)
  }
  
  mol <- mols[[1]]
  
  try(rcdk::do.aromaticity(mol), silent = TRUE)
  try(rcdk::convert.2d(mol),     silent = TRUE)
  
  depictor <- tryCatch(
    rcdk::get.depictor(
      width  = width,
      height = height,
      zoom   = 1.3,
      style  = "cow",
      annotate  = "off",
      abbr      = "on",
      suppressh = TRUE,
      showTitle = FALSE
    ),
    error = function(e) {
      if (DEBUG_FA) {
        message(sprintf("[DEBUG][draw_fa_from_smiles] ERROR in get.depictor: %s", e$message))
      }
      NULL
    }
  )
  
  img_raster <- tryCatch(
    rcdk::view.image.2d(mol, depictor = depictor),
    error = function(e) {
      if (DEBUG_FA) {
        message(sprintf("[DEBUG][draw_fa_from_smiles] ERROR in view.image.2d: %s", e$message))
      }
      NULL
    }
  )
  if (is.null(img_raster)) {
    if (DEBUG_FA) {
      message("[DEBUG][draw_fa_from_smiles] view.image.2d returned NULL.")
    }
    return(NULL)
  }
  
  tmp <- tempfile(fileext = ".png")
  grDevices::png(tmp, width = width, height = height, bg = "white", res = NA)
  par(mar = c(0, 0, 0, 0))
  plot(as.raster(img_raster))
  grDevices::dev.off()
  
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][draw_fa_from_smiles] Temporary PNG written to %s", tmp))
  }
  
  image_read(tmp)
}


########## 4.4) Generate FA PNG file (SMILES + rcdk only) ##########

download_fa_png <- function(
    nC, nDB,
    outdir = "fa_fatty_acids_png"
){
  if (!dir.exists(outdir)){
    dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  }
  
  outfile <- file.path(outdir, sprintf("FA_C%02d_%d.png", nC, nDB))
  
  if (file.exists(outfile) && !ALWAYS_OVERWRITE_FA){
    if (DEBUG_FA) {
      message(sprintf("[DEBUG][download_fa_png] File already exists, skipping: %s", outfile))
    }
    return(outfile)
  }
  
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][download_fa_png] Generating FA image for C%d:%d", nC, nDB))
  }
  
  rec    <- get_pubchem_record(nC, nDB)
  smiles <- rec$smiles
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][download_fa_png] PubChem result: CID='%s', SMILES='%s'",
                    ifelse(is.na(rec$cid),"NA", rec$cid),
                    ifelse(is.na(smiles),"NA", smiles)))
  }
  
  if (is.na(smiles) || !nzchar(smiles)){
    smiles <- generate_fa_smiles(nC, nDB)
    if (is.na(smiles) || !nzchar(smiles)){
      warning(sprintf(
        "[download_fa_png] Could not get/generate SMILES for C%d:%d; FA will be omitted.",
        nC, nDB
      ))
      return(NULL)
    }
  }
  
  img_raw <- draw_fa_from_smiles(smiles)
  if (is.null(img_raw)){
    warning(sprintf(
      "[download_fa_png] Failed to draw SMILES for C%d:%d; FA will be omitted.",
      nC, nDB
    ))
    return(NULL)
  }
  
  img_proc <- fa_red_nobg(img_raw)
  image_write(img_proc, outfile, "png")
  
  if (DEBUG_FA) {
    message(sprintf("[DEBUG][download_fa_png] FA image written to %s", outfile))
  }
  
  outfile
}

get_hw_units <- function(path, target_h = IMG_TARGET_H, max_w = IMG_MAX_W){
  c(
    size   = FA_CONST_SIZE,
    h      = target_h,
    half_h = target_h / 2
  )
}


########## 5) Read metadata and tree ##########

m1 <- read_xlsx(FILE_META1)
m2 <- read_xlsx(FILE_META2)
m3 <- read_xlsx(FILE_META3)


########## 6) Process and merge metadata ##########

stopifnot("Combined Signature description" %in% names(m1))

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
    MultiDomain = dplyr::case_when(
      !is.na(Signature) & str_detect(Signature,"FAAL-") ~ "Multidomain",
      !is.na(Signature) & str_detect(Signature,"FAAL")  ~ "Single",
      TRUE ~ "Single"
    )
  ) %>%
  tidyr::separate(
    Lineage,
    into = c("Domain","Phylum","Class","Order","Family","Extra1","Extra2"),
    sep = ";", fill = "right", remove = FALSE
  ) %>%
  mutate(
    across(c(Domain,Phylum,Class,Order,Family,Extra1,Extra2), ~trimws(.)),
    Genus = str_extract(Species,"^[^ ]+")
  ) %>%
  distinct(ProtNorm,.keep_all = TRUE)

meta2 <- tibble(
  Assembly = pick_col(m2, c("Assembly Accession","Assembly","Assembly_Accession")),
  Biome    = pick_col(m2, c("BiomeDistribution","Biome Distribution","Biome","Biome_Distribution")),
  Location = pick_col(m2, c("Location","Locality")),
  Latitude = suppressWarnings(as.numeric(pick_col(m2, c("Latitude","Lat")))),
  Longitude= suppressWarnings(as.numeric(pick_col(m2, c("Longitude","Long","Lon"))))
) %>% distinct()

meta3 <- m3 %>%
  rename(
    ProteinRaw     = `Protein Accession`,
    SubstrateLabel = Specificity,
    SubstrateColor = `Label color`
  ) %>%
  mutate(
    ProteinRaw     = str_replace_all(ProteinRaw,"\\s+",""),
    SubstrateLabel = normalize_fa_key(na_if(trimws(SubstrateLabel),"NA")),
    ProtNorm       = normalize_id(ProteinRaw)
  ) %>%
  select(ProtNorm,SubstrateLabel,SubstrateColor) %>%
  distinct(ProtNorm,.keep_all = TRUE)


########## 7) Tree and merged annotation ##########

tree_raw <- read.tree(FILE_TREE)
stopifnot(OUTGROUP_PROTEIN %in% tree_raw$tip.label)
tree <- root(tree_raw, outgroup = OUTGROUP_PROTEIN, resolve.root = TRUE)

tips <- tibble(
  Protein  = tree$tip.label,
  ProtNorm = normalize_id(tree$tip.label)
)

meta_tree <- tips %>%
  left_join(meta1 %>% select(-Protein,-ProteinRaw), by = "ProtNorm") %>%
  left_join(meta2, by = "Assembly") %>%
  left_join(meta3, by = "ProtNorm") %>%
  mutate(
    Biome = ifelse(
      Biome %in% c(NA,"","NA","Na","na","None"),
      NA_character_,
      as.character(Biome)
    )
  )

phylum_counts <- meta_tree %>%
  filter(!is.na(Phylum)) %>%
  count(Phylum, sort = TRUE)

top_phyla <- phylum_counts$Phylum[seq_len(min(TOP_N_PHYLA, nrow(phylum_counts)))]

ann <- meta_tree %>%
  transmute(
    Protein, Assembly, Lineage, Phylum, Class, Order, Family, Genus, Species,
    Signature, MultiDomain, Biome, Location, Latitude, Longitude,
    SubstrateLabel, SubstrateColor,
    MIBIG = if_else(str_detect(Protein,"^BGC"), "MIBIG", "Non-MIBIG"),
    PhylumCollapsed = if_else(
      !is.na(Phylum) & Phylum %in% top_phyla,
      Phylum,
      "Other phyla"
    )
  ) %>%
  distinct(Protein,.keep_all = TRUE) %>%
  slice(match(tree$tip.label, Protein))

rownames(ann) <- ann$Protein
ann$MultiDomain <- factor(ann$MultiDomain, levels = c("Multidomain","Single"))
ann$PhylumCollapsed[ann$Protein == OUTGROUP_PROTEIN] <- "Outgroup"


########## 8) Phylum color palette ##########

phylum_palette <- c(
  "Myxococcota"="#0000ff","Candidatus Riflebacteria"="#008000","Candidatus Tectomicrobia"="#008b8b",
  "Candidatus Eremiobacterota"="#00bfff","Cyanobacteriota"="#00ff4f","Ignavibacteriota"="#00ffff",
  "Euryarchaeota"="#040404","Bacillota"="#073763","Thermodesulfobacteriota"="#15ffff",
  "Deinococcota"="#16537e","Bdellovibrionota"="#1e90ff","Armatimonadota"="#20b2aa",
  "Viridiplantae"="#274e13","Planctomycetota"="#331900","Sar"="#331900","Pseudomonadati"="#34ae8f",
  "NA"="#36abaf","Metazoa"="#36abb2","Actinobacteria"="#37a9c3","Amoebozoa"="#39a6d3",
  "Candidatus Neomarinimicrobiota"="#3ba3ec","Cyanobacteria"="#46a0f4","Abditibacteriota"="#483d8b",
  "Candidatus Aerophobetes"="#4b0082","Candidatus Blackallbacteria"="#556b2f","Deltaproteobacteria"="#5f9ea0",
  "Candidatus Margulisiibacteriota"="#61ae31","Actinomycetota"="#6a329f","Candidatus Deferrimicrobiota"="#719af4",
  "Apicomplexa"="#744700","Candidatus Margulisbacteria"="#7cfc00","Lentisphaerota"="#7fffd4",
  "Rhodothermota"="#808080","candidate division KSB3"="#8096f4","Candidatus Shapirobacteria"="#87cefa",
  "Candidatus Tectimicrobiota"="#89a631","Candidatus Aminicenantes"="#8b008b","Candidatus Woesearchaeota"="#8fbc8f",
  "Candidatus Nealsonbacteria"="#9400d3","Candidatus Binatota"="#9932cc","Candidatus Moduliflexota"="#998ff4",
  "Candidatus Latescibacterota"="#a38cf4","uncultured bacterium"="#a52a2a","Candidatus Cloacimonadota"="#b19b31",
  "Calditrichota"="#b8860b","Fungi"="#bb9731","Chloroflexota"="#c90076","Candidatus Hydrogenedentota"="#cb9131",
  "Nitrospinota"="#d2691e","Chlamydiota"="#dcdcdc","Candidatus Sericytochromatia"="#dd6ef4",
  "Metamonada"="#de8731","Gemmatimonadota"="#deb887","Vulcanimicrobiota"="#e68231","Chlorobiota"="#e6e6fa",
  "Nitrospirota"="#ea9999","Discoba"="#ef7c32","Candidatus Melainabacteria"="#f0fff0","Candidatus Hinthialibacterota"="#f25cf4",
  "Acidobacteriota"="#f47936","Candidatus Moraniibacteriota"="#f562d9","Candidatus Omnitrophota"="#f569ba",
  "Candidatus Rokuibacteriota"="#f66bab","Candidatus Tharpellota"="#f6717e","Haptista"="#f77272",
  "candidate division KSB1"="#f77553","Bacillati"="#f77639","Candidatus Sumerlaeota"="#f8f8ff",
  "Elusimicrobiota"="#faf0e6","Pseudomonadota"="#ff0000","Candidatus Krumholzibacteriota"="#ff00ff",
  "Campylobacterota"="#ff69b4","bacterium"="#ff7f50","Spirochaetota"="#ff8c00","environmental samples"="#ffa07a",
  "Candidatus Rokubacteria"="#ffa312","Verrucomicrobiota"="#ffd34b","Caldisericota"="#fff0f5",
  "candidate division NC10"="#fff2cc","Candidatus Gracilibacteria"="#fff8dc","Candidatus Eisenbacteria"="#fffaf0",
  "Bacteroidota"="#ffff00"
)

phylum_levels_raw <- ann$PhylumCollapsed %>%
  unique() %>%
  na.omit()

other_label <- "Other phyla"
phylum_main <- setdiff(phylum_levels_raw, c(other_label, "Outgroup"))
phylum_levels <- c(phylum_main, other_label)

phylum_colors <- setNames(rep("#8D8D8D", length(phylum_levels)), phylum_levels)
matchable <- intersect(names(phylum_palette), phylum_levels)
phylum_colors[matchable] <- phylum_palette[matchable]

phylum_colors[other_label] <- "#A0A0A0"
phylum_colors["Outgroup"]  <- "#4D4D4D"


########## 9) Base circular tree ##########

p_tree <- ggtree(
  tree,
  layout = "circular",
  size   = TREE_SIZE,
  color  = "black"
) %<+% ann +
  geom_tippoint(
    aes(color = PhylumCollapsed),
    size  = 1.2,
    alpha = 1,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values   = phylum_colors,
    na.value = "grey60"
  ) +
  guides(
    color     = "none",
    size      = "none",
    linewidth = "none",
    alpha     = "none"
  )

r_tip <- max(layer_data(p_tree)$x, na.rm = TRUE)

tips_xy <- p_tree$data %>%
  dplyr::filter(isTip) %>%
  select(label, x, y) %>%
  rename(Protein = label, x_tip = x)


########## 10) Rectangular rings for MIBIG and Multi-domain ##########

first_gap_units <- cm_to_offset(FIRST_GAP_CM, r_ref = r_tip)
w_units         <- RING_WIDTH

ann_mibig <- ann %>%
  transmute(Protein, FillKey = paste0("MIBIG:", MIBIG))

ann_multi <- ann %>%
  transmute(Protein, FillKey = paste0("Multi:", MultiDomain))

ring_df <- function(df, x_inner){
  df %>%
    left_join(tips_xy, by = "Protein") %>%
    mutate(
      xmin = r_tip + x_inner,
      xmax = r_tip + x_inner + w_units,
      ymin = y - 0.5,
      ymax = y + 0.5
    )
}

ring1 <- ring_df(ann_mibig, x_inner = first_gap_units)
ring2 <- ring_df(ann_multi, x_inner = first_gap_units + w_units)

p_tree <- p_tree +
  geom_rect(
    data  = ring1,
    aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fill=FillKey),
    color = NA,
    inherit.aes = FALSE
  ) +
  geom_rect(
    data  = ring2,
    aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fill=FillKey),
    color = NA,
    inherit.aes = FALSE
  )


########## 11) Legend construction ##########

# Legend group keys
TITLE_PHY         <- "TITLE_Phylum"
TITLE_MIBIG       <- "TITLE_MIBIG"
TITLE_MULTI       <- "TITLE_Multi_Architect"
TITLE_BLANK_MIBIG <- "TITLE_Blank_MIBIG"
TITLE_BLANK_MULTI <- "TITLE_Blank_MULTI"

# Outgroup label in legend
OUTGROUP_LAB <- "NNJ93123.1 Halobacteria archaeon"

# Build phylum keys for legend
phylum_keys  <- paste0("Phy:", phylum_levels)
phylum_cols2 <- setNames(phylum_colors[phylum_levels], phylum_levels)
phylum_keys2 <- c(phylum_keys, "Phy:Outgroup")
phylum_cols2 <- c(phylum_cols2, setNames(phylum_colors["Outgroup"], "Outgroup"))

# Keys for MIBIG / Non-MIBIG
mibig_keys <- paste0("MIBIG:", c("MIBIG","Non-MIBIG"))

# Keys for Multi-domain / Single
multi_keys <- paste0("Multi:", c("Multidomain","Single"))

# Mapping FillKey -> colors in legend
# Titles and blanks têm fill = NA (sem cor)
ring_fill_colors <- c(
  setNames(NA, TITLE_PHY),
  setNames(phylum_cols2[c(phylum_levels,"Outgroup")], phylum_keys2),
  setNames(NA, TITLE_BLANK_MIBIG),
  setNames(NA, TITLE_MIBIG),
  setNames(c(MIBIG_COLOR, NON_MIBIG_COLOR), mibig_keys),
  setNames(NA, TITLE_BLANK_MULTI),
  setNames(NA, TITLE_MULTI),
  setNames(c(MULTI_BLACK, MULTI_SINGLE_GRAY), multi_keys)
)

# Labels para a legenda
phylum_labels <- phylum_levels

all_breaks <- c(
  TITLE_PHY,
  phylum_keys2,
  TITLE_BLANK_MIBIG,
  TITLE_MIBIG,
  mibig_keys,
  TITLE_BLANK_MULTI,
  TITLE_MULTI,
  multi_keys
)

all_labels <- c(
  " Phyla",
  phylum_labels,
  OUTGROUP_LAB,
  " ",                 # linha em branco antes de MIBIG
  " MIBIG",
  c("MIBIG","Non-MIBIG"),
  " ",                 # linha em branco antes de Multi-Architecture
  " Multi-Architecture",
  c("Multidomain","Single")
)

# Dummy data para forçar todas as chaves da legenda
dummy_df <- data.frame(
  FillKey = unique(all_breaks),
  x = 0,
  y = 0
)

# Vetores de override para NÃO mostrar quadrinho nos títulos e espaços em branco
n_keys <- length(all_breaks)
shape_vec <- rep(22, n_keys)
size_vec  <- rep(6,  n_keys)
alpha_vec <- rep(1,  n_keys)

no_box_keys <- c(TITLE_PHY, TITLE_BLANK_MIBIG, TITLE_MIBIG,
                 TITLE_BLANK_MULTI, TITLE_MULTI)
idx_nobox <- all_breaks %in% no_box_keys

shape_vec[idx_nobox] <- NA  # sem símbolo
size_vec[idx_nobox]  <- 0   # sem tamanho
alpha_vec[idx_nobox] <- 0   # evita qualquer traço

# Attach legend to plot via scale_fill_manual + dummy layer
p_tree <- p_tree +
  geom_point(
    data  = dummy_df,
    aes(x = x, y = y, fill = FillKey),
    shape = 22,
    size  = 0,
    alpha = 0,
    inherit.aes = FALSE,
    show.legend = TRUE
  ) +
  scale_fill_manual(
    name   = NULL,
    values = ring_fill_colors,
    breaks = all_breaks,
    labels = all_labels,
    limits = all_breaks,
    na.translate = FALSE,
    guide  = guide_legend(
      override.aes = list(
        shape  = shape_vec,
        size   = size_vec,
        alpha  = alpha_vec,
        colour = rep(NA, n_keys),
        stroke = rep(0,  n_keys)
      ),
      ncol  = 1,
      byrow = FALSE,
      order = 1
    )
  ) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position  = "right",
    legend.text      = element_text(size = 12, face = "bold"),
    legend.title     = element_blank(),
    legend.key       = element_blank(),
    legend.background= element_rect(fill = "white", color = "white")
  )

# Save base tree (rings + legend) to reuse before adding FAs
p_tree_base <- p_tree

# Inner radius of first ring (for FA distance calculations)
ring1_inner <- r_tip + first_gap_units


########## 12) Parse FA labels (C:nDB) ##########

ann <- ann %>%
  mutate(
    nC  = as.integer(sapply(SubstrateLabel, function(s) parse_fa(s)[1])),
    nDB = as.integer(sapply(SubstrateLabel, function(s) parse_fa(s)[2])),
    nDB = ifelse(is.na(nDB), 0L, nDB)
  )

if (DEBUG_FA) {
  message(sprintf("[DEBUG] Number of tips with FA labels: %d",
                  sum(!is.na(ann$nC) & !is.na(ann$SubstrateLabel))))
}


########## 13) Generate FA PNGs and add to tree ##########

fa_base_dir <- "fa_fatty_acids_png"
fa_dir <- file.path(fa_base_dir, "red")

message("=== Generating figure for red FAs (SMILES + rcdk, red only, standardized size) ===")

p_tree <- p_tree_base

if (!dir.exists(fa_dir)){
  dir.create(fa_dir, recursive = TRUE, showWarnings = FALSE)
}

fa_list <- ann %>%
  filter(!is.na(nC), !is.na(SubstrateLabel)) %>%
  mutate(img = file.path(fa_dir, sprintf("FA_C%02d_%d.png", nC, nDB))) %>%
  distinct(Protein, .keep_all = TRUE)

if (DEBUG_FA) {
  message(sprintf("[DEBUG] Number of distinct proteins with FA to draw: %d", nrow(fa_list)))
}

if (nrow(fa_list) > 0){
  for (i in seq_len(nrow(fa_list))){
    if (DEBUG_FA) {
      message(sprintf("[DEBUG] Processing FA %d/%d: Protein=%s, SubstrateLabel=%s, C=%d, DB=%d",
                      i, nrow(fa_list),
                      fa_list$Protein[i],
                      fa_list$SubstrateLabel[i],
                      fa_list$nC[i],
                      fa_list$nDB[i]))
    }
    invisible(try(
      download_fa_png(
        fa_list$nC[i],
        fa_list$nDB[i],
        outdir = fa_dir
      ),
      silent = FALSE
    ))
  }
}

close_magick_devices()
gc()

fa_img_df <- ann %>%
  filter(!is.na(nC), !is.na(SubstrateLabel)) %>%
  mutate(img = file.path(fa_dir, sprintf("FA_C%02d_%d.png", nC, nDB))) %>%
  filter(file.exists(img)) %>%
  rowwise() %>%
  mutate(
    vals        = list(get_hw_units(img, IMG_TARGET_H, IMG_MAX_W)),
    img_size    = vals["size"],
    img_h_units = vals["h"],
    img_half_h  = vals["half_h"]
  ) %>%
  ungroup() %>%
  select(-vals)

if (DEBUG_FA) {
  message(sprintf("[DEBUG] Number of FA images found on disk: %d", nrow(fa_img_df)))
}

fa_plot_df <- fa_img_df %>%
  left_join(tips_xy, by = "Protein") %>%
  filter(!is.na(y)) %>%
  mutate(
    dist_tip_to_ring = ring1_inner - x_tip,
    x_img    = x_tip + dist_tip_to_ring * FA_POS_FRAC,
    y_img    = y,
    angle_img = ANGLE_FIXED_DEG
  )

if (DEBUG_FA) {
  message(sprintf("[DEBUG] Number of FA images with valid coordinates: %d", nrow(fa_plot_df)))
}

if (nrow(fa_plot_df) > 0){
  fa_conn_df <- fa_plot_df %>%
    mutate(
      x_start   = x_tip,
      x_end_raw = x_img - img_size * 0.60,
      x_end     = ifelse(
        x_end_raw <= x_start,
        x_start + dist_tip_to_ring * 0.40,
        x_end_raw
      )
    )
  
  fa_color_df <- fa_plot_df %>%
    filter(!is.na(SubstrateLabel)) %>%
    select(SubstrateLabel, SubstrateColor) %>%
    distinct()
  
  missing_cols <- is.na(fa_color_df$SubstrateColor) | fa_color_df$SubstrateColor == ""
  if (any(missing_cols)){
    pal_tmp <- scales::hue_pal()(sum(missing_cols))
    fa_color_df$SubstrateColor[missing_cols] <- pal_tmp
  }
  fa_cols <- setNames(fa_color_df$SubstrateColor, fa_color_df$SubstrateLabel)
  
  p_tree <- p_tree +
    ggnewscale::new_scale_color() +
    geom_segment(
      data  = fa_conn_df,
      aes(
        x = x_start, xend = x_end,
        y = y_img,   yend = y_img,
        colour = SubstrateLabel
      ),
      inherit.aes = FALSE,
      linewidth   = 0.10,
      lineend     = "round"
    ) +
    scale_color_manual(values = fa_cols, guide = "none")
  
  p_tree <- p_tree +
    ggimage::geom_image(
      data  = fa_plot_df,
      aes(
        x     = x_img,
        y     = y_img,
        image = img,
        angle = angle_img,
        size  = img_size
      ),
      inherit.aes = FALSE,
      by = "width"
    ) +
    scale_size_identity()
  
  fa_label_df <- fa_plot_df %>%
    filter(!is.na(SubstrateLabel)) %>%
    mutate(
      x_label = x_img + img_size * 0.40,
      y_label = y_img
    )
  
  p_tree <- p_tree +
    geom_text(
      data  = fa_label_df,
      aes(x = x_label, y = y_label, label = SubstrateLabel),
      inherit.aes = FALSE,
      hjust  = 0,
      vjust  = 0.5,
      size   = 1.5,
      colour = "#222222",
      fontface = "bold"
    )
}

p_all <- p_tree

while (!is.null(dev.list())) dev.off()


########## 14) Export figures (SVG, PNG, TIFF, PDF) ##########

base_name <- "figure_tree_publication_circular_full_FAred_version_simple_nobarplot_redonly_stdsize"

ggsave(
  paste0(base_name, ".svg"),
  p_all,
  width  = FIG_W,
  height = FIG_H,
  device = "svg"
)

ggsave(
  paste0(base_name, ".png"),
  p_all,
  width  = FIG_W,
  height = FIG_H,
  dpi    = 900,
  limitsize = FALSE
)

tiff_try <- try({
  tiff(
    paste0(base_name, ".tiff"),
    width  = TIFF_W_IN,
    height = TIFF_H_IN,
    units  = "in",
    res    = TIFF_DPI,
    compression = "lzw"
  )
  print(p_all)
  dev.off()
}, silent = TRUE)

if (inherits(tiff_try, "try-error")) {
  message("[WARN] Could not create TIFF at ",
          TIFF_W_IN, "x", TIFF_H_IN, " inches, ",
          TIFF_DPI, " dpi. Skipping TIFF export.")
}

ggsave(
  paste0(base_name, ".pdf"),
  p_all,
  width  = FIG_W,
  height = FIG_H,
  device = "pdf"
)

gc()

message("Done: red FAs rendered from SMILES using rcdk (standardized size, only red, no black outline), with transparent background, no barplots, with Phyla/MIBIG/Multi legends and no boxes on legend titles/blank rows.")

