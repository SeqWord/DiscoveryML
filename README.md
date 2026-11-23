# DiscoveryML

**DiscoveryML** is a modular machine-learning pipeline for exploratory hypothesis testing and constructing hierarchical classifiers when predefined ground-truth labels are unavailable. Developed primarily for microbiological, bacterial-genomic and metagenomic applications, it combines metadata-based grouping, pangenomics, unsupervised clustering, feature selection, classifier training and validation.

**Release:** `v1.0.0-beta` · **Status:** Beta

> **Scope:** Initial ecological categories may be inferred from metadata or unsupervised clustering. Subsequent supervised models are trained against these derived labels; they should not be interpreted as independently verified biological ground truth.

## Contents

- [Workflow](#workflow)
- [Installation and dependencies](#installation-and-dependencies)
- [Input conventions](#input-conventions)
- [Pipeline steps](#pipeline-steps)
- [Reproducibility and limitations](#reproducibility-and-limitations)

## Workflow

| Step | Numbered subfolder (logical name) | Executable script(s) | Purpose |
|---|---|---|---|
| 1 | `Data_retrieving` | `ncbi_harvesting.py` | Retrieve NCBI assemblies, metadata and sequences |
| 2 | `ReferenceSelectionByKyewords` | `select_genomes_by_keywords.py` | Assign metadata categories and select representative genomes |
| 3 | `Pangenomics` | `pangenomics.py` | Generate accessory-gene and core-SNP matrices |
| 4 | `Filter_features` | `filter_features.py` | Filter SNPs and create encoded matrices |
| 5 *(optional)* | `PhyloEcoOverlap` | `phylo_ecology_overlap.py` | Test ecology–phylogeny overlap |
| 6 | `Unsupervised_clustering` | `algorithm_selector.py`, `run.py` | Discover initial genomic clusters |
| 7 | `Feature_scoring` | `score_features.py` | Score feature–cluster associations |
| 8 | `Feature_selector` | `select_features.py` | Select diagnostic, nonredundant markers |
| 9 | `feature_calling` | `feature_caller.py` | Call selected markers in additional genomes |
| 10 | `Filter_insensitive_features` | `filter_insensitive_features.py` | Remove poorly called markers and genomes |
| 11 | `Unsupervised_clustering` (second pass) | `algorithm_selector.py`, `run.py` | Cluster the expanded matrix |
| 12 *(optional)* | `Hypothesis_testing` | `hypothesis_testing.py` | Evaluate genotype–group association |
| 13 | `SplitTrainingTesterDatasets` | `split_dataset.py` | Create training and independent test cohorts |
| 14 | `Training` | `model_selector.py`, `run.py` | Select and train hierarchical classifiers |
| 15 | `Testing` | `tester.py`, `per_cluster_statistics.py`, `model_reader.py` | Evaluate and inspect trained models |
| 16 *(optional refinement)* | `Crossgrouping` | `crossgrouping.py` | Add cross-identification groups |

> Folder names above reproduce the supplied names, including `ReferenceSelectionByKyewords` and `Grossgrouping`. Actual numbered directory prefixes were not provided. Run commands from the respective step directory, or supply its path explicitly. Steps 6 and 11 use the same clustering programs in separate passes.

## Installation and dependencies

Use a dedicated virtual environment and install the dependencies required by the modules you intend to run. Python **3.12.3 or later** is the stated baseline for Step 7; compatibility across the entire pipeline must be tested for each release.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install biopython pandas requests lxml numpy scipy scikit-learn statsmodels matplotlib scikit-bio
# Optional clustering backends:
python -m pip install hdbscan python-igraph leidenalg
```

**External software:** Step 3 invokes **Panaroo** (and may require its own annotation/toolchain prerequisites). Step 9 requires an appropriate **NCBI BLAST** backend: `blastn` and `makeblastdb` on Linux/Unix where available, or the documented bundled legacy Windows wrapper. Install and test these separately.

### Dependency summary by module

The table distinguishes **directly visible imports** from **documented or likely runtime requirements**. Python standard-library modules (`argparse`, `csv`, `os`, `pathlib`, etc.) are not listed. Some scripts import local modules whose transitive requirements were not supplied.

| Step | Script(s) | Third-party Python dependencies | External tools / notes |
|---|---|---|---|
| 1 | `ncbi_harvesting.py` | `pandas`, `requests`, `biopython`, `lxml` | NCBI Entrez/network access |
| 2 | `select_genomes_by_keywords.py` | None in supplied imports | Standard library |
| 3 | `pangenomics.py` | `pandas`, `biopython` | **Panaroo** and its required toolchain |
| 4 | `filter_features.py` | `biopython` | — |
| 5 | `phylo_ecology_overlap.py` | `numpy`, `biopython` | — |
| 6 | `algorithm_selector.py` | None in supplied imports | Additional imports may exist in omitted code |
| 6, 11 | `run.py` | `numpy`, `scikit-learn`; optional `hdbscan`, `python-igraph`, `leidenalg` | Local `algorithms.py`; MCL described as NumPy implementation |
| 7 | `score_features.py` | `numpy`, `pandas`, `scipy`, `statsmodels`; documented: `scikit-learn`, `matplotlib`, `scikit-bio` | Analysis backends may use additional modules |
| 8 | `select_features.py` | None in supplied imports | RF evaluation (`--evaluate`) may require additional dependencies |
| 9 | `feature_caller.py` | `biopython` | BLAST+ or bundled legacy BLAST tools |
| 10 | `filter_insensitive_features.py` | None in supplied imports | Standard library |
| 12 | `hypothesis_testing.py` | `numpy`, `pandas`, `scipy`, `scikit-learn`; plot generation suggests `matplotlib` | — |
| 13 | `split_dataset.py` | None in supplied imports | Standard library |
| 14 | `model_selector.py`, `run.py` | `numpy`, `pandas`, `scikit-learn` | Local selector/model/bundle modules; XGBoost options are exposed, but installation/use of `xgboost` is not established by supplied imports |
| 15 | `tester.py`, `per_cluster_statistics.py` | `numpy`, `pandas`; model loading may require `scikit-learn` and training-time dependencies; plotting may require `matplotlib` | Compatible local bundle classes are essential |
| 15 | `model_reader.py` | No third-party direct imports shown; model unpickling may need training-time dependencies | Requires compatible bundle definition |
| 16 | `crossgrouping.py` | None in supplied imports | Standard library |

**Versions explicitly documented for Step 7:** Python 3.12.3, scikit-learn 1.7.0, statsmodels 0.14.5, matplotlib 3.10.1, SciPy 1.15.2, pandas 2.2.3, NumPy 2.2.3, and scikit-bio (unspecified latest). These are **reported minimums, not a verified compatible lockfile**. Record a tested environment for each release (e.g. `python -m pip freeze > requirements-lock.txt`).

## Input conventions

Most steps use project-specific paths:

```text
input/
  MyProject/
    metadata.tsv
    keywords.tsv
    core_snps.tsv
output/
  MyProject/
tmp/
  MyProject/
```

Some steps expect files to be copied from an earlier step's `output/MyProject/` into the next step's `input/MyProject/`. This README does **not** assume automatic inter-step file transfer.

Hierarchical matrix row identifiers generally follow:

```text
Category1|Category2|GenomicCluster|GenomeID
```

The final field is the genome/sample identifier; preceding fields are hierarchy labels. **Check each script's level numbering:** Steps 5 and 12 interpret levels according to their own documented rules, and Step 12 counts hierarchy levels from the *bottom*.

Use a consistent delimiter (`|` by default), stable genome identifiers, and matching feature columns across training and testing. Examples below use placeholder file names where exact output naming cannot be verified from the supplied code.

## Pipeline steps

### Step 1 — Data retrieving

**Script:** `ncbi_harvesting.py`

Searches NCBI Assembly by TaxID or assembly query; compiles Assembly, BioSample, BioProject and linked PubMed metadata; optionally downloads compressed genomic FASTA (`.fna.gz`) and GenBank (`.gbff.gz`) files.

```bash
python ncbi_harvesting.py --email researcher@example.org \
  --assembly-term "Bacillus subtilis group[Organism]" \
  --outdir ncbi_bsubtilis --limit 200 --download-fna --download-gbff

python ncbi_harvesting.py --email researcher@example.org \
  --taxid 65393 --latest --prefer-refseq \
  --outdir ncbi_taxid65393 --limit 500 --download-fna
```

| Argument | Meaning / default |
|---|---|
| `--email` | **Required** NCBI contact email |
| `--api-key` | Optional NCBI API key; default empty |
| `--taxid` | Taxonomy ID; default `0` |
| `--assembly-term` | Assembly search expression; default empty |
| `--limit` | Maximum retrieved assemblies; `20000` |
| `--outdir` | Output directory; `ncbi_bacillus_harvest` |
| `--prefer-refseq` | Prefer RefSeq FTP path |
| `--latest` | Latest assemblies only |
| `--only-refseq`, `--only-genbank` | Source filters; use appropriately |
| `--assembly-level` | Text filter, e.g. `Complete Genome` |
| `--download-fna`, `--download-gbff` | Enable corresponding downloads |
| `--max-download` | Maximum downloaded assemblies; `20000` |
| `--exclude-surveillance` | Optional surveillance exclusion toggle |

**Dependencies:** `biopython`, `pandas`, `requests`, `lxml`. Network access and NCBI usage limits apply.

### Step 2 — Reference selection by keywords

**Script:** `select_genomes_by_keywords.py`

Assigns hierarchical ecological/metadata categories through editable keyword rules, then samples representative genomes per terminal category. Rule columns: `Category_1` … `Category_N`, `Metadata_fields`, `Match_type` (`exact`/`contains`), `Keyword`. Multiple metadata fields are separated by `;` and combined with OR logic; all matching rules contribute category paths.

```bash
python select_genomes_by_keywords.py MyProject \
  -s ../genomes -f metadata.tsv -k keywords.tsv \
  -n 5 -r 41 -t folders

python select_genomes_by_keywords.py MyProject \
  -s ../genomes -f metadata.tsv -k keywords.tsv \
  -t list -L selected_genomes.txt -d "|"
```

| Argument | Meaning / default |
|---|---|
| `project` | **Required** project subfolder |
| `-i, --input_folder` / `-o, --output_folder` | Roots; `input` / `output` |
| `-s, --source_folder` | **Required** recursive genome source |
| `-f, --metadata_file` | **Required** metadata file in project input |
| `-k, --keywords` | Rule file; `keywords.tsv` |
| `-t, --output_format` | `folders` (default) or `list` |
| `-d, --level_delimiter` | List hierarchy separator; `|` |
| `-L, --output_list_file` | `selected_genomes.txt` |
| `-n, --genomes_per_category` | Maximum per terminal category; `5` (minimum `1`) |
| `-r, --random_seed` | Reproducible sampling; `41` |

**Dependencies:** Python standard library.

### Step 3 — Pangenomics

**Script:** `pangenomics.py`

Runs Panaroo-based bacterial pangenomics to generate an accessory-gene presence/absence matrix, core-genome SNP matrix and reference-based SNP annotations.

```bash
python pangenomics.py MyProject -f folders -m 5

python pangenomics.py MyProject -f list \
  -L selected_genomes.txt -s ../genomes \
  -r reference_genomic.gbff.gz --rerun_panaroo
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | Project name (parser marks positional as optional) |
| `-i, --input_folder` / `-o, --output_folder` / `-t, --tmp_folder` | `input` / `output` / `tmp` |
| `-f, --input_format` | `folders` (default) or `list` |
| `-L, --list_file` | Required in list mode; rows like `Category1|Category2|genome_filename` |
| `-s, --source_folder` | Original genome source; required in list mode |
| `-m, --minimal_diversity` | Gene-frequency percentage; `5` retains genes at >5% and <95% frequency |
| `-r, --reference` | Reference GBFF; default first GBFF |
| `--rerun_panaroo` | Force Panaroo rerun |

**Dependencies:** `pandas`, `biopython`, external **Panaroo**. Confirm Panaroo-compatible input annotations and external prerequisites.

### Step 4 — Filter features

**Script:** `filter_features.py`

Filters unidentified and nonsynonymous-related SNP features as implemented by the script, and creates character or binary matrices plus optional FASTA representations. The supplied description says “non-sense SNPs”; verify whether the implementation retains nonsynonymous substitutions, excludes nonsense substitutions, or applies another rule before claiming a specific biological filter.

```bash
python filter_features.py MyProject \
  -f core_snps.tsv -a core_snp_positions.tsv \
  -t digits --no-fasta
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-f, --input_matrix_file` | `core_snps.tsv` |
| `-a, --input_annotation_file` | `core_snp_positions.tsv` |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-t, --output_format` | `digits` (default) or `characters` |
| `-x, --no_fasta` / `--no-fasta` | Disable FASTA output; default false |

**Dependencies:** `biopython`.

### Step 5 — Phylogeny–ecology overlap (optional)

**Script:** `phylo_ecology_overlap.py`

Measures within-category mean pairwise patristic distance (MPD), compares it with permuted labels, and reports standardized effect size (SES), one-sided permutation P value, between-category MPD and overlap effect. Positive SES indicates stronger-than-random phylogenetic clustering by ecology.

```bash
python phylo_ecology_overlap.py MyProject \
  -t core_snps_filtered.nwk -L 1,2 \
  -n 10000 -s 12345
```

| Argument | Meaning / default |
|---|---|
| `project` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-t, --tree` | **Required** Newick filename |
| `-L, -l, --level` | **Required** comma-separated ecological levels |
| `-n, --permutations` | `10000` |
| `-s, --seed` | `12345` |
| `-d, --delimiter` | `|` |
| `-f, --output_file` | Optional TSV output stem |

**Dependencies:** `numpy`, `biopython`.

### Step 6 — Initial unsupervised clustering

**Scripts:** `algorithm_selector.py`, `run.py`

The selector recommends an algorithm and parameters; it **does not perform clustering**. The runner executes one of `SA`, `DBSCAN`, `HDBSCAN`, `LEIDEN`, `MCL`, `AGGLO`, or `KMEANS`. Output can include a cluster-annotated matrix, text report and/or Newick tree.

```bash
python algorithm_selector.py MyProject -m core_snps_digits.csv \
  -t genome_features -u binary

python run.py MyProject core_snps_digits.csv HDBSCAN \
  --min_cluster_size 5 --output_format matrix \
  --remove_noise_rows true
```

**Selector arguments**

| Argument | Meaning / default |
|---|---|
| `project` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-m, -f, --input_file` | Optional matrix; otherwise inspect CSV/TSV files |
| `--label_delimeter`, `--label_delimiter` | Label separator; `|` |
| `-a, --algorithm` | Force `SA`, `DBSCAN`, `HDBSCAN`, `LEIDEN`, `MCL`, `AGGLO`, `KMEANS`; default auto-select |
| `-t, --task` | `auto`, `kmer`, `metagenome_abundance`, `expression`, `genome_features`; `auto` |
| `-u, --units` | `auto`, `continuous`, `binary`, `categorical`; `auto` |
| `-s, --size` | `auto`, `small`, `average`, `large`; `auto` |
| `-l, --kmer_length` | `na`, `short`, `average`, `long`; `na` |
| `-c, --kmer_encoding` | `na`, `continuous`, `zscore5`, `qual5`; `na` |
| `-b, --abundance_encoding` | `na`, `binary`, `continuous`; `na` |
| `-e, --expression_encoding` | `na`, `continuous`, `zscore5`, `qual5`; `na` |
| `-d, --square_distance` | Treat input as a square distance matrix |
| `--show_help` | Extended usage information |

**Runner: common arguments**

| Argument | Meaning / default |
|---|---|
| `project`, `input_file`, `algorithm` | **Required** positionals |
| `-i, --input_directory` / `-d, --output_directory` | `input` / `output` |
| `-f, --output_format` | `all`, `newick`, `text`, `matrix`, `pathway`; default `matrix` |
| `-o, --out` | Optional output basename |
| `-l, --delimiter` | `|` |
| `-m, --max_cluster_levels` | `1` |
| `--seed` | `13` |
| `--scale` | Standardize features |
| `--transform` | `none` (default) or `clr` |
| `--pca`, `--svd` | Embedding dimensions; both `0` (off) |
| `--remove_noise_rows` | Optional boolean; default false |
| `--empty_symbol` | Missing-value symbol; empty |
| `--remove_empty_filed` | Empty-column threshold; `1.0` |
| `--show_help` | Extended help |

**Runner: algorithm-specific arguments**

| Algorithm | Options (defaults) |
|---|---|
| SA | `--metric` (`jaccard`/`euclidean`, unset), `--similarity gaussian`, `--sigma` (unset), `--epsilon` (unset), `--expected_k` (unset), `--max_cluster_number 5`, `--max_cluster_content 5`, `--bootstrap 0`, `--boot_mode replace`, `--boot_frac 0.1` |
| DBSCAN | `--eps 0.5`, `--min_samples 5`, `--dbscan_metric` (`euclidean`/`jaccard`/`braycurtis`, unset) |
| HDBSCAN | `--min_cluster_size 5`, `--hdbscan_metric euclidean`, `--cluster_selection_epsilon 0.0`, `--allow_single_cluster` |
| LEIDEN / MCL | `--knn_k 30`, `--graph_metric euclidean`, `--resolution 1.0`, `--from_distance` |
| MCL | `--inflation 2.0`, `--expansion 2`, `--prune_threshold 1e-6`, `--mcl_max_iter 100` |
| AGGLO | `--agg_linkage average`, `--agg_metric euclidean` |
| KMEANS | `--kmeans_k` (unset), `--n_init 10`, `--max_iter 300` |

**Dependencies:** `numpy`, `scikit-learn`; `hdbscan` for HDBSCAN; `python-igraph` and `leidenalg` for Leiden; local `algorithms.py`.

### Step 7 — Feature scoring

**Script:** `score_features.py`

Uses chi-square association tests and false-discovery-rate (FDR) correction to score polymorphic features. The supplied help also describes group-separation analyses using Random Forest, PERMANOVA or Mantel methods; however, **no argument selecting those algorithms appears in the supplied parser**, so the actual execution behavior must be checked.

```bash
python score_features.py MyProject clustered_matrix.csv \
  -p 0.05 -s true -d "|"
```

| Argument | Meaning / default |
|---|---|
| `project`, `input_file` | Optional positionals in parser; provide both for normal use |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-d, --delimiter` | `|` |
| `-p, --p_value` | Raw P-value threshold; unset |
| `-s, --significance` | Boolean for FDR-significant output; unset |
| `--empty_symbol` | Missing-value symbol; empty |
| `--remove_empty_filed` | Missing-column threshold; `1.0` |

**Dependencies:** `numpy`, `pandas`, `scipy`, `statsmodels`; additional documented `scikit-learn`, `matplotlib`, `scikit-bio`. **Note:** supplied help incorrectly names `run.py`; use `score_features.py`.

### Step 8 — Diagnostic feature selection

**Script:** `select_features.py`

Selects diagnostic SNP/probe markers from feature annotation, chi-square/FDR predictions, genomic proximity and redundancy of allele-state patterns.

```bash
python select_features.py MyProject \
  -m clustered_matrix_filtered_matrix.csv \
  -c clustered_matrix_chi2_fdr_results.csv \
  -f core_snp_positions_filtered.tsv \
  -a core_digit_alleles_filtered.csv \
  -p 0.05 -d 0.05 -g 0.05 \
  -s 10 -r 5 -x 160 -n 500 \
  --evaluate 50,100,200
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-m, --matrix_file` | **Required** allelic-state matrix |
| `-c, --prediction_file` | **Required** chi-square/FDR results |
| `-f, --feature_file` | **Required** annotated features |
| `-a, --allele_file` | **Required** common allelic states |
| `-p, --p_value` | Maximum raw P value; `0.05` |
| `-d, --fdr_value` | Maximum level FDR; `0.05` |
| `-g, --global_fdr_value` | Maximum global FDR; `0.05` |
| `-s, --similarity_threshold` | Max mismatching allelic-state percentage in redundancy group; `10.0` (0–50); typo alias `--similarity_threshols` |
| `-r, --redundancy` | Maximum retained SNPs per redundancy cluster; `5` |
| `-x, --minimum_distance_between_SNP` | Minimum reference distance; `160` bp |
| `-n, --feature_number_to_keep` | Maximum selected features; `500` |
| `--evaluate` | Optional comma-separated marker counts for RF accuracy evaluation |

**Dependencies:** Standard library in supplied imports; evaluate-mode dependencies need verification.

### Step 9 — Feature calling in additional genomes

**Script:** `feature_caller.py`

Uses sequence-context searches and BLAST to call marker states in FASTA/GenBank genomes. `character` returns a base, `digit` encodes reference/non-reference as `0/1`, and `probe` reports presence within a mismatch threshold.

```bash
python feature_caller.py MyProject selected_features.tsv \
  called_features.csv -s ../genomes \
  -y digit -a selected_alleles.csv -T 8 -m 6

python feature_caller.py MyProject probes.tsv \
  probe_calls.csv -s ../genomes -y probe -T 8
```

| Argument | Meaning / default |
|---|---|
| `project_folder`, `input_file`, `output_file` | Positional project, annotation/probe file, output filename |
| `-s, --source_folder` | Genome source; required in normal use |
| `-i, --input_folder` / `-o, --output_folder` / `-t, --tmp_folder` | `input` / `output` / `tmp` |
| `-a, --allele_file` | Required for `digit` mode |
| `-m, --mismatch` | Maximum BLAST substitutions; `6` |
| `-L, --context_shoulder` | Called-base query position; `0` means center |
| `-y, --data_type` | `character`, `digit` (default), `probe` |
| `-n, --no_data_mark` | Missing-call symbol; `n` |
| `-F, --flash_after_genome_number` | Flush interval; `50`, `0` disables |
| `-f, --filter_using_metadata_file` | Optional metadata filtering file |
| `-k, --keywords` | Rules for metadata filtering; `keywords.tsv` |
| `-d, --level_delimiter` | `|` |
| `-T, --threads` | Parallel locus/BLAST workers; `8` |
| `-h, --help` | Detailed help |

**Dependencies:** `biopython`, BLAST backend. Avoid oversubscribing CPUs when running many parallel workers.

### Step 10 — Filter insensitive features

**Script:** `filter_insensitive_features.py`

Removes genomes and features exceeding missing-data thresholds, and optionally replaces missing symbols.

```bash
python filter_insensitive_features.py MyProject \
  -m called_features.csv \
  -f selected_features.tsv \
  -a selected_alleles.csv \
  -t digit -x 10 -z 10 -y n -r 1
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-m, --matrix_file` | **Required** allele matrix |
| `-f, --feature_file` | **Required** annotation |
| `-a, --allele_file` | Required for digit data |
| `-t, --data_type` | `digit` (default) or `character` |
| `-x, --no_data_per_feature_threshold` | Max missing percentage per feature; `10` |
| `-z, --no_data_per_genome_threshold` | Max missing percentage per genome; `10` |
| `-y, --no_data_symbol` | `n` |
| `-r, --no_data_replacement` | **Actual parser default `1`** (help text inconsistently says `0`) |
| `-n, --feature_number_to_keep` | Maximum retained features; `0` = no limit |

**Dependencies:** Python standard library. Choose missing-data replacement carefully; replacing missing values with a valid allele can bias analysis.

### Step 11 — Expanded unsupervised clustering

**Scripts:** `algorithm_selector.py`, `run.py` (same interface as Step 6).

Repeat clustering on the larger matrix generated after feature calling and missing-data filtering. The resulting hierarchy is used for hypothesis testing (optional) and training/test splitting.

```bash
python algorithm_selector.py MyProject -m expanded_filtered.csv \
  -t genome_features -u binary

python run.py MyProject expanded_filtered.csv HDBSCAN \
  --min_cluster_size 10 --output_format matrix
```

**Arguments and dependencies:** See [Step 6](#step-6--initial-unsupervised-clustering).

### Step 12 — Hypothesis testing (optional)

**Script:** `hypothesis_testing.py` · script version `3.0`

Tests genotype/group separation using RF label randomization, PERMANOVA on Jaccard distances and Mantel correlation. Expects a hierarchical sample identifier in column 1 and binary SNP/features in subsequent columns.

```bash
python hypothesis_testing.py MyProject \
  -m expanded_clustered.csv -L 1,2 \
  -a ALL -n 100 -t 100 --seed 42 --no-show
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-m, -f, --input_matrix_file, --input_file` | Optional file; if omitted, exactly one CSV/TSV must exist |
| `-n, --replicates` | Permutation/randomization replicates; `100` |
| `-L, --level` | Bottom-up levels; `1,2` |
| `-a, --algorithm` | `ALL`, `RF`, `BOOTSTRAP`, `PERMANOVA`, `MANTEL`; `ALL` |
| `--seed` | `42` |
| `-t, --trees` | RF trees; `100` |
| `--no-show` | Save plot without opening window |
| `-v, --version` | Print script version |

**Dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`; likely `matplotlib` for plots. Permutation results depend on replicate count and seed.

### Step 13 — Split training and test datasets

**Script:** `split_dataset.py`

Splits a labeled matrix into training and held-out test cohorts; optionally balances training cluster sizes and excludes noise identified by a clustering report.

```bash
python split_dataset.py MyProject \
  -m expanded_clustered.csv \
  -t 30 -b -u 1.5 \
  -f clustering_report.txt
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-m, --matrix_file` | **Required** matrix |
| `-t, --tester_dataset_proportion` | Test percentage; `30` |
| `-b, --balanced_training_dataset` | Enable training balance |
| `-u, --unbalancing_order` | Max largest/smallest cluster ratio; `1.5` |
| `-f, --filter_noise` | Optional clustering report with NOISE section |
| `-d, --delimiter` | `|` |

**Dependencies:** Python standard library. Keep related or duplicate genomes from leaking across training/test cohorts where possible.

### Step 14 — Hierarchical classifier training

**Scripts:** `model_selector.py`, `run.py`

The selector recommends supervised models for hierarchical categorical marker matrices. The runner trains node-specific models (`AUTO`) or a selected fixed algorithm, computes allele profiles and packages a deployable classifier bundle.

```bash
python model_selector.py MyProject \
  -f training_matrix.csv --cv_folds 5 --top_alternatives 3

python run.py MyProject AUTO \
  -m training_matrix.csv \
  -f selected_features.tsv \
  -a selected_alleles.csv \
  --min_cluster_size 10 --max_entropy 0.80 \
  --author "Research team" --version "1.0.0-beta"
```

**Selector arguments**

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-f, --input_file` | Matrix; empty means analyze all CSV/TSV |
| `-d, --delimiter` | `|` |
| `--empty_symbol` | Additional missing symbol; empty |
| `--remove_empty_field` (`--remove_empty_filed`) | Missing-column threshold; `1.0` |
| `--cv_folds` | Max stratified CV folds; `5` |
| `--top_alternatives` | Alternative model count; `3` |

**Runner arguments: core and hierarchy**

| Argument | Meaning / default |
|---|---|
| `project_folder`, `algorithm` | **Required**; `AUTO` or an algorithm in `SUPPORTED_ALGORITHMS` |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-m, --matrix_file, --input_file` | **Required** training matrix |
| `-f, --feature_file` | **Required** annotations |
| `-a, --allele_file, --allele_matrix` | **Required** two-row common-allele file |
| `-d, --delimiter` | `|` |
| `--marker_style` | Only `plain` |
| `--empty_symbol` | Empty |
| `--remove_empty_field` (`--remove_empty_filed`) | `1.0` |
| `--cv_folds` | `5` |
| `--author`, `--description`, `--version`, `--contact_details` | Optional bundle metadata |
| `--min_cluster_size` | Minimum node size for ML model; `10` |
| `--max_entropy` (`--max_entroby`) | Maximum normalized prediction entropy before allele-profile fallback; `0.80` |

**Runner arguments: algorithm-specific overrides** (all default to `None` unless specified)

| Model family | Arguments |
|---|---|
| Logistic regression / SVC | `--C`, `--penalty`, `--solver`, `--max_iter`, `--class_weight`, `--kernel`, `--degree`, `--gamma`, `--coef0` |
| Random Forest / Decision Tree | `--n_estimators`, `--max_depth`, `--max_features`, `--min_samples_split`, `--min_samples_leaf`, `--bootstrap`, `--oob_score`, `--criterion`, `--splitter` |
| Multilayer perceptron | `--hidden_layer_sizes`, `--activation`, `--alpha`, `--batch_size`, `--learning_rate`, `--learning_rate_init`, `--early_stopping` |
| K-nearest neighbors | `--n_neighbors`, `--weights`, `--metric`, `--p` |
| Naive Bayes | `--fit_prior` |
| XGBoost-style options | `--subsample`, `--colsample_bytree`, `--reg_lambda`, `--reg_alpha`, `--min_child_weight`, `--scale_pos_weight` |

The precise values of `SUPPORTED_ALGORITHMS` were not supplied. The visible imports include Random Forest, logistic regression, categorical Naive Bayes, kNN, MLP, SVC and decision trees; do not assume XGBoost is enabled without checking the implementation.

**Dependencies:** `numpy`, `pandas`, `scikit-learn` plus the repository's compatible local model-selector and bundle modules.

### Step 15 — Model testing and inspection

**Scripts:** `tester.py`, `per_cluster_statistics.py`, `model_reader.py`

Test a serialized hierarchical classifier on a held-out, already-called marker matrix; summarize per-cluster performance; inspect bundle contents. The test matrix must contain known comparison labels, although the original grouping may have been derived without external ground truth.

```bash
python tester.py MyProject \
  -m tester_matrix.csv -b model.pkl \
  --sensitivity 0.85 --specificity 0.70 \
  --report detailed --show_graph

python per_cluster_statistics.py MyProject \
  -m tester_matrix.csv -b model.pkl \
  -l 2 --show_graph percent

python model_reader.py MyProject --model_file model.pkl
```

**`tester.py` arguments**

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-m, --matrix_file` | **Required** held-out matrix |
| `-d, --delimiter` | `|` |
| `-b, --model_file` | `model.pkl`; relative to script unless absolute |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `--max_no_data_for_NN` | Max missing percentage for ML prediction; `20` |
| `--max_no_data_allowed` | Max missing percentage before no identification; `50` |
| `--sensitivity` | Terminal identification score threshold; `0.85` |
| `--specificity` | Intermediate branch score threshold; `0.70` |
| `--report` | `short` (default) or `detailed` |
| `--report_level` | Lowest labels shown in short report; `2` |
| `--show_graph` | Display heatmap as well as save PNG |

**`per_cluster_statistics.py` arguments**

Shares `project_folder`, `-m`, `-d`, `-b`, `-i`, `-o`, `--max_no_data_for_NN` (`20`), `--max_no_data_allowed` (`50`), `--sensitivity` (`0.85`), and `--specificity` (`0.70`) with `tester.py`. Additional options:

| Argument | Meaning / default |
|---|---|
| `-l, -L, --level, --cluster_level` | Hierarchy level from bottom; `2` |
| `--show_graph` | `no` (default), `count`, or `percent`; both PNG heatmaps saved |

**`model_reader.py` arguments**

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-d, --delimiter` | `|` |
| `-m, --model_file` | `model.pkl`; exact relative-path handling should be checked |

**Dependencies:** `numpy`, `pandas`; trained-model dependencies (especially compatible `scikit-learn`), local model-bundle definitions, and potentially `matplotlib` for heatmaps. The supplied bundle interface is **version 2**. Python pickle bundles should only be loaded from trusted sources and may not work across incompatible code/library versions.

### Step 16 — Cross-group refinement (optional)

**Script:** `crossgrouping.py`

Uses successful terminal predictions from one or more cross-prediction reports to insert a new `CrossGroup_N` level under the original top cluster. `rearrange` builds cross-groups; `prune` applies branch-size and optional unidentified-genome filtering.

```bash
python crossgrouping.py MyProject \
  -m training_matrix.csv \
  -c cross_predictions_1.txt cross_predictions_2.txt \
  --mode rearrange -b MyProject

python crossgrouping.py MyProject \
  -m training_matrix.csv -c cross_predictions_1.txt \
  --mode prune -s 10 --remove_unidentified
```

| Argument | Meaning / default |
|---|---|
| `project_folder` | **Required** project |
| `-i, --input_folder` / `-o, --output_folder` | `input` / `output` |
| `-b, --output_base` | Optional basename; otherwise project-derived filenames |
| `-m, --matrix_file` | **Required** hierarchical matrix |
| `-c, --cross_prediction_files` | **Required** one or more report filenames |
| `-d, --delimiter` | `|` |
| `--mode` | `rearrange` (default) or `prune` |
| `-s, --min_size` | Minimum branch size in prune mode; `10` |
| `--remove_unidentified` | Drop genomes lacking successful predictions in prune mode |

Successful terminal predictions are those whose final report field is `True`; otherwise the cross-group signature is `Not identified`. Expected default outputs are a cross-group matrix and summary TSV. With `-b`, outputs use `<output_base>_crossgroup.tsv` and `<output_base>_summary.tsv`.

**Dependencies:** Python standard library.

## Reproducibility and limitations

1. **Ground-truth-free does not mean validation-free.** Categories inferred from metadata or clustering may be uncertain. Report whether validation measures agreement with derived labels or with independently established biological categories.
2. **Use a held-out test cohort.** Avoid selecting features or tuning model settings using test data. When genomes are closely related, consider lineage-aware or group-aware splitting.
3. **Record exact software versions.** In particular, pickle-based scikit-learn classifiers are sensitive to version changes. Archive the tested Python environment with each GitHub release.
4. **Preserve input provenance.** Retain NCBI accession IDs, download dates, reference genome details, keyword rules, seeds and pipeline parameters.
5. **Check paths and filenames.** Examples use illustrative names. Actual outputs and file movement between numbered directories should be verified against the complete scripts.
6. **Review source-code discrepancies.** The provided excerpts contain a few likely copy/paste errors or inconsistent help strings (e.g., `mport`, `rom`, duplicate `__future__` imports, misspelled `--remove_empty_filed`, and the Step 10 replacement default). These should be checked before release.

## License and citation

Add the repository's actual `LICENSE` file and a citation for the specific tagged release. For reproducible scientific use, cite both the software release and the publication describing the method when available.
