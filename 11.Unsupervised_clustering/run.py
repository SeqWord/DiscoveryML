#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MASTER RUNNER for multiple unsupervised clustering algorithms.

Usage:
  py run.py input_file.(csv|tsv) ALGORITHM [options]

Behavior (common):
- Reads ./input/<input_file.(csv|tsv)> with header row and first-column row labels.
- Outputs can be:
    newick -> .nwk
    text   -> .txt
    matrix -> .csv/.tsv (original table + Cluster column)

Algorithms supported (implemented as classes in algorithms.py):
  SA      - Spectral analysis clustering (hierarchical spectral)
  DBSCAN  - Density-based clustering (sklearn)
  HDBSCAN - Hierarchical density-based clustering (hdbscan package)
  LEIDEN  - Leiden community detection on kNN graph (igraph+leidenalg)
  MCL     - Markov Clustering on kNN similarity graph (pure numpy)
  AGGLO   - Agglomerative hierarchical clustering (average/UPGMA, ward)
  KMEANS  - KMeans clustering (sklearn)

Notes:
- Some algorithms require extra packages:
    HDBSCAN: pip install hdbscan
    LEIDEN:  pip install python-igraph leidenalg
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse
import csv
import re

# Add lib directory to path

path = os.path.abspath(
    os.path.join(os.getcwd(), "..", "lib")
)
sys.path.append(path)

'''
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, "..", "lib"))
sys.path.append(lib_dir)
'''

import clustering_algorithms as algorithms

date_of_creation = "August 14, 2026"
__version__ = "1.0"


def show_help():
    help_text = f"""
    {'='*80}
    UNSUPERVISED CLUSTERING ALGORITHMS (MASTER RUNNER)
        📅 Created: {date_of_creation};
        ‍💻 Version: {__version__};
        👨 Author: Oleg Reva (oleg.reva@up.ac.za)
           Centre for Bioinformatics and Computational Biology,
           BGM, University of Pretoria, South Africa

    🚀 Usage:
        py run.py <input.csv|input.tsv> <ALGORITHM> [options]

    ✅ Algorithms:
        SA      - Spectral Analysis (hierarchical spectral clustering)
        DBSCAN  - Density clustering (sklearn)
        HDBSCAN - Hierarchical density clustering (requires hdbscan)
        LEIDEN  - Leiden on kNN graphs (requires python-igraph + leidenalg)
        MCL     - Markov clustering on similarity graphs
        AGGLO   - Agglomerative hierarchical (UPGMA/Ward/Bray–Curtis)
        KMEANS  - k-Means clustering

    🔧 Common Options:
        --input_directory,  -i    • Input folder, default: 'input'
        --output_directory, -d    • Output folder, default: 'output'
        --output_format,    -f    • newick | text | matrix, default: 'newick'
        --out,              -o    • Base output filename (in output folder). Extension auto-set.
        --seed                     • Random seed, default: 13
        --scale                    • Standardize features (feature-mode algorithms), default: False
        --transform                • none | clr (for abundance-like matrices), default: none
        --pca                      • PCA embedding dimensions (0 = off), default: 0
        --svd                      • TruncatedSVD dimensions (0 = off), default: 0
        --show_help                • Show this help and exit
        --remove_noise_rows [BOOL] • Remove NOISE rows from matrix outputs, default: False
        ▶ Empty-field filtering
        --empty_symbol             • Symbol/string representing empty values in feature columns ('', 'nd', '-', 0, etc), default: ''
        --remove_empty_filed       • If < 1.0, drop feature columns where empty fraction >= this threshold (range: 0.0..1.0), default: 1.0

    ✳ Algorithm-specific options:

      ▶ SA (Spectral Analysis)
        --metric                 • jaccard | euclidean (for non-square inputs; auto if not set)
        --similarity             • gaussian | epsilon, default: gaussian
        --sigma                  • Fixed sigma for gaussian kernel
        --epsilon                • ε for epsilon-neighborhood similarity (required if similarity=epsilon)
        --expected_k             • Force k clusters (optional)
        --max_cluster_number     • Max top-level clusters, default: 5
        --max_cluster_content    • Max leaves per cluster, default: 5
        --bootstrap              • Bootstrap replicates, default: 0
        --boot_mode              • replace | zero, default: replace
        --boot_frac              • Fraction of cells perturbed, default: 0.1

      ▶ DBSCAN
        --eps                    • Neighborhood radius, default: 0.5
        --min_samples            • Min samples, default: 5
        --dbscan_metric          • euclidean | jaccard | braycurtis (feature mode)

      ▶ HDBSCAN
        --min_cluster_size       • default: 5
        --min_samples            • optional
        --hdbscan_metric         • euclidean | jaccard | braycurtis (feature mode), default: euclidean
        --cluster_selection_epsilon • default: 0.0
        --allow_single_cluster   • allow 1 cluster, default: False

      ▶ LEIDEN
        --knn_k                  • k in kNN graph, default: 30
        --graph_metric           • euclidean | jaccard | hamming, default: euclidean
        --resolution             • Leiden resolution parameter, default: 1.0
        --from_distance          • Treat square input as distances (recommended), default: False

      ▶ MCL
        --knn_k                  • k in kNN graph, default: 30
        --graph_metric           • euclidean | jaccard | hamming, default: euclidean
        --inflation              • MCL inflation, default: 2.0
        --expansion              • MCL expansion, default: 2
        --prune_threshold        • default: 1e-6
        --mcl_max_iter           • default: 100

      ▶ AGGLO
        --agg_linkage            • average | ward, default: average
        --agg_metric             • euclidean | jaccard | braycurtis, default: euclidean

      ▶ KMEANS
        --kmeans_k               • number of clusters k (optional; heuristic if absent)
        --n_init                 • default: 10
        --max_iter               • default: 300

    {'='*80}
    """
    print(help_text)
    sys.exit(0)


# Ensure we can import algorithms.py in same folder
sys.path.append(os.path.dirname(__file__))

def str_to_bool(value):
    """Parse common command-line boolean values."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def remove_noise_from_outputs(outdir: Path, output_stem: str | None = None) -> tuple[int, list[Path]]:
    """Remove TXT-reported NOISE rows from matrix output files.

    Matching is deliberately tolerant: a TXT report may contain a complete
    hierarchical row label (for example category|subcategory|genome), while
    the matrix may use either that complete label or only the terminal genome
    identifier in its first column.
    """
    files = [p for p in outdir.iterdir() if p.is_file()]

    # Prefer files belonging to this run/output basename. Fall back to all files
    # in the project output directory for compatibility with custom writers.
    if output_stem:
        related = [p for p in files if p.stem.startswith(output_stem)]
        if related:
            files = related

    txt_files = sorted(
        (p for p in files if str(p).endswith("_text.txt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not txt_files:
        raise RuntimeError(
            "--remove_noise_rows requires a TXT clustering output. "
            "Use --output_format all so that both TXT and matrix outputs are created."
        )
    report = txt_files[0]
    lines = report.read_text(encoding="utf-8").splitlines()
    noise_labels: set[str] = set()
    noise_ids: set[str] = set()
    in_noise = False
    header_index = None
    noise_count = None

    for i, line in enumerate(lines):
        m = re.match(r"^NOISE\s+\(n=(\d+)(?:\s+removed)?\)\s*$", line)
        if m:
            in_noise = True
            header_index = i
            noise_count = int(m.group(1))
            continue
        if in_noise:
            if re.match(r"^Cluster\s+", line):
                break
            label = line.strip()
            if label:
                noise_labels.add(label)
                # The last hierarchy component is normally the genome/sample ID.
                noise_ids.add(label.rsplit("|", 1)[-1].strip())

    if header_index is None:
        return 0, []

    matrix_files = sorted(
        (p for p in files if p.suffix.lower() in {".csv", ".tsv"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    if not matrix_files:
        raise RuntimeError(
            "NOISE rows were found in the TXT report, but no CSV/TSV matrix "
            "output was found to filter."
        )

    revised: list[Path] = []
    total_removed = 0

    for matrix in matrix_files:
        delimiter = "\t" if matrix.suffix.lower() == ".tsv" else ","
        with matrix.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh, delimiter=delimiter))
        if len(rows) < 2:
            continue

        kept = [rows[0]]
        removed_here = 0
        for row in rows[1:]:
            if not row:
                kept.append(row)
                continue

            row_label = row[0].strip()
            row_id = row_label.rsplit("|", 1)[-1].strip()

            # Match either the complete hierarchical label or the terminal ID.
            '''
            is_noise = (
                row_label in noise_labels
                or row_label in noise_ids
                or row_id in noise_ids
            )
            '''
            is_noise = (
                row_id in noise_ids
            )
            if is_noise:
                removed_here += 1
            else:
                kept.append(row)

        if removed_here:
            with matrix.open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh, delimiter=delimiter, lineterminator="\n").writerows(kept)
            revised.append(matrix)
            total_removed += removed_here
            print(f"🧹 {matrix.name}: removed {removed_here} NOISE rows")

    if total_removed == 0 and noise_labels:
        example_noise = next(iter(noise_labels))
        example_matrix = ""
        for matrix in matrix_files:
            delimiter = "\t" if matrix.suffix.lower() == ".tsv" else ","
            with matrix.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh, delimiter=delimiter)
                next(reader, None)
                first = next(reader, None)
                if first:
                    example_matrix = first[0]
                    break
        raise RuntimeError(
            f"Found {len(noise_labels)} NOISE entries, but none matched matrix rows. "
            f"Example TXT label: {example_noise!r}; "
            f"example matrix row label: {example_matrix!r}."
        )

    # Only claim '(removed)' after successful matrix filtering.
    lines[header_index] = f"NOISE\t(n={noise_count} removed)"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return total_removed, revised


def pars_args():
    ap = argparse.ArgumentParser(description="Master runner for multiple unsupervised clustering algorithms.")

    ap.add_argument("project", help="Project foldere inside ./input")
    ap.add_argument("input_file", help="CSV/TSV filename inside ./input/project (e.g., matrix.csv or matrix.tsv)")
    ap.add_argument("algorithm", choices=["SA", "DBSCAN", "HDBSCAN", "LEIDEN", "MCL", "AGGLO", "KMEANS"],
                    help="Select algorithm class implemented in algorithms.py")

    # Common
    ap.add_argument("--input_directory", "-i", default="input", help="INPUT directory")
    ap.add_argument("--output_directory", "-o", default="output", help="OUTPUT directory")
    ap.add_argument("--output_format", "-t", choices=["all", "newick", "text", "matrix", "pathway"], default="matrix", help="Output format")
    ap.add_argument("-f", "--out", dest="out_file", default=None, help="Base output filename")
    ap.add_argument("-d", "--delimiter", type=str, default="|", help="Delemeter of hierarchical clustering levels")
    ap.add_argument("-l", "--max_cluster_levels", type=int, default=1, help="Maximum cluster levels to add to the hierarchical clustering levels")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument("--scale", action="store_true", help="Standardize features (feature-mode algorithms)")
    ap.add_argument("--transform", choices=["none", "clr"], default="none", help="Optional transform")
    ap.add_argument("--pca", type=int, default=0, help="PCA embedding dims (0=off)")
    ap.add_argument("--svd", type=int, default=0, help="TruncatedSVD dims (0=off)")
    ap.add_argument("--show_help", action="store_true", help="Show detailed help")
    ap.add_argument("--remove_noise_rows", nargs="?", const=True, default=False, type=str_to_bool,
                    help="Remove rows classified as NOISE from matrix output (default: False)")
    # Empty-field filtering
    ap.add_argument("--empty_symbol", default="", help="Symbol/string representing empty values in feature columns")
    ap.add_argument("--remove_empty_filed", type=float, default=1.0,
                    help="If < 1.0, drop feature columns where empty fraction >= this threshold (range: 0.0..1.0)")

    # SA
    ap.add_argument("--metric", choices=["jaccard", "euclidean"], default=None)
    ap.add_argument("--similarity", choices=["gaussian", "epsilon"], default="gaussian")
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--expected_k", type=int, default=None)
    ap.add_argument("--max_cluster_number", type=int, default=5)
    ap.add_argument("--max_cluster_content", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--boot_mode", choices=["replace", "zero"], default="replace")
    ap.add_argument("--boot_frac", type=float, default=0.1)

    # DBSCAN
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--min_samples", type=int, default=5)
    ap.add_argument("--dbscan_metric", choices=["euclidean", "jaccard", "braycurtis"], default=None)

    # HDBSCAN
    ap.add_argument("--min_cluster_size", type=int, default=5)
    ap.add_argument("--hdbscan_metric", choices=["euclidean", "jaccard", "braycurtis"], default="euclidean")
    ap.add_argument("--cluster_selection_epsilon", type=float, default=0.0)
    ap.add_argument("--allow_single_cluster", action="store_true")

    # LEIDEN / MCL
    ap.add_argument("--knn_k", type=int, default=30)
    ap.add_argument("--graph_metric", choices=["euclidean", "jaccard", "hamming"], default="euclidean")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--from_distance", action="store_true")

    # MCL-specific
    ap.add_argument("--inflation", type=float, default=2.0)
    ap.add_argument("--expansion", type=int, default=2)
    ap.add_argument("--prune_threshold", type=float, default=1e-6)
    ap.add_argument("--mcl_max_iter", type=int, default=100)

    # AGGLO
    ap.add_argument("--agg_linkage", choices=["average", "ward"], default="average")
    ap.add_argument("--agg_metric", choices=["euclidean", "jaccard", "braycurtis"], default="euclidean")

    # KMEANS
    ap.add_argument("--kmeans_k", type=int, default=None)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--max_iter", type=int, default=300)

    # Early help (works even if positional args missing)
    if "--show_help" in sys.argv:
        show_help()

    return ap.parse_args()
    

def main() -> int:

    args = pars_args()

    # Basic checks
    # Validate remove_empty_filed
    try:
        thr = float(args.remove_empty_filed)
    except Exception:
        raise SystemExit("[ERROR] --remove_empty_filed must be a floating number.")
    if thr < 0.0 or thr > 1.0:
        raise SystemExit("[ERROR] --remove_empty_filed must be within [0.0, 1.0].")

    input_file = Path(args.input_directory) / args.project / args.input_file
    if not input_file.exists():
        print(f"\n❌ Error: input matrix file {input_file} does not exist!", file=sys.stderr)
        return 2

    outdir = Path(args.output_directory) / args.project
    outdir.mkdir(parents=True, exist_ok=True)

    '''
    # ---- import algprithms from ./lib ----
    try:
        from lib import algorithms  # type: ignore
    except Exception:
        sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
        import algorithms  # type: ignore
    '''
    
    algo = str(args.algorithm).upper()

    cls_map = {
        "SA": algorithms.SA,
        "DBSCAN": algorithms.DBSCAN,
        "HDBSCAN": algorithms.HDBSCAN,
        "LEIDEN": algorithms.LEIDEN,
        "MCL": algorithms.MCL,
        "AGGLO": algorithms.AGGLO,
        "KMEANS": algorithms.KMEANS,
    }

    if algo not in cls_map:
        print(f"\n❌ Error: algorithm {algo} is not available!", file=sys.stderr)
        return 2

    runner = cls_map[algo](args)
    out_path = runner.execute()

    if args.remove_noise_rows:
        try:
            output_stem = Path(str(out_path)).name if out_path is not None else None
            n_removed, revised = remove_noise_from_outputs(outdir, output_stem)
        except RuntimeError as exc:
            print(f"\n❌ Error: {exc}", file=sys.stderr)
            return 2
        if n_removed:
            print(f"\n🧹 Removed {n_removed} NOISE rows from {len(revised)} matrix output file(s).")

    print(f"\n✅ Done. Output saved to: {out_path}*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
