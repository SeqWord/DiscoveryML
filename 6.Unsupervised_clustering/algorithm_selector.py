#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def show_help():
    help_text = f"""
    SELECTOR OF UNSUPERVISED CLUSTERING ALGORITHMS

    A meta-selector for unsupervised clustering in genomics/metagenomics.
    This script does NOT perform clustering itself. It recommends:
      - one "best" clustering algorithm for your situation,
      - why it was chosen,
      - and suggested parameter settings / an example command to run.

    NEW: FORCING A SPECIFIC ALGORITHM
      You can now force a specific clustering algorithm using:

        --algorithm <ABBREV>

      where <ABBREV> is one of:
        SA, DBSCAN, HDBSCAN, LEIDEN, MCL, AGGLO, KMEANS

      If --algorithm is provided, the script will NOT search for the best algorithm.
      Instead, it will report the best recommended options (metric/k/eps/min_samples/etc.)
      for THAT algorithm, given your matrix properties and/or keywords.

    ───────────────────────────────────────────────────────────────────────────────
    DETAILED HELP (Narrative)
    ───────────────────────────────────────────────────────────────────────────────

    WHAT PROBLEM DOES THIS SCRIPT SOLVE?
      In genomics and metagenomics you often face one of these issues:
        • You have a matrix, but you are not sure which clustering algorithm is most appropriate.
        • The matrix may represent features (k-mers, taxa abundance, gene presence/absence),
          or it may already be a distance matrix (square matrix with distances).
        • The data could be continuous, binary, or categorical/encoded.
        • The dataset size may be too small for graph methods, or too large for hierarchical methods.
        • You may want a tree-like interpretation (Newick), noise detection (unclustered items),
          or scalable partitioning.

      The meta-selector takes either:
        (A) an actual matrix file, and infers basic properties; and/or
        (B) user-provided keywords describing the planned task and data semantics,
      then prints:
        • a recommended algorithm (or, if --algorithm is set, the best options for that algorithm),
        • a short explanation,
        • recommended run parameters and an example command line.

    ───────────────────────────────────────────────────────────────────────────────
    END OF DETAILED HELP
    ───────────────────────────────────────────────────────────────────────────────
    """
    print(help_text)
    sys.exit()


import sys, os, argparse

# Add lib directory to path
path = os.path.abspath(
    os.path.join(os.getcwd(), "..", "lib")
)
sys.path.append(path)
import clustering_algorithm_selector as CAS

def parse_args():
    ap = argparse.ArgumentParser(
        prog="meta_selector.py",
        description="Meta-selector that recommends an unsupervised clustering algorithm and suggested parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("project",
                    help="Mandatory positional argument - project folder name within 'input' folder")

    ap.add_argument("-i", "--input_folder", default="input",
                    help="Input folder containing project folders")

    ap.add_argument("-o", "--output_folder", default="output",
                    help="Output folder containing project folders and output files")

    ap.add_argument("-m", "-f", "--input_file", default="",
                    help="Optional input CSV/TSV matrix filename inside ./input/project. "
                         "If not provided, all CSV/TSV files within the project folder will be considered as input matrices.")

    ap.add_argument(
        "--label_delimeter", "--label_delimiter",
        dest="label_delimiter",
        default="|",
        help="Delimiter separating hierarchical labels and the final row identifier "
             "inside the first matrix column (default: '|').",
    )

    ap.add_argument("-a", "--algorithm", default="",
                    choices=["", "SA", "DBSCAN", "HDBSCAN", "LEIDEN", "MCL", "AGGLO", "KMEANS"],
                    help="Force a specific algorithm (SA, DBSCAN, HDBSCAN, LEIDEN, MCL, AGGLO, KMEANS). "
                         "If set, the program will NOT select the best algorithm; it will tune parameters for the chosen one.")

    ap.add_argument("-t", "--task",
                    choices=["auto", "kmer", "metagenome_abundance", "expression", "genome_features"],
                    default="auto",
                    help="High-level task type describing the biological meaning of the matrix.")
    ap.add_argument("-u", "--units", choices=["auto", "continuous", "binary", "categorical"],
                    default="auto",
                    help="Data unit type: continuous numbers, binary presence/absence, or categorical symbols. "
                         "Use 'auto' to infer from the matrix or encoding hints.")
    ap.add_argument("-s", "--size", choices=["auto", "small", "average", "large"],
                    default="auto",
                    help="Dataset size bucket (based on number of rows if auto).")

    ap.add_argument("-l", "--kmer_length", choices=["na", "short", "average", "long"], default="na",
                    help="k-mer length bucket: short(2–4), average(5–13), long(>13), or na.")
    ap.add_argument("-c", "--kmer_encoding", choices=["na", "continuous", "zscore5", "qual5"], default="na",
                    help="k-mer frequency encoding: continuous numeric, zscore5([-2,-1,0,1,2]), "
                         "or qual5([rare,common,not-found,frequent,abundant]).")
    ap.add_argument("-b", "--abundance_encoding", choices=["na", "binary", "continuous"], default="na",
                    help="Taxon abundance encoding: binary(+/-,1/0) or continuous(normalized).")
    ap.add_argument("-e", "--expression_encoding", choices=["na", "continuous", "zscore5", "qual5"], default="na",
                    help="Expression encoding: continuous numeric, zscore5([-2,-1,0,1,2]), "
                         "or qual5(words).")
    ap.add_argument("-d", "--square_distance", action="store_true",
                    help="Force treat the matrix as a square distance matrix (overrides inference).")
    ap.add_argument("--show_help", action="store_true",
                    help="Show detailed instruction on this program use.")

    if "--show_help" in sys.argv:
        show_help()

    return ap.parse_args()

def main() -> int:
    args = parse_args()
    return CAS.process(args)


if __name__ == "__main__":
    raise SystemExit(main())
