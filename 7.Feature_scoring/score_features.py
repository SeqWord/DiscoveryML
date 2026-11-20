#!/usr/bin/env python3
import sys
import os
import argparse

# Add ../lib directory to path
path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lib")
)
sys.path.append(path)

import top_fdr_corrected_symbols_multilevel as top_fdr_corrected 

date_of_creation = "August 14, 2026"
__version__ = "2.0"

def show_help():
    help_text = f"""
    {'='*80}
    ■ FDR Corrected Polymorphisms v{__version__} ■
    ■ Feature Selection Tool ■
    {'='*80}
    📅 Created: {date_of_creation}
    👨‍💻 Author: Oleg Reva (oleg.reva@up.ac.za)
       Centre for Bioinformatics and Computational Biology,
       BGM, University of Pretoria, South Africa

    🔬 Purpose:
    Select statistically significant polymorphic sites (features) using chi-square
    testing and FDR correction. Assess group separability using one of the following:
        • RF        – Random Forest bootstrap classification (default)
        • PERMANOVA – Permutational Multivariate Analysis of Variance
        • MANTEL    – Mantel test for distance matrix correlation

    ⚙️ Dependencies:
    ┌───────────────────┬───────────────────────┐
    │ Tool              │ Minimum Version       │
    ├───────────────────┼───────────────────────┤
    │ Python            │ 3.12.3                │
    │ sklearn           │ 1.7.0                 │
    │ statsmodels       │ 0.14.5                │
    │ matplotlib        │ 3.10.1                │
    │ scipy             │ 1.15.2                │
    │ pandas            │ 2.2.3                 │
    │ numpy             │ 2.2.3                 │
    │ scikit-bio        │ latest                │
    └───────────────────┴───────────────────────┘

    🚀 Usage:
        python3 run.py project_folder input.csv [options]

        The input file should be a CSV matrix of binary (1/0) data with column headers.
        • The first column must contain sample names.
        • The second column must contain group labels (at least two unique groups).
        • Empty rows or rows without valid group labels will be ignored.

        Example:
        Strain, Group, locus_1, locus_2, locus_3, ...
        S1,     N,     0,       0,       1,       ...
        S2,     N,     1,       0,       1,       ...
        S3,     P,     1,       1,       0,       ...
        S4,            0,       1,       0,       ...

    🔧 Options:
        POSITIONAL:
        project               • Project folder with input files: .input/project_folder
        input_file            • Input TSV/CSV matrix file: .input/project_folder/input_file
                                Use file */clusterin_algorithm/_matrix.csv  created on Step 6.
        
        OPTIONAL:
        --input_folder  -i    • Input folder, default: 'input'
        --output_folder -o    • Output folder, default: 'output'
        --delimiter     -d    • Delimiter of cluster levels in row headings, default: '|'
                                Like labels = '0|1|Environment|sponges|GCA_053410975'
]
        --p_value       -p    • Filter by maximum p-value (e.g. 0.05)
        --redundancy    -r    • Float number between (e.g. 0.05)
        --significance  -s    • Show only FDR-significant results (TRUE/FALSE)
        
        EMPTY FIELD MANAGEMENT
        --empty_symbol        • Symbol to indicate that this value is missing, default: ''
        --remove_empty_filed  • Allowed percentage of empty cells per collumn, default: 1.0 - any number of empty cells is allowed.

    🆘 Help Options:
        -h, --help            • Show this help message
        -v, --version         • Show version information
    {'='*80}
    """
    print(help_text)
    
def parse_args() -> argparse.Namespace:
    # Manual parser setup (disabling default help/version)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("project", nargs="?", default="")
    parser.add_argument("input_file", nargs="?", default="")
    parser.add_argument("--input_folder", "-i", type=str, default="input")
    parser.add_argument("--output_folder", "-o", type=str, default="output")
    parser.add_argument("--delimiter", "-d", type=str, default="|")
    parser.add_argument("--p_value", "-p", type=float, default=None)
    parser.add_argument("--significance", "-s", type=lambda x: x.lower() in ('true', 't', '1'), default=None)
    
    # Empty-field filtering
    parser.add_argument("--empty_symbol", default="", help="Symbol/string representing empty values in feature columns")
    parser.add_argument("--remove_empty_filed", type=float, default=1.0,
                    help="If < 1.0, drop feature columns where empty fraction >= this threshold (range: 0.0..1.0)")

    return parser.parse_args()

def main():
        
    args = parse_args()
    
    # Validate remove_empty_filed
    try:
        thr = float(args.remove_empty_filed)
    except Exception:
        raise SystemExit("[ERROR] --remove_empty_filed must be a floating number.")
    if thr < 0.0 or thr > 1.0:
        raise SystemExit("[ERROR] --remove_empty_filed must be within [0.0, 1.0].")

    # Locate input file
    input_path = os.path.join(args.input_folder, args.project, args.input_file)
    if not os.path.exists(input_path):
        show_help()
        print(f"\n❌ Error: Path '{input_path}' does not exist!")
        sys.exit(1)
        
    outpath = os.path.join(args.output_folder, args.project)
    os.makedirs(outpath, exist_ok=True)

    # Prepare basename
    basename = args.input_file[:args.input_file.rfind(".")] if '.' in args.input_file else args.input_file

    # Run main analysis for FDR filtering
    top_fdr_corrected.main(file_path=input_path, outpath=outpath, basename=basename, delimiter=args.delimiter,
           p_value=args.p_value, significance=args.significance, thr=thr, empty_symbol=str(args.empty_symbol))


if __name__ == "__main__":
    # Check for help/version manually before argparse
    if len(sys.argv) > 1:
        if sys.argv[1].strip() in ("-h", "--help", ""):
            show_help()
            sys.exit()
        if sys.argv[1].strip() in ("-v", "--version"):
            print(f"🌟 VARCALL v{__version__} | Created: {date_of_creation}")
            sys.exit()
    else:
        show_help()
        sys.exit()
        
    main()
    
