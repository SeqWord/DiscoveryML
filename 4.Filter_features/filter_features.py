
#!/usr/bin/env python3

import os
import sys
import argparse
from Bio import SeqIO

# Add ../lib directory to path
path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lib")
)
sys.path.append(path)
import progressbar

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter unidentified and non-sense SNPs."
    )

    # Positional argument
    parser.add_argument(
        "project_folder",
        help="Project subfolder inside input_folder"
    )

    # Input matrix file
    '''
    gbff_file	                    location	                                SNP_000001	SNP_000002	SNP_000003
    GCA_053410235.1_genomic.gbff.gz	Environment>Environmental swabs and sponges	C	        A	        C
    GCA_053410975.1_genomic.gbff.gz	Environment>Environmental swabs and sponges	C	        C	        T
    '''
    parser.add_argument(
        "-f", "--input_matrix_file",
        default="core_snps.tsv",
        help="Input SNP matrix TSV/CSV file (default: core_snps.tsv)"
    )

    # SNP Annotation file
    # SNP	alignment_position,	Location in reference sequence,	Strand in reference,	alleles,	CDS	annotation,	codon,	context
    parser.add_argument(
        "-a", "--input_annotation_file",
        default="core_snp_positions.tsv",
        help="Input SNP annotation TSV/CSV file (default: core_snp_positions.tsv)"
    )

    parser.add_argument(
        "-i", "--input_folder",
        default="input",
        help="Input root folder (default: input)"
    )

    parser.add_argument(
        "-o", "--output_folder",
        default="output",
        help="Output root folder (default: output)"
    )

    parser.add_argument(
        "-t", "--output_format",
        choices=("characters", "digits"),
        default="digits",
        help=("Output format can be either a comma-separated (CSV) character or 0/1 digit matrix (default: 'digits')")
    )

    parser.add_argument(
        "-x", "--no_fasta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "By default the program creates additional FASTA representations of the matrices. "
            "Use --no-fasta to disable creating additional fasta files (default: False)."
        )
    )

    return parser.parse_args()

def parse_matrix_file(in_file):
    if in_file.lower().endswith(".tsv"):
        with open(in_file) as f:
            data = [line.split("\t") for line in f.read().split("\n") if line.strip()]
            
    elif in_file.lower().endswith(".csv"):
        with open(in_file) as f:
            data = [line.split(",") for line in f.read().split("\n") if line.strip()]
            
    elif in_file.lower().endswith(".fasta"):
        fasta = list(SeqIO.parse(in_file, "fasta"))
        data = [[rec.description] + list(str(rec.seq).upper()) for rec in fasta]
        
    return data

def format_output(fname, matrix):
    if fname.endswith(".tsv"):
        return "\n".join(["\t".join(ls) for ls in matrix])
    elif fname.endswith(".csv"):
        return "\n".join([",".join(ls) for ls in matrix])
    elif fname.endswith(".fasta"):
        return "\n".join([f">{ls[0]}\n{''.join(ls[1:])}" for ls in matrix])
    return ""

# Convert the initial SNP matrix to FASTA    
def snp2fasta(in_path: str, in_file: str, out_path: str):

    # Replace extension safely
    base, ext = os.path.splitext(in_file)
    out_fasta_file = os.path.join(
        out_path,
        base + ".fasta"
    )
    in_file = os.path.join(in_path, in_file)
    
    if ext.lower() in (".tsv", ".txt"):
        delimiter = "\t"
    elif ext.lower() == ".csv":
        delimiter = ","
    else:
        sys.exit(
            f"ERROR: input file must have extensions tsv, csv or txt: {in_file}"
        )

    fasta = []

    with open(in_file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 0):
            line = line.rstrip("\n\r")

            if not line:
                continue
            
            if line_number == 0:
                titles = line.split(delimiter)
                continue
            else:
                fields = line.split(delimiter)

            if len(fields) < 3:
                continue

            # First two columns:
            #   0 = GBFF file
            #   1 = location
            #
            # Remaining columns are SNP nucleotides.
            gbff_file = fields[0].replace(">", "|")
            location = fields[1].replace(">", "|")

            sequence = "".join(fields[2:]).upper()

            fasta.append(
                f">{location}|{gbff_file}\n{sequence}"
            )
            

    if not fasta:
        sys.exit(
            f"ERROR: no sequence records were found in {in_file}"
        )

    with open(out_fasta_file, "w", encoding="utf-8") as f:
        f.write("\n".join(fasta))

    print(
        f"Created {len(fasta)} FASTA sequences and one-title-column matrix:\n{out_fasta_file}"
    )
    return out_fasta_file

# Conver initial SNP matrix to labeled CSV matrix 
def snp2matrix(in_path: str, in_file: str, out_path: str):

    # Replace extension safely
    base, ext = os.path.splitext(in_file)
    out_matrix_file = os.path.join(
        out_path,
        base + "_matrix.csv"
    )
    in_file = os.path.join(in_path, in_file)

    if ext.lower() in (".tsv", ".txt"):
        delimiter = "\t"
    elif ext.lower() == ".csv":
        delimiter = ","
    else:
        sys.exit(
            f"ERROR: input file must have extensions tsv, csv or txt: {in_file}"
        )

    data = []

    with open(in_file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 0):
            line = line.rstrip("\n\r")

            if not line:
                continue
            
            if line_number == 0:
                titles = line.split(delimiter)
                continue
            else:
                fields = line.split(delimiter)

            if len(fields) < 3:
                continue

            # First two columns:
            #   0 = GBFF file
            #   1 = location
            #
            # Remaining columns are SNP nucleotides.
            gbff_file = fields[0].replace(">", "|")
            location = fields[1].replace(">", "|")

            sequence = "".join(fields[2:]).upper()

            data.append(
                f"{location.replace(',',' ').replace('(','').replace(')','')}|"
                f"{gbff_file.replace(',','.').replace('(','').replace(')','').replace('|',' ')},"
                f"{','.join(fields[2:]).upper()}"
            )
            

    if not data:
        sys.exit(
            f"ERROR: no sequence records were found in {in_file}"
        )

    with open(out_matrix_file, "w", encoding="utf-8") as f:
        f.write("\n".join([",".join(["Genome"] + titles[2:])] + data))

    print(
        f"Created {len(data)} FASTA sequences and one-title-column matrix:\n{out_matrix_file}"
    )
    return out_matrix_file

# Convert initial SNP matrix to digital labeled matrix
def snp2digital_matrix(in_path: str, in_file: str, out_path: str):

    # Replace extension safely
    base, ext = os.path.splitext(in_file)
    out_matrix_file = os.path.join(
        out_path,
        base + "_digit_matrix.csv"
    )
    in_file = os.path.join(in_path, in_file)

    out_ref_allele_file = os.path.join(
        out_path,
        base + "_digit_alleles.csv"
    )

    if ext.lower() in (".tsv", ".txt"):
        delimiter = "\t"
    elif ext.lower() == ".csv":
        delimiter = ","
    else:
        sys.exit(
            f"ERROR: input file must have extensions tsv, csv or txt: {in_file}"
        )

    data = []

    with open(in_file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 0):
            line = line.rstrip("\n\r")

            if not line:
                continue
            
            if line_number == 0:
                titles = line.split(delimiter)
                continue
            else:
                fields = line.split(delimiter)

            if len(fields) < 3:
                continue

            # First two columns:
            #   0 = GBFF file
            #   1 = location
            #
            # Remaining columns are SNP nucleotides.
            gbff_file = fields[0].replace(">", "|")
            location = fields[1].replace(">", "|")

            sequence = "".join(fields[2:]).upper()

            data.append([
                f"{location.replace(',',' ').replace('(','').replace(')','')}|"
                f"{gbff_file.replace(',','.').replace('(','').replace(')','').replace('|',' ')}"
                ] + fields[2:])
    
    if not data:
        sys.exit(
            f"ERROR: no sequence records were found in {in_file}"
        )

    ref_alleles = ["Major alleles"]
    for j in range(1, len(data[0])):
        column = "".join([ls[j] for ls in data]).upper()

        alleles = [
            [column.count(nuc), nuc]
            for nuc in ["A", "T", "G", "C"]
        ]

        alleles.sort(reverse=True)
        mj = alleles[0][1].strip()
        ref_alleles.append(mj)
        states = ["0" if s==mj else "1" for s in list(column)]
        for i in range(len(data)):
            data[i][j] = states[i]
            

    with open(out_matrix_file, "w", encoding="utf-8") as f:
        f.write("\n".join([",".join(["Genome"] + titles[2:])] + [",".join(ls) for ls in data]))

    with open(out_ref_allele_file, "w", encoding="utf-8") as f:
        f.write(",".join(["SNP_IDs"] + titles[2:]) + "\n" + ",".join(ref_alleles))

    print(
        f"{len(data)} records are saved to the matrix:\n{out_matrix_file}"
    )
    return out_matrix_file, out_ref_allele_file
    

def main():
    args = parse_args()

    out_path = os.path.join(
        args.output_folder,
        args.project_folder
    )
    os.makedirs(out_path, exist_ok=True)

    in_path = os.path.join(
        args.input_folder,
        args.project_folder
    )

    # Core SNP positions
    in_core_snp_positions = os.path.join(
        in_path,
        args.input_annotation_file
    )
    if not os.path.exists(in_core_snp_positions):
        sys.exit(f"ERROR: input file {in_core_snp_positions} does not exists!")
        
    # Parse initial SNP annotation matrix
    core_snp_positions = parse_matrix_file(in_core_snp_positions)
       
    # Matrix format transformation
    in_fasta = None
    in_matrix = None
    in_digit_matrix = None
    in_digit_alleles = None
    
    # Convert matrix to fasta
    if not args.no_fasta:
        in_fasta = snp2fasta(in_path=in_path, in_file=args.input_matrix_file, out_path=out_path)
    
    # Convert initial SNP matrix into CSV labeled digital or character matrix    
    if args.output_format.lower() == 'characters':
        in_matrix = snp2matrix(in_path=in_path, in_file=args.input_matrix_file, out_path=out_path)
    elif args.output_format.lower() == 'digits':
        in_digit_matrix, in_digit_alleles = snp2digital_matrix(in_path=in_path, in_file=args.input_matrix_file, out_path=out_path)
    
    out_files = {
        "core_snps_filetered.fasta":None if in_fasta is None else parse_matrix_file(in_fasta),
        "core_snps_matrix_filtered.csv":None if in_matrix is None else parse_matrix_file(in_matrix),
        "core_snps_digit_matrix_filtered.csv":None if in_digit_matrix is None else parse_matrix_file(in_digit_matrix),
        "core_snps_digit_alleles_filtered.csv":None if in_digit_alleles is None else parse_matrix_file(in_digit_alleles),
    }
    
    bar = progressbar.indicator(len(core_snp_positions), "Process SNPs...")
    for i in range(len(core_snp_positions) - 1, 1, -1):
        codon = core_snp_positions[i][7]
        if codon and (codon.find(",") > -1 or codon[0] != codon[-1]):
            continue
        else:
            j = i - 1
            del core_snp_positions[i]
            for key in out_files:
                if out_files[key] is not None:
                    for k in range(len(out_files[key])):
                        del out_files[key][k][j]
        bar(len(core_snp_positions) - i)
    bar.stop()
                    
    with open(os.path.join(out_path, "core_snps_positions_filtered.tsv"), "w") as f:
        f.write(format_output("core_snps_positions_filtered.tsv", core_snp_positions))
    
    # Save initial matrices 
    for out_file in out_files:
        if out_files[out_file] is None:
            continue
        with open(os.path.join(out_path, out_file), "w") as f:
            f.write(format_output(out_file, out_files[out_file]))
    

if __name__ == "__main__":
    main()

