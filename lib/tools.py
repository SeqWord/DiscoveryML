#!/usr/bin/env python3

import os
import sys
import copy
from Bio import SeqIO

fasta_extensions = (".fasta", ".fa", ".fas", ".fna", ".fst")
genbank_extensions = (".gbk", ".gb", ".gbf", ".gbff")


def get_delimiter(path) -> str:
    if path.lower().endswith((".tsv", ".txt")):
        return "\t"

    if path.lower().endswith(".csv"):
        return ","

    sys.exit(f"ERROR: Unrecognized extension in the path {path}!")


def parse_matrix_file(in_path) -> list[list]:
    """
    Read a matrix from FASTA, GenBank, TSV, TXT, or CSV file.

    FASTA/GenBank records are converted to rows:
        [record_description, nucleotide1, nucleotide2, ...]

    Text tables are returned as lists of rows.
    """

    path_lower = in_path.lower()

    if path_lower.endswith(fasta_extensions):
        records = list(SeqIO.parse(in_path, "fasta"))
        data = [
            [rec.description] + list(str(rec.seq).upper())
            for rec in records
        ]

    elif path_lower.endswith(genbank_extensions):
        records = list(SeqIO.parse(in_path, "genbank"))
        data = [
            [rec.description] + list(str(rec.seq).upper())
            for rec in records
        ]

    else:
        delimiter = get_delimiter(in_path)

        with open(in_path, "r", encoding="utf-8-sig") as f:
            data = [
                line.split(delimiter)
                for line in f.read().splitlines()
                if line.strip()
            ]

    return data

def format_output(
        file_path,
        matrix: list[list],
        title_rows: int = 1,
        title_columns: int = 1
    ) -> str:
    """
    Convert a matrix into output text according to the extension of file_path.

    FASTA:
        first title column is used as the sequence title;
        matrix body is concatenated into the sequence.

    TSV/TXT/CSV:
        the complete matrix is written using the corresponding delimiter.
    """

    if not matrix:
        sys.exit("ERROR: matrix is empty!")

    if title_rows < 0 or title_columns < 0:
        sys.exit("ERROR: title_rows and title_columns cannot be negative!")

    if title_rows > len(matrix):
        sys.exit("ERROR: title_rows exceeds matrix size!")

    if title_columns > len(matrix[0]):
        sys.exit("ERROR: title_columns exceeds matrix size!")

    path_lower = file_path.lower()

    if path_lower.endswith(fasta_extensions):

        row_titles = [
            row[:title_columns]
            for row in matrix[title_rows:]
        ]

        matrix_body = [
            row[title_columns:]
            for row in matrix[title_rows:]
        ]

        data = "\n".join(
            f">{' | '.join(str(v) for v in row_titles[i])}\n"
            f"{''.join(str(v) for v in matrix_body[i])}"
            for i in range(len(matrix_body))
        )

    else:
        delimiter = get_delimiter(file_path)

        data = "\n".join(
            delimiter.join(str(v) for v in row)
            for row in matrix
        )

    return data

def save_matrix(
        file_path: str,
        matrix: list[list],
        title_rows: int = 1,
        title_columns: int = 1,
        comment: str = ""
    ) -> str:
    """
    Format and save a matrix to a file.

    The output format is determined from the file extension.
    """

    folder_path = os.path.dirname(os.path.abspath(file_path))

    if not os.path.isdir(folder_path):
        sys.exit(
            f"ERROR: The output folder {folder_path} "
            "does not exist or is not a directory!"
        )

    # Convert the matrix to a formatted string.
    data = format_output(
        file_path=file_path,
        matrix=matrix,
        title_rows=title_rows,
        title_columns=title_columns
    )

    with open(file_path, "w", encoding="utf-8-sig") as f:
        f.write(data)

    if comment:
        print(comment)

    return file_path
    
def transpose_matrix(matrix: list[list]) -> list[list]:
    transposed_matrix = []

    for i in range(len(matrix[0])):
        transposed_matrix.append([])

        for j in range(len(matrix)):
            transposed_matrix[-1].append(
                copy_data(matrix[j][i])
            )

    return transposed_matrix


def copy_data(val):
    if isinstance(val, (str, int, float)):
        return val

    return copy.deepcopy(val)
