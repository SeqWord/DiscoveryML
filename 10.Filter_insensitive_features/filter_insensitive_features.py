#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

# Add ../lib to sys.path (relative to this script, not the current working directory)
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "lib"
sys.path.append(str(LIB_DIR))

import tools


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter genomes and diagnostic SNP/probe features containing excessive missing data."
        )
    )

    parser.add_argument(
        "project_folder",
        type=str,
        help="Project subfolder inside input_folder. Mandatory positional argument."
    )

    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        default="input",
        help="Input root folder (default: input)"
    )

    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="output",
        help="Output root folder (default: output)"
    )

    parser.add_argument(
        "-m", "--matrix_file",
        type=str,
        required=True,
        help=(
            "Mandatory TSV/CSV matrix of allelic states represented as 0/1 or A,T,G,C "
            "characters (for example, '*/cluster_algorithm/_matrix_filtered_matrix.csv'; "
            "this file was created at step 6.Feature_scoring)."
        )
    )

    parser.add_argument(
        "-f", "--feature_file",
        type=str,
        required=True,
        help=(
            "Mandatory TSV/CSV file with the initial list of annotated features "
            "(for example, '*_positions_filtered.tsv'; this file was created at "
            "step 4.Filter_features)."
        )
    )

    parser.add_argument(
        "-a", "--allele_file",
        type=str,
        help=(
            "CSV/TSV file containing common allelic states at each polymorphism; "
            "required when data type is 'digit' "
            "(for example, '*_digit_alleles_filtered.csv'; this file was created "
            "at step 4.Filter_features)."
        )
    )

    parser.add_argument(
        "-t", "--data_type",
        type=str,
        choices=["digit", "character"],
        default="digit",
        help="Data type: 0/1 ('digit') or A,T,G,C ('character') (default: 'digit')"
    )

    parser.add_argument(
        "-x",
        "--no_data_per_feature_threshold",
        type=float,
        default=10.0,
        help="Maximum percentage of missing data per feature (default: 10)"
    )

    parser.add_argument(
        "-z",
        "--no_data_per_genome_threshold",
        type=float,
        default=10.0,
        help="Maximum percentage of missing data per genome (default: 10)"
    )

    parser.add_argument(
        "-y", "--no_data_symbol",
        type=str,
        default="n",
        help="Placeholder symbol for missing data (default: 'n')"
    )

    parser.add_argument(
        "-r", "--no_data_replacement",
        type=str,
        default="1",
        help="Replacement for no-data symbols (default: '0')"
    )

    parser.add_argument(
        "-n", "--feature_number_to_keep",
        type=int,
        default=0,
        help="Maximum number of features to keep after filtering (default: 0 - does not control feature number)"
    )

    return parser.parse_args()


def missing_percentage(values, no_data_symbol):
    """
    Return the percentage of cells containing the no-data placeholder.
    """
    if not values:
        return 0.0

    marker = str(no_data_symbol).lower()
    missing_count = sum(str(value).lower() == marker for value in values)

    return 100.0 * missing_count / len(values)


def get_threshold_from_user(
        current_threshold,
        items_to_delete,
        total_items,
        item_name
    ):
    """
    Ask the user to confirm filtering or enter a new threshold.

    Returns
    -------
    tuple
        (action, threshold), where action is one of:
        "accept", "keep_all", or "retry".
    """
    remaining = total_items - items_to_delete

    response = input(
        f"\n\tFiltering with threshold {current_threshold}% of missing data "
        f"selected {items_to_delete} {item_name}(s) for deletion.\n"
        f"\tThe remaining number of {item_name}s is {remaining}.\n"
        f"\tTo accept deletion, type Y+Enter,\n"
        f"\tor type 0+Enter to keep all {item_name}s,\n"
        f"\tor enter a new threshold in the range from 0 to 100% "
        f"instead of the current {current_threshold}%: "
    ).strip()

    if response.upper() == "Y":
        return "accept", current_threshold

    if response == "0":
        return "keep_all", 0.0

    try:
        new_threshold = float(response)
    except ValueError:
        print(f"\nERROR: Invalid threshold value: {response}!\n")
        return "retry", current_threshold

    if not 0 <= new_threshold <= 100:
        print(
            f"\nERROR: Threshold {new_threshold} must be in the range from 0 to 100!\n"
        )
        return "retry", current_threshold

    return "retry", new_threshold


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Validate scalar arguments
    # ------------------------------------------------------------

    if args.feature_number_to_keep < 0:
        raise ValueError("ERROR: --feature_number_to_keep must be >= 0!")

    if not 0 <= args.no_data_per_feature_threshold <= 100:
        raise ValueError(
            "ERROR: --no_data_per_feature_threshold must be in the range 0–100%!"
        )

    if not 0 <= args.no_data_per_genome_threshold <= 100:
        raise ValueError(
            "ERROR: --no_data_per_genome_threshold must be in the range 0–100%!"
        )

    if len(args.no_data_symbol) == 0:
        raise ValueError("ERROR: --no_data_symbol cannot be empty!")

    N = args.feature_number_to_keep
    feature_threshold = args.no_data_per_feature_threshold
    genome_threshold = args.no_data_per_genome_threshold

    # ------------------------------------------------------------
    # Input/output project folders
    # ------------------------------------------------------------

    in_path = os.path.join(args.input_folder, args.project_folder)

    if not os.path.isdir(in_path):
        raise FileNotFoundError(
            f"ERROR: Project folder does not exist: {in_path}!"
        )

    feature_file = os.path.join(in_path, args.feature_file)
    matrix_file = os.path.join(in_path, args.matrix_file)

    allele_file = None
    if args.data_type == "digit":
        if not args.allele_file:
            raise ValueError(
                "ERROR: --allele_file is required when --data_type is 'digit'!"
            )
        allele_file = os.path.join(in_path, args.allele_file)

    for label, path in (
        ("feature", feature_file),
        ("allele", allele_file),
        ("matrix", matrix_file),
    ):
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(
                f"ERROR: Input {label} file does not exist: {path}!"
            )

    # ------------------------------------------------------------
    # Load matrix and validate its basic structure
    # ------------------------------------------------------------

    matrix = tools.parse_matrix_file(matrix_file)

    if len(matrix) < 2:
        raise ValueError(
            "ERROR: The matrix must contain at least one header row and one genome row!"
        )

    if len(matrix[0]) < 2:
        raise ValueError(
            "ERROR: The matrix must contain at least one feature column!"
        )

    expected_columns = len(matrix[0])

    for row_number, row in enumerate(matrix, start=1):
        if len(row) != expected_columns:
            raise ValueError(
                f"ERROR: Matrix row {row_number} contains {len(row)} cells, "
                f"while {expected_columns} cells are expected!"
            )

    # List of feature IDs from the matrix header
    feature_IDs = matrix[0][1:]

    # ------------------------------------------------------------
    # Filter duplicated genomes 
    # ------------------------------------------------------------
    used_titles = []
    counter = 0
    for i in range(len(matrix) - 1, 0, -1):
        title = matrix[i][0]
        if title in used_titles:
            del matrix[i]
            counter += 1
        else:
            used_titles.append(title)
    print(f"\n\t{counter} duplicated genome records were removed!\n")
    
    # ------------------------------------------------------------
    # Filter genomes containing excessive missing data
    # ------------------------------------------------------------

    genomes_to_delete = []

    while True:
        genomes_to_delete = []

        # Rows 1..end contain genomes; column 0 contains the genome identifier.
        for i in range(len(matrix) - 1, 0, -1):
            genome_data = matrix[i][1:]
            missing = missing_percentage(genome_data, args.no_data_symbol)

            if missing > genome_threshold:
                genomes_to_delete.append(i)

        total_genomes = len(matrix) - 1

        if genome_threshold == 0:
            genomes_to_delete = []
            break

        action, new_threshold = get_threshold_from_user(
            current_threshold=genome_threshold,
            items_to_delete=len(genomes_to_delete),
            total_items=total_genomes,
            item_name="genome"
        )

        if action == "accept":
            break

        if action == "keep_all":
            genomes_to_delete = []
            break

        genome_threshold = new_threshold

    # Remove selected genomes. Indices are already collected in reverse order.
    for index in genomes_to_delete:
        del matrix[index]

    if len(matrix) < 2:
        raise ValueError(
            "ERROR: Genome filtering removed all genome rows from the matrix!"
        )

    # ------------------------------------------------------------
    # Transpose matrix and calculate missing-data percentages
    # for each feature
    # ------------------------------------------------------------

    # Remove the transposed row corresponding to the original first column,
    # which contains genome identifiers.
    transposed_matrix = tools.transpose_matrix(matrix)[1:]

    feature_rows = []

    for row in transposed_matrix:
        feature_id = row[0]
        feature_data = row[1:]
        missing = missing_percentage(feature_data, args.no_data_symbol)

        feature_rows.append({
            "id": feature_id,
            "missing_percentage": missing
        })

    # Sort features from the lowest to the highest percentage of missing data.
    feature_rows.sort(key=lambda item: item["missing_percentage"])

    # ------------------------------------------------------------
    # Filter features containing excessive missing data
    # ------------------------------------------------------------

    features_to_delete = []

    while True:
        features_to_delete = [
            item["id"]
            for item in feature_rows
            if item["missing_percentage"] > feature_threshold
        ]

        if feature_threshold == 0:
            features_to_delete = []
            break

        action, new_threshold = get_threshold_from_user(
            current_threshold=feature_threshold,
            items_to_delete=len(features_to_delete),
            total_items=len(feature_rows),
            item_name="feature"
        )

        if action == "accept":
            break

        if action == "keep_all":
            features_to_delete = []
            break

        feature_threshold = new_threshold

    # ------------------------------------------------------------
    # If necessary, reduce the remaining feature set to N features
    # ------------------------------------------------------------

    while True:
        deleted_set = set(features_to_delete)

        remaining_features = [
            item
            for item in feature_rows
            if item["id"] not in deleted_set
        ]

        remaining_number = len(remaining_features)
        additional_features_to_delete = []

        if remaining_number <= N:
            additional_features_to_delete = []
            break
        
        number_to_delete = 0
        if N > 0:
            number_to_delete = remaining_number - N

        if number_to_delete > 0:
            # remaining_features is sorted from the lowest to the highest
            # percentage of missing data, so remove features from the end.
            additional_features_to_delete = [
                item["id"]
                for item in remaining_features[-number_to_delete:]
            ]

        response = input(
            f"\n\tThe number of features after missing-data filtering is "
            f"{remaining_number}, while the requested maximum is {N}.\n"
            f"\tTo remove an additional {number_to_delete} feature(s) with the "
            f"highest percentages of missing data, type Y+Enter,\n"
            f"\tor type 0+Enter to keep all {remaining_number} filtered features,\n"
            f"\tor enter a new maximum number of features to keep instead of "
            f"the current {N}: "
        ).strip()

        if response.upper() == "Y":
            break

        if response == "0":
            additional_features_to_delete = []
            break

        try:
            new_N = int(response)
        except ValueError:
            print(
                f"\nERROR: Invalid number of features to keep: {response}!\n"
            )
            continue

        if new_N < 0:
            print(
                f"\nERROR: The number of features to keep must be >= 0!\n"
            )
            continue

        N = new_N

    # Add the additionally selected features to the deletion list.
    features_to_delete += additional_features_to_delete

    # Remove duplicate feature IDs while preserving order.
    features_to_delete = list(dict.fromkeys(features_to_delete))

    # ------------------------------------------------------------
    # Remove selected features from all corresponding files
    # ------------------------------------------------------------

    # Matrix feature columns are offset by one because column 0 contains
    # genome identifiers.
    indices = sorted(
        [feature_IDs.index(feature_id) + 1 for feature_id in features_to_delete],
        reverse=True
    )

    # Open feature annotation and allele files.
    features = tools.parse_matrix_file(feature_file)
    alleles = (
        tools.parse_matrix_file(allele_file)
        if args.data_type == "digit"
        else None
    )

    # Remove selected feature columns from the matrix and allele table,
    # and corresponding rows from the feature annotation table.
    #
    # This assumes that:
    #   1. the first row of the feature annotation table is a header;
    #   2. subsequent feature rows occur in the same order as matrix columns.
    for index in indices: 
        # ------------------------------------------------------------
        # Training matrix
        #
        # Column 0 contains genome identifiers, therefore matrix
        # feature indices are shifted by +1 relative to feature_IDs.
        # ------------------------------------------------------------
        matrix = [
            row[:index] + row[index + 1:]
            for row in matrix
        ]
    
        # ------------------------------------------------------------
        # Feature annotation file
        #
        # Row 0 = reference-genome metadata
        # Row 1 = column headers
        # Row 2+ = SNP records
        #
        # Matrix column 1 therefore corresponds to feature row 2.
        # ------------------------------------------------------------
        feature_index = index + 1
    
        if feature_index >= len(features):
            raise IndexError(
                f"ERROR: Feature index {feature_index} is outside "
                "the feature annotation table!"
            )
    
        del features[feature_index]
    
        # ------------------------------------------------------------
        # Allele matrix
        #
        # There is no leading genome/title column:
        # allele column 0 corresponds directly to feature_IDs[0].
        #
        # Matrix feature index is therefore shifted by -1.
        # ------------------------------------------------------------
        if alleles is not None:
            allele_index = index - 1
    
            alleles = [
                row[:allele_index] + row[allele_index + 1:]
                for row in alleles
            ]        
    # ------------------------------------------------------------
    # Replace no-data symbols with pseudo-values
    # ------------------------------------------------------------
    if args.no_data_replacement:
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] == args.no_data_symbol:
                    matrix[i][j] = args.no_data_replacement

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    out_path = os.path.join(args.output_folder, args.project_folder)
    os.makedirs(out_path, exist_ok=True)

    out_feature_file = os.path.join(out_path, args.feature_file)
    out_matrix_file = os.path.join(out_path, args.matrix_file)

    out_allele_file = (
        os.path.join(out_path, args.allele_file)
        if args.data_type == "digit"
        else None
    )

    for label, path, table in (
        ("feature", out_feature_file, features),
        ("allele", out_allele_file, alleles),
        ("matrix", out_matrix_file, matrix),
    ):
        if path is None or table is None:
            continue

        tools.save_matrix(
            file_path=path,
            matrix=table,
            comment=f"Output {label} file was saved to {path}."
        )


if __name__ == "__main__":
    main()
