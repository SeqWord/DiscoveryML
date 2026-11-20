import sys
import os
import csv
import glob
import math
import random
import argparse
from collections import defaultdict

# Add the ../lib directory relative to this script to the module search path.
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib"))
if lib_path not in sys.path:
    sys.path.append(lib_path)

import tools


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a dataset matrix into training and tester datasets."
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
            "Mandatory TSV/CSV matrix containing allelic states represented as "
            "0/1 or A,T,G,C characters (for example, "
            "'*/cluster_algorithm/_matrix_filtered_matrix.csv'). The matrix is "
            "expected inside input_folder/project_folder."
        )
    )

    parser.add_argument(
        "-t", "--tester_dataset_proportion",
        type=float,
        default=30.0,
        help="Percentage of records assigned to the tester dataset (default: 30)"
    )

    parser.add_argument(
        "-b", "--balanced_training_dataset",
        action="store_true",
        help="Create a balanced training dataset (default: False)"
    )

    parser.add_argument(
        "-u", "--unbalancing_order",
        type=float,
        default=1.5,
        help=(
            "Maximum permitted ratio between the largest and smallest clusters "
            "in a balanced training dataset (default: 1.5)"
        )
    )

    parser.add_argument(
        "-f", "--filter_noise",
        type=str,
        default=None,
        help=(
            "Optional clustering-report filename inside input_folder/project_folder. "
            "Records listed in its NOISE section are removed from the matrix before "
            "the training/tester split."
        )
    )

    parser.add_argument(
        "-d", "--delimiter",
        type=str,
        default="|",
        help=(
            "Delimiter used in row titles. All title fields except the last one "
            "define the cluster name (default: '|')"
        )
    )

    return parser.parse_args()


def get_matrix_delimiter(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".tsv":
        return "\t"
    if extension == ".csv":
        return ","
    sys.exit("ERROR: The input matrix must have a .csv or .tsv extension.")


def load_matrix(file_path: str) -> list[list[str]]:
    """Read a CSV/TSV matrix and return it as a list of rows."""
    matrix_delimiter = get_matrix_delimiter(file_path)
    with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle, delimiter=matrix_delimiter) if row]


def get_cluster_name(row_title: str, delimiter: str) -> str:
    """Return the cluster name encoded in a matrix row title."""
    title_parts = row_title.split(delimiter)
    if len(title_parts) < 2:
        sys.exit(
            f"ERROR: Row title '{row_title}' does not contain delimiter "
            f"'{delimiter}'."
        )
    return delimiter.join(title_parts[:-1])


def resolve_input_file(project_input_path: str, matrix_file: str) -> str:
    """Resolve a matrix filename or a relative glob pattern inside the project folder."""
    pattern = os.path.join(project_input_path, matrix_file)
    matches = [path for path in glob.glob(pattern) if os.path.isfile(path)]

    if not matches:
        sys.exit(f"ERROR: Input matrix was not found: {pattern}")
    if len(matches) > 1:
        sys.exit(
            "ERROR: The matrix_file pattern matches more than one file:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def resolve_project_file(project_input_path: str, file_name: str, description: str) -> str:
    """Resolve an ordinary filename relative to the project input folder."""
    file_path = os.path.join(project_input_path, file_name)
    if not os.path.isfile(file_path):
        sys.exit(f"ERROR: {description} was not found: {file_path}")
    return file_path


def load_noise_genome_names(report_file: str, delimiter: str) -> set[str]:
    """
    Read genome names from the NOISE section of a clustering report.

    Each indented NOISE record may contain metadata fields separated by the same
    delimiter used in matrix row titles. Only the final field (the genome name)
    is used for matching, so optional cluster labels at the beginning of matrix
    row titles do not affect filtering.
    """
    noise_genomes = set()
    in_noise_section = False

    with open(report_file, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if not in_noise_section:
                if line == "NOISE" or line.startswith("NOISE\t") or line.startswith("NOISE "):
                    in_noise_section = True
                continue

            # The NOISE block ends when the first named cluster begins.
            if line.startswith("Cluster "):
                break

            # Ignore comments or parameter lines if they occur unexpectedly.
            if line.startswith("#") or "=" in line and delimiter not in line:
                continue

            genome_name = line.split(delimiter)[-1].strip()
            if genome_name:
                noise_genomes.add(genome_name)

    if not in_noise_section:
        sys.exit(f"ERROR: No NOISE section was found in: {report_file}")

    if not noise_genomes:
        sys.exit(f"ERROR: The NOISE section contains no genome records: {report_file}")

    return noise_genomes


def filter_noise_rows(
        rows: list[list[str]],
        noise_genomes: set[str],
        delimiter: str
    ) -> tuple[list[list[str]], int, set[str]]:
    """Remove matrix rows whose final title field matches a NOISE genome name."""
    retained_rows = []
    removed_count = 0
    matched_genomes = set()

    for row in rows:
        genome_name = row[0].split(delimiter)[-1].strip()
        if genome_name in noise_genomes:
            removed_count += 1
            matched_genomes.add(genome_name)
        else:
            retained_rows.append(row)

    return retained_rows, removed_count, matched_genomes


def select_initial_tester_indices(
        cluster_indices: dict[str, list[int]],
        record_count: int,
        tester_proportion: float
    ) -> set[int]:
    """
    Select tester records randomly while ensuring that every cluster is represented.
    """
    target_tester_count = round(record_count * tester_proportion / 100.0)
    target_tester_count = max(target_tester_count, len(cluster_indices))
    target_tester_count = min(target_tester_count, record_count)

    tester_indices = set()

    # Reserve one random record from every cluster.
    for indices in cluster_indices.values():
        tester_indices.add(random.choice(indices))

    # Add random records until the requested tester size is reached.
    remaining_indices = [
        index for index in range(record_count)
        if index not in tester_indices
    ]
    random.shuffle(remaining_indices)

    additional_count = target_tester_count - len(tester_indices)
    tester_indices.update(remaining_indices[:additional_count])

    return tester_indices


def balance_training_indices(
        training_indices: set[int],
        rows: list[list[str]],
        title_delimiter: str,
        unbalancing_order: float
    ) -> tuple[set[int], set[int]]:
    """
    Trim oversized training clusters.

    Returns the retained training indices and indices removed for balancing.
    """
    training_by_cluster = defaultdict(list)
    for index in training_indices:
        cluster_name = get_cluster_name(rows[index][0], title_delimiter)
        training_by_cluster[cluster_name].append(index)

    nonempty_cluster_sizes = [
        len(indices) for indices in training_by_cluster.values() if indices
    ]
    if not nonempty_cluster_sizes:
        return training_indices, set()

    smallest_cluster_size = min(nonempty_cluster_sizes)
    maximum_cluster_size = max(
        1,
        math.floor(smallest_cluster_size * unbalancing_order)
    )

    removed_indices = set()

    for indices in training_by_cluster.values():
        excess_count = len(indices) - maximum_cluster_size
        if excess_count > 0:
            removed_indices.update(random.sample(indices, excess_count))

    return training_indices - removed_indices, removed_indices


def main():
    args = parse_args()

    if not 0.0 <= args.tester_dataset_proportion <= 100.0:
        sys.exit("ERROR: --tester_dataset_proportion must be between 0 and 100.")

    if args.unbalancing_order < 1.0:
        sys.exit("ERROR: --unbalancing_order must be greater than or equal to 1.0.")

    if not args.delimiter:
        sys.exit("ERROR: --delimiter cannot be an empty string.")

    project_input_path = os.path.join(args.input_folder, args.project_folder)
    if not os.path.isdir(project_input_path):
        sys.exit(
            f"ERROR: The project input folder does not exist: "
            f"{project_input_path}"
        )

    input_file = resolve_input_file(project_input_path, args.matrix_file)
    matrix = load_matrix(input_file)

    if len(matrix) < 2:
        sys.exit("ERROR: The input matrix contains no data records.")

    header = matrix[0]
    rows = matrix[1:]
    expected_column_count = len(header)

    for row_number, row in enumerate(rows, start=2):
        if len(row) != expected_column_count:
            sys.exit(
                f"ERROR: Row {row_number} contains {len(row)} columns; "
                f"expected {expected_column_count}."
            )

    original_record_count = len(rows)
    noise_record_count = 0
    noise_report_count = 0

    if args.filter_noise:
        noise_report_file = resolve_project_file(
            project_input_path,
            args.filter_noise,
            "Noise-filter report"
        )
        noise_genomes = load_noise_genome_names(
            report_file=noise_report_file,
            delimiter=args.delimiter
        )
        noise_report_count = len(noise_genomes)
        rows, noise_record_count, matched_noise_genomes = filter_noise_rows(
            rows=rows,
            noise_genomes=noise_genomes,
            delimiter=args.delimiter
        )

        if not rows:
            sys.exit("ERROR: Noise filtering removed all matrix records.")

        unmatched_noise_count = noise_report_count - len(matched_noise_genomes)
        if unmatched_noise_count:
            print(
                f"WARNING: {unmatched_noise_count} NOISE genome(s) from the report "
                "were not present in the input matrix."
            )

    cluster_indices = defaultdict(list)
    for index, row in enumerate(rows):
        cluster_name = get_cluster_name(row[0], args.delimiter)
        cluster_indices[cluster_name].append(index)
        
    # Check if th clustering results are supportive for model training
    requested_tester_count = round(
        len(rows) * args.tester_dataset_proportion / 100.0
    )
    
    cluster_count = len(cluster_indices)
    
    singleton_clusters = sum(
        1
        for indices in cluster_indices.values()
        if len(indices) == 1
    )
    
    if cluster_count > requested_tester_count:
        print(
            "\nERROR: The selected clustering level is too detailed for the "
            "requested training/tester split.\n"
            f"  Records after filtering: {len(rows)}\n"
            f"  Clusters at the selected level: {cluster_count}\n"
            f"  Singleton clusters: {singleton_clusters}\n"
            f"  Requested tester proportion: "
            f"{args.tester_dataset_proportion:.1f}%\n"
            f"  Requested tester records: {requested_tester_count}\n"
            "\n"
            "The program requires at least one tester record per cluster, "
            "which would exceed the requested tester dataset size.\n"
            "\n"
            "Recommendation: reduce the number of clustering levels used "
            "for stratification, for example with --cluster_level 1."
        )
        sys.exit(1)

    tester_indices = select_initial_tester_indices(
        cluster_indices=cluster_indices,
        record_count=len(rows),
        tester_proportion=args.tester_dataset_proportion
    )

    training_indices = set(range(len(rows))) - tester_indices
    balancing_removed_indices = set()

    if args.balanced_training_dataset:
        training_indices, balancing_removed_indices = balance_training_indices(
            training_indices=training_indices,
            rows=rows,
            title_delimiter=args.delimiter,
            unbalancing_order=args.unbalancing_order
        )

        # Records removed during balancing are retained as tester records.
        tester_indices.update(balancing_removed_indices)

    training_matrix = [header] + [
        rows[index] for index in sorted(training_indices)
    ]
    tester_matrix = [header] + [
        rows[index] for index in sorted(tester_indices)
    ]
    removed_matrix = [header] + [
        rows[index] for index in sorted(balancing_removed_indices)
    ]

    project_output_path = os.path.join(args.output_folder, args.project_folder)
    os.makedirs(project_output_path, exist_ok=True)

    matrix_filename = os.path.basename(input_file)
    matrix_name, matrix_extension = os.path.splitext(matrix_filename)

    training_file = os.path.join(
        project_output_path,
        f"{matrix_name}_training{matrix_extension}"
    )
    tester_file = os.path.join(
        project_output_path,
        f"{matrix_name}_tester{matrix_extension}"
    )
    removed_file = os.path.join(
        project_output_path,
        f"{matrix_name}_removed_for_balancing{matrix_extension}"
    )

    tools.save_matrix(
        training_file, training_matrix, title_rows=1, title_columns=1
    )
    tools.save_matrix(
        tester_file, tester_matrix, title_rows=1, title_columns=1
    )
    if args.balanced_training_dataset:
        tools.save_matrix(
            removed_file, removed_matrix, title_rows=1, title_columns=1
        )

    print(f"Input records: {original_record_count}")
    if args.filter_noise:
        print(f"NOISE genomes listed in filter report: {noise_report_count}")
        print(f"NOISE records removed from matrix: {noise_record_count}")
        print(f"Records after noise filtering: {len(rows)}")
    print(f"Clusters: {len(cluster_indices)}")
    print(f"Training records: {len(training_indices)}")
    print(f"Tester records: {len(tester_indices)}")
    if args.balanced_training_dataset:
        print(f"Records removed for balancing: {len(balancing_removed_indices)}")


if __name__ == "__main__":
    main()
