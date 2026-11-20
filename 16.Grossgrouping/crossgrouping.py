#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Insert a cross-identification hierarchy level into a training matrix.

The program groups genomes within each original top-level cluster according to
their complete set of successful terminal predictions from one or more
cross-prediction report files.

Input structure
---------------
./INPUT_FOLDER/PROJECT/
    MATRIX_FILE
    CROSS_PREDICTION_FILE(S)

Output structure
----------------
./OUTPUT_FOLDER/PROJECT/
    PROJECT_matrix_crossgroups.tsv
    PROJECT_crossgroup_summary.tsv

Only prediction records whose final field is ``True`` are treated as successful
terminal identifications. Records with no successful terminal prediction are
assigned to a cross-group whose signature is ``Not identified``.

The inserted hierarchy is:

    top_cluster|CrossGroup_N|label_1|label_2|...|genome_name
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Import shared tools from ../lib
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = (SCRIPT_DIR / "../lib").resolve()

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    import tools
except Exception as exc:
    raise SystemExit(
        f"ERROR: Cannot import tools.py from {LIB_DIR}: {exc}"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group cross-identified genomes and insert a new CrossGroup_N "
            "hierarchy level between the first numbered cluster and the "
            "following labeled levels."
        )
    )

    parser.add_argument(
        "-i", "--input_folder",
        default="input",
        help="Input root folder (default: input).",
    )

    parser.add_argument(
        "-o", "--output_folder",
        default="output",
        help="Output root folder (default: output).",
    )

    parser.add_argument(
        "-b", "--output_base",
        default="",
        help=(
            "Base name for output files. If set, outputs are named "
            "<output_base>_crossgroup.tsv and <output_base>_summary.tsv."
        ),
    )

    parser.add_argument(
        "project_folder",
        help="Project folder inside the input and output root folders.",
    )

    parser.add_argument(
        "-m", "--matrix_file",
        required=True,
        help="Input hierarchical matrix file inside input/PROJECT.",
    )

    parser.add_argument(
        "-c", "--cross_prediction_files",
        required=True,
        nargs="+",
        help=(
            "One or more cross-prediction report files inside input/PROJECT. "
            "Separate multiple file names by spaces."
        ),
    )

    parser.add_argument(
        "-d", "--delimiter",
        default="|",
        help="Hierarchy delimiter in matrix/report titles (default: '|').",
    )

    parser.add_argument("--mode", choices=("rearrange", "prune"),
                        default="rearrange",
                        help="Operation mode: rearrange (default) or prune.")
    parser.add_argument("-s", "--min_size", type=int, default=10,
                        help="Minimum branch size in prune mode (default: 10).")
    parser.add_argument("--remove_unidentified", action="store_true",
                        help="In prune mode, remove genomes without a successful identification.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_matrix(file_path: Path) -> List[List[str]]:
    """Read a CSV/TSV/TXT matrix using the shared tools.py delimiter logic."""
    delimiter = tools.get_delimiter(str(file_path))

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = [
            [str(cell).strip() for cell in row]
            for row in csv.reader(handle, delimiter=delimiter)
            if row and any(str(cell).strip() for cell in row)
        ]

    if len(rows) < 2:
        raise ValueError(
            f"Input matrix contains no data rows: {file_path}"
        )

    expected_columns = len(rows[0])

    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"Matrix row {row_number} has {len(row)} columns; "
                f"expected {expected_columns}."
            )

    return rows


def save_tsv(file_path: Path, rows: Sequence[Sequence[object]]) -> None:
    """Save rows as UTF-8-SIG tab-separated text."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Cross-prediction parsing
# ---------------------------------------------------------------------------

_SCORE_SUFFIX = re.compile(
    r",\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)


def prediction_label_from_field(value: str) -> str:
    """
    Remove the trailing score from a short-report prediction field.

    Example
    -------
    "Food chain|Chicken products, 0.9825"
        -> "Food chain|Chicken products"
    """
    return _SCORE_SUFFIX.sub("", str(value).strip()).strip()


def parse_short_prediction_report(
    file_path: Path,
) -> Dict[str, List[str]]:
    """
    Parse the compact report produced by tester.py/classifier.py.

    The first non-indented line is the query title. Indented lines contain
    predictions. Only lines whose final tab-separated field equals ``True``
    are retained.

    Returns
    -------
    dict
        query_title -> list of terminal prediction labels
    """
    predictions: Dict[str, List[str]] = defaultdict(list)
    current_query: str | None = None

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")

            if not line.strip():
                continue

            if not line.startswith((" ", "\t")):
                current_query = line.strip()
                predictions.setdefault(current_query, [])
                continue

            if current_query is None:
                continue

            stripped = line.strip()

            if stripped.lower() == "not identified":
                continue

            fields = stripped.split("\t")

            # Short report format:
            # prediction, score <TAB> Root... <TAB> True/False
            if len(fields) < 3:
                continue

            terminal_flag = fields[-1].strip().lower()

            if terminal_flag != "true":
                continue

            label = prediction_label_from_field(fields[0])

            if label and label not in predictions[current_query]:
                predictions[current_query].append(label)

    return dict(predictions)


def merge_prediction_reports(
    file_paths: Sequence[Path],
) -> Dict[str, List[str]]:
    """
    Merge terminal predictions from multiple report files.

    Repeated predictions for the same genome are deduplicated.
    """
    merged: Dict[str, List[str]] = defaultdict(list)

    for file_path in file_paths:
        report = parse_short_prediction_report(file_path)

        for query, labels in report.items():
            for label in labels:
                if label not in merged[query]:
                    merged[query].append(label)

            merged.setdefault(query, [])

    return dict(merged)


# ---------------------------------------------------------------------------
# Cross-group construction
# ---------------------------------------------------------------------------

def split_matrix_heading(
    heading: str,
    delimiter: str,
) -> Tuple[str, str]:
    """
    Return the top cluster and report-compatible query title.

    Matrix heading:
        2|Environment|Water and wastewater|GCA_....fna.gz

    Returns:
        ("2", "Environment|Water and wastewater|GCA_....fna.gz")
    """
    parts = [
        part.strip()
        for part in str(heading).split(delimiter)
    ]

    if len(parts) < 3:
        raise ValueError(
            f"Matrix heading has too few hierarchy components: {heading!r}"
        )

    top_cluster = parts[0]
    query_title = delimiter.join(parts[1:])

    return top_cluster, query_title


def normalize_signature(
    predictions: Iterable[str],
) -> Tuple[str, ...]:
    """Create a deterministic terminal-prediction signature."""
    return tuple(
        sorted(
            {str(value).strip() for value in predictions if str(value).strip()},
            key=str.casefold,
        )
    )


def top_cluster_sort_key(value: str) -> Tuple[int, object]:
    """Sort numeric top clusters numerically, then non-numeric labels."""
    text = str(value).strip()

    try:
        return 0, int(text)
    except ValueError:
        return 1, text.casefold()


def create_crossgroups(
    matrix: List[List[str]],
    predictions: Dict[str, List[str]],
    delimiter: str,
) -> Tuple[List[List[str]], List[List[object]], Dict[str, int]]:
    """
    Insert CrossGroup_N into every matrix row.

    CrossGroup numbering is local to each original top-level cluster. The most
    frequent cross-identification signature receives CrossGroup_1. Ties are
    resolved alphabetically by the signature text.

    Returns
    -------
    revised_matrix
    summary_rows
    statistics
    """
    row_records: List[
        Tuple[List[str], str, str, Tuple[str, ...]]
    ] = []

    signatures_by_top: Dict[
        str,
        Counter[Tuple[str, ...]],
    ] = defaultdict(Counter)

    matched_report_records = 0
    unresolved_records = 0

    for row in matrix[1:]:
        top_cluster, query_title = split_matrix_heading(
            row[0],
            delimiter,
        )

        report_predictions = predictions.get(query_title)

        if report_predictions is None:
            # The genome may be absent from the cross-prediction file.
            signature: Tuple[str, ...] = tuple()
            unresolved_records += 1
        else:
            signature = normalize_signature(report_predictions)
            matched_report_records += 1

            if not signature:
                unresolved_records += 1

        signatures_by_top[top_cluster][signature] += 1

        row_records.append(
            (
                row,
                top_cluster,
                query_title,
                signature,
            )
        )

    group_lookup: Dict[
        Tuple[str, Tuple[str, ...]],
        str,
    ] = {}

    summary_rows: List[List[object]] = []

    for top_cluster in sorted(
        signatures_by_top,
        key=top_cluster_sort_key,
    ):
        signature_counts = signatures_by_top[top_cluster]

        ordered_signatures = sorted(
            signature_counts.items(),
            key=lambda item: (
                -item[1],
                " | ".join(item[0]).casefold(),
            ),
        )

        for group_number, (
            signature,
            genome_count,
        ) in enumerate(
            ordered_signatures,
            start=1,
        ):
            group_name = f"CrossGroup_{group_number}"

            group_lookup[
                (top_cluster, signature)
            ] = group_name

            summary_rows.append([
                top_cluster,
                group_name,
                genome_count,
                len(signature),
                (
                    " ; ".join(signature)
                    if signature
                    else "Not identified"
                ),
            ])

    revised_matrix: List[List[str]] = [
        list(matrix[0])
    ]

    for (
        row,
        top_cluster,
        _query_title,
        signature,
    ) in row_records:
        parts = [
            part.strip()
            for part in row[0].split(delimiter)
        ]

        group_name = group_lookup[
            (top_cluster, signature)
        ]

        new_heading = delimiter.join(
            [parts[0], group_name] + parts[1:]
        )

        revised_matrix.append(
            [new_heading] + list(row[1:])
        )

    statistics = {
        "records": len(row_records),
        "top_clusters": len(signatures_by_top),
        "crossgroups": len(summary_rows),
        "matched_report_records": matched_report_records,
        "unresolved_records": unresolved_records,
    }

    return revised_matrix, summary_rows, statistics


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_rearranged_matrix(matrix, predictions, delimiter, min_size,
                            remove_unidentified=False):
    """Prune the already rearranged CrossGroup matrix."""
    if min_size < 1:
        raise ValueError("--min_size must be greater than 0.")

    header = list(matrix[0])
    retained = []
    removed_unidentified = 0

    for row in matrix[1:]:
        parts = [p.strip() for p in str(row[0]).split(delimiter)]
        if len(parts) < 4:
            raise ValueError(
                f"Rearranged heading has too few hierarchy components: {row[0]!r}"
            )

        # top_cluster|CrossGroup_N are absent from prediction-report titles.
        query_title = delimiter.join(parts[2:])
        signature = normalize_signature(predictions.get(query_title, []))

        if remove_unidentified and not signature:
            removed_unidentified += 1
            continue

        retained.append((list(row), parts))

    removed_small = 0

    # Recalculate branch counts until pruning no longer changes the matrix.
    while retained:
        prefix_counts = Counter()
        for _row, parts in retained:
            hierarchy = parts[:-1]  # exclude genome name
            for depth in range(1, len(hierarchy) + 1):
                prefix_counts[tuple(hierarchy[:depth])] += 1

        next_retained = []
        for row, parts in retained:
            hierarchy = parts[:-1]

            # Preserve the original first-level cluster itself. Pruning starts
            # at CrossGroup_N and applies to every downstream branch.
            too_small = any(
                prefix_counts[tuple(hierarchy[:depth])] < min_size
                for depth in range(2, len(hierarchy) + 1)
            )

            if too_small:
                removed_small += 1
            else:
                next_retained.append((row, parts))

        if len(next_retained) == len(retained):
            break
        retained = next_retained

    return [header] + [row for row, _parts in retained], {
        "removed_unidentified": removed_unidentified,
        "removed_small_branches": removed_small,
        "retained_records": len(retained),
    }


def summarize_final_crossgroups(matrix, predictions, delimiter):
    """Rebuild the summary so it describes the final pruned matrix."""
    counts = Counter()

    for row in matrix[1:]:
        parts = [p.strip() for p in str(row[0]).split(delimiter)]
        if len(parts) < 4:
            continue
        query_title = delimiter.join(parts[2:])
        signature = normalize_signature(predictions.get(query_title, []))
        counts[(parts[0], parts[1], signature)] += 1

    output = []
    for (top, group, signature), count in sorted(
        counts.items(),
        key=lambda item: (
            top_cluster_sort_key(item[0][0]),
            item[0][1].casefold(),
            " | ".join(item[0][2]).casefold(),
        ),
    ):
        output.append([
            top, group, count, len(signature),
            " ; ".join(signature) if signature else "Not identified",
        ])
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.delimiter:
        raise ValueError("--delimiter cannot be empty.")
    if args.min_size < 1:
        raise ValueError("--min_size must be greater than 0.")

    input_root = Path(args.input_folder)
    output_root = Path(args.output_folder)

    project_input = input_root / args.project_folder
    project_output = output_root / args.project_folder

    if not project_input.is_dir():
        raise FileNotFoundError(
            f"Project input folder does not exist: {project_input}"
        )

    matrix_path = project_input / args.matrix_file

    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Input matrix does not exist: {matrix_path}"
        )

    prediction_paths = [
        project_input / file_name
        for file_name in args.cross_prediction_files
    ]

    missing_files = [
        str(path)
        for path in prediction_paths
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Cross-prediction file(s) do not exist: "
            + ", ".join(missing_files)
        )

    matrix = read_matrix(matrix_path)

    predictions = merge_prediction_reports(
        prediction_paths
    )

    (
        revised_matrix,
        summary_rows,
        statistics,
    ) = create_crossgroups(
        matrix=matrix,
        predictions=predictions,
        delimiter=args.delimiter,
    )

    prune_statistics = None
    if args.mode == "prune":
        revised_matrix, prune_statistics = prune_rearranged_matrix(
            revised_matrix,
            predictions,
            args.delimiter,
            args.min_size,
            args.remove_unidentified,
        )
        summary_rows = summarize_final_crossgroups(
            revised_matrix,
            predictions,
            args.delimiter,
        )

    project_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output_base:
        matrix_output = project_output / f"{args.output_base}_crossgroup.tsv"
        summary_output = project_output / f"{args.output_base}_summary.tsv"
    else:
        matrix_output = (
            project_output
            / f"{args.project_folder}_matrix_crossgroups.tsv"
        )
        summary_output = (
            project_output
            / f"{args.project_folder}_crossgroup_summary.tsv"
        )

    save_tsv(
        matrix_output,
        revised_matrix,
    )

    save_tsv(
        summary_output,
        [
            [
                "top_cluster",
                "cross_group",
                "genome_count",
                "number_of_terminal_predictions",
                "cross_identified_categories",
            ]
        ]
        + summary_rows,
    )

    print(f"Input matrix records: {statistics['records']}")
    print(f"Original top-level clusters: {statistics['top_clusters']}")
    print(f"Generated cross-groups: {statistics['crossgroups']}")
    print(
        "Matrix records found in prediction report(s): "
        f"{statistics['matched_report_records']}"
    )
    print(
        "Records with no terminal cross-identification: "
        f"{statistics['unresolved_records']}"
    )
    print(f"Mode: {args.mode}")
    if prune_statistics is not None:
        print(f"Minimum retained branch size: {args.min_size}")
        print(f"Remove unidentified: {args.remove_unidentified}")
        print(f"Unidentified records removed: {prune_statistics['removed_unidentified']}")
        print(
            "Records removed with branches smaller than min_size: "
            f"{prune_statistics['removed_small_branches']}"
        )
        total_removed = (
            prune_statistics["removed_unidentified"]
            + prune_statistics["removed_small_branches"]
        )
        print(f"Total genomes removed: {total_removed}")
        print(f"Records retained after pruning: {prune_statistics['retained_records']}")
    print(f"Revised matrix saved to: {matrix_output}")
    print(f"Cross-group summary saved to: {summary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
