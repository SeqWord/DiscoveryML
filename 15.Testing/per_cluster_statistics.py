#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate per-cluster performance of a hierarchical classifier.

The input matrix is read from:
    ./input/PROJECT/MATRIX_FILE

The model bundle is loaded from:
    ./MODEL_FILE

The report is written to:
    ./output/PROJECT/PROJECT.txt

No sequence parsing or BLAST allele calling is performed because the tester
matrix already contains the allelic states used by the classifier.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = (SCRIPT_DIR / "../lib").resolve()

if not LIB_DIR.is_dir():
    raise SystemExit(
        f"ERROR: Required library directory does not exist: {LIB_DIR}"
    )

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    from model_bundle import ModelBundle
except Exception as exc:
    raise SystemExit(
        f"ERROR: Cannot import model_bundle.py from {LIB_DIR}: {exc}"
    )


TERMINAL_LABEL = "__TERMINAL__"
MISSING_STATE = "n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test a hierarchical classifier using a matrix that already contains "
            "the true hierarchical labels and allelic states of tester genomes."
        )
    )

    parser.add_argument(
        "project_folder",
        help="Project subfolder inside ./input and ./output."
    )

    parser.add_argument(
        "-m", "--matrix_file",
        required=True,
        help=(
            "Tester CSV/TSV matrix inside ./input/PROJECT. "
            "The first column must contain hierarchical cluster labels followed "
            "by the genome name; remaining columns contain allelic states."
        ),
    )

    parser.add_argument(
        "-d", "--delimiter",
        default="|",
        help=(
            "Delimiter separating hierarchical labels and the genome name "
            "in the first matrix column (default: '|')."
        ),
    )

    parser.add_argument(
        "-b", "--model_file",
        default="model.pkl",
        help=(
            "PKL model-bundle file relative to tester.py, unless an absolute "
            "path is supplied (default: model.pkl)."
        ),
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
        "--max_no_data_for_NN",
        type=float,
        default=20.0,
        help=(
            "Maximum percentage of missing markers at which an NN/ML model "
            "may still be used. Above this value allele-profile similarity "
            "is used (default: 20)."
        ),
    )

    parser.add_argument(
        "--max_no_data_allowed",
        type=float,
        default=50.0,
        help=(
            "Maximum percentage of missing markers allowed for classification. "
            "Above this value the genome is reported as not identified "
            "(default: 50)."
        ),
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.85,
        help=(
            "Minimum score required for a terminal-cluster identification "
            "(0..1; default: 0.85)."
        ),
    )

    parser.add_argument(
        "--specificity",
        type=float,
        default=0.70,
        help=(
            "Minimum score required to continue through an intermediate "
            "hierarchical branch (0..1; default: 0.70)."
        ),
    )

    parser.add_argument(
        "-l", "-L", "--level", "--cluster_level",
        dest="level", type=int, default=2,
        help=("Hierarchy level counted from the bottom: 1=terminal, "
              "2=parent of terminal; values at or above root select "
              "the first genomic-cluster level (default: 2)."),
    )

    parser.add_argument(
        "--show_graph",
        choices=("no", "count", "percent"),
        default="no",
        help="Heatmap to display: no (default), count, or percent. Both PNGs are always saved.",
    )    

    return parser.parse_args()


def get_matrix_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".tsv":
        return "\t"

    if suffix == ".csv":
        return ","

    if suffix == ".txt":
        # The project convention is tab-delimited TXT.
        return "\t"

    raise ValueError(
        f"Unsupported matrix extension {suffix!r}; expected .tsv, .csv, or .txt."
    )


def load_matrix(path: Path) -> List[List[str]]:
    """Read the complete tester matrix as stripped strings."""
    delimiter = get_matrix_delimiter(path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            [str(cell).strip() for cell in row]
            for row in csv.reader(handle, delimiter=delimiter)
            if row and any(str(cell).strip() for cell in row)
        ]

    if len(rows) < 2:
        raise ValueError(
            f"Tester matrix contains no data records: {path}"
        )

    expected_columns = len(rows[0])

    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"Matrix row {row_number} contains {len(row)} columns; "
                f"expected {expected_columns}."
            )

    return rows


def load_bundle(path: Path) -> ModelBundle:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle does not exist: {path}"
        )

    bundle = ModelBundle.load(path)

    if "tree" not in bundle:
        raise ValueError(
            "The PKL file is not a compatible hierarchical ModelBundle."
        )

    return bundle


def parse_heading(
    heading: str,
    delimiter: str,
) -> Tuple[str, str]:
    """Return expected hierarchical path and genome name."""
    parts = [part.strip() for part in str(heading).split(delimiter)]

    if len(parts) < 2:
        raise ValueError(
            f"Matrix row heading {heading!r} does not contain delimiter "
            f"{delimiter!r}."
        )

    genome_name = parts[-1]
    expected_path = delimiter.join(parts[:-1])

    if not genome_name:
        raise ValueError(
            f"Matrix row heading {heading!r} has an empty genome name."
        )

    if not expected_path:
        raise ValueError(
            f"Matrix row heading {heading!r} has no cluster labels."
        )

    return expected_path, genome_name


def is_missing(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"", "n", "nan", "nd", "none", "na", "?"}


def normalize_marker_value(
    value: Any,
    data_type: str,
) -> str:
    """Normalize one tester-matrix marker state."""
    text = str(value).strip()

    if is_missing(text):
        return MISSING_STATE

    if data_type in {"binary", "digit"}:
        try:
            numeric = float(text)
        except ValueError:
            return text

        if numeric == 0:
            return "0"

        if numeric == 1:
            return "1"

        return text

    return text.upper()


def build_markers(
    header_features: Sequence[str],
    row_values: Sequence[str],
    bundle: ModelBundle,
) -> Dict[str, str]:
    """Build marker dictionary in the feature order required by the model."""
    input_map = {
        str(title).strip(): value
        for title, value in zip(header_features, row_values)
    }

    data_type = str(bundle.get("data_type", "character")).lower()

    return {
        str(title): normalize_marker_value(
            input_map.get(str(title), MISSING_STATE),
            data_type,
        )
        for title in bundle["feature_titles"]
    }


def validate_feature_titles(
    matrix_features: Sequence[str],
    bundle: ModelBundle,
) -> None:
    """Ensure that model features are represented in the tester matrix."""
    matrix_set = set(str(v).strip() for v in matrix_features)
    model_features = [str(v) for v in bundle["feature_titles"]]

    missing = [
        feature
        for feature in model_features
        if feature not in matrix_set
    ]

    if missing:
        raise ValueError(
            "Tester matrix is missing feature(s) required by the model: "
            + ", ".join(missing[:20])
        )


def missing_fraction(markers: Dict[str, str]) -> float:
    if not markers:
        return 1.0

    missing = sum(
        str(value).strip().lower() == MISSING_STATE
        for value in markers.values()
    )

    return missing / len(markers)


def _impute_for_nn(
    markers: Dict[str, str],
    bundle: ModelBundle,
) -> Dict[str, str]:
    """Impute missing states only for NN/ML prediction."""
    data_type = str(bundle.get("data_type", "character")).lower()
    common = bundle.get("common_alleles") or {}

    result: Dict[str, str] = {}

    for title in bundle["feature_titles"]:
        title = str(title)
        value = str(markers.get(title, MISSING_STATE)).strip()

        if value.lower() != MISSING_STATE:
            result[title] = value
            continue

        if data_type in {"binary", "digit"}:
            # Training convention for binary matrices.
            result[title] = "1"
        else:
            common_state = str(common.get(title, "")).strip().upper()
            result[title] = common_state if common_state else "N"

    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)

    exp_values = np.exp(values)
    total = float(exp_values.sum())

    if total <= 0:
        return np.ones_like(values) / max(1, len(values))

    return exp_values / total


def _normalize_scores(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.maximum(arr, 0.0)

    total = float(arr.sum())

    if total <= 0:
        return np.ones(len(arr), dtype=float) / max(1, len(arr))

    return arr / total


def normalized_entropy(scores: Sequence[float]) -> float:
    """Return normalized Shannon entropy in the range 0..1."""
    probs = _normalize_scores(scores)

    if len(probs) <= 1:
        return 0.0

    positive = probs[probs > 0]

    entropy = -float(
        np.sum(positive * np.log2(positive))
    )

    return entropy / math.log2(len(probs))


def predict_stored_model(
    model_info: Dict[str, Any],
    markers: Dict[str, str],
    bundle: ModelBundle,
) -> Tuple[Dict[str, float], float]:
    """Return child scores normalized to 0..1 and prediction entropy."""
    pipeline = model_info["pipeline"]
    '''
    # Ensure compatibility between different sclearn version
    pipeline = repair_pipeline(pipeline)
    '''

    feature_titles = [
        str(value)
        for value in model_info["feature_titles"]
    ]

    classes = [
        str(value)
        for value in model_info["classes"]
    ]

    imputed = _impute_for_nn(
        markers=markers,
        bundle=bundle,
    )

    row = {
        title: str(
            imputed.get(title, MISSING_STATE)
        )
        for title in feature_titles
    }

    known_categories = model_info.get("known_categories")

    if known_categories:
        for index, title in enumerate(feature_titles):
            if index >= len(known_categories):
                continue

            known = [
                str(value)
                for value in known_categories[index]
            ]

            if row[title] not in known and known:
                row[title] = (
                    "1"
                    if "1" in known
                    else known[0]
                )

    X = pd.DataFrame(
        [row],
        columns=feature_titles,
    ).astype(str)

    if hasattr(pipeline, "predict_proba"):
        raw = np.asarray(
            pipeline.predict_proba(X)[0],
            dtype=float,
        )
        probs = _normalize_scores(raw)

    elif hasattr(pipeline, "decision_function"):
        decision = np.asarray(
            pipeline.decision_function(X),
            dtype=float,
        )

        if decision.ndim > 1:
            decision = decision[0]

        if decision.size == 1 and len(classes) == 2:
            p1 = 1.0 / (
                1.0 + math.exp(-float(decision[0]))
            )
            probs = np.asarray(
                [1.0 - p1, p1],
                dtype=float,
            )
        else:
            probs = _softmax(decision)

    else:
        prediction = pipeline.predict(X)[0]

        probs = np.zeros(
            max(1, len(classes)),
            dtype=float,
        )

        try:
            index = int(prediction)

            if 0 <= index < len(probs):
                probs[index] = 1.0
            else:
                raise ValueError

        except Exception:
            label = str(prediction)

            if label in classes:
                probs[classes.index(label)] = 1.0
            else:
                probs[:] = 1.0 / len(probs)

    if len(probs) != len(classes):
        count = min(len(probs), len(classes))
        probs = _normalize_scores(probs[:count])
        classes = classes[:count]

    result = {
        label: float(score)
        for label, score in zip(classes, probs)
    }

    entropy = normalized_entropy(
        list(result.values())
    )

    return result, entropy


def profile_child_scores(
    node: Dict[str, Any],
    markers: Dict[str, str],
    data_type: str,
) -> Dict[str, float]:
    """Compare tester markers with child allele profiles.

    Binary/digit:
        score = 1 - mean(abs(query_state - child_mean_state))

    Character:
        score = mean(child_frequency_of_query_allele)

    Missing tester states are ignored.
    """
    child_profiles = node.get("child_profiles") or {}
    scores: Dict[str, float] = {}

    for child, profile in child_profiles.items():
        components: List[float] = []

        for feature, raw_value in markers.items():
            value = str(raw_value).strip()

            if value.lower() == MISSING_STATE:
                continue

            if data_type in {"binary", "digit"}:
                try:
                    query_state = float(value)
                    profile_state = profile.get(feature)

                    if profile_state is None:
                        continue

                    profile_state = float(profile_state)

                except (ValueError, TypeError):
                    continue

                components.append(
                    1.0 - abs(
                        query_state - profile_state
                    )
                )

            else:
                ratios = profile.get(feature) or {}

                if not isinstance(ratios, dict):
                    continue

                components.append(
                    float(
                        ratios.get(value, 0.0)
                    )
                )

        score = (
            float(sum(components) / len(components))
            if components
            else 0.0
        )

        scores[str(child)] = max(
            0.0,
            min(1.0, score),
        )

    return scores


def node_scores(
    node: Dict[str, Any],
    markers: Dict[str, str],
    bundle: ModelBundle,
    max_no_data_for_nn: float,
) -> Tuple[Dict[str, float], str, float | None]:
    """Choose NN/ML or allele-profile scoring for one hierarchy node."""
    default_child = node.get("default_child")

    if default_child:
        return {
            str(default_child): 1.0
        }, "single-child", 0.0

    missing_pct = (
        100.0 * missing_fraction(markers)
    )

    model_info = node.get("model")
    data_type = str(
        bundle.get("data_type", "character")
    ).lower()

    if model_info is None:
        return (
            profile_child_scores(
                node=node,
                markers=markers,
                data_type=data_type,
            ),
            "profile:model-none",
            None,
        )

    if missing_pct > max_no_data_for_nn:
        return (
            profile_child_scores(
                node=node,
                markers=markers,
                data_type=data_type,
            ),
            "profile:missing-data",
            None,
        )

    nn_scores, entropy = predict_stored_model(
        model_info=model_info,
        markers=markers,
        bundle=bundle,
    )

    max_entropy = float(
        bundle.get("max_entropy", 0.80)
    )

    if entropy > max_entropy:
        return (
            profile_child_scores(
                node=node,
                markers=markers,
                data_type=data_type,
            ),
            "profile:high-entropy",
            entropy,
        )

    return nn_scores, "NN", entropy



def format_report_path(
    full_path: str,
    report_level: int,
    delimiter: str,
) -> Tuple[str, str]:
    """Return the visible classification and the omitted upper hierarchy path.

    ``report_level`` is the number of lowest hierarchy labels displayed in the
    prediction text.  Upper labels that are hidden from the visible prediction
    are represented in the path column beginning with ``Root``.

    Examples for ``2|Food chain|Chicken products``:
        level 1 -> Chicken products          / Root.2.Food chain
        level 2 -> Food chain|Chicken products / Root.2
        level 3 -> 2|Food chain|Chicken products / Root.0
    """
    parts = [
        part.strip()
        for part in str(full_path).split(delimiter)
        if part.strip()
    ]

    if not parts:
        return "", "Root.0"

    visible_count = min(report_level, len(parts))
    visible = delimiter.join(parts[-visible_count:])
    omitted = parts[:-visible_count]

    root_path = (
        "Root." + ".".join(omitted)
        if omitted
        else "Root.0"
    )

    return visible, root_path


def format_short_prediction(
    prediction: Dict[str, Any],
    report_level: int,
    delimiter: str,
) -> str:
    """Format one indented prediction line for the short report."""
    visible, root_path = format_report_path(
        full_path=str(prediction["path"]),
        report_level=report_level,
        delimiter=delimiter,
    )

    terminal = bool(prediction.get("terminal", False))
    score = float(prediction.get("score", 0.0))

    return (
        f"     {visible}, {score:.4f}\t"
        f"{root_path}\t{terminal}"
    )


def format_short_record(
    title: str,
    predictions: Sequence[Dict[str, Any]],
    report_level: int,
    delimiter: str,
) -> str:
    """Format one query and its identification results."""
    lines = [title]

    if predictions:
        for prediction in sorted(
            predictions,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        ):
            lines.append(
                format_short_prediction(
                    prediction=prediction,
                    report_level=report_level,
                    delimiter=delimiter,
                )
            )
    else:
        lines.append("     Not identified")

    return "\n".join(lines)

def classify_markers(
    markers: Dict[str, str],
    bundle: ModelBundle,
    sensitivity: float,
    specificity: float,
    max_no_data_for_nn: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Traverse all hierarchical branches passing the specificity threshold."""
    delimiter = str(
        bundle.get("delimiter", "|")
    )

    terminal_predictions: List[Dict[str, Any]] = []
    intermediate_predictions: List[Dict[str, Any]] = []
    trace: List[str] = []

    stack: List[
        Tuple[
            Dict[str, Any],
            Tuple[str, ...],
            float,
        ]
    ] = [
        (
            bundle["tree"],
            tuple(),
            1.0,
        )
    ]

    while stack:
        node, path, path_score = stack.pop()
        children = node.get("children", {})

        if not children and path:
            if path_score > sensitivity:
                terminal_predictions.append({
                    "path": delimiter.join(path),
                    "score": path_score,
                    "terminal": True,
                })

            continue

        scores, method, entropy = node_scores(
            node=node,
            markers=markers,
            bundle=bundle,
            max_no_data_for_nn=max_no_data_for_nn,
        )

        node_name = (
            delimiter.join(path)
            if path
            else "<root>"
        )

        entropy_text = (
            ""
            if entropy is None
            else f", entropy={entropy:.4f}"
        )

        trace.append(
            f"{node_name}: {method}{entropy_text}"
        )

        for child, local_score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            local_score = max(
                0.0,
                min(1.0, float(local_score)),
            )

            new_path_score = min(
                path_score,
                local_score,
            )

            if child == TERMINAL_LABEL:
                if path and new_path_score > sensitivity:
                    terminal_predictions.append({
                        "path": delimiter.join(path),
                        "score": new_path_score,
                        "terminal": True,
                    })

                continue

            if local_score <= specificity:
                continue

            new_path = path + (child,)
            child_node = children.get(child)

            if (
                child_node is not None
                and child_node.get("children")
            ):
                intermediate_predictions.append({
                    "path": delimiter.join(new_path),
                    "score": new_path_score,
                    "terminal": False,
                })

            if child_node is not None:
                stack.append(
                    (
                        child_node,
                        new_path,
                        new_path_score,
                    )
                )

    def deduplicate(
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}

        for record in records:
            path = record["path"]

            if (
                path not in best
                or record["score"] > best[path]["score"]
            ):
                best[path] = record

        return sorted(
            best.values(),
            key=lambda record: record["score"],
            reverse=True,
        )

    terminals = deduplicate(
        terminal_predictions
    )

    if terminals:
        return terminals, trace

    intermediates = [
        record
        for record in deduplicate(intermediate_predictions)
        if float(record.get("score", 0.0)) > sensitivity
    ]

    if intermediates:
        max_depth = max(
            record["path"].count(delimiter) + 1
            for record in intermediates
        )

        deepest = [
            record
            for record in intermediates
            if (
                record["path"].count(delimiter) + 1
                == max_depth
            )
        ]

        return sorted(
            deepest,
            key=lambda record: record["score"],
            reverse=True,
        ), trace

    return [], trace


def classification_matches(
    expected_path: str,
    predictions: Sequence[Dict[str, Any]],
    report_level: int,
    delimiter: str,
) -> bool:
    """Return True if the expected category occurs among terminal predictions.

    Both the expected classification and each prediction are reduced to the
    hierarchy depth displayed by ``report_level``.  A match may occur anywhere
    in the list of terminal predictions; it does not have to be the top-scoring
    prediction.
    """
    expected = category_at_report_level(
        full_path=expected_path,
        report_level=report_level,
        delimiter=delimiter,
    ).strip()

    return any(
        bool(record.get("terminal", False))
        and category_at_report_level(
            full_path=str(record.get("path", "")),
            report_level=report_level,
            delimiter=delimiter,
        ).strip() == expected
        for record in predictions
    )



def category_at_report_level(
    full_path: str,
    report_level: int,
    delimiter: str,
) -> str:
    """Return the category label displayed at the requested report level."""
    visible, _ = format_report_path(
        full_path=full_path,
        report_level=report_level,
        delimiter=delimiter,
    )
    return visible


def build_overlap_matrix(
    records: Sequence[Dict[str, Any]],
    report_level: int,
    delimiter: str,
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Build expected-vs-predicted overlap counts.

    Every prediction returned for a tester genome is counted independently.
    If a genome has no prediction, the ``Not identified`` column is incremented.

    Categories are represented at ``report_level`` using the same shortening
    rules as the short textual report.
    """
    categories = set()
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in records:
        expected_full = str(record["expected_path"])
        expected = category_at_report_level(
            full_path=expected_full,
            report_level=report_level,
            delimiter=delimiter,
        )
        categories.add(expected)

        predictions = record.get("prediction_records") or []

        # Only successful terminal identifications contribute to predicted
        # category columns. Intermediate (terminal=False) records represent
        # incomplete identification and are therefore counted as
        # "Not identified" for the overlap heatmap.
        terminal_predictions = [
            prediction
            for prediction in predictions
            if bool(prediction.get("terminal", False))
        ]

        if not terminal_predictions:
            counts[expected]["Not identified"] += 1
            continue

        for prediction in terminal_predictions:
            predicted = category_at_report_level(
                full_path=str(prediction["path"]),
                report_level=report_level,
                delimiter=delimiter,
            )
            categories.add(predicted)
            counts[expected][predicted] += 1

    sorted_categories = sorted(categories, key=lambda value: value.casefold())
    return sorted_categories, counts


def save_overlap_tsv(
    file_path: Path,
    categories: Sequence[str],
    counts: Dict[str, Dict[str, int]],
) -> None:
    """Save the overlap matrix as a tab-separated text file."""
    with file_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )

        # Numbered prediction columns; Not identified is always last.
        writer.writerow(
            [""] +
            [str(index) for index in range(1, len(categories) + 1)] +
            ["Not identified"]
        )

        for row_index, expected in enumerate(categories, start=1):
            row = [f"{row_index}.{expected}"]

            for predicted in categories:
                row.append(
                    counts.get(expected, {}).get(predicted, 0)
                )

            row.append(
                counts.get(expected, {}).get("Not identified", 0)
            )

            writer.writerow(row)


def build_expected_category_totals(
    records: Sequence[Dict[str, Any]],
    report_level: int,
    delimiter: str,
) -> Dict[str, int]:
    """Return the number of tester genomes belonging to each expected category."""
    totals: Dict[str, int] = defaultdict(int)

    for record in records:
        expected = category_at_report_level(
            full_path=str(record["expected_path"]),
            report_level=report_level,
            delimiter=delimiter,
        )
        totals[expected] += 1

    return dict(totals)

'''
def save_overlap_heatmap(
    file_path: Path,
    categories: Sequence[str],
    counts: Dict[str, Dict[str, int]],
    expected_totals: Dict[str, int],
) -> bool:
    """Create a row-normalized PNG heatmap if matplotlib is available.

    The TSV retains absolute counts, but heatmap colors represent the percentage
    of tester genomes in each expected category.  Thus every row uses the same
    fixed 0..100% color scale despite different row sizes.  A tester genome may
    contribute to more than one predicted-category cell when several terminal
    predictions are returned, so percentages across a row need not sum to 100%.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False

    if not categories:
        return False

    row_count = len(categories)
    column_count = len(categories) + 1

    count_matrix = np.zeros((row_count, column_count), dtype=int)
    percent_matrix = np.zeros((row_count, column_count), dtype=float)

    for row_index, expected in enumerate(categories):
        for column_index, predicted in enumerate(categories):
            count_matrix[row_index, column_index] = (
                counts.get(expected, {}).get(predicted, 0)
            )

        count_matrix[row_index, -1] = (
            counts.get(expected, {}).get("Not identified", 0)
        )

        row_total = int(expected_totals.get(expected, 0))
        if row_total > 0:
            percent_matrix[row_index, :] = (
                100.0 * count_matrix[row_index, :] / row_total
            )

    width = max(7.0, min(20.0, 0.45 * column_count + 2.5))
    height = max(6.0, min(20.0, 0.45 * row_count + 2.0))

    fig, ax = plt.subplots(figsize=(width, height))

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "white_yellow_brown",
        ["white", "yellow", "saddlebrown"],
    )

    image = ax.imshow(
        percent_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=heatmap_cmap,
        vmin=0.0,
        vmax=100.0,
    )

    ax.set_xlabel(
        "Predicted category number "
        f"(column {column_count} = Not identified)"
    )
    ax.set_ylabel("Expected category number")

    ax.set_xticks(range(column_count))
    ax.set_xticklabels(
        [str(index) for index in range(1, column_count + 1)],
        rotation=90 if column_count > 15 else 0,
    )

    ax.set_yticks(range(row_count))
    ax.set_yticklabels(
        [str(index) for index in range(1, row_count + 1)]
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        label="Tester genomes within expected category (%)",
    )
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])

    fig.tight_layout()
    fig.savefig(file_path, dpi=200, bbox_inches="tight")

    try:
        plt.show()
    finally:
        plt.close(fig)

    return True
'''

def save_overlap_heatmap(
    file_path: Path,
    categories: Sequence[str],
    counts: Dict[str, Dict[str, int]],
    expected_totals: Dict[str, int],
) -> bool:
    """Create a row-normalized PNG heatmap if matplotlib is available.

    The TSV retains absolute counts, but heatmap colors represent the percentage
    of tester genomes in each expected category. Thus every row uses the same
    fixed 0..100% color scale despite different row sizes. A tester genome may
    contribute to more than one predicted-category cell when several terminal
    predictions are returned, so percentages across a row need not sum to 100%.

    Thin black borders highlight:
      - Diagonal cells, excluding the last row.
      - All cells in the last row.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Rectangle
    except ImportError:
        return False

    if not categories:
        return False

    row_count = len(categories)
    column_count = len(categories) + 1

    count_matrix = np.zeros((row_count, column_count), dtype=int)
    percent_matrix = np.zeros((row_count, column_count), dtype=float)

    for row_index, expected in enumerate(categories):
        for column_index, predicted in enumerate(categories):
            count_matrix[row_index, column_index] = (
                counts.get(expected, {}).get(predicted, 0)
            )

        count_matrix[row_index, -1] = (
            counts.get(expected, {}).get("Not identified", 0)
        )

        row_total = int(expected_totals.get(expected, 0))
        if row_total > 0:
            percent_matrix[row_index, :] = (
                100.0 * count_matrix[row_index, :] / row_total
            )

    width = max(7.0, min(20.0, 0.45 * column_count + 2.5))
    height = max(6.0, min(20.0, 0.45 * row_count + 2.0))

    fig, ax = plt.subplots(figsize=(width, height))

    heatmap_cmap = LinearSegmentedColormap.from_list(
        "white_yellow_brown",
        ["white", "yellow", "saddlebrown"],
    )

    image = ax.imshow(
        percent_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=heatmap_cmap,
        vmin=0.0,
        vmax=100.0,
    )

    # ------------------------------------------------------------
    # Highlight diagonal cells, excluding the last row
    # ------------------------------------------------------------
    border_width = 0.5
    border_color = "black"

    for index in range(row_count):
        ax.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor=border_color,
                linewidth=border_width,
                zorder=3,
            )
        )

    # ------------------------------------------------------------
    # Highlight the entire last column with one rectangular border
    # ------------------------------------------------------------
    ax.add_patch(
        Rectangle(
            (column_count - 1.5, -0.5),
            1,
            row_count,
            fill=False,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=3,
        )
    )
    
    # ------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------
    ax.set_xlabel(
        "Predicted category number "
        f"(column {column_count} = Not identified)"
    )
    ax.set_ylabel("Expected category number")

    ax.set_xticks(range(column_count))
    ax.set_xticklabels(
        [str(index) for index in range(1, column_count + 1)],
        rotation=90 if column_count > 15 else 0,
    )

    ax.set_yticks(range(row_count))
    ax.set_yticklabels(
        [str(index) for index in range(1, row_count + 1)]
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        label="Tester genomes within expected category (%)",
    )
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])

    fig.tight_layout()
    fig.savefig(file_path, dpi=200, bbox_inches="tight")

    try:
        plt.show()
    finally:
        plt.close(fig)

    return True
    
'''
def repair_pipeline(pipeline):
    """
    Attempt to repair missing attributes in pipelines
    serialized with older scikit-learn versions.
    """

    visited = set()

    def repair(obj):

        if id(obj) in visited:
            return

        visited.add(id(obj))

        # Repair Pipeline
        if isinstance(obj, Pipeline):

            if not hasattr(obj, "transform_input"):
                obj.transform_input = None
                print("Repaired Pipeline.transform_input")

            for _, step in obj.steps:
                repair(step)

        # Repair ColumnTransformer
        elif isinstance(obj, ColumnTransformer):

            for _, transformer, _ in obj.transformers:
                if transformer not in ("drop", "passthrough"):
                    repair(transformer)

            if hasattr(obj, "transformers_"):
                for _, transformer, _ in obj.transformers_:
                    if isinstance(transformer, str):
                        continue
                    repair(transformer)

        # Repair OneHotEncoder
        elif isinstance(obj, OneHotEncoder):

            if not hasattr(obj, "_drop_idx_after_grouping"):
                obj._drop_idx_after_grouping = None
                print(
                    "Repaired OneHotEncoder."
                    "_drop_idx_after_grouping"
                )

    repair(pipeline)

    return pipeline
'''


# ---------------------------------------------------------------------------
# Per-cluster evaluation. Level is counted UP from the terminal category.
# ---------------------------------------------------------------------------

def evaluated_prefix(full_path: str, level: int, delimiter: str) -> str:
    parts = [p.strip() for p in str(full_path).split(delimiter) if p.strip()]
    if not parts:
        return ""
    # level 1 -> full terminal path; level 2 -> parent; ... root -> first label.
    depth = max(1, len(parts) - level + 1)
    return delimiter.join(parts[:depth])


def category_for_prediction(path: str, required_depth: int, delimiter: str) -> str | None:
    parts = [p.strip() for p in str(path).split(delimiter) if p.strip()]
    if len(parts) < required_depth:
        return None
    return delimiter.join(parts[:required_depth])


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def natural_key(label: str):
    import re
    return [int(x) if x.isdigit() else x.casefold()
            for x in re.split(r"(\d+)", str(label))]


def write_heatmap(
    path: Path,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    normalize: bool = False,
    show_graph: bool = False
) -> None:

    import matplotlib.pyplot as plt

    data = matrix.astype(float)

    # Convert absolute counts to row percentages if requested.
    if normalize:
        sums = data.sum(axis=1, keepdims=True)
        data = np.divide(
            data * 100,
            sums,
            out=np.zeros_like(data),
            where=sums != 0
        )

    nrows, ncols = data.shape

    fig, ax = plt.subplots(
        figsize=(
            max(8, min(26, 0.48 * ncols + 4)),
            max(6, min(26, 0.48 * nrows + 3))
        )
    )

    im = ax.imshow(
        data,
        cmap="YlOrBr",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=100 if normalize else None
    )

    # Use exactly the same labels as in the text confusion matrix.
    ax.set_xticks(
        range(ncols),
        labels=column_labels,
        rotation=90,
        fontsize=8
    )

    ax.set_yticks(
        range(nrows),
        labels=row_labels,
        fontsize=8
    )

    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("Expected cluster")
    ax.set_title(title)

    # Display numerical values inside heatmap cells.
    if nrows <= 35 and ncols <= 35:
        for i in range(nrows):
            for j in range(ncols):
                value = data[i, j]

                ax.text(
                    j,
                    i,
                    f"{value:.1f}" if normalize else str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if value > (
                            65 if normalize
                            else max(1, data.max() * 0.65)
                        )
                        else "black"
                    )
                )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Row percentage (%)"
            if normalize
            else "Genome count"
        )
    )

    fig.tight_layout()

    # Always save the heatmap.
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    # Display only when requested.
    try:
        if show_graph:
            plt.show()
    finally:
        plt.close(fig)
        

def main() -> int:
    args = parse_args()
    if not args.delimiter:
        raise ValueError("--delimiter cannot be empty")
    if args.level < 1:
        raise ValueError("--level must be >= 1")
    for name in ("max_no_data_for_NN", "max_no_data_allowed"):
        if not 0 <= getattr(args, name) <= 100:
            raise ValueError(f"--{name} must be between 0 and 100")
    if args.max_no_data_for_NN > args.max_no_data_allowed:
        raise ValueError("--max_no_data_for_NN cannot exceed --max_no_data_allowed")
    for name in ("sensitivity", "specificity"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name} must be between 0 and 1")

    input_root = Path(args.input_folder)
    if not input_root.is_absolute():
        input_root = SCRIPT_DIR / input_root
    project_input = input_root / args.project_folder
    matrix_path = project_input / args.matrix_file
    model_path = Path(args.model_file)
    if not model_path.is_absolute():
        model_path = project_input / model_path
    bundle = load_bundle(model_path)
    matrix = load_matrix(matrix_path)
    feature_titles = matrix[0][1:]
    validate_feature_titles(feature_titles, bundle)

    output_root = Path(args.output_folder)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    output_dir = output_root / args.project_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = model_path.stem
    level = args.level
    # Output file names
    report_path = output_dir / f"{stem}_performance_{level}.txt"
    counts_path = output_dir / f"{stem}_confusion_{level}.tsv"
    raw_png = output_dir / f"{stem}_confusion_{level}.png"
    pct_png = output_dir / f"{stem}_confusion_percent_{level}.png"

    observations = []
    for idx, row in enumerate(matrix[1:], 1):
        expected, genome = parse_heading(row[0], args.delimiter)
        markers = build_markers(feature_titles, row[1:], bundle)
        missing = 100.0 * missing_fraction(markers)
        predictions = []
        if missing <= args.max_no_data_allowed:
            predictions, trace = classify_markers(
                markers, bundle, args.sensitivity, args.specificity,
                args.max_no_data_for_NN)
        else:
            trace = ["missing-data:rejected"]
        true_category = evaluated_prefix(expected, args.level, args.delimiter)
        depth = len(true_category.split(args.delimiter))
        # A prediction must REACH the requested hierarchy depth. A correct
        # intermediate node counts as correct when evaluating that node's depth.
        reached = []
        for p in predictions:
            label = category_for_prediction(str(p["path"]), depth, args.delimiter)
            if label is not None:
                reached.append((label, float(p["score"])))
        reached.sort(key=lambda x: x[1], reverse=True)
        predicted = reached[0][0] if reached else "Not identified"
        # Accuracy is based on the highest-scoring prediction, not oracle
        # matching against any candidate. Preserve candidate details separately.
        observations.append((genome, true_category, predicted, reached, missing))
        print(f"{idx}/{len(matrix)-1}: {genome}: {true_category} -> {predicted}")

    labels = sorted({x[1] for x in observations} |
                    {x[2] for x in observations if x[2] != "Not identified"},
                    key=natural_key)
    # Actual classes are rows; the extra outcome is a prediction column only.
    # Always include the Not identified column for consistent report layouts.
    column_labels = labels + ["Not identified"]
    row_lookup = {name: i for i, name in enumerate(labels)}
    col_lookup = {name: i for i, name in enumerate(column_labels)}
    confusion = np.zeros((len(labels), len(column_labels)), dtype=int)
    for _, expected, predicted, _, _ in observations:
        confusion[row_lookup[expected], col_lookup[predicted]] += 1

    # Classification metrics on the identified genomes, plus coverage and
    # end-to-end accuracy on ALL genomes. No identification is never correct.
    n = len(observations)
    identified = sum(p != "Not identified" for _, _, p, _, _ in observations)
    correct = sum(t == p for _, t, p, _, _ in observations)
    wrong = identified - correct
    unidentified = n - identified
    per_class = []
    for label in labels:
        tp = sum(t == label and p == label for _, t, p, _, _ in observations)
        fp = sum(t != label and p == label for _, t, p, _, _ in observations)
        fn = sum(t == label and p != label for _, t, p, _, _ in observations)
        support = tp + fn
        precision = safe_div(tp, tp+fp)
        recall = safe_div(tp, support)
        f1 = safe_div(2*precision*recall, precision+recall)
        per_class.append((label, precision, recall, f1, support))
    total_support = sum(x[4] for x in per_class)
    macro = [safe_div(sum(x[k] for x in per_class), len(per_class)) for k in (1,2,3)]
    weighted = [safe_div(sum(x[k]*x[4] for x in per_class), total_support) for k in (1,2,3)]
    balanced = macro[1]
    accuracy = safe_div(correct, n)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("FINAL HELD-OUT TEST PERFORMANCE\n" + "="*31 + "\n\n")
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Project: {args.project_folder}\n")
        f.write(f"Evaluation level: {args.level} (1 = terminal; higher = toward genomic root)\n")
        f.write(f"Held-out test genomes: {n}\n")
        f.write(f"SNP features: {len(bundle['feature_titles'])}\n")
        f.write(f"Target: {len(labels)} evaluated hierarchy clusters\n\n")
        f.write(f"Accuracy (all genomes): {accuracy:.10f} ({accuracy*100:.2f}%)\n")
        f.write(f"Balanced accuracy: {balanced:.4f} ({balanced*100:.2f}%)\n")
        f.write(f"Correct predictions: {correct}/{n}\n")
        f.write(f"Incorrect predictions: {wrong}/{n}\n")
        f.write(f"Not identified: {unidentified}/{n}\n")
        f.write(f"Identification rate: {safe_div(identified,n)*100:.2f}%\n")
        f.write(f"Accuracy among identified: {safe_div(correct,identified)*100:.2f}%\n\n")
        f.write("Classification report:\n\n")
        f.write(f"{'Cluster':<38} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>9}\n")
        for label, precision, recall, f1, support in per_class:
            f.write(f"{label:<38} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>9}\n")
        # Abstentions are an outcome, not a true class. Display their count
        # explicitly without incorrectly adding them to the macro averages.
        f.write(f"{'Not identified':<38} {'—':>10} {'—':>10} {'—':>10} {unidentified:>9}\n")
        f.write(f"\n{'Macro avg':<38} {macro[0]:>10.4f} {macro[1]:>10.4f} {macro[2]:>10.4f} {n:>9}\n")
        f.write(f"{'Weighted avg':<38} {weighted[0]:>10.4f} {weighted[1]:>10.4f} {weighted[2]:>10.4f} {n:>9}\n")
        f.write("\nConfusion matrix:\nRows = true cluster\nColumns = predicted cluster\n")
        f.write("Row labels = " + ", ".join(labels) + "\n")
        f.write("Column labels = " + ", ".join(column_labels) + "\n\n")
        f.write(np.array2string(confusion) + "\n\n")
        f.write("Interpretation:\n")
        f.write(f"{correct} genomes correctly assigned; {wrong} assigned to an incorrect "
                f"cluster; {unidentified} did not reach the requested evaluation level.\n")
        f.write("\nNotes: Precision, recall and F1 treat unidentified predictions as "
                "false negatives for their expected categories.\n")
        f.write("The Not identified row in the classification report gives the number of abstentions, not true-class support.\n")
        f.write("Balanced accuracy is macro-average recall.\n")
        f.write("Only the highest-scoring eligible prediction is scored.\n")
    with counts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Expected / Predicted", *column_labels])
        for label in labels:
            writer.writerow([label, *confusion[row_lookup[label]].tolist()])

    # Select the backend once, before creating either heatmap.
    import matplotlib
    
    if args.show_graph == "no":
        matplotlib.use("Agg")
    else:
        backend = matplotlib.get_backend()
    
        if str(backend).lower().endswith("agg"):
            try:
                matplotlib.use("TkAgg")
            except ImportError as exc:
                raise RuntimeError(
                    "--show_graph requires an interactive Matplotlib backend "
                    "(e.g. TkAgg). Check that tkinter is installed and that "
                    "the script is running in a graphical desktop session."
                ) from exc
    
        print(f"Matplotlib backend: {matplotlib.get_backend()}")
    
    # Absolute-count confusion matrix.
    write_heatmap(
        path=raw_png,
        matrix=confusion,
        row_labels=labels,
        column_labels=column_labels,
        title=f"{stem}: confusion matrix (level {args.level})",
        normalize=False,
        show_graph=(args.show_graph == "count")
    )
    
    # Row-normalized percentage confusion matrix.
    write_heatmap(
        path=pct_png,
        matrix=confusion,
        row_labels=labels,
        column_labels=column_labels,
        title=f"{stem}: row-normalized confusion (level {args.level})",
        normalize=True,
        show_graph=(args.show_graph == "percent")
    )                  
                  
    print(f"Correct: {correct}/{n}; incorrect: {wrong}; not identified: {unidentified}")
    print(f"Accuracy: {accuracy*100:.2f}%; balanced accuracy: {balanced*100:.2f}%")
    for path in (report_path, counts_path, raw_png, pct_png):
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
