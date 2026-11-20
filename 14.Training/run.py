#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a hierarchical classifier with neural-network/model and allele-profile fallbacks."""

from __future__ import annotations

import argparse
import inspect
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR.parent / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

try:
    import NeuralNetwork as NN
    import nn__algorithm_selector as NAS
    from model_bundle import ModelBundle
except Exception as exc:
    raise SystemExit(f"ERROR: Cannot import required modules from {LIB_DIR}: {exc}")

SUPPORTED_ALGORITHMS = tuple(NAS.SUPPORTED_MODELS)
TERMINAL_LABEL = "__TERMINAL__"
BUNDLE_VERSION = 2


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a Boolean value, got {value!r}")


def none_or_int(value: str) -> int | None:
    text = str(value).strip().lower()
    return None if text in {"none", "null", "na", ""} else int(value)


def none_or_str(value: str) -> str | None:
    text = str(value).strip()
    return None if text.lower() in {"none", "null", "na", ""} else text


def parse_hidden_layers(value: str) -> Tuple[int, ...]:
    values = tuple(int(v.strip()) for v in str(value).split(",") if v.strip())
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            "--hidden_layer_sizes must contain positive comma-separated integers"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train hierarchical classifiers from a categorical SNP/marker matrix, "
            "calculate per-cluster allele profiles, and create a deployable classifier package."
        )
    )
    parser.add_argument("project_folder", help="Project subfolder inside input_folder.")
    parser.add_argument(
        "algorithm",
        choices=["AUTO", *SUPPORTED_ALGORITHMS],
        help="Fixed model at all eligible nodes, or AUTO for node-specific model selection.",
    )

    parser.add_argument("-i", "--input_folder", default="input")
    parser.add_argument("-o", "--output_folder", default="output")
    parser.add_argument(
        "-m", "--matrix_file", "--input_file",
        dest="matrix_file", required=True,
        help="Training CSV/TSV matrix inside input_folder/project_folder."
    )
    parser.add_argument(
        "-f", "--feature_file", required=True,
        help="Feature annotation TSV/CSV file inside input_folder/project_folder."
    )
    parser.add_argument(
        "-a", "--allele_file", "--allele_matrix",
        dest="allele_file", required=True,
        help=(
            "Required two-row allele CSV/TSV matrix: feature titles in row 1 "
            "and common allelic states in row 2. Required for both binary and character matrices."
        ),
    )
    parser.add_argument(
        "-d", "--delimiter", default="|",
        help="Delimiter in hierarchical row headings (default: '|')."
    )
    parser.add_argument("--marker_style", choices=["plain"], default="plain")
    parser.add_argument("--empty_symbol", default="")
    parser.add_argument(
        "--remove_empty_field", "--remove_empty_filed",
        dest="remove_empty_field", type=float, default=1.0,
        help="Remove feature columns with missing fraction above this value (0..1)."
    )
    parser.add_argument("--cv_folds", type=int, default=5)

    # Bundle metadata.
    parser.add_argument("--author", default="", help="Model author (default: empty).")
    parser.add_argument("--description", default="", help="Model description (default: empty).")
    parser.add_argument("--version", default="", help="Model/data release version (default: empty).")
    parser.add_argument("--contact_details", default="", help="Contact details (default: empty).")

    # Hierarchical training/fallback controls.
    parser.add_argument(
        "--min_cluster_size", type=int, default=10,
        help=(
            "Minimum number of training genomes at a splitting node required to train "
            "a neural-network/ML model. Smaller nodes retain model=None and use allele profiles "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--max_entropy", "--max_entroby",
        dest="max_entropy", type=float, default=0.80,
        help=(
            "Maximum normalized Shannon entropy (0..1) accepted for an NN/model prediction. "
            "Higher-entropy predictions fall back to allele-profile similarity (default: 0.80)."
        ),
    )

    # Logistic regression / SVC.
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--penalty", default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--class_weight", type=none_or_str, default=None)
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--gamma", default=None)
    parser.add_argument("--coef0", type=float, default=None)

    # RF / DT.
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=none_or_int, default=None)
    parser.add_argument("--max_features", type=none_or_str, default=None)
    parser.add_argument("--min_samples_split", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=None)
    parser.add_argument("--bootstrap", type=str2bool, default=None)
    parser.add_argument("--oob_score", type=str2bool, default=None)
    parser.add_argument("--criterion", default=None)
    parser.add_argument("--splitter", default=None)

    # MLP.
    parser.add_argument("--hidden_layer_sizes", type=parse_hidden_layers, default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--learning_rate", default=None)
    parser.add_argument("--learning_rate_init", type=float, default=None)
    parser.add_argument("--early_stopping", type=str2bool, default=None)

    # KNN.
    parser.add_argument("--n_neighbors", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--p", type=int, default=None)

    # Naive Bayes.
    parser.add_argument("--fit_prior", type=str2bool, default=None)

    # XGBoost.
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample_bytree", type=float, default=None)
    parser.add_argument("--reg_lambda", type=float, default=None)
    parser.add_argument("--reg_alpha", type=float, default=None)
    parser.add_argument("--min_child_weight", type=float, default=None)
    parser.add_argument("--scale_pos_weight", type=float, default=None)

    return parser.parse_args()


def _read_delimited_file(path: Path) -> List[List[str]]:
    import csv
    suffix = path.suffix.lower()
    if suffix == ".csv":
        delimiter = ","
    elif suffix in {".tsv", ".txt"}:
        delimiter = "\t"
    else:
        raise ValueError(f"Unsupported tabular file extension: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row for row in csv.reader(handle, delimiter=delimiter)
            if any(str(c).strip() for c in row)
        ]


def read_feature_file(path: Path) -> Tuple[str, List[List[str]], List[Dict[str, str]]]:
    rows = _read_delimited_file(path)
    if len(rows) < 3:
        raise ValueError(
            "Feature file must contain a reference-name row, a header row, "
            "and at least one feature row."
        )

    reference_line = next((cell.strip() for cell in rows[0] if cell.strip()), "")
    if reference_line.lower().startswith("#reference genome:"):
        reference_name = reference_line.split(":", 1)[1].strip()
    else:
        reference_name = reference_line.lstrip("#").strip()

    headers = [cell.strip() for cell in rows[1]]
    normalized = {h.lower(): i for i, h in enumerate(headers)}
    required = {"snp", "location in reference sequence", "alleles", "context"}
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(
            "Feature file is missing required column(s): " + ", ".join(missing)
        )

    feature_records: List[Dict[str, str]] = []
    for row_number, row in enumerate(rows[2:], start=3):
        padded = row + [""] * (len(headers) - len(row))
        record = {headers[i]: padded[i].strip() for i in range(len(headers))}
        snp = padded[normalized["snp"]].strip()
        if not snp:
            raise ValueError(f"Feature file row {row_number} has an empty SNP identifier.")
        feature_records.append(record)

    return reference_name, rows, feature_records


def read_allele_file(path: Path) -> Tuple[List[List[str]], Dict[str, str]]:
    rows = _read_delimited_file(path)
    if len(rows) < 2:
        raise ValueError(
            "Allele file must contain two rows: feature titles and common allelic states."
        )
    titles = [str(v).strip() for v in rows[0]]
    values = [str(v).strip() for v in rows[1]]
    if len(titles) != len(values):
        raise ValueError("The two allele-file rows contain different numbers of cells.")
    if len(set(titles)) != len(titles):
        raise ValueError("The allele-file title row contains duplicate feature IDs.")
    return rows[:2], dict(zip(titles, values))


def _feature_id(record: Dict[str, str]) -> str:
    for key, value in record.items():
        if key.strip().lower() == "snp":
            return str(value).strip()
    return ""


def validate_metadata(
    matrix: NAS.MatrixData,
    feature_records: Sequence[Dict[str, str]],
    allele_map: Dict[str, str],
) -> None:
    annotation_ids = [_feature_id(record) for record in feature_records]
    annotation_set = set(annotation_ids)
    missing_annotations = [f for f in matrix.feature_titles if f not in annotation_set]
    if missing_annotations:
        raise ValueError(
            "Training feature(s) missing from the feature annotation file: "
            + ", ".join(missing_annotations[:20])
        )

    missing_alleles = [f for f in matrix.feature_titles if f not in allele_map]
    if missing_alleles:
        raise ValueError(
            "Training feature(s) missing from the allele matrix: "
            + ", ".join(missing_alleles[:20])
        )

    empty_common_alleles = [
        f for f in matrix.feature_titles
        if not str(allele_map.get(f, "")).strip()
    ]
    if empty_common_alleles:
        raise ValueError(
            "Common allelic state is empty for training feature(s): "
            + ", ".join(empty_common_alleles[:20])
        )


def _context_for(X: pd.DataFrame, y: Sequence[str], data_type: str) -> NAS.SelectionContext:
    counts = Counter(str(v) for v in y)
    imbalance = max(counts.values()) / max(1, min(counts.values()))
    missing_fraction = (
        float((X.astype(str) == NAS.MISSING_TOKEN).to_numpy().mean())
        if X.size else 0.0
    )
    return NAS.SelectionContext(
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=len(counts),
        class_counts=dict(counts),
        imbalance_ratio=imbalance,
        data_type=data_type,
        missing_fraction=missing_fraction,
    )


def _cli_overrides(args: argparse.Namespace, algorithm: str) -> Dict[str, Any]:
    keys: Dict[str, Tuple[str, ...]] = {
        "LR": ("C", "penalty", "solver", "max_iter", "class_weight"),
        "SVC": ("C", "kernel", "degree", "gamma", "coef0", "class_weight"),
        "RF": (
            "n_estimators", "max_depth", "max_features", "min_samples_split",
            "min_samples_leaf", "bootstrap", "class_weight", "oob_score",
        ),
        "DT": (
            "criterion", "splitter", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features", "class_weight",
        ),
        "MLP": (
            "hidden_layer_sizes", "activation", "solver", "alpha", "batch_size",
            "learning_rate", "learning_rate_init", "max_iter", "early_stopping",
        ),
        "KNN": ("n_neighbors", "weights", "metric", "p"),
        "NBayes": ("alpha", "fit_prior"),
        "XGBoost": (
            "n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "reg_lambda", "reg_alpha", "min_child_weight",
            "gamma", "scale_pos_weight",
        ),
    }
    return {
        key: getattr(args, key)
        for key in keys.get(algorithm, ())
        if getattr(args, key, None) is not None
    }


def _make_model(algorithm: str, parameters: Dict[str, Any]) -> Any:
    mapping = {
        "LR": NN.LR,
        "SVC": NN.SVC,
        "RF": NN.RF,
        "DT": NN.DT,
        "MLP": NN.MLP,
        "KNN": NN.KNN,
        "NBayes": NN.NBayes,
        "XGBoost": NN.XGBoost,
    }
    cls = mapping[algorithm]
    signature = inspect.signature(cls.__init__)
    valid = set(signature.parameters) - {"self"}
    params = {k: v for k, v in parameters.items() if k in valid}
    params["marker_style"] = "plain"
    return cls(**params)



def _calculate_training_accuracy(
    model: Any,
    X: pd.DataFrame,
    y: Sequence[str],
) -> float:
    """Calculate classification accuracy on the data used to train one node.

    The stored pipelines may return either original class labels or integer
    class indices. This helper normalizes both representations before
    comparing predictions with the original hierarchical child labels.
    """
    try:
        feature_titles = list(X.columns.astype(str))
        X_predict = X.copy()
        X_predict.columns = feature_titles
        X_predict = X_predict.astype(str)

        predictions = np.asarray(
            model.pipeline.predict(X_predict),
            dtype=object,
        ).reshape(-1)

        classes = [
            str(value)
            for value in np.asarray(
                getattr(model, "classes_", []),
                dtype=object,
            ).reshape(-1)
        ]

        normalized_predictions: List[str] = []

        for prediction in predictions:
            text_prediction = str(prediction)

            try:
                index = int(prediction)
            except (TypeError, ValueError):
                index = -1

            if 0 <= index < len(classes):
                normalized_predictions.append(classes[index])
            else:
                normalized_predictions.append(text_prediction)

        expected = [str(value) for value in y]

        if len(normalized_predictions) != len(expected) or not expected:
            return float("nan")

        correct = sum(
            predicted == observed
            for predicted, observed in zip(
                normalized_predictions,
                expected,
            )
        )

        return float(correct / len(expected))

    except Exception as exc:
        print(
            "WARNING: Could not calculate training accuracy "
            f"for a node: {exc}"
        )
        return float("nan")


def _train_node_model(
    X: pd.DataFrame,
    y: Sequence[str],
    algorithm_mode: str,
    data_type: str,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], float, str, float]:
    if algorithm_mode == "AUTO":
        selection = NAS.recommend_model(
            X=X,
            y=y,
            data_type=data_type,
            cv_folds=args.cv_folds,
            top_alternatives=0,
        )
        if selection.best is None:
            raise ValueError(f"AUTO could not select a model: {selection.status}")
        algorithm = selection.best.algorithm
        parameters = dict(selection.best.parameters)
        selection_score = float(selection.best.score)
        selection_status = selection.status
    else:
        algorithm = algorithm_mode
        ctx = _context_for(X, y, data_type)
        parameters = dict(NAS.model_parameters(algorithm, ctx))
        parameters.update(_cli_overrides(args, algorithm))
        selection_score = float("nan")
        selection_status = "fixed algorithm"

    if algorithm == "KNN" and "n_neighbors" in parameters:
        approximate_train_n = max(1, int(math.floor(0.8 * len(y))))
        parameters["n_neighbors"] = max(
            1, min(int(parameters["n_neighbors"]), approximate_train_n)
        )

    model = _make_model(algorithm, parameters)
    model.train(
        X.to_numpy(dtype=object),
        np.asarray(y, dtype=object),
        feature_titles=list(X.columns.astype(str)),
        drop_singletons=False,
    )

    # Calculate training-set accuracy explicitly from the fitted pipeline.
    # Some model wrappers do not populate training_metrics["accuracy"].
    accuracy = _calculate_training_accuracy(
        model=model,
        X=X,
        y=y,
    )

    # Use a wrapper-provided value only as a fallback.
    if not math.isfinite(accuracy):
        training_metrics = getattr(model, "training_metrics", None)

        if training_metrics:
            value = training_metrics.get("accuracy")

            if value is not None:
                try:
                    wrapper_accuracy = float(value)

                    if math.isfinite(wrapper_accuracy):
                        accuracy = wrapper_accuracy

                except (TypeError, ValueError):
                    pass

    stored_model: Dict[str, Any] = {
        "pipeline": model.pipeline,
        "classes": [str(v) for v in np.asarray(model.classes_, dtype=object)],
        "feature_titles": list(model.feature_titles or []),
    }

    if algorithm == "NBayes":
        encoder = model.pipeline.named_steps.get("pre")
        if encoder is not None and hasattr(encoder, "categories_"):
            stored_model["known_categories"] = [
                [str(v) for v in values] for values in encoder.categories_
            ]

    return algorithm, parameters, stored_model, accuracy, selection_status, selection_score


def _is_missing(value: Any) -> bool:
    text = str(value).strip()
    return text in {"", NAS.MISSING_TOKEN} or text.lower() in {"n", "nan", "nd", "none"}


def _similarity_profile(
    X: pd.DataFrame,
    indices: Sequence[int],
    data_type: str,
) -> Dict[str, Any]:
    """Calculate an allele profile for a cluster.

    binary:
        feature -> mean of observed 0/1 states.

    character:
        feature -> {allelic_symbol: observed_ratio, ...}
    """
    profile: Dict[str, Any] = {}

    for title in X.columns:
        values = [
            str(X.iloc[i][title]).strip()
            for i in indices
            if not _is_missing(X.iloc[i][title])
        ]

        if data_type == "binary":
            numeric = []
            for value in values:
                try:
                    number = float(value)
                except ValueError:
                    continue
                if number in (0.0, 1.0):
                    numeric.append(number)
            profile[str(title)] = (
                float(sum(numeric) / len(numeric)) if numeric else None
            )
        else:
            counts = Counter(values)
            total = sum(counts.values())
            profile[str(title)] = (
                {symbol: count / total for symbol, count in counts.items()}
                if total else {}
            )

    return profile


def _build_tree(
    matrix: NAS.MatrixData,
    path: Tuple[str, ...],
    algorithm_mode: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    level = len(path)
    indices = [
        i for i, labels in enumerate(matrix.labels)
        if tuple(labels[:level]) == path
    ]

    next_labels: List[str] = []
    child_indices: Dict[str, List[int]] = {}
    for i in indices:
        labels = matrix.labels[i]
        child = labels[level].strip() if level < len(labels) else ""
        child = child if child else TERMINAL_LABEL
        next_labels.append(child)
        child_indices.setdefault(child, []).append(i)

    unique_children = list(dict.fromkeys(next_labels))
    real_children = [c for c in unique_children if c != TERMINAL_LABEL]

    node: Dict[str, Any] = {
        "path": list(path),
        "n_samples": len(indices),
        "children": {},
        "model": None,
        "model_skip_reason": "",
        "default_child": None,
        "similarity_profile": _similarity_profile(matrix.X, indices, matrix.data_type),
        "child_profiles": {
            child: _similarity_profile(matrix.X, idxs, matrix.data_type)
            for child, idxs in child_indices.items()
        },
        "child_sample_counts": {
            child: len(idxs) for child, idxs in child_indices.items()
        },
    }

    if len(unique_children) == 0 or unique_children == [TERMINAL_LABEL]:
        return node

    if len(unique_children) == 1:
        node["default_child"] = unique_children[0]
    elif len(indices) < args.min_cluster_size:
        node["model_skip_reason"] = (
            f"cluster size {len(indices)} is smaller than --min_cluster_size "
            f"{args.min_cluster_size}"
        )
        print(
            f"Node {args.delimiter.join(path) if path else '<root>'}: "
            f"model=None, samples={len(indices)}, classes={len(unique_children)}; "
            f"using allele-profile fallback ({node['model_skip_reason']})"
        )
    else:
        X_node = matrix.X.iloc[indices, :].reset_index(drop=True)
        try:
            algorithm, params, stored_model, accuracy, status, score = _train_node_model(
                X=X_node,
                y=next_labels,
                algorithm_mode=algorithm_mode,
                data_type=matrix.data_type,
                args=args,
            )
            node["model"] = {
                "algorithm": algorithm,
                "parameters": params,
                "training_accuracy": accuracy,
                "selection_status": status,
                "selection_score": score,
                **stored_model,
            }
            message = (
                f"Node {args.delimiter.join(path) if path else '<root>'}: "
                f"{algorithm}, samples={len(indices)}, classes={len(unique_children)}"
            )
            if math.isfinite(accuracy):
                message += f", accuracy={accuracy:.4f}"
            print(message)
        except Exception as exc:
            node["model_skip_reason"] = f"model training failed: {exc}"
            print(
                f"WARNING: Node {args.delimiter.join(path) if path else '<root>'}: "
                f"model=None; using allele-profile fallback ({exc})"
            )

    for child in real_children:
        node["children"][child] = _build_tree(
            matrix=matrix,
            path=path + (child,),
            algorithm_mode=algorithm_mode,
            args=args,
        )

    return node


def _count_models(node: Dict[str, Any]) -> int:
    return (1 if node.get("model") else 0) + sum(
        _count_models(child) for child in node.get("children", {}).values()
    )


def _count_profile_nodes(node: Dict[str, Any]) -> int:
    return 1 + sum(
        _count_profile_nodes(child) for child in node.get("children", {}).values()
    )


def _parse_feature_specs(
    feature_records: Sequence[Dict[str, str]],
    selected_titles: Sequence[str],
) -> List[Dict[str, str]]:
    by_id = {_feature_id(record): dict(record) for record in feature_records}
    return [by_id[title] for title in selected_titles]


def main() -> int:
    args = parse_args()

    if not args.delimiter:
        raise ValueError("--delimiter cannot be empty")
    if not 0.0 <= args.remove_empty_field <= 1.0:
        raise ValueError("--remove_empty_field must be in the range 0..1")
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be >= 2")
    if args.min_cluster_size < 1:
        raise ValueError("--min_cluster_size must be >= 1")
    if not 0.0 <= args.max_entropy <= 1.0:
        raise ValueError("--max_entropy must be in the range 0..1")

    project_input = Path(args.input_folder) / args.project_folder
    if not project_input.is_dir():
        raise FileNotFoundError(f"Project input folder does not exist: {project_input}")

    matrix_path = project_input / args.matrix_file
    feature_path = project_input / args.feature_file
    allele_path = project_input / args.allele_file

    for label, path in (
        ("matrix", matrix_path),
        ("feature", feature_path),
        ("allele", allele_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Input {label} file does not exist: {path}")

    initial_training_matrix = _read_delimited_file(matrix_path)
    matrix = NAS.read_hierarchical_matrix(
        file_path=str(matrix_path),
        delimiter=args.delimiter,
        empty_threshold=args.remove_empty_field,
        empty_symbol=args.empty_symbol,
    )
    reference_name, feature_matrix, feature_records = read_feature_file(feature_path)
    allele_matrix, allele_map = read_allele_file(allele_path)

    validate_metadata(matrix, feature_records, allele_map)

    print(
        f"Training matrix: {matrix.X.shape[0]} genomes x {matrix.X.shape[1]} features; "
        f"hierarchy levels={matrix.n_levels}; data type={matrix.data_type}"
    )
    print(
        f"Training mode: {args.algorithm}; min_cluster_size={args.min_cluster_size}; "
        f"max_entropy={args.max_entropy:.2f}"
    )

    tree = _build_tree(
        matrix=matrix,
        path=tuple(),
        algorithm_mode=args.algorithm,
        args=args,
    )

    classifier_root = Path(args.output_folder) / f"{args.project_folder}_classifier"
    classifier_input = classifier_root / "input"
    classifier_output = classifier_root / "output"
    classifier_model = classifier_root / "model"
    for folder in (classifier_input, classifier_output, classifier_model):
        folder.mkdir(parents=True, exist_ok=True)

    bundle = ModelBundle({
        "bundle_version": BUNDLE_VERSION,
        "project": args.project_folder,
        "author": args.author,
        "description": args.description,
        "version": args.version,
        "contact_details": args.contact_details,
        "algorithm_mode": args.algorithm,
        "delimiter": args.delimiter,
        "data_type": matrix.data_type,
        "reference_name": reference_name,
        "feature_titles": list(matrix.feature_titles),
        "feature_matrix": feature_matrix,
        "feature_records": _parse_feature_specs(feature_records, matrix.feature_titles),
        "allele_matrix": allele_matrix,
        "common_alleles": allele_map,
        "missing_token": "n",
        "hierarchy_levels": matrix.n_levels,
        "min_cluster_size": args.min_cluster_size,
        "max_entropy": args.max_entropy,
        "tree": tree,
        "label_paths": [list(labels) for labels in matrix.labels],
        "training_matrix": initial_training_matrix,
        "training": {
            "matrix_file": args.matrix_file,
            "feature_file": args.feature_file,
            "allele_file": args.allele_file,
            "genome_names": list(matrix.genome_names),
            "removed_feature_columns": list(matrix.removed_columns),
        },
    })

    model_path = classifier_model / "model.pkl"
    bundle.save(model_path)

    template_classifier = SCRIPT_DIR / "template" / "classifier.py"
    if not template_classifier.is_file():
        raise FileNotFoundError(f"Classifier template does not exist: {template_classifier}")
    shutil.copy2(template_classifier, classifier_root / "classifier.py")

    # The deployed classifier imports its helper modules from ./lib.
    classifier_lib = classifier_root / "lib"
    classifier_lib.mkdir(parents=True, exist_ok=True)

    for module_name in ("model_bundle.py", "blast.py"):
        module_source = LIB_DIR / module_name
        if not module_source.is_file():
            raise FileNotFoundError(
                f"Required classifier module does not exist: {module_source}"
            )
        shutil.copy2(module_source, classifier_lib / module_name)

    # Copy bundled BLAST executables, when present.  On Linux the classifier
    # can alternatively use blastn/makeblastdb from PATH.
    source_bin = LIB_DIR / "bin"
    target_bin = classifier_lib / "bin"
    if source_bin.is_dir():
        shutil.copytree(source_bin, target_bin, dirs_exist_ok=True)

    print(f"Trained NN/ML node models: {_count_models(tree)}")
    print(f"Nodes with stored allele profiles: {_count_profile_nodes(tree)}")
    print(f"Model bundle saved to: {model_path}")
    print(f"Classifier package created in: {classifier_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
