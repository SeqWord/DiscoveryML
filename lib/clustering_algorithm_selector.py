#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# -------------------- I/O --------------------

def _strip(s: str) -> str:
    return str(s).strip().strip('"').strip("'")


def _read_labeled_matrix_raw(
    path: str,
    delimiter: str,
    label_delimiter: str = "|",
    check_record_uniquness: bool = True,
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Read a matrix whose first column contains hierarchical row metadata.

    The first-column format is::

        level_1<label_delimiter>level_2<label_delimiter>...<label_delimiter>row_id

    The final element is treated as the unique row identifier (strain/genome),
    while all preceding elements are hierarchical label levels.

    Rows may contain different numbers of hierarchy levels.  In that case,
    levels are left-aligned and missing deeper levels are padded with empty
    strings so that ``labels`` is always a rectangular list of lists.
    """
    if not label_delimiter:
        raise ValueError("label_delimiter must not be empty.")

    rows: List[List[str]] = []

    with open(path, newline="", encoding="utf-8") as fp:
        rdr = csv.reader(fp, delimiter=delimiter)

        for r in rdr:
            if not r:
                continue

            r = [_strip(c) for c in r]

            if not any(r):
                continue

            rows.append(r)

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(
            "Input must have a header row and at least one data row with labels."
        )

    # Column names of the numeric/data matrix. Do not silently remove empty
    # headers because that would shift feature columns relative to the body.
    col_labels = rows[0][1:]
    if any(c == "" for c in col_labels):
        raise ValueError("Feature column names in the header must not be empty.")

    expected_columns = 1 + len(col_labels)
    for line_no, r in enumerate(rows[1:], start=2):
        if len(r) != expected_columns:
            raise ValueError(
                f"Row {line_no} has {len(r)} columns; expected {expected_columns}."
            )

    # Split the compound row heading into hierarchy levels + unique row ID.
    split_headings: List[List[str]] = [
        [_strip(part) for part in r[0].split(label_delimiter)]
        for r in rows[1:]
    ]

    for line_no, parts in enumerate(split_headings, start=2):
        if len(parts) < 2:
            raise ValueError(
                f"Row {line_no} heading must contain at least one hierarchical "
                f"label and a row identifier separated by {label_delimiter!r}."
            )
        if not parts[-1]:
            raise ValueError(f"Row {line_no} has an empty row identifier.")

    # Last element is the unique strain/genome identifier.
    row_labels: List[str] = [parts[-1] for parts in split_headings]
    if check_record_uniquness and len(set(row_labels)) != len(row_labels):
        seen: set[str] = set()
        duplicates: List[str] = []
        for row_id in row_labels:
            if row_id in seen and row_id not in duplicates:
                duplicates.append(row_id)
            seen.add(row_id)
        raise ValueError(
            "Row identifiers must be unique. Duplicate identifier(s): "
            + ", ".join(duplicates)
        )

    # Everything before the identifier represents hierarchical label levels.
    raw_labels: List[List[str]] = [parts[:-1] for parts in split_headings]
    n_label_levels = max((len(levels) for levels in raw_labels), default=0)

    # Convert the original compound heading into a rectangular set of label
    # columns. Missing deeper levels are represented by an empty string.
    labels: List[List[str]] = [
        levels + [""] * (n_label_levels - len(levels))
        for levels in raw_labels
    ]

    # Matrix values start immediately after the original compound heading.
    body: List[List[str]] = [r[1:] for r in rows[1:]]

    return col_labels, row_labels, labels, body
    
def _try_parse_float(x: str) -> float:
    s = _strip(x)
    if s in {"", "NA", "N/A", "NaN", "nan", "-", ".", "*"}:
        return float("nan")
    try:
        return float(s)
    except Exception:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return float("nan")


# -------------------- Known-label cluster validation --------------------

def _cells_to_numeric_matrix(
    cells: List[List[str]],
    units: str,
) -> Optional[np.ndarray]:
    """Convert matrix cells to a finite numeric NumPy array when possible."""
    binary_map = {
        "0": 0.0, "-": 0.0, "false": 0.0, "no": 0.0, "absent": 0.0,
        "1": 1.0, "+": 1.0, "true": 1.0, "yes": 1.0, "present": 1.0,
    }

    values: List[List[float]] = []
    for row in cells:
        numeric_row: List[float] = []
        for cell in row:
            token = _strip(cell)
            if units == "binary" and token.lower() in binary_map:
                value = binary_map[token.lower()]
            else:
                value = _try_parse_float(token)
            if not math.isfinite(value):
                return None
            numeric_row.append(value)
        values.append(numeric_row)

    if not values:
        return None
    return np.asarray(values, dtype=float)


def _evaluate_hierarchical_labels(
    cells: List[List[str]],
    labels: List[List[str]],
    units: str,
    square_distance: bool,
    label_delimiter: str,
) -> List[Dict[str, Any]]:
    """
    Evaluate how well known hierarchical labels agree with matrix structure.

    Each hierarchy level is evaluated independently.  At level N the cluster
    identity is the complete path from Level 1 through Level N, preventing
    equal child names under different parents from being merged.

    Rows missing the current level are excluded from that level's evaluation.
    """
    results: List[Dict[str, Any]] = []
    n_levels = len(labels[0]) if labels else 0

    if units == "categorical":
        for level_idx in range(n_levels):
            results.append({
                "level": level_idx + 1,
                "status": "not calculated: categorical feature values require an explicit numeric/distance encoding",
            })
        return results

    X = _cells_to_numeric_matrix(cells, units)
    if X is None:
        for level_idx in range(n_levels):
            results.append({
                "level": level_idx + 1,
                "status": "not calculated: matrix contains missing or non-numeric values",
            })
        return results

    if square_distance:
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            for level_idx in range(n_levels):
                results.append({
                    "level": level_idx + 1,
                    "status": "not calculated: inferred distance matrix is not square",
                })
            return results

        # silhouette_score(metric='precomputed') expects a zero diagonal.
        if not np.allclose(np.diag(X), 0.0, atol=1e-9):
            for level_idx in range(n_levels):
                results.append({
                    "level": level_idx + 1,
                    "status": "not calculated: precomputed distance matrix must have a zero diagonal",
                })
            return results

    for level_idx in range(n_levels):
        # A row participates only if it has a value at this hierarchy level.
        row_indices = [
            i for i, row_levels in enumerate(labels)
            if _strip(row_levels[level_idx]) != ""
        ]

        if not row_indices:
            results.append({
                "level": level_idx + 1,
                "status": "not calculated: no rows contain this label level",
            })
            continue

        # Use the complete hierarchy path through the current level.
        cluster_names = [
            label_delimiter.join(labels[i][:level_idx + 1])
            for i in row_indices
        ]
        n_used = len(row_indices)
        n_clusters = len(set(cluster_names))

        base: Dict[str, Any] = {
            "level": level_idx + 1,
            "n_samples": n_used,
            "n_clusters": n_clusters,
        }

        # Internal cluster validation requires at least two clusters, and fewer
        # clusters than samples (otherwise every sample is effectively unique).
        if n_clusters < 2:
            base["status"] = "not calculated: fewer than 2 groups"
            results.append(base)
            continue
        if n_clusters >= n_used:
            base["status"] = "not calculated: number of groups must be smaller than number of samples"
            results.append(base)
            continue

        try:
            if square_distance:
                D = X[np.ix_(row_indices, row_indices)]
                base["silhouette"] = float(
                    silhouette_score(D, cluster_names, metric="precomputed")
                )
                base["calinski_harabasz"] = None
                base["davies_bouldin"] = None
                base["status"] = (
                    "Silhouette calculated from precomputed distances; "
                    "Calinski-Harabasz and Davies-Bouldin require feature vectors"
                )
            else:
                X_level = X[row_indices, :]
                base["silhouette"] = float(
                    silhouette_score(X_level, cluster_names, metric="euclidean")
                )
                base["calinski_harabasz"] = float(
                    calinski_harabasz_score(X_level, cluster_names)
                )
                base["davies_bouldin"] = float(
                    davies_bouldin_score(X_level, cluster_names)
                )
                base["status"] = "ok"
        except ValueError as exc:
            base["status"] = f"not calculated: {exc}"

        results.append(base)

    return results


# -------------------- Heuristics --------------------

BINARY_TOKENS = {"0", "1", "+", "-", "true", "false", "yes", "no", "present", "absent"}


def _infer_units_from_cells(cells: List[List[str]]) -> str:
    flat = [c for row in cells for c in row if _strip(c) != ""]
    if not flat:
        return "continuous"

    numeric_count = 0
    non_numeric_count = 0
    bin_like_count = 0

    sample = flat[: min(len(flat), 20000)]
    for s in sample:
        ss = _strip(s).lower()
        if ss in BINARY_TOKENS:
            bin_like_count += 1
        v = _try_parse_float(ss)
        if not math.isnan(v):
            numeric_count += 1
        else:
            non_numeric_count += 1

    if non_numeric_count > 0 and non_numeric_count > numeric_count * 0.2:
        return "categorical"

    if bin_like_count > 0.7 * len(sample):
        return "binary"

    vals = []
    for s in sample:
        v = _try_parse_float(s)
        if not math.isnan(v):
            vals.append(v)
    if vals:
        if set(vals).issubset({0.0, 1.0}):
            return "binary"

    return "continuous"


def _infer_size_category(n_samples: int) -> str:
    if n_samples <= 200:
        return "small"
    if n_samples <= 5000:
        return "average"
    return "large"


def _infer_square_distance(n: int, m: int, units: str, cells: List[List[str]]) -> bool:
    if n != m:
        return False
    if units == "categorical":
        return False

    diag = []
    for i in range(min(n, 200)):
        v = _try_parse_float(cells[i][i])
        if not math.isnan(v):
            diag.append(v)

    if diag and np.nanmedian(np.abs(np.asarray(diag))) < 1e-9:
        nonneg = 0
        checked = 0
        for r in cells[: min(n, 50)]:
            for s in r[: min(m, 50)]:
                vv = _try_parse_float(s)
                if not math.isnan(vv):
                    checked += 1
                    if vv >= 0:
                        nonneg += 1
        if checked > 0 and (nonneg / checked) > 0.95:
            return True

    return False


# -------------------- Recommendation model --------------------

@dataclass
class Context:
    units: str
    size: str
    task: str
    kmer_length: str
    kmer_encoding: str
    expression_encoding: str
    abundance_encoding: str
    square_distance: bool
    n_samples: Optional[int] = None
    n_features: Optional[int] = None


@dataclass
class Recommendation:
    algorithm: str
    why: List[str]
    params: Dict[str, Any]
    example_cmd: str
    alternates: List[Tuple[str, str]]


def _suggest_graph_knn_k(n: int) -> int:
    if n <= 200:
        return min(30, max(10, int(round(math.sqrt(n)))))
    if n <= 5000:
        return min(50, max(15, int(round(math.sqrt(n)))))
    return min(100, max(30, int(round(math.sqrt(n)))))


def _cmd(algo: str, infile: str = "input.tsv", extra: str = "") -> str:
    extra = extra.strip()
    return f"py run.py {infile} --algorithm {algo}" + (f" {extra}" if extra else "")


def _out(fmt: str, out: str) -> str:
    return f"--output_format {fmt}"
    #return f"--output_format {fmt} -o {out}"


def _make_example_output_name(input_name: str) -> str:
    base = os.path.basename(input_name)
    stem, ext = os.path.splitext(base)
    if ext.lower() not in {".csv", ".tsv"}:
        return f"{stem}_clustered{ext}"
    return f"{stem}_clustered{ext}"


def _normalize_example_cmd_for_input_positional(example_cmd: str,
                                                project: str,
                                                input_file: str,
                                                algorithm_name: str) -> str:
    out_name = _make_example_output_name(input_file)
    inp_base = os.path.basename(input_file)

    cmd = example_cmd.strip()
    for old in (
        "py run_hdbscan.py", "py run_dbscan.py", "py run_kmeans.py",
        "py run_hclust.py", "py run_leiden.py", "py run_spectral.py",
        "python run_hdbscan.py", "python run_dbscan.py", "python run_kmeans.py",
        "python run_hclust.py", "python run_leiden.py", "python run_spectral.py",
    ):
        cmd = cmd.replace(old, "py run.py")

    parts = cmd.split()

    if "\n" in cmd:
        for line in cmd.splitlines():
            line = line.strip()
            if line.startswith("py run.py") or line.startswith("python run.py"):
                parts = line.replace("python", "py", 1).split()
                break

    base = ["py", "run.py", project, inp_base, algorithm_name]

    kept: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]

        if i < 2 and tok in {"py", "python"}:
            i += 1
            continue
        if tok == "run.py":
            i += 1
            continue

        if tok == "--algorithm":
            i += 2
            continue

        # skip the old positional input file if it looks like *.csv/*.tsv and is at the expected spot
        if tok.lower().endswith((".csv", ".tsv")) and len(parts) > 2 and tok == parts[2]:
            i += 1
            continue

        if tok == algorithm_name:
            i += 1
            continue

        kept.append(tok)
        i += 1

    out_parts = base + kept

    if "--output_format" in out_parts:
        j = out_parts.index("--output_format")
        if j + 1 < len(out_parts):
            out_parts[j + 1] = "all"
    elif "-f" in out_parts:
        j = out_parts.index("-f")
        if j + 1 < len(out_parts):
            out_parts[j + 1] = "all"
    else:
        out_parts += ["--output_format", "all"]
    '''
    if "-o" in out_parts:
        j = out_parts.index("-o")
        if j + 1 < len(out_parts):
            out_parts[j + 1] = out_name
    elif "--out" in out_parts:
        j = out_parts.index("--out")
        if j + 1 < len(out_parts):
            out_parts[j + 1] = out_name
    else:
        out_parts += ["-o", out_name]
    '''
    cleaned: list[str] = []
    seen_of = 0
    seen_o = 0
    i = 0
    while i < len(out_parts):
        t = out_parts[i]
        if t in {"--output_format", "-f"}:
            seen_of += 1
            if seen_of > 1:
                i += 2
                continue
        if t in {"-o", "--out"}:
            seen_o += 1
            if seen_o > 1:
                i += 2
                continue
        cleaned.append(t)
        i += 1

    return " ".join(cleaned)


def _format_alternative_label(algorithm: str, description: str) -> str:
    """Format an alternative algorithm label for the report."""
    description = (description or "").strip()
    return f"{algorithm} ({description})" if description else algorithm


# -------------------- Forced algorithm option tuning --------------------

_ALLOWED_ALGOS = {"SA", "DBSCAN", "HDBSCAN", "LEIDEN", "MCL", "AGGLO", "KMEANS"}

def _normalize_algo_name(s: str) -> str:
    ss = (s or "").strip()
    if not ss:
        return ""
    ss = ss.upper()
    # accept common aliases
    alias = {
        "SPECTRAL": "SA",
        "SPECTRALANALYSIS": "SA",
        "SPECTRAL_CLUSTERING": "SA",
        "HC": "AGGLO",
        "HCLUST": "AGGLO",
        "HIERARCHICAL": "AGGLO",
        "K-MEANS": "KMEANS",
        "KMEAN": "KMEANS",
    }
    ss = alias.get(ss, ss)
    return ss


def _choose_metric(ctx: Context, for_graph: bool = False) -> str:
    # Graph metric set differs slightly from feature-metric
    if ctx.units == "binary":
        return "jaccard" if not for_graph else "jaccard"
    if ctx.units == "categorical":
        return "hamming"  # "matching/hamming"
    # continuous
    if ctx.task == "metagenome_abundance":
        # many people like braycurtis for abundances, but graph defaults allow euclidean
        return "braycurtis" if not for_graph else "euclidean"
    return "euclidean"


def _suggest_min_samples(n: int) -> int:
    return max(5, int(round(0.01 * n)))


def _suggest_min_cluster_size(n: int) -> int:
    return max(5, int(round(0.02 * n)))


def _recommend_fixed(ctx: Context, forced_algo: str) -> Recommendation:
    """
    Provide the best parameter set for a user-forced algorithm (no algorithm selection).
    """
    n = ctx.n_samples or 500
    algo = forced_algo

    # --- SA ---
    if algo == "SA":
        metric = _choose_metric(ctx, for_graph=False)
        similarity = "gaussian"
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "Spectral methods are most sensitive to the similarity construction; recommended settings focus on a sensible metric and stable affinity.",
        ]
        params: Dict[str, Any] = {
            "metric": metric,
            "similarity": similarity,
            "sigma": "auto (or set --sigma if distances have a known scale)",
            "epsilon": "only if similarity=epsilon",
            "expected_k": "optional; set if you know k",
            "max_cluster_number": 5 if ctx.size != "large" else 10,
            "max_cluster_content": 5 if ctx.size != "large" else 10,
            "bootstrap": 0,
            "boot_mode": "replace",
            "boot_frac": 0.1,
        }
        extra = f"--metric {metric} --similarity {similarity} --max_cluster_number {params['max_cluster_number']} --max_cluster_content {params['max_cluster_content']} {_out('text','result.txt')}"
        if ctx.square_distance:
            why.append("Note: a square distance-like matrix was inferred/forced; SA typically expects similarities—use an appropriate metric/similarity (or transform distances to similarity).")
        return Recommendation(algo, why, params, _cmd("SA", "input.tsv", extra), alternates=[])

    # --- DBSCAN ---
    if algo == "DBSCAN":
        metric = _choose_metric(ctx, for_graph=False)
        ms = _suggest_min_samples(n)
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "DBSCAN requires a distance scale (eps). The defaults below are safe starting points; refine eps if clusters merge/split unexpectedly.",
        ]
        params = {
            "dbscan_metric": metric,
            "min_samples": ms,
            "eps": 0.5 if metric == "euclidean" else 0.2,
        }
        if ctx.square_distance:
            why.append("Warning: your matrix looks like a square distance matrix; DBSCAN typically needs either feature vectors or explicit support for precomputed distances.")
        extra = f"--dbscan_metric {metric} --min_samples {ms} --eps {params['eps']} {_out('text','result.txt')}"
        return Recommendation(algo, why, params, _cmd("DBSCAN", "input.tsv", extra), alternates=[("HDBSCAN", "more robust to variable density")])

    # --- HDBSCAN ---
    if algo == "HDBSCAN":
        metric = _choose_metric(ctx, for_graph=False)
        mcs = _suggest_min_cluster_size(n)
        ms = _suggest_min_samples(n)
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "HDBSCAN is robust for omics data because it handles variable-density clusters and can label uncertain items as noise.",
        ]
        params = {
            "hdbscan_metric": metric,
            "min_cluster_size": mcs,
            "min_samples": ms,
            "cluster_selection_epsilon": 0.0,
            "allow_single_cluster": False,
        }
        if ctx.square_distance:
            why.append("Note: your matrix looks like distances; if your runner supports it, using precomputed distances is typically best (otherwise consider LEIDEN --from_distance or AGGLO).")
        extra = (
            f"--hdbscan_metric {metric} --min_cluster_size {mcs} --min_samples {ms} "
            f"--cluster_selection_epsilon 0.0 {_out('text','result.txt')}"
        )
        return Recommendation(algo, why, params, _cmd("HDBSCAN", "input.tsv", extra), alternates=[("LEIDEN", "graph clustering"), ("AGGLO", "tree output if distances")])

    # --- LEIDEN ---
    if algo == "LEIDEN":
        k = _suggest_graph_knn_k(n)
        gmetric = _choose_metric(ctx, for_graph=True)
        from_distance = bool(ctx.square_distance)
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "Leiden is a scalable community-detection method; kNN graph construction choices (k, metric, from_distance) dominate performance.",
        ]
        params = {
            "knn_k": k,
            "graph_metric": gmetric,
            "resolution": 1.0,
            "from_distance": from_distance,
        }
        extra = f"--knn_k {k} --graph_metric {gmetric} --resolution 1.0" + (" --from_distance" if from_distance else "") + f" {_out('text','result.txt')}"
        return Recommendation(algo, why, params, _cmd("LEIDEN", "input.tsv", extra), alternates=[("MCL", "same graph")])

    # --- MCL ---
    if algo == "MCL":
        k = _suggest_graph_knn_k(n)
        gmetric = _choose_metric(ctx, for_graph=True)
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "MCL is driven mainly by graph construction and inflation; inflation controls cluster granularity (higher → more/smaller clusters).",
        ]
        params = {
            "knn_k": k,
            "graph_metric": gmetric,
            "inflation": 2.0,
            "expansion": 2,
            "prune_threshold": 1e-6,
            "mcl_max_iter": 100,
        }
        if ctx.square_distance:
            why.append("Note: your matrix looks like distances; if your runner builds a kNN graph from distances, keep k moderate and consider comparing with LEIDEN --from_distance.")
        extra = (
            f"--knn_k {k} --graph_metric {gmetric} --inflation 2.0 --expansion 2 "
            f"--prune_threshold 1e-6 --mcl_max_iter 100 {_out('text','result.txt')}"
        )
        return Recommendation(algo, why, params, _cmd("MCL", "input.tsv", extra), alternates=[("LEIDEN", "often simpler to tune")])

    # --- AGGLO ---
    if algo == "AGGLO":
        metric = _choose_metric(ctx, for_graph=False)
        # Ward only really makes sense for Euclidean feature space; average is more general (and common for distances)
        linkage = "ward" if (metric == "euclidean" and not ctx.square_distance and ctx.units == "continuous") else "average"
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "Agglomerative clustering is interpretable and naturally supports tree/dendrogram output; linkage/metric must match the data geometry.",
        ]
        params = {"agg_metric": metric, "agg_linkage": linkage}
        # if user wants tree, keep newick in example; otherwise still OK
        extra = f"--agg_metric {metric} --agg_linkage {linkage} {_out('newick','tree.nwk')}"
        return Recommendation(algo, why, params, _cmd("AGGLO", "input.tsv", extra), alternates=[("SA", "partitions, not a tree"), ("LEIDEN", "scalable for large n")])

    # --- KMEANS ---
    if algo == "KMEANS":
        # k heuristic: keep it textual rather than guessing a single number
        why = [
            "Algorithm was forced by --algorithm; parameters are tuned to your inferred matrix properties.",
            "KMEANS assumes roughly spherical clusters in a Euclidean space; it is best used with scaled/normalized continuous features (or embeddings).",
        ]
        params = {
            "kmeans_k": "choose via silhouette/stability (e.g., try 5..50 small/avg; 20..200 large)",
            "n_init": 10 if ctx.size == "large" else 50,
            "max_iter": 300 if ctx.size == "large" else 500,
        }
        extra = f"--kmeans_k <K> --n_init {params['n_init']} --max_iter {params['max_iter']} {_out('text','result.txt')}"
        if ctx.units != "continuous":
            why.append("Warning: your units are non-continuous; consider embedding/encoding to continuous space, or prefer graph/density methods.")
        return Recommendation(algo, why, params, _cmd("KMEANS", "input.tsv", extra), alternates=[("LEIDEN", "better structure"), ("HDBSCAN", "noise-aware")])

    # fallback (should not happen if validated)
    return Recommendation(algo, ["Unknown forced algorithm."], {}, _cmd(algo, "input.tsv", _out('text','result.txt')), [])


# -------------------- Original selection logic --------------------

def _recommend(ctx: Context, forced_algo: str = "") -> Recommendation:
    """
    If forced_algo is given, return best options for that algorithm (no selection).
    Otherwise, run the original algorithm selection logic.
    """
    if forced_algo:
        return _recommend_fixed(ctx, forced_algo)

    # ----- ORIGINAL LOGIC (UNCHANGED BELOW, EXCEPT FOR CALL SIGNATURE) -----
    n = ctx.n_samples or 500

    if ctx.units == "categorical":
        k = _suggest_graph_knn_k(n)
        if ctx.size in {"small", "average"}:
            return Recommendation(
                algorithm="LEIDEN",
                why=[
                    "Data are categorical/symbolic, so Euclidean geometry and centroids are not meaningful.",
                    "Graph community detection (Leiden) on a similarity graph works well for discrete encodings.",
                ],
                params={"knn_k": k, "graph_metric": "hamming/matching", "resolution": 1.0},
                example_cmd=_cmd("LEIDEN", "input.tsv", f"--knn_k {k} --graph_metric hamming --resolution 1.0 {_out('text','result.txt')}"),
                alternates=[("MCL", "same graph"), ("AGGLO", "matching/Hamming distance; small n, tree output")],
            )
        return Recommendation(
            algorithm="LEIDEN",
            why=[
                "Large categorical matrices are better handled via a similarity graph than by centroid methods.",
                "Leiden on a kNN graph scales well and is robust for discrete/symbolic encodings.",
            ],
            params={"knn_k": k, "graph_metric": "hamming/matching", "resolution": 1.0},
            example_cmd=_cmd("LEIDEN", "input.tsv", f"--knn_k {k} --graph_metric hamming --resolution 1.0 {_out('text','result.txt')}"),
            alternates=[("MCL", "Markov clustering on the same similarity graph")],
        )

    if ctx.units == "binary":
        if ctx.square_distance and ctx.size != "large":
            return Recommendation(
                algorithm="AGGLO",
                why=[
                    "Binary feature profiles are naturally compared using Jaccard distance (ignores joint zeros).",
                    "Agglomerative (average/UPGMA) is interpretable and supports Newick tree output.",
                ],
                params={"agg_metric": "jaccard", "agg_linkage": "average (UPGMA)"},
                example_cmd=_cmd("AGGLO", "input.tsv", f"--agg_metric jaccard --agg_linkage average {_out('newick','tree.nwk')}"),
                alternates=[("SA", "if you want fixed-k partitions"), ("LEIDEN", "kNN graph"), ("HDBSCAN", "noise-aware")],
            )

        if ctx.size == "large":
            k = _suggest_graph_knn_k(n)
            return Recommendation(
                algorithm="LEIDEN",
                why=[
                    "For large binary matrices, graph clustering scales better than O(n²) hierarchical methods.",
                    "Jaccard-based similarity reflects shared features and is robust for sparse presence/absence data.",
                ],
                params={"knn_k": k, "graph_metric": "jaccard", "resolution": 1.0},
                example_cmd=_cmd("LEIDEN", "input.tsv", f"--knn_k {k} --graph_metric jaccard --resolution 1.0 {_out('text','result.txt')}"),
                alternates=[("HDBSCAN", "on Jaccard distances; noise-aware"), ("MCL", "on the same kNN graph")],
            )

        mcs = max(10, int(round(0.02 * n)))
        ms = max(5, int(round(0.01 * n)))
        return Recommendation(
            algorithm="HDBSCAN",
            why=[
                "Binary genomics/metagenomics matrices often contain noise and missing calls.",
                "HDBSCAN handles variable-density clusters and labels ambiguous items as noise without choosing k.",
            ],
            params={"hdbscan_metric": "jaccard (or precomputed)", "min_cluster_size": mcs, "min_samples": ms},
            example_cmd=_cmd("HDBSCAN", "input.tsv", f"--hdbscan_metric jaccard --min_cluster_size {mcs} --min_samples {ms} {_out('text','result.txt')}"),
            alternates=[("DBSCAN", "if eps can be tuned"), ("AGGLO", "tree on Jaccard"), ("SA", "fixed-k partitions")],
        )

    if ctx.square_distance:
        if ctx.size == "large":
            k = _suggest_graph_knn_k(n)
            return Recommendation(
                algorithm="LEIDEN",
                why=[
                    "A square distance matrix is supplied; for large n, hierarchical clustering is often too costly.",
                    "Leiden on a kNN graph built from distances scales well and performs strongly in omics datasets.",
                ],
                params={"knn_k": k, "distance_to_similarity": "rank/exp(-d/σ)", "resolution": 1.0},
                example_cmd=_cmd("LEIDEN", "input.tsv", f"--knn_k {k} --from_distance --resolution 1.0 {_out('text','result.txt')}"),
                alternates=[("HDBSCAN", "precomputed distances if many outliers/noise are expected"), ("SA", "similarity derived from distances")],
            )
        return Recommendation(
            algorithm="AGGLO",
            why=[
                "Square distance matrix supplied; agglomerative clustering is a direct and interpretable choice.",
                "It naturally supports dendrograms and Newick output.",
            ],
            params={"agg_linkage": "average (UPGMA)"},
            example_cmd=_cmd("AGGLO", "input.tsv", f"--agg_linkage average {_out('newick','tree.nwk')}"),
            alternates=[("SA", "if you want partitions, not a tree"), ("HDBSCAN", "precomputed distances if noise is expected")],
        )

    if ctx.task == "metagenome_abundance":
        if ctx.size == "large":
            k = _suggest_graph_knn_k(n)
            return Recommendation(
                algorithm="LEIDEN",
                why=[
                    "Taxon abundance is compositional; CLR/Aitchison preprocessing often improves geometry.",
                    "Leiden on a kNN graph scales well and is robust for high-dimensional abundance profiles.",
                ],
                params={"transform": "CLR with pseudocount", "knn_k": k, "graph_metric": "euclidean (CLR space)", "resolution": 1.0},
                example_cmd=_cmd("LEIDEN", "input.tsv", f"--transform clr --knn_k {k} --graph_metric euclidean --resolution 1.0 {_out('text','result.txt')}"),
                alternates=[("HDBSCAN", "in CLR space if outliers/noise are expected"), ("KMEANS", "baseline in CLR space")],
            )
        mcs = max(10, int(round(0.02 * n)))
        ms = max(5, int(round(0.01 * n)))
        return Recommendation(
            algorithm="HDBSCAN",
            why=[
                "Abundance profiles often form clusters with variable density due to sparsity and depth differences.",
                "HDBSCAN is robust and can label outlier samples as noise without choosing k.",
            ],
            params={"transform": "CLR with pseudocount", "hdbscan_metric": "euclidean", "min_cluster_size": mcs, "min_samples": ms},
            example_cmd=_cmd("HDBSCAN", "input.tsv", f"--transform clr --hdbscan_metric euclidean --min_cluster_size {mcs} --min_samples {ms} {_out('text','result.txt')}"),
            alternates=[("AGGLO", "Ward linkage in CLR space for tree-like interpretation"), ("LEIDEN", "kNN graph in CLR space")],
        )

    if ctx.task == "expression":
        if ctx.size == "large":
            k = _suggest_graph_knn_k(n)
            return Recommendation(
                algorithm="LEIDEN",
                why=[
                    "Expression matrices are high-dimensional; clustering on a reduced embedding is robust and scalable.",
                    "Leiden is widely used in transcriptomics because it captures complex structure beyond spherical clusters.",
                ],
                params={"preprocess": "log/VST + z-score; PCA(30–100)", "knn_k": k, "resolution": 1.0},
                example_cmd=_cmd("LEIDEN", "input.tsv", f"--pca 50 --knn_k {k} --resolution 1.0 {_out('text','result.txt')}"),
                alternates=[("HDBSCAN", "on the same embedding; noise-aware"), ("KMEANS", "on PCA as baseline")],
            )
        k = max(10, min(30, int(round(math.sqrt(n)))))
        return Recommendation(
            algorithm="SA",
            why=[
                "Expression-driven clusters can be non-convex; spectral analysis captures manifold structure.",
                "For small/average datasets, spectral methods are feasible and often strong.",
            ],
            params={"similarity": "gaussian", "expected_k": "choose via stability/silhouette", "n_neighbors (if knn affinity)": k},
            example_cmd=_cmd("SA", "input.tsv", f"--similarity gaussian --expected_k <K> {_out('text','result.txt')}"),
            alternates=[("KMEANS", "on PCA as baseline"), ("AGGLO", "Ward linkage on embedding for tree-like output"), ("HDBSCAN", "if noise is expected")],
        )

    if ctx.task == "kmer" and ctx.kmer_length == "long":
        mcs = max(10, int(round(0.02 * n)))
        ms = max(5, int(round(0.01 * n)))
        return Recommendation(
            algorithm="HDBSCAN",
            why=[
                "Long k-mers produce very high-dimensional spaces; embedding (SVD/PCA) reduces noise and distance concentration.",
                "HDBSCAN handles variable density and can label ambiguous items as noise.",
            ],
            params={"embedding": "SVD/PCA(50–100)", "hdbscan_metric": "euclidean (embedding)", "min_cluster_size": mcs, "min_samples": ms},
            example_cmd=_cmd("HDBSCAN", "input.tsv", f"--svd 100 --hdbscan_metric euclidean --min_cluster_size {mcs} --min_samples {ms} {_out('text','result.txt')}"),
            alternates=[("LEIDEN", "kNN graph of the same embedding"), ("SA", "if you want fixed-k partitions")],
        )

    if ctx.units == "continuous" and (not ctx.square_distance) and ctx.size in {"small", "average"}:
        return Recommendation(
            algorithm="DBSCAN",
            why=[
                "Continuous feature matrix (not distances): DBSCAN is a simple, robust baseline for discovering dense groups.",
                "It can label outliers as noise (-1) without choosing k. For variable density, HDBSCAN is a stronger alternative.",
            ],
            params={"eps": 0.5, "min_samples": max(5, int(round(0.01 * n))), "scale": True},
            example_cmd=_cmd("DBSCAN", "input.tsv", f"--eps 0.5 --min_samples {max(5, int(round(0.01*n)))} --scale {_out('text','result.txt')}"),
            alternates=[("HDBSCAN", "variable density"), ("LEIDEN", "kNN graph"), ("KMEANS", "baseline")],
        )

    if ctx.size == "large":
        return Recommendation(
            algorithm="KMEANS",
            why=[
                "For very large continuous datasets, KMEANS is a fast baseline and often reasonable.",
                "If you need better structure or noise-handling, refine with LEIDEN or HDBSCAN on an embedding.",
            ],
            params={"kmeans_k": "start with 20–200", "scale": True, "n_init": 10},
            example_cmd=_cmd("KMEANS", "input.tsv", f"--kmeans_k <K> --scale --n_init 10 {_out('text','result.txt')}"),
            alternates=[("LEIDEN", "often better structure"), ("HDBSCAN", "noise-aware on an embedding")],
        )

    return Recommendation(
        algorithm="KMEANS",
        why=[
            "In continuous space, KMEANS is a reliable baseline if clusters are roughly spherical and k is known.",
            "If you expect outliers or variable-density clusters, prefer HDBSCAN or DBSCAN.",
        ],
        params={
            "KMEANS": {"scale": True, "n_init": 50, "max_iter": 500, "kmeans_k": "choose via silhouette/stability"},
            "HDBSCAN_alt": {"min_cluster_size": "≈2% of n", "min_samples": "≈1% of n"},
            "DBSCAN_alt": {"eps": "tune via k-distance plot", "min_samples": "≈1% of n"},
        },
        example_cmd=(
            "KMEANS baseline:\n"
            "  " + _cmd("KMEANS", "input.tsv", f"--scale --kmeans_k <K> --n_init 50 --max_iter 500 {_out('text','result.txt')}") + "\n"
            "Noise-aware alternative (HDBSCAN):\n"
            "  " + _cmd("HDBSCAN", "input.tsv", f"--hdbscan_metric euclidean --min_cluster_size <MCS> --min_samples <MS> {_out('text','result.txt')}") + "\n"
            "Noise-aware alternative (DBSCAN):\n"
            "  " + _cmd("DBSCAN", "input.tsv", f"--eps <EPS> --min_samples <MS> --scale {_out('text','result.txt')}")
        ),
        alternates=[("SA", "if non-convex clusters are likely"), ("AGGLO", "tree-like output"), ("LEIDEN", "scalable graph")],
    )

def process(args, check_record_uniquness: bool = True) -> int:
    forced_algo = _normalize_algo_name(args.algorithm)
    if forced_algo and forced_algo not in _ALLOWED_ALGOS:
        print(
            f"ERROR: Unknown --algorithm '{args.algorithm}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_ALGOS))}",
            file=sys.stderr,
        )
        return 1

    input_folder = os.path.join(args.input_folder, args.project)
    if not os.path.isdir(input_folder):
        print(f"ERROR: project folder {input_folder} does not exist!", file=sys.stderr)
        return 2

    input_files: List[str] = []
    if args.input_file:
        in_path = os.path.join(input_folder, args.input_file)
        if not os.path.isfile(in_path):
            print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
            return 3
        input_files.append(in_path)
    else:
        for fname in sorted(os.listdir(input_folder)):
            if fname.lower().endswith((".tsv", ".csv")):
                input_files.append(os.path.join(input_folder, fname))

    if not input_files:
        print(f"ERROR: No TSV/CSV matrix files were found in {input_folder}!", file=sys.stderr)
        return 4

    output_path = os.path.join(args.output_folder, args.project)
    os.makedirs(output_path, exist_ok=True)

    for i, input_file in enumerate(input_files):
        process_file(input_file, output_path, args, forced_algo, check_record_uniquness=check_record_uniquness)
        if i < len(input_files) - 1:
            input("Press ENTER to continue with next file...")

    return 0


def process_file(
    input_file: str,
    output_path: str,
    args: argparse.Namespace,
    forced_algo: str = "",
    check_record_uniquness: bool = True
) -> int:
    in_base = os.path.basename(input_file)
    ext = os.path.splitext(in_base)[1].lower()

    if ext not in {".tsv", ".csv"}:
        raise ValueError(f"Unsupported matrix format: {input_file}")

    delimiter = "\t" if ext == ".tsv" else ","

    col_labels, row_labels, labels, cells = _read_labeled_matrix_raw(
        input_file,
        delimiter,
        label_delimiter=args.label_delimiter,
        check_record_uniquness=check_record_uniquness,
    )

    n_samples = len(row_labels)
    n_features = len(col_labels)

    # Keep user-specified settings unchanged between files.  Inferred values
    # are local to the current input matrix only.
    units = args.units
    size = args.size
    task = args.task

    if units == "auto":
        units = _infer_units_from_cells(cells)

        # Encoding hints may be more informative than the generic cell parser.
        if args.kmer_encoding == "qual5" or args.expression_encoding == "qual5":
            units = "categorical"
        elif args.abundance_encoding == "binary":
            units = "binary"

    if size == "auto":
        size = _infer_size_category(n_samples)

    inferred_square = _infer_square_distance(
        n_samples, n_features, units, cells
    )
    square_distance = bool(args.square_distance or inferred_square)

    ctx = Context(
        units=units,
        size=size,
        task=task,
        kmer_length=args.kmer_length,
        kmer_encoding=args.kmer_encoding,
        expression_encoding=args.expression_encoding,
        abundance_encoding=args.abundance_encoding,
        square_distance=square_distance,
        n_samples=n_samples,
        n_features=n_features,
    )

    rec = _recommend(ctx, forced_algo=forced_algo)

    label_metrics = _evaluate_hierarchical_labels(
        cells=cells,
        labels=labels,
        units=ctx.units,
        square_distance=ctx.square_distance,
        label_delimiter=args.label_delimiter,
    )

    report: List[str] = []

    def add(line: str = "") -> None:
        report.append(line)

    def add_params(d: Any, indent: int = 2) -> None:
        sp = " " * indent
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    add(f"{sp}{key}:")
                    add_params(value, indent + 2)
                else:
                    add(f"{sp}{key}: {value}")
        else:
            add(f"{sp}{d}")

    add("=== META-SELECTOR RECOMMENDATION ===")
    add(f"Input:            {input_file}")
    add(f"Matrix shape:     {n_samples} x {n_features}")
    n_label_levels = len(labels[0]) if labels else 0
    add(f"Label delimiter:  {args.label_delimiter!r}")
    add(f"Label levels:     {n_label_levels}")
    add(f"Task:             {ctx.task}")
    add(f"Units:            {ctx.units}")
    add(f"Dataset size:     {ctx.size}")
    add(f"Square distances: {ctx.square_distance}")

    if ctx.task == "kmer":
        add(f"k-mer length:     {ctx.kmer_length}")
        add(f"k-mer encoding:   {ctx.kmer_encoding}")
    if ctx.task == "expression":
        add(f"expr encoding:    {ctx.expression_encoding}")
    if ctx.task == "metagenome_abundance":
        add(f"abund encoding:   {ctx.abundance_encoding}")

    add()
    add("Known-label cluster validation:")
    add("  Silhouette:         higher is better (-1 to 1)")
    add("  Calinski-Harabasz:  higher is better")
    add("  Davies-Bouldin:     lower is better (0 is ideal)")
    for metric_result in label_metrics:
        level = metric_result["level"]
        add(f"  Level {level}:")
        if "n_samples" in metric_result:
            add(f"    samples used:       {metric_result['n_samples']}")
            add(f"    groups:             {metric_result['n_clusters']}")
        if metric_result.get("silhouette") is not None:
            add(f"    Silhouette:         {metric_result['silhouette']:.6f}")
        if metric_result.get("calinski_harabasz") is not None:
            add(f"    Calinski-Harabasz:  {metric_result['calinski_harabasz']:.6f}")
        if metric_result.get("davies_bouldin") is not None:
            add(f"    Davies-Bouldin:     {metric_result['davies_bouldin']:.6f}")
        if metric_result.get("status") != "ok":
            add(f"    note:               {metric_result.get('status', '')}")

    if forced_algo:
        add()
        add(f"Forced algorithm (--algorithm): {forced_algo}")

    add()
    add("Best algorithm:" if not forced_algo else "Recommended options for algorithm:")
    add(f"  {rec.algorithm}")

    add()
    add("Why selected:" if not forced_algo else "Why these options:")
    for why in rec.why:
        add(f"  - {why}")

    add()
    add("Recommended parameters:")
    add_params(rec.params, 2)

    add()
    add("Example run command:")
    example_cmd = _normalize_example_cmd_for_input_positional(
        example_cmd=rec.example_cmd,
        project=args.project,
        input_file=input_file,
        algorithm_name=rec.algorithm,
    )
    add(example_cmd)

    if rec.alternates:
        add()
        add("Good alternatives (if needed):")
        for alternative_algorithm, alternative_description in rec.alternates:
            add(f"  - {_format_alternative_label(alternative_algorithm, alternative_description)}")

            alt_rec = _recommend_fixed(ctx, alternative_algorithm)
            alt_cmd = _normalize_example_cmd_for_input_positional(
                example_cmd=alt_rec.example_cmd,
                project=args.project,
                input_file=input_file,
                algorithm_name=alternative_algorithm,
            )
            add(f"    {alt_cmd}")

    add("===================================")

    report_text = "\n".join(report) + "\n"

    # Print exactly the same report that will be saved.
    print("\n" + report_text)

    # Save under the input basename, replacing .csv/.tsv with .txt.
    output_file = os.path.join(
        output_path,
        os.path.splitext(in_base)[0] + "_report.txt",
    )
    with open(output_file, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(report_text)

    print(f"Report saved to: {output_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(print("This module provides functions of selection the most appropriate clustring algorithms. It should be used by other programs."))
