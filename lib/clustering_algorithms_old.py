# algorithms.py
# -*- coding: utf-8 -*-
"""
Collection of unsupervised clustering algorithm classes.

Implemented (class names match run.py --algorithm choices):
  SA       - Spectral Analysis (hierarchical spectral clustering; implementation included here)
  DBSCAN   - sklearn DBSCAN (feature or precomputed distance)
  HDBSCAN  - hdbscan.HDBSCAN (feature or precomputed distance)
  LEIDEN   - Leiden community detection on kNN graphs (igraph + leidenalg)
  MCL      - Markov Clustering on similarity graphs (pure numpy implementation)
  AGGLO    - Agglomerative hierarchical clustering (average/UPGMA, ward) + Newick via UPGMA
  KMEANS   - sklearn KMeans on features (optional scaling)

The Basic base class provides:
  - labeled CSV/TSV reading
  - median imputation
  - binary detection
  - distance matrix building
  - output formatting: matrix/text/newick
  - UPGMA newick builder (from distances)
"""

from __future__ import annotations
import os
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Basic helpers
# =============================================================================

def _strip(s: str) -> str:
    return str(s).strip().strip('"').strip("'")


def _swap_extension(out_path: str, new_ext: str) -> str:
    base, _ = os.path.splitext(out_path)
    return base + new_ext


def _ensure_output_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _all_binary(data: List[List[float]]) -> bool:
    for row in data:
        for v in row:
            if math.isnan(v):
                continue
            if v not in (0.0, 1.0):
                return False
    return True


def _impute_col_median_inplace(data: List[List[float]]) -> None:
    if not data or not data[0]:
        return
    n, m = len(data), len(data[0])
    for j in range(m):
        col = [data[i][j] for i in range(n) if not math.isnan(data[i][j])]
        if not col:
            med = 0.0
        else:
            col_sorted = sorted(col)
            med = col_sorted[len(col_sorted) // 2]
        for i in range(n):
            if math.isnan(data[i][j]):
                data[i][j] = med


def _jaccard_pairwise_distance_bool(Xb: np.ndarray) -> np.ndarray:
    """
    Jaccard distance between boolean rows: 1 - |A∩B|/|A∪B|
    Xb: shape (n, m) boolean
    """
    n = Xb.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        ai = Xb[i]
        for j in range(i + 1, n):
            aj = Xb[j]
            inter = np.logical_and(ai, aj).sum()
            union = np.logical_or(ai, aj).sum()
            d = 1.0 - (inter / union) if union > 0 else 1.0
            D[i, j] = D[j, i] = float(d)
    return D


def _euclidean_pairwise_distance(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(X[i] - X[j]))
            D[i, j] = D[j, i] = d
    return D


def _braycurtis_pairwise_distance(X: np.ndarray) -> np.ndarray:
    """
    Bray–Curtis distance: sum |xi-xj| / sum |xi+xj|
    """
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            num = np.sum(np.abs(xi - xj))
            den = np.sum(np.abs(xi + xj))
            d = float(num / den) if den > 0 else 0.0
            D[i, j] = D[j, i] = d
    return D


class _Node:
    __slots__ = ("name", "left", "right", "height")
    def __init__(self, name: Optional[str] = None, left: Optional["_Node"] = None,
                 right: Optional["_Node"] = None, height: float = 0.0):
        self.name = name
        self.left = left
        self.right = right
        self.height = float(height)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _upgma_newick(labels: List[str], dist: np.ndarray) -> str:
    """
    O(n^3) UPGMA producing Newick with branch lengths.
    dist: symmetric NxN, diagonal 0
    """
    n = len(labels)
    if dist.shape != (n, n):
        raise ValueError("Distance matrix shape does not match labels.")

    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    nodes: Dict[int, _Node] = {i: _Node(name=labels[i], height=0.0) for i in range(n)}

    def cdist(a: List[int], b: List[int]) -> float:
        s = 0.0
        cnt = 0
        for i in a:
            for j in b:
                s += float(dist[i, j])
                cnt += 1
        return s / cnt if cnt else 0.0

    next_id = n

    while len(clusters) > 1:
        ids = list(clusters.keys())
        best_i = -1
        best_j = -1
        best_d = float("inf")
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                ci = ids[ii]
                cj = ids[jj]
                d = cdist(clusters[ci], clusters[cj])
                if d < best_d:
                    best_d = d
                    best_i = ci
                    best_j = cj

        a_id, b_id, dmin = best_i, best_j, best_d
        a_members = clusters.pop(a_id)
        b_members = clusters.pop(b_id)
        clusters[next_id] = a_members + b_members

        parent_h = float(dmin) / 2.0
        parent = _Node(left=nodes[a_id], right=nodes[b_id], height=parent_h)
        nodes[next_id] = parent
        next_id += 1

    root_id = next(iter(clusters.keys()))
    root = nodes[root_id]

    def safe_name(s: str) -> str:
        if any(c in s for c in " \t\r\n,:;()[]"):
            return "'" + s.replace("'", "''") + "'"
        return s

    def rec(node: _Node, parent_height: float) -> str:
        if node.is_leaf():
            bl = max(0.0, parent_height - node.height)
            return f"{safe_name(node.name or '')}:{bl:.6f}"
        left = rec(node.left, node.height)    # type: ignore[arg-type]
        right = rec(node.right, node.height)  # type: ignore[arg-type]
        bl = max(0.0, parent_height - node.height)
        return f"({left},{right}):{bl:.6f}"

    newick = rec(root, root.height)
    if newick.endswith(":0.000000"):
        newick = newick[: -len(":0.000000")]
    return newick + ";"


# =============================================================================
# Spectral-analysis implementation (moved from spectra.py)
# =============================================================================

# --------------------------- Type aliases ---------------------------
Number = Union[int, float]
TriList = List[List[Number]]
MatrixLike = Union[TriList, List[List[Number]], np.ndarray]

# --------------------------- Distance expansion utilities ---------------------------

def _expand_lower_triangle(lower: TriList) -> np.ndarray:
    """
    lower[i] holds distances from taxon i+1 down to 0 (length i).
    Example for n=4:
        lower = [
            [d10],
            [d20, d21],
            [d30, d31, d32],
        ]
    """
    n = len(lower) + 1
    D = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        row = lower[i - 1]
        if len(row) != i:
            raise ValueError(f"Lower-tri row {i-1} length {len(row)} != expected {i}")
        for j in range(i):
            v = float(row[j])
            D[i, j] = v
            D[j, i] = v
    return D

def _expand_upper_triangle(upper: TriList) -> np.ndarray:
    """
    upper[i] holds distances from taxon i to (i+1 ... n-1).
    Example for n=4:
        upper = [
            [d01, d02, d03],
            [     d12, d13],
            [          d23],
        ]
    """
    n = len(upper) + 1
    D = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        row = upper[i]
        expected = n - i - 1
        if len(row) != expected:
            raise ValueError(f"Upper-tri row {i} length {len(row)} != expected {expected}")
        for k, val in enumerate(row, start=i + 1):
            v = float(val)
            D[i, k] = v
            D[k, i] = v
    return D

def _validate_full_square_symmetric(M: Sequence[Sequence[Number]], atol: float = 1e-12) -> np.ndarray:
    """Validate a full matrix is square and (near-)symmetric; return as float ndarray."""
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Full matrix must be square; got shape {arr.shape}")
    if not np.allclose(arr, arr.T, atol=atol):
        raise ValueError("Full matrix must be symmetric (within tolerance).")
    return arr

def _expand_distance_any(matrix: MatrixLike) -> np.ndarray:
    """
    Accepts lower-triangle, upper-triangle, or full matrix and returns a full symmetric ndarray.
    Heuristic:
        - len(first row) < len(last row)  -> lower triangle
        - len(first row) > len(last row)  -> upper triangle
        - len(first row) == len(last row) -> full matrix
    """
    if isinstance(matrix, np.ndarray):
        return _validate_full_square_symmetric(matrix)

    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix must be a non-empty list (or numpy array).")

    # 1x1 special-case
    if isinstance(matrix[0], (int, float)) or (isinstance(matrix[0], list) and len(matrix) == 1 and len(matrix[0]) == 1):
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        return _validate_full_square_symmetric(arr)

    first_len = len(matrix[0])
    last_len = len(matrix[-1])

    if first_len < last_len:
        return _expand_lower_triangle(matrix)  # lower-tri
    elif first_len > last_len:
        return _expand_upper_triangle(matrix)  # upper-tri
    else:
        return _validate_full_square_symmetric(matrix)  # full

# --------------------------- Similarity & Laplacian ---------------------------

def _median_nonzero_upper(a: np.ndarray) -> float:
    vals = a[np.triu_indices_from(a, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))

def _gaussian_similarity(dist: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    if sigma is None:
        sigma = _median_nonzero_upper(dist)
        if sigma <= 0:
            sigma = 1.0
    W = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)  # standard for spectral clustering
    return W

def _normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    eps = 1e-12
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = np.diag(inv_sqrt)
    return np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

def _eigengap_k(evals: np.ndarray, k_min: int = 2, k_max: Optional[int] = None) -> int:
    n = len(evals)
    if k_max is None:
        k_max = max(2, n // 2)
    k_max = min(k_max, n - 1)
    k_min = max(k_min, 2)
    vals = np.sort(evals)
    gaps = vals[1:k_max+1] - vals[0:k_max]
    idx = int(np.argmax(gaps)) + 1
    return max(k_min, min(k_max, idx))

# --------------------------- K-means (simple, deterministic seed) ---------------------------

def _kmeans_pp_init(X: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    n, d = X.shape
    centers = np.empty((k, d), dtype=float)
    i0 = rng.randrange(n)
    centers[0] = X[i0]
    dist2 = np.full(n, np.inf)
    for c in range(1, k):
        d2_new = np.sum((X - centers[c-1])**2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
        s = float(dist2.sum())
        if not np.isfinite(s) or s <= 0:
            centers[c] = X[rng.randrange(n)]
            continue
        probs = dist2 / s
        r = rng.random()
        cum = 0.0
        chosen = 0
        for i in range(n):
            cum += probs[i]
            if r <= cum:
                chosen = i
                break
        centers[c] = X[chosen]
    return centers

def _kmeans(X: np.ndarray, k: int, max_iter: int = 200, n_init: int = 5, seed: int = 13) -> np.ndarray:
    rng = random.Random(seed)
    best_labels = None
    best_inertia = math.inf
    for _ in range(n_init):
        centers = _kmeans_pp_init(X, k, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(max_iter):
            d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                centers[c] = X[mask].mean(axis=0) if np.any(mask) else X[rng.randrange(X.shape[0])]
        inertia = float(np.sum((X - centers[labels])**2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels

def _spectral_cluster_from_distance(dist: np.ndarray,
                                    force_k: Optional[int] = None) -> List[List[int]]:
    W = _gaussian_similarity(dist)
    Lsym = _normalized_laplacian(W)
    evals, evecs = np.linalg.eigh(Lsym)  # ascending
    if not force_k:
        k = _eigengap_k(evals, k_min=2, k_max=None)
        k = max(2, min(k, dist.shape[0]))
    else:
        k = max(2, min(force_k, dist.shape[0]))
    U = evecs[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y = U / norms
    labels = _kmeans(Y, k=k)
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    return list(clusters.values())

# --------- Average inter-cluster distance (computed from ORIGINAL distances) ---------

def _avg_distance_between_groups(group_a: Sequence[int],
                                 group_b: Sequence[int],
                                 full_dist: np.ndarray) -> float:
    a = np.array(group_a, dtype=int)
    b = np.array(group_b, dtype=int)
    d = full_dist[np.ix_(a, b)]
    return float(d.mean()) if d.size else 0.0

def _reduce_distance_matrix_by_average(clusters: List[List[int]],
                                       full_dist: np.ndarray) -> np.ndarray:
    m = len(clusters)
    R = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            Rij = _avg_distance_between_groups(clusters[i], clusters[j], full_dist)
            R[i, j] = R[j, i] = Rij
    return R

# --------- Split clusters that exceed max_cluster_content using submatrix spectral ---------

def _split_large_once(nested: List[List[object]],
                      idx_level: List[List[int]],
                      D_full: np.ndarray,
                      labels: List[str],
                      max_cluster_content: int,
                      force_k: Optional[int]) -> Tuple[bool, List[List[object]], List[List[int]]]:
    """
    Split any cluster whose LEAF COUNT exceeds max_cluster_content by reclustering
    its submatrix (from ORIGINAL distances). Returns (changed, new_nested, new_idx_level).

    Invariant:
      - idx_level: lists of GLOBAL indices (ints)
      - nested:    parallel lists of LABEL STRINGS (presentation only)
    """
    changed = False
    new_nested: List[List[object]] = []
    new_idx_level: List[List[int]] = []

    for node, leaves in zip(nested, idx_level):
        if len(leaves) <= max_cluster_content:
            new_nested.append(node)
            new_idx_level.append(leaves)
            continue

        # Re-cluster this big cluster using submatrix
        subD = D_full[np.ix_(leaves, leaves)]
        sub_groups = _spectral_cluster_from_distance(subD, force_k=force_k)

        # If no effective split happened, try forcing k=2 once
        if len(sub_groups) <= 1 and force_k is None and len(leaves) >= 2:
            sub_groups = _spectral_cluster_from_distance(subD, force_k=2)

        if len(sub_groups) <= 1:
            # cannot split; keep as is
            new_nested.append(node)
            new_idx_level.append(leaves)
            continue

        # Build children using GLOBAL indices -> labels
        child_indices: List[List[int]] = [[leaves[i] for i in g] for g in sub_groups]
        child_nodes:   List[List[str]] = [[labels[idx] for idx in grp] for grp in child_indices]

        # Insert children as separate top-level nodes (expansion)
        new_nested.extend(child_nodes)
        new_idx_level.extend(child_indices)
        changed = True

    return changed, new_nested, new_idx_level

# ----------------- Pathways & Newick formatting -----------------

def nested_list_to_pathways(cl: list, root: str = "Root") -> List[str]:
    """
    Convert a multilevel nested list (internal nodes as lists/tuples, leaves as labels)
    into a list of path strings like 'Root>0>1>Leaf'.
    """
    def _is_list_like(x: Any) -> bool:
        return isinstance(x, (list, tuple))

    # Special-case: top-level flat => no indices at root
    top_is_flat = _is_list_like(cl) and all(not _is_list_like(ch) for ch in cl)
    paths: List[str] = []

    def _walk(node: Any, prefix: str, at_root: bool) -> None:
        if _is_list_like(node):
            for idx, child in enumerate(node):
                if _is_list_like(child):
                    next_prefix = prefix if (at_root and top_is_flat) else f"{prefix}>{idx}"
                    _walk(child, next_prefix, at_root=False)
                else:
                    leaf = str(child)
                    if at_root and top_is_flat:
                        paths.append(f"{prefix}>{leaf}")
                    else:
                        paths.append(f"{prefix}>{idx}>{leaf}")
        else:
            paths.append(f"{prefix}>{str(node)}")

    _walk(cl, root, at_root=True)
    return paths

def nested_list_to_newick(cl: list) -> str:
    """
    Convert a multilevel nested list (internal nodes as lists, leaves as labels)
    into a Newick tree string without branch lengths.
    """
    def _is_list_like(x: Any) -> bool:
        return isinstance(x, (list, tuple))

    def _escape_label(x: Any) -> str:
        s = str(x)
        if any(c in s for c in " \t\r\n,:;()[]"):
            s = "'" + s.replace("'", "''") + "'"
        return s

    def _to_newick(node: Any) -> str:
        if _is_list_like(node):
            children = [ _to_newick(ch) for ch in node if not (_is_list_like(ch) and len(ch) == 0) ]
            if not children:
                return "''"
            return "(" + ",".join(children) + ")"
        else:
            return _escape_label(node)

    out = _to_newick(cl)
    if not (out.startswith("(") and out.endswith(")")):
        out = f"({out})"
    return out + ";"

def saveTextFile(strText: str, fname: Optional[str] = None) -> Optional[str]:
    # Minimal saver; if you need an interactive dialog, integrate externally.
    if not fname:
        return None
    with open(fname, "w", encoding="utf-8") as ofp:
        ofp.write(strText)
        ofp.flush()
    return fname

# ---------- Utilities for hierarchical label codes & table I/O ----------

def _is_list_like(x: Any) -> bool:
    return isinstance(x, (list, tuple))

def _tree_to_hier_codes(nested: list) -> Dict[str, str]:
    """
    Convert a nested list tree into mapping {leaf_label -> '1.2.3'}.
    Child indices are 1-based at every level.
    """
    mapping: Dict[str, str] = {}
    def walk(node: Any, path: List[int]) -> None:
        if _is_list_like(node):
            for idx, child in enumerate(node, start=1):
                walk(child, path + [idx])
        else:
            mapping[str(node)] = ".".join(map(str, path))
    walk(nested, [])
    return mapping

def _read_table_any(path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Read CSV/TSV robustly or accept a DataFrame.
    Uses sniffing; falls back to TSV; ensures at least 1 column.
    """
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    try:
        df = pd.read_csv(path_or_df, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path_or_df, sep="\t")
    if df.shape[1] == 1:
        # Try explicit tab if sniffing failed
        try:
            df = pd.read_csv(path_or_df, sep="\t")
        except Exception:
            pass
    if df.shape[1] < 1:
        raise ValueError("Input table appears to have no columns.")
    return df

def _write_table_like_input(df_out: pd.DataFrame, output_file: str) -> None:
    sep = "\t" if output_file.lower().endswith(".tsv") else ","
    df_out.to_csv(output_file, index=False, sep=sep)

# ----------------- Main: hierarchical spectral analysis -----------------

def spectral_analysis(matrix: MatrixLike,
                      labels: List[str],
                      max_cluster_number: int = 5,
                      max_cluster_content: int = 5,
                      output_file: str = "",          # save output to file (CSV/TSV if output_format='matrix')
                      output_format: str = "",        # 'newick'/'pathways'/'matrix'
                      force_k: Optional[int] = None,
                      max_levels: int = 100,
                      input_csv: Optional[Union[str, pd.DataFrame]] = None
                      ) -> Union[List[List[object]], str, List[str], pd.DataFrame]:
    """
    Hierarchical spectral clustering with:
      (A) top-level reduction if number of clusters > max_cluster_number
      (B) within-cluster splitting if any cluster contains > max_cluster_content leaves

    Returns:
      - nested list (default),
      - Newick string if output_format='newick',
      - list of 'Root>..' path strings if output_format='pathways',
      - DataFrame if output_format='matrix' (requires input_csv).
    """
    # Expand to a full matrix first, then validate labels
    D_full = _expand_distance_any(matrix)
    n = D_full.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) must equal number of taxa ({n})")

    # Level-1 spectral clustering on leaves
    first_groups_idx = _spectral_cluster_from_distance(D_full, force_k=force_k)
    nested: List[List[object]] = [[labels[i] for i in grp] for grp in first_groups_idx]
    idx_level: List[List[int]]   = [grp[:] for grp in first_groups_idx]

    # Iterate hierarchy construction
    levels_done = 0
    while levels_done < max_levels:
        changed_any = False

        # (B) Split oversized clusters by reclustering submatrix until none remain
        while True:
            changed, nested, idx_level = _split_large_once(
                nested, idx_level, D_full, labels, max_cluster_content, force_k
            )
            changed_any = changed_any or changed
            if not changed:
                break

        # (A) If too many top-level clusters, reduce by clustering cluster-centroids via average distances
        if len(idx_level) > max_cluster_number:
            R = _reduce_distance_matrix_by_average(idx_level, D_full)
            higher_groups_idx = _spectral_cluster_from_distance(R, force_k=force_k)

            # Wrap: new parents grouping current top-level nodes
            new_nested: List[List[object]] = []
            new_idx_level: List[List[int]] = []
            for group in higher_groups_idx:
                parent_node: List[object] = []
                parent_leaves: List[int] = []
                for g in group:
                    parent_node.append(nested[g])      # keep entire subtree
                    parent_leaves.extend(idx_level[g]) # accumulate leaf indices
                new_nested.append(parent_node)
                # keep unique order (stable) for leaves inside the parent
                seen = set()
                ordered = []
                for x in parent_leaves:
                    if x not in seen:
                        seen.add(x)
                        ordered.append(x)
                new_idx_level.append(ordered)
            nested, idx_level = new_nested, new_idx_level
            changed_any = True

        # If both constraints satisfied and no changes, stop
        if (len(idx_level) <= max_cluster_number) and not changed_any:
            break

        levels_done += 1
        if not changed_any:
            break
    
    # ---------- Outputs ----------
    fmt = (output_format or "").lower()

    if fmt == "newick":
        out: Union[List[List[object]], str, List[str], pd.DataFrame] = nested_list_to_newick(nested)
    elif fmt == "pathways":
        out = nested_list_to_pathways(nested)
    elif fmt == "matrix":
        if input_csv is None:
            raise ValueError("output_format='matrix' requires input_csv (path or DataFrame) matching the passed labels.")
        df_in = _read_table_any(input_csv)

        # Ensure row order matches 'labels'; assume the first column contains sample IDs
        first_col = df_in.columns[0]
        # Fast path if already ordered
        if list(df_in[first_col].astype(str).values) != [str(x) for x in labels]:
            df_idx = df_in.set_index(first_col)
            # Convert to str to avoid type mismatches during lookup
            df_idx.index = df_idx.index.astype(str)
            ordered = [str(x) for x in labels]
            missing = [lab for lab in ordered if lab not in df_idx.index]
            if missing:
                raise ValueError(f"The following label IDs are missing in input_csv first column '{first_col}': {missing}")
            df_in = df_idx.loc[ordered].reset_index().rename(columns={"index": first_col})

        # Compute hierarchical codes and insert as second column "Labels"
        code_map = _tree_to_hier_codes(nested)
        codes = [code_map[str(x)] for x in labels]  # preserves order of 'labels'
        df_out = df_in.copy()
        df_out.insert(1, "Labels", codes)

        out = df_out

        if output_file:
            _write_table_like_input(df_out, output_file)
    else:
        out = nested  # default nested structure

    if fmt in ("newick", "pathways") and output_file:
        out_text = str(out)
        if fmt == "pathways":
            # Pretty indent by depth (optional)
            out_text = "\n".join([("   " * (len(line.split(">")) - 3)) + line for line in out])  # type: ignore[arg-type]
        saveTextFile(out_text, output_file)

    return out


# =============================================================================
# Basic base class
# =============================================================================

class Basic:
    def __init__(self, args: Any):
        self.args = args
        self.options: Dict[str, Any] = {}

        # Common options expected in run.py
        project = getattr(args, "project", "")

        # Common options expected in run.py
        self.options["input_directory"] = os.path.join(
            getattr(args, "input_directory", "input"),
            project
        )
        
        self.options["output_directory"] = os.path.join(
            getattr(args, "output_directory", "output"),
            project
        )
        self.options["output_format"] = getattr(args, "output_format", "matrix")
        self.options["out_file"] = getattr(args, "out_file", None)
        self.options["seed"] = getattr(args, "seed", 13)

        # Optional preprocessing hooks used by selectors
        self.options["scale"] = getattr(args, "scale", False)
        self.options["transform"] = getattr(args, "transform", "none")  # none|clr
        self.options["pca"] = getattr(args, "pca", 0)
        self.options["svd"] = getattr(args, "svd", 0)

    # ---------- paths ----------

    def resolve_paths(self) -> Tuple[Path, Path, str, bool]:
        """
        Resolve the input path and the base output path.

        output_format may be one of:
            all | newick | matrix | pathway | text

        For a specific format, ``out_path`` is the final output path.
        For ``all``, ``out_path`` is a common basename used to derive the five
        report filenames via ``report_output_path()``.
        """
        in_base = os.path.basename(getattr(self.args, "input_file"))
        in_path = Path(self.options["input_directory"]) / in_base
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        ext = in_base.lower().split(".")[-1]
        is_tsv = (ext == "tsv")
        delimiter = "\t" if is_tsv else ","

        fmt = str(self.options["output_format"]).lower()
        valid_formats = {"all", "newick", "matrix", "pathway", "text"}
        if fmt not in valid_formats:
            raise ValueError(
                f"Unknown output_format: {fmt}. Expected one of: "
                + ", ".join(sorted(valid_formats))
            )

        algo = str(getattr(self.args, "algorithm", "")).upper()
        stem, orig_ext = os.path.splitext(in_base)

        if self.options["out_file"]:
            out_base = os.path.basename(self.options["out_file"])
            stem, _ = os.path.splitext(out_base)
            
        if fmt == "all":
            out_path = Path(self.options["output_directory"]) / f"{stem}_{algo}"
        elif fmt == "matrix":
            out_path = Path(self.options["output_directory"]) / f"{stem}_matrix_{algo}{orig_ext}"
        else:
            out_path = Path(self.options["output_directory"]) / f"{stem}_{algo}{orig_ext}"

        _ensure_output_dir(str(out_path))

        if fmt == "newick":
            out_path = Path(_swap_extension(str(out_path), ".nwk"))
        elif fmt == "matrix":
            out_path = Path(_swap_extension(str(out_path), ".tsv" if is_tsv else ".csv"))
        elif fmt in {"text", "pathway"}:
            out_path = Path(_swap_extension(str(out_path), ".txt"))

        return in_path, out_path, delimiter, is_tsv

    def report_output_path(
        self,
        base_out_path: Path,
        report_format: str,
        delimiter: str,
        all_mode: bool,
    ) -> Path:
        """Return the output filename for one report type."""
        report_format = report_format.lower()
        if not all_mode:
            return Path(base_out_path)

        base = str(base_out_path)
        if report_format == "newick":
            return Path(base + ".nwk")
        if report_format == "matrix":
            return Path(base + "_matrix" + (".tsv" if delimiter == "\t" else ".csv"))
        if report_format == "pathway":
            return Path(base + "_pathway.txt")
        if report_format == "text":
            return Path(base + "_text.txt")
        raise ValueError(f"Unknown report format: {report_format}")

    # ---------- I/O ----------
    '''
    def read_labeled_matrix(self, path: Path, delimiter: str) -> Tuple[List[str], List[str], List[List[float]]]:
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

        if not rows or len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError("Input must have a header row and at least one data row with labels.")

        col_labels = [c for c in rows[0][1:] if c != ""]
        row_labels = [r[0] for r in rows[1:]]
        body_str = [r[1:1 + len(col_labels)] for r in rows[1:]]

        def to_float(x: str) -> float:
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

        data = [[to_float(x) for x in r] for r in body_str]
        return col_labels, row_labels, data
    '''
    
    def read_labeled_matrix(self, path: Path, delimiter: str) -> Tuple[List[str], List[str], List[List[float]]]:
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
    
        if not rows or len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError("Input must have a header row and at least one data row with labels.")
    
        col_labels = [c for c in rows[0][1:] if c != ""]
        row_labels = [r[0] for r in rows[1:]]
    
        # Make sure each row has at least len(col_labels) cells (pad with "")
        body_str: List[List[str]] = []
        for r in rows[1:]:
            vals = r[1:1 + len(col_labels)]
            if len(vals) < len(col_labels):
                vals = vals + [""] * (len(col_labels) - len(vals))
            body_str.append(vals)
    
        # ------------------------------------------------------------
        # Column filtering (remove_empty_filed / empty_symbol)
        # ------------------------------------------------------------
        thr_raw = getattr(self.args, "remove_empty_filed", None)
        if thr_raw is not None:
            try:
                thr = float(thr_raw)
            except Exception:
                raise ValueError("--remove_empty_filed must be a floating number.")
            if thr < 0.0 or thr > 1.0:
                raise ValueError("--remove_empty_filed must be within [0.0, 1.0].")
    
            if thr < 1.0:
                empty_symbol = getattr(self.args, "empty_symbol", "")
                # IMPORTANT: remove_empty_columns is expected to work on string matrix
                # and return (new_col_labels, new_body_str, removed_idx)
                col_labels, body_str, removed_idx = self.remove_empty_columns(
                    col_labels=col_labels,
                    data=body_str,
                    thr=thr,
                    empty_symbol=empty_symbol
                )
    
        def to_float(x: str) -> float:
            s = _strip(x)
    
            # Treat configured empty symbol as empty too (string compare even if symbol is numeric)
            empty_sym = str(getattr(self.args, "empty_symbol", "")).strip()
            if empty_sym != "" and str(s).strip() == empty_sym:
                return float("nan")
    
            if s in {"", "NA", "N/A", "NaN", "nan", "-", ".", "*"}:
                return float("nan")
            try:
                return float(s)
            except Exception:
                try:
                    return float(s.replace(",", "."))
                except Exception:
                    return float("nan")
    
        data = [[to_float(x) for x in r] for r in body_str]
        return col_labels, row_labels, data

    def remove_empty_columns(self,
        col_labels: List[str],
        data: List[List[str]],
        thr: float,
        empty_symbol: object
        ) -> Tuple[List[str], List[List[str]], List[int]]:
        """
        Remove columns where fraction of empty cells is > thr.
    
        Parameters
        ----------
        col_labels : list[str]
            Column names (features).
        data : list[list[str]]
            Matrix body as strings (same shape as col_labels).
        thr : float
            Threshold in [0,1]. Column removed if empty_fraction > thr.
        empty_symbol : object
            Symbol representing empty cells (may be "", "nd", "-", 0, etc).
            Compared as STRING.
    
        Returns
        -------
        new_col_labels : list[str]
        new_data : list[list[str]]
        removed_idx : list[int]
            0-based indices (relative to original col_labels) that were removed.
        """
        if not data:
            return col_labels, data, []
        n = len(data)
        m = len(col_labels)
    
        # normalize empty symbol to string (even if user passed 0)
        empty_sym = str(empty_symbol).strip().lower()
    
        # helper: decide if a cell is empty
        def is_empty(cell: object) -> bool:
            s = "" if cell is None else str(cell)
            s2 = s.strip()
            if s2 == "":
                return True
            if empty_sym != "" and s2.lower() == empty_sym:
                return True
            return False
    
        # determine which columns to keep
        keep = []
        removed_idx = []
        for j in range(m):
            empty_count = 0
            for i in range(n):
                # protect against ragged rows
                cell = data[i][j] if j < len(data[i]) else ""
                if is_empty(cell):
                    empty_count += 1
            frac = empty_count / n
            if frac > thr:
                removed_idx.append(j)
            else:
                keep.append(j)
    
        # apply filtering
        new_col_labels = [col_labels[j] for j in keep]
        new_data = []
        for i in range(n):
            row = data[i]
            new_row = [(row[j] if j < len(row) else "") for j in keep]
            new_data.append(new_row)
    
        return new_col_labels, new_data, removed_idx

    def impute(self, data: List[List[float]]) -> None:
        _impute_col_median_inplace(data)

    def is_binary(self, data: List[List[float]]) -> bool:
        return _all_binary(data)

    def _build_pathway_headings(
        self,
        input_csv: Path | str | pd.DataFrame,
        labels: list[str],
        cluster_labels: list[str | int] | np.ndarray,
        delimiter: str
    ) -> list[str]:
        """
        Build row headings with clustering levels prepended.
    
        Returns
        -------
        list[str]
            Modified headings such as:
    
                1|1|1|Environment|Source|Genome_A
                1|2|1|Environment|Source|Genome_B
                0|0|0|Environment|Source|Genome_C
    
        The number of clustering levels prepended is controlled by:
    
            self.args.max_cluster_levels
    
        and the hierarchy delimiter by:
    
            self.args.delimiter
        """
    
        label_delimiter = self.args.delimiter
        max_levels = int(self.args.max_cluster_levels)
    
        if max_levels < 1:
            raise ValueError(
                "self.args.max_cluster_levels must be >= 1."
            )
    
        # ------------------------------------------------------------
        # Load input table
        # ------------------------------------------------------------
        if isinstance(input_csv, pd.DataFrame):
            df_in = input_csv.copy()
        else:
            df_in = pd.read_csv(
                input_csv,
                sep=delimiter,
                dtype=str
            )
    
        if df_in.shape[1] < 1:
            raise ValueError(
                "Input table contains no columns."
            )
    
        first_col = df_in.columns[0]
    
        # ------------------------------------------------------------
        # Validate lengths
        # ------------------------------------------------------------
        if len(cluster_labels) != len(labels):
            raise ValueError(
                f"cluster_labels length ({len(cluster_labels)}) "
                f"!= labels length ({len(labels)})"
            )
    
        # ------------------------------------------------------------
        # Original row headings
        # ------------------------------------------------------------
        wanted = [
            str(x).strip()
            for x in labels
        ]
    
        headings = (
            df_in[first_col]
            .astype(str)
            .str.strip()
            .tolist()
        )
    
        if headings != wanted:
            raise ValueError(
                "Row headings in the input matrix do not match "
                "the labels used for clustering."
            )
    
        # ------------------------------------------------------------
        # Split a clustering code
        # ------------------------------------------------------------
        def _split_cluster_code(value: str) -> list[str]:
    
            s = str(value).strip()
    
            if not s:
                return []
    
            if label_delimiter in s:
                parts = s.split(label_delimiter)
    
            elif "." in s:
                parts = s.split(".")
    
            else:
                parts = [s]
    
            return [
                part.strip()
                for part in parts
                if part.strip()
            ]
    
        # ------------------------------------------------------------
        # Detect already hierarchical labels
        # ------------------------------------------------------------
        def _is_numeric_code(value: str) -> bool:
            parts = _split_cluster_code(value)
            return bool(parts) and all(
                part.lstrip("-").isdigit()
                for part in parts
            )
    
        codes: list[list[str]] = []
    
        cluster_labels_list = list(cluster_labels)
    
        # ------------------------------------------------------------
        # Case A: labels already hierarchical
        # ------------------------------------------------------------
        parsed_depths = [
            len(_split_cluster_code(x)) if isinstance(x, str) else 0
            for x in cluster_labels_list
        ]
        has_hierarchy = (
            all(isinstance(x, str) and _is_numeric_code(x) for x in cluster_labels_list)
            and any(depth >= 2 for depth in parsed_depths)
        )

        if has_hierarchy:
    
            for x in cluster_labels_list:
    
                parts = _split_cluster_code(x)
    
                parts = parts[:max_levels]
    
                if len(parts) < max_levels:
                    parts += ["0"] * (
                        max_levels - len(parts)
                    )
    
                codes.append(parts)
    
        # ------------------------------------------------------------
        # Case B: flat cluster labels
        # ------------------------------------------------------------
        else:
    
            norm: list[int] = []
    
            for x in cluster_labels_list:
    
                try:
                    norm.append(
                        int(float(str(x).strip()))
                    )
    
                except (ValueError, TypeError):
                    norm.append(-1)
    
            # Remap normal clusters to 1..K.
            # Noise remains represented by zeros.
            unique_clusters = sorted({
                value
                for value in norm
                if value >= 0
            })
    
            remap = {
                old: i + 1
                for i, old in enumerate(unique_clusters)
            }
    
            within_counter: dict[int, int] = {}
    
            for value in norm:
    
                if value < 0:
    
                    parts = ["0"] * max_levels
    
                else:
    
                    cluster_id = remap[value]
    
                    within_counter[cluster_id] = (
                        within_counter.get(cluster_id, 0)
                        + 1
                    )
    
                    within_id = within_counter[
                        cluster_id
                    ]
    
                    full_code = [
                        str(cluster_id),
                        str(within_id),
                        "1"
                    ]
    
                    parts = full_code[:max_levels]
    
                    if len(parts) < max_levels:
                        parts += ["0"] * (
                            max_levels - len(parts)
                        )
    
                codes.append(parts)
    
        # ------------------------------------------------------------
        # Build modified headings
        # ------------------------------------------------------------
        pathway_headings: list[str] = []
    
        for cluster_parts, original_heading in zip(
            codes,
            headings
        ):
    
            prefix = label_delimiter.join(
                cluster_parts
            )
    
            pathway_headings.append(
                prefix
                + label_delimiter
                + original_heading
            )
    
        return pathway_headings

    # ---------- preprocessing ----------

    def _apply_transform(self, X: np.ndarray) -> np.ndarray:
        tr = str(self.options.get("transform", "none")).lower()
        if tr != "clr":
            return X

        # CLR with pseudocount
        Xp = np.asarray(X, dtype=float)
        Xp = np.where(np.isfinite(Xp), Xp, 0.0)
        Xp = np.maximum(Xp, 0.0)
        Xp = Xp + 1e-6
        gm = np.exp(np.mean(np.log(Xp), axis=1, keepdims=True))
        gm = np.where(gm == 0.0, 1.0, gm)
        return np.log(Xp / gm)

    def _apply_embedding(self, X: np.ndarray) -> np.ndarray:
        # SVD/PCA hooks (optional; only active if sklearn is available)
        svd_k = int(self.options.get("svd", 0) or 0)
        pca_k = int(self.options.get("pca", 0) or 0)
        k = max(svd_k, pca_k)
        if k <= 0:
            return X

        try:
            from sklearn.decomposition import TruncatedSVD, PCA
        except Exception:
            return X

        k = min(k, max(2, min(X.shape[0] - 1, X.shape[1])))
        if svd_k > 0:
            return TruncatedSVD(n_components=k, random_state=int(self.options.get("seed", 13))).fit_transform(X)
        return PCA(n_components=k, random_state=int(self.options.get("seed", 13))).fit_transform(X)

    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        if not bool(self.options.get("scale", False)):
            return X
        try:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler().fit_transform(X)
        except Exception:
            return X

    # ---------- distances ----------

    def build_distance_matrix(self, X: np.ndarray, metric: str) -> np.ndarray:
        metric = metric.lower()
        if metric == "jaccard":
            Xb = (X != 0.0)
            return _jaccard_pairwise_distance_bool(Xb)
        if metric == "braycurtis":
            return _braycurtis_pairwise_distance(X)
        return _euclidean_pairwise_distance(X)

    # ---------- output writers ----------

    def write_text(self, text: str, out_path: Path) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    def write_df(self, df: pd.DataFrame, out_path: Path, delimiter: str) -> None:
        sep = "\t" if delimiter == "\t" else ","
        df.to_csv(out_path, index=False, sep=sep)

    def output_newick(self, labels: List[str], dist_mat: np.ndarray, out_path: Path) -> None:
        newick = _upgma_newick(labels, dist_mat)
        self.write_text(newick + "\n", out_path)

    def emit_requested_reports(
        self,
        *,
        out_fmt: str,
        base_out_path: Path,
        in_path: Path,
        delimiter: str,
        row_labels: List[str],
        cluster_labels: Union[List[int], List[str], np.ndarray],
        dist_mat: np.ndarray,
        text_title: str,
        text_extra_lines: List[str],
        custom_newick: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Save one requested report or all reports.

        Text and pathway reports are always generated from the same
        ``_build_pathway_headings()`` result so their hierarchical codes are
        guaranteed to be consistent.
        """
        out_fmt = str(out_fmt).lower()
        valid = {"all", "newick", "matrix", "pathway", "text"}
        if out_fmt not in valid:
            raise ValueError(
                f"Unknown output format {out_fmt!r}. Expected one of: "
                + ", ".join(sorted(valid))
            )

        all_mode = (out_fmt == "all")
        requested = {"newick", "matrix", "pathway", "text"} if all_mode else {out_fmt}
        outputs: Dict[str, Path] = {}
        labels = [str(x) for x in row_labels]

        pathway_headings: Optional[List[str]] = None
        if requested & {"pathway", "text"}:
            pathway_headings = self._build_pathway_headings(
                input_csv=in_path,
                labels=labels,
                cluster_labels=cluster_labels,
                delimiter=delimiter,
            )

        if "newick" in requested:
            p = self.report_output_path(base_out_path, "newick", delimiter, all_mode)
            _ensure_output_dir(str(p))
            if custom_newick is not None:
                self.write_text(custom_newick.rstrip() + "\n", p)
            else:
                self.output_newick(labels, np.asarray(dist_mat, dtype=float), p)
            outputs["newick"] = p

        if "matrix" in requested:
            p = self.report_output_path(base_out_path, "matrix", delimiter, all_mode)
            _ensure_output_dir(str(p))
            self.output_matrix_with_labels(
                input_csv=in_path,
                labels=labels,
                cluster_labels=list(cluster_labels),
                out_path=p,
                delimiter=delimiter,
            )
            outputs["matrix"] = p

        if "pathway" in requested:
            assert pathway_headings is not None
            p = self.report_output_path(base_out_path, "pathway", delimiter, all_mode)
            _ensure_output_dir(str(p))
            self.output_pathway_report(pathway_headings=pathway_headings, out_path=p)
            outputs["pathway"] = p

        if "text" in requested:
            assert pathway_headings is not None
            p = self.report_output_path(base_out_path, "text", delimiter, all_mode)
            _ensure_output_dir(str(p))
            self.output_text_report(
                title=text_title,
                in_name=in_path.name,
                pathway_headings=pathway_headings,
                extra_lines=text_extra_lines,
                out_path=p,
                levels=int(getattr(self.args, "max_cluster_levels", 1)),
            )
            outputs["text"] = p

        return outputs
        
    def output_pathway_report(
        self,
        pathway_headings: list[str],
        out_path: Path,
    ) -> None:
        """
        Save modified hierarchical row headings, one per line.
        """
    
        out_path = Path(out_path)
    
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
    
        with open(
            out_path,
            "w",
            encoding="utf-8",
            newline="\n"
        ) as fp:
    
            for heading in pathway_headings:
                fp.write(heading + "\n")

    '''
    def output_pathway_report(
        self,
        input_csv: Path | str | pd.DataFrame,
        labels: list[str],
        cluster_labels: list[str | int],
        out_path: Path,
        delimiter: str
    ) -> None:
        """
        Save a pathway report containing only modified row headings.
    
        Clustering hierarchy levels are prepended to the original first-column
        heading using:
    
            self.args.delimiter
            self.args.max_cluster_levels
    
        Example
        -------
        Original heading:
            Environmental|Environmental swabs|GCA_123
    
        Cluster label:
            1.0.1
    
        With:
            self.args.delimiter = "|"
            self.args.max_cluster_levels = 2
    
        Output:
            1|0|Environmental|Environmental swabs|GCA_123
    
        One modified heading is written per line.
        """
    
        label_delimiter = self.args.delimiter
        max_levels = int(self.args.max_cluster_levels)
    
        if max_levels < 1:
            raise ValueError(
                "self.args.max_cluster_levels must be >= 1."
            )
    
        # ------------------------------------------------------------
        # Load original table
        # ------------------------------------------------------------
        if isinstance(input_csv, pd.DataFrame):
            df_in = input_csv.copy()
        else:
            df_in = pd.read_csv(
                input_csv,
                sep=delimiter,
                dtype=str
            )
    
        if df_in.shape[1] < 1:
            raise ValueError("Input table contains no columns.")
    
        first_col = df_in.columns[0]
    
        # ------------------------------------------------------------
        # Check lengths
        # ------------------------------------------------------------
        if len(cluster_labels) != len(labels):
            raise ValueError(
                f"cluster_labels length ({len(cluster_labels)}) "
                f"!= labels length ({len(labels)})"
            )
    
        # ------------------------------------------------------------
        # Match input rows to labels
        #
        # Unique genome/strain ID is the last part of the heading.
        # ------------------------------------------------------------
        wanted = [str(x).strip() for x in labels]
        
        headings = (
            df_in[first_col]
            .astype(str)
            .str.strip()
            .tolist()
        )
        
        if headings != wanted:
            raise ValueError(
                "Row headings in the input matrix do not match "
                "the labels used for clustering."
            )
    
        # ------------------------------------------------------------
        # Parse hierarchical clustering code
        # ------------------------------------------------------------
        def _split_cluster_code(value: str) -> list[str]:
    
            s = str(value).strip()
    
            if not s:
                return []
    
            # Already uses current delimiter
            if label_delimiter in s:
                parts = s.split(label_delimiter)
    
            # Historical cluster notation: 1.0.1
            elif "." in s:
                parts = s.split(".")
    
            else:
                parts = [s]
    
            return [
                part.strip()
                for part in parts
                if part.strip() != ""
            ]
    
        # ------------------------------------------------------------
        # Determine whether labels are already hierarchical
        # ------------------------------------------------------------
        def _is_hierarchical(value: str) -> bool:
    
            parts = _split_cluster_code(value)
    
            return (
                len(parts) >= 2
                and all(
                    part.lstrip("-").isdigit()
                    for part in parts
                )
            )
    
        codes: list[list[str]] = []
    
        # ------------------------------------------------------------
        # Case A: existing hierarchical labels, e.g. 1.0.1
        # ------------------------------------------------------------
        if all(
            isinstance(x, str) and _is_hierarchical(x)
            for x in cluster_labels
        ):
    
            for x in cluster_labels:
    
                parts = _split_cluster_code(x)
    
                # Keep requested hierarchy depth
                parts = parts[:max_levels]
    
                # Pad missing levels
                if len(parts) < max_levels:
                    parts += ["0"] * (
                        max_levels - len(parts)
                    )
    
                codes.append(parts)
    
        # ------------------------------------------------------------
        # Case B: flat numeric cluster labels
        # ------------------------------------------------------------
        else:
    
            norm: list[int] = []
    
            for x in cluster_labels:
    
                try:
                    norm.append(int(float(str(x).strip())))
                except (ValueError, TypeError):
                    norm.append(-1)
    
            # Convert original cluster IDs to 1..K
            unique_clusters = sorted({
                value
                for value in norm
                if value >= 0
            })
    
            remap = {
                old: i + 1
                for i, old in enumerate(unique_clusters)
            }
    
            within_counter: dict[int, int] = {}
    
            for value in norm:
    
                # Noise / unassigned
                if value < 0:
    
                    parts = ["0"] * max_levels
    
                else:
    
                    cluster_id = remap[value]
    
                    within_counter[cluster_id] = (
                        within_counter.get(cluster_id, 0) + 1
                    )
    
                    within_id = within_counter[cluster_id]
    
                    full_code = [
                        str(cluster_id),
                        str(within_id),
                        "1"
                    ]
    
                    parts = full_code[:max_levels]
    
                    if len(parts) < max_levels:
                        parts += ["0"] * (
                            max_levels - len(parts)
                        )
    
                codes.append(parts)
    
        # ------------------------------------------------------------
        # Build modified headings
        # ------------------------------------------------------------
        pathway_headings: list[str] = []
    
        for cluster_parts, original_heading in zip(
            codes,
            headings
        ):
    
            prefix = label_delimiter.join(cluster_parts)
    
            pathway_headings.append(
                prefix
                + label_delimiter
                + original_heading
            )
    
        # ------------------------------------------------------------
        # Save pathway report
        # ------------------------------------------------------------
        out_path = Path(out_path)
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
    
        with open(
            out_path,
            "w",
            encoding="utf-8",
            newline="\n"
        ) as fp:
    
            for heading in pathway_headings:
                fp.write(heading + "\n")
    '''        

    def output_matrix_with_labels(
        self,
        input_csv: Path | str | pd.DataFrame,
        labels: list[str],
        cluster_labels: list[str | int],
        out_path: Path,
        delimiter: str
    ) -> None:
        """
        Write an output matrix by prepending clustering hierarchy levels
        to the text in the first column.
    
        The separator used inside the first-column heading is:
            self.args.delimiter
    
        The number of clustering hierarchy levels to prepend is:
            self.args.max_cluster_levels
    
        Examples
        --------
        Original first-column value:
            Environmental|Environmental swabs|GCA_123
    
        Cluster code:
            1.0.1
    
        max_cluster_levels = 1:
            1|Environmental|Environmental swabs|GCA_123
    
        max_cluster_levels = 2:
            1|0|Environmental|Environmental swabs|GCA_123
    
        max_cluster_levels = 3:
            1|0|1|Environmental|Environmental swabs|GCA_123
    
        Notes
        -----
        - Existing hierarchical string cluster labels may be separated by "."
          or by self.args.delimiter.
        - Numeric cluster labels are converted to hierarchical codes using
          the same logic as before.
        - If fewer cluster levels are available than requested,
          missing levels are padded with "0".
        """
    
        # ------------------------------------------------------------
        # Settings
        # ------------------------------------------------------------
        label_delimiter = self.args.delimiter
        max_levels = int(self.args.max_cluster_levels)
    
        if max_levels < 1:
            raise ValueError(
                "self.args.max_cluster_levels must be >= 1."
            )
    
        # ------------------------------------------------------------
        # Load original table
        # ------------------------------------------------------------
        if isinstance(input_csv, pd.DataFrame):
            df_in = input_csv.copy()
        else:
            df_in = pd.read_csv(input_csv, sep=delimiter)
    
        if df_in.shape[1] < 2:
            raise ValueError(
                "Input table must have at least 2 columns "
                "(row identifier + features)."
            )
    
        first_col = df_in.columns[0]
    
        # ------------------------------------------------------------
        # Ensure row order matches labels
        #
        # labels contains unique genome/strain identifiers,
        # whereas the first column may contain:
        #
        #   Environment|Source|Genome_ID
        #
        # Therefore compare labels with the LAST component of
        # the first-column heading.
        # ------------------------------------------------------------

        wanted = [str(x).strip() for x in labels]
        
        headings = (
            df_in[first_col]
            .astype(str)
            .str.strip()
            .tolist()
        )
        
        if headings != wanted:
            raise ValueError(
                "Row headings in the input matrix do not match "
                "the labels used for clustering."
            )
            
        # ------------------------------------------------------------
        # Validate cluster label count
        # ------------------------------------------------------------
        if len(cluster_labels) != len(labels):
            raise ValueError(
                f"cluster_labels length ({len(cluster_labels)}) "
                f"!= labels length ({len(labels)})"
            )
    
        # ------------------------------------------------------------
        # Recognize hierarchical cluster labels
        # ------------------------------------------------------------
        def _split_hierarchical_code(value: str) -> list[str]:
            """
            Convert a hierarchical clustering label into a list.
    
            Accepted examples:
                "1.0.1" -> ["1", "0", "1"]
                "1|0|1" -> ["1", "0", "1"]
    
            The output separator is always self.args.delimiter.
            """
    
            s = str(value).strip()
    
            if not s:
                return []
    
            # First try the configured delimiter
            if label_delimiter in s:
                parts = s.split(label_delimiter)
    
            # Historical clustering codes used dots
            elif "." in s:
                parts = s.split(".")
    
            else:
                parts = [s]
    
            return [
                p.strip()
                for p in parts
                if p.strip() != ""
            ]
    
        def _looks_hierarchical(value: str) -> bool:
            parts = _split_hierarchical_code(value)
    
            if len(parts) < 2:
                return False
    
            return all(
                part.lstrip("-").isdigit()
                for part in parts
            )
    
        # ------------------------------------------------------------
        # Build cluster codes
        # ------------------------------------------------------------
        codes: list[list[str]] = []
    
        # Case A:
        # cluster_labels already contain hierarchical cluster codes
        if all(
            isinstance(x, str) and _looks_hierarchical(x)
            for x in cluster_labels
        ):
            for x in cluster_labels:
                parts = _split_hierarchical_code(str(x))
    
                # Keep only requested levels
                parts = parts[:max_levels]
    
                # Pad if fewer levels are available
                if len(parts) < max_levels:
                    parts += ["0"] * (max_levels - len(parts))
    
                codes.append(parts)
    
        else:
            # --------------------------------------------------------
            # Case B:
            # flat numeric cluster IDs
            # --------------------------------------------------------
            norm: list[int] = []
    
            for x in cluster_labels:
    
                if isinstance(x, int):
                    norm.append(x)
    
                else:
                    xs = str(x).strip()
    
                    try:
                        norm.append(int(float(xs)))
                    except (ValueError, TypeError):
                        # unknown / invalid label -> noise
                        norm.append(-1)
    
            # Remap non-negative cluster IDs to 1..K
            uniq = sorted({
                v for v in norm
                if v >= 0
            })
    
            remap = {
                old: i + 1
                for i, old in enumerate(uniq)
            }
    
            within_counter: dict[int, int] = {}
    
            for v in norm:
    
                if v < 0:
                    # Noise/unassigned
                    parts = ["0"] * max_levels
    
                else:
                    cluster_id = remap[v]
    
                    within_counter[cluster_id] = (
                        within_counter.get(cluster_id, 0) + 1
                    )
    
                    within_id = within_counter[cluster_id]
    
                    # Historical format:
                    # cluster.within.1
                    full_code = [
                        str(cluster_id),
                        str(within_id),
                        "1"
                    ]
    
                    # Keep requested number of levels
                    parts = full_code[:max_levels]
    
                    # If >3 levels requested, extend with zeros
                    if len(parts) < max_levels:
                        parts += ["0"] * (
                            max_levels - len(parts)
                        )
    
                codes.append(parts)
    
        # ------------------------------------------------------------
        # Prepend clustering hierarchy to first column
        # ------------------------------------------------------------
        original_headings = (
            df_in[first_col]
            .astype(str)
            .tolist()
        )
    
        new_headings: list[str] = []
    
        for cluster_parts, original_heading in zip(
            codes,
            original_headings
        ):
            prefix = label_delimiter.join(cluster_parts)
    
            new_heading = (
                prefix
                + label_delimiter
                + original_heading
            )
    
            new_headings.append(new_heading)
    
        df_out = df_in.copy()
        df_out[first_col] = new_headings
    
        # ------------------------------------------------------------
        # Write output matrix
        # ------------------------------------------------------------
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
        df_out.to_csv(
            out_path,
            sep=delimiter,
            index=False
        )

    def output_text_report(
        self,
        title: str,
        in_name: str,
        pathway_headings: list[str],
        extra_lines: list[str],
        out_path: Path,
        levels: int = 1
    ) -> None:
        """
        Write a hierarchical clustering report from pathway headings.
    
        Example pathway heading:
    
            1|2|1|Environment|Source|Genome_A
    
        With:
            levels = 3
    
        hierarchy:
            1 -> 1.2 -> 1.2.1
    
        object heading:
            Environment|Source|Genome_A
        """
    
        if levels < 1:
            raise ValueError(
                "levels must be >= 1."
            )
    
        delimiter = self.args.delimiter
    
        # Do not request more levels than were actually generated.
        max_levels = int(
            self.args.max_cluster_levels
        )
    
        levels = min(
            levels,
            max_levels
        )
    
        # ------------------------------------------------------------
        # Parse pathway headings
        # ------------------------------------------------------------
    
        records: list[
            tuple[tuple[str, ...], str]
        ] = []
    
        for heading in pathway_headings:
    
            parts = [
                part.strip()
                for part in str(heading).split(
                    delimiter
                )
            ]
    
            if len(parts) <= levels:
                raise ValueError(
                    "Pathway heading does not contain "
                    f"{levels} cluster levels plus an "
                    f"object heading: {heading}"
                )
    
            cluster_path = tuple(
                parts[:levels]
            )
    
            object_heading = delimiter.join(
                parts[levels:]
            )
    
            records.append(
                (
                    cluster_path,
                    object_heading
                )
            )
    
        # ------------------------------------------------------------
        # Terminal groups
        # ------------------------------------------------------------
    
        terminal_groups: Dict[
            Tuple[str, ...],
            List[str]
        ] = {}
    
        for path, label in records:
    
            terminal_groups.setdefault(
                path,
                []
            ).append(label)
    
        # ------------------------------------------------------------
        # Calculate size of every hierarchy node
        # ------------------------------------------------------------
    
        node_sizes: Dict[
            Tuple[str, ...],
            int
        ] = {}
    
        for path, members in terminal_groups.items():
    
            count = len(members)
    
            for depth in range(
                1,
                len(path) + 1
            ):
    
                prefix = path[:depth]
    
                node_sizes[prefix] = (
                    node_sizes.get(prefix, 0)
                    + count
                )
    
        # ------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------
    
        n_samples = len(records)
    
        # 0|0|0 represents noise/unassigned in the pathway code.
        n_noise = sum(
            1
            for path, _ in records
            if all(x == "0" for x in path)
        )
    
        top_clusters = {
            path[0]
            for path, _ in records
            if path[0] != "0"
        }
    
        terminal_clusters = {
            path
            for path in terminal_groups
            if not all(
                x == "0"
                for x in path
            )
        }
    
        # ------------------------------------------------------------
        # Report header
        # ------------------------------------------------------------
    
        lines: list[str] = []
    
        lines.append(
            f"# {title}"
        )
    
        lines.append(
            f"# input={in_name}"
        )
    
        lines.append(
            f"# n_samples={n_samples}"
        )
    
        lines.append(
            "# n_clusters_level_1"
            f"(excluding_noise)="
            f"{len(top_clusters)}"
        )
    
        lines.append(
            "# n_terminal_clusters"
            f"(excluding_noise)="
            f"{len(terminal_clusters)}"
        )
    
        lines.append(
            f"# n_noise={n_noise}"
        )
    
        lines.append(
            f"# hierarchy_levels={levels}"
        )
    
        lines.extend(
            extra_lines
        )
    
        lines.append("")
    
        # ------------------------------------------------------------
        # Sort cluster paths numerically where possible
        # ------------------------------------------------------------
    
        def cluster_sort_key(
            path: Tuple[str, ...]
        ):
            return tuple(
                (
                    0,
                    int(value)
                )
                if value.lstrip("-").isdigit()
                else (
                    1,
                    value
                )
                for value in path
            )
    
        # ------------------------------------------------------------
        # Recursive output
        # ------------------------------------------------------------
    
        def write_node(
            prefix: Tuple[str, ...],
            indent: int = 0
        ) -> None:
    
            size = node_sizes[prefix]
    
            cluster_name = ".".join(
                prefix
            )
    
            # All-zero cluster path = noise
            if all(
                value == "0"
                for value in prefix
            ):
                tag = "NOISE"
    
            else:
                tag = (
                    f"Cluster {cluster_name}"
                )
    
            lines.append(
                f"{'    ' * indent}"
                f"{tag}\t(n={size})"
            )
    
            # Immediate child nodes
            children = {
                path[:len(prefix) + 1]
                for path in terminal_groups
                if (
                    len(path) > len(prefix)
                    and path[:len(prefix)]
                    == prefix
                )
            }
    
            children = sorted(
                children,
                key=cluster_sort_key
            )
    
            # --------------------------------------------------------
            # Terminal cluster
            # --------------------------------------------------------
    
            if not children:
    
                for member in terminal_groups.get(
                    prefix,
                    []
                ):
    
                    lines.append(
                        f"{'    ' * (indent + 1)}"
                        f"{member}"
                    )
    
                lines.append("")
    
                return
    
            # --------------------------------------------------------
            # Descend to child clusters
            # --------------------------------------------------------
    
            for child in children:
    
                write_node(
                    child,
                    indent + 1
                )
    
        # ------------------------------------------------------------
        # Top-level hierarchy
        # ------------------------------------------------------------
    
        top_nodes = sorted(
            {
                path[:1]
                for path in terminal_groups
            },
            key=cluster_sort_key
        )
    
        for node in top_nodes:
            write_node(node)
    
        # ------------------------------------------------------------
        # Save
        # ------------------------------------------------------------
    
        self.write_text(
            "\n".join(lines).rstrip()
            + "\n",
            out_path
        )

    '''
    def output_text_report(self, title: str, in_name: str, row_labels: List[str],
                           cluster_labels: np.ndarray, extra_lines: List[str],
                           out_path: Path) -> None:
        cluster_labels = np.asarray(cluster_labels, dtype=int)
        n = len(cluster_labels)
        n_noise = int(np.sum(cluster_labels == -1))
        uniq = sorted(set(int(x) for x in cluster_labels))
        n_clusters = len([c for c in uniq if c != -1])

        groups: Dict[int, List[str]] = {}
        for lab, cid in zip(row_labels, cluster_labels):
            groups.setdefault(int(cid), []).append(lab)

        lines: List[str] = []
        lines.append(f"# {title}")
        lines.append(f"# input={in_name}")
        lines.append(f"# n_samples={n}")
        lines.append(f"# n_clusters(excluding_noise)={n_clusters}")
        lines.append(f"# n_noise={n_noise}")
        lines.extend(extra_lines)
        lines.append("")

        for cid in sorted(groups.keys()):
            members = groups[cid]
            tag = "NOISE" if cid == -1 else f"Cluster {cid}"
            lines.append(f"{tag}\t(n={len(members)})")
            for x in members:
                lines.append(f"  {x}")
            lines.append("")

        self.write_text("\n".join(lines).rstrip() + "\n", out_path)
    '''

# =============================================================================
# SA (Spectral Analysis)
# =============================================================================

class SA(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "metric": getattr(args, "metric", None),
            "similarity": getattr(args, "similarity", "gaussian"),
            "sigma": getattr(args, "sigma", None),
            "epsilon": getattr(args, "epsilon", None),
            "expected_k": getattr(args, "expected_k", None),
            "max_cluster_number": getattr(args, "max_cluster_number", 5),
            "max_cluster_content": getattr(args, "max_cluster_content", 5),
            "bootstrap": getattr(args, "bootstrap", 0),
            "boot_mode": getattr(args, "boot_mode", "replace"),
            "boot_frac": getattr(args, "boot_frac", 0.1),
        })

    def execute(self):
        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        _, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")

        self.impute(data)
        n = len(data)
        m = len(data[0])
        is_square = (n == m)
        is_binary = self.is_binary(data)
        labels = [str(x) for x in row_labels]

        if is_square:
            dist = np.asarray(data, dtype=float)
            dist = np.nan_to_num(dist, nan=0.0)
            np.fill_diagonal(dist, 0.0)
            metric = "precomputed"
        else:
            X = np.asarray(data, dtype=float)
            X = np.nan_to_num(X, nan=0.0)
            metric = self.options["metric"] or ("jaccard" if is_binary else "euclidean")
            dist = self.build_distance_matrix(X, metric=str(metric))

        # spectral_analysis() uses the module-level similarity kernel. Temporarily
        # replace it when epsilon or a fixed sigma was requested.
        global _gaussian_similarity
        original_kernel = _gaussian_similarity
        try:
            if str(self.options["similarity"]).lower() == "epsilon":
                eps = self.options["epsilon"]
                if eps is None:
                    raise ValueError("Please provide --epsilon when --similarity epsilon.")
                epsilon = float(eps)

                def _eps_kernel(dist_arr: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
                    W = (dist_arr <= epsilon).astype(float)
                    np.fill_diagonal(W, 0.0)
                    return W

                _gaussian_similarity = _eps_kernel

            elif self.options["sigma"] is not None:
                fixed_sigma = float(self.options["sigma"])

                def _fixed_sigma(dist_arr: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
                    return original_kernel(dist_arr, sigma=fixed_sigma)

                _gaussian_similarity = _fixed_sigma

            force_k = self.options["expected_k"]
            force_k = None if force_k is None else int(force_k)

            nested = spectral_analysis(
                matrix=dist,
                labels=labels,
                max_cluster_number=int(self.options["max_cluster_number"]),
                max_cluster_content=int(self.options["max_cluster_content"]),
                output_file="",
                output_format="",
                force_k=force_k,
                max_levels=int(getattr(self.args, "max_cluster_levels", 100)),
            )
        finally:
            _gaussian_similarity = original_kernel

        if not isinstance(nested, list):
            raise RuntimeError("Spectral analysis did not return a nested cluster tree.")

        code_map = _tree_to_hier_codes(nested)
        cluster_codes = [code_map.get(str(label), "0") for label in labels]
        custom_newick = nested_list_to_newick(nested)
        out_fmt = str(self.options["output_format"]).lower()
        extra = [
            f"# metric={metric}",
            f"# similarity={self.options['similarity']}",
            f"# sigma={self.options['sigma']}",
            f"# epsilon={self.options['epsilon']}",
            f"# expected_k={self.options['expected_k']}",
            f"# max_cluster_number={self.options['max_cluster_number']}",
            f"# max_cluster_content={self.options['max_cluster_content']}",
        ]

        self.emit_requested_reports(
            out_fmt=out_fmt,
            base_out_path=out_path,
            in_path=in_path,
            delimiter=delimiter,
            row_labels=labels,
            cluster_labels=cluster_codes,
            dist_mat=np.asarray(dist, dtype=float),
            text_title="SA clustering report",
            text_extra_lines=extra,
            custom_newick=custom_newick,
        )
        return out_path


# =============================================================================

class DBSCAN(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "eps": getattr(args, "eps", 0.5),
            "min_samples": getattr(args, "min_samples", 5),
            "metric": getattr(args, "dbscan_metric", None),  # euclidean|jaccard|braycurtis
        })

    def execute(self):
        from sklearn.cluster import DBSCAN as _SK_DBSCAN

        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        _, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")
        self.impute(data)
        X = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
        n, m = X.shape
        is_square = (n == m)
        is_binary = self.is_binary(data)

        eps = float(self.options["eps"])
        ms = int(self.options["min_samples"])
        if eps <= 0:
            raise ValueError("--eps must be > 0")
        if ms <= 0:
            raise ValueError("--min_samples must be > 0")

        if is_square:
            dist_mat = X.copy()
            np.fill_diagonal(dist_mat, 0.0)
            labels = _SK_DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(dist_mat)
            metric = "precomputed"
        else:
            Xp = self._apply_scaling(self._apply_embedding(self._apply_transform(X)))
            metric = str(self.options["metric"] or ("jaccard" if is_binary else "euclidean")).lower()
            if metric in {"jaccard", "braycurtis"}:
                dist_mat = self.build_distance_matrix(Xp, metric)
                labels = _SK_DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(dist_mat)
            else:
                dist_mat = self.build_distance_matrix(Xp, "euclidean")
                labels = _SK_DBSCAN(eps=eps, min_samples=ms, metric="euclidean").fit_predict(Xp)

        labels = np.asarray(labels, dtype=int)
        out_fmt = str(self.options["output_format"]).lower()
        extra = [f"# eps={eps}", f"# min_samples={ms}", f"# metric={metric}"]
        self.emit_requested_reports(
            out_fmt=out_fmt, base_out_path=out_path, in_path=in_path, delimiter=delimiter,
            row_labels=[str(x) for x in row_labels], cluster_labels=labels,
            dist_mat=np.asarray(dist_mat, dtype=float), text_title="DBSCAN clustering report",
            text_extra_lines=extra,
        )
        return out_path


# =============================================================================

class HDBSCAN(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "min_cluster_size": getattr(args, "min_cluster_size", 5),
            "min_samples": getattr(args, "min_samples", None),
            "hdbscan_metric": getattr(args, "hdbscan_metric", "euclidean"),
            "cluster_selection_epsilon": getattr(args, "cluster_selection_epsilon", 0.0),
            "allow_single_cluster": getattr(args, "allow_single_cluster", False),
        })

    def execute(self):
        try:
            import hdbscan  # type: ignore
        except Exception as e:
            raise ImportError("HDBSCAN requires 'hdbscan' package (pip install hdbscan).") from e

        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        _, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")
        self.impute(data)
        X = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
        n, m = X.shape
        is_square = (n == m)

        mcs = int(self.options["min_cluster_size"])
        if mcs < 2:
            raise ValueError("--min_cluster_size must be >= 2")
        ms_opt = self.options["min_samples"]
        ms = None if ms_opt in (None, "None") else int(ms_opt)
        if ms is not None and ms < 1:
            raise ValueError("--min_samples must be >= 1 (or omit it)")
        metric = str(self.options["hdbscan_metric"]).lower()
        cse = float(self.options["cluster_selection_epsilon"])
        asc = bool(self.options["allow_single_cluster"])

        if is_square:
            dist_mat = X.copy()
            np.fill_diagonal(dist_mat, 0.0)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs, min_samples=ms, metric="precomputed",
                cluster_selection_epsilon=cse, allow_single_cluster=asc,
            )
            labels = clusterer.fit_predict(dist_mat)
            report_metric = "precomputed"
        else:
            Xp = self._apply_scaling(self._apply_embedding(self._apply_transform(X)))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs, min_samples=ms, metric=metric,
                cluster_selection_epsilon=cse, allow_single_cluster=asc,
            )
            labels = clusterer.fit_predict(Xp)
            report_metric = metric
            if metric == "jaccard":
                dist_mat = self.build_distance_matrix(Xp, "jaccard")
            elif metric == "braycurtis":
                dist_mat = self.build_distance_matrix(Xp, "braycurtis")
            else:
                dist_mat = self.build_distance_matrix(Xp, "euclidean")

        labels = np.asarray(labels, dtype=int)
        extra = [
            f"# min_cluster_size={mcs}", f"# min_samples={ms}", f"# metric={report_metric}",
            f"# cluster_selection_epsilon={cse}", f"# allow_single_cluster={asc}",
        ]
        self.emit_requested_reports(
            out_fmt=str(self.options["output_format"]).lower(), base_out_path=out_path,
            in_path=in_path, delimiter=delimiter, row_labels=[str(x) for x in row_labels],
            cluster_labels=labels, dist_mat=np.asarray(dist_mat, dtype=float),
            text_title="HDBSCAN clustering report", text_extra_lines=extra,
        )
        return out_path


# =============================================================================

class LEIDEN(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "knn_k": getattr(args, "knn_k", 30),
            "graph_metric": getattr(args, "graph_metric", "euclidean"),  # euclidean|jaccard|hamming
            "resolution": getattr(args, "resolution", 1.0),
            "from_distance": getattr(args, "from_distance", False),
        })

    def execute(self):
        import sys
        try:
            import igraph as ig  # type: ignore
            import leidenalg  # type: ignore
        except Exception as e:
            raise ImportError(
                "LEIDEN requires 'python-igraph' and 'leidenalg'. "
                "Install them (e.g., pip install python-igraph leidenalg)."
            ) from e

        from sklearn.neighbors import NearestNeighbors

        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        col_labels, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")
        self.impute(data)

        X = np.asarray(data, dtype=float)
        X = np.nan_to_num(X, nan=0.0)

        n, m = X.shape
        is_square = (n == m)
        out_fmt = str(self.options["output_format"]).lower()

        k = int(self.options["knn_k"])
        k = max(5, min(k, max(5, n - 1)))
        metric = str(self.options["graph_metric"]).lower()
        resolution = float(self.options["resolution"])
        from_distance = bool(self.options["from_distance"])

        if is_square and (from_distance or True):
            dist_mat = np.asarray(X, dtype=float)
            dist_mat = np.nan_to_num(dist_mat, nan=0.0)
            np.fill_diagonal(dist_mat, 0.0)

            nbrs = []
            weights = []
            for i in range(n):
                order = np.argsort(dist_mat[i])
                order = order[order != i][:k]
                for j in order:
                    d = float(dist_mat[i, j])
                    w = 1.0 / (1.0 + d)
                    nbrs.append((i, int(j)))
                    weights.append(w)
        else:
            Xf = self._apply_transform(X)
            Xf = self._apply_embedding(Xf)
            Xf = self._apply_scaling(Xf)

            if metric == "jaccard":
                Xb = (Xf != 0.0).astype(bool)
                nn = NearestNeighbors(n_neighbors=k + 1, metric="jaccard")
                nn.fit(Xb)
                dists, inds = nn.kneighbors(Xb)
            elif metric == "hamming":
                Xb = (Xf != 0.0).astype(int)
                nn = NearestNeighbors(n_neighbors=k + 1, metric="hamming")
                nn.fit(Xb)
                dists, inds = nn.kneighbors(Xb)
            else:
                nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
                nn.fit(Xf)
                dists, inds = nn.kneighbors(Xf)

            nbrs = []
            weights = []
            for i in range(n):
                for dd, j in zip(dists[i][1:], inds[i][1:]):
                    d = float(dd)
                    w = 1.0 / (1.0 + d)
                    nbrs.append((i, int(j)))
                    weights.append(w)

            if out_fmt in {"all", "newick"}:
                if metric == "jaccard":
                    dist_mat = self.build_distance_matrix(Xf, "jaccard")
                else:
                    dist_mat = self.build_distance_matrix(Xf, "euclidean")

        g = ig.Graph(n=n, edges=nbrs, directed=True)
        g.es["weight"] = weights
        g = g.as_undirected(combine_edges="max")

        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution
        )
        labels = np.asarray(part.membership, dtype=int)

        # A distance matrix is required by the Newick report and by "all".
        if "dist_mat" not in locals():
            if metric == "jaccard":
                dist_mat = self.build_distance_matrix(Xf, "jaccard")
            elif metric == "hamming":
                dist_mat = self.build_distance_matrix((Xf != 0.0).astype(float), "euclidean")
            else:
                dist_mat = self.build_distance_matrix(Xf, "euclidean")

        extra = [
            f"# knn_k={k}", f"# graph_metric={metric}",
            f"# resolution={resolution}", f"# graph_edges={g.ecount()}",
        ]
        self.emit_requested_reports(
            out_fmt=out_fmt, base_out_path=out_path, in_path=in_path, delimiter=delimiter,
            row_labels=[str(x) for x in row_labels], cluster_labels=labels,
            dist_mat=np.asarray(dist_mat, dtype=float), text_title="Leiden clustering report",
            text_extra_lines=extra,
        )

        return out_path

# =============================================================================
# MCL (Markov Clustering)
# =============================================================================

class MCL(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "knn_k": getattr(args, "knn_k", 30),
            "graph_metric": getattr(args, "graph_metric", "euclidean"),
            "inflation": getattr(args, "inflation", 2.0),
            "expansion": getattr(args, "expansion", 2),
            "prune_threshold": getattr(args, "prune_threshold", 1e-6),
            "max_iter": getattr(args, "mcl_max_iter", 100),
        })

    def _mcl(self, W: np.ndarray, expansion: int, inflation: float,
             prune_threshold: float, max_iter: int) -> List[List[int]]:
        """
        Simple MCL on adjacency-like matrix W (non-negative).
        """
        M = np.asarray(W, dtype=float).copy()
        n = M.shape[0]
        np.fill_diagonal(M, 1.0)

        # stochastic normalize columns
        def normalize(A: np.ndarray) -> np.ndarray:
            colsum = A.sum(axis=0, keepdims=True)
            colsum[colsum == 0.0] = 1.0
            return A / colsum

        M = normalize(M)

        for _ in range(max_iter):
            M_prev = M.copy()

            # expansion
            for _e in range(expansion - 1):
                M = M @ M

            # inflation
            M = np.power(M, inflation)

            # pruning
            M[M < prune_threshold] = 0.0

            # normalize
            M = normalize(M)

            # convergence (rough)
            if np.allclose(M, M_prev, atol=1e-6):
                break

        # Interpret clusters from attractors: nodes with strong self-loop neighborhoods
        # A practical heuristic: build graph of entries > 0 and take connected components.
        G = (M > 0).astype(int)
        visited = set()
        clusters: List[List[int]] = []

        for i in range(n):
            if i in visited:
                continue
            stack = [i]
            comp = []
            visited.add(i)
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = np.where(G[u] > 0)[0].tolist()
                neigh += np.where(G[:, u] > 0)[0].tolist()
                for v in neigh:
                    if v not in visited:
                        visited.add(v)
                        stack.append(int(v))
            clusters.append(sorted(comp))

        return clusters

    def execute(self):
        import sys
        from sklearn.neighbors import NearestNeighbors

        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        col_labels, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")
        self.impute(data)

        X = np.asarray(data, dtype=float)
        X = np.nan_to_num(X, nan=0.0)

        n, m = X.shape
        is_square = (n == m)
        out_fmt = str(self.options["output_format"]).lower()

        k = int(self.options["knn_k"])
        k = max(5, min(k, max(5, n - 1)))
        metric = str(self.options["graph_metric"]).lower()

        inflation = float(self.options["inflation"])
        expansion = int(self.options["expansion"])
        prune_threshold = float(self.options["prune_threshold"])
        max_iter = int(self.options["max_iter"])

        if is_square:
            dist_mat = np.asarray(X, dtype=float)
            dist_mat = np.nan_to_num(dist_mat, nan=0.0)
            np.fill_diagonal(dist_mat, 0.0)

            W = np.zeros((n, n), dtype=float)
            for i in range(n):
                order = np.argsort(dist_mat[i])
                order = order[order != i][:k]
                for j in order:
                    d = float(dist_mat[i, j])
                    w = 1.0 / (1.0 + d)
                    W[i, j] = max(W[i, j], w)
                    W[j, i] = max(W[j, i], w)

        else:
            Xf = self._apply_transform(X)
            Xf = self._apply_embedding(Xf)
            Xf = self._apply_scaling(Xf)

            if metric == "jaccard":
                Xb = (Xf != 0.0).astype(bool)
                nn = NearestNeighbors(n_neighbors=k + 1, metric="jaccard")
                nn.fit(Xb)
                dists, inds = nn.kneighbors(Xb)
            elif metric == "hamming":
                Xb = (Xf != 0.0).astype(int)
                nn = NearestNeighbors(n_neighbors=k + 1, metric="hamming")
                nn.fit(Xb)
                dists, inds = nn.kneighbors(Xb)
            else:
                nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
                nn.fit(Xf)
                dists, inds = nn.kneighbors(Xf)

            W = np.zeros((n, n), dtype=float)
            for i in range(n):
                for dd, j in zip(dists[i][1:], inds[i][1:]):
                    d = float(dd)
                    w = 1.0 / (1.0 + d)
                    W[i, int(j)] = max(W[i, int(j)], w)
                    W[int(j), i] = max(W[int(j), i], w)

            if metric == "jaccard":
                dist_mat = self.build_distance_matrix(Xf, "jaccard")
            elif metric == "hamming":
                dist_mat = self.build_distance_matrix((Xf != 0.0).astype(float), "euclidean")
            else:
                dist_mat = self.build_distance_matrix(Xf, "euclidean")

        clusters = self._mcl(W, expansion=expansion, inflation=inflation,
                             prune_threshold=prune_threshold, max_iter=max_iter)

        labels = np.full(n, -1, dtype=int)
        for cid, grp in enumerate(clusters):
            for i in grp:
                labels[int(i)] = cid

        extra = [
            f"# knn_k={k}", f"# graph_metric={metric}", f"# inflation={inflation}",
            f"# expansion={expansion}", f"# prune_threshold={prune_threshold}",
            f"# max_iter={max_iter}",
        ]
        self.emit_requested_reports(
            out_fmt=out_fmt, base_out_path=out_path, in_path=in_path, delimiter=delimiter,
            row_labels=[str(x) for x in row_labels], cluster_labels=labels,
            dist_mat=np.asarray(dist_mat, dtype=float), text_title="MCL clustering report",
            text_extra_lines=extra,
        )

        return out_path

# =============================================================================
# AGGLO (Agglomerative)
# =============================================================================

class AGGLO(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "agg_linkage": getattr(args, "agg_linkage", "average"),   # average|ward
            "agg_metric": getattr(args, "agg_metric", "euclidean"),   # euclidean|jaccard|braycurtis
        })

    def execute(self):
        import sys
        from sklearn.cluster import AgglomerativeClustering

        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        col_labels, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")
        self.impute(data)

        X = np.asarray(data, dtype=float)
        X = np.nan_to_num(X, nan=0.0)

        n, m = X.shape
        is_square = (n == m)
        out_fmt = str(self.options["output_format"]).lower()

        linkage = str(self.options["agg_linkage"]).lower()
        metric = str(self.options["agg_metric"]).lower()

        k = max(2, min(50, int(round(math.sqrt(n)))))

        labels = None  # set only when needed

        if is_square:
            dist_mat = np.asarray(X, dtype=float)
            dist_mat = np.nan_to_num(dist_mat, nan=0.0)
            np.fill_diagonal(dist_mat, 0.0)

            if out_fmt in {"all", "text", "matrix", "pathway"}:
                try:
                    model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed")
                except TypeError:
                    model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="precomputed")
                labels = model.fit_predict(dist_mat)

        else:
            Xf = self._apply_transform(X)
            Xf = self._apply_embedding(Xf)
            Xf = self._apply_scaling(Xf)

            if metric == "jaccard":
                dist_mat = self.build_distance_matrix(Xf, "jaccard")
                if out_fmt in {"all", "text", "matrix", "pathway"}:
                    try:
                        model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed")
                    except TypeError:
                        model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="precomputed")
                    labels = model.fit_predict(dist_mat)

            elif metric == "braycurtis":
                dist_mat = self.build_distance_matrix(Xf, "braycurtis")
                if out_fmt in {"all", "text", "matrix", "pathway"}:
                    try:
                        model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed")
                    except TypeError:
                        model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="precomputed")
                    labels = model.fit_predict(dist_mat)

            else:
                dist_mat = self.build_distance_matrix(Xf, "euclidean")
                if out_fmt in {"all", "text", "matrix", "pathway"}:
                    if linkage == "ward":
                        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                        labels = model.fit_predict(Xf)
                    else:
                        try:
                            model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean")
                        except TypeError:
                            model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean")
                        labels = model.fit_predict(Xf)

        if labels is None:
            # Only a pure Newick request can reach here without cluster labels.
            # For consistency, compute labels as well when any label-based report is requested.
            if out_fmt != "newick":
                raise RuntimeError("Agglomerative cluster labels were not calculated.")
            labels = np.zeros(n, dtype=int)

        extra = [
            f"# agg_linkage={linkage}", f"# agg_metric={metric}",
            f"# n_clusters(heuristic)={k}",
        ]
        self.emit_requested_reports(
            out_fmt=out_fmt, base_out_path=out_path, in_path=in_path, delimiter=delimiter,
            row_labels=[str(x) for x in row_labels], cluster_labels=np.asarray(labels, dtype=int),
            dist_mat=np.asarray(dist_mat, dtype=float), text_title="Agglomerative clustering report",
            text_extra_lines=extra,
        )

        return out_path

# =============================================================================
# KMEANS
# =============================================================================

class KMEANS(Basic):
    def __init__(self, args: Any):
        super().__init__(args)
        self.options.update({
            "kmeans_k": getattr(args, "kmeans_k", None),
            "n_init": getattr(args, "n_init", 10),
            "max_iter": getattr(args, "max_iter", 300),
        })

    def execute(self):
        import sys
        from sklearn.cluster import KMeans

        in_path, out_path, delimiter, is_tsv = self.resolve_paths()
        col_labels, row_labels, data = self.read_labeled_matrix(in_path, delimiter)
        if not data or not data[0]:
            raise ValueError("Empty matrix in table.")
        self.impute(data)

        X = np.asarray(data, dtype=float)
        X = np.nan_to_num(X, nan=0.0)

        n, m = X.shape
        is_square = (n == m)
        out_fmt = str(self.options["output_format"]).lower()

        # Feature space
        Xf = X
        Xf = self._apply_transform(Xf)
        Xf = self._apply_embedding(Xf)
        Xf = self._apply_scaling(Xf)

        k = self.options["kmeans_k"]
        if k is None:
            k = max(2, min(50, int(round(math.sqrt(n)))))
        k = int(k)

        model = KMeans(
            n_clusters=k,
            n_init=int(self.options["n_init"]),
            max_iter=int(self.options["max_iter"]),
            random_state=int(self.options.get("seed", 13)),
        )
        labels = np.asarray(model.fit_predict(Xf), dtype=int)

        dist_mat = self.build_distance_matrix(Xf, "euclidean")

        extra = [
            f"# kmeans_k={k}", f"# n_init={int(self.options['n_init'])}",
            f"# max_iter={int(self.options['max_iter'])}",
            f"# scale={bool(self.options.get('scale', False))}",
        ]
        self.emit_requested_reports(
            out_fmt=out_fmt, base_out_path=out_path, in_path=in_path, delimiter=delimiter,
            row_labels=[str(x) for x in row_labels], cluster_labels=labels,
            dist_mat=dist_mat, text_title="KMeans clustering report",
            text_extra_lines=extra,
        )

        return out_path
