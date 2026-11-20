#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Sequence, Dict, Union
import math, random
import numpy as np
import pandas as pd

# (networkx is not required by the core here; keep import only if you use it elsewhere)
# import networkx as nx

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
