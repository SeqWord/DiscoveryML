import os
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import progressbar


def _read_table_auto(file_path: str) -> pd.DataFrame:
    """
    Read a CSV/TSV-like table, keeping all cells as strings/symbols.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".tsv", ".tab"):
        return pd.read_csv(file_path, sep="\t", dtype=str)
    if ext == ".csv":
        return pd.read_csv(file_path, sep=",", dtype=str)

    # Fallback: sniff the table delimiter.  The hierarchy delimiter is NOT
    # used here; it applies only inside the first column.
    with open(file_path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
            sep = dialect.delimiter
        except Exception:
            sep = "\t"

        return pd.read_csv(fh, sep=sep, dtype=str)


def _split_row_headings(
    headings: pd.Series,
    delimiter: str = "|",
) -> Tuple[pd.Series, pd.DataFrame, List[pd.Series]]:
    """
    Split the first matrix column into hierarchical labels and a unique row ID.

    A heading such as
        Environment|Environmental swabs|GCA_000001.1
    is interpreted as
        level 1 = Environment
        level 2 = Environmental swabs
        row ID  = GCA_000001.1

    Rows may have different hierarchy depths.  Missing deeper levels are padded
    on the RIGHT with pd.NA so that Level 1 always means the same thing.

    Returns
    -------
    row_ids
        Unique genome/strain identifiers (last item after splitting).
    level_df
        Rectangular DataFrame containing raw hierarchical levels.
    cumulative_labels
        One Series per level.  Level n is represented by its full hierarchy
        path, e.g. "Environment|Environmental swabs".  This prevents equal
        child names under different parents from being merged.
    """
    if not delimiter:
        raise ValueError("delimiter must be a non-empty string")

    cleaned = headings.astype("string").str.strip()
    valid = cleaned.notna() & cleaned.ne("")
    if not valid.all():
        bad = headings.index[~valid].tolist()
        raise ValueError(
            f"The first column contains {len(bad)} empty row heading(s); "
            "each row must contain hierarchical label(s) and a unique row ID."
        )

    split_rows: List[List[str]] = []
    row_ids: List[str] = []
    hierarchy_rows: List[List[str]] = []

    for value in cleaned.tolist():
        parts = [part.strip() for part in str(value).split(delimiter)]

        if len(parts) < 2:
            raise ValueError(
                f"Row heading {value!r} does not contain delimiter {delimiter!r}. "
                "Expected at least one label followed by a unique row ID."
            )

        row_id = parts[-1]
        levels = parts[:-1]

        if row_id == "":
            raise ValueError(f"Row heading {value!r} has an empty unique row ID.")
        if any(level == "" for level in levels):
            # Empty internal levels are ambiguous.  Variable depth should be
            # expressed by simply omitting deeper levels, not by 'A||ID'.
            raise ValueError(
                f"Row heading {value!r} contains an empty hierarchy level."
            )

        split_rows.append(parts)
        row_ids.append(row_id)
        hierarchy_rows.append(levels)

    row_id_series = pd.Series(row_ids, index=headings.index, dtype="string", name="Row_ID")

    duplicated = row_id_series[row_id_series.duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().tolist()[:10]
        raise ValueError(
            "Unique genome/strain identifiers are duplicated in the first column: "
            + ", ".join(map(str, examples))
        )

    max_levels = max(len(levels) for levels in hierarchy_rows)
    padded = [levels + [pd.NA] * (max_levels - len(levels)) for levels in hierarchy_rows]
    level_df = pd.DataFrame(
        padded,
        index=headings.index,
        columns=[f"Label_level_{i}" for i in range(1, max_levels + 1)],
        dtype="string",
    )

    cumulative_labels: List[pd.Series] = []
    for level_idx in range(max_levels):
        labels_for_level = []
        for levels in hierarchy_rows:
            if len(levels) <= level_idx:
                labels_for_level.append(pd.NA)
            else:
                labels_for_level.append(delimiter.join(levels[: level_idx + 1]))

        cumulative_labels.append(
            pd.Series(
                labels_for_level,
                index=headings.index,
                dtype="string",
                name=f"Hierarchy_level_{level_idx + 1}",
            )
        )

    return row_id_series, level_df, cumulative_labels


def _chi2_for_level(
    data: pd.DataFrame,
    labels: pd.Series,
    level_number: int,
) -> pd.DataFrame:
    """Run a chi-square test for every feature at one hierarchy level."""
    valid_label_mask = labels.notna() & labels.ne("")
    level_labels = labels.loc[valid_label_mask]

    results = []
    for i, col in enumerate(data.columns):
        feature = data.loc[valid_label_mask, col]

        # Missing feature values are excluded rather than treated as a symbol.
        valid_feature_mask = feature.notna() & feature.ne("")
        y = level_labels.loc[valid_feature_mask]
        x = feature.loc[valid_feature_mask]

        contingency_table = pd.crosstab(y, x)
        n_groups = contingency_table.shape[0]
        n_states = contingency_table.shape[1]
        n_samples = int(contingency_table.to_numpy().sum()) if not contingency_table.empty else 0

        chi2 = np.nan
        p = np.nan
        dof = np.nan

        # A chi-square test requires at least two groups and two feature states.
        if n_groups >= 2 and n_states >= 2:
            try:
                chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
            except ValueError:
                pass

        results.append(
            (
                level_number,
                i + 1,
                col,
                n_samples,
                n_groups,
                n_states,
                chi2,
                dof,
                p,
            )
        )

    result_df = pd.DataFrame(
        results,
        columns=[
            "Level",
            "Column",
            "Location",
            "Samples",
            "Groups",
            "States",
            "Chi-square",
            "dof",
            "p-value",
        ],
    )

    # FDR correction across features WITHIN this hierarchy level.
    result_df["Level FDR p-value"] = np.nan
    valid = result_df["p-value"].notna()
    if valid.any():
        _, corrected, _, _ = multipletests(
            result_df.loc[valid, "p-value"].to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )
        result_df.loc[valid, "Level FDR p-value"] = corrected

    return result_df

def _remove_empty_columns(dm: pd.DataFrame, thr: float, empty_symbol: str) -> pd.DataFrame:
    """
    Remove feature columns (starting from 3rd column) where
    fraction of empty cells > thr.

    Empty is defined STRICTLY by string comparison:
      - NaN
      - stripped value == ""
      - stripped value == empty_symbol (as STRING, even if '0')
    """

    if dm.shape[1] < 3:
        return dm

    empty_sym = str(empty_symbol).strip()
    feature_cols = dm.columns[2:]

    keep_cols = list(dm.columns[:2])  # first two columns always preserved
    removed = []

    for c in feature_cols:
        s = dm[c]

        is_empty = s.isna()
        s_str = s.astype(str).str.strip()

        is_empty |= (s_str == "")
        if empty_sym != "":
            is_empty |= (s_str == empty_sym)

        frac_empty = is_empty.mean()
        if frac_empty <= thr:
            keep_cols.append(c)
        else:
            removed.append((c, frac_empty))

    if removed:
        msg = ", ".join(f"{c}({p:.2f})" for c, p in removed[:10])
        extra = "" if len(removed) <= 10 else f" ... +{len(removed)-10} more"
        print(f"⚠️  Removed {len(removed)} column(s) with empty fraction > {thr}: {msg}{extra}")

    return dm.loc[:, keep_cols].copy()


def main(
    file_path,
    outpath,
    empty_symbol: str = "",
    thr: float = 0,
    basename="",
    p_value=None,
    significance=None,
    delimiter="|",
):
    """
    Select symbolic features associated with at least one hierarchical label level.

    The first input column must contain one or more hierarchical labels followed
    by a unique strain/genome ID, separated by `delimiter`, for example:

        Environment|Environmental swabs|GCA_053410975.1_genomic.gbff.gz

    All remaining columns are treated as symbolic features.

    FDR strategy
    ------------
    1. `Level FDR p-value` is Benjamini-Hochberg correction across all features
       tested at a particular hierarchy level.
    2. `Global FDR p-value` is Benjamini-Hochberg correction across ALL valid
       feature x hierarchy-level tests.  This is the value used to decide which
       features are retained in the filtered matrix, because the selection rule
       is "significant at at least one hierarchy level".
    """
    df = _read_table_auto(file_path)
    
    # Remove incomple rows
    if thr < 1.0:
        df = _remove_empty_columns(df, thr, empty_symbol)


    if df.shape[1] < 2:
        raise ValueError(
            "Input matrix must contain a row-heading column and at least one feature column."
        )

    # Remove rows with missing/blank first-column headings.
    heading_col = df.columns[0]
    heading_valid = df.iloc[:, 0].notna() & df.iloc[:, 0].astype(str).str.strip().ne("")
    removed = int((~heading_valid).sum())
    if removed:
        print(f"Removed {removed} row(s) with missing/blank row headings.")
    df = df.loc[heading_valid].copy()

    # Parse the single combined heading column.
    row_ids, level_df, cumulative_labels = _split_row_headings(
        df.iloc[:, 0], delimiter=delimiter
    )

    # All columns after the first are features.  Keep symbols as strings, but
    # preserve missing cells as pd.NA so they are not treated as a category.
    data = df.iloc[:, 1:].astype("string")
    bar = progressbar.indicator(len(data.columns), "Score features...")
    counter = 0
    for col in data.columns:
        data[col] = data[col].str.strip()
        data.loc[data[col].eq(""), col] = pd.NA
        counter += 1
        bar(counter)
    bar.stop()

    print(f"Input matrix shape: {data.shape} (Rows x feature columns)")
    print(f"Heading delimiter: {delimiter!r}")
    print(f"Detected hierarchy levels: {level_df.shape[1]}")
    print(f"Unique row IDs: {row_ids.nunique()}")

    # Report group counts and available samples at each hierarchy level.
    for i, labels in enumerate(cumulative_labels, start=1):
        valid = labels.notna() & labels.ne("")
        print(
            f"  Level {i}: {labels.loc[valid].nunique()} groups; "
            f"{int(valid.sum())} rows available"
        )

    # Analyze each hierarchy level independently.
    level_results = []
    for i, labels in enumerate(cumulative_labels, start=1):
        valid = labels.notna() & labels.ne("")
        if labels.loc[valid].nunique() < 2:
            print(f"Skipping Level {i}: fewer than 2 groups.")
            continue
        level_results.append(_chi2_for_level(data, labels, i))

    if not level_results:
        raise ValueError("No hierarchy level contains at least two groups for chi-square analysis.")

    result_df = pd.concat(level_results, ignore_index=True)

    # Global FDR correction across every feature x level hypothesis.  This is
    # important when the final rule is to keep a feature significant at ANY level.
    result_df["Global FDR p-value"] = np.nan
    valid_global = result_df["p-value"].notna()
    if valid_global.any():
        _, global_fdr, _, _ = multipletests(
            result_df.loc[valid_global, "p-value"].to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )
        result_df.loc[valid_global, "Global FDR p-value"] = global_fdr

    result_df["Significant (global FDR<0.05)"] = (
        result_df["Global FDR p-value"].notna()
        & (result_df["Global FDR p-value"] < 0.05)
    )

    result_df = result_df.sort_values(
        ["Global FDR p-value", "p-value", "Level"],
        na_position="last",
    ).reset_index(drop=True)

    # Optional filtering affects the statistics table only.
    filtered_df = result_df.copy()
    if p_value is not None:
        filtered_df = filtered_df[filtered_df["p-value"] <= p_value]
    if significance:
        filtered_df = filtered_df[filtered_df["Significant (global FDR<0.05)"]]

    if p_value is not None or significance:
        print(filtered_df)
    else:
        print(filtered_df.head(10))

    # Keep every feature that is significant at AT LEAST ONE hierarchy level.
    significant_locations = (
        result_df.loc[
            result_df["Significant (global FDR<0.05)"], "Location"
        ]
        .drop_duplicates()
        .tolist()
    )

    # A compact per-feature summary is useful for seeing which level(s) selected it.
    significant_rows = result_df[result_df["Significant (global FDR<0.05)"]].copy()
    if significant_rows.empty:
        feature_summary = pd.DataFrame(
            columns=[
                "Location",
                "Best p-value",
                "Best Level FDR p-value",
                "Best Global FDR p-value",
                "Significant levels",
            ]
        )
    else:
        feature_summary = (
            significant_rows.groupby("Location", sort=False)
            .agg(
                **{
                    "Best p-value": ("p-value", "min"),
                    "Best Level FDR p-value": ("Level FDR p-value", "min"),
                    "Best Global FDR p-value": ("Global FDR p-value", "min"),
                    "Significant levels": (
                        "Level",
                        lambda s: "|".join(map(str, sorted(set(s.astype(int))))),
                    ),
                }
            )
            .reset_index()
            .sort_values("Best Global FDR p-value")
        )

    # Preserve the original combined heading column in the filtered matrix so
    # downstream programs can still parse the hierarchy from the same format.
    cols_to_keep = [df.columns[0]] + significant_locations
    filtered_matrix = df.loc[:, cols_to_keep].copy()

    os.makedirs(outpath, exist_ok=True)
    basename_prefix = basename + "_" if basename else ""

    out_file_stats = os.path.join(outpath, basename_prefix + "chi2_fdr_results.csv")
    filtered_df.to_csv(out_file_stats, index=False)

    out_file_summary = os.path.join(outpath, basename_prefix + "significant_features.csv")
    feature_summary.to_csv(out_file_summary, index=False)

    out_file_matrix = os.path.join(outpath, basename_prefix + "filtered_matrix.csv")
    filtered_matrix.to_csv(out_file_matrix, index=False)

    # Also save the parsed hierarchy explicitly for inspection/debugging.
    parsed_labels = pd.concat([row_ids, level_df], axis=1)
    out_file_labels = os.path.join(outpath, basename_prefix + "parsed_labels.csv")
    parsed_labels.to_csv(out_file_labels, index=False)

    print("Chi-square analysis with multi-level FDR correction completed.")
    print(f"Results table saved to: {out_file_stats}")
    print(f"Significant-feature summary saved to: {out_file_summary}")
    print(f"Parsed hierarchical labels saved to: {out_file_labels}")
    print(
        f"Matrix with non-significant columns removed saved to: {out_file_matrix} "
        f"({len(significant_locations)} feature(s) retained)"
    )

    return filtered_df, filtered_matrix
