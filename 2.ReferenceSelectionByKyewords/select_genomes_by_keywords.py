#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_SUFFIXES = (
    ".fna.gz", ".fasta.gz", ".fa.gz", ".fst.gz", ".fas.gz",
    ".gbff.gz", ".gbk.gz", ".gbf.gz", ".gb.gz",
    ".fna", ".fasta", ".fa", ".fst", ".fas",
    ".gbff", ".gbk", ".gbf", ".gb",
)

# Prefer compact FASTA files when several representations of one assembly exist.
# This matches the intended list output, e.g. GCA_..._genomic.fna.gz.
FILE_PRIORITY = {
    ".fna.gz": 0, ".fasta.gz": 1, ".fa.gz": 2, ".fst.gz": 3, ".fas.gz": 4,
    ".gbff.gz": 5, ".gbk.gz": 6, ".gbf.gz": 7, ".gb.gz": 8,
    ".fna": 9, ".fasta": 10, ".fa": 11, ".fst": 12, ".fas": 13,
    ".gbff": 14, ".gbk": 15, ".gbf": 16, ".gb": 17,
}

ASSEMBLY_RE = re.compile(r"(GC[AF]_\d+\.\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class KeywordRule:
    categories: Tuple[str, ...]
    metadata_fields: Tuple[str, ...]
    match_type: str
    keyword: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative genomes by metadata-defined hierarchical categories. "
            "Category/keyword rules are read from a user-editable CSV/TSV/TXT file."
        )
    )

    parser.add_argument(
        "project",
        help="Project subfolder inside --input_folder and --output_folder."
    )
    parser.add_argument(
        "-i", "--input_folder",
        default="input",
        help="Input root folder (default: input)."
    )
    parser.add_argument(
        "-o", "--output_folder",
        default="output",
        help="Output root folder (default: output)."
    )
    parser.add_argument(
        "-s", "--source_folder",
        required=True,
        help=(
            "Folder containing genome files in GenBank/FASTA format, optionally gzip-compressed. "
            "The folder is searched recursively."
        )
    )
    parser.add_argument(
        "-f", "--metadata_file",
        required=True,
        help="CSV/TSV/TXT metadata file located in ./input/PROJECT."
    )
    parser.add_argument(
        "-k", "--keywords",
        default="keywords.tsv",
        help=(
            "CSV/TSV/TXT keyword-rule file located in ./input/PROJECT "
            "(default: keywords.tsv)."
        )
    )
    parser.add_argument(
        "-t", "--output_format",
        choices=["folders", "list"],
        default="folders",
        help=(
            "Output mode: 'folders' copies selected genomes into hierarchical folders; "
            "'list' writes hierarchical genome headings only (default: folders)."
        )
    )
    parser.add_argument(
        "-d", "--level_delimiter",
        default="|",
        help="Delimiter used between hierarchy levels in list output (default: '|')."
    )
    parser.add_argument(
        "-L", "--output_list_file",
        default="selected_genomes.txt",
        help="List-output filename inside ./output/PROJECT (default: selected_genomes.txt)."
    )
    parser.add_argument(
        "-n", "--genomes_per_category",
        type=int,
        default=5,
        help="Maximum genomes selected per terminal category (default: 5)."
    )
    
    parser.add_argument(
        "-r", "--random_seed",
        type=int,
        default=41,
        help="Random seed used for reproducible subsampling (default: 41)."
    )

    args = parser.parse_args()

    if args.genomes_per_category < 1:
        parser.error("--genomes_per_category must be >= 1")
    if not args.level_delimiter:
        parser.error("--level_delimiter cannot be empty")

    return args


def detect_delimiter(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".tsv", ".txt"}:
        return "\t"
    if ext == ".csv":
        return ","
    raise ValueError(
        f"Unsupported table extension for {path}. Expected .tsv, .txt, or .csv."
    )


def read_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        rows = [
            [cell.strip() for cell in row]
            for row in reader
            if row and any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError(f"Input table is empty: {path}")

    header = rows[0]
    body = rows[1:]
    return header, body


def load_keyword_rules(path: Path, metadata_header: Sequence[str]) -> List[KeywordRule]:
    """
    Read keyword rules from a tidy table.

    Required columns:
        Category_1 ... Category_N
            One or more hierarchical category columns. Category_1 is required.

        Metadata_fields
            One or more metadata column names separated by semicolons.
            Fields are evaluated using OR logic.

        Match_type
            "exact" or "contains".

        Keyword
            One keyword per row.

    All rules have equal importance. There is no priority field.
    Every matching rule contributes its category path.
    """
    header, rows = read_table(path)
    header_map = {name.strip().lower(): i for i, name in enumerate(header)}

    category_cols: List[Tuple[int, int]] = []
    for idx, name in enumerate(header):
        match = re.fullmatch(r"category_(\d+)", name.strip(), flags=re.IGNORECASE)
        if match:
            category_cols.append((int(match.group(1)), idx))
    category_cols.sort()

    if not category_cols or category_cols[0][0] != 1:
        raise ValueError(
            "Keyword file must contain hierarchy columns named Category_1, Category_2, ..."
        )

    required = ["metadata_fields", "match_type", "keyword"]
    missing = [name for name in required if name not in header_map]
    if missing:
        raise ValueError(
            f"Keyword file is missing required column(s): {', '.join(missing)}"
        )

    metadata_names = set(metadata_header)
    rules: List[KeywordRule] = []

    for row_number, row in enumerate(rows, start=2):
        row = row + [""] * max(0, len(header) - len(row))

        raw_categories = [row[idx].strip() for _, idx in category_cols]

        seen_empty = False
        for value in raw_categories:
            if not value:
                seen_empty = True
            elif seen_empty:
                raise ValueError(
                    f"Keyword row {row_number} has a gap in hierarchical categories."
                )

        categories = tuple(value for value in raw_categories if value)
        if not categories:
            continue

        fields_text = row[header_map["metadata_fields"]].strip()
        metadata_fields = tuple(
            field.strip()
            for field in fields_text.split(";")
            if field.strip()
        )
        if not metadata_fields:
            raise ValueError(
                f"Keyword row {row_number} has no Metadata_fields value."
            )

        unknown_fields = [
            field
            for field in metadata_fields
            if field not in metadata_names
        ]
        if unknown_fields:
            raise ValueError(
                f"Keyword row {row_number} refers to metadata field(s) not present in the "
                f"metadata table: {', '.join(unknown_fields)}"
            )

        match_type = row[header_map["match_type"]].strip().lower() or "exact"
        if match_type not in {"exact", "contains"}:
            raise ValueError(
                f"Keyword row {row_number}: Match_type must be 'exact' or 'contains'."
            )

        keyword = row[header_map["keyword"]].strip().lower()
        if not keyword:
            continue

        rules.append(
            KeywordRule(
                categories=categories,
                metadata_fields=metadata_fields,
                match_type=match_type,
                keyword=keyword,
            )
        )

    if not rules:
        raise ValueError(f"No usable keyword rules were found in {path}")

    return rules

def classify_metadata_record(
    record: Dict[str, str],
    rules: Sequence[KeywordRule],
) -> List[Tuple[str, ...]]:
    """
    Return all category paths matched by the metadata record.

    All keyword rules have equal importance. If at least one keyword assigned
    to a category matches at least one of its configured metadata fields,
    that category is assigned.

    Multiple matching keywords for the same category do not duplicate the
    category. If keywords from different categories match, the genome may
    be assigned to more than one category.
    """
    matched_categories: set[Tuple[str, ...]] = set()

    for rule in rules:
        matched = False

        for field in rule.metadata_fields:
            value = str(record.get(field, "")).strip().lower()
            if not value:
                continue

            if rule.match_type == "exact":
                matched = value == rule.keyword
            else:
                matched = rule.keyword in value

            if matched:
                break

        if matched:
            matched_categories.add(rule.categories)

    return sorted(matched_categories)

def supported_suffix(name: str) -> Optional[str]:
    lower = name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower.endswith(suffix):
            return suffix
    return None


def file_priority(path: Path) -> Tuple[int, str]:
    suffix = supported_suffix(path.name)
    priority = FILE_PRIORITY.get(suffix or "", 999)
    return priority, path.name.lower()


def index_genome_files(source_folder: Path) -> Dict[str, Path]:
    """
    Recursively index supported genome files by GCA/GCF assembly accession.
    If several representations exist, prefer compressed FASTA, then GenBank.
    """
    candidates: Dict[str, List[Path]] = {}

    for path in source_folder.rglob("*"):
        if not path.is_file() or supported_suffix(path.name) is None:
            continue

        match = ASSEMBLY_RE.search(path.name)
        if match is None:
            # Also allow an accession-named parent directory.
            match = ASSEMBLY_RE.search(path.parent.name)
        if match is None:
            continue

        accession = match.group(1).upper()
        candidates.setdefault(accession, []).append(path)

    return {
        accession: sorted(paths, key=file_priority)[0]
        for accession, paths in candidates.items()
    }


def average_file_size(paths: Iterable[Path]) -> Tuple[int, int]:
    sizes: List[int] = []
    for path in paths:
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            continue
    if not sizes:
        return 0, 0
    return int(sum(sizes) / len(sizes)), len(sizes)


def select_genomes(
    candidates: Sequence[Tuple[str, Path]],
    number_to_select: int,
    min_file_size: int,
    max_file_size: int,
    rng: random.Random,
) -> List[Tuple[str, Path]]:
    """
    Prefer files within the global ±20% average-size range, then fill from the
    remaining category members if too few pass the size criterion.
    """
    candidates = list(candidates)
    if len(candidates) <= number_to_select:
        return candidates

    eligible: List[Tuple[str, Path]] = []
    fallback: List[Tuple[str, Path]] = []

    for item in candidates:
        try:
            size = item[1].stat().st_size
        except OSError:
            continue
        if min_file_size <= size <= max_file_size:
            eligible.append(item)
        else:
            fallback.append(item)

    rng.shuffle(eligible)
    rng.shuffle(fallback)

    selected = eligible[:number_to_select]
    if len(selected) < number_to_select:
        selected.extend(fallback[: number_to_select - len(selected)])
    return selected


def safe_folder_name(text: str) -> str:
    """Replace characters forbidden in Windows filenames while remaining Linux-safe."""
    text = re.sub(r'[<>:"/\\|?*]', "_", text.strip())
    return text.rstrip(". ") or "Unclassified"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)

    input_project = Path(args.input_folder) / args.project
    output_project = Path(args.output_folder) / args.project
    source_folder = Path(args.source_folder).expanduser().resolve()

    if not input_project.is_dir():
        raise FileNotFoundError(f"Input project folder does not exist: {input_project}")
    if not source_folder.is_dir():
        raise FileNotFoundError(f"Genome source folder does not exist: {source_folder}")

    metadata_path = input_project / args.metadata_file
    keywords_path = input_project / args.keywords

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    if not keywords_path.is_file():
        raise FileNotFoundError(f"Keyword file does not exist: {keywords_path}")

    output_project.mkdir(parents=True, exist_ok=True)

    metadata_header, metadata_rows = read_table(metadata_path)
    if "assembly_accession" not in metadata_header:
        raise ValueError(
            "Metadata file must contain the column 'assembly_accession' so records can be "
            "matched to local genome files."
        )

    rules = load_keyword_rules(keywords_path, metadata_header)
    genome_index = index_genome_files(source_folder)
    if not genome_index:
        raise ValueError(
            f"No supported genome files with GCA/GCF accessions were found under {source_folder}"
        )

    metadata_idx = {name: i for i, name in enumerate(metadata_header)}
    accession_col = metadata_idx["assembly_accession"]

    categorized: Dict[Tuple[str, ...], List[Tuple[str, Path]]] = {}
    classified_records = 0
    missing_genome_files = 0

    for row in metadata_rows:
        if len(row) < len(metadata_header):
            row = row + [""] * (len(metadata_header) - len(row))

        accession = row[accession_col].strip().upper()
        if not accession:
            continue

        genome_path = genome_index.get(accession)
        if genome_path is None:
            missing_genome_files += 1
            continue

        record = {
            name: row[i] if i < len(row) else ""
            for i, name in enumerate(metadata_header)
        }

        category_paths = classify_metadata_record(record, rules)
        if not category_paths:
            continue

        for category_path in category_paths:
            categorized.setdefault(category_path, []).append(
                (accession, genome_path)
            )

        classified_records += 1

    if not categorized:
        raise ValueError(
            "No metadata records matched both a local genome file and a keyword rule."
        )

    avg_size, n_sized_files = average_file_size(genome_index.values())
    min_file_size = int(avg_size * 0.8) if avg_size else 0
    max_file_size = int(avg_size * 1.2) if avg_size else sys.maxsize

    print(f"Metadata records: {len(metadata_rows):,}")
    print(f"Genome files indexed: {len(genome_index):,}")
    print(f"Metadata records assigned to at least one category: {classified_records:,}")
    print(f"Metadata records without a local genome file: {missing_genome_files:,}")
    print(f"Terminal categories represented: {len(categorized):,}")
    if avg_size:
        print(
            f"Preferred genome-file size range: {min_file_size:,} .. {max_file_size:,} bytes "
            f"(global mean {avg_size:,}; n={n_sized_files:,})"
        )

    selected_all: List[Tuple[Tuple[str, ...], str, Path]] = []

    for category_path in sorted(categorized):
        candidates = categorized[category_path]
        selected = select_genomes(
            candidates=candidates,
            number_to_select=args.genomes_per_category,
            min_file_size=min_file_size,
            max_file_size=max_file_size,
            rng=rng,
        )

        print(
            f"{' / '.join(category_path)}: "
            f"{len(candidates)} candidate(s), {len(selected)} selected"
        )

        for accession, genome_path in selected:
            selected_all.append((category_path, accession, genome_path))

    if args.output_format == "folders":
        copied = 0
        for category_path, _accession, genome_path in selected_all:
            destination = output_project.joinpath(*(safe_folder_name(x) for x in category_path))
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(genome_path, destination / genome_path.name)
            copied += 1
        print(f"Copied {copied:,} selected genome file(s) to: {output_project}")

    else:  # list
        list_path = output_project / args.output_list_file
        with list_path.open("w", encoding="utf-8", newline="\n") as fp:
            for category_path, _accession, genome_path in selected_all:
                heading = args.level_delimiter.join((*category_path, genome_path.name))
                fp.write(heading + "\n")
        print(f"Saved {len(selected_all):,} selected genome heading(s) to: {list_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
