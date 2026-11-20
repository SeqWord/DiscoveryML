#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import os
import shutil
import subprocess
import sys
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Bio import SeqIO


date_of_creation = "August 16, 2026"
__version__ = "1.3"

DNA_ALPHABET = set("ACGT")
GENBANK_EXTENSIONS = (".gb", ".gbk", ".gbf", ".gbff")
FASTA_EXTENSIONS = (".fa", ".fasta", ".fst", ".fas", ".fna")
SUPPORTED_GENOME_EXTENSIONS = GENBANK_EXTENSIONS + FASTA_EXTENSIONS

ASSEMBLY_RE = re.compile(r"(GC[AF]_\d+\.\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class KeywordRule:
    """One metadata keyword rule; all rules have equal importance."""
    categories: Tuple[str, ...]
    metadata_fields: Tuple[str, ...]
    match_type: str
    keyword: str


# -----------------------------------------------------------------------------
# Locate optional local library directory without depending on the current CWD.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LIB_CANDIDATES = [
    SCRIPT_DIR / "lib",
    SCRIPT_DIR.parent / "lib",
    Path.cwd() / "lib",
    Path.cwd().parent / "lib",
]
LIB_DIR: Optional[Path] = next((p for p in LIB_CANDIDATES if p.is_dir()), None)
if LIB_DIR is not None and str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))

try:
    import blast as custom_blast  # type: ignore
except Exception:
    custom_blast = None


########################################################################
#   ██████╗  ██╗    ██╗  ██████╗    ██████╗   █████╗   ██╗       ██╗
#  ██╔════╝  ███╗   ██║  ██╔══██╗  ██╔════╝  ██╔══██╗  ██║       ██║
#  ╚█████╗   ██╔██╗ ██║  ██████╔╝  ██║       ███████║  ██║       ██║
#   ╚═══██╗  ██║╚██╗██║  ██╔═══╝   ██║       ██╔══██║  ██║       ██║
#  ██████╔╝  ██║ ╚████║  ██║       ╚██████╗  ██║  ██║  ███████╗  ███████╗
# SNP calling pipeline
########################################################################

def show_banner() -> None:
    print(r"""
   ██████╗  ██╗    ██╗  ██████╗    ██████╗   █████╗   ██╗       ██╗
  ██╔════╝  ███╗   ██║  ██╔══██╗  ██╔════╝  ██╔══██╗  ██║       ██║
  ╚█████╗   ██╔██╗ ██║  ██████╔╝  ██║       ███████║  ██║       ██║
   ╚═══██╗  ██║╚██╗██║  ██╔═══╝   ██║       ██╔══██║  ██║       ██║
  ██████╔╝  ██║ ╚████║  ██║       ╚██████╗  ██║  ██║  ███████╗  ███████╗

    SNP calling pipeline
    """)


def show_help() -> None:
    help_text = f"""
    {'=' * 80}
    ■ SNPCALL (Polymorphic Site Caller) v{__version__} ■
    {'=' * 80}
    📅 Revised: {date_of_creation}
    👨‍💻 Author: Oleg Reva (oleg.reva@up.ac.za)
       Centre for Bioinformatics and Computational Biology,
       University of Pretoria, South Africa

    🔬 Purpose:
    Create character, digit, or probe-state matrices from genome sequences.

    🚀 Usage:
        python3 feature_caller.py PROJECT INPUT_FILE OUTPUT_FILE -s SOURCE_FOLDER [options]

    Positional arguments:
        PROJECT       Project subfolder inside --input_folder/--output_folder/--tmp_folder
        INPUT_FILE    Feature annotation/probe file inside input/PROJECT
        OUTPUT_FILE   Output CSV/TSV/TXT matrix name

    🔧 Options:
        -s, --source_folder DIR       Directory containing genome FASTA/GenBank files,
                                      including supported .gz archives (required)
        -i, --input_folder DIR        Input root directory (default: input)
        -o, --output_folder DIR       Output root directory (default: output)
        -t, --tmp_folder DIR          Temporary root directory (default: tmp)
        -a, --allele_file FILE        Two-row CSV/TSV reference-allele file;
                                      required for --data_type digit
        -m, --mismatch INT            Maximum BLAST substitutions (default: 6)
        -L, --context_shoulder INT    Zero-based query position of the called base.
                                      0 means use the center of each context for
                                      character/digit mode (default: 0)
        -y, --data_type TYPE          character | digit | probe (default: digit)
        -n, --no_data_mark SYMBOL     Missing-call symbol for character/digit
                                      (default: n)
        -F, --flash_after_genome_number INT
                                      Flush accumulated output after this many
                                      processed genomes; 0 disables intermediate
                                      flushing (default: 50)
        -f, --filter_using_metadata_file FILE
                                      Optional metadata CSV/TSV/TXT file in
                                      input/PROJECT. When specified, only genomes
                                      represented in metadata and matching at least
                                      one keyword category are processed.
        -k, --keywords FILE           Keyword-rule CSV/TSV/TXT file in input/PROJECT
                                      (default: keywords.tsv). Used only with -f.
        -d, --level_delimiter STR     Delimiter between hierarchical category levels
                                      in output genome headings (default: |).
        -T, --threads INT             Number of concurrent locus/BLAST worker threads
        -h, --help                    Show this help

    Data-type behavior:
        character  Return the A/T/G/C base aligned to the requested query position.
        digit      Compare that base with the SNP reference allele from --allele_file;
                   return 0 for reference state and 1 for another A/T/G/C state.
        probe      Return 1 when the complete probe/context is found within the
                   mismatch threshold, otherwise 0.

    BLAST backend:
        Windows uses the bundled legacy blast.py wrapper with blastall.exe/formatdb.exe.
        Linux/Unix prefers system NCBI BLAST+ (blastn + makeblastdb), then bundled
        BLAST+ under lib/bin, with the custom wrapper as a final fallback.

    Parallelization:
        Independent loci from one genome are processed concurrently. Each worker
        launches at most one single-threaded BLAST process at a time, so --threads 8
        normally uses up to about 8 CPU cores without nested BLAST oversubscription.
    {'=' * 80}
    """
    print(help_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create character, digit, or probe-state feature matrices from genome sequences."
        ),
        add_help=False,
    )

    parser.add_argument("-h", "--help", action="store_true", dest="show_help", help=argparse.SUPPRESS)
    parser.add_argument("project_folder", nargs="?", type=str)
    parser.add_argument("input_file", nargs="?", type=str)
    parser.add_argument("output_file", nargs="?", type=str)

    parser.add_argument("-s", "--source_folder", type=str, default="")
    parser.add_argument("-i", "--input_folder", type=str, default="input")
    parser.add_argument("-o", "--output_folder", type=str, default="output")
    parser.add_argument("-t", "--tmp_folder", type=str, default="tmp")
    parser.add_argument("-a", "--allele_file", type=str, default="")
    parser.add_argument("-m", "--mismatch", type=int, default=6)
    parser.add_argument("-L", "--context_shoulder", type=int, default=0)
    parser.add_argument(
        "-y",
        "--data_type",
        type=str,
        choices=["character", "digit", "probe"],
        default="digit",
    )
    parser.add_argument(
        "-n",
        "--no_data_mark",
        type=str,
        default="n",
        help="Missing-call symbol for character/digit modes (default: n)",
    )
    parser.add_argument(
        "-F",
        "--flash_after_genome_number",
        type=int,
        default=50,
        help="Flush output after this many processed genomes; 0 disables intermediate flushing",
    )
    parser.add_argument(
        "-f",
        "--filter_using_metadata_file",
        type=str,
        default="",
        help=(
            "Optional metadata CSV/TSV/TXT file inside input/PROJECT. "
            "When set, only genomes represented in this metadata and matching "
            "at least one keyword category are processed."
        ),
    )
    parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        default="keywords.tsv",
        help=(
            "Keyword-rule CSV/TSV/TXT file inside input/PROJECT "
            "(default: keywords.tsv). Used only with --filter_using_metadata_file."
        ),
    )
    parser.add_argument(
        "-d",
        "--level_delimiter",
        type=str,
        default="|",
        help="Delimiter used between hierarchical categories in output headings (default: '|').",
    )
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=8,
        help=(
            "Number of concurrent locus/BLAST worker threads "
            "(default: 8). '--threads' is an alias."
        ),
    )

    args = parser.parse_args()

    if args.show_help:
        show_help()
        sys.exit(0)

    missing = []
    if not args.project_folder:
        missing.append("project_folder")
    if not args.input_file:
        missing.append("input_file")
    if not args.output_file:
        missing.append("output_file")
    if not args.source_folder:
        missing.append("--source_folder")

    if missing:
        print(f"ERROR: mandatory argument(s) missing: {', '.join(missing)}\n", file=sys.stderr)
        show_help()
        sys.exit(1)

    if args.mismatch < 0:
        parser.error("--mismatch must be >= 0")
    if args.context_shoulder < 0:
        parser.error("--context_shoulder must be >= 0")
    if args.flash_after_genome_number < 0:
        parser.error("--flash_after_genome_number must be >= 0")
    if args.no_data_mark == "":
        parser.error("--no_data_mark cannot be an empty string")
    if not args.level_delimiter:
        parser.error("--level_delimiter cannot be empty")
    if args.threads < 1:
        parser.error("--threads must be >= 1")

    return args


# =============================================================================
# General helpers
# =============================================================================

def reverse_complement(dna_sequence: str) -> str:
    """Generate reverse complement including standard ambiguous DNA bases."""
    complement = {
        "A": "T", "T": "A", "G": "C", "C": "G",
        "R": "Y", "Y": "R", "S": "S", "W": "W",
        "K": "M", "M": "K", "B": "V", "D": "H",
        "H": "D", "V": "B", "N": "N",
        "a": "t", "t": "a", "g": "c", "c": "g",
        "r": "y", "y": "r", "s": "s", "w": "w",
        "k": "m", "m": "k", "b": "v", "d": "h",
        "h": "d", "v": "b", "n": "n",
    }
    try:
        return "".join(complement[base] for base in reversed(dna_sequence))
    except KeyError as exc:
        raise ValueError(f"Invalid nucleotide character: {exc.args[0]}") from exc


def _detect_table_delimiter(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".tsv", ".txt"}:
        return "\t"
    if ext == ".csv":
        return ","

    with path.open("r", encoding="utf-8", newline="") as fp:
        sample = fp.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t"



def _read_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV/TSV/TXT table as stripped strings."""
    delimiter = _detect_table_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(fp, delimiter=delimiter)
            if row and any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError(f"Input table is empty: {path}")

    return rows[0], rows[1:]


def _load_keyword_rules(
    path: Path,
    metadata_header: Sequence[str],
) -> List[KeywordRule]:
    """
    Load no-priority hierarchical keyword rules.

    Expected columns:
        Category_1 ... Category_N
        Metadata_fields
        Match_type
        Keyword

    Metadata_fields may contain several metadata columns separated by ';'.
    These fields use OR logic. All keyword rows have equal importance.
    """
    header, rows = _read_table(path)
    header_map = {
        name.strip().lower(): i
        for i, name in enumerate(header)
    }

    category_columns: List[Tuple[int, int]] = []
    for idx, name in enumerate(header):
        match = re.fullmatch(
            r"category_(\d+)",
            name.strip(),
            flags=re.IGNORECASE,
        )
        if match:
            category_columns.append(
                (int(match.group(1)), idx)
            )

    category_columns.sort()

    if (
        not category_columns
        or category_columns[0][0] != 1
    ):
        raise ValueError(
            "Keyword file must contain hierarchy columns "
            "Category_1, Category_2, ..."
        )

    required = (
        "metadata_fields",
        "match_type",
        "keyword",
    )
    missing = [
        field
        for field in required
        if field not in header_map
    ]
    if missing:
        raise ValueError(
            "Keyword file is missing required column(s): "
            + ", ".join(missing)
        )

    metadata_names = set(metadata_header)
    rules: List[KeywordRule] = []

    for row_number, row in enumerate(rows, start=2):
        row = row + [""] * max(
            0,
            len(header) - len(row)
        )

        raw_categories = [
            row[idx].strip()
            for _, idx in category_columns
        ]

        # Do not allow hierarchy gaps such as:
        # Category_1=A, Category_2="", Category_3=C.
        seen_empty = False
        for value in raw_categories:
            if not value:
                seen_empty = True
            elif seen_empty:
                raise ValueError(
                    f"Keyword row {row_number} has a gap "
                    "in hierarchical categories."
                )

        categories = tuple(
            value
            for value in raw_categories
            if value
        )
        if not categories:
            continue

        metadata_fields = tuple(
            field.strip()
            for field in row[
                header_map["metadata_fields"]
            ].split(";")
            if field.strip()
        )

        if not metadata_fields:
            raise ValueError(
                f"Keyword row {row_number} has no "
                "Metadata_fields value."
            )

        unknown_fields = [
            field
            for field in metadata_fields
            if field not in metadata_names
        ]
        if unknown_fields:
            raise ValueError(
                f"Keyword row {row_number} refers to "
                "metadata field(s) absent from metadata: "
                + ", ".join(unknown_fields)
            )

        match_type = (
            row[header_map["match_type"]]
            .strip()
            .lower()
            or "exact"
        )

        if match_type not in {
            "exact",
            "contains",
        }:
            raise ValueError(
                f"Keyword row {row_number}: Match_type "
                "must be 'exact' or 'contains'."
            )

        keyword = (
            row[header_map["keyword"]]
            .strip()
            .lower()
        )
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
        raise ValueError(
            f"No usable keyword rules found in {path}"
        )

    return rules


def _classify_metadata_record(
    record: Dict[str, str],
    rules: Sequence[KeywordRule],
) -> List[Tuple[str, ...]]:
    """
    Return every hierarchical category path matched by a metadata record.

    All keywords are equally important. A rule matches if its keyword is
    found in at least one configured metadata field (OR across fields).
    Multiple rules matching the same category path are collapsed.
    """
    categories: set[Tuple[str, ...]] = set()

    for rule in rules:
        matched = False

        for field in rule.metadata_fields:
            value = str(
                record.get(field, "")
            ).strip().lower()

            if not value:
                continue

            if rule.match_type == "exact":
                matched = value == rule.keyword
            else:
                matched = rule.keyword in value

            if matched:
                break

        if matched:
            categories.add(rule.categories)

    return sorted(categories)


def _build_metadata_category_index(
    metadata_file: Path,
    keyword_file: Path,
) -> Dict[str, List[Tuple[str, ...]]]:
    """
    Build:
        assembly_accession -> [category_path, ...]

    Only metadata records assigned to at least one category are included.
    """
    header, rows = _read_table(metadata_file)

    try:
        accession_idx = header.index(
            "assembly_accession"
        )
    except ValueError as exc:
        raise ValueError(
            "Metadata file must contain the column "
            "'assembly_accession'."
        ) from exc

    rules = _load_keyword_rules(
        keyword_file,
        header,
    )

    result: Dict[
        str,
        List[Tuple[str, ...]]
    ] = {}

    for row in rows:
        row = row + [""] * max(
            0,
            len(header) - len(row)
        )

        accession = (
            row[accession_idx]
            .strip()
            .upper()
        )
        if not accession:
            continue

        record = {
            name: row[i]
            if i < len(row)
            else ""
            for i, name in enumerate(header)
        }

        categories = _classify_metadata_record(
            record,
            rules,
        )

        if categories:
            result[accession] = categories

    return result


def _assembly_accession_from_name(
    name: str,
) -> Optional[str]:
    """Extract a GCA_/GCF_ assembly accession from a genome filename."""
    match = ASSEMBLY_RE.search(name)
    if match is None:
        return None
    return match.group(1).upper()


def _make_output_titles(
    source_name: str,
    category_paths: Sequence[Tuple[str, ...]],
    delimiter: str,
) -> List[str]:
    """
    Build output row titles.

    Without metadata filtering:
        GCA_..._genomic.fna.gz

    With one category:
        Environment|Water and wastewater|GCA_..._genomic.fna.gz

    If a genome matches several terminal categories, one title is returned
    for every category path.
    """
    if not category_paths:
        return [source_name]

    return [
        delimiter.join(
            (*category_path, source_name)
        )
        for category_path in category_paths
    ]


def _source_name_from_output_title(
    title: str,
    delimiter: str,
) -> str:
    """
    Recover the original genome filename from an output title.

    This allows restart detection to work for both plain titles and
    hierarchy-prefixed titles.
    """
    if delimiter and delimiter in title:
        return title.rsplit(
            delimiter,
            1,
        )[-1].strip()
    return title.strip()


def _resolve_project_file(input_directory: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return input_directory / path


def _clean_tmp_directory(tmp_directory: Path) -> None:
    """Remove the project-specific temporary directory and recreate it empty."""
    if tmp_directory.exists():
        shutil.rmtree(tmp_directory, ignore_errors=True)
    tmp_directory.mkdir(parents=True, exist_ok=True)


def _safe_remove(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except OSError:
        pass


def _is_supported_genome_name(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    return lower.endswith(SUPPORTED_GENOME_EXTENSIONS)


def _prepare_genome_file(source_path: Path, tmp_directory: Path) -> Path:
    """
    Materialize a genome file for processing.

    Plain files are read in place. GZ archives are decompressed into the
    project-specific temporary directory, as requested.
    """
    if not source_path.name.lower().endswith(".gz"):
        return source_path

    uncompressed_name = source_path.name[:-3]
    target = tmp_directory / uncompressed_name
    with gzip.open(source_path, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _read_genbank_concat(filepath: Path | str) -> str:
    """
    Read a plain GenBank/FASTA file and concatenate all sequences.
    Contigs are separated by ten Ns.
    """
    path = Path(filepath)
    name_lower = path.name.lower()

    if name_lower.endswith(GENBANK_EXTENSIONS):
        fmt = "genbank"
    elif name_lower.endswith(FASTA_EXTENSIONS):
        fmt = "fasta"
    else:
        raise ValueError(f"Unknown genome-file extension: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        seqs = [str(rec.seq).upper() for rec in SeqIO.parse(handle, fmt)]

    if not seqs:
        raise ValueError(f"No sequences found in file: {path}")

    return ("N" * 10).join(seqs)


def _load_feature_file(path: Path) -> Tuple[List[str], List[List[str]], int]:
    """
    Load the SNP/probe annotation table.

    Supports an optional first metadata line such as:
        #Reference genome: file.gbff
    followed by the true table header.
    """
    delimiter = _detect_table_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp, delimiter=delimiter)
        rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"Feature file is empty or unreadable: {path}")

    header_idx = 0
    if rows[0] and rows[0][0].lstrip().startswith("#"):
        header_idx = 1

    if header_idx >= len(rows):
        raise ValueError(f"Feature file contains no table header: {path}")

    header = rows[header_idx]
    header_lc = [h.strip().lower() for h in header]

    context_idx: Optional[int] = None
    for candidate in ("context", "probe", "sequence"):
        if candidate in header_lc:
            context_idx = header_lc.index(candidate)
            break

    if context_idx is None:
        raise ValueError(
            "No context column found in feature-file header. "
            "Expected one of: context, probe, sequence."
        )

    loci = [
        row
        for row in rows[header_idx + 1 :]
        if row and row[0].strip() and len(row) > context_idx and row[context_idx].strip()
    ]

    if not loci:
        raise ValueError(f"No valid feature records found in: {path}")

    return header, loci, context_idx


def _load_reference_alleles(path: Path) -> Dict[str, str]:
    """Read the required two-row SNP-ID/reference-allele CSV/TSV file."""
    delimiter = _detect_table_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(fp, delimiter=delimiter)
            if any(cell.strip() for cell in row)
        ]

    if len(rows) < 2:
        raise ValueError(
            f"Allele file must contain two non-empty rows (SNP IDs and reference alleles): {path}"
        )
    if len(rows[0]) != len(rows[1]):
        raise ValueError(
            f"Allele-file rows have different lengths ({len(rows[0])} vs {len(rows[1])}): {path}"
        )

    result: Dict[str, str] = {}
    for snp_id, allele in zip(rows[0], rows[1]):
        snp_id = snp_id.strip()
        allele = allele.strip().upper()
        if not snp_id:
            continue
        if allele not in DNA_ALPHABET:
            raise ValueError(f"Invalid reference allele '{allele}' for {snp_id} in {path}")
        if snp_id in result:
            raise ValueError(f"Duplicate SNP ID in allele file: {snp_id}")
        result[snp_id] = allele

    return result


def _get_used_genomes_from_file(
    fname: Path | str,
    level_delimiter: str = "|",
) -> set[str]:
    """
    Return original source genome filenames already present in an output matrix.

    Hierarchical prefixes are removed from the first column, e.g.:

        Environment|Water|GCA_123_genomic.fna.gz

    becomes:

        GCA_123_genomic.fna.gz
    """
    path = Path(fname)
    delimiter = _detect_table_delimiter(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fp:
        rows = list(
            csv.reader(
                fp,
                delimiter=delimiter,
            )
        )

    return {
        _source_name_from_output_title(
            row[0].strip(),
            level_delimiter,
        )
        for row in rows[1:]
        if row and row[0].strip()
    }


# =============================================================================
# BLAST backend discovery and execution
# =============================================================================

def _bundled_executable(name: str) -> Optional[str]:
    """Return a platform-compatible executable from LIB_DIR/bin, if present."""
    if LIB_DIR is None:
        return None

    bin_dir = LIB_DIR / "bin"

    if sys.platform == "win32":
        # Do not fall back to extensionless Linux binaries on Windows.
        candidates = [bin_dir / f"{name}.exe"]
    else:
        candidates = [bin_dir / name]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def _find_blast_backend() -> Dict[str, Any]:
    """
    Select the BLAST backend appropriate for the operating system.

    Windows
    -------
    Use the legacy custom ``blast.py`` wrapper with:
        LIB_DIR/bin/blastall.exe
        LIB_DIR/bin/formatdb.exe

    Linux/Unix
    ----------
    Prefer NCBI BLAST+ from PATH (blastn + makeblastdb), then bundled
    BLAST+ from LIB_DIR/bin, and finally the custom wrapper if available.
    """

    # ------------------------------------------------------------------
    # Windows: this project uses legacy blastall.exe + formatdb.exe.
    # ------------------------------------------------------------------
    if sys.platform == "win32":
        if custom_blast is None:
            raise RuntimeError(
                "The custom blast.py module could not be imported on Windows."
            )

        if LIB_DIR is None:
            raise RuntimeError(
                "The local lib directory was not found. Expected a layout such as "
                "NCBI/lib/bin and NCBI/8.feature_calling/feature_caller.py."
            )

        bin_dir = LIB_DIR / "bin"
        blastall_exe = bin_dir / "blastall.exe"
        formatdb_exe = bin_dir / "formatdb.exe"

        missing = [
            str(path)
            for path in (blastall_exe, formatdb_exe)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "Required Windows BLAST executable(s) not found: "
                + ", ".join(missing)
            )

        return {
            "kind": "custom",
            "module": custom_blast,
            "binpath": str(bin_dir),
            "blastall": str(blastall_exe),
            "formatdb": str(formatdb_exe),
            "description": "Windows legacy BLAST (blastall.exe/formatdb.exe)",
        }

    # ------------------------------------------------------------------
    # Linux/Unix: prefer BLAST+ installed in PATH.
    # ------------------------------------------------------------------
    blastn = shutil.which("blastn")
    makeblastdb = shutil.which("makeblastdb")

    if blastn and makeblastdb:
        return {
            "kind": "native",
            "blastn": blastn,
            "makeblastdb": makeblastdb,
            "description": "system NCBI BLAST+",
        }

    # ------------------------------------------------------------------
    # Then try bundled Linux BLAST+ in LIB_DIR/bin.
    # ------------------------------------------------------------------
    blastn = _bundled_executable("blastn")
    makeblastdb = _bundled_executable("makeblastdb")

    if blastn and makeblastdb:
        try:
            os.chmod(blastn, 0o755)
            os.chmod(makeblastdb, 0o755)
        except OSError:
            pass

        return {
            "kind": "native",
            "blastn": blastn,
            "makeblastdb": makeblastdb,
            "description": "bundled Linux NCBI BLAST+",
        }

    # ------------------------------------------------------------------
    # Final Linux/Unix fallback: use the custom wrapper if available.
    # ------------------------------------------------------------------
    if custom_blast is not None and LIB_DIR is not None:
        return {
            "kind": "custom",
            "module": custom_blast,
            "binpath": str(LIB_DIR / "bin"),
            "description": "legacy custom BLAST wrapper",
        }

    raise RuntimeError(
        "Compatible BLAST executables were not found. On Windows, place "
        "blastall.exe and formatdb.exe in NCBI/lib/bin. On Linux, install "
        "blastn and makeblastdb or place them in NCBI/lib/bin."
    )


def _run_checked(command: List[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _create_native_blast_db(
    sequence: str,
    tmp_directory: Path,
    makeblastdb_exe: str,
) -> Tuple[str, Path]:
    input_fasta = tmp_directory / "genome_for_blast.fasta"
    input_fasta.write_text(f">genome\n{sequence}\n", encoding="utf-8")
    dbname = str(tmp_directory / "genome_db")
    _run_checked([
        makeblastdb_exe,
        "-in", str(input_fasta),
        "-dbtype", "nucl",
        "-out", dbname,
    ])
    return dbname, input_fasta


def _query_base_from_alignment(qseq: str, sseq: str, qstart: int, center: int) -> Optional[str]:
    """Return the subject base aligned to zero-based query position `center`."""
    query_pos = qstart - 1
    for qchar, schar in zip(qseq, sseq):
        if qchar != "-":
            if query_pos == center:
                base = schar.upper()
                return base if base in DNA_ALPHABET else None
            query_pos += 1
    return None


def _native_blast_context(
    context: str,
    center: int,
    mismatch: int,
    blastn_exe: str,
    dbname: str,
    tmp_directory: Path,
) -> Optional[Tuple[str, str, int]]:
    """
    Search one context with native blastn.

    Returns (called_base_in_query_orientation, aligned_subject_context, zero_based_location)
    or None. Only full-length, ungapped hits with <= mismatch substitutions are accepted.
    """
    # One query file per worker thread prevents races when several
    # BLAST searches run concurrently.
    query_file = tmp_directory / f"context_query_{threading.get_ident()}.fasta"
    query_file.write_text(f">context\n{context}\n", encoding="utf-8")

    outfmt = "6 qstart qend sstart send length mismatch gaps bitscore qseq sseq"
    result = _run_checked([
        blastn_exe,
        "-query", str(query_file),
        "-db", dbname,
        "-task", "blastn-short",
        "-dust", "no",
        "-word_size", "7",
        "-evalue", "1000",
        "-max_target_seqs", "50",
        "-outfmt", outfmt,
    ])

    candidates: List[Tuple[int, float, int, str, int]] = []
    qlen = len(context)

    for line in result.stdout.splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 10:
            continue
        try:
            qstart = int(fields[0])
            qend = int(fields[1])
            sstart = int(fields[2])
            send = int(fields[3])
            length = int(fields[4])
            mismatches = int(fields[5])
            gaps = int(fields[6])
            bitscore = float(fields[7])
        except ValueError:
            continue
        qseq = fields[8].upper()
        sseq = fields[9].upper()

        if qstart != 1 or qend != qlen or length != qlen or gaps != 0:
            continue
        if mismatches > mismatch:
            continue

        called_base = _query_base_from_alignment(qseq, sseq, qstart, center)
        if called_base is None:
            continue

        # BLAST sseq is emitted in query-aligned orientation, so called_base is
        # already comparable with the reference allele from the input context.
        genomic_center = (sstart - 1 + center) if sstart <= send else (sstart - 1 - center)
        candidates.append((mismatches, -bitscore, genomic_center, sseq, ord(called_base)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    return chr(best[4]), best[3], best[2]


def _custom_blast_context(
    seq: str,
    context: str,
    center: int,
    mismatch: int,
    o_blast: Any,
    dbname: str,
    tmp_directory: Path,
) -> Optional[Tuple[str, str, int]]:
    # One query file per worker thread prevents races when several
    # BLAST searches run concurrently.
    query_file = tmp_directory / f"context_query_{threading.get_ident()}.fasta"
    query_file.write_text(f">context\n{context}\n", encoding="utf-8")

    o_blast.execute(str(query_file.resolve()), dbname)
    matches = o_blast.get_matches(query_length=len(context), mismatches=mismatch)
    matches = [
        m for m in matches
        if getattr(m, "alignment_length", 0) == len(context)
    ]
    if not matches:
        return None

    matches.sort(
        key=lambda m: (
            -int(getattr(m, "identities", 0)),
            min(int(m.sbjct_start), int(m.sbjct_end)),
        )
    )
    match = matches[0]

    sstart = int(match.sbjct_start)
    send = int(match.sbjct_end)
    start0 = min(sstart, send) - 1
    end0 = max(sstart, send)
    subject_context = seq[start0:end0].upper()

    if sstart > send or getattr(match, "strand", "") == "Plus/Minus":
        subject_context = reverse_complement(subject_context).upper()
        genomic_center = (sstart - 1) - center
    else:
        genomic_center = (sstart - 1) + center

    if len(subject_context) != len(context) or center >= len(subject_context):
        return None

    called_base = subject_context[center].upper()
    if called_base not in DNA_ALPHABET:
        return None

    return called_base, subject_context, genomic_center


# =============================================================================
# Allelic/probe state calling
# =============================================================================

def _format_context(subject_context: str, center: int) -> str:
    if not subject_context or center < 0 or center >= len(subject_context):
        return subject_context
    return (
        subject_context[:center].lower()
        + subject_context[center].upper()
        + subject_context[center + 1 :].lower()
    )


def _state_from_base(
    called_base: Optional[str],
    data_type: str,
    reference_allele: Optional[str],
    no_data_mark: str,
    success: bool,
) -> str:
    mode = data_type.lower()

    if mode == "probe":
        return "1" if success else "0"

    if not success or called_base is None or called_base.upper() not in DNA_ALPHABET:
        return no_data_mark

    called_base = called_base.upper()

    if mode == "character":
        return called_base

    if mode == "digit":
        if reference_allele is None:
            raise ValueError("reference_allele is required for data_type='digit'")
        reference_allele = reference_allele.upper()
        if reference_allele not in DNA_ALPHABET:
            raise ValueError(f"Invalid reference allele: {reference_allele}")
        return "0" if called_base == reference_allele else "1"

    raise ValueError(f"Unsupported data_type: {data_type}")


def _get_allelic_state(
    seq: str,
    context: str,
    mismatch: int,
    center: Optional[int] = None,
    data_type: str = "probe",
    reference_allele: Optional[str] = None,
    no_data_mark: str = "n",
    native_blastn: Optional[str] = None,
    dbname: str = "",
    tmp_directory: Optional[Path] = None,
    custom_blast_object: Any = None,
) -> Tuple[str, str, int]:
    """
    Determine character/digit/probe state.

    character:
        return A/T/G/C aligned immediately after the requested left shoulder.
    digit:
        compare that base to `reference_allele`: reference=0, alternative=1.
    probe:
        return 1 for a successful complete-context match, otherwise 0.

    For character/digit, failed searches return `no_data_mark`.
    Location is zero-based and refers to the called base for character/digit,
    or to the query center for probe mode.
    """
    mode = data_type.lower()
    if mode not in {"character", "digit", "probe"}:
        raise ValueError(f"Unsupported data_type: {data_type}")

    context = context.strip().upper()
    if not context:
        return _state_from_base(None, mode, reference_allele, no_data_mark, False), "", -1

    if center is None:
        center = len(context) // 2
    if center < 0 or center >= len(context):
        raise ValueError(
            f"Context shoulder/center {center} is outside context length {len(context)}."
        )

    seq_upper = seq.upper()

    # ------------------------------------------------------------------
    # Exact forward match
    # ------------------------------------------------------------------
    location = seq_upper.find(context)
    if location >= 0:
        called_base = context[center] if mode != "probe" else None
        state = _state_from_base(called_base, mode, reference_allele, no_data_mark, True)
        shown = _format_context(context, center)
        return state, shown, location + center

    # ------------------------------------------------------------------
    # Exact reverse-strand match. Convert the matched subject segment back
    # to query orientation before reading the requested allele.
    # ------------------------------------------------------------------
    rc_context = reverse_complement(context).upper()
    location = seq_upper.find(rc_context)
    if location >= 0:
        subject_query_orientation = reverse_complement(seq_upper[location : location + len(context)]).upper()
        called_base = subject_query_orientation[center] if mode != "probe" else None
        state = _state_from_base(called_base, mode, reference_allele, no_data_mark, True)
        shown = _format_context(subject_query_orientation, center)
        genomic_center = location + (len(context) - 1 - center)
        return state, shown, genomic_center

    # With mismatch=0, exact search is sufficient.
    if mismatch == 0:
        return _state_from_base(None, mode, reference_allele, no_data_mark, False), "", -1

    if tmp_directory is None:
        raise ValueError("tmp_directory is required for BLAST searches")

    # ------------------------------------------------------------------
    # Approximate full-length BLAST search
    # ------------------------------------------------------------------
    hit: Optional[Tuple[str, str, int]] = None
    if native_blastn and dbname:
        hit = _native_blast_context(
            context=context,
            center=center,
            mismatch=mismatch,
            blastn_exe=native_blastn,
            dbname=dbname,
            tmp_directory=tmp_directory,
        )
    elif custom_blast_object is not None and dbname:
        hit = _custom_blast_context(
            seq=seq_upper,
            context=context,
            center=center,
            mismatch=mismatch,
            o_blast=custom_blast_object,
            dbname=dbname,
            tmp_directory=tmp_directory,
        )

    if hit is None:
        return _state_from_base(None, mode, reference_allele, no_data_mark, False), "", -1

    called_base, subject_context, genomic_center = hit
    state = _state_from_base(called_base, mode, reference_allele, no_data_mark, True)
    shown = _format_context(subject_context, center)
    return state, shown, genomic_center


# =============================================================================
# Output
# =============================================================================

def save_output(
    output_path: Path | str,
    records: Dict[str, str],
    loci_headings: List[str],
    append: bool = False,
) -> None:
    """
    Save SNP/probe calls to matrix and FASTA files.

    Results are explicitly flushed to disk before the function returns
    to reduce the risk of losing recently written data if the program
    is interrupted later.
    """

    if not records:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ext = output_path.suffix.lower()

    if ext in {".tsv", ".txt"}:
        delimiter = "\t"
    elif ext == ".csv":
        delimiter = ","
    else:
        raise ValueError(
            f"Unsupported output file extension: {ext}"
        )

    mode = "a" if append else "w"

    # ------------------------------------------------------------
    # Save matrix
    # ------------------------------------------------------------
    with output_path.open(
        mode,
        encoding="utf-8",
        newline=""
    ) as fp:

        writer = csv.writer(
            fp,
            delimiter=delimiter,
            lineterminator="\n"
        )

        if not append:
            writer.writerow(
                ["Genome", *loci_headings]
            )

        for title, record in records.items():
            writer.writerow(
                [title, *list(record)]
            )

        # Force Python buffer -> operating system
        fp.flush()

        # Force operating-system buffer -> disk
        os.fsync(fp.fileno())

    # ------------------------------------------------------------
    # Save FASTA
    # ------------------------------------------------------------
    output_fasta_file = output_path.with_suffix(
        ".fasta"
    )

    with output_fasta_file.open(
        mode,
        encoding="utf-8",
        newline="\n"
    ) as fp:

        for title, record in records.items():
            fp.write(
                f">{title}\n{record}\n"
            )

        fp.flush()
        os.fsync(fp.fileno())
    
# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    cwd = Path.cwd()
    input_directory = (cwd / args.input_folder / args.project_folder).resolve()
    output_directory = (cwd / args.output_folder / args.project_folder).resolve()
    tmp_directory = (cwd / args.tmp_folder / args.project_folder).resolve()

    if not input_directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

    source_directory = Path(args.source_folder).expanduser().resolve()
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_directory}")

    in_file = _resolve_project_file(input_directory, args.input_file)
    if not in_file.is_file():
        raise FileNotFoundError(f"Input feature file does not exist: {in_file}")

    # ------------------------------------------------------------------
    # Optional metadata/category filter.
    # ------------------------------------------------------------------
    metadata_category_index: Optional[
        Dict[str, List[Tuple[str, ...]]]
    ] = None

    if args.filter_using_metadata_file:
        metadata_file = _resolve_project_file(
            input_directory,
            args.filter_using_metadata_file,
        )
        keyword_file = _resolve_project_file(
            input_directory,
            args.keywords,
        )

        if not metadata_file.is_file():
            raise FileNotFoundError(
                "Metadata filter file does not exist: "
                f"{metadata_file}"
            )

        if not keyword_file.is_file():
            raise FileNotFoundError(
                "Keyword file does not exist: "
                f"{keyword_file}"
            )

        metadata_category_index = (
            _build_metadata_category_index(
                metadata_file,
                keyword_file,
            )
        )

        if not metadata_category_index:
            raise ValueError(
                "Metadata filtering is enabled, but no metadata "
                "records could be assigned to a keyword category."
            )

        print(
            "Metadata filtering enabled: "
            f"{len(metadata_category_index):,} assembly accession(s) "
            "have at least one assigned category."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    out_file = output_directory / args.output_file
    if out_file.exists() and out_file.is_dir():
        raise ValueError(f"Output file path is a directory: {out_file}")
    if out_file.suffix.lower() not in {".tsv", ".csv", ".txt"}:
        raise ValueError("Output file extension must be .tsv, .txt, or .csv")

    # ------------------------------------------------------------------
    # Remove leftovers from an interrupted previous run BEFORE starting.
    # ------------------------------------------------------------------
    _clean_tmp_directory(tmp_directory)

    try:
        header, loci, context_idx = _load_feature_file(in_file)
        loci_headings = [row[0].strip() for row in loci]

        # ------------------------------------------------------------------
        # Reference alleles are mandatory and actively used in digit mode.
        # ------------------------------------------------------------------
        reference_alleles: Dict[str, str] = {}
        if args.data_type == "digit":
            if not args.allele_file:
                raise ValueError("--allele_file is required when --data_type digit")
            allele_file = _resolve_project_file(input_directory, args.allele_file)
            if not allele_file.is_file():
                raise FileNotFoundError(f"Allele file does not exist: {allele_file}")
            reference_alleles = _load_reference_alleles(allele_file)
            missing_alleles = [snp_id for snp_id in loci_headings if snp_id not in reference_alleles]
            if missing_alleles:
                preview = ", ".join(missing_alleles[:10])
                suffix = "..." if len(missing_alleles) > 10 else ""
                raise ValueError(
                    f"{len(missing_alleles)} SNP(s) from the feature file are absent from --allele_file: "
                    f"{preview}{suffix}"
                )

        backend = _find_blast_backend()
        print(f"BLAST backend: {backend['description']}")
        print(
            "Parallel BLAST/locus workers: "
            f"{args.threads}"
        )

        used_genomes: set[str] = set()
        append = False
        if out_file.exists():
            used_genomes = _get_used_genomes_from_file(
                out_file,
                args.level_delimiter,
            )
            append = True
            print(f"Existing output detected: {len(used_genomes)} genome(s) already present.")

        all_files = sorted(
            p
            for p in source_directory.iterdir()
            if (
                p.is_file()
                and _is_supported_genome_name(
                    p.name
                )
            )
        )

        if not all_files:
            raise ValueError(
                "No supported genome files found in: "
                f"{source_directory}"
            )

        # --------------------------------------------------------------
        # Pre-filter source genomes before any sequence/BLAST work.
        #
        # Each item:
        #   (source_path, category_paths)
        #
        # category_paths is empty when metadata filtering is disabled.
        # --------------------------------------------------------------
        selected_sources: List[
            Tuple[
                Path,
                List[Tuple[str, ...]],
            ]
        ] = []

        skipped_not_in_metadata = 0
        skipped_unclassified = 0
        skipped_no_accession = 0
        skipped_already_done = 0

        for source_path in all_files:
            if source_path.name in used_genomes:
                skipped_already_done += 1
                continue

            category_paths: List[
                Tuple[str, ...]
            ] = []

            if metadata_category_index is not None:
                accession = (
                    _assembly_accession_from_name(
                        source_path.name
                    )
                )

                if accession is None:
                    skipped_no_accession += 1
                    continue

                # Not represented among metadata records that have
                # a valid keyword classification.
                category_paths = (
                    metadata_category_index.get(
                        accession,
                        [],
                    )
                )

                if not category_paths:
                    # Distinguish "not in metadata" from "metadata but
                    # unclassified" is impossible after index compaction,
                    # so report these together as metadata/category skips.
                    skipped_unclassified += 1
                    continue

            selected_sources.append(
                (
                    source_path,
                    category_paths,
                )
            )

        print(
            f"Genome files found: {len(all_files)}"
        )
        print(
            "Genome files to process: "
            f"{len(selected_sources)}"
        )

        if metadata_category_index is not None:
            print(
                "Skipped because no GCA/GCF accession could be "
                f"parsed: {skipped_no_accession}"
            )
            print(
                "Skipped because accession was absent from classified "
                f"metadata or had no matching keyword category: "
                f"{skipped_unclassified}"
            )

        if skipped_already_done:
            print(
                "Skipped because already present in existing output: "
                f"{skipped_already_done}"
            )

        records: Dict[str, str] = {}
        total = len(selected_sources)
        processed = 0

        for file_index, (
            source_path,
            category_paths,
        ) in enumerate(
            selected_sources,
            start=1,
        ):
            source_name = source_path.name

            progress = (file_index - 1) / total if total else 1.0
            bar_width = 50
            filled = int(bar_width * progress)
            bar = "█" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\rProcessing genomes {file_index}/{total}: |{bar}| {progress * 100:5.1f}%"
            )
            sys.stdout.flush()

            # Remove per-genome temporary BLAST/database/query files while keeping
            # the project tmp directory itself.
            for child in list(tmp_directory.iterdir()):
                _safe_remove(child)

            prepared_path = _prepare_genome_file(source_path, tmp_directory)
            seq = _read_genbank_concat(prepared_path)

            native_blastn: Optional[str] = None
            dbname = ""

            if backend["kind"] == "native":
                # Build one read-only BLAST database per genome. All worker
                # processes can query this database concurrently.
                dbname, _ = _create_native_blast_db(
                    sequence=seq,
                    tmp_directory=tmp_directory,
                    makeblastdb_exe=backend["makeblastdb"],
                )
                native_blastn = backend["blastn"]

            else:
                # Build the database once. The BLAST wrapper object itself is
                # mutable and therefore must NOT be shared between threads.
                db_creator = backend["module"].BLAST(
                    seqtype="dna",
                    binpath=backend["binpath"],
                )
                input_fasta = tmp_directory / "genome_for_blast.fasta"
                input_fasta.write_text(
                    f">genome\n{seq}\n",
                    encoding="utf-8",
                )
                dbname = str(tmp_directory / "genome_db")
                db_creator.create_db(
                    fasta_file=str(input_fasta),
                    dbname=dbname,
                )

            # ----------------------------------------------------------
            # Process independent loci in parallel.
            #
            # executor.map preserves the order of `loci`, so the generated
            # state string remains aligned with loci_headings.
            #
            # For the legacy custom BLAST wrapper we create one mutable
            # BLAST object per worker thread using thread-local storage.
            # ----------------------------------------------------------
            thread_state = threading.local()

            def call_one_locus(locus: List[str]) -> str:
                snp_id = locus[0].strip()
                context = locus[context_idx].strip().upper()

                if args.data_type in {"character", "digit"}:
                    if args.context_shoulder == 0:
                        if len(context) % 2 != 1:
                            raise ValueError(
                                f"Context for {snp_id} has even length {len(context)}. "
                                "With --context_shoulder 0, character/digit contexts "
                                "must have odd length."
                            )
                        center = len(context) // 2
                    else:
                        center = args.context_shoulder
                else:
                    center = (
                        args.context_shoulder
                        if args.context_shoulder > 0
                        else len(context) // 2
                    )

                reference_allele = (
                    reference_alleles.get(snp_id)
                    if args.data_type == "digit"
                    else None
                )

                custom_blast_object: Any = None

                if backend["kind"] == "custom":
                    custom_blast_object = getattr(
                        thread_state,
                        "blast_object",
                        None,
                    )

                    if custom_blast_object is None:
                        custom_blast_object = (
                            backend["module"].BLAST(
                                seqtype="dna",
                                binpath=backend["binpath"],
                            )
                        )
                        thread_state.blast_object = (
                            custom_blast_object
                        )

                state, _current_context, _location = (
                    _get_allelic_state(
                        seq=seq,
                        context=context,
                        mismatch=args.mismatch,
                        center=center,
                        data_type=args.data_type,
                        reference_allele=reference_allele,
                        no_data_mark=args.no_data_mark,
                        native_blastn=native_blastn,
                        dbname=dbname,
                        tmp_directory=tmp_directory,
                        custom_blast_object=custom_blast_object,
                    )
                )

                return state

            if args.threads == 1 or len(loci) <= 1:
                states = [
                    call_one_locus(locus)
                    for locus in loci
                ]
            else:
                max_workers = min(
                    args.threads,
                    len(loci),
                )

                with ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    states = list(
                        executor.map(
                            call_one_locus,
                            loci,
                        )
                    )

            state_string = "".join(states)

            output_titles = _make_output_titles(
                source_name=source_name,
                category_paths=category_paths,
                delimiter=args.level_delimiter,
            )

            for output_title in output_titles:
                records[output_title] = state_string

            processed += 1
            used_genomes.add(source_name)

            flush_n = args.flash_after_genome_number
            if flush_n and processed % flush_n == 0:
                save_output(out_file, records, loci_headings, append=append)
                records = {}
                append = True

        if total:
            bar = "█" * 50
            sys.stdout.write(f"\rProcessing genomes {total}/{total}: |{bar}| 100.0%\n")
            sys.stdout.flush()

        save_output(out_file, records, loci_headings, append=append)
        print("✅ Pipeline completed successfully!")

    finally:
        # This runs on normal completion and on Python exceptions / Ctrl+C.
        # SIGKILL/power loss cannot execute cleanup, so the next run also clears
        # this directory before doing any work.
        _clean_tmp_directory(tmp_directory)

        # Legacy BLAST wrappers may leave these in the launch directory.
        for legacy_name in ("lintmp.out", "formatdb.log"):
            _safe_remove(cwd / legacy_name)


if __name__ == "__main__":
    args = parse_args()
    show_banner()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
