#!/usr/bin/env python3

import os
import sys
import argparse
import gzip
import shutil
import subprocess
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation

# Add ../lib to sys.path (relative to this script, not CWD)
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "lib"
sys.path.append(str(LIB_DIR))

import progressbar

# ============================================================
# Constants
# ============================================================

DNA_ALPHABET = set("ACGT")
STOP_CODONS = {"TAA", "TAG", "TGA"}

CORE_THRESHOLD = 0.95
PANAROO_THREADS = 8

# 40 nt + SNP + 40 nt
SNP_CONTEXT_FLANK = 40
SNP_CONTEXT_LENGTH = 81

GENBANK_EXTENSIONS = (".gb", ".gbk", ".gbf", ".gbff")
FASTA_EXTENSIONS = (".fa", ".fasta", ".fst", ".fas", ".fna")
SUPPORTED_SEQUENCE_EXTENSIONS = GENBANK_EXTENSIONS + FASTA_EXTENSIONS
LIST_LEVEL_DELIMITER = "|"


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Panaroo-based bacterial pangenome analysis. "
            "Creates an accessory-gene presence/absence matrix, "
            "a core-genome SNP matrix, and reference-based SNP annotation."
        )
    )

    parser.add_argument(
        "project_folder",
        nargs="?",
        type=str,
        help="Project folder name, e.g. Salmonella"
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
        "-t", "--tmp_folder",
        type=str,
        default="tmp",
        help="Temporary root folder (default: tmp)"
    )

    parser.add_argument(
        "-f", "--input_format",
        type=str,
        choices=["folders", "list"],
        default="folders",
        help=(
            "Input organization: 'folders' reads genome files recursively from "
            "input/project_folder and derives hierarchy from folders; 'list' reads "
            "--list_file from input/project_folder and resolves listed genome files "
            "under --source_folder (default: folders)."
        )
    )

    parser.add_argument(
        "-L", "--list_file",
        type=str,
        default="",
        help=(
            "List file used with --input_format list. Each non-empty line must be "
            "Category1|Category2|...|genome_filename. The file must be located in "
            "input/project_folder."
        )
    )

    parser.add_argument(
        "-s", "--source_folder",
        type=str,
        default="",
        help=(
            "Source folder containing original GenBank/FASTA genome files, optionally "
            "gzip-compressed. Required when --input_format list."
        )
    )

    parser.add_argument(
        "-m", "--minimal_diversity",
        type=float,
        default=5,
        help=(
            "Minimal gene-frequency percentage. "
            "5 keeps genes present in >5%% and <95%% of genomes; "
            "10 keeps >10%% and <90%% (default: 5)"
        )
    )

    parser.add_argument(
        "-r", "--reference",
        type=str,
        default="",
        help=(
            "Optional argument. "
            "Name of a GBFF file to be used as a reference; "
            "first GBFF file by default."
        )
    )

    parser.add_argument(
        "--rerun_panaroo",
        action="store_true",
        help="Rerun Panaroo even if existing results are present"
    )

    return parser.parse_args()


# ============================================================
# General helpers
# ============================================================

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):

    print("\n[RUN]")
    print(" ".join(map(str, cmd)))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def reverse_complement(sequence):
    return str(Seq(sequence).reverse_complement())


def complement_base(base):
    return str(Seq(base).complement())


# ============================================================
# Input discovery and metadata
# ============================================================

def _ungz_name(name):
    """Return filename without a final .gz suffix."""
    return name[:-3] if name.lower().endswith(".gz") else name


def _sequence_suffix(path_or_name):
    """Return the biological sequence suffix, ignoring a final .gz."""
    name = _ungz_name(Path(path_or_name).name).lower()
    return Path(name).suffix.lower()


def is_supported_sequence_file(path):
    path = Path(path)
    if not path.is_file():
        return False
    return _sequence_suffix(path) in SUPPORTED_SEQUENCE_EXTENSIONS


def sequence_format(path):
    """Return 'genbank' or 'fasta' from a supported filename."""
    suffix = _sequence_suffix(path)
    if suffix in GENBANK_EXTENSIONS:
        return "genbank"
    if suffix in FASTA_EXTENSIONS:
        return "fasta"
    raise ValueError(f"Unsupported genome sequence format: {path}")


def discover_files(root):
    """
    Recursively find supported GenBank/FASTA files and their .gz archives.
    Ordering is deterministic and is used to generate S0001, S0002, ...
    """
    root = Path(root)
    return sorted(
        (p for p in root.rglob("*") if is_supported_sequence_file(p)),
        key=lambda p: str(p).lower()
    )


def build_metadata(files, input_root):
    """
    Build metadata from the hierarchical folder organization under input_root.
    Folder names are converted to a '>'-separated location path.
    """
    input_root = Path(input_root)
    metadata = OrderedDict()

    for i, source_path in enumerate(files, 1):
        source_path = Path(source_path)
        sample_id = f"S{i:04d}"
        rel = source_path.relative_to(input_root)

        if str(rel.parent) == ".":
            location = ""
        else:
            location = str(rel.parent).replace(os.sep, ">")

        metadata[sample_id] = {
            "gbff_name": source_path.name,  # retained for downstream compatibility
            "location": location,
            "source_path": source_path,
            "relative_path": rel,
        }

    return metadata


def _index_source_files(source_root):
    """
    Index supported source files by basename.

    The list format produced by the genome-selection program contains a filename,
    not a full source path. Basenames therefore must be unique under source_root.
    """
    index = {}
    duplicates = {}

    for path in discover_files(source_root):
        key = path.name
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path

    return index, duplicates


def build_metadata_from_list(list_file, source_root, level_delimiter=LIST_LEVEL_DELIMITER):
    """
    Build S0001... metadata from a hierarchy list.

    Expected line format:
        Human clinical|Blood and tissues|GCA_..._genomic.fna.gz

    Every component before the final delimiter is a hierarchical level; the last
    component is the exact basename of a genome file available under source_root.
    """
    list_file = Path(list_file)
    source_root = Path(source_root)

    source_index, duplicates = _index_source_files(source_root)

    if not source_index:
        raise RuntimeError(
            f"No supported GenBank/FASTA genome files found under {source_root}"
        )

    metadata = OrderedDict()
    seen_names = set()
    missing = []

    with list_file.open("r", encoding="utf-8-sig") as fp:
        raw_lines = [
            line.strip()
            for line in fp
            if line.strip() and not line.lstrip().startswith("#")
        ]

    for line_number, line in enumerate(raw_lines, 1):
        parts = [part.strip() for part in line.split(level_delimiter)]

        if len(parts) < 2:
            raise ValueError(
                f"Invalid list record {line_number}: expected at least one hierarchy "
                f"level and a genome filename separated by '{level_delimiter}': {line}"
            )

        if any(not part for part in parts):
            raise ValueError(
                f"Invalid empty hierarchy/filename field in list record {line_number}: {line}"
            )

        genome_name = parts[-1]
        hierarchy = parts[:-1]

        if genome_name in seen_names:
            raise ValueError(
                f"Duplicate genome filename in list file: {genome_name}"
            )
        seen_names.add(genome_name)

        if genome_name in duplicates:
            paths = ", ".join(str(p) for p in duplicates[genome_name])
            raise ValueError(
                f"Genome filename '{genome_name}' is not unique under --source_folder: {paths}"
            )

        source_path = source_index.get(genome_name)
        if source_path is None:
            missing.append(genome_name)
            continue

        sample_id = f"S{len(metadata) + 1:04d}"
        relative_path = Path(*hierarchy) / genome_name

        metadata[sample_id] = {
            "gbff_name": genome_name,  # retained for downstream compatibility
            "location": ">".join(hierarchy),
            "source_path": source_path,
            "relative_path": relative_path,
        }

    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"{len(missing)} genome file(s) from {list_file} were not found under "
            f"{source_root}:\n  {preview}{suffix}"
        )

    if not metadata:
        raise RuntimeError(f"No usable genome records found in list file: {list_file}")

    return metadata


def save_metadata(metadata, output_file):
    rows = []

    for sample_id, info in metadata.items():
        rows.append({
            "sample_id": sample_id,
            "genome_file": info["gbff_name"],
            "location": info["location"],
            "source_path": str(info["source_path"]),
        })

    pd.DataFrame(rows).to_csv(
        output_file,
        sep="\t",
        index=False
    )


# ============================================================
# Temporary sequence files
# ============================================================

def get_tmp_gbff_path(info, input_root, sequence_root):
    """
    Return the project-temporary uncompressed/copy path for a genome.

    The historical function name is retained for compatibility with the rest of
    the script, but the path may contain either GenBank or FASTA data.
    """
    rel = Path(info.get("relative_path", info["source_path"].name))

    # Only remove the gzip suffix. Keep .gbff/.fna/etc.
    if rel.name.lower().endswith(".gz"):
        rel = rel.with_name(rel.name[:-3])

    return Path(sequence_root) / rel


def prepare_sequence_files(metadata, input_root, out_root):
    """
    Copy plain genome files or decompress .gz genome files into tmp/project.

    The hierarchy recorded in metadata is reproduced under out_root.
    """
    ensure_dir(out_root)

    for sample_id, info in metadata.items():
        source_path = Path(info["source_path"])
        out_path = get_tmp_gbff_path(info, input_root, out_root)
        ensure_dir(out_path.parent)

        if source_path.suffix.lower() == ".gz":
            print(f"Unzipping {source_path}")
            with gzip.open(source_path, "rb") as fin, open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            print(f"Copying {source_path}")
            shutil.copy2(source_path, out_path)

        # Historical key retained so downstream code need not change.
        info["gbff_path"] = out_path


# Backward-compatible alias used elsewhere in older revisions.
def unzip_files(metadata, input_root, out_root):
    prepare_sequence_files(metadata, input_root, out_root)


def attach_existing_tmp_paths(metadata, input_root, sequence_root):
    """
    Attach prepared temporary paths for an existing Panaroo run.
    Return missing paths so they can be recreated if necessary.
    """
    missing = []

    for sample_id, info in metadata.items():
        path = get_tmp_gbff_path(info, input_root, sequence_root)
        info["gbff_path"] = path

        if not path.exists():
            missing.append(path)

    return missing


# ============================================================
# GBFF -> GFF
# ============================================================

def is_partial_location(location):

    text = str(location)

    return "<" in text or ">" in text


def is_valid_cds_sequence(sequence):

    sequence = sequence.upper()

    if not sequence:
        return False

    if set(sequence) - DNA_ALPHABET:
        return False

    if len(sequence) % 3 != 0:
        return False

    if len(sequence) < 6:
        return False

    for i in range(0, len(sequence) - 3, 3):

        if sequence[i:i + 3] in STOP_CODONS:
            return False

    return True


def escape_gff(value):

    return (
        str(value)
        .replace(";", "%3B")
        .replace("=", "%3D")
        .replace("\t", " ")
        .replace("\n", " ")
    )


def gbff_to_gff(gbff, gff, sample_id):

    records = list(
        SeqIO.parse(str(gbff), "genbank")
    )

    if not records:
        raise RuntimeError(
            f"No GenBank records found in {gbff}"
        )

    with open(gff, "w") as out:

        out.write("##gff-version 3\n")

        feature_number = 0
        fasta_records = []

        for rec_number, rec in enumerate(records, 1):

            seqid = (
                rec.id
                if rec.id
                else f"{sample_id}_contig{rec_number}"
            )

            sequence = str(rec.seq).upper()

            fasta_records.append(
                (seqid, sequence)
            )

            out.write(
                f"##sequence-region {seqid} 1 {len(sequence)}\n"
            )

            for feat in rec.features:

                if feat.type != "CDS":
                    continue

                q = feat.qualifiers

                if "pseudo" in q:
                    continue

                if "pseudogene" in q:
                    continue

                if "transl_except" in q:
                    continue

                if "ribosomal_slippage" in q:
                    continue

                if "exception" in q:
                    continue

                if isinstance(
                    feat.location,
                    CompoundLocation
                ):
                    continue

                if is_partial_location(feat.location):
                    continue

                try:
                    start = int(feat.location.start) + 1
                    end = int(feat.location.end)
                except Exception:
                    continue

                strand = (
                    "-"
                    if feat.location.strand == -1
                    else "+"
                )

                try:
                    cds_sequence = str(
                        feat.extract(rec.seq)
                    ).upper()
                except Exception:
                    continue

                if not is_valid_cds_sequence(
                    cds_sequence
                ):
                    continue

                feature_number += 1

                locus = q.get(
                    "locus_tag",
                    [
                        f"{sample_id}_CDS_"
                        f"{feature_number:06d}"
                    ]
                )[0]

                gene = q.get("gene", [""])[0]
                product = q.get("product", [""])[0]

                attributes = [
                    f"ID={escape_gff(locus)}",
                    f"locus_tag={escape_gff(locus)}"
                ]

                if gene:
                    attributes.append(
                        f"gene={escape_gff(gene)}"
                    )

                    attributes.append(
                        f"Name={escape_gff(gene)}"
                    )

                if product:
                    attributes.append(
                        f"product={escape_gff(product)}"
                    )

                out.write(
                    f"{seqid}\t"
                    f"GBFF2GFF\t"
                    f"CDS\t"
                    f"{start}\t"
                    f"{end}\t"
                    f".\t"
                    f"{strand}\t"
                    f"0\t"
                    f"{';'.join(attributes)}\n"
                )

        out.write("##FASTA\n")

        for seqid, sequence in fasta_records:

            out.write(f">{seqid}\n")

            for i in range(
                0,
                len(sequence),
                60
            ):
                out.write(
                    sequence[i:i + 60] + "\n"
                )


def fasta_to_gff(fasta, gff, sample_id):
    """
    Predict CDS features from FASTA with Prodigal and create a Panaroo-compatible
    GFF3 containing the original FASTA sequences.

    Prodigal is required only when FASTA input files are used.
    """
    prodigal = shutil.which("prodigal")
    if not prodigal:
        raise RuntimeError(
            "FASTA input requires Prodigal to predict CDS features before Panaroo. "
            "Install/load 'prodigal' or provide annotated GenBank input files."
        )

    fasta = Path(fasta)
    gff = Path(gff)
    raw_gff = gff.with_suffix(".prodigal.tmp.gff")

    run_cmd([
        prodigal,
        "-i", str(fasta),
        "-o", str(raw_gff),
        "-f", "gff",
        "-p", "single",
        "-q",
    ])

    records = list(SeqIO.parse(str(fasta), "fasta"))
    if not records:
        raw_gff.unlink(missing_ok=True)
        raise RuntimeError(f"No FASTA records found in {fasta}")

    with gff.open("w", encoding="utf-8") as out:
        raw_text = raw_gff.read_text(encoding="utf-8", errors="replace")
        if raw_text.startswith("##gff-version"):
            out.write(raw_text.rstrip() + "\n")
        else:
            out.write("##gff-version 3\n")
            out.write(raw_text.rstrip() + "\n")

        out.write("##FASTA\n")
        for rec in records:
            sequence = str(rec.seq).upper()
            out.write(f">{rec.id}\n")
            for i in range(0, len(sequence), 60):
                out.write(sequence[i:i + 60] + "\n")

    raw_gff.unlink(missing_ok=True)


def convert_all(metadata, gff_dir):
    ensure_dir(gff_dir)
    gffs = []

    for sample_id, info in metadata.items():
        gff = gff_dir / f"{sample_id}.gff"
        genome_path = Path(info["gbff_path"])
        fmt = sequence_format(genome_path)

        print(
            f"Creating GFF: {sample_id} <- {info['gbff_name']} "
            f"({fmt})"
        )

        try:
            if fmt == "genbank":
                gbff_to_gff(
                    genome_path,
                    gff,
                    sample_id
                )
            else:
                fasta_to_gff(
                    genome_path,
                    gff,
                    sample_id
                )

            # Panaroo/gffutils requires at least one actual GFF feature line.
            # A file containing only directives and FASTA would otherwise make
            # Panaroo abort with gffutils.exceptions.EmptyInputError.
            feature_count = 0
            with gff.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("##FASTA"):
                        break
                    if line.strip() and not line.startswith("#"):
                        feature_count += 1

            if feature_count == 0:
                raise RuntimeError("GFF contains no valid annotation features")

        except Exception as exc:
            gff.unlink(missing_ok=True)
            info["skip_reason"] = str(exc)
            print(
                f"WARNING: skipping {sample_id} ({info['gbff_name']}): {exc}",
                file=sys.stderr
            )
            continue

        gffs.append(gff)

    return gffs


# ============================================================
# Accessory-gene matrix
# ============================================================

def make_unique_names(names):

    counts = {}
    result = []

    for name in names:

        if not name:
            name = "unnamed_gene"

        counts[name] = counts.get(name, 0) + 1

        if counts[name] == 1:
            result.append(name)
        else:
            result.append(
                f"{name}_{counts[name]}"
            )

    return result


def create_accessory_matrix(
        metadata,
        panaroo_out,
        output_root,
        pmin,
        pmax):

    input_file = (
        panaroo_out /
        "gene_presence_absence.csv"
    )

    if not input_file.exists():

        raise RuntimeError(
            f"Panaroo output file not found: {input_file}"
        )

    df = pd.read_csv(
        input_file,
        dtype=str,
        low_memory=False
    )

    expected_samples = list(metadata.keys())

    genome_cols = [
        sid
        for sid in expected_samples
        if sid in df.columns
    ]

    missing = [
        sid
        for sid in expected_samples
        if sid not in df.columns
    ]

    if missing:

        print(
            "\nWARNING: samples absent from "
            "gene_presence_absence.csv:"
        )

        print(", ".join(missing))

    if not genome_cols:

        raise RuntimeError(
            "No reconstructed sample IDs match "
            "Panaroo genome columns."
        )

    presence = (
        df[genome_cols].notna()
        &
        df[genome_cols].ne("")
    )

    frequency = (
        presence.sum(axis=1)
        /
        len(genome_cols)
    )

    mask = (
        (frequency > pmin)
        &
        (frequency < pmax)
    )

    accessory_df = df.loc[mask].copy()

    accessory_df.insert(
        0,
        "frequency",
        frequency.loc[mask].values
    )

    accessory_df.insert(
        0,
        "n_genomes",
        presence.loc[mask].sum(axis=1).values
    )

    accessory_df.to_csv(
        output_root /
        "accessory_filtered.csv",
        index=False
    )

    matrix = (
        presence.loc[mask]
        .T
        .astype(int)
    )

    if "Gene" in df.columns:

        gene_names = (
            df.loc[mask, "Gene"]
            .fillna("")
            .astype(str)
            .tolist()
        )

    else:

        gene_names = [
            f"gene_{i + 1}"
            for i in range(matrix.shape[1])
        ]

    matrix.columns = make_unique_names(
        gene_names
    )

    rows = []

    for sample_id in genome_cols:

        row = {
            "gbff_file":
                metadata[sample_id]["gbff_name"],

            "location":
                metadata[sample_id]["location"]
        }

        for gene in matrix.columns:

            row[gene] = int(
                matrix.loc[sample_id, gene]
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_root /
        "accessory_matrix.tsv",
        sep="\t",
        index=False
    )

    print(
        f"Accessory gene families retained: "
        f"{mask.sum()}"
    )


# ============================================================
# Core alignment
# ============================================================

def find_core_alignment(panaroo_out):

    # Requested standard location
    requested = (
        panaroo_out /
        "core_gene_alignment.aln"
    )

    if requested.exists():
        return requested

    # Compatibility fallback
    fallback = (
        panaroo_out /
        "core_alignment.aln"
    )

    if fallback.exists():
        return fallback

    raise RuntimeError(
        "Core alignment not found. Checked:\n"
        f"  {requested}\n"
        f"  {fallback}"
    )


# ============================================================
# Reference genome
# ============================================================

def get_reference_gbff(
        metadata,
        input_root,
        sequence_root):
    """
    Return the prepared sequence file for S0001.

    Historical function name retained for compatibility. The reference may be a
    GenBank or FASTA file; GenBank is preferred when annotations are needed.
    """
    if "S0001" not in metadata:
        raise RuntimeError("S0001 is absent from metadata.")

    info = metadata["S0001"]
    reference = get_tmp_gbff_path(
        info,
        input_root,
        sequence_root
    )

    if reference.exists():
        return reference

    raise RuntimeError(
        f"Prepared reference sequence for S0001 was not found: {reference}"
    )


def resolve_requested_reference(metadata, reference_name):
    """Find a prepared reference by exact source basename."""
    matches = [
        info["gbff_path"]
        for info in metadata.values()
        if info["gbff_name"] == reference_name
        or Path(info["gbff_path"]).name == reference_name
    ]

    if not matches:
        raise FileNotFoundError(
            f"Requested reference '{reference_name}' is not among the selected genomes."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Requested reference '{reference_name}' is ambiguous."
        )
    return Path(matches[0])


def load_reference_genome(reference_file):
    fmt = sequence_format(reference_file)

    records = list(
        SeqIO.parse(
            str(reference_file),
            fmt
        )
    )

    if not records:
        raise RuntimeError(
            f"Could not read reference sequence: {reference_file}"
        )

    if fmt == "fasta":
        print(
            "\nWARNING: the reference is FASTA, so CDS/codon annotations are "
            "unavailable. SNP genomic coordinates and strand can still be reported."
        )

    return records


# ============================================================
# SNP context
# ============================================================

def make_reference_context(
        reference_alignment,
        alignment_position,
        flank=SNP_CONTEXT_FLANK):

    """
    Obtain exactly 40 reference nucleotides before and after the
    SNP, ignoring alignment gaps.

    Returns an 81-nt sequence:

        40 nt + reference allele + 40 nt

    If fewer than 40 reference nucleotides are available on
    either side, returns None.
    """

    ref_base = reference_alignment[
        alignment_position
    ]

    if ref_base not in DNA_ALPHABET:
        return None

    # Left flank
    left = []

    i = alignment_position - 1

    while i >= 0 and len(left) < flank:

        base = reference_alignment[i]

        if base in DNA_ALPHABET:
            left.append(base)

        i -= 1

    if len(left) != flank:
        return None

    left.reverse()

    # Right flank
    right = []

    i = alignment_position + 1

    while (
        i < len(reference_alignment)
        and
        len(right) < flank
    ):

        base = reference_alignment[i]

        if base in DNA_ALPHABET:
            right.append(base)

        i += 1

    if len(right) != flank:
        return None

    context = (
        "".join(left)
        +
        ref_base
        +
        "".join(right)
    )

    if len(context) != (
        2 * flank + 1
    ):
        return None

    return context


# ============================================================
# Find context in reference genome
# ============================================================

def find_all_occurrences(sequence, query):

    starts = []

    start = sequence.find(query)

    while start != -1:

        starts.append(start)

        start = sequence.find(
            query,
            start + 1
        )

    return starts


def find_context_in_reference(
        records,
        context):

    """
    Search both strands of every reference sequence.

    Returns a list of:

        {
            record,
            record_id,
            genomic_position,
            context_orientation
        }

    genomic_position is zero-based.

    context_orientation:
        +1 -> context matches reference genomic sequence
        -1 -> context matches its reverse complement
    """

    hits = []

    context = context.upper()
    rc_context = reverse_complement(context)

    center = len(context) // 2

    for record in records:

        sequence = str(record.seq).upper()

        # Forward match
        for start in find_all_occurrences(
            sequence,
            context
        ):

            hits.append({
                "record": record,
                "record_id": record.id,
                "genomic_position":
                    start + center,
                "context_orientation": 1
            })

        # Reverse-complement match
        if rc_context != context:

            for start in find_all_occurrences(
                sequence,
                rc_context
            ):

                # For a symmetric odd-length context, the SNP is still
                # exactly at the central position.
                genomic_position = (
                    start
                    +
                    len(context)
                    - 1
                    - center
                )

                hits.append({
                    "record": record,
                    "record_id": record.id,
                    "genomic_position":
                        genomic_position,
                    "context_orientation": -1
                })

    return hits


# ============================================================
# Reference feature lookup
# ============================================================

def find_cds_at_position(
        record,
        genomic_position):

    hits = []

    for feature in record.features:

        if feature.type != "CDS":
            continue

        try:
            if genomic_position in feature.location:
                hits.append(feature)
        except Exception:
            continue

    return hits


def get_annotation(feature):

    q = feature.qualifiers

    gene = q.get("gene", [""])[0]
    product = q.get("product", [""])[0]

    if gene and product:
        return f"{gene} | {product}"

    if gene:
        return gene

    if product:
        return product

    return ""


def get_locus_tag(feature):

    return feature.qualifiers.get(
        "locus_tag",
        [""]
    )[0]


# ============================================================
# CDS coordinate handling
# ============================================================

def get_feature_position_list(feature):

    """
    Biopython FeatureLocation iteration follows the feature
    strand, which gives genomic positions in CDS sequence order.
    """

    try:
        return list(feature.location)
    except Exception:
        return []


def get_codon_start_offset(feature):

    value = feature.qualifiers.get(
        "codon_start",
        ["1"]
    )[0]

    try:
        value = int(value)
    except Exception:
        value = 1

    if value not in (1, 2, 3):
        value = 1

    return value - 1


# ============================================================
# Amino-acid consequence
# ============================================================

def nucleotide_to_genomic_orientation(
        allele,
        context_orientation):

    allele = allele.upper()

    if context_orientation == 1:
        return allele

    return complement_base(allele)


def genomic_to_coding_orientation(
        allele,
        strand):

    if strand == -1:
        return complement_base(allele)

    return allele


def annotate_codon_change(
        record,
        feature,
        genomic_position,
        alignment_alleles,
        reference_alignment_allele,
        context_orientation):

    """
    Produce values such as:

        A233A
        A233T
        A233T,G

    where the first amino-acid letter is the reference protein
    state and the number is the amino-acid/codon position.
    """

    positions = get_feature_position_list(
        feature
    )

    if not positions:
        return ""

    try:
        raw_index = positions.index(
            genomic_position
        )
    except ValueError:
        return ""

    frame_offset = get_codon_start_offset(
        feature
    )

    coding_nt_index = (
        raw_index - frame_offset
    )

    if coding_nt_index < 0:
        return ""

    codon_number = (
        coding_nt_index // 3
        + 1
    )

    codon_offset = (
        coding_nt_index % 3
    )

    try:
        cds_sequence = str(
            feature.extract(record.seq)
        ).upper()
    except Exception:
        return ""

    if frame_offset:
        cds_sequence = (
            cds_sequence[frame_offset:]
        )

    codon_start = (
        (codon_number - 1) * 3
    )

    codon = cds_sequence[
        codon_start:
        codon_start + 3
    ]

    if len(codon) != 3:
        return ""

    if set(codon) - DNA_ALPHABET:
        return ""

    transl_table = feature.qualifiers.get(
        "transl_table",
        ["11"]
    )[0]

    try:
        transl_table = int(transl_table)
    except Exception:
        transl_table = 11

    try:
        reference_aa = str(
            Seq(codon).translate(
                table=transl_table
            )
        )
    except Exception:
        return ""

    feature_strand = (
        feature.location.strand
    )

    # Determine which base in coding orientation corresponds
    # to the reference alignment allele.
    ref_genomic_base = (
        nucleotide_to_genomic_orientation(
            reference_alignment_allele,
            context_orientation
        )
    )

    ref_coding_base = (
        genomic_to_coding_orientation(
            ref_genomic_base,
            feature_strand
        )
    )

    # Verify mapping when possible.
    if codon[codon_offset] != ref_coding_base:

        return (
            f"{reference_aa}"
            f"{codon_number}"
            f"?"
        )

    alternative_aas = []

    for allele in alignment_alleles:

        if allele == reference_alignment_allele:
            continue

        if allele not in DNA_ALPHABET:
            continue

        genomic_allele = (
            nucleotide_to_genomic_orientation(
                allele,
                context_orientation
            )
        )

        coding_allele = (
            genomic_to_coding_orientation(
                genomic_allele,
                feature_strand
            )
        )

        alternative_codon = list(codon)

        alternative_codon[
            codon_offset
        ] = coding_allele

        alternative_codon = "".join(
            alternative_codon
        )

        try:
            alt_aa = str(
                Seq(
                    alternative_codon
                ).translate(
                    table=transl_table
                )
            )
        except Exception:
            continue

        if alt_aa not in alternative_aas:
            alternative_aas.append(
                alt_aa
            )

    if not alternative_aas:

        return (
            f"{reference_aa}"
            f"{codon_number}"
            f"{reference_aa}"
        )

    # If all alternatives are synonymous:
    if set(alternative_aas) == {
        reference_aa
    }:

        return (
            f"{reference_aa}"
            f"{codon_number}"
            f"{reference_aa}"
        )

    # Include every possible translated state.
    return (
        f"{reference_aa}"
        f"{codon_number}"
        +
        ",".join(alternative_aas)
    )


# ============================================================
# SNP table + SNP annotation
# ============================================================

def create_snp_outputs(
        metadata,
        panaroo_out,
        output_root,
        reference_gbff):

    alignment_file = find_core_alignment(
        panaroo_out
    )

    print(
        f"\nReading core alignment:\n"
        f"{alignment_file}"
    )

    alignment = list(
        SeqIO.parse(
            str(alignment_file),
            "fasta"
        )
    )

    if not alignment:
        raise RuntimeError(
            "Core alignment is empty."
        )

    seqs = OrderedDict(
        (
            rec.id,
            str(rec.seq).upper()
        )
        for rec in alignment
    )

    sample_ids = [
        sid
        for sid in metadata
        if sid in seqs
    ]

    if not sample_ids:
        raise RuntimeError(
            "No Panaroo alignment IDs match metadata."
        )

    # Reference must be S0001
    if "S0001" not in seqs:

        raise RuntimeError(
            "S0001 is absent from "
            "core_gene_alignment.aln."
        )

    reference_alignment = seqs["S0001"]

    lengths = {
        len(seqs[sid])
        for sid in sample_ids
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "Alignment sequences have unequal lengths."
        )

    alignment_length = next(iter(lengths))

    print(
        f"Core alignment length: "
        f"{alignment_length:,} positions"
    )

    # --------------------------------------------------------
    # Find true SNP columns
    # --------------------------------------------------------

    snp_positions = []
    
    bar = progressbar.indicator(alignment_length,"SNP calling...")
    
    for pos in range(alignment_length):

        bases = [
            seqs[sid][pos]
            for sid in sample_ids
        ]

        base_set = set(bases)

        # Exclude gaps and ambiguous bases.
        if base_set.issubset(DNA_ALPHABET):
            if len(base_set) > 1:
                snp_positions.append(pos)

        # Update progress for every alignment position.
        bar(pos + 1)
    bar.stop()
    
    print(
        f"\nCore SNPs found: "
        f"{len(snp_positions):,}"
    )

    snp_names = [
        f"SNP_{i + 1:06d}"
        for i in range(
            len(snp_positions)
        )
    ]

    # --------------------------------------------------------
    # Genome x SNP matrix
    # --------------------------------------------------------

    rows = []
    bar = progressbar.indicator(len(sample_ids),"Create matrix...")
    counter = 0
    for sid in sample_ids:

        row = {
            "gbff_file":
                metadata[sid]["gbff_name"],

            "location":
                metadata[sid]["location"]
        }

        seq = seqs[sid]

        for snp_name, pos in zip(
            snp_names,
            snp_positions
        ):
            row[snp_name] = seq[pos]

        rows.append(row)
        counter += 1
        bar(counter)
    bar.stop()
    print()

    print("Loading matrix...\n")
    pd.DataFrame(rows).to_csv(
        output_root /
        "core_snps.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Load reference GBFF
    # --------------------------------------------------------

    print(
        f"\nReference genome:\n"
        f"{reference_gbff}"
    )

    reference_records = (
        load_reference_genome(
            reference_gbff
        )
    )

    # --------------------------------------------------------
    # SNP annotation
    # --------------------------------------------------------

    annotation_rows = []

    unmapped = 0
    ambiguous = 0

    bar = progressbar.indicator(len(snp_names),"SNP annotation")
    counter = 0
    for snp_name, pos in zip(
        snp_names,
        snp_positions
    ):

        alleles = []

        for sid in sample_ids:

            allele = seqs[sid][pos]

            if allele not in alleles:
                alleles.append(allele)

        reference_allele = (
            reference_alignment[pos]
        )

        # Put reference allele first
        alleles = (
            [reference_allele]
            +
            [
                a
                for a in alleles
                if a != reference_allele
            ]
        )

        alleles_text = "/".join(alleles)

        context = make_reference_context(
            reference_alignment,
            pos,
            SNP_CONTEXT_FLANK
        )

        cds = ""
        annotation = ""
        codon_change = ""
        reference_location = ""
        reference_strand = ""

        if context is None:

            context_text = ""

            annotation = (
                "context unavailable"
            )

            unmapped += 1

        else:

            context_text = context

            hits = find_context_in_reference(
                reference_records,
                context
            )

            if len(hits) == 0:

                annotation = "not mapped"

                unmapped += 1

            elif len(hits) > 1:

                annotation = (
                    f"ambiguous context "
                    f"({len(hits)} matches)"
                )

                ambiguous += 1

            else:

                hit = hits[0]

                record = hit["record"]

                genomic_position = (
                    hit["genomic_position"]
                )

                orientation = (
                    hit["context_orientation"]
                )

                # Reference coordinate is written as 1-based.
                reference_location = genomic_position + 1
                reference_strand = "+" if orientation == 1 else "-"

                cds_hits = (
                    find_cds_at_position(
                        record,
                        genomic_position
                    )
                )

                if not cds_hits:

                    cds = "non coding"
                    annotation = "non coding"
                    codon_change = ""

                else:

                    # Normally only one bacterial CDS should
                    # contain the position. If CDS overlap,
                    # report all locus tags but use the first
                    # feature for codon consequence.
                    locus_tags = [
                        get_locus_tag(feature)
                        for feature
                        in cds_hits
                    ]

                    locus_tags = [
                        x for x in locus_tags if x
                    ]

                    cds = ",".join(
                        locus_tags
                    )

                    feature = cds_hits[0]

                    annotation = (
                        get_annotation(feature)
                    )

                    codon_change = (
                        annotate_codon_change(
                            record=record,
                            feature=feature,
                            genomic_position=
                                genomic_position,
                            alignment_alleles=
                                alleles,
                            reference_alignment_allele=
                                reference_allele,
                            context_orientation=
                                orientation
                        )
                    )

        annotation_rows.append({
            "SNP":
                snp_name,

            "alignment_position":
                pos + 1,

            "Location in reference sequence":
                reference_location,

            "Strand in reference":
                reference_strand,

            "alleles":
                alleles_text,

            "CDS":
                cds,

            "annotation":
                annotation,

            "codon":
                codon_change,

            "context":
                context_text
        })
        
        counter += 1
        bar(counter)
    bar.stop()

    # --------------------------------------------------------
    # Write requested SNP-position file
    # --------------------------------------------------------

    positions_file = (
        output_root /
        "core_snp_positions.tsv"
    )

    with open(
        positions_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as out:

        out.write(
            f"#Reference genome: "
            f"{reference_gbff.name}\n"
        )

        pd.DataFrame(
            annotation_rows,
            columns=[
                "SNP",
                "alignment_position",
                "Location in reference sequence",
                "Strand in reference",
                "alleles",
                "CDS",
                "annotation",
                "codon",
                "context"
            ]
        ).to_csv(
            out,
            sep="\t",
            index=False
        )

    print(
        f"\nSNP matrix saved:\n"
        f"{output_root / 'core_snps.tsv'}"
    )

    print(
        f"\nSNP annotations saved:\n"
        f"{positions_file}"
    )

    print(
        f"\nUnmapped SNP contexts: {unmapped}"
    )

    print(
        f"Ambiguous SNP contexts: {ambiguous}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    project = args.project_folder

    INPUT_ROOT = (
        Path(args.input_folder)
        / project
    )

    TMP_ROOT = (
        Path(args.tmp_folder)
        / project
    )

    OUTPUT_ROOT = (
        Path(args.output_folder)
        / project
    )
    

    GBFF_DIR = (
        TMP_ROOT /
        "sequences"
    )

    GFF_DIR = (
        TMP_ROOT /
        "gff"
    )

    PANAROO_OUT = (
        OUTPUT_ROOT /
        "panaroo"
    )

    REF = ""
    if args.reference:
        REF = (
            GBFF_DIR /
            args.reference
        )
        if not REF.is_file():
            sys.exit(f"ERROR: Reference file does not exist: {REF}")            
    
    ensure_dir(OUTPUT_ROOT)

    # --------------------------------------------------------
    # Diversity range
    # --------------------------------------------------------

    if (
        args.minimal_diversity < 0
        or
        args.minimal_diversity >= 50
    ):

        raise ValueError(
            "--minimal_diversity must be "
            ">= 0 and < 50."
        )

    pmin = (
        args.minimal_diversity
        / 100.0
    )

    pmax = 1.0 - pmin

    print(
        f"Diversity range: "
        f"{pmin:.2f}–{pmax:.2f}"
    )

    # --------------------------------------------------------
    # Reconstruct metadata
    # --------------------------------------------------------

    if not project:
        raise ValueError("project_folder is required.")

    if args.input_format == "folders":
        files = discover_files(INPUT_ROOT)

        if not files:
            raise RuntimeError(
                f"No supported GenBank/FASTA genome files found under {INPUT_ROOT}"
            )

        print(f"Genome files found in hierarchical folders: {len(files)}")

        metadata = build_metadata(
            files,
            INPUT_ROOT
        )

    else:
        if not args.list_file:
            raise ValueError(
                "--list_file is required when --input_format list."
            )

        if not args.source_folder:
            raise ValueError(
                "--source_folder is required when --input_format list."
            )

        list_path = INPUT_ROOT / args.list_file
        if not list_path.is_file():
            raise FileNotFoundError(
                f"List file does not exist: {list_path}"
            )

        source_root = Path(args.source_folder)
        if not source_root.is_dir():
            raise NotADirectoryError(
                f"Source genome folder does not exist: {source_root}"
            )

        print(f"Reading hierarchy list: {list_path}")
        print(f"Resolving genome files under: {source_root}")

        metadata = build_metadata_from_list(
            list_path,
            source_root,
            LIST_LEVEL_DELIMITER
        )

        print(f"Genome files resolved from list: {len(metadata)}")

    save_metadata(
        metadata,
        OUTPUT_ROOT /
        "genome_metadata.tsv"
    )

    print(
        f"Metadata reconstructed for "
        f"{len(metadata)} genomes."
    )

    print("\nFirst mappings:")

    for sid in list(metadata.keys())[:5]:
        print(
            f"  {sid} -> "
            f"{metadata[sid]['gbff_name']} "
            f"[{metadata[sid]['location']}]"
        )

    # --------------------------------------------------------
    # Determine whether Panaroo should run
    # --------------------------------------------------------

    panaroo_matrix = (
        PANAROO_OUT /
        "gene_presence_absence.csv"
    )

    run_panaroo = (
        args.rerun_panaroo
        or
        not panaroo_matrix.exists()
    )

    if run_panaroo:

        print(
            "\nPreparing Panaroo input..."
        )

        ensure_dir(GBFF_DIR)
        ensure_dir(GFF_DIR)
        ensure_dir(PANAROO_OUT)

        prepare_sequence_files(
            metadata,
            INPUT_ROOT,
            GBFF_DIR
        )

        gffs = convert_all(
            metadata,
            GFF_DIR
        )

        # Remove genomes that failed conversion so all downstream matrices,
        # reference selection and SNP analysis use exactly the genomes supplied
        # to Panaroo.
        skipped = [sid for sid, info in metadata.items() if info.get("skip_reason")]
        if skipped:
            print(f"\nSkipped erroneous genome files: {len(skipped)}", file=sys.stderr)
            for sid in skipped:
                info = metadata[sid]
                print(
                    f"  {sid}: {info['gbff_name']} -- {info['skip_reason']}",
                    file=sys.stderr
                )
            metadata = OrderedDict(
                (sid, info) for sid, info in metadata.items()
                if not info.get("skip_reason")
            )
            save_metadata(
                metadata,
                OUTPUT_ROOT / "genome_metadata.tsv"
            )

        if len(gffs) < 2:
            raise RuntimeError(
                f"Only {len(gffs)} valid genome(s) remain after input validation; "
                "at least two are required for Panaroo."
            )

        run_cmd([
            "panaroo",

            "-i",
            *map(str, gffs),

            "-o",
            str(PANAROO_OUT),

            "-t",
            str(PANAROO_THREADS),

            "--clean-mode",
            "strict",

            "--core_threshold",
            str(CORE_THRESHOLD),

            "--remove-invalid-genes",

            "-a",
            "core"
        ])

    else:

        print(
            "\nExisting Panaroo results found."
        )

        print(
            "Panaroo will NOT be rerun."
        )

        # Attach paths to existing uncompressed GBFF files.
        missing_tmp = (
            attach_existing_tmp_paths(
                metadata,
                INPUT_ROOT,
                GBFF_DIR
            )
        )

        if missing_tmp:

            print(
                "\nWARNING: some prepared genome sequence "
                "files are missing from TMP."
            )

            print(
                "They will be recreated."
            )

            prepare_sequence_files(
                metadata,
                INPUT_ROOT,
                GBFF_DIR
            )

    # --------------------------------------------------------
    # Accessory genes
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "Accessory gene analysis"
    )

    print(
        "======================================"
    )

    create_accessory_matrix(
        metadata,
        PANAROO_OUT,
        OUTPUT_ROOT,
        pmin,
        pmax
    )

    # --------------------------------------------------------
    # Reference genome
    # --------------------------------------------------------
    if args.reference:
        reference_gbff = resolve_requested_reference(
            metadata,
            args.reference
        )
    else:
        reference_gbff = get_reference_gbff(
            metadata,
            INPUT_ROOT,
            GBFF_DIR
        )

    # --------------------------------------------------------
    # Core SNP analysis
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "Core-genome SNP analysis"
    )

    print(
        "======================================"
    )

    create_snp_outputs(
        metadata,
        PANAROO_OUT,
        OUTPUT_ROOT,
        reference_gbff
    )

    print(
        "\n======================================"
    )

    print("Done.")

    print(
        f"Results directory:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr
        )

        sys.exit(1)
