#!/usr/bin/env python3
"""
NCBI genome harvester:
- Searches NCBI Assembly for a taxon query (e.g. "Bacillus subtilis group[Organism]" or a TaxID)
- Builds a comprehensive metadata table (Assembly + BioSample + BioProject + linked PubMed)
- Downloads sequence files (genomic FASTA; optionally GenBank) for each assembly

Requirements:
  pip install biopython pandas requests lxml

NCBI best practice:
  Provide --email (required) and optionally --api-key for faster throughput.

Examples:
  # Search by organism term:
  python ncbi_harvesting.py --email you@uni.ac.za \
    --assembly-term "Bacillus subtilis group[Organism]" \
    --outdir out_bsubtilis_group --limit 200 --download-fna --download-gbff

  # Search by TaxID (replace 65393 with whatever you choose):
  python ncbi_harvesting.py --email you@uni.ac.za \
    --taxid 65393 --outdir out_taxid65393 --limit 500 --download-fna
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from Bio import Entrez
from lxml import etree


# ----------------------------
# Helpers / NCBI rate limiting
# ----------------------------

def sleep_polite(api_key: Optional[str]) -> None:
    # NCBI guideline: ~3 req/sec without API key, ~10 req/sec with key.
    time.sleep(0.12 if api_key else 0.35)


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_ftp(ftp_refseq: str, ftp_genbank: str, prefer_refseq: bool = True) -> str:
    ftp_refseq = ftp_refseq or ""
    ftp_genbank = ftp_genbank or ""
    if prefer_refseq and ftp_refseq.strip():
        return ftp_refseq.strip()
    if ftp_genbank.strip():
        return ftp_genbank.strip()
    return ftp_refseq.strip()  # fallback


def ftp_to_https(ftp_path: str) -> str:
    # NCBI often provides ftp://...; https works reliably in most environments
    if ftp_path.startswith("ftp://"):
        return "https://" + ftp_path[len("ftp://"):]
    return ftp_path


def download_file(url: str, outpath: Path, timeout: int = 60) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False
            mkdir(outpath.parent)
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


# ----------------------------
# XML parsing: BioSample
# ----------------------------

def parse_biosample_xml(xml_bytes: bytes) -> Dict[str, str]:
    """
    Parses BioSample XML and returns a flat dict of attributes.
    Keeps both common fields and any user-defined attributes.
    """
    out: Dict[str, str] = {}
    root = etree.fromstring(xml_bytes)

    # Accession
    acc = root.get("accession")
    if acc:
        out["biosample_accession"] = acc

    # Title/Description sometimes exist as <Title> or <Description>
    title = root.findtext(".//Title")
    if title:
        out["biosample_title"] = title.strip()

    # Owner / Organization
    org = root.findtext(".//Owner/Name")
    if org:
        out["biosample_owner_name"] = org.strip()

    # Attributes: <Attribute attribute_name="...">value</Attribute>
    for attr in root.findall(".//Attributes/Attribute"):
        key = (attr.get("attribute_name") or "").strip()
        val = (attr.text or "").strip()
        if not key:
            continue
        # normalize keys: spaces -> underscore, collapse punctuation
        norm_key = re.sub(r"\s+", "_", key)
        norm_key = re.sub(r"[^A-Za-z0-9_]+", "_", norm_key).strip("_").lower()
        if not norm_key:
            continue
        # If duplicates, join
        if norm_key in out and val:
            out[norm_key] = out[norm_key] + " | " + val
        else:
            out[norm_key] = val

    return out


# ----------------------------
# Entrez wrappers
# ----------------------------

def entrez_search(db: str, term: str, retmax: int, api_key: Optional[str]) -> List[str]:
    handle = Entrez.esearch(db=db, term=term, retmax=retmax, usehistory="n")
    rec = Entrez.read(handle)
    handle.close()
    sleep_polite(api_key)
    return list(rec.get("IdList", []))


def entrez_esummary(db: str, ids: List[str], api_key: Optional[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    handle = Entrez.esummary(db=db, id=",".join(ids), retmode="xml")
    rec = Entrez.read(handle)
    handle.close()
    sleep_polite(api_key)
    # rec is a list-like of DocumentSummary
    return [dict(x) for x in rec]


def entrez_efetch_xml(db: str, id_: str, api_key: Optional[str]) -> bytes:
    handle = Entrez.efetch(db=db, id=id_, retmode="xml")
    data = handle.read()
    handle.close()
    sleep_polite(api_key)
    return data.encode("utf-8") if isinstance(data, str) else data


def entrez_elink(dbfrom: str, db: str, id_: str, api_key: Optional[str]) -> List[str]:
    handle = Entrez.elink(dbfrom=dbfrom, db=db, id=id_)
    rec = Entrez.read(handle)
    handle.close()
    sleep_polite(api_key)

    ids: List[str] = []
    try:
        for linkset in rec:
            for lsdb in linkset.get("LinkSetDb", []):
                for link in lsdb.get("Link", []):
                    ids.append(link.get("Id"))
    except Exception:
        return []
    return [x for x in ids if x]


# ----------------------------
# Metadata objects
# ----------------------------

@dataclass
class AssemblyRow:
    assembly_uid: str = ""
    assembly_accession: str = ""
    assembly_name: str = ""
    organism: str = ""
    strain: str = ""
    submitter: str = ""
    bioproject: str = ""
    biosample: str = ""
    refseq_category: str = ""
    assembly_level: str = ""
    genome_representation: str = ""
    seq_rel_date: str = ""
    ftp_refseq: str = ""
    ftp_genbank: str = ""
    chosen_ftp: str = ""
    fna_url: str = ""
    gbff_url: str = ""
    local_fna: str = ""
    local_gbff: str = ""

    # BioProject
    bioproject_title: str = ""
    bioproject_description: str = ""

    # PubMed
    pubmed_pmids: str = ""
    pubmed_titles: str = ""

    # Derived / tagging
    application_tags: str = ""
    application_evidence: str = ""


# ----------------------------
# Application tagging (simple, extendable)
# ----------------------------

APPLICATION_KEYWORDS = {
    "probiotic": ["probiotic", "gut", "microbiome", "synbiotic"],
    "biocontrol": ["biocontrol", "biocontrol agent", "antifungal", "antagonistic", "biopesticide"],
    "plant_growth": ["plant growth", "pgpr", "rhizosphere", "root", "phyllosphere"],
    "fermentation_food": ["fermentation", "starter culture", "food", "soy", "natto", "dairy", "beer", "wine"],
    "enzyme_industrial": ["enzyme", "industrial", "biotech", "detergent", "protease", "amylase"],
    "aquaculture_feed": ["aquaculture", "fish", "shrimp", "feed", "pond"],
    "bioremediation": ["bioremediation", "wastewater", "pollution", "hydrocarbon", "heavy metal"],
}


def tag_applications(text: str) -> List[str]:
    t = (text or "").lower()
    tags: List[str] = []
    for tag, kws in APPLICATION_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                tags.append(tag)
                break
    return sorted(set(tags))


# ----------------------------
# Main pipeline
# ----------------------------

def build_assembly_term(args: argparse.Namespace) -> str:
    filters = []
    if args.latest:
        filters.append("latest[filter]")
    if args.exclude_surveillance:
        # crude example filter; adjust to your preference
        filters.append("NOT surveillance[All Fields]")

    base = ""
    if args.taxid:
        # NCBI syntax: txidXXXX[Organism:exp]
        base = f"txid{args.taxid}[Organism:exp]"
    else:
        base = args.assembly_term.strip()

    if args.only_refseq:
        filters.append("refseq[filter]")

    if args.only_genbank:
        filters.append("genbank[filter]")

    if args.assembly_level:
        # e.g., "complete genome" / "chromosome" / "scaffold" / "contig"
        # Assembly DB filter field varies; safest is keyword match in summary fields,
        # but we implement as All Fields term.
        filters.append(f"({args.assembly_level}[All Fields])")

    term = base
    if filters:
        term = f"({base}) AND " + " AND ".join(f"({f})" for f in filters)
    return term


def assembly_to_urls(chosen_ftp: str, assembly_accession: str) -> Tuple[str, str]:
    """
    Given chosen ftp base path, infer genomic FASTA and GenBank file URLs.
    NCBI naming: <basename>_genomic.fna.gz and <basename>_genomic.gbff.gz
    basename is last path component.
    """
    if not chosen_ftp:
        return "", ""
    base = chosen_ftp.rstrip("/").split("/")[-1]
    chosen_https = ftp_to_https(chosen_ftp.rstrip("/"))
    fna = f"{chosen_https}/{base}_genomic.fna.gz"
    gbff = f"{chosen_https}/{base}_genomic.gbff.gz"
    return fna, gbff

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="NCBI requires an email for Entrez usage")
    ap.add_argument("--api-key", default="", help="NCBI API key (optional)")
    ap.add_argument("--taxid", type=int, default=0, help="Taxonomy ID (preferred if you know it)")
    ap.add_argument("--assembly-term", default="",
                    help='Assembly search term (used if --taxid not set); example: Bacillus subtilis group[Organism]')
    ap.add_argument("--limit", type=int, default=20000, help="Max assemblies to retrieve")
    ap.add_argument("--outdir", default="ncbi_bacillus_harvest", help="Output folder")
    ap.add_argument("--prefer-refseq", action="store_true", help="Prefer RefSeq FTP path when available")
    ap.add_argument("--latest", action="store_true", help="Restrict to latest assemblies")
    ap.add_argument("--only-refseq", action="store_true", help="RefSeq-only filter")
    ap.add_argument("--only-genbank", action="store_true", help="GenBank-only filter")
    ap.add_argument("--assembly-level", default="", help="Optional text filter, e.g. 'Complete Genome'")
    ap.add_argument("--download-fna", action="store_true", help="Download genomic FASTA (.fna.gz)")
    ap.add_argument("--download-gbff", action="store_true", help="Download GenBank (.gbff.gz)")
    ap.add_argument("--max-download", type=int, default=20000, help="Max number of assemblies to download")
    ap.add_argument("--exclude-surveillance", action="store_true", help="Example exclusion toggle (optional)")
    return ap.parse_args() 

def main() -> int:
    # Read command-prompt arguments
    args = parse_args()
    
    Entrez.email = args.email
    if args.api_key.strip():
        Entrez.api_key = args.api_key.strip()

    outdir = Path(args.outdir)
    mkdir(outdir)
    mkdir(outdir / "genomes")

    term = build_assembly_term(args)
    print(f"[INFO] Assembly search term:\n  {term}\n", file=sys.stderr)

    # 1) Search Assembly
    assembly_uids = entrez_search("assembly", term, retmax=args.limit, api_key=args.api_key)
    print(f"[INFO] Found {len(assembly_uids)} assembly records (limited to {args.limit}).", file=sys.stderr)

    # 2) Summarize Assembly in batches
    rows: List[AssemblyRow] = []
    batch_size = 200
    for i in range(0, len(assembly_uids), batch_size):
        batch = assembly_uids[i:i + batch_size]
        summaries = entrez_esummary("assembly", batch, api_key=args.api_key)
        for s in summaries:
            # The esummary structure for Assembly has nested keys; these are commonly present:
            # - AssemblyAccession, AssemblyName, Organism, BioSampleAccn, BioProjectAccn
            # - FtpPath_RefSeq, FtpPath_GenBank
            r = AssemblyRow(
                assembly_uid=safe_text(s.get("Id", "")),
                assembly_accession=safe_text(s.get("AssemblyAccession", "")),
                assembly_name=safe_text(s.get("AssemblyName", "")),
                organism=safe_text(s.get("Organism", "")),
                submitter=safe_text(s.get("SubmitterOrganization", "")),
                bioproject=safe_text(s.get("BioProjectAccn", "")),
                biosample=safe_text(s.get("BioSampleAccn", "")),
                refseq_category=safe_text(s.get("RefSeq_category", "")),
                assembly_level=safe_text(s.get("AssemblyStatus", "")),
                genome_representation=safe_text(s.get("GenomeRepresentation", "")),
                seq_rel_date=safe_text(s.get("SeqReleaseDate", "")),
                ftp_refseq=safe_text(s.get("FtpPath_RefSeq", "")),
                ftp_genbank=safe_text(s.get("FtpPath_GenBank", "")),
            )

            chosen = pick_ftp(r.ftp_refseq, r.ftp_genbank, prefer_refseq=args.prefer_refseq)
            r.chosen_ftp = chosen
            r.fna_url, r.gbff_url = assembly_to_urls(chosen, r.assembly_accession)
            rows.append(r)

    # 3) Enrich with BioSample + BioProject + PubMed
    # Cache to avoid refetching duplicates
    biosample_cache: Dict[str, Dict[str, str]] = {}
    bioproject_cache: Dict[str, Tuple[str, str]] = {}
    pubmed_cache: Dict[str, Tuple[str, str]] = {}

    def enrich_biosample(biosample_acc: str) -> Dict[str, str]:
        if not biosample_acc:
            return {}
        if biosample_acc in biosample_cache:
            return biosample_cache[biosample_acc]
        # BioSample efetch uses numeric IDs OR accessions; accessions typically work.
        xml = entrez_efetch_xml("biosample", biosample_acc, api_key=args.api_key)
        d = parse_biosample_xml(xml)
        biosample_cache[biosample_acc] = d
        return d

    def enrich_bioproject(bioproject_acc: str) -> Tuple[str, str]:
        if not bioproject_acc:
            return ("", "")
        if bioproject_acc in bioproject_cache:
            return bioproject_cache[bioproject_acc]
        # BioProject accessions look like PRJNAxxxx; esummary expects IDs too but often accepts accessions.
        try:
            s = entrez_esummary("bioproject", [bioproject_acc], api_key=args.api_key)
            if s:
                title = safe_text(s[0].get("Project_Title", "")) or safe_text(s[0].get("Title", ""))
                desc = safe_text(s[0].get("Project_Description", "")) or safe_text(s[0].get("Description", ""))
            else:
                title, desc = "", ""
        except Exception:
            title, desc = "", ""
        bioproject_cache[bioproject_acc] = (title, desc)
        return title, desc

    def enrich_pubmed(assembly_uid: str) -> Tuple[str, str]:
        if not assembly_uid:
            return ("", "")
        if assembly_uid in pubmed_cache:
            return pubmed_cache[assembly_uid]
        pmids = entrez_elink("assembly", "pubmed", assembly_uid, api_key=args.api_key)
        pmids = pmids[:20]  # cap
        titles: List[str] = []
        if pmids:
            try:
                sums = entrez_esummary("pubmed", pmids, api_key=args.api_key)
                for rec in sums:
                    t = safe_text(rec.get("Title", "")).strip()
                    if t:
                        titles.append(t)
            except Exception:
                pass
        pmid_str = ",".join(pmids)
        title_str = " | ".join(titles[:10])
        pubmed_cache[assembly_uid] = (pmid_str, title_str)
        return pmid_str, title_str

    enriched_records: List[Dict[str, Any]] = []
    for r in rows:
        bios = enrich_biosample(r.biosample)
        bpt, bpd = enrich_bioproject(r.bioproject)
        pmids, pmtitles = enrich_pubmed(r.assembly_uid)

        r.bioproject_title = bpt
        r.bioproject_description = bpd
        r.pubmed_pmids = pmids
        r.pubmed_titles = pmtitles

        # Try to grab "strain" from BioSample if present; fall back to AssemblyName heuristics
        strain = ""
        for k in ("strain", "isolate", "sample_name"):
            if k in bios and bios[k]:
                strain = bios[k]
                break
        r.strain = strain or r.assembly_name

        # Application tagging from combined text (BioSample + BioProject + PubMed titles)
        combined = " | ".join([
            safe_text(bios.get("isolation_source", "")),
            safe_text(bios.get("host", "")),
            safe_text(bios.get("environment", "")),
            safe_text(bios.get("biomaterial_provider", "")),
            safe_text(bios.get("biosample_title", "")),
            safe_text(r.bioproject_title),
            safe_text(r.bioproject_description),
            safe_text(r.pubmed_titles),
        ])
        tags = tag_applications(combined)
        r.application_tags = ",".join(tags)

        # Evidence summary
        ev = []
        if tags:
            if any(kw in (safe_text(bios.get("isolation_source", "")) + " " + safe_text(bios.get("biosample_title", ""))).lower()
                   for kws in APPLICATION_KEYWORDS.values() for kw in kws):
                ev.append("BioSample")
            if any(kw in (safe_text(r.bioproject_title) + " " + safe_text(r.bioproject_description)).lower()
                   for kws in APPLICATION_KEYWORDS.values() for kw in kws):
                ev.append("BioProject")
            if any(kw in safe_text(r.pubmed_titles).lower()
                   for kws in APPLICATION_KEYWORDS.values() for kw in kws):
                ev.append("PubMed_titles")
        r.application_evidence = ",".join(sorted(set(ev)))

        record = asdict(r)
        # merge in all BioSample attributes (prefixed to avoid collisions)
        for k, v in bios.items():
            record[f"biosample_attr__{k}"] = v
        enriched_records.append(record)

    df = pd.DataFrame(enriched_records)

    # 4) Download sequences (optional)
    downloads = 0
    if args.download_fna or args.download_gbff:
        for idx, rec in df.iterrows():
            if downloads >= args.max_download:
                break
            acc = rec.get("assembly_accession", "")
            base_dir = outdir / "genomes" / safe_text(acc)
            mkdir(base_dir)

            if args.download_fna and rec.get("fna_url"):
                out_fna = base_dir / f"{acc}_genomic.fna.gz"
                if not out_fna.exists():
                    ok = download_file(rec["fna_url"], out_fna)
                    if ok:
                        df.at[idx, "local_fna"] = str(out_fna)
                downloads += 1

            if downloads >= args.max_download:
                break

            if args.download_gbff and rec.get("gbff_url"):
                out_gbff = base_dir / f"{acc}_genomic.gbff.gz"
                if not out_gbff.exists():
                    ok = download_file(rec["gbff_url"], out_gbff)
                    if ok:
                        df.at[idx, "local_gbff"] = str(out_gbff)
                downloads += 1

    # 5) Save table
    out_csv = outdir / "metadata_table.csv"
    out_tsv = outdir / "metadata_table.tsv"
    df.to_csv(out_csv, index=False)
    df.to_csv(out_tsv, index=False, sep="\t")

    print(f"[DONE] Wrote:\n  {out_csv}\n  {out_tsv}", file=sys.stderr)
    if args.download_fna or args.download_gbff:
        print(f"[DONE] Download attempts: {min(downloads, args.max_download)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
