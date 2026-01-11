import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.error import URLError, HTTPError
import json
import re

from io import StringIO

import requests
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("docking_preparation")


@dataclass
class TaxonInfo:
    scientific_name: str
    taxid: int
    rank: str
    genus: str


@dataclass
class ProteinCandidate:
    summary_accession: str  # accession from esummary (may lack version)
    length: int
    title: str
    organism: str
    taxid: Optional[int]
    score_class: str
    same_species: bool


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    logger.debug("Logging initialised with verbosity level %s", verbosity)


def resolve_taxonomy(organism: str, email: str) -> TaxonInfo:
    """
    Resolve an organism string to NCBI Taxonomy information.
    """
    assert "@" in email, "NCBI requires a valid email address in Entrez.email"
    Entrez.email = email
    Entrez.tool = "docking_preparation"

    logger.info("Resolving organism in NCBI Taxonomy: '%s'", organism)
    try:
        handle = Entrez.esearch(db="taxonomy", term=organism)
        record = Entrez.read(handle)
    except (HTTPError, URLError, OSError) as e:
        logger.error("Network error when querying NCBI Taxonomy: %s", e)
        raise SystemExit(1)

    id_list = record.get("IdList", [])
    if not id_list:
        logger.error("No taxonomy record found for organism '%s'.", organism)
        raise SystemExit(1)

    taxid = int(id_list[0])
    try:
        handle = Entrez.efetch(db="taxonomy", id=str(taxid), retmode="xml")
        records = Entrez.read(handle)
    except (HTTPError, URLError, OSError) as e:
        logger.error("Network error when fetching NCBI Taxonomy details: %s", e)
        raise SystemExit(1)

    if not records:
        logger.error("Failed to fetch taxonomy details for taxid %s.", taxid)
        raise SystemExit(1)

    tax = records[0]
    sci_name = tax["ScientificName"]
    rank = tax.get("Rank", "unknown")
    genus = sci_name.split()[0]

    logger.info(
        "NCBI Taxonomy resolved: '%s' (taxid=%s, rank=%s).",
        sci_name,
        taxid,
        rank,
    )

    return TaxonInfo(scientific_name=sci_name, taxid=taxid, rank=rank, genus=genus)


def expand_keywords(user_keywords: str) -> List[str]:
    """
    Expand user-supplied keywords with common CYP51 / ERG11 synonyms.
    """
    base = [kw.strip() for kw in user_keywords.split(",") if kw.strip()]
    extra = [
        "sterol 14-alpha demethylase",
        "sterol 14-demethylase",
        "lanosterol 14-alpha demethylase",
        "lanosterol 14α-demethylase",
        "cytochrome P450 51",
        "cytochrome P450 family 51",
        "CYP51",
        "ERG11",
    ]
    # Ensure uniqueness, preserve order (user first, then extras).
    seen = set()
    expanded: List[str] = []
    for kw in base + extra:
        if kw not in seen:
            expanded.append(kw)
            seen.add(kw)
    logger.info("Using the following search keywords: %s", "; ".join(expanded))
    return expanded


def classify_protein_title(title: str) -> str:
    """
    Very lightweight heuristic classification of protein annotation.
    Used only to help the user choose plausible CYP51 candidates.
    """
    t = title.lower()
    if "cytochrome p450 51" in t or "sterol 14" in t or "eburicol 14" in t:
        return "CYP51-LIKE"
    if "p450" in t:
        return "OTHER-P450"
    return "OTHER"


def _search_ncbi_protein_internal(
    organism_term: str, keywords: List[str], retmax: int = 50
) -> List[ProteinCandidate]:
    """
    Search NCBI Protein for a set of CYP51-related keywords restricted to an
    organism term, returning a list of ProteinCandidate objects.
    """
    term_keywords = " OR ".join([f'"{kw}"[Title]' for kw in keywords])
    query = f"({term_keywords}) AND {organism_term}[Organism]"
    logger.debug("NCBI Protein search term: %s", query)

    try:
        handle = Entrez.esearch(db="protein", term=query, retmax=retmax)
        record = Entrez.read(handle)
    except (HTTPError, URLError, OSError) as e:
        logger.error("Network error when searching NCBI Protein: %s", e)
        raise SystemExit(1)

    ids = record.get("IdList", [])
    if not ids:
        return []

    try:
        handle = Entrez.esummary(db="protein", id=",".join(ids))
        summaries = Entrez.read(handle)
    except (HTTPError, URLError, OSError) as e:
        logger.error("Network error when fetching NCBI Protein summaries: %s", e)
        raise SystemExit(1)

    candidates: List[ProteinCandidate] = []
    for summ in summaries:
        caption = str(summ.get("Caption", ""))
        title = str(summ.get("Title", ""))
        length = int(summ.get("Length", 0) or 0)
        organism = str(summ.get("Organism", ""))
        taxid = int(summ.get("TaxId", 0) or 0) or None

        score_class = classify_protein_title(title)
        candidates.append(
            ProteinCandidate(
                summary_accession=caption,
                length=length,
                title=title,
                organism=organism,
                taxid=taxid,
                score_class=score_class,
                same_species=False,  # filled later by caller
            )
        )

    return candidates


def find_cyp51_candidates(
    taxon: TaxonInfo, keywords: List[str], search_level: str = "auto"
) -> Tuple[List[ProteinCandidate], str]:
    """
    Try to find CYP51-like proteins at species level; if none, broaden to genus.
    Returns (candidates, level_used).
    """
    level_used = "species"
    org_term_species = taxon.scientific_name
    logger.info("Searching NCBI Protein at species level ...")
    candidates = _search_ncbi_protein_internal(org_term_species, keywords)

    # Mark same-species flag
    for c in candidates:
        c.same_species = True

    if not candidates and search_level in ("auto", "genus"):
        logger.warning(
            "No hits found in NCBI Protein for the query at species level."
        )
        logger.warning(
            "Broadening search from species to genus (%s). If you did not intend this, "
            "consider using --search-level species.",
            taxon.genus,
        )
        level_used = "genus"
        org_term_genus = taxon.genus
        logger.info(
            "Searching NCBI Protein at genus level (%s) ...", org_term_genus
        )
        candidates = _search_ncbi_protein_internal(org_term_genus, keywords)
        # Genus-level hits are, by construction, not same-species.
        for c in candidates:
            c.same_species = False

    # Filter to plausible CYP51-like hits first.
    plausible = [c for c in candidates if c.score_class == "CYP51-LIKE"]
    if plausible:
        candidates = plausible
    else:
        logger.warning(
            "No clearly annotated CYP51-like proteins were found; "
            "showing broader cytochrome P450 hits (may be poor docking targets)."
        )

    if not candidates:
        logger.error(
            "No candidate proteins found in NCBI Protein. "
            "You may need to adjust your organism or keywords."
        )
        raise SystemExit(1)

    logger.info(
        "Candidate protein(s) retrieved (filtered to plausible CYP51-like hits where possible):"
    )
    logger.info(
        "Idx  Accession        Len  Class       SameSp?  Organism / Title"
    )
    logger.info("-" * 74)
    for i, c in enumerate(candidates):
        same = "yes" if c.same_species else "no"
        logger.info(
            "%3d  %-15s %5d  %-10s %-7s | %s [%s]",
            i,
            c.summary_accession,
            c.length,
            c.score_class,
            same,
            c.organism or "Unknown organism",
            c.title,
        )

    return candidates, level_used


def prompt_user_choice(
    candidates: List[ProteinCandidate],
    target_organism: str,
    allow_cross_species: bool,
) -> ProteinCandidate:
    """
    Interactively ask the user to choose a candidate. If the chosen one is not
    from the same species and cross-species docking is allowed, warn and ask
    for confirmation.
    """
    assert candidates, "Internal error: no candidates to choose from."

    while True:
        idx_raw = input(
            "Enter the index (Idx) of the protein you want to use for docking "
            "(or 'q' to abort): "
        ).strip()
        if idx_raw.lower() in {"q", "quit", "exit"}:
            logger.info("User aborted selection. Exiting.")
            raise SystemExit(0)
        if not idx_raw.isdigit():
            logger.warning("Please enter a valid integer index.")
            continue

        idx = int(idx_raw)
        if idx < 0 or idx >= len(candidates):
            logger.warning(
                "Index %d is out of range (0–%d). Please choose a valid candidate.",
                idx,
                len(candidates) - 1,
            )
            continue

        cand = candidates[idx]
        logger.info(
            "You selected %s (%s, length %d, organism '%s').",
            cand.summary_accession,
            cand.score_class,
            cand.length,
            cand.organism or "",
        )

        if not cand.same_species:
            msg = (
                "WARNING: Selected protein is from a different species "
                f"('{cand.organism or 'unknown'}') than your target "
                f"('{target_organism}')."
            )
            logger.warning(msg)
            if not allow_cross_species:
                logger.warning(
                    "Cross-species docking is currently disallowed. "
                    "Either re-choose a same-species candidate or "
                    "re-run with --allow-cross-species."
                )
                continue

            confirm = input(
                "Type 'yes' to confirm this cross-species choice anyway, "
                "anything else to re-choose: "
            ).strip()
            if confirm.lower() != "yes":
                logger.info("Cross-species choice not confirmed. Please re-choose.")
                continue

        return cand


def fetch_protein_fasta(
    accession: str, output_dir: str
) -> Tuple[str, str, str]:
    """
    Download FASTA sequence from NCBI Protein given a RefSeq accession or GI.
    Returns (fasta_path, sequence, full_accession_with_version).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Fetching FASTA for %s from NCBI Protein.", accession)
    try:
        handle = Entrez.efetch(
            db="protein", id=accession, rettype="fasta", retmode="text"
        )
        fasta_text = handle.read()
    except (HTTPError, URLError, OSError) as e:
        logger.error("Network error when fetching FASTA from NCBI Protein: %s", e)
        raise SystemExit(1)

    if not fasta_text.strip():
        logger.error("Received empty FASTA for accession '%s'.", accession)
        raise SystemExit(1)

    # Parse with SeqIO to recover a canonical accession with version.
    record = next(SeqIO.parse(StringIO(fasta_text), "fasta"))
    header = record.id  # e.g. "ref|XP_002845046.1|" or "XP_002845046.1"
    # Extract full accession with version.
    m = re.search(r"(XP_\d+\.\d+)", header)
    if m:
        full_acc = m.group(1)
    else:
        full_acc = header

    seq = str(record.seq)
    logger.info("Sequence length for selected protein: %d aa", len(seq))

    out_path = os.path.join(output_dir, f"{full_acc}.fasta")
    with open(out_path, "w") as f:
        f.write(fasta_text)
    logger.info("Saved FASTA to %s", out_path)

    return out_path, seq, full_acc


def detect_heme_motif(seq: str) -> bool:
    """
    Detect classical CYP heme-binding motif FxxGxxxCxG (allowing some variation).
    """
    motif_regex = re.compile(r"F..G...C.G", re.IGNORECASE)
    return motif_regex.search(seq) is not None


def detect_exxr_motif(seq: str) -> bool:
    """
    Detect EXXR motif in the conserved I-helix region of P450s.
    """
    motif_regex = re.compile(r"E..R", re.IGNORECASE)
    return motif_regex.search(seq) is not None


def run_blast_validation(
    seq: str, min_identity: float, min_coverage: float
) -> Optional[dict]:
    """
    Run BLASTP against NCBI nr restricted to fungi and summarise the top hits.
    Returns a dict with details of the best hit, or None on failure.
    """
    assert 0.0 <= min_identity <= 100.0
    assert 0.0 <= min_coverage <= 100.0

    logger.info(
        "Submitting BLASTP job to NCBI (database='nr', fungi[organism]; this may "
        "take a few minutes and is subject to NCBI usage limits)..."
    )
    try:
        result_handle = NCBIWWW.qblast(
            "blastp",
            "nr",
            seq,
            entrez_query="fungi[organism]",
            hitlist_size=25,
            expect=1e-20,
        )
        blast_xml = result_handle.read()
    except (HTTPError, URLError, OSError) as e:
        logger.error("Network error during BLASTP: %s", e)
        return None

    if not blast_xml.strip():
        logger.error("Received empty BLASTP result from NCBI.")
        return None

    blast_record = NCBIXML.read(StringIO(blast_xml))

    if not blast_record.alignments:
        logger.warning("BLASTP returned no alignments.")
        return None

    seq_len = len(seq)
    logger.info("Top BLASTP fungal hits (up to 5):")
    top_best = None

    for i, alignment in enumerate(blast_record.alignments[:5]):
        hsp = alignment.hsps[0]
        identity_pct = 100.0 * hsp.identities / hsp.align_length
        coverage_pct = 100.0 * hsp.align_length / seq_len
        desc = alignment.title
        logger.info(
            "  Hit %d: %s | len=%d | identities=%.1f%% | coverage=%.1f%%",
            i,
            desc,
            alignment.length,
            identity_pct,
            coverage_pct,
        )
        if i == 0:
            top_best = {
                "description": desc,
                "length": alignment.length,
                "identity_pct": identity_pct,
                "coverage_pct": coverage_pct,
            }

    if top_best:
        passed = (
            top_best["identity_pct"] >= min_identity
            and top_best["coverage_pct"] >= min_coverage
        )
        if passed:
            logger.info(
                "BLAST validation OK: best fungal hit has identity %.1f%% and "
                "coverage %.1f%% (>= thresholds).",
                top_best["identity_pct"],
                top_best["coverage_pct"],
            )
        else:
            logger.warning(
                "BLAST validation WARNING: best fungal hit has identity %.1f%% and "
                "coverage %.1f%%, which does not meet thresholds "
                "(identity>=%.1f%%, coverage>=%.1f%%). "
                "You should consider a different target or relax thresholds "
                "only with strong justification.",
                top_best["identity_pct"],
                top_best["coverage_pct"],
                min_identity,
                min_coverage,
            )
        top_best["passed"] = passed

    return top_best


import urllib.parse
import requests
import logging

logger = logging.getLogger(__name__)

def map_refseq_to_uniprot(refseq_acc: str) -> Optional[str]:
    """
    Map a RefSeq protein accession (e.g. XP_002845046.1) to a UniProt accession
    using the UniProt REST API.

    Returns the first UniProt accession found, or None if mapping fails.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    # Recommended syntax in current UniProt REST: database:RefSeq:ACCESSION
    query_str = f"database:RefSeq:{refseq_acc}"
    params = {
        "query": query_str,
        "fields": "accession,protein_name,organism_name",
        "format": "json",
        "size": 1,
    }

    logger.info(f"Attempting to map RefSeq accession {refseq_acc} -> UniProt via UniProt REST API.")
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to query UniProt: {e}")
        return None

    data = resp.json()
    results = data.get("results", [])
    if not results:
        logger.warning(f"No UniProt mapping found for RefSeq {refseq_acc}.")
        return None

    primary_acc = results[0]["primaryAccession"]
    protein_desc = results[0].get("proteinDescription", {})
    rec_name = protein_desc.get("recommendedName", {}).get("fullName", {}).get("value", "")
    org = results[0].get("organism", {}).get("scientificName", "")

    logger.info(
        f"Mapped RefSeq {refseq_acc} -> UniProt {primary_acc} "
        f"({rec_name} from {org})."
    )
    return primary_acc

def download_alphafold_model(uniprot_acc: str, output_dir: str) -> Optional[str]:
    """
    Download an AlphaFold model (if available) for a given UniProt accession.
    """
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_acc}-F1-model_v4.pdb"
    logger.info(
        "Attempting to download AlphaFold model from %s",
        url,
    )
    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as e:
        logger.error("Network error while trying to download AlphaFold model: %s", e)
        return None

    if resp.status_code == 404:
        logger.warning(
            "AlphaFold model not found for UniProt accession %s (HTTP 404).",
            uniprot_acc,
        )
        return None
    if resp.status_code != 200:
        logger.error(
            "Unexpected HTTP status %s when fetching AlphaFold model for %s.",
            resp.status_code,
            uniprot_acc,
        )
        return None

    out_path = os.path.join(output_dir, f"AF-{uniprot_acc}-F1-model_v4.pdb")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    logger.info("AlphaFold model saved to %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Semi-automated preparation of a CYP51 docking target: "
            "search NCBI Protein by organism & CYP51 keywords, let the user "
            "choose plausible candidates, check CYP51 motifs and (optionally) "
            "validate by BLAST and retrieve AlphaFold structure."
        )
    )

    parser.add_argument(
        "--email",
        required=True,
        help="Your email address (required by NCBI Entrez).",
    )
    parser.add_argument(
        "--organism",
        required=True,
        help="Target organism (e.g. 'Microsporum audouinii').",
    )
    parser.add_argument(
        "--keywords",
        default="sterol 14-alpha demethylase,cytochrome P450 51,CYP51,ERG11",
        help="Comma-separated CYP51-related keywords.",
    )
    parser.add_argument(
        "--min-identity",
        type=float,
        default=30.0,
        help="Minimum BLAST identity threshold (%%) for validation.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=70.0,
        help="Minimum BLAST coverage threshold (%%) for validation.",
    )
    parser.add_argument(
        "--output-dir",
        default="docking_target",
        help="Directory where the chosen FASTA (and optionally AlphaFold PDB) will be saved.",
    )
    parser.add_argument(
        "--alphafold",
        action="store_true",
        help="Attempt to map RefSeq to UniProt and download AlphaFold 3D structure.",
    )
    parser.add_argument(
        "--blast-validate",
        action="store_true",
        help="Run BLASTP against fungal proteins to confirm CYP51-like identity.",
    )
    parser.add_argument(
        "--allow-cross-species",
        action="store_true",
        help=(
            "Allow selection of proteins from a different species than the target "
            "organism (useful when only genus-level data exist). You will still "
            "be explicitly warned and asked to confirm."
        ),
    )
    parser.add_argument(
        "--search-level",
        choices=["auto", "species", "genus"],
        default="auto",
        help=(
            "How strictly to restrict NCBI search to your organism. "
            "'auto' = species first then genus if nothing is found."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (use -vv for debug-level logs).",
    )

    args = parser.parse_args()

    # Basic sanity checks / assertions that reviewers will appreciate.
    assert 0.0 <= args.min_identity <= 100.0, "--min-identity must be between 0 and 100."
    assert 0.0 <= args.min_coverage <= 100.0, "--min-coverage must be between 0 and 100."
    assert "@" in args.email, "--email must look like a valid email (required by NCBI)."

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("Starting docking_preparation.py")
    logger.info(
        "Organism: %s | Keywords: %s | min-identity: %.1f | min-coverage: %.1f",
        args.organism,
        args.keywords,
        args.min_identity,
        args.min_coverage,
    )

    taxon = resolve_taxonomy(args.organism, args.email)
    keywords = expand_keywords(args.keywords)

    candidates, _ = find_cyp51_candidates(
        taxon, keywords, search_level=args.search_level
    )

    chosen = prompt_user_choice(
        candidates,
        target_organism=taxon.scientific_name,
        allow_cross_species=args.allow_cross_species,
    )

    fasta_path, seq, full_acc = fetch_protein_fasta(
        chosen.summary_accession, args.output_dir
    )

    # Motif checks
    has_heme = detect_heme_motif(seq)
    has_exxr = detect_exxr_motif(seq)

    if has_heme:
        logger.info("Heme-binding motif (FxxGxxxCxG-like) detected.")
    else:
        logger.warning(
            "Heme-binding motif (FxxGxxxCxG-like) NOT detected. "
            "This is unusual for CYP51 and should be justified."
        )

    if has_exxr:
        logger.info("EXXR motif (E..R) detected.")
    else:
        logger.warning(
            "EXXR motif (E..R) NOT detected. This is unusual for CYP51 "
            "and should be justified."
        )

    if not (400 <= len(seq) <= 600):
        logger.warning(
            "Sequence length (%d aa) is outside the typical range for CYP51 "
            "(~500 aa). Reviewers may ask you to justify this choice.",
            len(seq),
        )
    else:
        logger.info(
            "Sequence length (%d aa) is in a reasonable range for CYP51 (~500 aa).",
            len(seq),
        )

    blast_summary = None
    if args.blast_validate:
        blast_summary = run_blast_validation(
            seq, min_identity=args.min_identity, min_coverage=args.min_coverage
        )

    alphafold_path = None
    uniprot_acc = None
    if args.alphafold:
        uniprot_acc = map_refseq_to_uniprot(full_acc)
        if uniprot_acc:
            alphafold_path = download_alphafold_model(uniprot_acc, args.output_dir)

    logger.info(
        "Docking target preparation complete. Chosen protein: %s (%s) from '%s', "
        "sequence length %d aa, FASTA saved to %s.",
        full_acc,
        chosen.title,
        chosen.organism or "",
        len(seq),
        fasta_path,
    )

    logger.info(
        "Motif summary: heme motif=%s | EXXR motif=%s",
        bool(has_heme),
        bool(has_exxr),
    )

    if blast_summary is not None:
        logger.info(
            "BLAST summary: best fungal hit '%s' with identity %.1f%% and coverage "
            "%.1f%% (thresholds: identity>=%.1f%%, coverage>=%.1f%%).",
            blast_summary.get("description", "N/A"),
            blast_summary.get("identity_pct", float("nan")),
            blast_summary.get("coverage_pct", float("nan")),
            args.min_identity,
            args.min_coverage,
        )

    if uniprot_acc:
        if alphafold_path:
            logger.info(
                "AlphaFold summary: UniProt=%s, model downloaded to %s.",
                uniprot_acc,
                alphafold_path,
            )
        else:
            logger.info(
                "AlphaFold summary: UniProt=%s, but no AlphaFold model was retrieved.",
                uniprot_acc,
            )

    logger.info(
        "Please inspect the logs above (motifs, BLAST validation if run, "
        "cross-species status, UniProt/AlphaFold mapping) before presenting this "
        "as your final docking target to reviewers."
    )


if __name__ == "__main__":
    main()
