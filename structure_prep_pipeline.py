import os
import sys
import argparse
import logging
import subprocess
import shutil
import glob
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from Bio import SeqIO, PDB, pairwise2
from Bio.Blast import NCBIWWW, NCBIXML

import urllib.request

try:
    import numpy as np
except ImportError:
    np = None  # RMSD and water-distance features will be skipped if NumPy is missing


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("structure_prep_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TemplateCandidate:
    pdb_id: str
    chain: str
    identity: float
    coverage: float
    title: str


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ensure_outdir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created output directory: {path}")


def read_fasta(path: str) -> Tuple[str, str]:
    assert os.path.isfile(path), f"FASTA file not found: {path}"
    records = list(SeqIO.parse(path, "fasta"))
    assert records, f"No sequence records found in FASTA: {path}"
    if len(records) > 1:
        logger.warning(f"FASTA contains {len(records)} sequences; using the first one.")
    rec = records[0]
    seq_str = str(rec.seq).replace("*", "").strip()
    logger.info(f"Read FASTA: id={rec.id}, length={len(seq_str)} aa")
    return rec.id, seq_str


# ---------------------------------------------------------------------------
# BLAST vs PDB to find templates
# ---------------------------------------------------------------------------

def run_blastp_pdb(sequence: str, out_xml: str) -> None:
    """Run BLASTP against PDB database at NCBI and save XML."""
    logger.info("Submitting BLASTP job against PDB database at NCBI...")
    # You may add an entrez_query if you want to further restrict (e.g. fungi/CYP51)
    handle = NCBIWWW.qblast("blastp", "pdb", sequence)
    xml_data = handle.read()
    with open(out_xml, "w") as fh:
        fh.write(xml_data)
    logger.info(f"BLASTP XML saved to: {out_xml}")


def _parse_pdb_id_and_chain(title: str) -> Tuple[str, str]:
    """
    Parse PDB id and chain from BLAST alignment.title.
    Typically: 'pdb|5FRB|A Chain A, ...'
    """
    token = title.split()[0]  # e.g. 'pdb|5FRB|A'
    parts = token.split("|")
    pdb_id = ""
    chain = "A"
    if len(parts) >= 3 and parts[0].lower() == "pdb":
        pdb_id = parts[1].upper()
        chain = parts[2] if parts[2] else "A"
    else:
        pdb_id = token[:4].upper()
        chain = token[4] if len(token) > 4 else "A"
    return pdb_id, chain


def parse_blastp_xml_for_templates(
    xml_path: str,
    query_len: int,
    max_hits: int = 25,
    min_identity: float = 30.0,
    min_coverage: float = 70.0,
) -> List[TemplateCandidate]:
    assert os.path.isfile(xml_path), f"BLAST XML not found: {xml_path}"
    logger.info("Parsing BLASTP XML for template candidates...")
    candidates: List[TemplateCandidate] = []

    with open(xml_path) as fh:
        blast_records = list(NCBIXML.parse(fh))

    if not blast_records:
        logger.warning("No BLAST records found.")
        return []

    count = 0
    for record in blast_records:
        for alignment in record.alignments:
            if count >= max_hits:
                break
            if not alignment.hsps:
                continue
            hsp = alignment.hsps[0]
            identity = 100.0 * float(hsp.identities) / float(hsp.align_length)
            coverage = 100.0 * float(hsp.align_length) / float(query_len)
            if identity < min_identity or coverage < min_coverage:
                continue
            pdb_id, chain = _parse_pdb_id_and_chain(alignment.title)
            candidates.append(
                TemplateCandidate(
                    pdb_id=pdb_id,
                    chain=chain,
                    identity=identity,
                    coverage=coverage,
                    title=alignment.title.strip(),
                )
            )
            count += 1
            if count >= max_hits:
                break

    if not candidates:
        logger.warning(
            "No template candidates passed the filters "
            f"(identity>={min_identity}%, coverage>={min_coverage}%)."
        )
        return []

    # Sort: highest identity first, then highest coverage
    candidates.sort(key=lambda c: (-c.identity, -c.coverage))
    logger.info("Top candidate PDB templates (after filters):")
    logger.info("Idx  PDB   Ch  Id%%   Cov%%  Title")
    logger.info("-------------------------------------------------------------")
    for idx, c in enumerate(candidates[:10]):
        logger.info(
            f"{idx:3d}  {c.pdb_id:4s}  {c.chain:1s}  "
            f"{c.identity:4.1f}  {c.coverage:5.1f}  {c.title[:60]}"
        )
    return candidates


def choose_best_template(candidates: List[TemplateCandidate]) -> Optional[TemplateCandidate]:
    """
    Automatically choose the 'best' template (no interactive input).
    Criteria:
      1) Highest identity
      2) Highest coverage
    """
    if not candidates:
        logger.warning("No candidates to choose from.")
        return None
    best = candidates[0]
    logger.info(
        "Automatically selected template "
        f"{best.pdb_id} chain {best.chain} (identity={best.identity:.1f}%, "
        f"coverage={best.coverage:.1f}%)."
    )
    return best


def download_pdb(pdb_id: str, outdir: str) -> str:
    """Download PDB from RCSB if not already present."""
    ensure_outdir(outdir)
    out_path = os.path.join(outdir, f"{pdb_id}.pdb")
    if os.path.isfile(out_path):
        logger.info(f"PDB {pdb_id} already exists at {out_path}, reusing it.")
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info(f"Downloading PDB {pdb_id} from {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(out_path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        logger.error(f"Failed to download PDB {pdb_id}: {exc}")
        raise
    logger.info(f"Saved template PDB to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# MODELLER homology modelling
# ---------------------------------------------------------------------------

def _extract_template_sequence_from_pdb(pdb_path: str, chain_id: str) -> str:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("TEMPLATE", pdb_path)
    model = structure[0]
    try:
        chain = model[chain_id]
    except KeyError:
        # Fallback: first chain
        logger.warning(
            f"Chain {chain_id} not found in {pdb_path}; using first chain instead."
        )
        chain = next(model.get_chains())
    ppb = PDB.PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        raise ValueError(f"No polypeptide could be built from chain {chain_id} in {pdb_path}")
    seq = "".join(str(pp.get_sequence()) for pp in peptides)
    logger.info(f"Extracted template sequence from {pdb_path}, chain {chain.id}, len={len(seq)} aa")
    return seq


def _wrap_pir_sequence(seq: str, line_length: int = 75) -> str:
    seq = seq.replace("*", "")
    return "\n".join(seq[i:i + line_length] for i in range(0, len(seq), line_length)) + "*"


def generate_modeller_inputs(
    target_id: str,
    target_seq: str,
    template_pdb_path: str,
    template_pdb_id: str,
    template_chain: str,
    outdir: str,
) -> Tuple[str, str]:
    """
    Build PIR alignment file and MODELLER script.
    Returns (alignment_path, modeller_script_path).
    """
    ensure_outdir(outdir)
    logger.info("Preparing MODELLER input files (alignment + run script)...")

    template_seq = _extract_template_sequence_from_pdb(template_pdb_path, template_chain)

    # Pairwise alignment template vs target
    aln = pairwise2.align.globalxx(template_seq, target_seq, one_alignment_only=True)[0]
    template_aln, target_aln, score, begin, end = aln
    logger.info(f"Pairwise alignment score (template vs target) = {score:.1f}")

    align_path = os.path.join(outdir, f"{target_id}_{template_pdb_id}.ali")
    modeller_script_path = os.path.join(outdir, f"run_modeller_{target_id}_{template_pdb_id}.py")

    # PIR alignment content
    pir_lines = []
    pir_lines.append(f">P1;{template_pdb_id}")
    pir_lines.append(
        f"structureX:{template_pdb_id}:1:{template_chain}:"
        f"{len(template_seq)}:{template_chain}::::"
    )
    pir_lines.append(_wrap_pir_sequence(template_aln))
    pir_lines.append("")
    pir_lines.append(f">P1;{target_id}")
    pir_lines.append(f"sequence:{target_id}::::::::")
    pir_lines.append(_wrap_pir_sequence(target_aln))
    pir_content = "\n".join(pir_lines) + "\n"

    with open(align_path, "w") as fh:
        fh.write(pir_content)
    logger.info(f"MODELLER alignment written to: {align_path}")

    # MODELLER script
    modeller_script = f"""from modeller import *
from modeller.automodel import *

log.verbose()
env = environ()
env.io.atom_files_directory = ['.', '../']

a = automodel(env,
              alnfile='{os.path.basename(align_path)}',
              knowns='{template_pdb_id}',
              sequence='{target_id}')

a.starting_model = 1
a.ending_model   = 5  # generate 5 models
a.make()
"""
    with open(modeller_script_path, "w") as fh:
        fh.write(modeller_script)
    logger.info(f"MODELLER script written to: {modeller_script_path}")

    return align_path, modeller_script_path


def run_modeller(modeller_script_path: str) -> Optional[List[str]]:
    """
    Try to run MODELLER if a 'mod*' command is in PATH.
    Returns list of generated model PDB paths or None.
    """
    modeller_cmd = None
    for cmd in ("mod9.25", "mod10.5", "mod", "modpy"):
        if shutil.which(cmd):
            modeller_cmd = cmd
            break

    workdir = os.path.dirname(os.path.abspath(modeller_script_path))
    script_name = os.path.basename(modeller_script_path)

    if modeller_cmd is None:
        logger.error(
            "MODELLER command not found in PATH. "
            "Install MODELLER and ensure e.g. 'mod9.25' is available, then run manually:\n"
            f"    cd {workdir}\n"
            f"    mod9.25 {script_name}"
        )
        return None

    logger.info(f"Attempting to run MODELLER via command: {modeller_cmd} {script_name}")
    try:
        subprocess.run([modeller_cmd, script_name], cwd=workdir, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"MODELLER failed with return code {exc.returncode}")
        return None

    # Collect resulting model files (*.B9999*.pdb)
    model_paths = sorted(
        glob.glob(os.path.join(workdir, "*.B9999*.pdb"))
    )
    if not model_paths:
        logger.warning(f"No MODELLER models (*.B9999*.pdb) found in {workdir}")
        return None

    logger.info("MODELLER produced the following model(s):")
    for p in model_paths:
        logger.info(f"  {os.path.basename(p)}")
    return model_paths


def select_best_modeller_model(model_paths: List[str]) -> Optional[str]:
    if not model_paths:
        return None
    # For now, simply pick the first in sorted order
    best = sorted(model_paths)[0]
    logger.info(f"Selected best MODELLER model: {os.path.basename(best)}")
    return best


# ---------------------------------------------------------------------------
# ColabFold (local)
# ---------------------------------------------------------------------------

def run_colabfold(
    fasta_path: str, outdir: str, colabfold_batch: str = "colabfold_batch"
) -> Optional[str]:
    """
    Run local ColabFold if available. Returns best model PDB path or None.
    """
    cf_cmd = shutil.which(colabfold_batch)
    cf_outdir = os.path.join(outdir, "colabfold")
    ensure_outdir(cf_outdir)

    if cf_cmd is None:
        logger.error(
            f"ColabFold command '{colabfold_batch}' not found. "
            "Install local ColabFold or adjust --colabfold-batch."
        )
        return None

    logger.info(
        f"Attempting to run ColabFold via: {colabfold_batch} "
        f"{fasta_path} {cf_outdir} --use-gpu --amber"
    )
    try:
        subprocess.run(
            [cf_cmd, fasta_path, cf_outdir, "--use-gpu", "--amber"],
            cwd=outdir,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(f"ColabFold failed with return code {exc.returncode}")
        return None

    best_model = find_best_colabfold_model(cf_outdir)
    if best_model:
        logger.info(f"Best ColabFold model: {best_model}")
    else:
        logger.warning("Could not identify best ColabFold model in output directory.")
    return best_model


def find_best_colabfold_model(cf_outdir: str) -> Optional[str]:
    """Heuristic: use ranking_debug.json if present, else 'ranked_0.pdb', else any PDB."""
    ranking_path = os.path.join(cf_outdir, "ranking_debug.json")
    if os.path.isfile(ranking_path):
        try:
            with open(ranking_path) as fh:
                data = json.load(fh)
            # 'order' is typically a list of model names in descending confidence
            order = data.get("order") or data.get("ranked_order")
            if order:
                best_name = order[0]
                patterns = [
                    f"relaxed_{best_name}.pdb",
                    f"unrelaxed_{best_name}.pdb",
                    f"{best_name}.pdb",
                ]
                for pattern in patterns:
                    candidate = os.path.join(cf_outdir, pattern)
                    if os.path.isfile(candidate):
                        return candidate
        except Exception as exc:
            logger.warning(f"Failed to parse ranking_debug.json: {exc}")

    # Fallbacks
    ranked = glob.glob(os.path.join(cf_outdir, "ranked_0*.pdb"))
    if ranked:
        return ranked[0]

    pdbs = glob.glob(os.path.join(cf_outdir, "*.pdb"))
    if pdbs:
        return sorted(pdbs)[0]

    return None


# ---------------------------------------------------------------------------
# Model comparison (RMSD)
# ---------------------------------------------------------------------------

def _get_ca_coordinates(pdb_path: str) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is required for RMSD comparison but is not installed.")
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("MODEL", pdb_path)
    model = structure[0]
    chain = next(model.get_chains())
    ca_coords = []
    for residue in chain:
        if "CA" in residue:
            ca_coords.append(residue["CA"].get_coord())
    if not ca_coords:
        raise ValueError(f"No CA atoms found in first chain of {pdb_path}")
    return np.array(ca_coords, dtype=float)


def compute_ca_rmsd(pdb_ref: str, pdb_mobile: str) -> float:
    """
    Compute backbone CA RMSD between two models (superimposing mobile onto ref).
    """
    if np is None:
        logger.warning("NumPy is missing; cannot compute RMSD.")
        return float("nan")
    from Bio.PDB import Superimposer

    coords_ref = _get_ca_coordinates(pdb_ref)
    coords_mob = _get_ca_coordinates(pdb_mobile)
    n = min(len(coords_ref), len(coords_mob))
    if n < 10:
        logger.warning("Very few CA atoms overlap; RMSD may be unreliable.")
    coords_ref = coords_ref[:n]
    coords_mob = coords_mob[:n]

    sup = Superimposer()
    sup.set_atoms(coords_ref, coords_mob)
    rmsd = float(sup.rms)
    logger.info(f"CA RMSD between models: {rmsd:.3f} Å (on {n} CA pairs)")
    return rmsd


# ---------------------------------------------------------------------------
# Docking preparation
# ---------------------------------------------------------------------------

def prepare_structure_for_docking(input_pdb: str, output_pdb: str) -> None:
    """
    Simple docking prep:
      - Keep first model only
      - Keep all protein residues
      - Keep HEM/HEC prosthetic group
      - Keep waters within 4.0 Å of HEM iron (if present)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("IN", input_pdb)
    model = structure[0]

    # Find heme centre (HEM/HEC FE atom)
    heme_atoms = []
    for chain in model:
        for res in chain:
            if res.get_resname().strip() in {"HEM", "HEC"}:
                for atom in res:
                    if atom.element.strip().upper() in {"FE", "ZN", "MG"}:
                        heme_atoms.append(atom)

    heme_center = None
    if heme_atoms and np is not None:
        coords = np.array([a.get_coord() for a in heme_atoms], dtype=float)
        heme_center = coords.mean(axis=0)
        logger.info(
            f"Detected {len(heme_atoms)} metal atoms in heme; using their centroid as active-site centre."
        )
    elif heme_atoms:
        logger.info(
            "Detected heme metal atoms but NumPy not installed; skipping water-distance filtering."
        )

    io = PDB.PDBIO()

    class DockingSelect(PDB.Select):
        def accept_residue(self, residue):
            resname = residue.get_resname().strip()
            het = residue.id[0].strip() != ""  # HETATM if non-blank
            # Always keep protein residues
            if not het:
                return 1
            # Keep heme-like groups
            if resname in {"HEM", "HEC"}:
                return 1
            # Optionally keep waters near heme centre
            if resname in {"HOH", "WAT"} and heme_center is not None and np is not None:
                min_dist = math.inf
                for atom in residue:
                    dist = np.linalg.norm(atom.get_coord() - heme_center)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist <= 4.0:
                    return 1
            return 0

    io.set_structure(structure)
    io.save(output_pdb, DockingSelect())
    logger.info(f"Docking-ready structure written to: {output_pdb}")


# ---------------------------------------------------------------------------
# Optional MD relaxation
# ---------------------------------------------------------------------------

def run_md_relaxation(outdir: str, md_command: str) -> None:
    """
    Run an arbitrary MD command in 'outdir'. This is just a wrapper:
    e.g. md_command="gmx mdrun -s topol.tpr -deffnm md"
    """
    logger.info(f"Running MD relaxation with command (cwd={outdir}): {md_command}")
    try:
        subprocess.run(md_command, cwd=outdir, shell=True, check=True)
        logger.info("MD relaxation command finished successfully.")
    except subprocess.CalledProcessError as exc:
        logger.error(f"MD command failed with return code {exc.returncode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structure preparation pipeline: template search, homology modelling, "
                    "ColabFold prediction, comparison, and docking prep."
    )
    parser.add_argument("--fasta", required=True, help="Target sequence FASTA file")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--search-templates", action="store_true", help="Search PDB for templates with BLASTP")
    parser.add_argument("--run-homology", action="store_true", help="Run MODELLER homology modelling")
    parser.add_argument("--run-colabfold", action="store_true", help="Run local ColabFold (colabfold_batch)")
    parser.add_argument("--compare-models", action="store_true", help="Compare homology vs ColabFold models (RMSD)")
    parser.add_argument("--prep-docking", action="store_true", help="Prepare a docking-ready PDB from best model")
    parser.add_argument("--run-md", action="store_true", help="Run an MD relaxation command in outdir")

    parser.add_argument("--min-template-identity", type=float, default=30.0,
                        help="Minimum BLAST identity for template selection (%%)")
    parser.add_argument("--min-template-coverage", type=float, default=70.0,
                        help="Minimum BLAST coverage for template selection (%%)")
    parser.add_argument("--colabfold-batch", default="colabfold_batch",
                        help="Name or path of colabfold_batch executable")
    parser.add_argument("--md-command", default=None,
                        help="Shell command to run MD relaxation (used with --run-md)")

    args = parser.parse_args()

    logger.info("===== Starting structure_prep_pipeline.py =====")
    logger.info(f"FASTA: {args.fasta}")
    logger.info(f"Outdir: {args.outdir}")

    ensure_outdir(args.outdir)
    target_id, target_seq = read_fasta(args.fasta)

    template_candidate: Optional[TemplateCandidate] = None
    template_pdb_path: Optional[str] = None

    # ------------------------------------------------------------------
    # 1) Search templates with BLASTP
    # ------------------------------------------------------------------
    blast_xml_path = os.path.join(args.outdir, "blastp_pdb.xml")
    if args.search_templates:
        run_blastp_pdb(target_seq, blast_xml_path)
        candidates = parse_blastp_xml_for_templates(
            blast_xml_path,
            query_len=len(target_seq),
            max_hits=25,
            min_identity=args.min_template_identity,
            min_coverage=args.min_template_coverage,
        )
        if candidates:
            template_candidate = choose_best_template(candidates)
            if template_candidate:
                template_pdb_path = download_pdb(template_candidate.pdb_id, args.outdir)
        else:
            logger.error("No suitable PDB templates found; homology modelling will be skipped.")
    else:
        logger.info("--search-templates was not requested; skipping BLASTP template search.")

    # ------------------------------------------------------------------
    # 2) Homology modelling with MODELLER
    # ------------------------------------------------------------------
    homology_models: Optional[List[str]] = None
    best_homology_model: Optional[str] = None

    if args.run_homology:
        if template_candidate is None or template_pdb_path is None:
            logger.error("--run-homology requested but no template candidate is available.")
        else:
            align_path, modeller_script_path = generate_modeller_inputs(
                target_id=target_id,
                target_seq=target_seq,
                template_pdb_path=template_pdb_path,
                template_pdb_id=template_candidate.pdb_id,
                template_chain=template_candidate.chain,
                outdir=args.outdir,
            )
            logger.info(f"MODELLER alignment file: {align_path}")
            logger.info(f"MODELLER script: {modeller_script_path}")

            homology_models = run_modeller(modeller_script_path)
            if homology_models:
                best_homology_model = select_best_modeller_model(homology_models)
    else:
        logger.info("--run-homology not requested; skipping MODELLER step.")

    # ------------------------------------------------------------------
    # 3) ColabFold prediction
    # ------------------------------------------------------------------
    best_colabfold_model: Optional[str] = None
    if args.run_colabfold:
        best_colabfold_model = run_colabfold(
            fasta_path=args.fasta,
            outdir=args.outdir,
            colabfold_batch=args.colabfold_batch,
        )
    else:
        logger.info("--run-colabfold not requested; skipping ColabFold step.")

    # ------------------------------------------------------------------
    # 4) Compare homology vs ColabFold models
    # ------------------------------------------------------------------
    if args.compare_models:
        if not best_homology_model or not best_colabfold_model:
            logger.error(
                "--compare-models was requested but we do not have both "
                "a homology model and a ColabFold model."
            )
        else:
            if np is None:
                logger.warning(
                    "NumPy not installed; cannot compute RMSD numerically, "
                    "but the PDBs are available for visual inspection."
                )
            else:
                rmsd = compute_ca_rmsd(best_homology_model, best_colabfold_model)
                rmsd_out = os.path.join(args.outdir, "homology_vs_colabfold_rmsd.txt")
                with open(rmsd_out, "w") as fh:
                    fh.write(f"CA RMSD (homology vs ColabFold) = {rmsd:.3f} Å\n")
                logger.info(f"Saved RMSD summary to: {rmsd_out}")
    else:
        logger.info("--compare-models not requested; skipping RMSD comparison.")

    # ------------------------------------------------------------------
    # 5) Prepare docking-ready structure
    # ------------------------------------------------------------------
    docking_model: Optional[str] = None
    if args.prep_docking:
        # Preference: ColabFold (amber-relaxed) > best homology > any available model
        candidate_for_docking = best_colabfold_model or best_homology_model
        if not candidate_for_docking:
            logger.error(
                "--prep-docking was requested but neither homology nor ColabFold "
                "models are available. Check previous steps."
            )
        else:
            docking_model = os.path.join(args.outdir, f"{target_id}_docking_ready.pdb")
            prepare_structure_for_docking(candidate_for_docking, docking_model)
    else:
        logger.info("--prep-docking not requested; skipping docking preparation.")

    # ------------------------------------------------------------------
    # 6) Optional MD relaxation
    # ------------------------------------------------------------------
    if args.run_md:
        if not args.md_command:
            logger.error("--run-md was requested but no --md-command was provided.")
        else:
            workdir = args.outdir
            if docking_model and os.path.isfile(docking_model):
                workdir = os.path.dirname(os.path.abspath(docking_model))
            run_md_relaxation(workdir, args.md_command)
    else:
        logger.info("--run-md not requested; skipping MD relaxation.")

    logger.info("===== structure_prep_pipeline.py finished =====")


if __name__ == "__main__":
    main()
