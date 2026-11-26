# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Nucleotide open-reading-frame (ORF) translation wrapper using TransDecoder.

Identifies candidate coding regions in transcriptomic or genomic FASTA files via
``TransDecoder.LongOrfs``, strips terminal stop symbols, and writes a clean 60-column
peptide FASTA file (``translated_proteins.fasta``) ready for ESM-2 representation
extraction and MINpred classification.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def _file_digest(path: Path) -> str:
    """Compute SHA-256 hex digest of ``path`` to namespace temporary TransDecoder runs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _executable(name: str) -> str:
    """Locate ``TransDecoder.LongOrfs`` on ``PATH`` or inside an active Conda prefix.

    Args:
        name: Executable binary name (``"TransDecoder.LongOrfs"``).

    Returns:
        Absolute path to the executable binary.

    Raises:
        RuntimeError: If ``name`` cannot be found on ``PATH`` or ``$CONDA_PREFIX``.
    """
    executable = shutil.which(name)
    if executable is None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            packaged = Path(conda_prefix) / "opt" / "transdecoder" / "util" / name
            if packaged.is_file() and os.access(packaged, os.X_OK):
                executable = str(packaged)
    if executable is None:
        raise RuntimeError(
            f"{name} was not found on PATH. Install TransDecoder with "
            "`conda install -c conda-forge -c bioconda transdecoder`, then "
            "activate that environment before running nucleotide input."
        )
    return executable


def translate_nucleotide_fasta(input_fasta: str | Path, output_dir: str | Path) -> str:
    """Predict open reading frames from nucleotide FASTA and write cleaned peptides.

    Args:
        input_fasta: Path to input DNA, RNA, or transcript FASTA file.
        output_dir: Destination directory where ``translated_proteins.fasta`` is saved.

    Returns:
        String path to ``<output_dir>/translated_proteins.fasta``.

    Raises:
        FileNotFoundError: If ``input_fasta`` does not exist.
        RuntimeError: If ``TransDecoder.LongOrfs`` fails or yields zero ORFs.
    """
    source = Path(input_fasta).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Nucleotide FASTA not found: {source}")

    out_root = Path(output_dir).expanduser().resolve()
    long_orfs = _executable("TransDecoder.LongOrfs")
    run_dir = out_root / f"transdecoder_{_file_digest(source)[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    local_input = run_dir / "transcripts.fasta"
    transdecoder_output = run_dir / "transdecoder_out"
    peptide_fasta = transdecoder_output / f"{local_input.name}.transdecoder_dir" / "longest_orfs.pep"
    legacy_peptide_fasta = transdecoder_output / "longest_orfs.pep"
    stable_peptide_fasta = out_root / "translated_proteins.fasta"

    for candidate in (peptide_fasta, legacy_peptide_fasta):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return _write_clean_peptides(candidate, stable_peptide_fasta)

    shutil.copy2(source, local_input)
    log_path = run_dir / "TransDecoder.LongOrfs.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [long_orfs, "-t", local_input.name, "--output_dir", str(transdecoder_output)],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{Path(long_orfs).name} failed with exit code {completed.returncode}. See {log_path}."
        )

    if (not peptide_fasta.is_file() or peptide_fasta.stat().st_size == 0) and (
        not legacy_peptide_fasta.is_file() or legacy_peptide_fasta.stat().st_size == 0
    ):
        raise RuntimeError(
            f"TransDecoder completed without producing a non-empty peptide FASTA at {peptide_fasta}."
        )
    source_peptides = peptide_fasta if peptide_fasta.is_file() else legacy_peptide_fasta
    return _write_clean_peptides(source_peptides, stable_peptide_fasta)


def _write_clean_peptides(source: Path, destination: Path) -> str:
    """Format raw TransDecoder peptide output into 60-character FASTA lines without ``*`` stops."""
    records: list[tuple[str, str]] = []
    cur_header: str | None = None
    chunks: list[str] = []

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_header is not None:
                    pep = "".join(chunks).rstrip("*")
                    if pep:
                        records.append((cur_header, pep))
                cur_header = line
                chunks = []
            else:
                chunks.append(line)
        if cur_header is not None:
            pep = "".join(chunks).rstrip("*")
            if pep:
                records.append((cur_header, pep))

    if not records:
        raise RuntimeError("TransDecoder found no open reading frames.")

    with destination.open("w", encoding="utf-8") as out:
        for header, pep in records:
            out.write(f"{header}\n")
            for idx in range(0, len(pep), 60):
                out.write(f"{pep[idx : idx + 60]}\n")

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Unable to write translated peptide FASTA at {destination}.")
    return str(destination)
