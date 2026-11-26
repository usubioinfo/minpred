# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Command-line interface (CLI) for standalone MINpred execution.

Provides the ``minpred`` console entry point for screening protein or nucleotide FASTA
files across Phases I-IV, plus ``minpred models-status`` to verify the SHA-256 integrity
of all 13 bundled ``.keras`` checkpoints and ``.npz`` scalers.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from minpred.__version__ import __version__
from minpred.model_store import load_manifest, model_status
from minpred.prediction import PHASE4_CLASS_MAP, run_pipeline

_PHASE_ALIASES: dict[str, str] = {
    "1": "Phase1",
    "2": "Phase2",
    "3": "Phase3",
    "4": "Phase4",
    "phase1": "Phase1",
    "phase2": "Phase2",
    "phase3": "Phase3",
    "phase4": "Phase4",
}


def _parse_phase(raw: str) -> str:
    """Normalize user phase input (``1``..``4`` or ``Phase1``..``Phase4``) to canonical form."""
    key = raw.strip().lower()
    if key in _PHASE_ALIASES:
        return _PHASE_ALIASES[key]
    raise argparse.ArgumentTypeError(
        f"Invalid phase '{raw}'. Choose from 1, 2, 3, 4 (or Phase1, Phase2, Phase3, Phase4)."
    )


def build_parser(version: str = __version__) -> argparse.ArgumentParser:
    """Construct the ``argparse.ArgumentParser`` for the ``minpred`` CLI.

    Args:
        version: Package version string displayed by ``-v`` / ``--version``.

    Returns:
        Configured ``ArgumentParser`` with clean POSIX short and long options.
    """
    parser = argparse.ArgumentParser(
        prog="minpred",
        description=(
            "MINpred: Alignment-free four-phase identification and Enzyme Commission "
            "classification of nitrogen-mineralization enzymes."
        ),
        epilog=(
            "Examples:\n"
            "  minpred -i proteins.fasta -o MINpred_results\n"
            "  minpred -i proteins.fasta -o MINpred_results -p 3\n"
            "  minpred -i proteins.fasta -o MINpred_results -p 4 -c serine\n"
            "  minpred -i transcripts.fasta -o MINpred_nucl -t nucleotide\n\n"
            "Author:  Naveen Duhan <naveen.duhan@usu.edu>\n"
            "Lab:     KAABiL, Utah State University (https://kaabil.net)\n"
            "License: MIT (https://github.com/usubioinfo/minpred)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"MINpred {version}")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="FASTA",
        help="Input protein or nucleotide FASTA file",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="MINpred_results",
        metavar="DIR",
        help="Output directory for per-phase and summary TSV tables (default: MINpred_results)",
    )
    parser.add_argument(
        "-f",
        "--outfile",
        default="MINpred_results.tsv",
        metavar="FILE",
        help="Summary TSV filename inside --outdir (default: MINpred_results.tsv)",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=_parse_phase,
        default="Phase4",
        metavar="{1,2,3,4}",
        help="Terminal prediction phase: 1..4 or Phase1..Phase4 (default: 4)",
    )
    parser.add_argument(
        "-c",
        "--enzyme-class",
        "--class",
        dest="enzyme_class",
        choices=["all", *sorted(PHASE4_CLASS_MAP)],
        default="all",
        metavar="CLASS",
        help=(
            "Phase IV enzyme class: all or {"
            + ", ".join(sorted(PHASE4_CLASS_MAP))
            + "} (default: all)"
        ),
    )
    parser.add_argument(
        "-t",
        "--seq-type",
        choices=["protein", "nucleotide"],
        default="protein",
        help="Input sequence alphabet; 'nucleotide' translates ORFs via TransDecoder first (default: protein)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        metavar="DEV",
        help="Compute device: auto, cuda, cuda:0, mps, or cpu (default: auto)",
    )
    return parser


def _print_models_status() -> int:
    """Verify and print the SHA-256 checksum status of all 13 bundled model pairs.

    Returns:
        ``0`` when all 13 ``.keras`` and ``.npz`` artifacts match their manifest SHA-256
        digests; ``1`` otherwise.
    """
    manifest = load_manifest()
    statuses = model_status(verify_hash=True)
    print(
        f"MINpred v{__version__} | Release Seed {manifest['release_seed']} | "
        f"Backbone: {manifest['esm_backbone']} ({manifest['embedding_dim']}-d)"
    )
    all_ok = True
    for key, k_path, s_path, ok in statuses:
        state = "ready" if ok else "INVALID/MISSING"
        if not ok:
            all_ok = False
        print(f"  {key:30s}: {state} ({k_path.name} + {s_path.name})")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    """Execute the ``minpred`` CLI workflow.

    Args:
        argv: Optional argument list overriding ``sys.argv[1:]`` (useful for testing).

    Returns:
        Process exit code (``0`` on success, non-zero on validation or runtime failure).
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args_list = list(sys.argv[1:] if argv is None else argv)

    if args_list and args_list[0] in {"models-status", "check-models"}:
        return _print_models_status()

    parser = build_parser()
    options = parser.parse_args(args_list)

    t0 = time.perf_counter()
    out_dir = Path(options.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MINpred v{__version__}] Output directory: {out_dir}")

    summary = run_pipeline(
        options.input,
        out_dir,
        output_file=options.outfile,
        level=options.phase,
        ecnumber=options.enzyme_class,
        sequence_type=options.seq_type,
        device=options.device,
    )
    elapsed = time.perf_counter() - t0
    n_total = len(summary)
    n_enz = int((summary["Phase1_Prediction"] == "Enzyme").sum())
    n_min = int((summary["Phase2_Prediction"] == "Nitrogen Mineralization").sum())
    print(
        f"[MINpred] Processed {n_total:,} sequence(s) in {elapsed:.2f}s | "
        f"Enzymes: {n_enz:,} | N-Mineralization: {n_min:,} | "
        f"Summary: {out_dir / options.outfile}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
