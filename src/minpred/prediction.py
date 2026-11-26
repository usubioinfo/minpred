# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Hierarchical four-phase MINpred decision cascade and TSV report generation.

Consumes a single ``(N, 1280)`` ESM-2 representation matrix per FASTA run so no protein
is ever re-embedded across Phases I-IV. Writes both:
    1. Per-phase probability tables (``Phase_1_dnn_log.{tsv,txt}``,
       ``Phase_2_dnn_log.{tsv,txt}``, ``Phase_3_dnn_log.{tsv,txt}``, and
       ``Phase_4_<class>_log.{tsv,txt}``) matching the schema consumed by ``minpred_web``.
    2. A unified end-to-end summary table (``MINpred_results.tsv`` / ``-o``) tracking
       each sequence's progression and softmax confidence scores across requested phases.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from minpred.embedding import EmbeddedProteome, embed_fasta
from minpred.inference import predict
from minpred.model_store import load_manifest

PHASE4_CLASS_MAP: dict[str, tuple[str, str]] = {
    "amidohydrolases": ("Amidohydrolases", "phase4_amidohydrolases"),
    "aminopeptidases": ("Aminopeptidases", "phase4_aminopeptidases"),
    "aspartic": ("Aspartic endopeptidases", "phase4_aspartic"),
    "cysteine": ("Cysteine endopeptidases", "phase4_cysteine"),
    "dipeptidases": ("Dipeptidases", "phase4_dipeptidases"),
    "dipeptidyl": ("Dipeptidyl-peptidases", "phase4_dipeptidyl"),
    "metalloendopeptidases": ("Metalloendopeptidases", "phase4_metalloendopeptidases"),
    "metallopeptidases": ("Metallopeptidases", "phase4_metallopeptidases"),
    "omega": ("Omega peptidases", "phase4_omega"),
    "serine": ("Serine endopeptidases", "phase4_serine"),
}


def _write_dual_tsv(table: pd.DataFrame, stem_path: Path) -> None:
    """Write identical tab-delimited tables to ``.txt`` and ``.tsv`` extensions.

    Args:
        table: Formatted per-phase probability DataFrame.
        stem_path: Output path stem without file suffix.
    """
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem_path.with_suffix(".txt"), sep="\t", index=False)
    table.to_csv(stem_path.with_suffix(".tsv"), sep="\t", index=False)


def _predict_subset(
    seq_ids: list[str],
    embeddings: np.ndarray,
    model_key: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate ``model_key`` on ``embeddings`` and format percentage probability columns.

    Args:
        seq_ids: Ordered sequence identifiers for the subset entering ``model_key``.
        embeddings: ``(M, 1280)`` ``float32`` ESM-2 embedding matrix.
        model_key: Manifest key of the bundled ``.keras`` classifier.

    Returns:
        Tuple ``(formatted_df, raw_probabilities)`` where ``formatted_df`` reports
        ``SampleID``, top-1 ``Prediction``, and percentage probabilities ``[0, 100]``.
    """
    labels: list[str] = load_manifest()["models"][model_key]["labels"]
    if len(seq_ids) == 0:
        empty = pd.DataFrame(columns=["SampleID", "Prediction", *labels])
        return empty, np.empty((0, len(labels)), dtype=np.float32)

    probs = predict(model_key, embeddings)
    if probs.shape[1] != len(labels):
        raise RuntimeError(
            f"Model '{model_key}' returned {probs.shape[1]} outputs, expected {len(labels)}."
        )
    pred_idx = probs.argmax(axis=1)
    pct_df = pd.DataFrame(probs * 100.0, columns=labels).round(4)
    pct_df.insert(0, "Prediction", [labels[i] for i in pred_idx])
    pct_df.insert(0, "SampleID", seq_ids)
    return pct_df, probs


def run_from_embeddings(
    proteome: EmbeddedProteome,
    output_dir: str | Path,
    *,
    level: str = "Phase4",
    ecnumber: str = "all",
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """Execute the four-phase MINpred cascade starting from an :class:`EmbeddedProteome`.

    Args:
        proteome: Pre-embedded sequences and ``(N, 1280)`` ESM-2 vectors.
        output_dir: Directory where per-phase ``Phase_*_log.{tsv,txt}`` tables are saved.
        level: Terminal phase to execute (``"Phase1"``, ``"Phase2"``, ``"Phase3"``, or ``"Phase4"``).
        ecnumber: ``"all"`` for automatic Phase III -> Phase IV routing, or one of the ten
            class keys in :data:`PHASE4_CLASS_MAP`.
        confidence_threshold: Optional minimum softmax probability ``[0.0, 1.0]`` required
            to advance to the next phase (default: ``0.0``, standard argmax routing).

    Returns:
        Unified summary DataFrame with one row per input sequence in original FASTA order.

    Raises:
        ValueError: If ``level`` or ``ecnumber`` is invalid.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_levels = {"Phase1": 1, "Phase2": 2, "Phase3": 3, "Phase4": 4}
    if level not in valid_levels:
        raise ValueError(f"Unsupported level '{level}'; choose from {sorted(valid_levels)}")
    max_phase = valid_levels[level]

    ec_choice = (ecnumber or "all").strip().lower()
    if ec_choice != "all" and ec_choice not in PHASE4_CLASS_MAP:
        raise ValueError(
            f"Unsupported Phase IV class '{ecnumber}'; choose 'all' or one of {sorted(PHASE4_CLASS_MAP)}"
        )

    n_seqs = len(proteome.seq_ids)
    summary = pd.DataFrame(
        {
            "SampleID": proteome.seq_ids,
            "Length": proteome.lengths,
            "Phase1_Prediction": "",
            "Phase1_Confidence": np.nan,
            "Phase2_Prediction": "",
            "Phase2_Confidence": np.nan,
            "Phase3_Class": "",
            "Phase3_Confidence": np.nan,
            "Phase4_EC": "",
            "Phase4_Confidence": np.nan,
            "Prediction": "",
        }
    )

    # ── Phase I: Enzyme vs Non-enzyme ─────────────────────────────────────────
    p1_table, p1_probs = _predict_subset(proteome.seq_ids, proteome.embeddings, "phase1")
    _write_dual_tsv(p1_table, out_dir / "Phase_1_dnn_log")

    p1_max = p1_probs.max(axis=1) * 100.0 if n_seqs else np.array([])
    summary["Phase1_Prediction"] = p1_table["Prediction"].values
    summary["Phase1_Confidence"] = np.round(p1_max, 4)
    summary["Prediction"] = p1_table["Prediction"].values

    if max_phase == 1:
        return summary

    enz_mask = (p1_table["Prediction"].values == "Enzyme") & (
        (p1_probs.max(axis=1) >= confidence_threshold) if n_seqs else False
    )
    enz_indices = np.flatnonzero(enz_mask)
    if len(enz_indices) == 0:
        return summary

    # ── Phase II: Nitrogen Mineralization vs Non-nitrogen Mineralization ──────
    enz_ids = [proteome.seq_ids[i] for i in enz_indices]
    enz_emb = proteome.embeddings[enz_indices]
    p2_table, p2_probs = _predict_subset(enz_ids, enz_emb, "phase2")
    _write_dual_tsv(p2_table, out_dir / "Phase_2_dnn_log")

    for local_i, global_i in enumerate(enz_indices):
        pred_lbl = str(p2_table.iloc[local_i]["Prediction"])
        conf = round(float(p2_probs[local_i].max() * 100.0), 4)
        summary.at[global_i, "Phase2_Prediction"] = pred_lbl
        summary.at[global_i, "Phase2_Confidence"] = conf
        summary.at[global_i, "Prediction"] = pred_lbl

    if max_phase == 2:
        return summary

    min_local_mask = (p2_table["Prediction"].values == "Nitrogen Mineralization") & (
        p2_probs.max(axis=1) >= confidence_threshold
    )
    min_indices = enz_indices[np.flatnonzero(min_local_mask)]
    if len(min_indices) == 0:
        return summary

    # ── Phase III: 10 Functional Nitrogen-Mineralization Classes ──────────────
    min_ids = [proteome.seq_ids[i] for i in min_indices]
    min_emb = proteome.embeddings[min_indices]
    p3_table, p3_probs = _predict_subset(min_ids, min_emb, "phase3")
    _write_dual_tsv(p3_table, out_dir / "Phase_3_dnn_log")

    for local_i, global_i in enumerate(min_indices):
        cls_lbl = str(p3_table.iloc[local_i]["Prediction"])
        conf = round(float(p3_probs[local_i].max() * 100.0), 4)
        summary.at[global_i, "Phase3_Class"] = cls_lbl
        summary.at[global_i, "Phase3_Confidence"] = conf
        summary.at[global_i, "Prediction"] = cls_lbl

    if max_phase == 3:
        return summary

    # ── Phase IV: 10 Class-Specific 4-Digit EC Predictors ────────────────────
    selected_classes = (
        list(PHASE4_CLASS_MAP.items())
        if ec_choice == "all"
        else [(ec_choice, PHASE4_CLASS_MAP[ec_choice])]
    )

    p3_pred_lower = p3_table["Prediction"].astype(str).str.lower().values
    for class_key, (class_label, model_key) in selected_classes:
        local_hits = np.flatnonzero(
            (p3_pred_lower == class_label.lower()) & (p3_probs.max(axis=1) >= confidence_threshold)
        )
        if len(local_hits) == 0:
            continue

        cls_global_indices = min_indices[local_hits]
        cls_ids = [proteome.seq_ids[i] for i in cls_global_indices]
        cls_emb = proteome.embeddings[cls_global_indices]
        p4_table, p4_probs = _predict_subset(cls_ids, cls_emb, model_key)
        p4_table.insert(1, "Enzyme_Class", class_label)
        _write_dual_tsv(p4_table, out_dir / f"Phase_4_{class_key}_log")

        for j, global_i in enumerate(cls_global_indices):
            ec_lbl = str(p4_table.iloc[j]["Prediction"])
            conf = round(float(p4_probs[j].max() * 100.0), 4)
            summary.at[global_i, "Phase4_EC"] = ec_lbl
            summary.at[global_i, "Phase4_Confidence"] = conf
            summary.at[global_i, "Prediction"] = f"{class_label} ({ec_lbl})"

    return summary


def run_pipeline(
    fasta_file: str | Path,
    output_dir: str | Path = "MINpred_results",
    *,
    output_file: str = "MINpred_results.tsv",
    level: str = "Phase4",
    ecnumber: str = "all",
    sequence_type: str = "protein",
    device: str = "auto",
) -> pd.DataFrame:
    """Run the end-to-end MINpred workflow from protein or nucleotide FASTA to TSV reports.

    Args:
        fasta_file: Input protein or nucleotide FASTA path.
        output_dir: Destination directory for per-phase logs and summary table.
        output_file: Summary TSV filename inside ``output_dir``.
        level: Terminal phase (``"Phase1"`` .. ``"Phase4"``).
        ecnumber: ``"all"`` or a specific Phase IV class key.
        sequence_type: ``"protein"`` or ``"nucleotide"`` (runs ``TransDecoder.LongOrfs`` first).
        device: PyTorch device for ESM-2 representation extraction (``"auto"``, ``"cuda"``,
            ``"mps"``, or ``"cpu"``).

    Returns:
        Summary DataFrame written to ``output_dir / output_file``.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_fasta: str | Path = fasta_file
    if sequence_type == "nucleotide":
        from minpred.transdecoder import translate_nucleotide_fasta

        target_fasta = translate_nucleotide_fasta(fasta_file, out_dir)

    proteome = embed_fasta(target_fasta, device=device)
    summary = run_from_embeddings(proteome, out_dir, level=level, ecnumber=ecnumber)
    summary.to_csv(out_dir / output_file, sep="\t", index=False)
    return summary
