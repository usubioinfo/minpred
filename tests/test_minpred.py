# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Unit and numerical-integrity test suite for the MINpred ``.keras`` + ``.npz`` package.

Verifies SHA-256 checksums of all 13 bundled ``.keras`` models and ``.npz`` scalers,
Keras 3 probability calibration across every output head, hierarchical cascade routing,
and end-to-end CLI argument parsing and TSV generation.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>

License:
    MIT License
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from minpred import __author__, __license__, __version__, load_manifest, model_status
from minpred.cli import build_parser
from minpred.cli import main as cli_main
from minpred.embedding import EmbeddedProteome, read_protein_fasta
from minpred.inference import predict
from minpred.prediction import run_from_embeddings


def test_package_metadata_and_bundled_models_integrity() -> None:
    """Verify package authorship/license metadata and SHA-256 digests of all 13 model pairs."""
    assert __version__ == "1.0.0"
    assert __author__ == "Naveen Duhan"
    assert __license__ == "MIT"

    manifest = load_manifest()
    assert manifest["schema_version"] == 2
    assert manifest["embedding_dim"] == 1280
    assert manifest["window_size"] == 1022
    assert manifest["window_stride"] == 511
    assert len(manifest["models"]) == 13

    statuses = model_status(verify_hash=True)
    assert len(statuses) == 13
    for key, k_path, s_path, ok in statuses:
        assert k_path.suffix == ".keras"
        assert s_path.suffix == ".npz"
        assert ok, f"Bundled model/scaler failed checksum verification: {key} ({k_path}, {s_path})"


def test_all_13_keras_models_forward_pass_probabilities() -> None:
    """Confirm every bundled .keras model produces finite, normalized softmax probabilities."""
    manifest = load_manifest()
    rng = np.random.default_rng(42)
    dummy_emb = rng.normal(0.0, 0.25, size=(5, 1280)).astype(np.float32)

    for key, meta in manifest["models"].items():
        probs = predict(key, dummy_emb)
        assert probs.shape == (5, len(meta["labels"]))
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-5, atol=1e-5)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_clean_cli_options_exclude_internal_developer_flags() -> None:
    """Ensure the public CLI exposes clean POSIX flags and hides internal developer flags."""
    help_text = build_parser().format_help()
    assert "-i" in help_text and "--input" in help_text
    assert "-o" in help_text and "--outdir" in help_text
    assert "-f" in help_text and "--outfile" in help_text
    assert "-p" in help_text and "--phase" in help_text
    assert "-c" in help_text and "--enzyme-class" in help_text
    assert "-t" in help_text and "--seq-type" in help_text
    assert "-d" in help_text and "--device" in help_text
    assert "-od" not in help_text
    assert "--fasta_file" not in help_text
    assert "--esm-model" not in help_text
    assert "--embeddings" not in help_text
    assert "--save-embeddings" not in help_text
    assert "--batch-tokens" not in help_text


def test_end_to_end_cli_pipeline_writes_tsv_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``minpred`` CLI writes Phase TSV logs and unified summary table in FASTA order."""
    repo_root = Path(__file__).resolve().parents[1]
    example_fasta = repo_root / "example" / "test.fasta"
    seq_ids, seqs = read_protein_fasta(example_fasta)
    assert len(seq_ids) >= 10

    rng = np.random.default_rng(7)
    mock_proteome = EmbeddedProteome(
        seq_ids=seq_ids,
        lengths=np.asarray([len(s) for s in seqs], dtype=np.int32),
        embeddings=rng.normal(0.0, 0.5, size=(len(seq_ids), 1280)).astype(np.float32),
    )
    monkeypatch.setattr("minpred.prediction.embed_fasta", lambda *_a, **_kw: mock_proteome)

    out_dir = tmp_path / "results_p4"
    rc = cli_main(
        [
            "-i",
            str(example_fasta),
            "-o",
            str(out_dir),
            "-f",
            "minpred_predictions.tsv",
            "-p",
            "4",
        ]
    )
    assert rc == 0
    assert (out_dir / "Phase_1_dnn_log.tsv").is_file()
    assert (out_dir / "Phase_1_dnn_log.txt").is_file()
    summary = pd.read_csv(out_dir / "minpred_predictions.tsv", sep="\t")
    assert len(summary) == len(seq_ids)
    assert list(summary["SampleID"]) == seq_ids


def test_zero_enzyme_graceful_cascade(tmp_path: Path) -> None:
    """Confirm cascade terminates cleanly at Phase I when zero sequences pass the gate."""
    proteome = EmbeddedProteome(
        seq_ids=["seq_a", "seq_b"],
        lengths=np.array([150, 220], dtype=np.int32),
        embeddings=np.zeros((2, 1280), dtype=np.float32),
    )
    summary = run_from_embeddings(
        proteome,
        tmp_path / "high_thresh",
        level="Phase4",
        confidence_threshold=1.01,
    )
    assert len(summary) == 2
    assert (tmp_path / "high_thresh" / "Phase_1_dnn_log.tsv").is_file()
    assert not (tmp_path / "high_thresh" / "Phase_2_dnn_log.tsv").exists()


def test_models_status_cli_command() -> None:
    """Verify ``minpred models-status`` exits cleanly."""
    assert cli_main(["models-status"]) == 0


def test_fasta_x_stripping_and_non_standard_residue_validation(tmp_path: Path) -> None:
    """Verify unknown 'X' residues are stripped with a warning while invalid characters raise."""
    valid_x = tmp_path / "with_x.fasta"
    valid_x.write_text(">seq1\nMKTXXLV*\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="removed 2 unknown residue"):
        ids, seqs = read_protein_fasta(valid_x)
    assert ids == ["seq1"]
    assert seqs == ["MKTLV"]

    bad_fasta = tmp_path / "bad_aa.fasta"
    bad_fasta.write_text(">seq_bad\nMKTLJVB\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid/non-standard amino acid"):
        read_protein_fasta(bad_fasta)
