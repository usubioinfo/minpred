# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Discovery and SHA-256 checksum verification for bundled MINpred ``.keras`` and ``.npz`` artifacts.

All 13 release models (``ESM2_650M_mlp_adamw_full``, Seed 2) and their fold-fit
``StandardScaler`` archives are packaged directly inside ``minpred.models`` via
``importlib.resources``. An optional ``MINPRED_MODEL_DIR`` environment variable can
override the bundled directory when custom checkpoints are staged on a cluster.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any


def load_manifest() -> dict[str, Any]:
    """Load and parse the bundled ``model_manifest.json`` specification.

    Returns:
        Parsed JSON dictionary describing the schema version, ESM-2 backbone, sliding-window
        parameters, and per-model file names, byte sizes, SHA-256 digests, and class labels.
    """
    override = os.environ.get("MINPRED_MODEL_DIR")
    if override:
        custom_manifest = Path(override).expanduser().resolve() / "model_manifest.json"
        if custom_manifest.is_file():
            return json.loads(custom_manifest.read_text(encoding="utf-8"))
    resource = files("minpred.models").joinpath("model_manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def model_directory() -> Path:
    """Resolve the filesystem directory containing the bundled ``.keras`` and ``.npz`` files.

    Returns:
        Absolute ``Path`` to ``src/minpred/models`` (or the ``MINPRED_MODEL_DIR`` override
        directory when present and populated).
    """
    override = os.environ.get("MINPRED_MODEL_DIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        if (candidate / "phase1.keras").is_file():
            return candidate
    return Path(str(files("minpred.models")))


def _sha256(path: Path) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of ``path`` in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_artifact(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    verify_hash: bool = True,
) -> bool:
    """Check file existence, exact byte length, and optional SHA-256 checksum.

    Args:
        path: Candidate ``.keras`` or ``.npz`` file path.
        expected_bytes: Exact byte count recorded in ``model_manifest.json``.
        expected_sha256: Expected lowercase SHA-256 digest string.
        verify_hash: Whether to compute and compare the full SHA-256 digest.

    Returns:
        ``True`` if ``path`` exists and satisfies size (and optional hash) checks.
    """
    if not path.is_file() or path.stat().st_size != expected_bytes:
        return False
    return not (verify_hash and _sha256(path) != expected_sha256)


def get_model_paths(model_key: str, verify_hash: bool = False) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve and validate ``(keras_path, scaler_path, metadata)`` for ``model_key``.

    Args:
        model_key: Manifest key (``"phase1"``, ``"phase2"``, ``"phase3"``, or
            ``"phase4_<class>"``).
        verify_hash: Whether to recompute SHA-256 digests prior to returning paths.

    Returns:
        Tuple ``(keras_path, scaler_path, metadata)``.

    Raises:
        KeyError: If ``model_key`` is not registered in ``model_manifest.json``.
        FileNotFoundError: If either the ``.keras`` model or ``.npz`` scaler is missing
            or fails validation.
    """
    manifest = load_manifest()
    try:
        metadata = manifest["models"][model_key]
    except KeyError as exc:
        raise KeyError(f"Unknown MINpred model key: {model_key}") from exc

    directory = model_directory()
    keras_path = directory / metadata["keras_filename"]
    scaler_path = directory / metadata["scaler_filename"]

    if not is_valid_artifact(keras_path, metadata["keras_bytes"], metadata["keras_sha256"], verify_hash=verify_hash):
        raise FileNotFoundError(
            f"Bundled Keras model missing or corrupted for '{model_key}': {keras_path}"
        )
    if not is_valid_artifact(scaler_path, metadata["scaler_bytes"], metadata["scaler_sha256"], verify_hash=verify_hash):
        raise FileNotFoundError(
            f"Bundled StandardScaler (.npz) missing or corrupted for '{model_key}': {scaler_path}"
        )
    return keras_path, scaler_path, metadata


def select_models(selection: str = "all") -> list[str]:
    """Expand a phase or group selector into concrete manifest model keys.

    Args:
        selection: ``"all"``, ``"phase1"``, ``"phase2"``, ``"phase3"``, ``"phase4"``,
            or an individual ``"phase4_<class>"`` key.

    Returns:
        Ordered list of matching model keys.

    Raises:
        ValueError: If ``selection`` does not match any supported group or model key.
    """
    models = load_manifest()["models"]
    normalized = selection.lower()
    if normalized == "all":
        return list(models)
    if normalized in {"phase1", "phase2", "phase3"}:
        return [normalized]
    if normalized == "phase4":
        return [key for key in models if key.startswith("phase4_")]
    if normalized in models:
        return [normalized]
    raise ValueError(f"Unknown model selection: {selection}")


def model_status(verify_hash: bool = True) -> list[tuple[str, Path, Path, bool]]:
    """Inspect presence and integrity of all 13 bundled ``(.keras, .npz)`` model pairs.

    Args:
        verify_hash: Whether to verify SHA-256 digests in addition to byte lengths.

    Returns:
        List of ``(model_key, keras_path, scaler_path, is_valid)`` tuples across all
        13 classification units.
    """
    manifest = load_manifest()
    directory = model_directory()
    statuses: list[tuple[str, Path, Path, bool]] = []
    for key, meta in manifest["models"].items():
        k_path = directory / meta["keras_filename"]
        s_path = directory / meta["scaler_filename"]
        ok = is_valid_artifact(
            k_path, meta["keras_bytes"], meta["keras_sha256"], verify_hash=verify_hash
        ) and is_valid_artifact(
            s_path, meta["scaler_bytes"], meta["scaler_sha256"], verify_hash=verify_hash
        )
        statuses.append((key, k_path, s_path, ok))
    return statuses
