# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Keras 3 neural inference engine for bundled MINpred ``.keras`` models and ``.npz`` scalers.

Loads each classification unit's release checkpoint (``.keras``) directly via Keras 3
(configured automatically to run on the PyTorch backend when TensorFlow is absent) together
with its training-fit ``StandardScaler`` parameters (``mean`` and ``scale`` in ``.npz``),
caches loaded models in memory across phases, and evaluates calibrated softmax class
probabilities.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from minpred.model_store import get_model_paths

_MODEL_CACHE: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}


def _configure_keras_backend() -> None:
    """Configure Keras 3 to use the PyTorch backend when TensorFlow is not loaded."""
    if "KERAS_BACKEND" not in os.environ and "tensorflow" not in sys.modules:
        os.environ["KERAS_BACKEND"] = "torch"


def _get_loaded_unit(model_key: str) -> tuple[Any, np.ndarray, np.ndarray]:
    """Load and cache a ``(.keras model, mean, scale)`` tuple for ``model_key``.

    Args:
        model_key: Identifier of the classification unit in ``model_manifest.json``
            (for example ``"phase1"``, ``"phase3"``, or ``"phase4_serine"``).

    Returns:
        A 3-tuple ``(keras_model, mean, scale)`` where ``keras_model`` is a compiled-free
        Keras ``Model`` instance and ``mean`` / ``scale`` are 1D ``float32`` arrays of
        length 1,280.
    """
    if model_key not in _MODEL_CACHE:
        _configure_keras_backend()
        import keras

        keras_path, scaler_path, _ = get_model_paths(model_key)
        scaler = np.load(scaler_path, allow_pickle=False)
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        scale = np.asarray(scaler["scale"], dtype=np.float32)
        model = keras.saving.load_model(str(keras_path), compile=False)
        _MODEL_CACHE[model_key] = (model, mean, scale)
    return _MODEL_CACHE[model_key]


def predict(model_key: str, embeddings: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Standardize 1,280-d ESM-2 embeddings and compute softmax class probabilities.

    Args:
        model_key:Bundled model identifier (``"phase1"``, ``"phase2"``, ``"phase3"``,
            or ``"phase4_<class>"``).
        embeddings: Two-dimensional ``(N, 1280)`` array of overlap-averaged ESM-2
            sequence representations.
        batch_size: Mini-batch size passed to ``model.predict`` (default: ``1024``).

    Returns:
        A ``float32`` probability matrix of shape ``(N, K)`` whose rows sum to ``1.0``
        across the ``K`` output classes of ``model_key``.

    Raises:
        ValueError: If ``embeddings`` is not two-dimensional or its feature width does
            not match the 1,280-dimensional scaler of ``model_key``.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D (N, 1280) embedding matrix; got shape {arr.shape}")
    if arr.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    model, mean, scale = _get_loaded_unit(model_key)
    if arr.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Model '{model_key}' expects {mean.shape[0]}-d embeddings, but received {arr.shape[1]}-d."
        )

    x_std = ((arr - mean) / scale).astype(np.float32)
    reshaped = x_std.reshape(x_std.shape[0], 1, x_std.shape[1], 1)
    probs = model.predict(reshaped, batch_size=batch_size, verbose=0)
    return np.asarray(probs, dtype=np.float32)
