# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""MINpred: Alignment-free hierarchical ESM-2 + MLP prediction of nitrogen-mineralization enzymes.

MINpred classifies protein and translated nucleotide FASTA sequences across a four-phase
hierarchical deep-learning cascade:
    - Phase I: Enzyme vs. Non-enzyme binary screening.
    - Phase II: Nitrogen Mineralization vs. Non-nitrogen Mineralization enzyme screening.
    - Phase III: Assignment into ten functional peptidase and amidohydrolase classes.
    - Phase IV: Class-specific four-digit Enzyme Commission (EC) number prediction across
      52 explicit catalytic specificities plus rare-EC ``Others`` categories.

All 13 release Keras 3 checkpoints (``.keras``) and fold-fit ``StandardScaler`` archives
(``.npz``) are bundled directly inside ``minpred.models`` and checksum-verified via SHA-256.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

from minpred.__version__ import __author__, __copyright__, __email__, __license__, __version__
from minpred.embedding import EmbeddedProteome, ESM2Embedder
from minpred.model_store import load_manifest, model_status
from minpred.prediction import run_from_embeddings, run_pipeline

__all__ = [
    "ESM2Embedder",
    "EmbeddedProteome",
    "__author__",
    "__copyright__",
    "__email__",
    "__license__",
    "__version__",
    "load_manifest",
    "model_status",
    "run_from_embeddings",
    "run_pipeline",
]
