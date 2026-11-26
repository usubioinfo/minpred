# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Overlap-averaged ESM-2 (650M) representation extraction for MINpred.

Implements the exact protein language-model feature pipeline used to train all 13 release
classifiers:
    - Backbone: ``facebook/esm2_t33_650M_UR50D`` (33 Transformer layers, 1,280-d hidden state).
    - Sequences <= 1,022 aa: single forward pass, strip ``<cls>`` and ``<eos>`` tokens, and
      mean-pool per-residue vectors across sequence length.
    - Sequences > 1,022 aa: overlapping 1,022-aa windows at a 511-aa stride plus a terminal
      window ``[L - 1022 : L]``; per-residue hidden states are averaged across overlapping
      windows prior to global residue mean-pooling so C-terminal domains are never truncated.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
WINDOW_SIZE = 1022
WINDOW_STRIDE = 511
DEFAULT_BATCH_TOKENS = 16384
_WS_RE = re.compile(r"\s+")
_NON_STD_AA_RE = re.compile(r"[^ARNDCQEGHILKMFPSTWYV]")


@dataclass(frozen=True)
class EmbeddedProteome:
    """Immutable container holding ordered FASTA sequence identifiers, lengths, and embeddings.

    Attributes:
        seq_ids: Sequence identifiers in original FASTA record order.
        lengths: 1D ``int32`` array of cleaned amino-acid lengths ``(N,)``.
        embeddings: 2D ``float32`` matrix of 1,280-d ESM-2 representations ``(N, 1280)``.
    """

    seq_ids: list[str]
    lengths: np.ndarray
    embeddings: np.ndarray


def read_protein_fasta(fasta_path: str | Path) -> tuple[list[str], list[str]]:
    """Parse a protein FASTA file, strip stop/whitespace symbols, and deduplicate headers.

    Args:
        fasta_path: Path to the input protein FASTA file.

    Returns:
        Tuple ``(seq_ids, sequences)`` preserving first-seen FASTA record order.

    Raises:
        FileNotFoundError: If ``fasta_path`` does not exist.
        ValueError: If no valid non-empty protein sequences are present.
    """
    path = Path(fasta_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input FASTA file not found: {path}")

    seq_ids: list[str] = []
    sequences: list[str] = []
    seen: set[str] = set()

    current_id: str | None = None
    chunks: list[str] = []

    def _commit() -> None:
        if not current_id or current_id in seen:
            return
        raw_seq = _WS_RE.sub("", "".join(chunks)).upper().rstrip("*")
        if not raw_seq:
            return
        removed_x = raw_seq.count("X")
        if removed_x:
            raw_seq = raw_seq.replace("X", "")
            warnings.warn(
                f"Sequence '{current_id}': removed {removed_x} unknown residue(s) marked 'X' before embedding.",
                RuntimeWarning,
                stacklevel=2,
            )
        invalid = sorted(set(_NON_STD_AA_RE.findall(raw_seq)))
        if invalid:
            bad = ", ".join(invalid)
            raise ValueError(
                f"Sequence '{current_id}' contains invalid/non-standard amino acid character(s): '{bad}'. "
                "Only the 20 standard amino acids are permitted ('X' is removed automatically)."
            )
        if not raw_seq:
            raise ValueError(f"Sequence '{current_id}' is empty after removing unknown residues marked 'X'.")
        seen.add(current_id)
        seq_ids.append(current_id)
        sequences.append(raw_seq)

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _commit()
                header = line[1:].strip()
                current_id = header.split()[0] if header else None
                chunks = []
            else:
                chunks.append(line)
        _commit()

    if not seq_ids:
        raise ValueError(f"No valid protein sequences found in FASTA: {path}")
    return seq_ids, sequences


def resolve_device(requested: str = "auto") -> str:
    """Resolve the PyTorch compute device string (``cuda`` -> ``mps`` -> ``cpu``).

    Args:
        requested: Explicit PyTorch device specification or ``"auto"``.

    Returns:
        Resolved device string suitable for ``torch.device(...)``.
    """
    import torch

    req = (requested or "auto").strip().lower()
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cached_embeddings(
    npz_path: str | Path,
    seq_ids: list[str],
    sequences: list[str],
) -> EmbeddedProteome:
    """Load precomputed ``(N, 1280)`` ESM-2 embeddings from an ``.npz`` archive.

    Args:
        npz_path: Path to an ``.npz`` archive containing ``accession`` and ``emb`` arrays.
        seq_ids: Ordered target sequence identifiers to extract.
        sequences: Cleaned amino-acid sequences corresponding to ``seq_ids``.

    Returns:
        Aligned :class:`EmbeddedProteome` matching ``seq_ids`` order.

    Raises:
        KeyError: If one or more ``seq_ids`` are absent from ``npz_path``.
    """
    archive = np.load(Path(npz_path).expanduser().resolve(), allow_pickle=False)
    acc_list = [str(a) for a in archive["accession"]]
    emb_mat = np.asarray(archive["emb"], dtype=np.float32)
    index_map = {acc: idx for idx, acc in enumerate(acc_list)}
    missing = [sid for sid in seq_ids if sid not in index_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"Precomputed embedding archive {npz_path} is missing {len(missing)} sequence(s), e.g.: {preview}"
        )
    rows = [index_map[sid] for sid in seq_ids]
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int32)
    return EmbeddedProteome(seq_ids=list(seq_ids), lengths=lengths, embeddings=emb_mat[rows])


class ESM2Embedder:
    """Reusable ESM-2 (650M) representation extractor for single or multi-proteome runs.

    Loads ``facebook/esm2_t33_650M_UR50D`` (or a local directory specified via the
    ``MINPRED_ESM_MODEL`` environment variable on air-gapped clusters) once onto the
    target accelerator in ``float16`` (GPU/MPS) or ``float32`` (CPU) and computes
    overlap-averaged 1,280-d sequence vectors.
    """

    def __init__(self, device: str = "auto") -> None:
        """Initialize the tokenizer and frozen 33-layer ESM-2 encoder on ``device``.

        Args:
            device: PyTorch device string (``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``).
        """
        import torch
        from transformers import AutoTokenizer, EsmModel
        from transformers import logging as tf_logging

        tf_logging.set_verbosity_error()

        self.backbone = os.environ.get("MINPRED_ESM_MODEL") or DEFAULT_ESM_MODEL
        self.device_str = resolve_device(device)
        self.device = torch.device(self.device_str)
        use_fp16 = self.device.type in {"cuda", "mps"}
        dtype = torch.float16 if use_fp16 else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        try:
            self.model: Any = (
                EsmModel.from_pretrained(self.backbone, add_pooling_layer=False, dtype=dtype)
                .to(self.device)
                .eval()
            )
        except TypeError:
            self.model = (
                EsmModel.from_pretrained(self.backbone, add_pooling_layer=False, torch_dtype=dtype)
                .to(self.device)
                .eval()
            )

    def _residue_embeddings(self, batch_seqs: list[str]) -> list[Any]:
        """Extract per-residue last-layer ``float32`` hidden states excluding ``<cls>``/``<eos>``."""
        import torch

        with torch.inference_mode():
            encoded = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, add_special_tokens=True)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            hidden = self.model(**encoded).last_hidden_state.float()
            return [hidden[i, 1 : len(seq) + 1] for i, seq in enumerate(batch_seqs)]

    def embed_sequences(
        self,
        seq_ids: list[str],
        sequences: list[str],
        *,
        max_tokens: int = DEFAULT_BATCH_TOKENS,
        window_size: int = WINDOW_SIZE,
        window_stride: int = WINDOW_STRIDE,
    ) -> EmbeddedProteome:
        """Compute 1,280-d overlap-averaged ESM-2 embeddings in original sequence order.

        Args:
            seq_ids: Ordered sequence identifiers.
            sequences: Cleaned amino-acid strings matching ``seq_ids``.
            max_tokens: Token budget per forward batch for length-sorted short sequences.
            window_size: Maximum window length in residues (``1022`` for ESM-2).
            window_stride: Sliding-window stride in residues (``511``).

        Returns:
            :class:`EmbeddedProteome` containing ``(N, 1280)`` ``float32`` embeddings.

        Raises:
            ValueError: If ``seq_ids`` and ``sequences`` differ in length.
            RuntimeError: If non-finite values arise during forward inference.
        """
        import torch

        if len(seq_ids) != len(sequences):
            raise ValueError("seq_ids and sequences must have identical length.")
        if not seq_ids:
            return EmbeddedProteome(
                seq_ids=[],
                lengths=np.empty((0,), dtype=np.int32),
                embeddings=np.empty((0, 1280), dtype=np.float32),
            )

        n_total = len(seq_ids)
        out_emb = np.zeros((n_total, 1280), dtype=np.float32)

        indexed = list(enumerate(sequences))
        short_items = sorted(
            [(idx, seq) for idx, seq in indexed if len(seq) <= window_size],
            key=lambda item: len(item[1]),
        )
        long_items = [(idx, seq) for idx, seq in indexed if len(seq) > window_size]

        batch: list[tuple[int, str]] = []

        def _flush_short() -> None:
            nonlocal batch
            if not batch:
                return
            tensors = self._residue_embeddings([seq for _, seq in batch])
            for (orig_idx, _), res_tensor in zip(batch, tensors):
                out_emb[orig_idx] = res_tensor.mean(dim=0).cpu().numpy()
            batch = []

        for orig_idx, seq in short_items:
            if batch and (len(batch) + 1) * (len(seq) + 2) > max_tokens:
                _flush_short()
            batch.append((orig_idx, seq))
        _flush_short()

        windows_per_step = max(1, max_tokens // (window_size + 2))
        for orig_idx, seq in long_items:
            seq_len = len(seq)
            starts = list(range(0, seq_len - window_size + 1, window_stride))
            if starts[-1] != seq_len - window_size:
                starts.append(seq_len - window_size)

            acc_sum: torch.Tensor | None = None
            counts = torch.zeros((seq_len, 1), dtype=torch.float32, device=self.device)
            for offset in range(0, len(starts), windows_per_step):
                chunk_starts = starts[offset : offset + windows_per_step]
                subseqs = [seq[st : st + window_size] for st in chunk_starts]
                for st, res_tensor in zip(chunk_starts, self._residue_embeddings(subseqs)):
                    if acc_sum is None:
                        acc_sum = torch.zeros(
                            (seq_len, res_tensor.shape[1]), dtype=torch.float32, device=self.device
                        )
                    acc_sum[st : st + window_size] += res_tensor
                    counts[st : st + window_size] += 1.0
            assert acc_sum is not None
            out_emb[orig_idx] = (acc_sum / counts).mean(dim=0).cpu().numpy()

        if not np.isfinite(out_emb).all():
            raise RuntimeError("Non-finite values encountered during ESM-2 embedding extraction.")

        lengths = np.asarray([len(s) for s in sequences], dtype=np.int32)
        return EmbeddedProteome(seq_ids=list(seq_ids), lengths=lengths, embeddings=out_emb)

    def embed_fasta(
        self,
        fasta_path: str | Path,
        *,
        max_tokens: int = DEFAULT_BATCH_TOKENS,
    ) -> EmbeddedProteome:
        """Read ``fasta_path`` and compute 1,280-d ESM-2 embeddings using this loaded backbone.

        Args:
            fasta_path: Path to protein FASTA file.
            max_tokens: Maximum tokens per forward batch.

        Returns:
            :class:`EmbeddedProteome` for all valid records in ``fasta_path``.
        """
        seq_ids, sequences = read_protein_fasta(fasta_path)
        return self.embed_sequences(seq_ids, sequences, max_tokens=max_tokens)


def embed_fasta(
    fasta_path: str | Path,
    *,
    device: str = "auto",
) -> EmbeddedProteome:
    """High-level helper to read a protein FASTA file and compute 1,280-d ESM-2 embeddings.

    Args:
        fasta_path: Input protein FASTA path.
        device: PyTorch execution device (``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``).

    Returns:
        :class:`EmbeddedProteome` containing sequence IDs, lengths, and ``(N, 1280)`` embeddings.
    """
    embedder = ESM2Embedder(device=device)
    return embedder.embed_fasta(fasta_path)
