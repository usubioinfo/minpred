# MINpred

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/usubioinfo/minpred/releases/tag/v1.0.0)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Web Server](https://img.shields.io/badge/Web_Server-kaabil.net%2Fminpred-0f766e.svg)](https://kaabil.net/minpred)

**MINpred** is an alignment-free, four-phase hierarchical deep-learning package for identifying and classifying microbial and soil nitrogen-mineralization enzymes down to four-digit Enzyme Commission (EC) numbers. It couples overlap-averaged representations from the frozen 650-million-parameter **ESM-2** protein language model (`facebook/esm2_t33_650M_UR50D`) with regularized multilayer perceptrons (`minpred_mlp`) trained with decoupled **AdamW** on single-linkage homology-disjoint UniProtKB/Swiss-Prot splits.

All **13 release Keras 3 checkpoints (`.keras`) and fold-fit feature scalers (`.npz`)** are bundled directly inside the Python package and verified via SHA-256 checksums — no separate model download step is required.

---

## Hierarchical Four-Phase Architecture

1. **Phase I — Enzyme vs. Strict Non-Enzyme:** Binary screening on 1,280-d ESM-2 representations (`<40%` sequence-identity held-out test `MCC = 0.954 ± 0.007`, `AUROC = 0.998`, `AUPRC = 0.998`).
2. **Phase II — Nitrogen Mineralization vs. Other Enzyme:** Separates target peptidases and linear C–N amidohydrolases from non-target enzymes, including homologous EC 3.4/3.5 controls (`MCC = 0.936 ± 0.011`, `AUROC = 0.998`, `AUPRC = 0.986`).
3. **Phase III — Ten Functional N-Mineralization Classes:** Classifies candidates into `Aminopeptidases` (EC 3.4.11), `Dipeptidases` (3.4.13), `Dipeptidyl-peptidases` (3.4.14), `Metallopeptidases` / metallocarboxypeptidases (3.4.17), `Omega peptidases` (3.4.19), `Serine endopeptidases` (3.4.21), `Cysteine endopeptidases` (3.4.22), `Aspartic endopeptidases` (3.4.23), `Metalloendopeptidases` (3.4.24), and `Amidohydrolases` (3.5.1) (`<60%` identity `MCC = 0.975 ± 0.007`, macro `AUROC = 0.998`).
4. **Phase IV — Ten Class-Specific 4-Digit EC Predictors:** Assigns explicit four-digit EC numbers across 52 catalytic specificities plus class-specific `Others` pools (`Accuracy = 96.1% ± 0.5%`, `MCC = 0.955 ± 0.006`).

Proteins exceeding ESM-2's native `1,022`-residue context window are embedded without C-terminal truncation using `1,022`-residue sliding windows at a `511`-residue stride with per-residue overlap averaging prior to mean-pooling.

---

## Installation

MINpred uses modern **PEP 621 / Hatchling** `src/` packaging and requires **Python 3.10+**.

### Install with `uv` (recommended)
```bash
uv sync --python 3.12
uv run minpred models-status
```

### Install with `pip`
```bash
pip install .
minpred models-status
```

### Install with Conda / Mamba (includes `TransDecoder` for nucleotide inputs)
```bash
conda env create -f minpred_environment.yml
conda activate minpred
minpred models-status
```

*(Note: On first run, Hugging Face `transformers` automatically caches `facebook/esm2_t33_650M_UR50D` locally.)*

---

## Command-Line Usage

```bash
# End-to-end 4-phase prediction (default: Phase 4 with automatic EC routing)
minpred -i example/test.fasta -o MINpred_results

# Restrict Phase IV to a single enzyme class
minpred -i example/test.fasta -o MINpred_serine -p 4 -c serine

# Screen only Phase I (Enzyme vs Non-enzyme) or Phase III (10 Classes)
minpred -i example/test.fasta -o MINpred_phase1 -p 1
minpred -i example/test.fasta -o MINpred_phase3 -p 3

# Predict from nucleotide transcripts / contigs (requires TransDecoder on PATH)
minpred -i transcripts.fasta -o MINpred_nucl -t nucleotide
```

### CLI Options
| Option | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | *(required)* | Input protein or nucleotide FASTA file |
| `-o`, `--outdir` | `MINpred_results` | Output directory for per-phase and summary TSV tables |
| `-f`, `--outfile` | `MINpred_results.tsv` | Summary TSV filename inside `--outdir` |
| `-p`, `--phase` | `4` (`Phase4`) | Terminal phase to execute (`1`–`4` or `Phase1`–`Phase4`) |
| `-c`, `--enzyme-class`, `--class` | `all` | Phase IV enzyme class (`all` or one of the 10 classes) |
| `-t`, `--seq-type` | `protein` | `protein` or `nucleotide` (`TransDecoder.LongOrfs` ORF translation) |
| `-d`, `--device` | `auto` | Compute device (`auto`, `cuda`, `cuda:0`, `mps`, `cpu`) |
| `-v`, `--version` | — | Show MINpred version |

### Output Files
Inside `--outdir`, MINpred writes:
- `MINpred_results.tsv` (or `-f <filename>`): Unified summary table across all input sequences (`SampleID`, `Length`, `Phase1_Prediction`, `Phase1_Confidence`, `Phase2_Prediction`, `Phase2_Confidence`, `Phase3_Class`, `Phase3_Confidence`, `Phase4_EC`, `Phase4_Confidence`, `Prediction`).
- `Phase_1_dnn_log.tsv` (and `.txt`): Per-sequence Phase I class probabilities (`%`).
- `Phase_2_dnn_log.tsv` (and `.txt`): Per-sequence Phase II probabilities for predicted enzymes.
- `Phase_3_dnn_log.tsv` (and `.txt`): Per-sequence 10-class probabilities for predicted N-mineralization enzymes.
- `Phase_4_<class>_log.tsv` (and `.txt`): Per-sequence 4-digit EC probabilities within each routed enzyme class.

---

## Author & License

- **Developer & Author:** Naveen Duhan ([naveen.duhan@usu.edu](mailto:naveen.duhan@usu.edu))
- **Lab:** Kaundal Artificial Intelligence & Advanced Bioinformatics Lab ([KAABiL](https://kaabil.net)), Utah State University, Logan, UT, USA *(PhD Supervisor: Dr. Rakesh Kaundal; PhD Committee Member: Dr. Jeanette M. Norton)*
- **License:** Released under the **[MIT License](LICENSE)** (`SPDX-License-Identifier: MIT`).
