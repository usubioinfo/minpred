#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
#SBATCH --job-name=minpred
#SBATCH --partition=mahaguru
#SBATCH --nodelist=chela-g01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
umask 077

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 input.fasta Phase1|Phase2|Phase3|Phase4 enzyme-class output-directory protein|nucleotide" >&2
  exit 64
fi

input_fasta=$1
level=$2
enzyme_class=$3
output_dir=$4
sequence_type=$5
app_dir="${MINPRED_APP_DIR:-$HOME/naveen_tools/minpred}"

[[ -f "$input_fasta" && -s "$input_fasta" ]] || { echo "Input FASTA is missing or empty: $input_fasta" >&2; exit 66; }
[[ -f "$app_dir/minpred.py" ]] || { echo "MINpred is missing: $app_dir" >&2; exit 69; }
[[ "$level" =~ ^Phase[1-4]$ ]] || { echo "Invalid prediction level: $level" >&2; exit 64; }
[[ "$sequence_type" == protein || "$sequence_type" == nucleotide ]] || { echo "Invalid sequence type: $sequence_type" >&2; exit 64; }
[[ "$enzyme_class" =~ ^(all|amidohydrolases|aminopeptidases|aspartic|cysteine|dipeptidases|dipeptidyl|metalloendopeptidases|metallopeptidases|omega|serine)$ ]] || { echo "Invalid Phase IV enzyme class: $enzyme_class" >&2; exit 64; }

module_name="${MINPRED_MODULE:-dl-gpu}"
if command -v module >/dev/null 2>&1; then
  module load "$module_name"
elif [[ -x /usr/bin/modulecmd ]]; then
  eval "$(/usr/bin/modulecmd bash load "$module_name")"
fi

if [[ -x "$app_dir/.venv/bin/python" ]]; then
  python_bin="$app_dir/.venv/bin/python"
else
  python_bin="$(command -v python3 || command -v python || true)"
fi
[[ -n "$python_bin" && -x "$python_bin" ]] || { echo "Python interpreter was not found." >&2; exit 69; }

module_prefix="${CONDA_PREFIX:-$(dirname "$(dirname "$python_bin")")}"
transdecoder_util="$module_prefix/opt/transdecoder/util"
if ! command -v TransDecoder.LongOrfs >/dev/null 2>&1 && [[ -d "$transdecoder_util" ]]; then
  export PATH="$transdecoder_util:$PATH"
fi
if [[ "$sequence_type" == nucleotide ]] && ! command -v TransDecoder.LongOrfs >/dev/null 2>&1; then
  echo "TransDecoder.LongOrfs is unavailable in PATH / $module_name." >&2
  exit 69
fi

mkdir -p "$output_dir"
export PYTHONPATH="$app_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

"$python_bin" -c 'import Bio, keras, numpy, pandas, torch, transformers'
"$python_bin" "$app_dir/minpred.py" \
  -i "$input_fasta" \
  -o "$output_dir" \
  -f minpred_predictions.tsv \
  -p "$level" \
  -c "$enzyme_class" \
  -t "$sequence_type" \
  -d "${MINPRED_DEVICE:-auto}"

prediction_file=$(find "$output_dir" -maxdepth 1 -type f -name '*.tsv' -size +0c -print -quit)
if [[ -z "$prediction_file" ]]; then
  echo "MINpred finished without a non-empty prediction table." >&2
  exit 70
fi

echo "MINpred results: $output_dir"
