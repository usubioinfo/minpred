#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Repository-root launcher script for direct CLI and SLURM cluster invocations.

Adjusts ``sys.path`` so ``src/minpred`` takes precedence over this wrapper file when
invoked as ``python minpred.py ...`` from a source checkout or HPC job script
(``minpred.sl`` / ``run_minpred_web.slurm``), then delegates to :func:`minpred.cli.main`.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>
    Kaundal Artificial Intelligence & Advanced Bioinformatics Lab (KAABiL),
    Utah State University, Logan, UT, USA.

License:
    MIT License
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from minpred.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
