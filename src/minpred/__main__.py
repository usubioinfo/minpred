# SPDX-FileCopyrightText: 2026 Naveen Duhan <naveen.duhan@usu.edu>
# SPDX-License-Identifier: MIT
"""Module entry point enabling ``python -m minpred`` execution.

Delegates directly to :func:`minpred.cli.main` and propagates its integer exit status to
the operating system.

Author:
    Naveen Duhan <naveen.duhan@usu.edu>

License:
    MIT License
"""

from __future__ import annotations

from minpred.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
