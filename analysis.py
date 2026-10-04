"""Command-line entry point: run the full financial analysis pipeline.

Usage
-----
    python analysis.py                  # live data, run everything, write reports/
    python analysis.py --offline        # use the committed data/ snapshot only
    python analysis.py --mc 20000 --rf 0.03

The engine itself lives in the ``finrisk`` package; this script only wires it to
the command line.
"""

from __future__ import annotations

from finrisk.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
