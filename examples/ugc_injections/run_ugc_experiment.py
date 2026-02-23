"""
Convenience wrapper: run UGC manifest via run_local_parallel with sensible defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from examples.batch.run_local_parallel import main as run_local_parallel_main
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.batch.run_local_parallel import main as run_local_parallel_main  # type: ignore[no-redef]


def main() -> int:
    """Invoke run_local_parallel; parse minimal args and pass through."""
    return run_local_parallel_main()


if __name__ == "__main__":
    sys.exit(main())
