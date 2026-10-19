import sys
from pathlib import Path

import pytest

# The relay is an optional component with its own dependency (aiohttp);
# skip its whole suite cleanly where that dependency isn't installed.
# The product suite does not collect this directory (see pyproject testpaths).
pytest.importorskip("aiohttp")

_RELAY_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _RELAY_DIR.parents[1]  # archive/relay -> repository root
for path in (str(_RELAY_DIR), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
