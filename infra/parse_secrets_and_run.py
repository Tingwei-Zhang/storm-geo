#!/usr/bin/env python3
"""If STORM_GEO_SECRETS_JSON is set, parse it and add keys to os.environ, then exec the batch entrypoint."""
import json
import os
import sys
from typing import Any, Dict, Optional


def parse_secrets_json(raw: Optional[str]) -> Dict[str, Any]:
    """Parse STORM_GEO_SECRETS_JSON string into a dict. Returns {} if raw is None or invalid."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except (json.JSONDecodeError, TypeError):
        return {}


def main() -> None:
    raw = os.environ.get("STORM_GEO_SECRETS_JSON")
    data = parse_secrets_json(raw)
    if raw and not data:
        try:
            json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: could not parse STORM_GEO_SECRETS_JSON: {e}", file=sys.stderr)
    for k, v in data.items():
        os.environ[k] = v
    # Exec the real entrypoint (no return)
    os.execv("/app/batch_entrypoint.sh", ["/app/batch_entrypoint.sh"])


if __name__ == "__main__":
    main()
