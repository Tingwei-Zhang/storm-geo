#!/usr/bin/env python3
"""If STORM_GEO_SECRETS_JSON is set, parse it and add keys to os.environ, then exec the batch entrypoint."""
import json
import os
import sys


def main() -> None:
    raw = os.environ.get("STORM_GEO_SECRETS_JSON")
    if raw:
        try:
            data = json.loads(raw)
            for k, v in data.items():
                if isinstance(v, str):
                    os.environ[k] = v
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: could not parse STORM_GEO_SECRETS_JSON: {e}", file=sys.stderr)
    # Exec the real entrypoint (no return)
    os.execv("/app/batch_entrypoint.sh", ["/app/batch_entrypoint.sh"])

if __name__ == "__main__":
    main()
