"""Write the staged release's file index (release-files.json) into the payload.

The launcher reads it on every start to move files an older release left behind
out of src/ and bin/, which is what lets a user upgrade by extracting a new
archive over the old folder. The schema lives in bootstrap.py, next to its only
reader.

    uv run python packaging/write_release_index.py --payload dist/setup-payload/jav-trans
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap import read_release_index, write_release_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", required=True, help="staged release directory")
    args = parser.parse_args(argv)
    payload = Path(args.payload).resolve()
    if not payload.is_dir():
        print(f"[packaging] payload not found: {payload}", file=sys.stderr)
        return 1
    index = write_release_index(payload)
    shipped = read_release_index(payload)
    if shipped is None:
        # The launcher ignores an index without launcher.py and src/; shipping
        # one would silently turn the upgrade cleanup off.
        print(f"[packaging] {index.name} is not a usable index for {payload}", file=sys.stderr)
        return 1
    print(f"[packaging] indexed {len(shipped)} files -> {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
