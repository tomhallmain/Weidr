#!/usr/bin/env python3
"""Write assets/clip_reference_embedding.json, the embedding that
compare/clip_reference.py checks builds and test runs against.

Run it from a source checkout with working torch and clip (TorchScript
enabled), after a change that is meant to alter the embedding, e.g. a new
reference image:
    python scripts/generate_clip_reference_embedding.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import clip  # noqa: E402
import torch  # noqa: E402

from compare import clip_reference  # noqa: E402


def main() -> int:
    # clip.load itself, so the reference does not depend on the app's own loader.
    embedding = clip_reference.compute_embedding(load=clip.load)
    reference = {
        "model": clip_reference.REFERENCE_MODEL,
        "torch_version": torch.__version__,
        "embedding": [round(value, 8) for value in embedding],
    }
    path = clip_reference.reference_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=1)
        f.write("\n")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
