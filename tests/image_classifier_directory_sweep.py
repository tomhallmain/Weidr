"""
Batch-run an image classifier from config and write per-file scores to CSV.

No UI. Resolves ``image_classifier_models`` from ``--config-json`` or, by default,
from Weidr's active ``configs/config.json`` (via ``utils.config``).

Usage (from repository root):
  python tests/image_classifier_directory_sweep.py nsfw_model ./folder1 ./folder2 -o out.csv
  python tests/image_classifier_directory_sweep.py nsfw_model D:/pics --recursive
  python tests/image_classifier_directory_sweep.py nsfw_model ./pics --config-json C:/path/config.json

Columns: path, argmax_index, argmax_label, error, then score_<category> for each
``model_categories`` entry in index order (same order as model output vector).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from image.image_classifier import ImageClassifierWrapper  # noqa: E402
from image.image_classifier_model_config import ImageClassifierModelConfig  # noqa: E402

_FALLBACK_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".avif",
)


def _load_classifier_models(config_json: Path | None) -> list:
    if config_json is not None:
        with config_json.open(encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("image_classifier_models", []))
    from utils.config import config

    return list(getattr(config, "image_classifier_models", None) or [])


def _image_suffixes() -> tuple[str, ...]:
    try:
        from utils.config import config

        exts = []
        for t in getattr(config, "image_types", []) or []:
            t = str(t).lower()
            if not t.startswith("."):
                t = "." + t
            exts.append(t)
        return tuple(exts) if exts else _FALLBACK_EXTS
    except Exception:
        return _FALLBACK_EXTS


def _gather_images(directories: list[Path], recursive: bool, suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for d in directories:
        d = d.resolve()
        if not d.is_dir():
            print(f"Skip (not a directory): {d}", file=sys.stderr)
            continue
        if recursive:
            for root, _, files in os.walk(d):
                for name in files:
                    p = Path(root) / name
                    if p.suffix.lower() in suffixes:
                        out.append(p.resolve())
        else:
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in suffixes:
                    out.append(p.resolve())
    return sorted(out)


def _find_model_dict(models: list, model_name: str) -> dict:
    name = model_name.strip()
    for m in models:
        if str(m.get("model_name", "")).strip() == name:
            return dict(m)
    raise SystemExit(f"No image_classifier_models entry with model_name={name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_name", help="model_name in image_classifier_models")
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Directories to scan for images",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("classifier_sweep.csv"))
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Path to config.json (uses only image_classifier_models). Default: Weidr config singleton.",
    )
    args = parser.parse_args()

    models = _load_classifier_models(args.config_json)
    raw = _find_model_dict(models, args.model_name)
    try:
        mc = ImageClassifierModelConfig.from_dict(raw, logger=None, warn_unknown_keys=False)
    except Exception as e:
        raise SystemExit(f"Invalid model config: {e}") from e

    wrapper = ImageClassifierWrapper(mc)
    if not wrapper.can_run:
        raise SystemExit("Classifier wrapper failed to initialize (can_run=False); check model path and logs.")

    cats = list(mc.model_categories)
    suffixes = _image_suffixes()
    images = _gather_images(list(args.directories), args.recursive, suffixes)
    if not images:
        raise SystemExit("No image files found (check paths and extensions).")

    score_headers = [f"score_{c}" for c in cats]
    fieldnames = ["path", "argmax_index", "argmax_label", "error"] + score_headers

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for path in images:
            row: dict[str, object] = {h: "" for h in fieldnames}
            row["path"] = str(path)
            try:
                pred = wrapper.predict_image(str(path))
                scores = [float(pred.get(c, 0.0)) for c in cats]
                if not scores:
                    raise ValueError("no scores")
                best_i = max(range(len(scores)), key=lambda i: scores[i])
                row["argmax_index"] = best_i
                row["argmax_label"] = cats[best_i]
                row["error"] = ""
                for c, v in zip(cats, scores):
                    row[f"score_{c}"] = f"{v:.8f}"
            except Exception as e:
                row["argmax_index"] = ""
                row["argmax_label"] = ""
                row["error"] = str(e)
                for c in cats:
                    row[f"score_{c}"] = ""
            w.writerow(row)

    print(f"Wrote {len(images)} rows to {args.output}")


if __name__ == "__main__":
    main()
