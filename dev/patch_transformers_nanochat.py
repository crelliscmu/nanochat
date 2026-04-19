"""Patch an installed `transformers` package to register the NanoChat model
with the Auto* classes.

Adds the NanoChat-specific entries to:
  - transformers/models/__init__.py
  - transformers/models/auto/configuration_auto.py
  - transformers/models/auto/modeling_auto.py
  - transformers/models/auto/tokenization_auto.py

The patch is idempotent: if an entry already exists, it is left untouched.

Usage:
    python dev/patch_transformers_nanochat.py            # auto-detect installed transformers
    python dev/patch_transformers_nanochat.py --path /path/to/site-packages/transformers
    python dev/patch_transformers_nanochat.py --dry-run  # print diffs without writing
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

# Directory containing the NanoChat model files to copy into transformers/models/nanochat/
NANOCHAT_PKG_DIR = Path(__file__).resolve().parent / "nanochat_pkg"


# (anchor_line, new_line) pairs. Insertion happens on the line immediately
# after the first exact-match occurrence of `anchor_line` in the target file.
# Every new_line is checked for existence first; duplicates are skipped.
PATCHES: dict[str, list[tuple[str, str]]] = {
    "models/__init__.py": [
        (
            "    from .myt5 import *",
            "    from .nanochat import *",
        ),
    ],
    "models/auto/configuration_auto.py": [
        (
            '        ("mvp", "MvpConfig"),',
            '        ("nanochat", "NanoChatConfig"),',
        ),
        (
            '        ("myt5", "myt5"),',
            '        ("nanochat", "NanoChat"),',
        ),
    ],
    "models/auto/modeling_auto.py": [
        # MODEL_MAPPING_NAMES (base models)
        (
            '        ("mvp", "MvpModel"),',
            '        ("nanochat", "NanoChatModel"),',
        ),
        # MODEL_FOR_PRE_TRAINING_MAPPING_NAMES
        (
            '        ("mvp", "MvpForConditionalGeneration"),',
            '        ("nanochat", "NanoChatForCausalLM"),',
        ),
        # MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        (
            '        ("mvp", "MvpForCausalLM"),',
            '        ("nanochat", "NanoChatForCausalLM"),',
        ),
    ],
    "models/auto/tokenization_auto.py": [
        (
            '        ("myt5", ("MyT5Tokenizer", None)),',
            '        ("nanochat", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),',
        ),
    ],
}


def copy_nanochat_model_files(root: Path, dry_run: bool = False) -> int:
    """Copy nanochat_pkg/ files into transformers/models/nanochat/."""
    dest = root / "models" / "nanochat"
    if not NANOCHAT_PKG_DIR.is_dir():
        print(f"[skip] nanochat_pkg not found at {NANOCHAT_PKG_DIR}")
        return 0

    src_files = list(NANOCHAT_PKG_DIR.iterdir())
    if not src_files:
        print(f"[skip] nanochat_pkg is empty")
        return 0

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(src_files):
        if not src.is_file():
            continue
        dst = dest / src.name
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"[ok]   models/nanochat/{src.name}: already up to date")
            continue
        if dry_run:
            print(f"[copy] models/nanochat/{src.name} (dry-run)")
        else:
            shutil.copy2(src, dst)
            print(f"[copy] models/nanochat/{src.name}")
        copied += 1

    return copied


def find_transformers_root() -> Path:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate an installed `transformers` package. "
            "Pass --path explicitly."
        )
    return Path(spec.origin).parent


def apply_patches(root: Path, dry_run: bool = False) -> int:
    if not root.is_dir():
        raise FileNotFoundError(f"Transformers root not found: {root}")

    total_insertions = 0
    for rel_path, edits in PATCHES.items():
        target = root / rel_path
        if not target.is_file():
            print(f"[skip] {target} (file not found)")
            continue

        original = target.read_text().splitlines(keepends=True)
        lines = list(original)
        file_insertions = 0

        for anchor, new_line in edits:
            new_line_stripped = new_line.strip()
            if any(line.strip() == new_line_stripped for line in lines):
                print(f"[ok]   {rel_path}: already contains `{new_line_stripped}`")
                continue

            anchor_idx = None
            for i, line in enumerate(lines):
                if line.rstrip("\n") == anchor:
                    anchor_idx = i
                    break
            if anchor_idx is None:
                print(
                    f"[warn] {rel_path}: anchor not found, cannot insert "
                    f"`{new_line_stripped}`"
                )
                continue

            lines.insert(anchor_idx + 1, new_line + "\n")
            file_insertions += 1
            print(f"[add]  {rel_path}: inserted `{new_line_stripped}`")

        if file_insertions and not dry_run:
            target.write_text("".join(lines))
        total_insertions += file_insertions

    return total_insertions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to the installed transformers package "
        "(the directory containing models/). Auto-detected by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing to disk.",
    )
    args = parser.parse_args()

    root = args.path if args.path is not None else find_transformers_root()
    print(f"Patching transformers at: {root}")
    if args.dry_run:
        print("(dry-run: no files will be modified)")

    copied = copy_nanochat_model_files(root, dry_run=args.dry_run)
    inserted = apply_patches(root, dry_run=args.dry_run)
    print(f"Done. {copied} file(s) copied, {inserted} line(s) inserted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
