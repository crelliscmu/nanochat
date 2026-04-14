"""
NoLiMa: long-context evaluation beyond literal matching.
https://github.com/adobe-research/NoLiMa
HuggingFace: https://huggingface.co/datasets/amodaresi/NoLiMa

This is a *simplified* port designed to be self-contained: it pulls only the
needle JSON files from the upstream HF dataset, and uses TinyShakespeare as a
zero-config haystack source instead of the much larger book corpus the upstream
benchmark relies on. The needle insertion, depth sweep, and `lastline_contains`
scoring follow the upstream protocol.

Each Task example corresponds to one (needle, test, character, depth) tuple at
a fixed `context_length`. The driver script
(`scripts/longcontext_eval.py`) is responsible for tokenizing, optionally
re-trimming the haystack to fit a target token budget, and dispatching to
generation.

Differences from upstream NoLiMa (acceptable for a small-model nanochat eval):
- Single haystack source (TinyShakespeare) repeated to reach target length,
  instead of randomly shuffled book pages from the upstream `rand_shuffle/` set.
- Character selection is deterministic per example index (seeded RNG) so runs
  are reproducible.
- We always use `metric="lastline_contains"`; upstream supports four metric
  variants.
"""

import os
import json
import random

from tasks.common import Task
from tasks.longcontext_metrics import em_contains
from nanochat.common import get_base_dir, download_file_with_lock

NEEDLE_SET_URL_TEMPLATE = (
    "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/{name}.json"
)
HAYSTACK_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
HAYSTACK_FILENAME = "longcontext/nolima/tinyshakespeare.txt"


def _ensure_needle_set(name):
    rel_path = f"longcontext/nolima/{name}.json"
    target = os.path.join(get_base_dir(), rel_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return download_file_with_lock(NEEDLE_SET_URL_TEMPLATE.format(name=name), rel_path)


def _ensure_haystack():
    target = os.path.join(get_base_dir(), HAYSTACK_FILENAME)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return download_file_with_lock(HAYSTACK_URL, HAYSTACK_FILENAME)


def _load_haystack_text():
    path = _ensure_haystack()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_haystack(base_text, target_chars):
    """Repeat / truncate base_text to roughly target_chars characters."""
    if len(base_text) >= target_chars:
        return base_text[:target_chars]
    n_repeats = (target_chars // len(base_text)) + 1
    return (base_text * n_repeats)[:target_chars]


def _substitute(template, char_name, input_args):
    """Replace {CHAR} and {1}, {2}, ... in a template string."""
    out = template.replace("{CHAR}", char_name)
    for i, arg in enumerate(input_args, start=1):
        out = out.replace("{" + str(i) + "}", arg)
    return out


def _flatten_needles(needle_set):
    """
    Flatten the needle JSON into a list of (needle_template, test_id) base tuples.
    Each needle template typically defines several `tests`; we expand them.
    """
    base = []
    for needle_template in needle_set:
        for test_id, test_def in needle_template.get("tests", {}).items():
            base.append((needle_template, test_id, test_def))
    return base


class NoLiMa(Task):

    def __init__(
        self,
        needle_set="needle_set",
        context_length=4096,
        depths=(0.0, 0.25, 0.5, 0.75, 1.0),
        question_kind="onehop",
        seed=1337,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.needle_set_name = needle_set
        self.context_length = context_length
        self.depths = tuple(depths)
        self.question_kind = question_kind
        self.seed = seed

        needle_set_path = _ensure_needle_set(needle_set)
        with open(needle_set_path, "r", encoding="utf-8") as f:
            self.needle_set = json.load(f)
        self.base_tuples = _flatten_needles(self.needle_set)
        # Each example is one (base_tuple, depth) pair
        self.length = len(self.base_tuples) * len(self.depths)

        self._haystack_text = _load_haystack_text()
        # Approximate target chars: nanochat tokenizer averages ~3.5–4 chars/token.
        # Slightly overshoot so the script's tokenizer-based middle-truncation has
        # room to clip rather than starve the haystack.
        self.target_chars = context_length * 4

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        assert 0 <= index < self.length, f"Index {index} out of range"
        n_depths = len(self.depths)
        base_idx = index // n_depths
        depth_idx = index % n_depths
        depth = self.depths[depth_idx]

        needle_template, test_id, test_def = self.base_tuples[base_idx]
        input_args = test_def["input_args"]
        character_set = needle_template.get("character_set", ["Alex"])

        # Deterministic character pick based on (seed, base_idx, depth_idx)
        rng = random.Random((self.seed, base_idx, depth_idx))
        char_name = rng.choice(character_set)

        needle_text = _substitute(needle_template["needle"], char_name, input_args)
        question_template = needle_template["questions"][self.question_kind]
        question_text = _substitute(question_template, char_name, input_args)

        # Build the haystack: repeat the base text to roughly target_chars,
        # then splice the needle in at the chosen depth.
        haystack = _build_haystack(self._haystack_text, self.target_chars)
        cut = int(len(haystack) * depth)
        # Snap the cut to the nearest space so we don't insert mid-word.
        if 0 < cut < len(haystack):
            space_back = haystack.rfind(" ", 0, cut)
            if space_back > 0:
                cut = space_back
        haystack_with_needle = haystack[:cut] + " " + needle_text + " " + haystack[cut:]

        prompt = needle_template["task_template"].format(
            haystack=haystack_with_needle,
            question=question_text,
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
        return {
            "messages": messages,
            "gold_answers": [char_name],
            "metric": "lastline_contains",
            "context_length": self.context_length,
            "depth": depth,
            "needle_id": needle_template.get("id", "?"),
            "test_id": test_id,
        }

    def evaluate(self, conversation, completion):
        return float(em_contains(
            completion,
            conversation["gold_answers"],
            metric=conversation.get("metric", "lastline_contains"),
        ))
