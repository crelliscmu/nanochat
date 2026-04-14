"""
LongGenBench: long-form generation under structured constraints.
https://github.com/mozhu621/LongGenBench

Each example asks the model to produce a long structured document (e.g. 52
weekly diary entries) where particular content must appear at particular
positions. Scoring:
  1. Split the completion into entries by `prefix` (e.g. "#*# Week").
  2. For each (entry_index, check_string) in checks_once / checks_range /
     checks_periodic, check whether the lower-cased check_string appears in the
     lower-cased entry at that index.
  3. Return the fraction of checks that pass, in [0, 1].

This collapses the upstream two-stage protocol (generate-then-quiz) into a
single generative pass. For a small model, the output budget is the limiting
factor — set `--max-new-tokens` generously when running this task.
"""

import os
import re
import json

from tasks.common import Task
from nanochat.common import get_base_dir, download_file_with_lock

DATASET_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/mozhu621/LongGenBench/main/Dataset/Dataset_{split}.json"
)


def _ensure_dataset(split):
    rel_path = f"longcontext/longgenbench/Dataset_{split}.json"
    target = os.path.join(get_base_dir(), rel_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return download_file_with_lock(DATASET_URL_TEMPLATE.format(split=split), rel_path)


def _split_into_entries(completion, prefix):
    """
    Split a completion into a list of per-entry strings using the section
    prefix. Index 0 is unused (entries are 1-indexed in the upstream JSON).

    The upstream prefix is something like "#*# Week 1 (January 1st - January
    7th):"; we use only the leading portion before the entry number to find
    section boundaries. Entries are returned in the order they appear in the
    completion.
    """
    # Strip the per-entry index from the prefix to get the section delimiter,
    # e.g. "#*# Week 1 (...)" -> "#*# Week"
    delimiter_match = re.match(r"^(\S+\s+\S+?)\s*\d", prefix)
    delimiter = delimiter_match.group(1) if delimiter_match else prefix.split()[0]

    # Split on the delimiter; the first chunk is anything before the first marker.
    parts = re.split(re.escape(delimiter), completion)
    # parts[0] is preamble; entries start at parts[1].
    entries = {}  # entry_index (int) -> entry_text (str)
    entry_re = re.compile(r"^\s*(\d+)")
    for chunk in parts[1:]:
        m = entry_re.match(chunk)
        if not m:
            continue
        idx = int(m.group(1))
        entries[idx] = chunk.lower()
    return entries


def _score_completion(completion, example):
    entries = _split_into_entries(completion, example["prefix"])
    total = 0
    passed = 0
    for kind in ("checks_once", "checks_range", "checks_periodic"):
        for idx_str, check in example.get(kind, {}).items():
            total += 1
            entry = entries.get(int(idx_str))
            if entry is not None and check.lower() in entry:
                passed += 1
    if total == 0:
        return 0.0
    return passed / total


class LongGenBench(Task):

    def __init__(self, split="short", **kwargs):
        super().__init__(**kwargs)
        assert split in ("short", "long"), f"split must be 'short' or 'long', got {split!r}"
        self.split = split
        path = _ensure_dataset(split)
        with open(path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        # max_gen_len scales with the number of entries the model needs to produce.
        # ~150 tokens per entry is a generous heuristic for a small model.
        self.max_gen_len = max(1024, 150 * max(ex.get("number", 1) for ex in self.examples))

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        ex = self.examples[index]
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ""},
        ]
        return {
            "messages": messages,
            "longgenbench_example": ex,
            "max_gen_len": self.max_gen_len,
        }

    def evaluate(self, conversation, completion):
        return float(_score_completion(completion, conversation["longgenbench_example"]))
