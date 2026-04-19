"""
Template for adding a new SFT dataset.

Copy this file, rename the class, and fill in the TODOs. The dataset must
yield examples in the standard conversation format expected by
Tokenizer.render_conversation:
    {"messages": [
        {"role": "system",    "content": "..."},  # optional, leading only
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]}
with strict user/assistant alternation starting with user, and string content.

Wire the new class into scripts/chat_sft.py alongside SmolTalk / TuluLongMix.
"""

import random
from datasets import load_dataset
from tasks.common import Task
from nanochat.common import print0


class TemplateSFT(Task):
    """
    TODO: one-line description of the dataset, row count, and any quirks.
    """

    # TODO: set to the HF dataset id, e.g. "org/name"
    HF_DATASET = "TODO/dataset-name"
    # TODO: list of splits supported upstream
    SUPPORTED_SPLITS = ("train", "test")

    def __init__(self, split="train", size=None, **kwargs):
        super().__init__(**kwargs)
        assert split in self.SUPPORTED_SPLITS, (
            f"{type(self).__name__} split must be one of {self.SUPPORTED_SPLITS}"
        )
        ds = load_dataset(self.HF_DATASET, split=split)

        # Pre-pass: keep only rows that satisfy the conversation contract.
        valid_indices = [i for i, row in enumerate(ds) if self._is_valid(self._extract_messages(row))]
        n_total = len(ds)
        n_valid = len(valid_indices)
        print0(f"{type(self).__name__}: {n_valid:,}/{n_total:,} rows valid ({n_total - n_valid:,} dropped)")

        # Deterministic shuffle so subsampling is stable across runs and ranks.
        rng = random.Random(42)
        rng.shuffle(valid_indices)

        if size is not None:
            assert size > 0, f"size must be positive, got {size}"
            valid_indices = valid_indices[:size]

        self.ds = ds
        self.indices = valid_indices
        self.length = len(valid_indices)

    @staticmethod
    def _extract_messages(row):
        """
        TODO: return a list[{"role", "content"}] from a raw dataset row.
        For datasets already in the messages schema, this is just row["messages"].
        Override for datasets that store prompt/response pairs or custom fields.
        """
        return row.get("messages")

    @staticmethod
    def _is_valid(messages):
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        if messages[0].get("role") == "system":
            messages = messages[1:]
        if len(messages) < 2:
            return False
        for i, m in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            if m.get("role") != expected:
                return False
            if not isinstance(m.get("content"), str):
                return False
        return True

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[self.indices[index]]
        return {"messages": self._extract_messages(row)}
