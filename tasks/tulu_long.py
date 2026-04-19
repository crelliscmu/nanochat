"""
allenai/tulu-v2-sft-long-mixture: long-context SFT mixture curated by AI2.
Combines long subsets of UltraChat, ShareGPT, OASST, and others, all in the
standard `messages` format. Used to teach the model to attend across long
conversations during SFT.

https://huggingface.co/datasets/allenai/tulu-v2-sft-long-mixture
"""

import random
from datasets import load_dataset
from tasks.common import Task
from nanochat.common import print0


class TuluLongMix(Task):
    """
    allenai/tulu-v2-sft-long-mixture. Only a `train` split exists upstream.
    Pass `size` to subsample the (deterministically shuffled) dataset.

    Filters out rows that don't fit the conversation contract expected by
    Tokenizer.render_conversation: optional leading system message, then
    strict user/assistant alternation starting with user, with string content.
    """

    def __init__(self, size=None, split="train", val_size=1500, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "TuluLongMix split must be train|val"
        assert val_size > 0, f"val_size must be positive, got {val_size}"
        ds = load_dataset("allenai/tulu-v2-sft-long-mixture", split="train")

        # Pre-pass: collect indices of rows with a valid conversation shape.
        valid_indices = []
        for i, row in enumerate(ds):
            if self._is_valid(row.get("messages")):
                valid_indices.append(i)
        n_total = len(ds)
        n_valid = len(valid_indices)
        n_dropped = n_total - n_valid
        print0(f"TuluLongMix: {n_valid:,}/{n_total:,} rows valid ({n_dropped:,} dropped)")

        # Deterministic shuffle so the train/val split is stable across runs and ranks.
        rng = random.Random(42)
        rng.shuffle(valid_indices)

        # Hold out the last `val_size` shuffled indices as val; the rest is train.
        assert val_size < len(valid_indices), f"val_size {val_size} >= valid rows {len(valid_indices)}"
        if split == "val":
            selected = valid_indices[-val_size:]
        else:
            selected = valid_indices[:-val_size]
            if size is not None:
                assert size > 0, f"size must be positive, got {size}"
                selected = selected[:size]

        self.ds = ds
        self.indices = selected
        self.length = len(selected)

    @staticmethod
    def _is_valid(messages):
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        # Strip an optional leading system message (render_conversation merges it).
        if messages[0].get("role") == "system":
            messages = messages[1:]
        if len(messages) < 2:
            return False
        # Strict user/assistant alternation starting with user, string content.
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
        return {"messages": row["messages"]}
