"""
LongBench v1 (English subsets only).
https://github.com/THUDM/LongBench
HuggingFace: https://huggingface.co/datasets/THUDM/LongBench

The Chinese subsets (multifieldqa_zh, dureader, vcsum, lsht, passage_retrieval_zh)
are intentionally omitted because the nanochat tokenizer is English-only.

Each task example carries the verbatim LongBench prompt template along with the
fields the metric needs (answers, all_classes). The driver script
(`scripts/longcontext_eval.py`) handles middle-truncation of the rendered token
sequence to fit the model's context window.

Data loading note: THUDM/LongBench is published as a HuggingFace dataset *script*
(LongBench.py), which the modern `datasets` library (v4+) refuses to execute.
We bypass this by downloading the upstream `data.zip` once and reading the
per-subset JSONL files directly.
"""

import os
import json
import shutil
import zipfile

from tasks.common import Task
from tasks.longcontext_metrics import DATASET2METRIC
from nanochat.common import get_base_dir, download_file_with_lock

DATA_ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
DATA_DIR_REL = "longcontext/longbench"


def _ensure_longbench_data():
    """Download data.zip once and extract to <base_dir>/longcontext/longbench/data/."""
    base_dir = get_base_dir()
    data_root = os.path.join(base_dir, DATA_DIR_REL)
    extracted_dir = os.path.join(data_root, "data")
    os.makedirs(data_root, exist_ok=True)
    if os.path.isdir(extracted_dir):
        return extracted_dir

    def _extract(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        # Some upstream zips wrap content in an extra folder; flatten if needed.
        if not os.path.isdir(extracted_dir):
            # Look for any directory containing *.jsonl and rename it to "data"
            for entry in os.listdir(data_root):
                candidate = os.path.join(data_root, entry)
                if os.path.isdir(candidate) and any(
                    f.endswith(".jsonl") for f in os.listdir(candidate)
                ):
                    shutil.move(candidate, extracted_dir)
                    break

    zip_rel = f"{DATA_DIR_REL}/data.zip"
    download_file_with_lock(DATA_ZIP_URL, zip_rel, postprocess_fn=_extract)
    assert os.path.isdir(extracted_dir), (
        f"Failed to extract LongBench data into {extracted_dir}"
    )
    return extracted_dir

# Verbatim copy of THUDM/LongBench/LongBench/config/dataset2prompt.json
# (English subsets only).
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a "
        "question. Answer the question asconcisely as you can, using a single phrase if "
        "possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", "
        "or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        " Answer the question based on the above article as concisely as you can, using "
        "a single phrase or sentence if possible. If the question cannot be answered "
        "based on the information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the "
        "answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or "
        "instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more "
        "sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of "
        "questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do "
        "not output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be "
        "duplicates. Please carefully read these paragraphs and determine how many "
        "unique paragraphs there are after removing duplicates. In other words, how many "
        "non-repeating paragraphs are there in total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer "
        "format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Verbatim copy of THUDM/LongBench/LongBench/config/dataset2maxlen.json
# (English subsets only).
DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

ENGLISH_SUBSETS = tuple(DATASET2PROMPT.keys())


class LongBench(Task):

    def __init__(self, subset, split="test", **kwargs):
        super().__init__(**kwargs)
        assert subset in ENGLISH_SUBSETS, (
            f"LongBench subset {subset!r} is not in the supported English set: "
            f"{ENGLISH_SUBSETS}"
        )
        assert split == "test", "LongBench v1 only ships a test split"
        self.subset = subset
        self.split = split
        self.prompt_format = DATASET2PROMPT[subset]
        self.max_gen_len = DATASET2MAXLEN[subset]
        self.metric_fn = DATASET2METRIC[subset]

        data_dir = _ensure_longbench_data()
        jsonl_path = os.path.join(data_dir, f"{subset}.jsonl")
        assert os.path.isfile(jsonl_path), f"Missing LongBench file: {jsonl_path}"
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        user_message = self.prompt_format.format(context=row["context"], input=row["input"])
        # Empty assistant turn is the convention used by render_for_completion:
        # it pops the last assistant message and primes the model with <|assistant_start|>.
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ""},
        ]
        return {
            "messages": messages,
            "answers": list(row["answers"]),
            "all_classes": row.get("all_classes"),
            "subset": self.subset,
            "max_gen_len": self.max_gen_len,
        }

    def evaluate(self, conversation, completion):
        return float(self.metric_fn(
            completion,
            conversation["answers"],
            all_classes=conversation.get("all_classes"),
        ))
