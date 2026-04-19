"""
Scoring functions for long-context evaluations.

The LongBench-derived metrics (qa_f1_score, rouge_score, classification_score,
retrieval_score, count_score, code_sim_score) are ports of the English half of
THUDM/LongBench/LongBench/metrics.py. The Chinese variants are intentionally
omitted because the nanochat tokenizer is English-only.

em_contains is the NoLiMa scoring function (EM | contains | lastline_EM |
lastline_contains).
"""

import re
import string
from collections import Counter

# rouge_score and fuzzywuzzy are optional imports — see pyproject.toml extras.
# Tasks that need them check availability at construction time and raise a
# clear error if missing.
try:
    from rouge_score import rouge_scorer
    _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except ImportError:
    _ROUGE_SCORER = None

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None


def normalize_answer(s):
    """Lowercase, strip punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truths, **_kwargs):
    """Token-level F1, max over the list of gold answers."""
    if not ground_truths:
        return 0.0
    scores = []
    for gt in ground_truths:
        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(gt).split()
        if not pred_tokens or not gold_tokens:
            scores.append(0.0)
            continue
        scores.append(_f1(pred_tokens, gold_tokens))
    return max(scores)


def rouge_score(prediction, ground_truths, **_kwargs):
    """ROUGE-L F1, max over gold answers. Requires `rouge-score` package."""
    if _ROUGE_SCORER is None:
        raise ImportError(
            "rouge_score requires the `rouge-score` package. "
            "Install with `pip install -e \".[longcontext]\"`."
        )
    if not ground_truths:
        return 0.0
    scores = [_ROUGE_SCORER.score(gt, prediction)["rougeL"].fmeasure for gt in ground_truths]
    return max(scores)


def classification_score(prediction, ground_truths, all_classes=None, **_kwargs):
    """
    LongBench classification score: count how many classes (other than the
    correct one) appear as substrings in the prediction. Score is 1.0 only when
    the gold class appears AND no wrong class does.
    """
    if not all_classes:
        return float(any(gt in prediction for gt in ground_truths))
    em_match_list = []
    for c in all_classes:
        if c in prediction:
            em_match_list.append(c)
    for gt in ground_truths:
        if gt in em_match_list:
            score = 1.0 / len(em_match_list)
            return score
    return 0.0


def retrieval_score(prediction, ground_truths, **_kwargs):
    """LongBench passage_retrieval_en: fraction of numbers in prediction that match the gold paragraph id."""
    pred_numbers = re.findall(r"\d+", prediction)
    if not pred_numbers:
        return 0.0
    scores = []
    for gt in ground_truths:
        gt_matches = re.findall(r"Paragraph (\d+)", gt)
        if not gt_matches:
            gt_matches = re.findall(r"\d+", gt)
        if not gt_matches:
            continue
        gt_id = gt_matches[0]
        right = sum(1 for n in pred_numbers if n == gt_id)
        scores.append(right / len(pred_numbers))
    return max(scores) if scores else 0.0


def count_score(prediction, ground_truths, **_kwargs):
    """LongBench passage_count: fraction of predicted numbers that match the gold count."""
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    scores = []
    for gt in ground_truths:
        gt_num = str(gt).strip()
        right = sum(1 for n in numbers if n == gt_num)
        scores.append(right / len(numbers))
    return max(scores) if scores else 0.0


def code_sim_score(prediction, ground_truths, **_kwargs):
    """LongBench lcc / repobench-p: fuzzy similarity on the first non-comment line.

    Matches THUDM upstream: pick the first line that contains none of `, #, //.
    """
    if fuzz is None:
        raise ImportError(
            "code_sim_score requires the `fuzzywuzzy` package. "
            "Install with `pip install -e \".[longcontext]\"`."
        )
    if not ground_truths:
        return 0.0
    all_lines = prediction.lstrip("\n").split("\n")
    pred_first = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            pred_first = line
            break
    scores = [fuzz.ratio(pred_first, gt) / 100.0 for gt in ground_truths]
    return max(scores)


def em_contains(prediction, ground_truths, metric="contains", **_kwargs):
    """
    NoLiMa scoring. metric is one of:
      - "EM"               : exact match against (stripped) prediction
      - "contains"         : gold appears anywhere in prediction
      - "lastline_EM"      : exact match against the prediction's last non-empty line
      - "lastline_contains": gold appears in the prediction's last non-empty line
    """
    if not ground_truths:
        return 0.0
    target = prediction.strip()
    if metric.startswith("lastline_"):
        lines = [ln for ln in target.splitlines() if ln.strip()]
        target = lines[-1].strip() if lines else ""
    target_lower = target.lower()
    if metric.endswith("EM"):
        for gt in ground_truths:
            if target_lower == gt.strip().lower():
                return 1.0
        return 0.0
    # contains variants
    for gt in ground_truths:
        if gt.strip().lower() in target_lower:
            return 1.0
    return 0.0


# LongBench English dataset → metric function lookup
DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
