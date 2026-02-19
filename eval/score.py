"""
Scoring functions for OOLONG, OOLONG-Pairs, and S-NIAH benchmarks.
"""

import re


# ---------------------------------------------------------------------------
# OOLONG
# ---------------------------------------------------------------------------

def score_oolong(prediction: str, answer) -> float:
    """Score a single OOLONG prediction.

    answer may be a list (as returned by the dataset) or a plain string.
    Numeric answers use 0.75^|y - y_hat|; all others use exact match.
    """
    if isinstance(answer, list):
        answer = answer[0]

    pred = prediction.strip()
    ans = str(answer).strip()

    # Try numeric scoring first
    try:
        y_hat = float(pred.replace(",", ""))
        y = float(ans.replace(",", ""))
        return 0.75 ** abs(y - y_hat)
    except ValueError:
        pass

    # Exact match (case-insensitive)
    return 1.0 if pred.lower() == ans.lower() else 0.0


def evaluate_oolong_results(results: list) -> float:
    """Return average OOLONG score as a percentage (0–100)."""
    scores = [score_oolong(r["prediction"], r["answer"]) for r in results]
    return sum(scores) / len(scores) * 100


# ---------------------------------------------------------------------------
# OOLONG-Pairs
# ---------------------------------------------------------------------------

def parse_pairs(text: str) -> set:
    """Parse (user_id_1, user_id_2) pairs from model output."""
    pairs = set()
    for match in re.finditer(r'\((\d+),\s*(\d+)\)', text):
        a, b = int(match.group(1)), int(match.group(2))
        pairs.add((min(a, b), max(a, b)))
    return pairs


def f1_pairs(predicted: str, gold: str) -> float:
    """F1 score over predicted vs. gold user-ID pairs."""
    pred_set = parse_pairs(predicted)
    gold_set = parse_pairs(gold)

    #if not pred_set and not gold_set:
    #    return 1.0
    if not pred_set or not gold_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_oolong_pairs_results(results: list) -> float:
    """Return average F1 over OOLONG-Pairs tasks as a percentage (0–100)."""
    scores = [f1_pairs(r["prediction"], r["answer"]) for r in results]
    return sum(scores) / len(scores) * 100


# ---------------------------------------------------------------------------
# S-NIAH
# ---------------------------------------------------------------------------

def score_sniah(prediction: str, answer: str) -> float:
    """1.0 if the answer appears anywhere in the prediction, else 0.0."""
    return 1.0 if answer.strip().lower() in prediction.strip().lower() else 0.0


def evaluate_sniah_results(results_by_length: dict) -> dict:
    """Return % correct per context length."""
    scores = {}
    for length, results in results_by_length.items():
        correct = sum(score_sniah(r["prediction"], r["answer"]) for r in results)
        scores[length] = correct / len(results) * 100
    return scores
