"""
Dataset loading helpers for OOLONG, OOLONG-Pairs, and S-NIAH.
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# OOLONG-Pairs: 20 task definitions (verbatim from Appendix D.1)
# Each is the question text that follows "In the above data, "
# ---------------------------------------------------------------------------

OOLONG_PAIRS_TASKS = [
    # Task 1
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a numeric value or location. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 2
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with an entity or human being. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 3
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a description and abstract concept or "
    "abbreviation. Each of the questions can be labelled as one of the labels (the data "
    "does not provide the labels, you need to figure out the label from the semantics of "
    "the question): description and abstract concept, entity, human being, numeric value, "
    "location, abbreviation. In your answer, list all pairs in the format "
    "(user_id_1, user_id_2), separated by newlines.",

    # Task 4
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a human being or location, and all instances "
    "that are a human being for both users must be after January 6, 2023. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 5
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with an entity or numeric value, and all instances "
    "that are an entity for both users must be before March 15, 2023. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 6
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a location or abbreviation. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 7
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a description and abstract concept or numeric "
    "value, and all instances that are a numeric value for both users must be after "
    "February 1, 2023. Each of the questions can be labelled as one of the labels (the "
    "data does not provide the labels, you need to figure out the label from the "
    "semantics of the question): description and abstract concept, entity, human being, "
    "numeric value, location, abbreviation. In your answer, list all pairs in the format "
    "(user_id_1, user_id_2), separated by newlines.",

    # Task 8
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a human being or description and abstract "
    "concept. Each of the questions can be labelled as one of the labels (the data does "
    "not provide the labels, you need to figure out the label from the semantics of the "
    "question): description and abstract concept, entity, human being, numeric value, "
    "location, abbreviation. In your answer, list all pairs in the format "
    "(user_id_1, user_id_2), separated by newlines.",

    # Task 9
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with an entity or location, and all instances that "
    "are a location for both users must be after April 10, 2023. Each of the questions "
    "can be labelled as one of the labels (the data does not provide the labels, you "
    "need to figure out the label from the semantics of the question): description and "
    "abstract concept, entity, human being, numeric value, location, abbreviation. In "
    "your answer, list all pairs in the format (user_id_1, user_id_2), separated by "
    "newlines.",

    # Task 10
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) where both "
    "users have at least one instance with a numeric value or abbreviation, and all "
    "instances that are an abbreviation for both users must be before May 20, 2023. Each "
    "of the questions can be labelled as one of the labels (the data does not provide "
    "the labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 11
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least one instance with entity and one with abbreviation, and the other "
    "user has exactly one instance with entity. Each of the questions can be labelled as "
    "one of the labels (the data does not provide the labels, you need to figure out the "
    "label from the semantics of the question): description and abstract concept, entity, "
    "human being, numeric value, location, abbreviation. In your answer, list all pairs "
    "in the format (user_id_1, user_id_2), separated by newlines.",

    # Task 12
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least two instances with numeric value, and the other user has at least "
    "one instance with location and at least one instance with human being. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 13
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has exactly one instance with description and abstract concept, and the other "
    "user has at least one instance with abbreviation and at least one instance with "
    "entity. Each of the questions can be labelled as one of the labels (the data does "
    "not provide the labels, you need to figure out the label from the semantics of the "
    "question): description and abstract concept, entity, human being, numeric value, "
    "location, abbreviation. In your answer, list all pairs in the format "
    "(user_id_1, user_id_2), separated by newlines.",

    # Task 14
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least one instance with human being and at least one instance with "
    "numeric value, and the other user has exactly two instances with location. Each of "
    "the questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",

    # Task 15
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least one instance with entity, at least one instance with location, "
    "and at least one instance with abbreviation, and the other user has exactly one "
    "instance with numeric value. Each of the questions can be labelled as one of the "
    "labels (the data does not provide the labels, you need to figure out the label from "
    "the semantics of the question): description and abstract concept, entity, human "
    "being, numeric value, location, abbreviation. In your answer, list all pairs in the "
    "format (user_id_1, user_id_2), separated by newlines.",

    # Task 16
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least one instance with description and abstract concept and at least "
    "one instance with human being, and the other user has at least two instances with "
    "entity and exactly one instance with abbreviation. Each of the questions can be "
    "labelled as one of the labels (the data does not provide the labels, you need to "
    "figure out the label from the semantics of the question): description and abstract "
    "concept, entity, human being, numeric value, location, abbreviation. In your "
    "answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",

    # Task 17
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has exactly one instance with numeric value, and the other user has at least "
    "one instance with location and at least one instance with description and abstract "
    "concept. Each of the questions can be labelled as one of the labels (the data does "
    "not provide the labels, you need to figure out the label from the semantics of the "
    "question): description and abstract concept, entity, human being, numeric value, "
    "location, abbreviation. In your answer, list all pairs in the format "
    "(user_id_1, user_id_2), separated by newlines.",

    # Task 18
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least one instance with abbreviation and exactly one instance with human "
    "being, and the other user has at least one instance with entity and at least one "
    "instance with numeric value. Each of the questions can be labelled as one of the "
    "labels (the data does not provide the labels, you need to figure out the label from "
    "the semantics of the question): description and abstract concept, entity, human "
    "being, numeric value, location, abbreviation. In your answer, list all pairs in the "
    "format (user_id_1, user_id_2), separated by newlines.",

    # Task 19
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least two instances with location and at least one instance with entity, "
    "and the other user has exactly one instance with description and abstract concept "
    "and exactly one instance with abbreviation. Each of the questions can be labelled "
    "as one of the labels (the data does not provide the labels, you need to figure out "
    "the label from the semantics of the question): description and abstract concept, "
    "entity, human being, numeric value, location, abbreviation. In your answer, list "
    "all pairs in the format (user_id_1, user_id_2), separated by newlines.",

    # Task 20
    "List all pairs of user IDs (no duplicate pairs, list lower ID first) such that one "
    "user has at least one instance with numeric value and at least one instance with "
    "human being, and the other user has at least one instance with location, at least "
    "one instance with entity, and exactly one instance with abbreviation. Each of the "
    "questions can be labelled as one of the labels (the data does not provide the "
    "labels, you need to figure out the label from the semantics of the question): "
    "description and abstract concept, entity, human being, numeric value, location, "
    "abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), "
    "separated by newlines.",
]

assert len(OOLONG_PAIRS_TASKS) == 20, "Expected exactly 20 OOLONG-Pairs tasks"


# ---------------------------------------------------------------------------
# OOLONG dataset loading
# ---------------------------------------------------------------------------

def load_oolong(context_len: int = 131072) -> list[dict]:
    """Load OOLONG trec_coarse examples at the specified context length.

    Returns a list of dicts with keys: context, question, answer, id.
    """
    from datasets import load_dataset

    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    examples = [
        x for x in ds
        if x["dataset"] == "trec_coarse" and x["context_len"] == context_len
    ]
    return [
        {
            "id": ex["id"],
            "context": ex["context_window_text"],
            "question": ex["question"],
            "answer": ex["answer"],
        }
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# OOLONG-Pairs: ground-truth computation
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"Date:\s*(.+?)\s*\|\|\s*User:\s*(\d+)\s*\|\|\s*Instance:\s*.+?\s*\|\|\s*Label:\s*(.+?)$"
)
_DATE_FMT = "%b %d, %Y"


def _parse_labeled_context(text: str) -> dict:
    """Parse a labeled context into {user_id: [{"date": datetime|None, "label": str}]}."""
    users = defaultdict(list)
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        date_str, user_id, label = m.group(1), int(m.group(2)), m.group(3).strip()
        try:
            date = datetime.strptime(date_str, _DATE_FMT)
        except ValueError:
            date = None
        users[user_id].append({"date": date, "label": label})
    return dict(users)


def _count(instances, label):
    return sum(1 for inst in instances if inst["label"] == label)


def _has(instances, label):
    return any(inst["label"] == label for inst in instances)


def _all_after(instances, label, cutoff: datetime):
    labeled = [inst for inst in instances if inst["label"] == label]
    return bool(labeled) and all(
        inst["date"] is not None and inst["date"] > cutoff for inst in labeled
    )


def _all_before(instances, label, cutoff: datetime):
    labeled = [inst for inst in instances if inst["label"] == label]
    return bool(labeled) and all(
        inst["date"] is not None and inst["date"] < cutoff for inst in labeled
    )


def _check_pair_condition(instances_a: list[dict], instances_b: list[dict], task_idx: int) -> bool:
    """Check whether a pair of users qualifies for a given OOLONG-Pairs task.

    Args:
        instances_a: List of {"date": datetime|None, "label": str} for user A
        instances_b: List of {"date": datetime|None, "label": str} for user B
        task_idx: 1-indexed task number (1-20)

    Returns:
        True if the pair qualifies for the task condition.
    """
    a, b = instances_a, instances_b

    # Temporal cutoffs used in tasks 4, 5, 7, 9, 10
    JAN6_2023 = datetime(2023, 1, 6)
    MAR15_2023 = datetime(2023, 3, 15)
    FEB1_2023 = datetime(2023, 2, 1)
    APR10_2023 = datetime(2023, 4, 10)
    MAY20_2023 = datetime(2023, 5, 20)

    if task_idx == 1:
        return (
            (_has(a, "numeric value") or _has(a, "location"))
            and (_has(b, "numeric value") or _has(b, "location"))
        )
    elif task_idx == 2:
        return (
            (_has(a, "entity") or _has(a, "human being"))
            and (_has(b, "entity") or _has(b, "human being"))
        )
    elif task_idx == 3:
        return (
            (_has(a, "description and abstract concept") or _has(a, "abbreviation"))
            and (_has(b, "description and abstract concept") or _has(b, "abbreviation"))
        )
    elif task_idx == 4:
        def q4(u):
            return (
                (_has(u, "human being") or _has(u, "location"))
                and _all_after(u, "human being", JAN6_2023)
            )
        return q4(a) and q4(b)
    elif task_idx == 5:
        def q5(u):
            return (
                (_has(u, "entity") or _has(u, "numeric value"))
                and _all_before(u, "entity", MAR15_2023)
            )
        return q5(a) and q5(b)
    elif task_idx == 6:
        return (
            (_has(a, "location") or _has(a, "abbreviation"))
            and (_has(b, "location") or _has(b, "abbreviation"))
        )
    elif task_idx == 7:
        def q7(u):
            return (
                (_has(u, "description and abstract concept") or _has(u, "numeric value"))
                and _all_after(u, "numeric value", FEB1_2023)
            )
        return q7(a) and q7(b)
    elif task_idx == 8:
        return (
            (_has(a, "human being") or _has(a, "description and abstract concept"))
            and (_has(b, "human being") or _has(b, "description and abstract concept"))
        )
    elif task_idx == 9:
        def q9(u):
            return (
                (_has(u, "entity") or _has(u, "location"))
                and _all_after(u, "location", APR10_2023)
            )
        return q9(a) and q9(b)
    elif task_idx == 10:
        def q10(u):
            return (
                (_has(u, "numeric value") or _has(u, "abbreviation"))
                and _all_before(u, "abbreviation", MAY20_2023)
            )
        return q10(a) and q10(b)
    elif task_idx == 11:
        def role11_a(u):
            return _has(u, "entity") and _has(u, "abbreviation")
        def role11_b(u):
            return _count(u, "entity") == 1
        return (role11_a(a) and role11_b(b)) or (role11_a(b) and role11_b(a))
    elif task_idx == 12:
        def role12_a(u):
            return _count(u, "numeric value") >= 2
        def role12_b(u):
            return _has(u, "location") and _has(u, "human being")
        return (role12_a(a) and role12_b(b)) or (role12_a(b) and role12_b(a))
    elif task_idx == 13:
        def role13_a(u):
            return _count(u, "description and abstract concept") == 1
        def role13_b(u):
            return _has(u, "abbreviation") and _has(u, "entity")
        return (role13_a(a) and role13_b(b)) or (role13_a(b) and role13_b(a))
    elif task_idx == 14:
        def role14_a(u):
            return _has(u, "human being") and _has(u, "numeric value")
        def role14_b(u):
            return _count(u, "location") == 2
        return (role14_a(a) and role14_b(b)) or (role14_a(b) and role14_b(a))
    elif task_idx == 15:
        def role15_a(u):
            return _has(u, "entity") and _has(u, "location") and _has(u, "abbreviation")
        def role15_b(u):
            return _count(u, "numeric value") == 1
        return (role15_a(a) and role15_b(b)) or (role15_a(b) and role15_b(a))
    elif task_idx == 16:
        def role16_a(u):
            return _has(u, "description and abstract concept") and _has(u, "human being")
        def role16_b(u):
            return _count(u, "entity") >= 2 and _count(u, "abbreviation") == 1
        return (role16_a(a) and role16_b(b)) or (role16_a(b) and role16_b(a))
    elif task_idx == 17:
        def role17_a(u):
            return _count(u, "numeric value") == 1
        def role17_b(u):
            return _has(u, "location") and _has(u, "description and abstract concept")
        return (role17_a(a) and role17_b(b)) or (role17_a(b) and role17_b(a))
    elif task_idx == 18:
        def role18_a(u):
            return _has(u, "abbreviation") and _count(u, "human being") == 1
        def role18_b(u):
            return _has(u, "entity") and _has(u, "numeric value")
        return (role18_a(a) and role18_b(b)) or (role18_a(b) and role18_b(a))
    elif task_idx == 19:
        def role19_a(u):
            return _count(u, "location") >= 2 and _has(u, "entity")
        def role19_b(u):
            return (
                _count(u, "description and abstract concept") == 1
                and _count(u, "abbreviation") == 1
            )
        return (role19_a(a) and role19_b(b)) or (role19_a(b) and role19_b(a))
    elif task_idx == 20:
        def role20_a(u):
            return _has(u, "numeric value") and _has(u, "human being")
        def role20_b(u):
            return (
                _has(u, "location")
                and _has(u, "entity")
                and _count(u, "abbreviation") == 1
            )
        return (role20_a(a) and role20_b(b)) or (role20_a(b) and role20_b(a))
    else:
        raise ValueError(f"Unknown task index: {task_idx}")


def compute_gold_pairs(labeled_context: str, task_idx: int) -> str:
    """Compute ground-truth pair string for one OOLONG-Pairs task (1-indexed).

    Returns a newline-separated string of "(uid1, uid2)" with uid1 < uid2.
    """
    users = _parse_labeled_context(labeled_context)
    user_ids = sorted(users.keys())
    gold_pairs = []

    # Temporal cutoffs used in tasks 4, 5, 7, 9, 10
    JAN6_2023  = datetime(2023, 1, 6)
    MAR15_2023 = datetime(2023, 3, 15)
    FEB1_2023  = datetime(2023, 2, 1)
    APR10_2023 = datetime(2023, 4, 10)
    MAY20_2023 = datetime(2023, 5, 20)

    for uid1, uid2 in combinations(user_ids, 2):
        a, b = users[uid1], users[uid2]
        qualifies = False

        if task_idx == 1:
            qualifies = (
                (_has(a, "numeric value") or _has(a, "location")) and
                (_has(b, "numeric value") or _has(b, "location"))
            )
        elif task_idx == 2:
            qualifies = (
                (_has(a, "entity") or _has(a, "human being")) and
                (_has(b, "entity") or _has(b, "human being"))
            )
        elif task_idx == 3:
            qualifies = (
                (_has(a, "description and abstract concept") or _has(a, "abbreviation")) and
                (_has(b, "description and abstract concept") or _has(b, "abbreviation"))
            )
        elif task_idx == 4:
            # both have ≥1 of (human being, location); all human being instances after Jan 6 2023
            def q4(u):
                return (
                    (_has(u, "human being") or _has(u, "location")) and
                    _all_after(u, "human being", JAN6_2023)
                )
            qualifies = q4(a) and q4(b)
        elif task_idx == 5:
            # both have ≥1 of (entity, numeric value); all entity instances before Mar 15 2023
            def q5(u):
                return (
                    (_has(u, "entity") or _has(u, "numeric value")) and
                    _all_before(u, "entity", MAR15_2023)
                )
            qualifies = q5(a) and q5(b)
        elif task_idx == 6:
            qualifies = (
                (_has(a, "location") or _has(a, "abbreviation")) and
                (_has(b, "location") or _has(b, "abbreviation"))
            )
        elif task_idx == 7:
            # both have ≥1 of (description, numeric value); all numeric value after Feb 1 2023
            def q7(u):
                return (
                    (_has(u, "description and abstract concept") or _has(u, "numeric value")) and
                    _all_after(u, "numeric value", FEB1_2023)
                )
            qualifies = q7(a) and q7(b)
        elif task_idx == 8:
            qualifies = (
                (_has(a, "human being") or _has(a, "description and abstract concept")) and
                (_has(b, "human being") or _has(b, "description and abstract concept"))
            )
        elif task_idx == 9:
            # both have ≥1 of (entity, location); all location instances after Apr 10 2023
            def q9(u):
                return (
                    (_has(u, "entity") or _has(u, "location")) and
                    _all_after(u, "location", APR10_2023)
                )
            qualifies = q9(a) and q9(b)
        elif task_idx == 10:
            # both have ≥1 of (numeric value, abbreviation); all abbreviation before May 20 2023
            def q10(u):
                return (
                    (_has(u, "numeric value") or _has(u, "abbreviation")) and
                    _all_before(u, "abbreviation", MAY20_2023)
                )
            qualifies = q10(a) and q10(b)
        elif task_idx == 11:
            # one: ≥1 entity AND ≥1 abbreviation; other: exactly 1 entity
            def role11_a(u):
                return _has(u, "entity") and _has(u, "abbreviation")
            def role11_b(u):
                return _count(u, "entity") == 1
            qualifies = (role11_a(a) and role11_b(b)) or (role11_a(b) and role11_b(a))
        elif task_idx == 12:
            # one: ≥2 numeric value; other: ≥1 location AND ≥1 human being
            def role12_a(u):
                return _count(u, "numeric value") >= 2
            def role12_b(u):
                return _has(u, "location") and _has(u, "human being")
            qualifies = (role12_a(a) and role12_b(b)) or (role12_a(b) and role12_b(a))
        elif task_idx == 13:
            # one: exactly 1 description; other: ≥1 abbreviation AND ≥1 entity
            def role13_a(u):
                return _count(u, "description and abstract concept") == 1
            def role13_b(u):
                return _has(u, "abbreviation") and _has(u, "entity")
            qualifies = (role13_a(a) and role13_b(b)) or (role13_a(b) and role13_b(a))
        elif task_idx == 14:
            # one: ≥1 human being AND ≥1 numeric value; other: exactly 2 location
            def role14_a(u):
                return _has(u, "human being") and _has(u, "numeric value")
            def role14_b(u):
                return _count(u, "location") == 2
            qualifies = (role14_a(a) and role14_b(b)) or (role14_a(b) and role14_b(a))
        elif task_idx == 15:
            # one: ≥1 entity, ≥1 location, ≥1 abbreviation; other: exactly 1 numeric value
            def role15_a(u):
                return _has(u, "entity") and _has(u, "location") and _has(u, "abbreviation")
            def role15_b(u):
                return _count(u, "numeric value") == 1
            qualifies = (role15_a(a) and role15_b(b)) or (role15_a(b) and role15_b(a))
        elif task_idx == 16:
            # one: ≥1 description AND ≥1 human being; other: ≥2 entity AND exactly 1 abbreviation
            def role16_a(u):
                return _has(u, "description and abstract concept") and _has(u, "human being")
            def role16_b(u):
                return _count(u, "entity") >= 2 and _count(u, "abbreviation") == 1
            qualifies = (role16_a(a) and role16_b(b)) or (role16_a(b) and role16_b(a))
        elif task_idx == 17:
            # one: exactly 1 numeric value; other: ≥1 location AND ≥1 description
            def role17_a(u):
                return _count(u, "numeric value") == 1
            def role17_b(u):
                return _has(u, "location") and _has(u, "description and abstract concept")
            qualifies = (role17_a(a) and role17_b(b)) or (role17_a(b) and role17_b(a))
        elif task_idx == 18:
            # one: ≥1 abbreviation AND exactly 1 human being; other: ≥1 entity AND ≥1 numeric value
            def role18_a(u):
                return _has(u, "abbreviation") and _count(u, "human being") == 1
            def role18_b(u):
                return _has(u, "entity") and _has(u, "numeric value")
            qualifies = (role18_a(a) and role18_b(b)) or (role18_a(b) and role18_b(a))
        elif task_idx == 19:
            # one: ≥2 location AND ≥1 entity; other: exactly 1 description AND exactly 1 abbreviation
            def role19_a(u):
                return _count(u, "location") >= 2 and _has(u, "entity")
            def role19_b(u):
                return (_count(u, "description and abstract concept") == 1 and
                        _count(u, "abbreviation") == 1)
            qualifies = (role19_a(a) and role19_b(b)) or (role19_a(b) and role19_b(a))
        elif task_idx == 20:
            # one: ≥1 numeric value AND ≥1 human being; other: ≥1 location, ≥1 entity, exactly 1 abbreviation
            def role20_a(u):
                return _has(u, "numeric value") and _has(u, "human being")
            def role20_b(u):
                return (_has(u, "location") and _has(u, "entity") and
                        _count(u, "abbreviation") == 1)
            qualifies = (role20_a(a) and role20_b(b)) or (role20_a(b) and role20_b(a))
        else:
            raise ValueError(f"Unknown task index: {task_idx}")

        if qualifies:
            gold_pairs.append(f"({min(uid1, uid2)}, {max(uid1, uid2)})")

    return "\n".join(gold_pairs)


def load_oolong_pairs(context_len: int = 131072, gold_file: str | None = None) -> list[dict]:
    """Build OOLONG-Pairs dataset: 20 tasks over the trec_coarse corpus.

    Returns a list of dicts with keys: task_id, context, question, answer.
    The context shown to the model is context_window_text (without labels).

    If gold_file is provided, gold answers are loaded from that JSON file
    (format: list of {id, question, answer: [pair_str, ...], type}).
    Otherwise, gold answers are computed from the labeled context.
    """
    from datasets import load_dataset

    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    trec_at_len = [
        x for x in ds
        if x["dataset"] == "trec_coarse" and x["context_len"] == context_len
    ]
    if not trec_at_len:
        raise ValueError(f"No trec_coarse examples at context_len={context_len}")

    corpus_example = trec_at_len[0]
    context_text   = corpus_example["context_window_text"]

    if gold_file is not None:
        print(f"Loading gold answers from {gold_file}...")
        with open(gold_file) as f:
            gold_data = json.load(f)
        # gold_data is a list of {id, question, answer: [pair_str, ...], type}
        # answer list → newline-joined string for scoring
        gold_by_id = {int(entry["id"]): "\n".join(entry["answer"]) for entry in gold_data}
        question_by_id = {int(entry["id"]): entry["question"] for entry in gold_data}
        examples = []
        for task_idx in range(1, 21):
            examples.append({
                "task_id": task_idx,
                "context": context_text,
                "question": question_by_id[task_idx],
                "answer": gold_by_id[task_idx],
            })
    else:
        labeled_text = corpus_example["context_window_text_with_labels"]
        print(f"Computing gold pairs for 20 tasks (context_len={context_len})...")
        examples = []
        for task_idx in range(1, 21):
            question = f"In the above data, {OOLONG_PAIRS_TASKS[task_idx - 1]}"
            gold = compute_gold_pairs(labeled_text, task_idx)
            examples.append({
                "task_id": task_idx,
                "context": context_text,
                "question": question,
                "answer": gold,
            })

    return examples


# ---------------------------------------------------------------------------
# S-NIAH dataset loading
# ---------------------------------------------------------------------------

SNIAH_CONTEXT_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def load_sniah_tasks(sniah_dir: str, length: int) -> list[dict]:
    """Load S-NIAH tasks for a given context length.

    Expects files named {length}.json in sniah_dir, each containing a list of:
      {"haystack": str, "question": str, "answer": str}
    """
    path = Path(sniah_dir) / f"{length}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"S-NIAH task file not found: {path}\n"
            "Generate tasks with the RULER repo: https://github.com/hsiehjackson/RULER"
        )
    with open(path) as f:
        return json.load(f)
