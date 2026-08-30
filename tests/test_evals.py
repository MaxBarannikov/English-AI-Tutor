"""Scoring is arithmetic, not prompting — it gets its own tests."""

from collections import Counter

from run import DATASET, Example, f_score, load, score


def _example(correct: bool, *expected: str) -> Example:
    return Example(
        {
            "id": "x",
            "text": "t",
            "level": "B1",
            "correct": correct,
            "expected": [{"error_type": e} for e in expected],
        }
    )


def test_f_score_weights_precision_twice() -> None:
    # F0.5 sits closer to precision than to recall.
    assert f_score(1.0, 1.0) == 1.0
    assert f_score(1.0, 0.5) > f_score(0.5, 1.0)
    assert f_score(0.0, 0.0) == 0.0


def test_perfect_predictions() -> None:
    examples = [_example(False, "tense"), _example(True)]
    table, fpr = score(examples, [Counter({"tense": 1}), Counter()])

    assert "| tense | 1.00 | 1.00 | 1.00 | 1 |" in table
    assert fpr == 0.0


def test_false_positive_on_a_correct_sentence_is_counted() -> None:
    examples = [_example(True), _example(True)]
    _, fpr = score(examples, [Counter({"article": 1}), Counter()])

    assert fpr == 0.5


def test_missed_error_lowers_recall_not_precision() -> None:
    table, _ = score([_example(False, "tense")], [Counter()])

    assert "| tense | 0.00 | 0.00 | 0.00 | 1 |" in table


def test_dataset_contains_negative_controls() -> None:
    examples = load(DATASET, limit=None)

    assert any(e.correct for e in examples), "correct sentences are the FP control"
    assert all(bool(e.expected) != e.correct for e in examples)
