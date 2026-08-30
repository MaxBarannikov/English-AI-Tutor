from conftest import correction
from tutor.severity import select_corrections


def test_all_critical_are_shown() -> None:
    corrections = [correction("critical", "a"), correction("critical", "b")]
    assert select_corrections(corrections) == corrections


def test_at_most_one_minor_per_turn() -> None:
    shown = select_corrections(
        [correction("minor", "a"), correction("minor", "b"), correction("minor", "c")]
    )
    assert [c.original for c in shown] == ["a"]


def test_style_is_never_shown_mid_dialogue() -> None:
    assert select_corrections([correction("style"), correction("style")]) == []


def test_critical_comes_before_minor() -> None:
    shown = select_corrections(
        [
            correction("minor", "m"),
            correction("style", "s"),
            correction("critical", "c"),
        ]
    )
    assert [c.original for c in shown] == ["c", "m"]


def test_empty_input() -> None:
    assert select_corrections([]) == []
