import pytest

from src.utils import contains_urdu_script


@pytest.mark.parametrize(
    "text",
    [
        "مجھے سر درد ہے",
        "بخار",
    ],
)
def test_detects_urdu_script(text):
    assert contains_urdu_script(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "high fever",
        "mujhe bukhaar hai",
        "",
    ],
)
def test_does_not_flag_non_urdu_script(text):
    assert contains_urdu_script(text) is False


def test_handles_none_gracefully():
    assert contains_urdu_script(None) is False
