import pytest

from src.emergency import check_emergency


@pytest.mark.parametrize(
    "message",
    [
        "I have chest pain",
        "seenay mein dard ho raha hai",
        "cannot breathe",
        "bohat zyada bleeding ho rahi hai",
    ],
)
def test_detects_known_emergencies(message):
    assert check_emergency(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I have a mild headache",
        "mujhe halka bukhaar hai",
        "",
    ],
)
def test_does_not_flag_non_emergencies(message):
    assert check_emergency(message) is False


def test_handles_none_gracefully():
    assert check_emergency(None) is False
