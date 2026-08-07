from __future__ import annotations

import contextlib
from unittest.mock import patch

import numpy as np

from quantum import CX, H, Measure, QRegister


@contextlib.contextmanager
def patch_random_choice(return_value):
    with patch.object(np.random, "choice", return_value=return_value):
        yield

def test_register_starts_in_zero_state() -> None:
    register = QRegister(3)

    assert register.count == 3
    assert str(register) == "|000❯"


def test_getitem_rejects_out_of_range_index() -> None:
    register = QRegister(2)

    try:
        _ = register[2]
        assert False, "Expected IndexError"
    except IndexError:
        pass


def test_h_creates_equal_superposition() -> None:
    register = QRegister(1)

    H.apply(register[0])

    np.testing.assert_allclose(
        register.amps,
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128),
    )


def test_cx_flips_target_when_control_is_one() -> None:
    register = QRegister(2)
    register.amps = np.array([0, 0, 0, 1], dtype=np.complex128)

    CX.apply(register[0], register[1])

    np.testing.assert_allclose(
        register.amps,
        np.array([0, 0, 1, 0], dtype=np.complex128),
    )


def test_measure_collapses_state() -> None:
    register = QRegister(1)
    H.apply(register[0])

    with patch_random_choice(1):
        measured_value = Measure.apply(register[0])

    assert measured_value == 1
    np.testing.assert_allclose(
        register.amps,
        np.array([0, 1], dtype=np.complex128),
    )
