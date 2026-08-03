from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gates

quantum = importlib.import_module("main")


class QuantumSimulatorTests(unittest.TestCase):
    def test_register_starts_in_zero_state(self) -> None:
        register = quantum.QRegister(3)

        self.assertEqual(register.count, 3)
        self.assertEqual(str(register), "|000❯")

    def test_getitem_rejects_out_of_range_index(self) -> None:
        register = quantum.QRegister(2)

        with self.assertRaises(IndexError):
            _ = register[2]

    def test_h_creates_equal_superposition(self) -> None:
        register = quantum.QRegister(1)

        quantum.H.apply(register[0])

        np.testing.assert_allclose(
            register.amps,
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128),
        )

    def test_cx_flips_target_when_control_is_one(self) -> None:
        register = quantum.QRegister(2)
        register.amps = np.array([0, 0, 0, 1], dtype=np.complex128)

        quantum.CX.apply(register[0], register[1])

        np.testing.assert_allclose(
            register.amps,
            np.array([0, 0, 1, 0], dtype=np.complex128),
        )

    def test_measure_collapses_state(self) -> None:
        register = quantum.QRegister(1)
        quantum.H.apply(register[0])

        with patch.object(gates.np.random, "choice", return_value=1):
            measured_value = quantum.Measure.apply(register[0])

        self.assertEqual(measured_value, 1)
        np.testing.assert_allclose(
            register.amps,
            np.array([0, 1], dtype=np.complex128),
        )


if __name__ == "__main__":
    unittest.main()