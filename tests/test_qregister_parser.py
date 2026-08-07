from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum import parser as quantum_parser
from quantum.qregister import QRegister


class QRegisterFormattingTests(unittest.TestCase):
    def test_symbolic_amplitudes_keep_common_ratios(self) -> None:
        state_vector = np.array(
            [
                1 / 2,
                1 / 2,
                0,
                0,
                0,
                0,
                0,
                1 / np.sqrt(2),
            ],
            dtype=np.complex128,
        )
        register = QRegister(3, density=np.outer(state_vector, state_vector.conj()))

        text = str(register)

        self.assertIn("1/2|000❯", text)
        self.assertIn("1/2|001❯", text)
        self.assertIn("1/√2|111❯", text)

    def test_single_qubit_symbolic_states_use_ket_shorthand(self) -> None:
        cases = [
            (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128), "|+❯"),
            (np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128), "|-❯"),
            (np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=np.complex128), "|j❯"),
            (
                np.array([1 / np.sqrt(2), -1j / np.sqrt(2)], dtype=np.complex128),
                "|-j❯",
            ),
        ]

        for amps, expected in cases:
            register = QRegister(1, density=np.outer(amps, amps.conj()))
            self.assertEqual(str(register), expected)


class GeneratorTests(unittest.TestCase):
    def test_ghz_generator_builds_expected_density(self) -> None:
        register = QRegister.ghz(3)
        state_vector = np.zeros(8, dtype=np.complex128)
        state_vector[0] = 1 / np.sqrt(2)
        state_vector[-1] = 1 / np.sqrt(2)

        np.testing.assert_allclose(register.density, np.outer(state_vector, state_vector.conj()))

    def test_w_state_generator_builds_expected_density(self) -> None:
        register = QRegister.w_state(3)
        state_vector = np.zeros(8, dtype=np.complex128)
        for index in range(8):
            if bin(index).count("1") == 1:
                state_vector[index] = 1 / np.sqrt(3)

        np.testing.assert_allclose(register.density, np.outer(state_vector, state_vector.conj()))

    def test_werner_generator_builds_expected_density(self) -> None:
        purity = 0.75
        register = QRegister.werner(purity)
        singlet = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2)
        expected = purity * np.outer(singlet, singlet.conj())
        expected += (1 - purity) * np.eye(4, dtype=np.complex128) / 4

        np.testing.assert_allclose(register.density, expected)
        np.testing.assert_allclose(register[0].density, np.diag([0.5, 0.5]))
        np.testing.assert_allclose(register[1].density, np.diag([0.5, 0.5]))


class ParserTests(unittest.TestCase):
    def test_parser_supports_rz_and_swap(self) -> None:
        state = quantum_parser.State()

        quantum_parser.parse(
            """
            qbits[2] q
            bits[1] b
            x q[0]
            swap q[0] q[1]
            rz(pi) q[0]  # add a phase to the |0❯ branch
            b[0] = measure q[0]
            """,
            state,
        )

        np.testing.assert_allclose(
            state.qbits_dict["q"].density,
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.complex128,
            ),
        )
        self.assertEqual(state.bits_dict["b"][0], 0)

    def test_qbit_views_expose_reduced_density_matrices(self) -> None:
        register = QRegister(2)

        quantum_parser.H.apply(register[0])
        quantum_parser.CX.apply(register[0], register[1])

        self.assertEqual(register[0].density.shape, (2, 2))
        self.assertEqual(register[1].density.shape, (2, 2))
        np.testing.assert_allclose(register[0].density, np.diag([0.5, 0.5]))
        np.testing.assert_allclose(register[1].density, np.diag([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()