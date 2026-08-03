from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import parser as quantum_parser
from qregister import QRegister


class QRegisterFormattingTests(unittest.TestCase):
    def test_symbolic_amplitudes_keep_common_ratios(self) -> None:
        register = QRegister(3)
        register.amps = np.array(
            [
                1 / 2,
                1 / np.sqrt(2),
                0,
                0,
                0,
                np.sqrt(3) / 2 + 0.5j,
                0,
                -1j,
            ],
            dtype=np.complex128,
        )

        text = str(register)

        self.assertIn("1/2|000❯", text)
        self.assertIn("1/√2|001❯", text)
        self.assertIn("(√3/2 + 1/2j)|101❯", text)
        self.assertIn("-j|111❯", text)

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
            register = QRegister(1)
            register.amps = amps
            self.assertEqual(str(register), expected)


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
            state.qbits_dict["q"].amps,
            np.array([0, -1j, 0, 0], dtype=np.complex128),
        )
        self.assertEqual(state.bits_dict["b"][0], 0)

    def test_collect_unique_qbit_groups_returns_unique_indices(self) -> None:
        register = QRegister(3)

        quantum_parser.H.apply(register[0])
        quantum_parser.X.apply(register[1])
        quantum_parser.H.apply(register[2])

        groups = register.dissociate()
        indices = [tuple(group.indices) for group in groups]

        self.assertEqual(
            indices,
            [
                (0,),
                (1,),
                (2,),
            ],
        )

    def test_collect_unique_qbit_groups_finds_two_factor_groups(self) -> None:
        register = QRegister(5)

        quantum_parser.H.apply(register[0])
        quantum_parser.CX.apply(register[0], register[1])

        quantum_parser.H.apply(register[2])
        quantum_parser.CX.apply(register[2], register[3])
        quantum_parser.CX.apply(register[3], register[4])

        groups = register.dissociate()
        indices = [tuple(group.indices) for group in groups]

        self.assertEqual(indices, [(0, 1), (2, 3, 4)])


if __name__ == "__main__":
    unittest.main()