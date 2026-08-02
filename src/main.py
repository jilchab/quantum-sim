from __future__ import annotations

import functools as ft
from dataclasses import dataclass

import numpy as np

FULL_MATRIX_CALCULATION = False

class QRegister:
    def __init__(self, count: int):
        self.count = count
        self.amps = np.zeros((2 ** self.count,), dtype=np.complex128)
        self.amps[0] = 1.0

    def probabilities(self):
        return np.abs(self.amps) ** 2

    def __getitem__(self, index: int) -> QbitRef:
        if index < 0 or index >= self.count:
            raise IndexError("QRegister index out of range.")
        return QbitRef(register=self, index=index)

    def __str__(self):
        s = ""
        for i in range(2 ** self.count):
            if np.isclose(self.amps[i], 0):
                continue

            bin_str = format(i, f"0{self.count}b")
            match np.round(self.amps[i], 3):
                case 1j:
                    amp = "+j"
                case -1j:
                    amp = "-j"
                case 1:
                    amp = "+"
                case -1:
                    amp = "-"
                case 0.5:
                    amp = "+1/2"
                case -0.5:
                    amp = "-1/2"
                case 0.707:
                    amp = "+1/√2"
                case -0.707:
                    amp = "-1/√2"
                case 0.707 + 0.707j:
                    amp = "+(1/√2 + 1/√2j)"
                case 0.707 - 0.707j:
                    amp = "+(1/√2 - 1/√2j)"
                case -0.707 + 0.707j:
                    amp = "-(1/√2 - 1/√2j)"
                case -0.707 - 0.707j:
                    amp = "-(1/√2 + 1/√2j)"
                case _:
                    amp = str(np.round(self.amps[i], 3))
            s += f"{amp}|{bin_str}> "
            if s.startswith("+"):
                s = s[1:]
        return s.strip()


@dataclass(frozen=True)
class QbitRef:
    register: QRegister
    index: int

class MatrixGate:
    matrix: np.ndarray

class OneQbitGate(MatrixGate):
    @classmethod
    def apply(cls, qbit: QbitRef) -> QRegister:
        register = qbit.register

        if FULL_MATRIX_CALCULATION:
            # Calculate the full matrix for the gate operation on the entire register
            operations = [cls.matrix if i == qbit.index else np.eye(2) for i in range(register.count)]
            full_matrix = ft.reduce(np.kron, operations)  # ty:ignore[invalid-argument-type]
            register.amps = full_matrix @ register.amps
            return register


        # Find pairs to apply the gate to the target qubit while leaving others unchanged
        for i in range(2 ** register.count):
            if ((i >> (register.count - 1 - qbit.index)) & 1) == 0:
                # Apply the gate to the target qubit
                j = i | (1 << (register.count - 1 - qbit.index))  # Flip the target qubit
                amp_i = register.amps[i]
                amp_j = register.amps[j]
                register.amps[i] = cls.matrix[0, 0] * amp_i + cls.matrix[0, 1] * amp_j
                register.amps[j] = cls.matrix[1, 0] * amp_i + cls.matrix[1, 1] * amp_j

        return register

class QControlledGate(MatrixGate):
    @classmethod
    def apply(cls, control: QbitRef, target: QbitRef) -> QRegister:
        if control.register is not target.register:
            raise ValueError("Control and target qubits must belong to the same register.")

        register = control.register

        if FULL_MATRIX_CALCULATION:
            # Calculate the full matrix for the controlled gate operation on the entire register

            projector_0 = np.array([[1, 0], [0, 0]])
            projector_1 = np.array([[0, 0], [0, 1]])

            operations = []

            for i in range(register.count):
                if i == control.index:
                    operations.append(projector_0)
                else:
                    operations.append(np.eye(2))
            full_matrix_0 = ft.reduce(np.kron, operations)  # ty:ignore[invalid-argument-type]

            operations = []
            for i in range(register.count):
                if i == control.index:
                    operations.append(projector_1)
                elif i == target.index:
                    operations.append(cls.matrix)
                else:
                    operations.append(np.eye(2))
            full_matrix_1 = ft.reduce(np.kron, operations)  # ty:ignore[invalid-argument-type]

            register.amps = full_matrix_0 @ register.amps + full_matrix_1 @ register.amps
            return register


        old_register_amps = register.amps.copy()
        # Find pairs to apply the controlled gate to the target qubit while leaving others unchanged
        for i in range(2 ** register.count):
            if ((i >> (register.count - 1 - control.index)) & 1) == 1:
                # Control qubit is |1>, apply the gate to the target qubit
                j = i ^ (1 << (register.count - 1 - target.index))  # Flip the target qubit
                amp_i = old_register_amps[i]
                amp_j = old_register_amps[j]
                register.amps[i] = cls.matrix[0, 0] * amp_i + cls.matrix[0, 1] * amp_j
                register.amps[j] = cls.matrix[1, 0] * amp_i + cls.matrix[1, 1] * amp_j

        return register

class X_Matrix(MatrixGate):
    matrix = np.array([[0, 1], [1, 0]])

class Y_Matrix(MatrixGate):
    matrix = np.array([[0, -1j], [1j, 0]])

class Z_Matrix(MatrixGate):
    matrix = np.array([[1, 0], [0, -1]])

class H_Matrix(MatrixGate):
    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

class X(X_Matrix, OneQbitGate):
    ...

class Y(Y_Matrix, OneQbitGate):
    ...

class Z(Z_Matrix, OneQbitGate):
    ...

class H(H_Matrix, OneQbitGate):
    ...

class CX(X_Matrix, QControlledGate):
    ...

class CY(Y_Matrix, QControlledGate):
    ...

class CZ(Z_Matrix, QControlledGate):
    ...

class SWAP:
    @classmethod
    def apply(cls, qbit1: QbitRef, qbit2: QbitRef) -> QRegister:
        if qbit1.register is not qbit2.register:
            raise ValueError("Qubits must belong to the same register.")

        register = qbit1.register

        # Apply the SWAP operation using three CNOT gates
        CX.apply(qbit1, qbit2)
        CX.apply(qbit2, qbit1)
        CX.apply(qbit1, qbit2)

        return register


class Measure:
    @classmethod
    def apply(cls, qbit: QbitRef) -> int:
        register = qbit.register
        probabilities = register.probabilities()

        p = np.array(
            [
                sum(
                    probabilities[i]
                    for i in range(len(probabilities))
                    if (i >> (register.count - 1 - qbit.index)) & 1 == 0
                ),
                sum(
                    probabilities[i]
                    for i in range(len(probabilities))
                    if (i >> (register.count - 1 - qbit.index)) & 1 == 1
                ),
            ],
            dtype=float,
        )

        total_probability = p.sum()
        if np.isclose(total_probability, 0):
            raise ValueError("Cannot measure a register with zero total probability.")

        p /= total_probability

        measured_value = np.random.choice([0, 1], p=p)

        projector_0 = np.array([[1, 0], [0, 0]])
        projector_1 = np.array([[0, 0], [0, 1]])

        operations = []
        for i in range(register.count):
            if i == qbit.index:
                operations.append(projector_1 if measured_value else projector_0)
            else:
                operations.append(np.eye(2))

        full_matrix = ft.reduce(np.kron, operations)  # ty:ignore[invalid-argument-type]
        register.amps = (full_matrix @ register.amps) / np.sqrt(p[measured_value])

        return int(measured_value)


def main() -> None:
    q = QRegister(count=3)
    print(f"{f'qbits[{q.count}] q':20} -> q: {q}")

    b = [0, 0]
    print(f"{'bit[2] b':20} -> q: {q}")

    H.apply(q[1])
    print(f"{'h q[1]':20} -> q: {q}")

    CX.apply(q[1], q[2])
    print(f"{'cx q[1] q[2]':20} -> q: {q}")

    CX.apply(q[0], q[1])
    print(f"{'cx q[0] q[1]':20} -> q: {q}")

    H.apply(q[0])
    print(f"{'h q[0]':20} -> q: {q}")

    b[0] = Measure.apply(q[0])
    print(f"{'b[0] = measure q[0]':20} -> q: {q}")

    b[1] = Measure.apply(q[1])
    print(f"{'b[1] = measure q[1]':20} -> q: {q}")

    CZ.apply(q[0], q[2])
    print(f"{'cz q[0] q[2]':20} -> q: {q}")
    CX.apply(q[1], q[2])
    print(f"{'cx q[1] q[2]':20} -> q: {q}")


if __name__ == "__main__":
    main()
