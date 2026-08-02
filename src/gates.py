import functools as ft

import numpy as np

from qregister import QbitRef, QRegister


class MatrixGate:
    matrix: np.ndarray

class RMatrixGate:
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the matrix method.")


def _apply_matrix_to_pair(register: QRegister, index_a: int, index_b: int, matrix: np.ndarray) -> None:
    amp_a = register.amps[index_a]
    amp_b = register.amps[index_b]
    register.amps[index_a] = matrix[0, 0] * amp_a + matrix[0, 1] * amp_b
    register.amps[index_b] = matrix[1, 0] * amp_a + matrix[1, 1] * amp_b

class OneQbitGate(MatrixGate):
    @classmethod
    def apply(cls, qbit: QbitRef) -> QRegister:
        register = qbit.register
        shift = register.count - 1 - qbit.index

        for i in range(2 ** register.count):
            if ((i >> shift) & 1) == 0:
                j = i | (1 << shift)
                _apply_matrix_to_pair(register, i, j, cls.matrix)

        return register

class QControlledGate(MatrixGate):
    @classmethod
    def apply(cls, control: QbitRef, target: QbitRef) -> QRegister:
        if control.register is not target.register:
            raise ValueError("Control and target qubits must belong to the same register.")

        register = control.register
        old_register_amps = register.amps.copy()
        control_shift = register.count - 1 - control.index
        target_shift = register.count - 1 - target.index

        for i in range(2 ** register.count):
            if ((i >> control_shift) & 1) == 1:
                j = i ^ (1 << target_shift)
                amp_i = old_register_amps[i]
                amp_j = old_register_amps[j]
                register.amps[i] = cls.matrix[0, 0] * amp_i + cls.matrix[0, 1] * amp_j
                register.amps[j] = cls.matrix[1, 0] * amp_i + cls.matrix[1, 1] * amp_j

        return register

class RotationGate(RMatrixGate):
    @classmethod
    def apply(cls, qbit: QbitRef, theta: float) -> QRegister:
        register = qbit.register
        matrix = cls.matrix(theta)
        shift = register.count - 1 - qbit.index

        for i in range(2 ** register.count):
            if ((i >> shift) & 1) == 0:
                j = i | (1 << shift)
                _apply_matrix_to_pair(register, i, j, matrix)

        return register

class H_Matrix(MatrixGate):
    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

class RX(RotationGate):
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

class RY(RotationGate):
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2), np.cos(theta / 2)]])


class RZ(RotationGate):
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        )

class X_Matrix(MatrixGate):
    matrix = np.array([[0, 1], [1, 0]])

class Y_Matrix(MatrixGate):
    matrix = np.array([[0, -1j], [1j, 0]])

class Z_Matrix(MatrixGate):
    matrix = np.array([[1, 0], [0, -1]])

class H(H_Matrix, OneQbitGate):
    ...

class X(X_Matrix, OneQbitGate):
    ...

class Y(Y_Matrix, OneQbitGate):
    ...

class Z(Z_Matrix, OneQbitGate):
    ...

class CH(H_Matrix, QControlledGate):
    ...

class CX(X_Matrix, QControlledGate):
    ...

class CY(Y_Matrix, QControlledGate):
    ...

class CZ(Z_Matrix, QControlledGate):
    ...


class SWAPGate:
    @classmethod
    def apply(cls, qbit1: QbitRef, qbit2: QbitRef) -> QRegister:
        if qbit1.register is not qbit2.register:
            raise ValueError("Qubits must belong to the same register.")

        register = qbit1.register
        old_register_amps = register.amps.copy()
        shift_1 = register.count - 1 - qbit1.index
        shift_2 = register.count - 1 - qbit2.index

        if shift_1 == shift_2:
            return register

        for i in range(2 ** register.count):
            bit_1 = (i >> shift_1) & 1
            bit_2 = (i >> shift_2) & 1
            swapped = i ^ ((bit_1 ^ bit_2) << shift_1) ^ ((bit_1 ^ bit_2) << shift_2)
            register.amps[swapped] = old_register_amps[i]

        return register

class SWAP:
    @classmethod
    def apply(cls, qbit1: QbitRef, qbit2: QbitRef) -> QRegister:
        return SWAPGate.apply(qbit1, qbit2)


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
