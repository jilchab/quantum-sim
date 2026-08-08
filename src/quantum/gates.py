import numpy as np

from .qregister import Qbit, QRegister

type KlausNoiseTransformations = list[tuple[float, np.ndarray]]


class MatrixGate:
    matrix: np.ndarray


class RMatrixGate:
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the matrix method.")


def _full_operator(
    register: QRegister, qbit_index: int, matrix: np.ndarray
) -> np.ndarray:
    operations = [np.eye(2, dtype=np.complex128) for _ in range(register.count)]
    operations[qbit_index] = matrix.astype(np.complex128)
    full_operator = operations[0]
    for operation in operations[1:]:
        full_operator = np.kron(full_operator, operation)

    return full_operator


def _apply_unitary(register: QRegister, unitary: np.ndarray) -> None:
    register.density = unitary @ register.density @ unitary.conj().T


def _controlled_unitary(
    register: QRegister,
    control_index: int,
    target_index: int,
    matrix: np.ndarray,
) -> np.ndarray:
    operations_0 = [np.eye(2, dtype=np.complex128) for _ in range(register.count)]
    operations_0[control_index] = np.array([[1, 0], [0, 0]], dtype=np.complex128)

    operations_1 = [np.eye(2, dtype=np.complex128) for _ in range(register.count)]
    operations_1[control_index] = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    operations_1[target_index] = matrix.astype(np.complex128)

    full_operator_0 = operations_0[0]
    for operation in operations_0[1:]:
        full_operator_0 = np.kron(full_operator_0, operation)

    full_operator_1 = operations_1[0]
    for operation in operations_1[1:]:
        full_operator_1 = np.kron(full_operator_1, operation)

    return full_operator_0 + full_operator_1


def apply_single_qubit_gate(rho, gate, qubit, n_qubits):
    """
    Apply a 2x2 gate to one qubit of an n-qubit density matrix.

    rho:      (2**n, 2**n) complex density matrix
    gate:     (2, 2) complex unitary
    qubit:    qubit index, 0 = first/leftmost qubit
    n_qubits: number of qubits
    """

    dim = 2**n_qubits

    assert rho.shape == (dim, dim)
    assert gate.shape == (2, 2)

    # ---------------------------------------------------------
    # 1. Convert matrix into a 2n-dimensional tensor
    #
    # rho_tensor[i0, i1, ..., iN, j0, j1, ..., jN]
    # ---------------------------------------------------------
    rho_tensor = rho.reshape([2] * (2 * n_qubits))

    # ---------------------------------------------------------
    # 2. Apply G to the row index of the target qubit
    #
    # New tensor index:
    #   i_k
    #
    # old tensor index:
    #   a
    #
    # rho'[..., i_k, ...] = sum_a G[i_k, a] rho[..., a, ...]
    # ---------------------------------------------------------

    rho_tensor = np.tensordot(gate, rho_tensor, axes=([1], [qubit]))

    # tensordot moved the new gate axis to the front.
    # Move it back to the original qubit position.
    rho_tensor = np.moveaxis(rho_tensor, 0, qubit)

    # ---------------------------------------------------------
    # 3. Apply G† to the column index
    #
    # Column index corresponding to qubit k is n_qubits + k.
    # ---------------------------------------------------------

    gate_dagger = gate.conj().T

    rho_tensor = np.tensordot(rho_tensor, gate_dagger, axes=([n_qubits + qubit], [0]))

    # The new column axis was appended at the end.
    # Move it back to n_qubits + qubit.
    rho_tensor = np.moveaxis(rho_tensor, -1, n_qubits + qubit)

    # ---------------------------------------------------------
    # 4. Convert tensor back into the normal density matrix
    # ---------------------------------------------------------

    return rho_tensor.reshape(dim, dim)


class OneQbitGate(MatrixGate):
    @classmethod
    def apply(cls, qbit: Qbit) -> QRegister:
        register = qbit.register

        new_density = apply_single_qubit_gate(
            register.density, cls.matrix, qbit.index, register.count
        )
        register.density = new_density

        return register


def apply_controlled_gate(
    rho: np.ndarray,
    gate: np.ndarray,
    controls: list[int],
    target: int,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    gate = np.asarray(gate, dtype=np.complex128)

    dim = rho.shape[0]
    n = dim.bit_length() - 1

    if 1 << n != dim:
        raise ValueError("rho dimension must be a power of 2")

    if gate.shape != (2, 2):
        raise ValueError("gate must be 2x2")

    if target in controls:
        raise ValueError("target cannot be a control")

    # ---------------------------------------------------------
    # Bit masks
    # q0 = MSB
    # ---------------------------------------------------------

    target_mask = 1 << (n - 1 - target)

    control_mask = 0

    for c in controls:
        control_mask |= 1 << (n - 1 - c)

    indices = np.arange(dim)

    # ---------------------------------------------------------
    # Active states:
    #
    # ALL control bits = 1
    # ---------------------------------------------------------

    active = indices[(indices & control_mask) == control_mask]

    # ---------------------------------------------------------
    # For every active state, find its partner with the
    # target bit flipped.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Only keep one member of each pair.
    #
    # target bit = 0
    # ---------------------------------------------------------

    active0 = active[(active & target_mask) == 0]

    active1 = active0 ^ target_mask

    # ---------------------------------------------------------
    # Result starts as unchanged rho.
    # ---------------------------------------------------------

    result = rho.copy()

    # =========================================================
    # ACTIVE / ACTIVE
    #
    # For every pair of target states:
    #
    #     [a0]
    #     [a1]
    #
    # G transforms the target dimension.
    #
    # =========================================================

    rows = np.concatenate([active0, active1])

    block = rho[np.ix_(rows, rows)]

    m = len(active0)

    # The ordering is now:
    #
    # [all target=0 states,
    #  all target=1 states]
    #
    # therefore G ⊗ I
    #

    big_gate = np.kron(gate, np.eye(m, dtype=np.complex128))

    block_new = big_gate @ block @ big_gate.conj().T

    result[np.ix_(rows, rows)] = block_new

    # =========================================================
    # ACTIVE / INACTIVE
    #
    # The gate acts only on the active side.
    # =========================================================

    inactive = indices[(indices & control_mask) != control_mask]

    block = rho[np.ix_(rows, inactive)]

    block_new = big_gate @ block

    result[np.ix_(rows, inactive)] = block_new

    # =========================================================
    # INACTIVE / ACTIVE
    # =========================================================

    block = rho[np.ix_(inactive, rows)]

    block_new = block @ big_gate.conj().T

    result[np.ix_(inactive, rows)] = block_new

    return result


class QControlledGate(MatrixGate):
    @classmethod
    def apply(cls, controls: list[Qbit] | Qbit, target: Qbit) -> QRegister:
        if isinstance(controls, Qbit):
            controls = [controls]

        if any(c.register is not target.register for c in controls):
            raise ValueError(
                "Control and target qubits must belong to the same register."
            )

        # register = target.register
        # unitary = _controlled_unitary(register, control.index, target.index, cls.matrix)
        # _apply_unitary(register, unitary)

        register = target.register
        new_density = apply_controlled_gate(
            register.density, cls.matrix, [c.index for c in controls], target.index
        )
        register.density = new_density

        return register


class KlausNoise:
    """
    General Kraus operator noise channel.
    Transformations should be a list of tuples (p, K) where p is the probability and K is the 2x2 matrix to apply.
    Example for a depolarizing channel with probability p:
    transformations = [
        (1 - p, I.matrix),
        (p / 3, X.matrix),
        (p / 3, Y.matrix),
        (p / 3, Z.matrix),
    ]
    """

    def __init__(self, transformations: KlausNoiseTransformations):
        self.transformations = transformations

    def apply(self, qbit: Qbit, **kwargs) -> QRegister:
        register = qbit.register
        n_qubits = register.count
        new_density = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)

        for p, matrix in self.transformations:
            kraus_op = np.sqrt(p) * matrix
            new_density += apply_single_qubit_gate(
                register.density, kraus_op, qbit.index, n_qubits
            )

        register.density = new_density
        return register


class DepolarizingNoise(KlausNoise):
    """Single-qubit depolarizing channel: p·ρ + (1-p)/3·(X·ρ·X† + Y·ρ·Y† + Z·ρ·Z†)."""

    def __init__(self, probability: float):
        if not (0 <= probability <= 1):
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability

        self.transformations: KlausNoiseTransformations = [
            (1 - probability, np.eye(2, dtype=np.complex128)),
            (probability / 3, X.matrix),
            (probability / 3, Y.matrix),
            (probability / 3, Z.matrix),
        ]


class BitFlipNoise(KlausNoise):
    """Bit flip channel: (1-p)·ρ + p·X·ρ·X†."""

    def __init__(self, flip_probability: float):
        if not (0 <= flip_probability <= 1):
            raise ValueError("flip_probability must be in [0, 1]")
        self.flip_probability = flip_probability

        self.transformations: KlausNoiseTransformations = [
            (1 - flip_probability, np.eye(2, dtype=np.complex128)),
            (flip_probability, X.matrix),
        ]


class PhaseFlipNoise(KlausNoise):
    """Phase flip channel: (1-p)·ρ + p·Z·ρ·Z†."""

    def __init__(self, flip_probability: float):
        if not (0.0 <= flip_probability <= 1.0):
            raise ValueError("flip_probability must be in [0, 1]")
        self.flip_probability = flip_probability

        self.transformations: KlausNoiseTransformations = [
            (1 - flip_probability, np.eye(2, dtype=np.complex128)),
            (flip_probability, Z.matrix),
        ]


class AmplitudeDampingNoise(KlausNoise):
    """Amplitude damping channel: models energy loss. K_0 = [[1, 0], [0, sqrt(1-γ)]], K_1 = [[0, sqrt(γ)], [0, 0]]."""

    def __init__(self, gamma: float):
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = gamma

        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
        self.transformations: KlausNoiseTransformations = [(1, K0), (1, K1)]


class RotationGate(RMatrixGate):
    @classmethod
    def apply(cls, qbit: Qbit, theta: float) -> QRegister:
        register = qbit.register
        unitary = _full_operator(register, qbit.index, cls.matrix(theta))
        _apply_unitary(register, unitary)

        return register


class H_Matrix(MatrixGate):
    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class I_Matrix(MatrixGate):
    matrix = np.eye(2, dtype=np.complex128)


class RX(RotationGate):
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )


class RY(RotationGate):
    @classmethod
    def matrix(cls, theta: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )


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


class H(H_Matrix, OneQbitGate): ...


class X(X_Matrix, OneQbitGate): ...


class Y(Y_Matrix, OneQbitGate): ...


class Z(Z_Matrix, OneQbitGate): ...


class CH(H_Matrix, QControlledGate): ...


class CX(X_Matrix, QControlledGate): ...


class CY(Y_Matrix, QControlledGate): ...


class CZ(Z_Matrix, QControlledGate): ...


class SWAPGate:
    @classmethod
    def apply(cls, qbit1: Qbit, qbit2: Qbit) -> QRegister:
        if qbit1.register is not qbit2.register:
            raise ValueError("Qubits must belong to the same register.")

        CX.apply(qbit1, qbit2)
        CX.apply(qbit2, qbit1)
        CX.apply(qbit1, qbit2)

        return qbit1.register


class SWAP:
    @classmethod
    def apply(cls, qbit1: Qbit, qbit2: Qbit) -> QRegister:
        return SWAPGate.apply(qbit1, qbit2)


class Measure:
    @classmethod
    def apply(cls, qbit: Qbit) -> int:
        register = qbit.register
        reduced_density = qbit.density
        p = np.array(
            [np.real(reduced_density[0, 0]), np.real(reduced_density[1, 1])],
            dtype=float,
        )

        total_probability = p.sum()
        if np.isclose(total_probability, 0):
            raise ValueError("Cannot measure a register with zero total probability.")

        p /= total_probability

        measured_value = np.random.choice([0, 1], p=p)

        projector = np.array(
            [[1, 0], [0, 0]] if measured_value == 0 else [[0, 0], [0, 1]],
            dtype=np.complex128,
        )
        full_projector = _full_operator(register, qbit.index, projector)
        register.density = (
            full_projector @ register.density @ full_projector.conj().T
        ) / p[measured_value]

        return int(measured_value)
