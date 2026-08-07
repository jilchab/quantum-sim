from __future__ import annotations
import numpy as np

_SYMBOLIC_MAGNITUDES: tuple[tuple[float, str], ...] = (
    (1 / 2, "1/2"),
    (1 / np.sqrt(2), "1/√2"),
    (1 / np.sqrt(3), "1/√3"),
    (2 / np.sqrt(3), "2/√3"),
    (1 / np.sqrt(5), "1/√5"),
    (2 / np.sqrt(5), "2/√5"),
    (3 / np.sqrt(5), "3/√5"),
    (4 / np.sqrt(5), "4/√5"),
    (1 / np.sqrt(6), "1/√6"),
    (2 / np.sqrt(6), "2/√6"),
    (3 / np.sqrt(6), "3/√6"),
    (4 / np.sqrt(6), "4/√6"),
    (5 / np.sqrt(6), "5/√6"),
    (np.sqrt(3) / 2, "√3/2"),
)


def _format_magnitude(value: float) -> str:
    for target, label in _SYMBOLIC_MAGNITUDES:
        if np.isclose(value, target):
            return label

    rounded = str(np.round(value, 3))
    return rounded.rstrip("0").rstrip(".") if "." in rounded else rounded


def _format_real(value: float) -> str:
    if np.isclose(value, 1):
        return "+"
    if np.isclose(value, -1):
        return "-"

    sign = "+" if value >= 0 else "-"
    return f"{sign}{_format_magnitude(abs(value))}"


def _format_imag(value: float) -> str:
    if np.isclose(value, 1):
        return "+j"
    if np.isclose(value, -1):
        return "-j"

    sign = "+" if value >= 0 else "-"
    return f"{sign}{_format_magnitude(abs(value))}j"


def _format_complex(value: complex) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))

    if np.isclose(imag, 0):
        return _format_real(real)
    if np.isclose(real, 0):
        return _format_imag(imag)

    outer_sign = "+" if real >= 0 else "-"
    inner_sign = "+" if (real >= 0) == (imag >= 0) else "-"
    return (
        f"{outer_sign}({_format_magnitude(abs(real))} {inner_sign} "
        f"{_format_magnitude(abs(imag))}j)"
    )


def _format_single_qubit_state(amps: np.ndarray) -> str | None:
    if len(amps) != 2:
        return None

    canonical_states: tuple[tuple[str, np.ndarray], ...] = (
        ("|+❯", np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)),
        ("|-❯", np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128)),
        ("|j❯", np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=np.complex128)),
        (
            "|-j❯",
            np.array([1 / np.sqrt(2), -1j / np.sqrt(2)], dtype=np.complex128),
        ),
    )

    for label, canonical_amps in canonical_states:
        if np.allclose(amps, canonical_amps):
            return label

    return None


def format_qbits(count: int, amps: np.ndarray) -> str:
    if count == 1:
        symbolic_state = _format_single_qubit_state(amps)
        if symbolic_state is not None:
            return symbolic_state

    parts: list[str] = []
    for i in range(2**count):
        if np.isclose(amps[i], 0):
            continue

        bin_str = format(i, f"0{count}b")
        amp = _format_complex(amps[i])
        parts.append(f"{amp}|{bin_str}❯")

    if not parts:
        return ""

    first, *rest = parts
    if first.startswith("+"):
        first = first[1:]

    return " ".join([first, *rest])


def _density_from_state_vector(state_vector: np.ndarray) -> np.ndarray:
    return np.outer(state_vector, state_vector.conj())


class QRegister:
    def __init__(self, count: int, density: np.ndarray | None = None) -> None:
        self.count = count
        if density is None:
            self.density = np.zeros([2**count, 2**count], dtype=np.complex128)
            self.density[0, 0] = 1
        else:
            if density.shape != (2**count, 2**count):
                raise ValueError(
                    f"Shape of density ({density.shape}) does not match (2**count, 2**count) ({(2**count, 2**count)})."
                )
            self.density = density.astype(np.complex128)

    def probabilities(self):
        return np.diag(self.density)

    def state_vector(self) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(self.density)

        # # Largest eigenvalue
        i = np.argmax(eigenvalues)
        eigenvalue = eigenvalues[i]

        # A density matrix represents a pure state
        # only if its largest eigenvalue is ~1.
        # if not np.isclose(eigenvalue, 1.0):
        #      raise ValueError("Mixed state has no unique state vector")

        # state_vector = eigenvectors[:, i].copy()

        # # Remove arbitrary global phase
        # for value in state_vector:
        #     if not np.isclose(value, 0):
        #         state_vector /= value / abs(value)
        #         break

        # return np.real_if_close(state_vector)

        if self.is_pure() is False:
            raise ValueError("Mixed state has no unique state vector")
        return self.probabilities() ** 0.5 * np.exp(1j * np.angle(eigenvectors[:, i]))

    def ket(self) -> np.ndarray:
        return self.state_vector().reshape((-1, 1))

    def bra(self) -> np.ndarray:
        return self.state_vector().conj().T

    def is_pure(self) -> bool:
        purity = np.trace(self.density @ self.density)
        return np.isclose(purity, 1.0)

    def bloch_vector(self) -> np.ndarray:
        vectors = np.array([self[i].bloch_vector() for i in range(self.count)])
        return vectors.reshape((-1, 3))

    def add_depolarizing_noise(self, purity: float) -> QRegister:
        new_density = purity * self.density + (1 - purity) * np.eye(2**self.count, dtype=np.complex128) / 2**self.count
        return QRegister(self.count, new_density)

    def __getitem__(self, index: int) -> Qbit:
        if index < 0 or index >= self.count:
            raise IndexError("QRegister index out of range.")
        return Qbit(register=self, index=index)

    def __repr__(self):
        if self.is_pure():
            return f"PureQRegister<{format_qbits(self.count, self.state_vector())}>"
        else:
            return f"ImpureQRegister<{self.bloch_vector()}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, QRegister):
            return self.count == other.count and np.allclose(self.density, other.density)
        raise NotImplementedError(
            "Equality comparison is only implemented for QRegister instances."
        )

    def __add__(self, other: QRegister) -> QRegister:
        if not isinstance(other, QRegister):
            raise NotImplementedError(
                "Addition is only implemented for QRegister instances."
            )

        new_count = self.count + other.count
        new_density = np.kron(self.density, other.density)
        return QRegister(new_count, new_density)


    @staticmethod
    def zeros(count: int) -> QRegister:
        return QRegister(count)

    @staticmethod
    def ones(count: int) -> QRegister:
        density = np.zeros([2**count, 2**count], dtype=np.complex128)
        density[-1, -1] = 1
        return QRegister(count, density)

    @staticmethod
    def bell() -> QRegister:
        return QRegister.ghz(2)

    @staticmethod
    def ghz(count: int) -> QRegister:
        if count < 2:
            raise ValueError("GHZ state requires at least 2 qubits.")
        state_vector = np.zeros(2**count, dtype=np.complex128)
        state_vector[0] = 1 / np.sqrt(2)
        state_vector[-1] = 1 / np.sqrt(2)
        return QRegister(count, _density_from_state_vector(state_vector))

    @staticmethod
    def w_state(count: int) -> QRegister:
        if count < 2:
            raise ValueError("W state requires at least 2 qubits.")
        state_vector = np.zeros(2**count, dtype=np.complex128)
        for index in range(2**count):
            if bin(index).count("1") == 1:
                state_vector[index] = 1 / np.sqrt(count)
        return QRegister(count, _density_from_state_vector(state_vector))

    @staticmethod
    def werner(purity: float) -> QRegister:
        return QRegister.bell().add_depolarizing_noise(purity)

class Qbit(QRegister):
    def __init__(self, register: QRegister, index: int, entangled_with: list[int] | None = None) -> None:
        if index < 0 or index >= register.count:
            raise IndexError("QbitRef index out of range.")
        self.register = register
        self.index = index
        self.entangled_with = entangled_with if entangled_with is not None else []
        self.count = 1

    @property
    def density(self) -> np.ndarray:
        tensor = self.register.density.reshape([2] * (2 * self.register.count))
        other_indices = [i for i in range(self.register.count) if i != self.index]
        permutation = [self.index, *other_indices, self.index + self.register.count]
        permutation.extend(i + self.register.count for i in other_indices)
        tensor = np.transpose(tensor, permutation)
        tensor = tensor.reshape(2, 2 ** (self.register.count - 1), 2, 2 ** (self.register.count - 1))
        return np.einsum("aibi->ab", tensor)

    def bloch_vector(self) -> np.ndarray:
        x = 2 * np.real(self.density[0, 1])
        y = 2 * np.imag(self.density[0, 1])
        z = np.real(self.density[0, 0] - self.density[1, 1])
        return np.array([x, y, z])

    def is_entangled(self) -> bool:
        return len(self.entangled_with) > 0

    def __repr__(self):
        if self.is_pure():
            return f"PureQbit<{format_qbits(self.count, self.state_vector())}>"
        else:
            return f"EntangledQbit<{self.bloch_vector()}>"