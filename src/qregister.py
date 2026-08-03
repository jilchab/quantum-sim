from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

_SYMBOLIC_MAGNITUDES: tuple[tuple[float, str], ...] = (
    (1 / 2, "1/2"),
    (1 / np.sqrt(2), "1/√2"),
    (1 / np.sqrt(3), "1/√3"),
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


def _canonicalize_schmidt_pair(
    u: np.ndarray, v: np.ndarray, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    for value in u:
        if not np.isclose(value, 0, atol=tol):
            phase = value / abs(value)
            return u / phase, v * phase

    return u, v


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

    def __repr__(self):
        return format_qbits(self.count, self.amps)

    def dissociate(self) -> list[QbitGroup]:
        groups: list[QbitGroup] = []
        seen: set[tuple[int, ...]] = set()

        def add_group(group: QbitGroup) -> None:
            key = tuple(group.indices)
            if key not in seen:
                seen.add(key)
                groups.append(group)

        def factorize(
            amps: np.ndarray,
            indices: list[int],
            left_positions: list[int],
            right_positions: list[int],
        ) -> tuple[QbitGroup, QbitGroup] | None:
            positions = left_positions + right_positions
            if sorted(positions) != list(range(len(indices))):
                raise ValueError(
                    "left_positions and right_positions must partition the current indices."
                )

            tensor = amps.reshape([2] * len(indices))
            tensor = tensor.transpose(positions)
            matrix = tensor.reshape(2 ** len(left_positions), 2 ** len(right_positions))

            if not np.isclose(np.linalg.matrix_rank(matrix), 1):
                return None

            u, _, vh = np.linalg.svd(matrix, full_matrices=False)
            u0, v0 = _canonicalize_schmidt_pair(u[:, 0], vh[0, :].conj())
            left_indices = [indices[position] for position in left_positions]
            right_indices = [indices[position] for position in right_positions]
            return (
                QbitGroup(register=self, indices=left_indices, amps=u0.copy()),
                QbitGroup(register=self, indices=right_indices, amps=v0.copy()),
            )

        def recurse(amps: np.ndarray, indices: list[int]) -> None:
            if len(indices) <= 1:
                if indices:
                    add_group(
                        QbitGroup(
                            register=self, indices=indices.copy(), amps=amps.copy()
                        )
                    )
                return

            local_positions = list(range(len(indices)))
            for left_size in range(1, len(indices)):
                for left_positions in combinations(local_positions, left_size):
                    right_positions = [
                        position
                        for position in local_positions
                        if position not in left_positions
                    ]
                    result = factorize(
                        amps,
                        indices,
                        list(left_positions),
                        right_positions,
                    )
                    if result is None:
                        continue

                    left_group, right_group = result
                    add_group(left_group)
                    recurse(right_group.amps, right_group.indices)
                    return

            add_group(
                QbitGroup(register=self, indices=indices.copy(), amps=amps.copy())
            )

        recurse(self.amps, list(range(self.count)))
        return sorted(groups, key=lambda group: group.indices[0])


@dataclass(frozen=True)
class QbitRef:
    register: QRegister
    index: int

    def __repr__(self):
        return f"{self.register}[{self.index}]"


@dataclass(frozen=True)
class QbitGroup:
    register: QRegister
    indices: list[int]
    amps: np.ndarray

    def __repr__(self):
        return (
            format_qbits(len(self.indices), self.amps)
            + f" {{{', '.join(map(str, self.indices))}}}"
        )
