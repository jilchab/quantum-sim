from __future__ import annotations

from dataclasses import dataclass

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
        parts: list[str] = []
        for i in range(2 ** self.count):
            if np.isclose(self.amps[i], 0):
                continue

            bin_str = format(i, f"0{self.count}b")
            amp = _format_complex(self.amps[i])
            parts.append(f"{amp}|{bin_str}>")

        if not parts:
            return ""

        first, *rest = parts
        if first.startswith("+"):
            first = first[1:]

        return " ".join([first, *rest])


@dataclass(frozen=True)
class QbitRef:
    register: QRegister
    index: int