
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

import numpy as np

from gates import CH, CX, CY, CZ, RX, RY, RZ, SWAP, H, Measure, X, Y, Z
from qregister import QRegister

qbits_pattern = re.compile(r"^\s*(?:qbits|qubits)\s*\[\s*(\d+)\s*\]\s+(\w+)\s*$")
bits_pattern = re.compile(r"^\s*bits\s*\[\s*(\d+)\s*\]\s+(\w+)\s*$")
one_qbit_gate_pattern = re.compile(r"^\s*(h|x|y|z)\s+(\w+)\[(\d+)\]\s*$")
control_gate_pattern = re.compile(
    r"^\s*(ch|cx|cy|cz)\s+(\w+)\[(\d+)\]\s+(\w+)\[(\d+)\]\s*$",
)
measure_pattern = re.compile(r"^\s*(\w+)\[(\d+)\]\s*=\s*measure\s+(\w+)\[(\d+)\]\s*$")
rotation_gate_pattern = re.compile(
    r"^\s*r(x|y|z)\s*\(\s*([^\)]+)\s*\)\s+(\w+)\[(\d+)\]\s*$",
)
swap_pattern = re.compile(r"^\s*swap\s+(\w+)\[(\d+)\]\s+(\w+)\[(\d+)\]\s*$")

_ONE_QBIT_GATES = {
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
}

_CONTROL_GATES = {
    "ch": CH,
    "cx": CX,
    "cy": CY,
    "cz": CZ,
}

_ROTATION_GATES = {
    "x": RX,
    "y": RY,
    "z": RZ,
}

_ALLOWED_NAMES = {
    "e": np.e,
    "pi": np.pi,
    "tau": np.pi * 2,
}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Div: operator.truediv,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.Sub: operator.sub,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_scalar_expression(expression: str) -> float:
    def _evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
            return float(_ALLOWED_NAMES[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                _evaluate(node.left),
                _evaluate(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
        raise ValueError(f"Unsupported expression '{expression}'.")

    parsed = ast.parse(expression, mode="eval")
    return _evaluate(parsed)

@dataclass(slots=True)
class State:
    qbits_dict: dict[str, QRegister] = field(default_factory=dict)
    bits_dict: dict[str, list[int]] = field(default_factory=dict)

def parse(input_str: str, state: State) -> State:
    lines = input_str.strip().splitlines()

    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("#"):
            continue

        qbits = qbits_pattern.match(line)
        if qbits:
            count = int(qbits.group(1))
            name = qbits.group(2)
            state.qbits_dict[name] = QRegister(count)
            continue

        bits = bits_pattern.match(line)
        if bits:
            count = int(bits.group(1))
            name = bits.group(2)
            state.bits_dict[name] = [0] * count
            continue

        one_qbit_gate = one_qbit_gate_pattern.match(line)
        if one_qbit_gate:
            gate_name = one_qbit_gate.group(1)
            qbit = one_qbit_gate.group(2)
            qbit_index = int(one_qbit_gate.group(3))

            if qbit not in state.qbits_dict:
                raise ValueError(f"Qbits '{qbit}' not found.")

            qregister = state.qbits_dict[qbit]
            if qbit_index < 0 or qbit_index >= qregister.count:
                raise IndexError(f"QRegister '{qbit}' index out of range.")

            _ONE_QBIT_GATES[gate_name].apply(qregister[qbit_index])
            continue

        control_gate = control_gate_pattern.match(line)
        if control_gate:
            gate_name = control_gate.group(1)
            control_qbit = control_gate.group(2)
            control_index = int(control_gate.group(3))
            target_qbit = control_gate.group(4)
            target_index = int(control_gate.group(5))

            if control_qbit not in state.qbits_dict:
                raise ValueError(f"Control Qbits '{control_qbit}' not found.")
            if target_qbit not in state.qbits_dict:
                raise ValueError(f"Target Qbits '{target_qbit}' not found.")

            control_register = state.qbits_dict[control_qbit]
            target_register = state.qbits_dict[target_qbit]

            if control_index < 0 or control_index >= control_register.count:
                raise IndexError(f"Control QRegister '{control_qbit}' index out of range.")
            if target_index < 0 or target_index >= target_register.count:
                raise IndexError(f"Target QRegister '{target_qbit}' index out of range.")

            control_ref = control_register[control_index]
            target_ref = target_register[target_index]

            _CONTROL_GATES[gate_name].apply(control_ref, target_ref)
            continue

        measure = measure_pattern.match(line)
        if measure:
            bit = measure.group(1)
            bit_index = int(measure.group(2))
            qbit = measure.group(3)
            qbit_index = int(measure.group(4))
            if bit not in state.bits_dict:
                raise ValueError(f"Bits '{bit}' not found.")
            if qbit not in state.qbits_dict:
                raise ValueError(f"Qbits '{qbit}' not found.")

            bits_list = state.bits_dict[bit]
            qregister = state.qbits_dict[qbit]

            if bit_index < 0 or bit_index >= len(bits_list):
                raise IndexError(f"Bits '{bit}' index out of range.")
            if qbit_index < 0 or qbit_index >= qregister.count:
                raise IndexError(f"QRegister '  {qbit}' index out of range.")

            qbit_ref = qregister[qbit_index]
            measured_value = Measure.apply(qbit_ref)
            bits_list[bit_index] = measured_value

            continue

        swap_gate = swap_pattern.match(line)
        if swap_gate:
            left_qbit = swap_gate.group(1)
            left_index = int(swap_gate.group(2))
            right_qbit = swap_gate.group(3)
            right_index = int(swap_gate.group(4))

            if left_qbit not in state.qbits_dict:
                raise ValueError(f"Qbits '{left_qbit}' not found.")
            if right_qbit not in state.qbits_dict:
                raise ValueError(f"Qbits '{right_qbit}' not found.")

            left_register = state.qbits_dict[left_qbit]
            right_register = state.qbits_dict[right_qbit]

            if left_index < 0 or left_index >= left_register.count:
                raise IndexError(f"QRegister '{left_qbit}' index out of range.")
            if right_index < 0 or right_index >= right_register.count:
                raise IndexError(f"QRegister '{right_qbit}' index out of range.")

            SWAP.apply(left_register[left_index], right_register[right_index])
            continue

        rotation_gate = rotation_gate_pattern.match(line)
        if rotation_gate:
            gate = rotation_gate.group(1)
            theta_str = rotation_gate.group(2)
            qbit = rotation_gate.group(3)
            qbit_index = int(rotation_gate.group(4))

            if qbit not in state.qbits_dict:
                raise ValueError(f"Qbits '{qbit}' not found.")

            qregister = state.qbits_dict[qbit]
            if qbit_index < 0 or qbit_index >= qregister.count:
                raise IndexError(f"QRegister '{qbit}' index out of range.")

            qbit_ref = qregister[qbit_index]

            theta = _evaluate_scalar_expression(theta_str)
            _ROTATION_GATES[gate].apply(qbit_ref, theta)

            continue

        raise ValueError(f"Could not parse line: '{line}'.")

    return state