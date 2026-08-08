"""Display utilities for quantum states."""

from __future__ import annotations

import numpy as np
from .qregister import Qbit, QRegister


def display_state(state: QRegister | Qbit) -> None:
    """Display detailed quantum state information.

    Args:
        state: Either a QRegister or a Qbit to display.
    """
    if isinstance(state, Qbit):
        display_qbit(state)
    else:
        display_register(state)


def display_register(register: QRegister) -> None:
    """Display detailed information about a quantum register.

    Args:
        register: The quantum register to display.
    """
    print(f"QRegister ({register.count} qubits):")
    if register.is_pure():
        print(f"  State:   {register}")
    print(f"  Purity:  {np.round(register.purity(), 4)}")
    print(f"  Entropy: {np.round(register.entropy(), 4)} bits")
    print()
    print("  Qubit Bloch vectors:")
    for i, qbit in enumerate(register.qbits):
        bv = np.round(qbit.bloch_vector(), 3)
        state = str(qbit) if qbit.is_pure() else "mixed"
        ent_str = (
            f" entangled with {qbit.entangled_indices}" if qbit.is_entangled() else ""
        )
        print(f"    q[{i}]: {bv} ({state}{ent_str})")


def display_qbit(qbit: Qbit) -> None:
    """Display detailed information about a single qubit.

    Args:
        qbit: The qubit to display.
    """
    bv = qbit.bloch_vector()
    ent_str = (
        f"entangled with {qbit.entangled_indices}"
        if qbit.is_entangled()
        else "not entangled"
    )

    print(f"Qbit {qbit.index} (register size {qbit.register.count}):")
    if qbit.is_pure():
        print(f"  State:          {qbit}")
    print(f"  Pure:           {qbit.is_pure()}")
    print(f"  Entanglement:   {ent_str}")
    print(f"  Bloch vector:   {np.round(bv, 3)}")
    print("  Density matrix:")
    for row in qbit.density:
        print(f"    {row}")
