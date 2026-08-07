## Quantum Simulator

This repository contains a small quantum circuit simulator implemented with NumPy.

### Features

- Register-based state vector simulator
- Single-qubit gates: `X`, `Y`, `Z`, `H`
- Controlled gates: `CX`, `CY`, `CZ`
- `SWAP` built from controlled gates
- Projective measurement with state collapse

### Run

```bash
uv run quantum
```

To run a specific circuit file, pass its path:

```bash
uv run quantum teleport.qasm
```

### Notes

- Qubit indices are ordered from left to right in the printed basis states.
- The implementation uses a state vector and collapses the register on measurement.
