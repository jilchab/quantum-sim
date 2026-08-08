# Quantum Simulator

A density-matrix based quantum circuit simulator implemented in Python with NumPy. Features full support for quantum gates, entanglement detection via Kronecker factorization, and realistic quantum noise models for quantum error correction (QEC) applications.

## Features

- **Density Matrix Representation**: Supports both pure and mixed quantum states
- **Entanglement Detection**: Automatic detection via Kronecker product factorization
- **Quantum Gates**: Single-qubit, rotation, and multi-qubit controlled gates
- **Realistic Noise Models**: Depolarizing, bit-flip, phase-flip, and amplitude damping channels
- **State Visualization**: Rich display of quantum states with purity, entropy, and Bloch vectors
- **Qubit Access**: Direct access to individual qubits with reduced density matrices


## Quick Start

### Basic Circuit Example

```python
from quantum import QRegister, H, CX, display_state

# Create a 2-qubit register in |00⟩ state
q = QRegister.zeros(2)

# Apply Hadamard to first qubit
H.apply(q[0])

# Apply CNOT
CX.apply(q[0], q[1])

# Display the result
display_state(q)
```

Output:
```
QRegister (2 qubits):
  State:   1/√2|00❯ +1/√2|11❯
  Purity:  1.0
  Entropy: 0.0 bits

  Qubit Bloch vectors:
    q[0]: [1. 0. 0.] (|+❯ entangled with [1])
    q[1]: [0. 0. 1.] (|0❯ entangled with [0])
```

### Noisy Quantum Circuit

```python
from quantum import QRegister, H, CX, PhaseFlipNoise, AmplitudeDampingNoise, display_state

# Create a 2-qbits entangled system (Bell state)
q = QRegister.bell()

# Apply phase flip noise (10% error rate)
PhaseFlipNoise(0.1).apply(q[0])

# Apply amplitude damping (50% energy loss)
AmplitudeDampingNoise(0.5).apply(q[1])
```

Run the full example:
```bash
uv run python examples/simple.py
```

## Quantum State Generator Functions

| Function | Description |
|----------|-------------|
| `QRegister.zeros(n)` | n-qubit \|00...0⟩ state |
| `QRegister.ones(n)` | n-qubit \|11...1⟩ state |
| `QRegister.ghz(n)` | GHZ state: (1/√2)(\|00...0⟩ + \|11...1⟩) |
| `QRegister.bell()` | Bell state: (1/√2)(\|00⟩ + \|11⟩) |
| `QRegister.w_state(n)` | W state: equal superposition of single-excitation states |
| `QRegister.werner(purity)` | Werner state from singlet with mixed component |

## Quantum Gates

### Single-Qubit Gates

| Gate | Description | Usage |
|------|-------------|-------|
| `H` | Hadamard | `H.apply(q[0])` |
| `X` | Pauli-X (bit flip) | `X.apply(q[0])` |
| `Y` | Pauli-Y | `Y.apply(q[0])` |
| `Z` | Pauli-Z (phase flip) | `Z.apply(q[0])` |
| `RX(θ)` | X-rotation | `RX.apply(q[0], np.pi/4)` |
| `RY(θ)` | Y-rotation | `RY.apply(q[0], np.pi/4)` |
| `RZ(θ)` | Z-rotation | `RZ.apply(q[0], np.pi/4)` |

### Multi-Qubit Gates

| Gate | Description | Usage |
|------|-------------|-------|
| `CX` | CNOT (Controlled-X) | `CX.apply(q[0], q[1])` |
| `CY` | Controlled-Y | `CY.apply(q[0], q[1])` |
| `CZ` | Controlled-Z | `CZ.apply(q[0], q[1])` |
| `CH` | Controlled-Hadamard | `CH.apply(q[0], q[1])` |
| `SWAP` | Swap two qubits | `SWAP.apply(q[0], q[1])` |

## Noise Models

All noise models inherit from `KlausNoise` and use the Kraus operator formalism: $\rho' = \sum_{i=0}^n E_i ρ {E_i}^\dagger$

| Noise Type | Kraus Operators | Usage |
|-----------|-----------------|-------|
| **Depolarizing** | $E_0 = \sqrt{p} I$, $E_i = \sqrt{(1-p)/3} \sigma_i$ for $i \in \{X,Y,Z\}$ | `DepolarizingNoise(purity=0.9).apply(q[0])` |
| **Bit Flip** | $E_0 = \sqrt{1-p} I$, $E_1 = \sqrt{p} X$ | `BitFlipNoise(flip_prob=0.2).apply(q[0])` |
| **Phase Flip** | $E_0 = \sqrt{1-p} I$, $E_1 = \sqrt{p} Z$ | `PhaseFlipNoise(flip_prob=0.15).apply(q[0])` |
| **Amplitude Damping** | $E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$ | `AmplitudeDampingNoise(gamma=0.3).apply(q[0])` |

## Display Functions

### Display Full Register State

```python
from quantum import display_state

display_state(q)
```

Shows:
- Quantum state (if pure)
- Purity and Von Neumann entropy
- Bloch vector for each qubit
- Entanglement information

### Display Single Qubit

```python
display_state(q[0])
```

Shows:
- Reduced density matrix
- Bloch vector
- Entanglement status
- Pure/mixed classification

## State Properties

```python
# Check if state is pure
if q.is_pure():
    print("State is pure")
    print("State vector:", q.state_vector())
else:
    print("State is mixed")

# Measure decoherence
print(f"Purity: {q.purity()}")  # 1.0 for pure, 0 for maximally mixed
print(f"Entropy: {q.entropy()}")  # bits of entropy

# Access individual qubits
qbit = q[0]
print(f"Reduced density matrix:\n{qbit.density}")
print(f"Bloch vector: {qbit.bloch_vector()}")
print(f"Is entangled: {qbit.is_entangled()}")
print(f"Entangled with qubits: {qbit.entangled_indices}")
```

## Entanglement Detection

The simulator automatically detects entangled subsystems via Kronecker factorization of the density matrix. Each qubit tracks which other qubits it is entangled with:

```python
q = QRegister.bell()
print(q[0].entangled_indices)  # [1]
print(q[1].entangled_indices)  # [0]

# After applying noise, if subsystems become factorizable again:
BitFlipNoise(1.0).apply(q[0])  # Completely flip q[0]
print(q[0].entangled_indices)  # May change based on factorization
```

## Implementation Notes

- **Density Matrix Representation**: All states stored as 2^n × 2^n Hermitian matrices
- **Optimized Tensor Contractions**: Gates applied via tensor contractions, not full matrix multiplication
- **Dissociation Algorithm**: Entanglement detected by factorizing ρ into Kronecker products
- **Per-Qubit Noise**: All noise models apply independently to each qubit (local noise), preserving product structure

## References

- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Kraus operators: https://en.wikipedia.org/wiki/Quantum_channel#Kraus_representation
- Density matrices: https://en.wikipedia.org/wiki/Density_matrix
