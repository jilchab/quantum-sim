from quantum import H, CX, AmplitudeDampingNoise, QRegister, PhaseFlipNoise, display_state

q = QRegister.zeros(2)

print("Initial state:")
display_state(q)

H.apply(q[0])

print("\nAfter applying H to q[0]:")
display_state(q)

PhaseFlipNoise(0.1).apply(q[0])

print("\nAfter applying PhaseFlipNoise to q[0]:")
display_state(q)

CX.apply(q[0], q[1])
print("\nAfter applying CX to q[0] on q[1]:")
display_state(q)

AmplitudeDampingNoise(0.5).apply(q[1])

print("\nAfter applying AmplitudeDampingNoise to q[1]:")
display_state(q)