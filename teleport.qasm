qbits[3] q
bits[2] b

h q[1]
cx q[1] q[2]
cx q[0] q[1]
h q[0]
b[0] = measure q[0]
b[1] = measure q[1]
cz q[0] q[2]
cx q[1] q[2]
