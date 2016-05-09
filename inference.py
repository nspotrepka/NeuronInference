import math
import numpy as np
import matplotlib.pyplot as plt

from network import Network

def normalize(v):
    s = 0
    for x in v:
        if math.fabs(x) > s:
            s = math.fabs(x)
    if s == 0:
        return v
    else:
        return v/s

dt = 0.0001
totalTime = 0.3
numIterations = int(totalTime/dt)
m = 32
n = 256
network = Network(m, n)
voltage = np.zeros((n, numIterations))
for t in range(numIterations):
    for i in range(n):
        voltage[i, t] = network.S[i].voltage
    network.update(dt)


#plt.plot(voltage[0], 'red')
#plt.plot(voltage[1], 'green')
#plt.plot(voltage[2], 'blue')
#plt.show()

plt.plot(normalize(network.X))
S = np.zeros(n)
for i in range(n):
    S[i] = network.S[i].spikeRate

print("Neuron firing rates:")
print(S)
X2 = np.dot(network.A, S)
plt.plot(normalize(X2))
plt.show()