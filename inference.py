import numpy as np
import matplotlib.pyplot as plt

from network import Network

dt = 0.001
totalTime = 0.1
numIterations = int(totalTime/dt)
network = Network(64, 8)
voltage = np.zeros(numIterations)
for t in range(numIterations):
    voltage[t] = network.S[7].voltage
    #print(network.getSpikingIndices())
    network.update(dt)
plt.plot(voltage)
plt.show()