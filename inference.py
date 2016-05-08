import numpy as np
import matplotlib.pyplot as plt

from network import Network

dt = 0.001
totalTime = 0.1
numIterations = int(totalTime/dt)
m = 64
n = 8
network = Network(m, n)
voltage = np.zeros((n, numIterations))
for t in range(numIterations):
    for i in range(n):
        voltage[i, t] = network.S[i].voltage
    #print(voltage[i, t])
    print(network.getSpikingIndices())
    network.update(dt)
for i in range(n):
    plt.plot(voltage[i], ['#FF0000','#FF8800','#FFFF00','#00FF00','#00FFFF','#0088FF','#8800FF','#FF00FF'][i])
    plt.show()