import network

dt = 0.001
totalTime = 1
numIterations = int(totalTime/dt)
network = network.Network(100)
for t in range(numIterations):
    network.update(dt)