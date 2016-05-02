from network import Network

dt = 0.001
totalTime = 0.01
numIterations = int(totalTime/dt)
network = Network(10, 100)
for t in range(numIterations):
    print(network.getSpikingIndices())
    network.update(dt)