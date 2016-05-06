from network import Network

dt = 0.001
totalTime = 0.1
numIterations = int(totalTime/dt)
network = Network(8, 4)
for t in range(numIterations):
    print(network.getSpikingIndices())
    network.update(dt)