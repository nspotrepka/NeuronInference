import network

dt = 0.001
totalTime = 1
numIterations = int(totalTime/dt)
network = network.Network(30)
network.neurons[0].setBinaryValue(True)
"""network.neurons[3].setBinaryValue(True)
network.neurons[7].setBinaryValue(True)
network.neurons[8].setBinaryValue(True)"""
for t in range(numIterations):
    print(network.getNetworkStateReadable())
    network.update(dt)