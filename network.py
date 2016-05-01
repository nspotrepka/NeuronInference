import math
import random as random
import numpy as np

import neuron

class Network(object):
    def __init__(self, numNeurons):
        self.numNeurons = numNeurons

        self.neurons = [neuron.Neuron(0, 10, -10, 0.02, 0.01) for i in range(numNeurons)]
        self.weights = np.random.randn(numNeurons, numNeurons)

    def update(self, dt):
        binaryValues = np.array([n.getBinaryValue() for n in self.neurons])
        dv = np.dot(self.weights, binaryValues.transpose())
        for i in range(self.numNeurons):
            self.neurons[i].update(dv[i], dt)

dt = 0.001
totalTime = 1
numIterations = int(totalTime/dt)
network = Network(100)
for t in range(numIterations):
    network.update(dt)