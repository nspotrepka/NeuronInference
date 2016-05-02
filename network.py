import numpy as np

import neuron

class Network(object):
    def __init__(self, numNeurons):
        self.numNeurons = numNeurons

        self.neurons = [neuron.Neuron(0, 10, -10, 0.02, 0.01) for i in range(numNeurons)]
        self.weights = np.random.randn(numNeurons, numNeurons)
        for i in range(numNeurons):
            self.weights[i,i] = 0

    def update(self, dt):
        binaryValues = np.array([n.getBinaryValue() for n in self.neurons])
        dv = np.dot(self.weights, binaryValues)
        for i in range(self.numNeurons):
            self.neurons[i].update(dv[i], dt)

    def getNetworkStateReadable(self):
        x = ""
        counter = 0;
        for i in range(self.numNeurons):
            if self.neurons[i].getBinaryValue() == True:
                x += str(i) + " "
                counter += 1
        x += " (" + str(counter) + ")"
        return x


