import numpy as np

from neuron import Neuron

class Network(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.X = np.zeros(m)
        self.S = [Neuron() for i in range(n)]
        self.A = np.zeros((m, n))
        self.alpha = 1
        self.sigma = 1

    def update(self, dt):
        excitatory = np.dot(self.A.transpose(), self.X)
        binaryS = np.array([n.getBinaryValue() for n in self.S])
        squaredA = np.dot(self.A.transpose(), self.A)

        # Neurons do not inhibit themselves
        for i in range(self.n):
            squaredA[i, i] = 0

        inhibitory = np.dot(squaredA, binaryS)
        addVector = self.alpha * (excitatory - inhibitory - self.sigma * self.sigma)
        for i in range(self.n):
            self.S[i].voltage += addVector[i]
            self.S[i].decay(dt)

    def getSpikingIndices(self):
        x = []
        for i in range(self.n):
            if self.S[i].getBinaryValue():
                x.append(i)
        return x