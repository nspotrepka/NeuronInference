import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from neuron import Neuron

def gabor1D(phi, m):
    sigmaX = 1
    scalar = 1.0/(2*math.pi*sigmaX)
    k = 2
    width = 10
    g = np.zeros(m)
    for i in range(m):
        x = (i - m/2.0) / m * width
        g[i] = scalar * math.exp(-x*x/(2*sigmaX*sigmaX)) * math.cos(k*x-phi)
    return g

def normalize(v):
    s = 0
    for x in v:
        s += x
    if s == 0:
        return v
    else:
        return v/s

class Network(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.X = normalize(gabor1D(0, m))
        self.S = [Neuron() for i in range(n)]
        self.A = np.zeros((m, n))
        for i in range(n):
            phi = i * 2.0 * math.pi / n
            self.A[:,i] = normalize(gabor1D(phi, m)).transpose()
        self.alpha = 1
        self.sigma = 1

    def update(self, dt):
        excitatory = np.dot(self.A.transpose(), self.X)
        binaryS = np.array([n.getBinaryValue() for n in self.S])
        squaredA = np.dot(self.A.transpose(), self.A)

        # Neurons do not inhibit themselves
        #for i in range(self.n):
        #    squaredA[i, i] = 0

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