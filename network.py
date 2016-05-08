import math
import numpy as np
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
        if x > s:
            s = x
    if s == 0:
        return v
    else:
        if s == float("inf") or s == -float("inf"):
            return np.array([(1 if (x == float("inf") or x == -float("inf")) else 0) for x in v])
        else:
            return v/s

class Network(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.X = gabor1D(0, m)+gabor1D(math.pi*3/2, m)
        self.S = [Neuron() for i in range(n)]
        self.A = np.zeros((m, n))
        for i in range(n):
            phi = i * 2.0 * math.pi / n
            self.A[:,i] = gabor1D(phi, m).transpose()
        self.alpha = 2
        self.sigma = 0

    def update(self, dt):
        excitatory = normalize(1 / ( np.array([np.sum((self.A.transpose()[i]-self.X)**2) for i in range(self.n)]) / self.m))
        binaryS = np.array([n.getBinaryValue() for n in self.S])
        squaredA = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                squaredA[i,j] = np.sum((self.A.transpose()[i] - self.A.transpose()[j])**2)
        squaredA = 1 / squaredA

        # Neurons do not inhibit themselves
        for i in range(self.n):
            squaredA[i, i] = 0

        # Normalize each row
        for i in range(self.n):
            squaredA[i] = normalize(squaredA[i])

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