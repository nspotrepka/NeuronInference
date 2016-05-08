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

def sine(k, m):
    g = np.zeros(m)
    for i in range(m):
        x = (i - m/2.0) / m * k
        g[i] = math.sin(x)
    return g

def cosine(k, m):
    g = np.zeros(m)
    for i in range(m):
        x = (i - m/2.0) / m * k
        g[i] = math.cos(x)
    return g

class Network(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.X = gabor1D(math.pi/4, m)+gabor1D(0, m)
        self.S = [Neuron() for i in range(n)]
        self.A = np.zeros((m, n))
        for i in range(n):
            phi = i * 2.0 * math.pi / n
            #if i < n/2:
            #    self.A[:,i] = cosine(i, m)
            #else:
            #    self.A[:,i] = sine(i-n/2+1, m)
            self.A[:,i] = gabor1D(phi, m).transpose()
        self.alpha = 250000000
        self.sigma = 0.001

        excitatory = np.dot(self.A.transpose(), self.X) / self.m
        # Half-wave square
        excitatory = np.array([x*x if x >= 0 else 0 for x in excitatory])
        self.excitatory = excitatory

        squaredA = np.dot(self.A.transpose(), self.A)
        # Half-wave square
        for i in range(self.n):
            for j in range(self.n):
                x = squaredA[i, j]
                squaredA[i, j] = x*x/self.n if x >= 0 else 0
        # Neurons do not inhibit themselves
        for i in range(self.n):
            squaredA[i, i] = 0
        self.squaredA = squaredA

        #plt.plot(self.A.transpose()[0])
        #plt.show()

    def update(self, dt):
        binaryS = np.array([n.getBinaryValue() for n in self.S])
        inhibitory = np.dot(self.squaredA, binaryS)
        addVector = self.alpha * (self.excitatory - inhibitory - self.sigma * self.sigma)
        for i in range(self.n):
            self.S[i].voltage += addVector[i]*dt
            self.S[i].decay(dt)

    def getSpikingIndices(self):
        x = []
        for i in range(self.n):
            if self.S[i].getBinaryValue():
                x.append(i)
        return x