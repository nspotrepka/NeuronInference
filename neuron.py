class Neuron(object):
    def __init__(self):
        """Return a new Neuron object."""
        self.resting = 0
        self.upper = 10
        self.lower = -10
        self.voltageDecay = 0.01
        self.spikeDecay = 0.001

        self.voltage = self.resting
        self.spikeTimer = 0
        self.timeSinceSpike = 0
        self.spikeRate = 0

    def decay(self, dt):
        """Decay the neuron by one time step."""
        self.timeSinceSpike += dt
        if self.voltage < self.lower:
            self.voltage = self.lower
        if self.voltage > self.upper:
            self.voltage = self.lower
            self.spikeTimer = self.spikeDecay
            self.spikeRate = 1/self.timeSinceSpike
            self.timeSinceSpike = 0
        else:
            self.voltage += -1.0/self.voltageDecay * (self.voltage - self.resting) * dt
            self.spikeTimer = max(self.spikeTimer - dt, 0)

    def getBinaryValue(self):
        """Get the binary value of the neuron."""
        return self.spikeTimer > 0