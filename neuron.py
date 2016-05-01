class Neuron(object):
    def __init__(self, resting, upper, lower, voltageDecay, spikeDecay):
        """Return a new Neuron object."""
        self.resting = resting
        self.upper = upper
        self.lower = lower
        self.voltageDecay = voltageDecay
        self.spikeDecay = spikeDecay

        self.voltage = resting
        self.spikeTimer = 0
        self.running = True

    def update(self, dv, dt):
        """Update the neuron by one time step."""
        if self.running:
            self.voltage += dv
            if self.voltage < self.lower:
                self.voltage = self.lower
            if self.voltage > self.upper:
                self.voltage = self.lower
                self.spikeTimer = self.spikeDecay
            else:
                self.voltage += -1.0/self.voltageDecay*(self.voltage-self.resting)*dt
                self.spikeTimer -= max(self.spikeTimer-1, 0)

    def getBinaryValue(self):
        """Get the binary value of the neuron."""
        self.running = True
        return self.spikeTimer > 0

    def setBinaryValue(self, value):
        """Set the binary value of the neuron."""
        self.running = False
        if value:
            self.spikeTimer = self.spikeDecay
        else:
            self.spikeTimer = 0