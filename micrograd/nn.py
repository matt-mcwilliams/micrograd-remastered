from .engine import Value
import random

class Module:
    """
    Base class for backpropagation.
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        return []


class Neuron(Module):
    """
    A single neuron, capable of calculating the weighted sums of all of the previous layer.
    """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, inputs):
        n = sum([w*x for w,x in zip(inputs,self.w)], self.b)
        return n.tanh()
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):
    """
    A layer of neurons, which outputs nout values to be passed to the next layer.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, inputs):
        return [n(inputs) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
    """
    A simple multi-layer perceptron capable of learning many patterns in the data.
    """

    def __init__(self, nin, nouts):
        self.layers = [Layer(ni, no) for ni,no in zip([nin]+nouts,nouts)]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs[0] if len(inputs)==1 else inputs
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]