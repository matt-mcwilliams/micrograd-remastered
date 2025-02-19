import math

class Value:

    def __init__(self, data):
        self.data = data
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)

        return out
    
    def __radd__(self, other):  # other + self
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)

        return out
    
    def __rmul__(self, other):  # other * self
        return self * other
    
    def tanh(self):
        x = self.data
        out = Value(math.tanh(x))

        return out