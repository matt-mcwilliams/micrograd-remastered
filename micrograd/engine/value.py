import math

class Value:
    """
    Base class for everything that needs to be backpropagated.
    """

    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value({self.data})'
    
    def backward(self):
        topo = [] # this will contain all the sorted values
        stack = [self]
        while len(stack) > 0:
            stack[0].grad = 0.0
            if len([g for g in stack[0]._prev if g not in topo]) > 0:
                stack = list(stack[0]._prev) + stack
                stack = [i for n, i in enumerate(stack) if i not in stack[:n] and i not in topo] # remove duplicates
                continue
            topo.append(stack.pop(0))
        
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _prev=(self,other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):  # other + self
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):  # other - self
        return other + (-self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _prev=(self,other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward =_backward

        return out
    
    def __rmul__(self, other):  # other * self
        return self * other
    
    def __pow__(self, k):
        assert isinstance(k, (int, float)), "Only float/integer exponents for now!"
        out = Value(self.data**k, _prev=(self,), _op=f'**{k}')

        def _backward():
            self.grad = k * (self.data ** (k-1) * out.grad)
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):      # other / self
        return other * self**-1
    
    def exp(self):
        n = math.exp(self.data)
        out = Value(n, _prev=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        n = math.tanh(x)
        out = Value(n, _prev=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - n ** 2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        x = self.data
        n = x * int(x>0)
        out = Value(n, _prev=(self,), _op='relu')

        def _backward():
            self.grad += int(x>0)
        out._backward = _backward

        return out

