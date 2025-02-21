import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


class TestNN(unittest.TestCase):

    def test_neuron(self):
        xs = [[Value(0), Value(0)], [Value(0), Value(1)], [Value(1), Value(0)], [Value(1), Value(1)]]  # List of lists (input features)
        ys = [Value(-1), Value(-1), Value(-1), Value(1)]                    # List of floats (labels)

        n = Neuron(2)

        for i in range(200):

            # Forward pass
            mse = sum((n(x)-y)**2 for x,y in zip(xs,ys)) / len(xs)
            print(f'{i}: {mse.data}')

            # backward pass
            mse.backward()

            # apply gradients
            for p in n.parameters():
                p.data -= p.grad * 1
        
        
        mse = sum((n(x)-y)**2 for x,y in zip(xs,ys)) / len(xs)
        self.assertLess(mse.data, 0.05, "MSE not low enough")
    
    def test_mlp(self):
        xs = [
            [Value(1), Value(0), Value(1)],  # First and third active
            [Value(0), Value(1), Value(1)],  # Second and third active
            [Value(1), Value(1), Value(0)],  # First and second active
            [Value(0), Value(0), Value(1)]   # Third active
        ]

        ys = [
            Value(1),    # Positive
            Value(-1),   # Negative
            Value(1),    # Positive
            Value(-1)    # Negative
        ]

        n = MLP(3, [3, 4, 1])

        for i in range(200):

            # Forward pass
            mse = sum((n(x)-y)**2 for x,y in zip(xs,ys)) / len(xs)
            print(f'{i}: {mse.data}')

            # backward pass
            mse.backward()

            # apply gradients
            for p in n.parameters():
                p.data -= p.grad * 1
        
        
        mse = sum((n(x)-y)**2 for x,y in zip(xs,ys)) / len(xs)
        self.assertLess(mse.data, 0.05, "MSE not low enough")

        

        





if __name__ == '__main__':
    unittest.main()