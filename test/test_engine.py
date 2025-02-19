import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
from micrograd.engine import Value


class TestValue(unittest.TestCase):

    def test_add(self):
        # Value + Value
        a = Value(1.0)
        b = Value(4.0)
        c = a + b

        ta = torch.Tensor([1.0]).double();      ta.requires_grad = True
        tb = torch.Tensor([4.0]).double();      tb.requires_grad = True
        tc = ta + tb

        self.assertAlmostEqual(c.data, tc.item(), 4)

        # Value + float
        a = Value(-6.5)
        b = a + 0.5

        ta = torch.Tensor([-6.5]).double();     ta.requires_grad = True
        tb = ta + 0.5

        self.assertAlmostEqual(b.data, tb.item(), 4)

        # float + Value
        a = Value(9.8)
        b = 0.4 + a

        ta = torch.Tensor([9.8]).double();      ta.requires_grad = True
        tb = 0.4 + ta

        self.assertAlmostEqual(b.data, tb.item(), 4)
    
    def test_mul(self):
        # Value * Value
        a = Value(1.0)
        b = Value(4.0)
        c = a * b

        ta = torch.Tensor([1.0]).double();      ta.requires_grad = True
        tb = torch.Tensor([4.0]).double();      tb.requires_grad = True
        tc = ta * tb

        self.assertAlmostEqual(c.data, tc.item(), 4)

        # Value * float
        a = Value(-6.5)
        b = a * 0.5

        ta = torch.Tensor([-6.5]).double();     ta.requires_grad = True
        tb = ta * 0.5

        self.assertAlmostEqual(b.data, tb.item(), 4)

        # float * Value
        a = Value(9.8)
        b = 0.4 * a

        ta = torch.Tensor([9.8]).double();      ta.requires_grad = True
        tb = 0.4 * ta

        self.assertAlmostEqual(b.data, tb.item(), 4)    

    def test_neuron(self):
        x1 = Value(2.0)
        w1 = Value(-3.0)
        x2 = Value(0.0)
        w2 = Value(1.0)
        b = Value(-1.0)
        x1w1 = x1 * w1
        x2w2 = x2 * w2
        x1w1x2w2 = x1w1 + x2w2
        n = x1w1x2w2 + b
        o = n.tanh()

        tx1 = torch.Tensor([2.0]).double()
        tw1 = torch.Tensor([-3.0]).double()
        tx2 = torch.Tensor([0.0]).double()
        tw2 = torch.Tensor([1.0]).double()
        tb = torch.Tensor([-1.0]).double()
        tx1w1 = tx1 * tw1
        tx2w2 = tx2 * tw2
        tx1w1x2w2 = tx1w1 + tx2w2
        tn = tx1w1x2w2 + tb
        to = tn.tanh()

        self.assertAlmostEqual(o.data, to.item(), 4)




if __name__ == '__main__':
    unittest.main()