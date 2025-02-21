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
    
    def test_tanh(self):
        a = Value(9.8)
        b = a.tanh()

        ta = torch.Tensor([9.8]).double();      ta.requires_grad = True
        tb = ta.tanh()

        self.assertAlmostEqual(b.data, tb.item(), 4)
    
    def test_tanh_alt(self):
        a = Value(9.8)
        b = (2*a).exp()
        c = (b-1)/(b+1)

        c.backward()

        ta = torch.Tensor([9.8]).double();      ta.requires_grad = True
        tb = ta.tanh()

        tb.backward()

        self.assertAlmostEqual(c.data, tb.item(), 4)
        self.assertAlmostEqual(a.grad, ta.grad.item(), 4)

    def test_neuron_value(self):
        x1 = Value(2.0)
        w1 = Value(-3.0)
        x2 = Value(0.0)
        w2 = Value(1.0)
        b = Value(6.881374)
        x1w1 = x1 * w1
        x2w2 = x2 * w2
        x1w1x2w2 = x1w1 + x2w2
        n = x1w1x2w2 + b
        o = n.tanh()

        o.backward()  # TODO: Add tests for backward pass

        tx1 = torch.Tensor([2.0]).double();         tx1.requires_grad = True
        tw1 = torch.Tensor([-3.0]).double();        tw1.requires_grad = True
        tx2 = torch.Tensor([0.0]).double();         tx2.requires_grad = True
        tw2 = torch.Tensor([1.0]).double();         tw2.requires_grad = True
        tb = torch.Tensor([6.881374]).double();     tb.requires_grad = True
        tx1w1 = tx1 * tw1
        tx2w2 = tx2 * tw2
        tx1w1x2w2 = tx1w1 + tx2w2
        tn = tx1w1x2w2 + tb
        to = tn.tanh()

        to.backward()

        self.assertAlmostEqual(o.data, to.item(), 4)

        self.assertAlmostEqual(x1.grad, tx1.grad.item(), 4)
        self.assertAlmostEqual(x2.grad, tx2.grad.item(), 4)
        self.assertAlmostEqual(w1.grad, tw1.grad.item(), 4)
        self.assertAlmostEqual(w2.grad, tw2.grad.item(), 4)
        self.assertAlmostEqual(b.grad, tb.grad.item(), 4)
        
    def test_relu(self):

        a = Value(4.0)
        b = a.relu()

        b.backward()

        ta = torch.Tensor([4.0]).double();    ta.requires_grad = True
        tb = ta.relu()

        tb.backward()

        self.assertAlmostEqual(b.data, tb.item(), 4)
        self.assertAlmostEqual(a.grad, ta.grad.item(), 4)
    
    def test_comprehensive(self):
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f

        g.backward()

        ta = torch.Tensor([-4.0]).double();     ta.requires_grad = True
        tb = torch.Tensor([2.0]).double();      tb.requires_grad = True
        tc = ta + tb
        td = ta * tb + tb**3
        tc += tc + 1
        tc += 1 + tc + (-ta)
        td += td * 2 + (tb + ta).relu()
        td += 3 * td + (tb - ta).relu()
        te = tc - td
        tf = te**2
        tg = tf / 2.0
        tg += 10.0 / tf

        tg.backward()

        self.assertAlmostEqual(g.data, tg.item(), 4)
        self.assertAlmostEqual(a.grad, ta.grad.item(), delta=10)  # use delta because imprecise at large numbers
        self.assertAlmostEqual(b.grad, tb.grad.item(), delta=10)



if __name__ == '__main__':
    unittest.main()