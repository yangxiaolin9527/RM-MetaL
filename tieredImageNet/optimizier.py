import torch
from torch.optim.optimizer import Optimizer


class Base(Optimizer):
    def __init__(self, params, name=None, device=None):
        defaults = dict(params=params, name=name, device=device)

        super(Base, self).__init__(params, defaults)


class MAML(Base):
    def __init__(self, *args, **kwargs):
        super(MAML, self).__init__(*args, **kwargs)


class ANIL(Base):
    def __init__(self, *args, **kwargs):
        super(ANIL, self).__init__(*args, **kwargs)


class Algorithm1(Base):
    def __init__(self, *args, **kwargs):
        super(Algorithm1, self).__init__(*args, **kwargs)

    def head_update(self, loss, head, lr=None):
        param = list(head.parameters())
        grads = torch.autograd.grad(loss, param)
        for l, g in zip(param, grads):
            if l.grad is None:
                l.grad = torch.zeros_like(l)
            if g is not None:
                l.grad += g
        for _, param_key in enumerate(head._parameters):
            p = head._parameters[param_key]
            if p is not None and p.grad is not None:
                p.data = p.data - lr * p.grad.data


class ITDBiO(Base):
    def __init__(self, *args, **kwargs):
        super(ITDBiO, self).__init__(*args, **kwargs)

