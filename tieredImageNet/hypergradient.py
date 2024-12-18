import torch
from torch.autograd import grad as torch_grad
from torch import Tensor
from CG_torch import cg
from typing import Union, Tuple, List, Callable, Any


def CG_Centralized(
        y: List[Tensor],
        x: List[Tensor],
        CG_steps: int,
        g_loss,
        f_loss,
        epsilon=1e-8,
):
    grad_f_y, grad_f_x = get_outer_gradients(f_loss, y, x)
    grad_g_y = torch.autograd.grad(g_loss, y, create_graph=True)

    vector = cg(lambda v: torch_grad(grad_g_y, y, grad_outputs=v, retain_graph=True),
                grad_f_y,
                CG_steps,
                epsilon=epsilon,
                )
    jvp = torch_grad(grad_g_y, x, grad_outputs=vector)
    hypergradient = [g - j for g, j in zip(grad_f_x, jvp)]

    update_tensor_grads(x, hypergradient)
    return vector


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g
