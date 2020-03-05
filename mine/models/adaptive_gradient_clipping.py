
import numpy as np
import torch
import torch.nn as nn
EPS = 1e-9


def grad_norm(module):
    parameters = module.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm = param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def adaptive_gradient_clipping_(generator_module: nn.Module, mi_module: nn.Module):
    """
    Clips the gradient according to the min norm of the generator and mi estimator

    Arguments:
        generator_module -- nn.Module 
        mi_module -- nn.Module
    """
    norm_generator = grad_norm(generator_module)
    norm_estimator = grad_norm(mi_module)

    min_norm = np.minimum(norm_generator, norm_estimator)

    parameters = list(
        filter(lambda p: p.grad is not None, mi_module.parameters()))
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    for p in parameters:
        p.grad.data.mul_(min_norm/(norm_estimator + EPS))
