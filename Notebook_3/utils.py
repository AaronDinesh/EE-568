import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
from matplotlib.table import Table

import time
import numpy as np

import gym
from gym import wrappers


# Hessian vector product H*x
def hessian_vector_product(actor_params, grad_kl, vector, retain_graph=True, cg_damping=0.2):
    # J*x   
    J_x = (grad_kl * vector).sum()

    # compute gradient
    grads = torch.autograd.grad(
        J_x,
        actor_params,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=False
    )
    # flatten
    H_x = torch.cat([torch.ravel(grad) for grad in grads if grad is not None])

    return H_x + cg_damping * vector
        

# compute gradients of parameters
def compute_actor_gradient(policy_obj, kl_div, actor_params):

    grad_kl = []
    grad_obj = []
    grad_shape = []
    relevant_actor_params = []

    # compute kl_div grad
    for param in actor_params:
        # compute policy obj gradient
        grad_kl_param, *_ = torch.autograd.grad(
            kl_div,
            param,
            retain_graph=True, # to all higher order graph
            create_graph=True, # to compute higher order graph
            allow_unused=True,
            only_inputs=True
        )

        if grad_kl_param is None:
            continue

        # compute kl div gradient
        grad_obj_param, *_ = torch.autograd.grad(
            policy_obj,
            param,
            retain_graph=True, # to all higher order graph
            only_inputs=True # to compute higher order graph
        )

        grad_kl.append(grad_kl_param.reshape(-1))
        grad_obj.append(grad_obj_param.reshape(-1))
        grad_shape.append(grad_kl_param.shape)
        relevant_actor_params.append(param)

    return relevant_actor_params, torch.cat(grad_kl), torch.cat(grad_obj), grad_shape


# conjugate gradient solver (from stable baseline)
def conjugate_gradient_solver(
    matrix_vector_dot_fn,
    b,
    max_iter=20,
    residual_tol=1e-10,
):
    """
    Finds an approximate solution to a set of linear equations Ax = b

    Sources:
     - https://github.com/ajlangley/trpo-pytorch/blob/master/conjugate_gradient.py
     - https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py#L122

    Reference:
     - https://epubs.siam.org/doi/abs/10.1137/1.9781611971446.ch6

    :param matrix_vector_dot_fn:
        a function that right multiplies a matrix A by a vector v
    :param b:
        the right hand term in the set of linear equations Ax = b
    :param max_iter:
        the maximum number of iterations (default is 10)
    :param residual_tol:
        residual tolerance for early stopping of the solving (default is 1e-10)
    :return x:
        the approximate solution to the system of equations defined by `matrix_vector_dot_fn`
        and b
    """

    # The vector is not initialized at 0 because of the instability issues when the gradient becomes small.
    # A small random gaussian noise is used for the initialization.
    x = 1e-4 * torch.randn_like(b)
    residual = b - matrix_vector_dot_fn(x)
    # Equivalent to th.linalg.norm(residual) ** 2 (L2 norm squared)
    residual_squared_norm = torch.matmul(residual, residual)

    if residual_squared_norm < residual_tol:
        # If the gradient becomes extremely small
        # The denominator in alpha will become zero
        # Leading to a division by zero
        return x

    p = residual.clone()

    for i in range(max_iter):
        # A @ p (matrix vector multiplication)
        A_dot_p = matrix_vector_dot_fn(p)

        alpha = residual_squared_norm / p.dot(A_dot_p)
        x += alpha * p

        if i == max_iter - 1:
            return x

        residual -= alpha * A_dot_p
        new_residual_squared_norm = torch.matmul(residual, residual)

        if new_residual_squared_norm < residual_tol:
            return x

        beta = new_residual_squared_norm / residual_squared_norm
        residual_squared_norm = new_residual_squared_norm
        p = residual + beta * p