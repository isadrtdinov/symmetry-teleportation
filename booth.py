import torch


def booth(x):
    return (x[0] + 2 * x[1] - 7).square() + (2 * x[0] + x[1] - 5).square()


def h(x):
    return torch.stack([
        x[0] + 2 * x[1] - 7,
        2 * x[0] + x[1] - 5
    ])


def h_inv(u):
    return torch.stack([
        -1/3 * u[0] + 2/3 * u[1] + 1,
        2/3 * u[0] - 1/3 * u[1] + 3
    ])


def rotate(u, theta):
    R = torch.stack([
        torch.cat([theta.cos(), -theta.sin()]),
        torch.cat([theta.sin(), theta.cos()]),
    ], dim=0)
    return R @ u


def teleport_booth(x, theta):
    return h_inv(rotate(h(x), theta))
