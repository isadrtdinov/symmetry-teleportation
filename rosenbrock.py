import torch


def rosenbrock(x):
    return 100 * (x[0].square() - x[1]).square() + (x[0] - 1).square()


def h(x):
    return torch.stack([
        10 * (x[0].square() - x[1]),
        x[0] - 1
    ])


def h_inv(u):
    return torch.stack([
        u[1] + 1, (u[1] + 1).square() - 0.1 * u[0]
    ])


def rotate(u, theta):
    R = torch.stack([
        torch.cat([theta.cos(), -theta.sin()]),
        torch.cat([theta.sin(), theta.cos()]),
    ], dim=0)
    return R @ u


def teleport_rosenbrock(x, theta):
    return h_inv(rotate(h(x), theta))
