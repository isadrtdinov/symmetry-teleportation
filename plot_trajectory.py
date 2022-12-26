import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def compute_levels(x_range, y_range, f, num=1000):
    x, y = torch.linspace(x_range[0], x_range[1], num), torch.linspace(y_range[0], y_range[1], num)
    A, B = torch.meshgrid(x, y)
    levels = f(
        torch.stack([A.reshape(-1), B.reshape(-1)], dim=0)
    ).reshape(A.shape).T
    return x, y, levels


def plot_trajectory(w_list, w_true, func, save_path, teleports=None, name='Experiment', label='trajectory',
                    size=(10, 10), xlim=(-10, 10), ylim=(-10, 10), cbar_ticks=np.logspace(-3, 4, 8)):
    plt.figure(figsize=(10, 10))
    x, y, levels = compute_levels(xlim, ylim, func)

    plt.contourf(x, y, levels, levels=np.logspace(-3, 4, 50), norm=LogNorm())
    cb = plt.colorbar(ticks=cbar_ticks)
    cb.ax.set_yticklabels(['$10^{' + str(int(np.log10(val))) + '}$' for val in cbar_ticks])

    if teleports is None:
        plt.plot(
            w_list[:, 0], w_list[:, 1],
            color='red', marker='o', markersize=5,
            label=label
        )
    else:
        teleports = [0] + teleports + [w_list.shape[0]]
        for i in range(len(teleports) - 1):
            plt.plot(
                w_list[teleports[i]:teleports[i + 1], 0],
                w_list[teleports[i]:teleports[i + 1], 1],
                color='red', marker='o', markersize=5,
                label=label if i == 0 else None
            )
            if i > 0:
                plt.plot(
                    w_list[teleports[i] - 1:teleports[i] + 1, 0],
                    w_list[teleports[i] - 1:teleports[i] + 1, 1],
                    color='orange', linewidth=2,
                    label='symmetry teleports' if i == 1 else None
                )

    plt.gca().set_aspect('equal')
    plt.scatter([w_true[0]], [w_true[1]], color='black', edgecolors='white', marker='*', label='optimum',
                s=[500], linewidth=2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(name)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
