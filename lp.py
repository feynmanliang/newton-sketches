import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as anp
from autograd import grad
from numpy import random
from scipy import linalg
from scipy import optimize

from fjlt import fjlt_usp

np.random.seed(42)

def plot_region(ax, A, b):
    # Construct lines
    x = np.linspace(0, 20, 2000)

    y = []
    for i in range(np.size(A, 0)):
        y.append((b[i] - A[i,0]*x) / A[i,1])
        ax.plot(x, y[i])
    y = np.array(y)

    # y >= 2
    y1 = (x*0) + 2
    # 2y <= 25 - x
    y2 = (25-x)/2.0
    # 4y >= 2x - 8
    y3 = (2*x-8)/4.0
    # y <= 2x - 5
    y4 = 2 * x -5

    # Make plot
    ax.set_xlim((2.5, 13))
    ax.set_ylim((1, 9))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    # Fill feasible region
    # mins = np.min(y, axis=0)
    # maxes = np.max(y, axis=0)
    # plt.fill_between(x, mins, maxes, color='grey', alpha=0.5)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

X = np.array([
    [0, -1, -2],
    [1, 2, 25],
    [2, -4, 8],
    [-2, 1, -5],
    [-0.1, -1, -2.5],
    [1, 2, 25],
    [-1, 5, 35],
    [-2, 1, -5],
    [10, -1, 120],
    [-10, 1, -35],
    [1, 1, 18],
    [1, 10, 92],
    [0.1, -1, -3],
    [0.9, 2, 24],
    [2.1, -4, 9],
    [-1.99, 1, -3],
], dtype=float)
A = X[:,0:2]
b = X[:,2]
n = np.size(A, 0) # NOTE: must be power of 2 for hadamard
d = np.size(A, 1)

c = np.array([-3, -1], dtype=float)

def log_barrier_obj(x, tau):
    return tau * anp.dot(c, x) - anp.sum(anp.log(b - anp.dot(A, x)))

def hess_sqrt(x, A):
    hess_sqrt = A.transpose() * (1.0 / np.abs(b - A.dot(x)))
    hess_sqrt = hess_sqrt.transpose()
    return hess_sqrt

def sketched_hess_sqrt(x, m, A):
    sketched = np.sqrt(A.shape[0]) * fjlt_usp(hess_sqrt(x, A), m)
    return sketched

def compute_traj(hess_fn):
    path = []
    x = np.array([8, 5], dtype=float)
    tau = 1e-8
    jac = grad(lambda x: log_barrier_obj(x, tau))
    for _ in range(300):
        path.append(x)
        while True:
            SH = hess_fn(x)
            step = linalg.solve(SH.transpose().dot(SH), jac(x))
            newton_dec = np.dot(jac(x), step)
            mu = 1
            while np.dot(c, x - mu * step) > np.dot(c, x) + 0.1 * mu * newton_dec \
                    or np.any(np.dot(A, x - mu * step) > b): # constraint violation
                mu *= 0.5
            x = x - mu * step
            if mu * newton_dec < 1e-2:
                break
        tau *= 1.1
    return np.array(path)

def plot_path(ax, path, color, label):
    x = path[:,0]
    y = path[:,1]
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1],
            scale_units='xy', angles='xy', scale=1, width=0.003, color=color, label=label)

lp_opt = optimize.linprog(c, A, b)
ip_path = compute_traj(lambda x: hess_sqrt(x, A))

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 10))
ms = [2, 4, 8, 16]
for row, m in enumerate(ms):
    for trial in range(3):
        ax = axes[row][trial]
        plot_region(ax, A, b)
        ip_sketch_path = compute_traj(lambda x: sketched_hess_sqrt(x, m, A))
        ax.plot(lp_opt.x[0], lp_opt.x[1], 'g*', markersize=3)
        plot_path(ax, ip_path, 'r', label="Exact Newton")
        plot_path(ax, ip_sketch_path, 'b', label="Newton Sketch")

        if row == 0:
            ax.set_title('Trial {}'.format(trial))
        if trial == 0:
            ax.set_ylabel('m = {}'.format(m))
        if row == 3 and trial == 2:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center')
fig.tight_layout()
plt.savefig("lp-central-path.pdf", dpi=150)
plt.show()

plt.figure(1, figsize=(6,4))
for m in ms:
    print(m)
    ip_sketch_path = np.array([compute_traj(lambda x: sketched_hess_sqrt(x, m, A)) for _ in range(10)]).mean(axis=0)
    plt.semilogy(
            np.arange(ip_sketch_path.shape[0]),
            ip_sketch_path.dot(c) - lp_opt.fun,
            label="Sketched Hessian (m = {})".format(m)
    )
plt.semilogy(
        np.arange(ip_path.shape[0]),
        ip_path.dot(c) - lp_opt.fun,
        label="Exact Hessian")

plt.xlabel("Num. iterations")
plt.ylabel("Optimality gap")
plt.legend()
plt.grid()
fig.tight_layout()
plt.savefig("lp.pdf", dpi=150)
plt.show()
