import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as anp
from autograd import grad
from numpy import random
from scipy import linalg
from scipy import optimize

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

def sketched_hess_sqrt(x, m):
    R = np.diag(random.binomial(1, 0.5, size=n))
    H = linalg.hadamard(n)
    S = np.sqrt(n/m) * \
            np.dot(
                    H[random.choice(n, size=m, replace=False),:],
                    R)
    return S.dot(np.diag(1.0 / np.abs(b - A.dot(x)))).dot(A)

def hess_sqrt(x):
    return np.diag(1.0 / np.abs(b - A.dot(x))).dot(A)

def compute_traj(hess_fn):
    path = []
    x = np.array([8, 5], dtype=float)
    tau = 1e-8
    jac = grad(lambda x: log_barrier_obj(x, tau))
    for _ in range(300):
        path.append(x)
        while True:
            SH = hess_fn(x)
            step, _, _, _ = linalg.lstsq(SH.transpose().dot(SH), jac(x))
            if np.any(b - np.dot(A, x - step) <= 0) or linalg.norm(step) < 1e-2:
                break
            x = x - step
        tau *= 1.1
    return np.array(path)

def plot_path(ax, path, color, label):
    x = path[:,0]
    y = path[:,1]
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1],
            scale_units='xy', angles='xy', scale=1, width=0.003, color=color, label=label)

lp_opt = optimize.linprog(c, A, b)
ip_path = compute_traj(hess_sqrt)

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
for row, m in enumerate([2, 4, 8, 16]):
    for trial in range(3):
        ax = axes[row][trial]
        plot_region(ax, A, b)
        ip_sketch_path = compute_traj(lambda x: sketched_hess_sqrt(x, m))
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
plt.show()

plt.figure(1)
plt.semilogy(
        np.arange(ip_path.shape[0]),
        ip_path.dot(c) - lp_opt.fun,
        label="Exact Hessian")

for m in [2, 4, 8, 16]:
    print(m)
    ip_sketch_path = np.array([compute_traj(lambda x: sketched_hess_sqrt(x, m)) for _ in range(10)]).mean(axis=0)
    plt.semilogy(
            np.arange(ip_sketch_path.shape[0]),
            ip_sketch_path.dot(c) - lp_opt.fun,
            label="Sketched Hessian (10 trial average, m = {})".format(m)
    )
plt.legend()
plt.grid()
plt.show()
