import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as anp
from autograd import grad
from numpy import random
from scipy import linalg
from scipy import optimize

#np.random.seed(42)

# def objective(x):
#     return (x[0] - 10.0)**2 + (x[1] + 5.0)**2

# jac, hess = gh(objective)

# x0 = np.array([24, 17])
# bnds = ((0, None), (0, None))
# method = 'Newton-CG'
# res = optimize.minimize(objective, x0, method=method, jac=jac, hess=hess, bounds=bnds,
#                options={'disp': True})

# print(res.x)  # optimal parameter values
# print(res.fun)  # optimal objective
# print(res.jac)  # gradient at optimum

def plot_region(A, b):
    # Construct lines
    x = np.linspace(0, 20, 2000)

    y = []
    for i in range(np.size(A, 0)):
        y.append((b[i] - A[i,0]*x) / A[i,1])
        plt.plot(x, y[i])
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
    plt.xlim((0, 16))
    plt.ylim((0, 11))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

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
m = 8

c = np.array([-3, -1], dtype=float)

def log_barrier_obj(x, tau):
    return tau * anp.dot(c, x) - anp.sum(anp.log(b - anp.dot(A, x)))

def sketched_hess_sqrt(x):
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
    for _ in range(250):
        path.append(x)
        while True:
            SH = hess_fn(x)
            step, _, _, _ = linalg.lstsq(SH.transpose().dot(SH), jac(x))
            if np.any(b - np.dot(A, x - step) <= 0) or linalg.norm(step) < 1e-2:
                break
            x = x - step
        tau *= 1.1
    return np.array(path)




def plot_path(path, color):
    x = path[:,0]
    y = path[:,1]
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.003, color=color)

lp_opt = optimize.linprog(c, A, b)
ip_path = compute_traj(hess_sqrt)
ip_sketch_path = compute_traj(sketched_hess_sqrt)

plt.figure(0)
plot_region(A, b)
plot_path(ip_path, 'r')
plot_path(ip_sketch_path, 'b')
plt.plot(lp_opt.x[0], lp_opt.x[1], 'g*')
plt.show()

plt.figure(1)
plt.semilogy(
        np.arange(ip_path.shape[0]),
        ip_path.dot(c) - lp_opt.fun,
        label="Exact Hessian")

plt.semilogy(
        np.arange(ip_sketch_path.shape[0]),
        ip_sketch_path.dot(c) - lp_opt.fun,
        label="Sketched Hessian"
)
plt.legend()
plt.grid()
plt.show()
