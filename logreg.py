import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as anp
from autograd import grad, jacobian
from numpy import random
from scipy import linalg, optimize
from scipy.optimize import OptimizeResult
from time import time

from fjlt import fjlt_usp

def logddpsi(u):
    return u - 2*np.log(np.exp(u) + 1)

def sketched_hess_sqrt(x, m, A):
    hess_sqrt = A.transpose() * np.exp(0.5 * logddpsi(A.dot(x)))
    hess_sqrt = hess_sqrt.transpose()
    sketched = np.sqrt(A.shape[0]) * fjlt_usp(hess_sqrt, m)
    return sketched

def ihs(fun, x0, jac, m, A, args=(), callback=None, **options):
    x = x0
    nit = 0
    for _ in range(25):
        nit += 1
        SH = sketched_hess_sqrt(x, m, A)
        try:
            step = linalg.solve(SH.transpose().dot(SH), jac(x))
            if (fun(x - step) < fun(x)):
                x = x - step
        except:
            continue
        cb(x)

    return OptimizeResult(
            fun=fun(x),
            x=x,
            nit=nit,
            nfev=0,
            success=True)

if __name__ == "__main__":
    random.seed(42)

    d = 100
    m = 4*d # for ROS IHS
    n = 65536

    plt.rc('text', usetex=True)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))


    for row, rho in enumerate([0.0, 0.7, 0.9]):
        # generate problem
        A = random.multivariate_normal(
                np.zeros(d),
                rho*np.ones((d, d)) + np.diag(np.ones(d) - rho),
                size=n)
        x_opt = random.normal(0, 1, size=d)
        x_opt = x_opt / linalg.norm(x_opt)
        y = np.ones(n)
        y[np.where(A.dot(x_opt) < 0)] = -1

        def objective(x):
            return anp.sum(anp.log(1 + anp.exp(anp.dot(A, x) * y)))
        jac = grad(objective)
        hess = jacobian(jac)

        methods = ["BFGS", "TNC", "trust-ncg", "IHS"]
        methods = ["IHS"]
        markers = ['^', 'o', 'D', 'x']
        for i,method in enumerate(methods):
            print(rho, method)
            x_trace = []
            clock_time = []
            def cb(x):
                global x_trace
                x_trace.append(x)
                clock_time.append(time())
            start = time()
            method_arg = method
            if method_arg is "IHS":
                method_arg = ihs
            opt = optimize.minimize(
                    objective,
                    x0=np.zeros(d),
                    method=method_arg,
                    jac=jac,
                    hess=hess,
                    callback=cb,
                    options={
                        'm': m,
                        'A': A
                        })
            obj_trace = np.array(list(map(objective, x_trace)))
            ax = axes[row][0]
            ax.semilogy(np.arange(obj_trace.shape[0]), obj_trace,
                    label=method, marker=markers[i])
            ax.set_title(r'Optimality vs time, $\rho=${}'.format(rho))
            ax.set_xlabel("Num. iterations")
            ax.set_ylabel("Optimality gap")
            ax.grid(True)
            ax.legend(methods)

            clock_time = np.array(list(map(lambda x: x - start, clock_time)))
            ax = axes[row][1]
            ax.semilogy(clock_time, obj_trace,
                    label=method, marker=markers[i])
            ax.set_title(r'Optimality vs time, $\rho=${}'.format(rho))
            ax.set_xlabel("Wall clock time (seconds)")
            ax.set_ylabel("Optimality gap")
            ax.grid(True)
            ax.legend(methods)

    plt.savefig("logreg.pdf", dpi=150)
    plt.show()
