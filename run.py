from ad import gh
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def objective(x):
    return (x[0] - 10.0)**2 + (x[1] + 5.0)**2

jac, hess = gh(objective)

x0 = np.array([24, 17])
bnds = ((0, None), (0, None))
method = 'Newton-CG'
res = minimize(objective, x0, method=method, jac=jac, hess=hess, bounds=bnds,
               options={'disp': True})

print(res.x)  # optimal parameter values
print(res.fun)  # optimal objective
print(res.jac)  # gradient at optimum

print(hess(x0))
res = minimize(objective, x0, method=method, jac=jac, hess=hess_sketch, bounds=bnds,
               options={'disp': True})

print(res.x)  # optimal parameter values
print(res.fun)  # optimal objective
print(res.jac)  # gradient at optimum
