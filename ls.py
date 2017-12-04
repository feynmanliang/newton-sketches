import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import autograd.numpy as anp
from autograd import grad
from numpy import random
from scipy import linalg
from scipy import optimize

random.seed(42)


data = []
for d in [32, 64, 128, 256]:
    n = 100*d

    for trial in range(10):
        print(d, trial)
        A = random.normal(0, 1, size=(n,d))

        # random regression vector from sphere
        x_opt = random.normal(0, 1, size=d)
        x_opt = x_opt / linalg.norm(x_opt)

        sigma = 1
        y = np.dot(A, x_opt) + random.normal(0, sigma, size=n)

        # LS fit
        x_fit, _, _, _ = linalg.lstsq(A, y)
        ls_error = (1./np.sqrt(n)) * linalg.norm(A.dot(x_fit - x_opt))
        data.append({
            'd': d,
            'trial': trial,
            'method': 'LS',
            'error': ls_error
        })

        # IHS fit
        m = 6*d
        num_iter = int(1 + np.ceil(np.log2(np.sqrt(n/d))))
        x_fit = np.zeros(d)
        for _ in range(num_iter):
            S = random.normal(0, 1, size=(m,n))
            objective = lambda x: \
                    (1. / (2*m)) * anp.sum(anp.power(anp.dot(S, anp.dot(A, x - x_fit)), 2)) \
                    - anp.dot(y - anp.dot(A, x_fit), anp.dot(A, x))
            jac = grad(objective)
            opt = optimize.minimize(objective, x0 = x_fit, jac=jac)
            x_fit = opt.x
        ihs_error = (1./np.sqrt(n)) * linalg.norm(A.dot(x_fit - x_opt))
        data.append({
            'd': d,
            'trial': trial,
            'method': 'IHS',
            'error': ihs_error
        })

        # sketched LS (suboptimal)
        m = m * num_iter
        S = random.normal(0, 1, size=(m,n))
        x_fit, _, _, _ = linalg.lstsq(S.dot(A), S.dot(y))
        sls_error = (1./np.sqrt(n)) * linalg.norm(A.dot(x_fit - x_opt))

        data.append({
            'd': d,
            'trial': trial,
            'method': 'SLS',
            'error': sls_error
        })

df = pd.DataFrame(data)\
        .groupby(['d','method']).mean()\
        .pivot_table(index="d", columns="method", values="error")

fig, ax = plt.subplots()
pos = np.arange(len(df['IHS']))
width = 0.25

for i,method in enumerate(df.columns):
    ax.bar(pos + i*width, df[method], width, label=method)

ax.set_ylabel('Error')
ax.set_xlabel('d')
ax.set_xticks(pos + width / 2)
ax.set_xticklabels(df.index)
ax.legend()
plt.grid(True)

plt.savefig("ls.pdf", dpi=150)
plt.show()
