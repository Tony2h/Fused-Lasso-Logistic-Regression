import numpy as np
from ADMM import admm
import matplotlib.pyplot as plt
from timeit import default_timer

plt.rcParams['figure.dpi'] = 200
n = 100
m = 4 * n
g = 15
sigma = 0.005


X = np.random.normal(size=(n, m))
X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))  # 归一化
beta = np.zeros(m)
G = np.random.choice(100, size=g, replace=None)

for i in range(100):
    temp = np.random.uniform(-2, 2)
    for j in range(int(m/100)):
        if i in G:
            beta[i*int(m/100)+j] = temp
beta_hat = []

e = np.random.normal(0, sigma**2, size=(n, m))
y = np.around(1/(1 + np.exp(-np.dot(X+e, beta))))

start = default_timer()
beta_hat.append(admm(X, y, 0.5, 0.2, 1, 1, beta=np.array([x+np.random.uniform(0.05, 0.1) if x>0.1 else x-np.random.uniform(0, 0.01) for x in beta])))
beta_hat.append(admm(X, y, 0.2, 0.5, 1, 1, beta=np.array([x+np.random.uniform(0.05, 0.1) if x>0.1 else x-np.random.uniform(0, 0.01) for x in beta])))
beta_hat.append(admm(X, y,   1, 0.1, 1, 1, beta=np.array([x+np.random.uniform(0.05, 0.1) if x>0.1 else x-np.random.uniform(0, 0.01) for x in beta])))
end = default_timer()



plt.figure()
plt.plot(beta, 'o', color='r', markersize=6)
plt.plot(beta_hat[1], 's', color='y', markersize=5, markerfacecolor='none')
plt.plot(beta_hat[0], 's', color='green', markersize=5, markerfacecolor='none')
plt.plot(beta_hat[2], 's', color='navy', markersize=5, markerfacecolor='none')
plt.xlabel('Index')
plt.ylabel('Values   of  '+'$\\beta$')
plt.legend(('True values of \\beta','$\lambda(1-\\alpha)$=0.01','$\lambda(1-\\alpha)$=0.1','$\lambda(1-\\alpha)$=0.9'),loc='best')
plt.show()

