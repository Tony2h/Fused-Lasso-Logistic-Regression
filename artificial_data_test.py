import numpy as np
from ADMM import admm
import matplotlib.pyplot as plt
from timeit import default_timer

plt.rcParams['figure.dpi'] = 200
n = 200
m = 4 * n
g = 15
sigma = [0.001, 0.005, 0.01]


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
for i in range(3):
    e = np.random.normal(0, sigma[i]**2, size=(n, m))
    y = np.around(1/(1 + np.exp(-np.dot(X+e, beta))))

    start = default_timer()
    beta_hat.append(admm(X, y, 2, 0.3, 1, 1, beta=np.zeros(m)))
    end = default_timer()
    time_spent = end - start
    odis = np.linalg.norm(beta-beta_hat[i], 2)
    err = odis/np.linalg.norm(beta, 2)
    y_hat = np.around(1/(1 + np.exp(-np.dot(X, beta_hat[i]))))
    print(i, ": ", time_spent, " ", odis, " ", err)


plt.figure()
plt.plot(beta, 'o', color='r', markersize=6)
plt.plot(beta_hat[0], 's', color='dodgerblue', markersize=5, markerfacecolor='none')
plt.xlabel('Index')
plt.ylabel('Values   of  '+'$\\beta$')
plt.legend(('True  values  of  '+'$\\beta$', 'Estimated  values  of  '+'$\\beta$'), loc='upper right')
plt.show()

