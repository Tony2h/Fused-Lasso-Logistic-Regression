from read_data import get_data
from ADMM import admm
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.metrics import f1_score

# 数据预处理
X_train, y_train, X_test, y_test = get_data(standard=1)

# 模型拟合
beta=[]
ti = []  # 记录运行时间
lamb = [0.1, 0.2, 1, 10]
alpha = [1, 0.5, 0.1, 0.01]
for i in range(1):
    start = timer()
    beta.append(admm(X_train, y_train, lamb[0], alpha[0], 1, 1))
    end = timer()
    ti.append(end-start)


# 计算模型在训练集上的预测准确率
accurate_factor = []
f1score = []
for i in range(1):
    y_hat = np.around(1/(1 + np.exp(-np.dot(X_test, beta[i]))))
    accurate_factor.append(1 - sum(np.abs(y_test - y_hat))/y_test.shape[0])
    f1score.append(f1_score(y_test, y_hat))
'''
# plot
colors = ['orange','olivedrab','navy']
plt.rcParams['figure.dpi'] = 200
plt.cla()
plt.figure(0)
plt.plot(beta[3],'.',color='r', markersize=5)
for i in range(3):
    plt.plot(beta[i],'.',color=colors[i], markersize=7-1.5*i)
plt.xlabel('Index')
plt.ylabel('Values   of   '+'$\\beta_(\lambda,\\alpha)$')
plt.legend( ('$\lambda(1-\\alpha)$=9.9','$\lambda(1-\\alpha)$=0','$\lambda(1-\\alpha)$=0.1','$\lambda(1-\\alpha)$=0.9'),loc='best')
plt.show()
'''