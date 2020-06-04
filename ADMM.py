import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def penalty2(beta):
    m, = beta.shape
    R = -np.eye(m - 1, m, 0) + np.eye(m - 1, m, 1)
    return np.dot(R, beta)


def object_f(X, y, beta, lam, alpha):
    f = -np.dot(y, np.dot(X, beta)) + np.sum(np.log(1+np.exp(np.dot(X, beta)))) \
        + lam * alpha * norm(beta, 1) + lam * (1 - alpha) * norm(penalty2(beta), 1)
    return f


def aug_lagrangian(X, y, beta, lam, alpha, a, b, u, v, mu1, mu2):
    """
    :param X: Data Matrix
    :param y: Class
    :param beta: Coefficient vector
    :param lam: Penalty factor
    :param alpha: Adjustment factor
    :param a: New variable, a=beta
    :param b: New variable, b=R*beta
    :param u: Dual variable
    :param v: Dual variable
    :param mu1: Augmented Lagrangian variable
    :param mu2: Augmented Lagrangian variable
    :return: Value of Augmented Lagrangian function
    """
    augL = object_f(X, y, beta, lam, alpha)\
         + np.dot(u, beta - a) + np.dot(v, penalty2(beta) - b)\
         + mu1/2 * norm(beta - a, 2)**2 + mu2/2 * norm(penalty2(beta) - b, 2)**2
    return augL


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def soft_threshold(x1, x2):
    """

    :param x1: Variable 1 is a value
    :param x2: Variable 2 is a numpy vector
    :return: Vector with same shape of x2
    """
    ans = np.array([np.sign(x)*max(0, np.abs(x)-x1) for x in x2])
    return ans


def beta_update_newton(X, y, beta, lam, alpha, a, b, u, v, mu1, mu2):
    """
    Solve the beta update
        minimize  -logistic(beta) + u(beta-a) + v(R*beta-b)+ mu1/2 * ||beta-a||^2 + mu2/2 * ||R*beta-b||^2
    via Newton's method with Armijo rule
    """
    ALPHA = 0.3
    BETA = 0.5
    TOLERANCE = 1e-4
    MAX_ITER = 10
    n, m = X.shape
    R = -np.eye(m - 1, m, 0) + np.eye(m - 1, m, 1)

    B = mu1*np.ones((m, m))+mu2*(-np.eye(m, m, 0) + np.eye(m, m, 1))  # maybe something wrong
    '''for j in range(m):
        for k in range(m):
            if k == j + 1:
                B[j, k] += mu2
            if k == j:
                B[j, k] -= mu2'''

    for i in range(MAX_ITER):
        l = aug_lagrangian(X, y, beta, lam, alpha, a, b, u, v, mu1, mu2)
        gradient = np.dot(X.T, sigmoid(np.dot(X, beta)) - y) \
                   + u + np.dot(R.T, v) \
                   + mu1 * (beta - a) \
                   + mu2 * (np.dot(np.dot(R.T, R), beta) - np.dot(R.T, b))
        A = np.eye(n)
        for j in range(n):
            h = sigmoid(np.dot(X[j], beta))
            A[j, j] = h * (1 - h)
        Hessian = -np.dot(np.dot(X.T, A), X) + B
        dbeta = np.linalg.solve(Hessian, gradient)  # = -(Hessian)^(-1) * gradient, Newton step
        dl = np.dot(gradient.T, dbeta)  # Newton decrement

        if norm(dbeta, 1) < TOLERANCE:
            break

        # backtracking
        delta = 1
        # count = 0
        while aug_lagrangian(X, y, beta + delta * dbeta, lam, alpha, a, b, u, v, mu1, mu2) > l + ALPHA * delta * dl:
            delta *= BETA
            # count += 1
            # print(count)
        beta += delta*dbeta
        # print(beta[0])
    return beta


def admm(X, y, lam, alpha, mu1, mu2, beta=0):

    PRINT = 1
    MAX_ITER = 100
    ABSTOL = 1e-4
    RELTOL = 1e-3
    error_history = []


    n, m = X.shape  # n:# of examples, m:# of features
    if beta == 0:
        beta = np.zeros(m)
    beta_new = beta
    a = beta_new
    b = np.zeros(m-1)
    u = np.zeros(m)
    v = np.zeros(m-1)

    if PRINT:
        print("Begin!")
    for i in range(MAX_ITER):
        beta_old = beta_new.copy()
        beta_new = beta_update_newton(X, y, beta_old.copy(), lam, alpha, a, b, u, v, mu1, mu2)
        a = soft_threshold(lam*alpha/mu1, beta_new+u/mu1)
        b = soft_threshold(lam*(1-alpha)/mu2, penalty2(beta_new)+v/mu2)
        u = u + mu1*(beta_new - a)
        v = v + mu2*(penalty2(beta_new) - b)
        RelE = np.abs(object_f(X, y, beta_old, lam, alpha) - object_f(X, y, beta_new, lam, alpha)) \
                / object_f(X, y, beta_old, lam, alpha)
        error_history.append(RelE)
        if PRINT:
            print("the {}th relative error is {}".format(i+1, RelE))
        if RelE < RELTOL:
            break
    if PRINT:
        print(error_history)
        plt.figure()
        plt.plot(error_history)
    return beta_new
