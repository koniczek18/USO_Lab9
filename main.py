import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy


def model(x, t, u, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    x1p = x2
    x2p = (u - d * x2 - m * g * l * np.sin(x1)) / J
    return np.array([x1p, x2p])


def modelMod(x, t, K, l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    X=np.array([[x1], [x2]])
    X0=np.array([[np.pi], [0]])
    u = -K @ (X - X0)
    x1p = x2
    x2p = (u - d * x2 - m * g * l * np.sin(x1)) / J
    return np.array([x1p, x2p[0]])


def zadanie1(active):
    if active:
        l = 1
        u = 0
        m = 9
        J = 1
        d = 0.5
        g = 9.81
        t = np.linspace(0, 5, 1001)
        ans1 = odeint(model, y0=[np.pi / 4, 0], t=t, args=(u, l, m, J, d, g))
        plt.figure('Base model')
        plt.plot(t, ans1[:, 0], label='x1')
        plt.plot(t, ans1[:, 1], label='x2')
        plt.legend()
        A = np.array([[0, 1], [(m * g * l) / J, -d / J]])
        B = np.array([[0], [1 / J]])
        r = 1
        q = np.array([[1, 0], [0, 1]])
        P = scipy.linalg.solve_continuous_are(A, B, q, r)
        R_inv = np.array([1])
        BT = np.transpose(B)
        K = R_inv @ BT @ P
        ans2 = odeint(modelMod, y0=[np.pi-0.1, 0], t=t, args=(K, l, m, J, d, g))
        plt.figure('With LQR')
        plt.plot(t, ans2[:, 0], label='x1')
        plt.plot(t, ans2[:, 1], label='x2')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    zadanie1(True)
