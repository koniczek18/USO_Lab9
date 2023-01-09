import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy
import scipy.interpolate


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

def modelModified(x,t,k0,k1,k2,k3,R_inv,B,l, m, J, d, g):
    x1 = x[0]
    x2 = x[1]
    Pt=np.array([[k0(t),k1(t)],[k2(t),k3(t)]])
    Kt = R_inv @ B.T @ Pt
    X = np.array([[x1], [x2]])
    X0 = np.array([[np.pi], [0]])
    u = -Kt @ (X - X0)
    x1p = x2
    x2p = (u - d * x2 - m * g * l * np.sin(x1)) / J
    return np.array([x1p, x2p[0]])
def riccati(p,t,A,B,R,q):
    Pt=p.reshape((2,2))
    Pp=(Pt@A-Pt@B*R@B.T@Pt+A.T@Pt+q)
    return Pp.reshape(4)

def zadanie1(active):
    if active:
        #def variables
        l = 1
        u = 0
        m = 9
        J = 1
        d = 0.5
        g = 9.81
        t = np.linspace(0, 5, 1001)
        #
        #base
        ans1 = odeint(model, y0=[np.pi / 4, 0], t=t, args=(u, l, m, J, d, g))
        plt.figure('Base model')
        plt.plot(t, ans1[:, 0], label='x1')
        plt.plot(t, ans1[:, 1], label='x2')
        plt.legend()
        #
        # y0 def
        value = 0.1
        Y0 = [np.pi - value, 0]
        # y0 end def
        #
        #infinite
        A = np.array([[0, 1], [(m * g * l) / J, -d / J]])
        B = np.array([[0], [1 / J]])
        r = 1
        q = np.array([[1, 0], [0, 1]])
        P = scipy.linalg.solve_continuous_are(A, B, q, r)
        R_inv = np.array([1])
        BT = np.transpose(B)
        K = R_inv @ BT @ P
        ans2 = odeint(modelMod, y0=Y0, t=t, args=(K, l, m, J, d, g))
        plt.figure('With LQR - inf')
        plt.plot(t, ans2[:, 0]-np.pi, label='x1')
        plt.plot(t, ans2[:, 1], label='x2')
        plt.legend()
        #
        #finite
        t = np.linspace(0, 1, 101)
        t_inv = np.flip(t, 0)
        S = np.array([[1, 0], [0, 1]])
        P_t = odeint(riccati, y0=[1, 0, 0, 1], t=t, args=(A,B,r,q))
        P_t1 = np.array([[P_t[1, 0], P_t[-1, 1]], [P_t[1, 2], P_t[-1, 3]]])
        print(P_t1)
        Pt_inter0 = scipy.interpolate.interp1d(t_inv, P_t[:, 0], fill_value='extrapolate')
        Pt_inter1 = scipy.interpolate.interp1d(t_inv, P_t[:, 1], fill_value='extrapolate')
        Pt_inter2 = scipy.interpolate.interp1d(t_inv, P_t[:, 2], fill_value='extrapolate')
        Pt_inter3 = scipy.interpolate.interp1d(t_inv, P_t[:, 3], fill_value='extrapolate')
        SystemSym = odeint(modelModified, y0=Y0, t=t, args=(Pt_inter0, Pt_inter1, Pt_inter2, Pt_inter3,R_inv,B,l,m,J,d,g))
        plt.figure('With LQR - fin t=(0,1)')
        plt.plot(t, SystemSym[:, 0]-np.pi, label='x1')
        plt.plot(t, SystemSym[:, 1], label='x2')
        plt.legend()
        #
        plt.show()


if __name__ == '__main__':
    zadanie1(True)
