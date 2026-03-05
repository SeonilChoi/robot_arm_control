"""
LQR control: u = -K @ (z - z_e), tau = G(q_e) + u.
K from continuous-time ARE: A'P + PA - P B R^{-1} B' P + Q = 0, K = R^{-1} B' P.
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import M, C_vec, G, rk4_step, linearize

# Equilibrium (upright: q = [0,0,0] gives gravity; use a stable pose)
Q_E = np.array([0.4, 0.3, 0.2])
TAU_E = G(Q_E)
Z_E = np.concatenate([Q_E, np.zeros(3)])

A, B = linearize(Q_E)
Q_lqr = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
R_lqr = np.diag([0.1, 0.1, 0.1])

# Solve CARE: A'P + PA - P B R^{-1} B' P + Q = 0  =>  K = R^{-1} B' P
def solve_care(A, B, Q, R):
    Rinv = np.linalg.inv(R)
    H = np.block([
        [A, -B @ Rinv @ B.T],
        [-Q, -A.T]
    ])
    eigvals, eigvecs = np.linalg.eig(H)
    n = A.shape[0]
    idx = np.argsort(np.real(eigvals))
    stable_idx = idx[:n]
    X = eigvecs[:n, stable_idx].real
    Y = eigvecs[n:, stable_idx].real
    P = Y @ np.linalg.solve(X.T, np.eye(n)).T
    P = 0.5 * (P + P.T)
    K = Rinv @ B.T @ P
    return K, P

K_lqr, _ = solve_care(A, B, Q_lqr, R_lqr)

DT = 0.002
T_MAX = 4.0
N = int(T_MAX / DT)

q = np.array([0.5, 0.35, 0.25])
qd = np.array([0.1, 0.0, 0.0])
t_hist = []
q_hist = []
qd_hist = []

for i in range(N):
    t = i * DT
    z = np.concatenate([q, qd])
    u = -K_lqr @ (z - Z_E)
    tau = TAU_E + u
    tau = np.clip(tau, -50.0, 50.0)

    t_hist.append(t)
    q_hist.append(q.copy())
    qd_hist.append(qd.copy())

    q, qd = rk4_step(q, qd, tau, DT)

q_hist = np.array(q_hist)
qd_hist = np.array(qd_hist)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for j in range(3):
    axes[0].plot(t_hist, q_hist[:, j], label=r'$q_{}$'.format(j+1))
    axes[0].axhline(Q_E[j], color='gray', linestyle='--')
axes[0].set_ylabel('q (rad)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('LQR: joint angles')
for j in range(3):
    axes[1].plot(t_hist, qd_hist[:, j], label=r'$\dot{q}_%d$' % (j+1))
axes[1].set_ylabel(r'$\dot{q}$ (rad/s)')
axes[1].set_xlabel('t (s)')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('lqr_result.png', dpi=120)
plt.show()
