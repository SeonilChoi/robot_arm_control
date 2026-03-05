"""
QP-based joint angle tracking.
min_a  (1/2) || b - a ||^2 + (lambda/2) ||a||^2  =>  a* = (I + lambda*I)^{-1} b,
where b = qdd_d + Kp*(q_d - q) + Kd*(qd_d - qd). Then tau = M(q)*a* + C_vec + G.
(Unconstrained QP: H = I + lambda*I, c = -b, a* = -H^{-1} c.)
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import M, C_vec, G, rk4_step

DT = 0.002
T_MAX = 5.0
N = int(T_MAX / DT)

Q_D = np.array([0.5, 0.4, 0.3])
QD_D = np.zeros(3)
QDD_D = np.zeros(3)

Kp = np.diag([60.0, 50.0, 40.0])
Kd = np.diag([18.0, 14.0, 10.0])
lam = 0.1  # regularization

q = np.array([0.0, 0.0, 0.0])
qd = np.zeros(3)
t_hist = []
q_hist = []
e_hist = []

for i in range(N):
    t = i * DT
    e = Q_D - q
    edot = QD_D - qd
    b = QDD_D + Kp @ e + Kd @ edot

    # QP: min_a (1/2)||b - a||^2 + (lambda/2)||a||^2 = (1/2) a'Ha + c'a, H = (1+lambda)*I, c = -b
    H_qp = (1.0 + lam) * np.eye(3)
    c_qp = -b
    a_star = np.linalg.solve(H_qp, -c_qp)

    tau = M(q) @ a_star + C_vec(q, qd) + G(q)
    tau = np.clip(tau, -50.0, 50.0)

    t_hist.append(t)
    q_hist.append(q.copy())
    e_hist.append(e.copy())

    q, qd = rk4_step(q, qd, tau, DT)

q_hist = np.array(q_hist)
e_hist = np.array(e_hist)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for j in range(3):
    axes[0].plot(t_hist, q_hist[:, j], label=r'$q_{}$'.format(j+1))
    axes[0].axhline(Q_D[j], color='gray', linestyle='--')
axes[0].set_ylabel('q (rad)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('QP: joint angles')
for j in range(3):
    axes[1].plot(t_hist, e_hist[:, j], label=r'$e_{}$'.format(j+1))
axes[1].set_ylabel('error (rad)')
axes[1].set_xlabel('t (s)')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('qp_result.png', dpi=120)
plt.show()
