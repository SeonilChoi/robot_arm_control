"""
Impedance control in task space.
F = M_d*(xdd_d - xdd) + B_d*(xd_d - xd) + K_d*(x_d - x)
Desired acceleration: xdd_des = xdd_d + M_d^{-1}[ B_d*(xd_d - xd) + K_d*(x_d - x) ]
qdd_des = J_pinv @ (xdd_des - J_dot @ qd),  tau = M @ qdd_des + C_vec + G
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import (
    M, C_vec, G, forward_kinematics, jacobian, jacobian_dot,
    rk4_step
)

DT = 0.002
T_MAX = 5.0
N = int(T_MAX / DT)

# Desired end-effector position (constant)
X_D = np.array([0.6, 0.3])
XD_D = np.zeros(2)
XDD_D = np.zeros(2)

# Impedance gains (task space)
M_d = np.diag([2.0, 2.0])
B_d = np.diag([25.0, 25.0])
K_d = np.diag([120.0, 120.0])

q = np.array([0.3, 0.2, 0.1])
qd = np.zeros(3)

t_hist = []
x_hist = []
x_d_hist = []
q_hist = []

for i in range(N):
    t = i * DT
    x = forward_kinematics(q)
    xd = jacobian(q) @ qd
    J = jacobian(q)
    J_dot = jacobian_dot(q, qd)

    # F_impedance = M_d*(xdd_d - xdd) + B_d*(xd_d - xd) + K_d*(x_d - x)
    # Desired xdd from impedance: M_d @ xdd_des = M_d @ xdd_d + B_d*(xd_d - xd) + K_d*(x_d - x)
    xdd_des = XDD_D + np.linalg.solve(M_d, B_d @ (XD_D - xd) + K_d @ (X_D - x))

    # qdd_des such that J @ qdd_des + J_dot @ qd = xdd_des  =>  qdd_des = J_pinv @ (xdd_des - J_dot @ qd)
    J_pinv = np.linalg.pinv(J)
    qdd_des = J_pinv @ (xdd_des - J_dot @ qd)

    tau = M(q) @ qdd_des + C_vec(q, qd) + G(q)
    tau = np.clip(tau, -50.0, 50.0)

    t_hist.append(t)
    x_hist.append(x.copy())
    x_d_hist.append(X_D.copy())
    q_hist.append(q.copy())

    q, qd = rk4_step(q, qd, tau, DT)

x_hist = np.array(x_hist)
q_hist = np.array(q_hist)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(t_hist, x_hist[:, 0], label='x')
axes[0].plot(t_hist, x_hist[:, 1], label='y')
axes[0].axhline(X_D[0], color='gray', linestyle='--')
axes[0].axhline(X_D[1], color='gray', linestyle='--')
axes[0].set_ylabel('EE position (m)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Impedance: end-effector position')
for j in range(3):
    axes[1].plot(t_hist, q_hist[:, j], label=r'$q_{}$'.format(j+1))
axes[1].set_ylabel('Joint (rad)')
axes[1].set_xlabel('t (s)')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('impedance_result.png', dpi=120)
plt.show()
