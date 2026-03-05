"""
Dynamics-based PID (Computed Torque) control for 3-DOF planar arm.
tau = M(q)*u + C_vec(q,qd) + G(q),  u = qdd_d + Kp*e + Kd*edot + Ki*e_int
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import M, C_vec, G, rk4_step

# Simulation
DT = 0.002
T_MAX = 5.0
N = int(T_MAX / DT)

# Desired (constant setpoint)
Q_D = np.array([0.5, 0.4, 0.3])
QD_D = np.zeros(3)
QDD_D = np.zeros(3)

# Gains
Kp = np.diag([80.0, 60.0, 40.0])
Kd = np.diag([20.0, 15.0, 10.0])
Ki = np.diag([5.0, 3.0, 2.0])

# State
q = np.array([0.0, 0.0, 0.0])
qd = np.zeros(3)
e_int = np.zeros(3)

t_hist = []
q_hist = []
qd_hist = []
e_hist = []
tau_hist = []

for i in range(N):
    t = i * DT
    e = Q_D - q
    edot = QD_D - qd
    e_int = e_int + e * DT
    e_int = np.clip(e_int, -2.0, 2.0)

    # u = qdd_d + Kp*e + Kd*edot + Ki*e_int
    u = QDD_D + Kp @ e + Kd @ edot + Ki @ e_int

    # tau = M(q)*u + C_vec(q,qd) + G(q)
    tau = M(q) @ u + C_vec(q, qd) + G(q)
    tau = np.clip(tau, -50.0, 50.0)

    t_hist.append(t)
    q_hist.append(q.copy())
    e_hist.append(e.copy())
    tau_hist.append(tau.copy())

    q, qd = rk4_step(q, qd, tau, DT)

q_hist = np.array(q_hist)
e_hist = np.array(e_hist)
tau_hist = np.array(tau_hist)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for j in range(3):
    axes[j].plot(t_hist, q_hist[:, j], label=r'$q_{}$'.format(j+1))
    axes[j].axhline(Q_D[j], color='gray', linestyle='--', label=r'$q_{d,' + str(j+1) + '}$')
    axes[j].set_ylabel(r'$q_{}$ (rad)'.format(j+1))
    axes[j].legend(loc='right')
    axes[j].grid(True)
axes[0].set_title('Joint angles (PID)')
axes[-1].set_xlabel('t (s)')
plt.tight_layout()
plt.savefig('pid_result.png', dpi=120)
plt.show()
