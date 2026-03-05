"""
MPC: receding horizon, unconstrained QP.
z_{k+1} = A_d z_k + B_d u_k. Cost = sum_{k=0}^{N-1} (|z_k - z_r|_Q^2 + |u_k|_R^2) + |z_N - z_r|_{Q_N}^2.
Solve for U = [u_0; ...; u_{N-1}], apply u_0 only.
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import G, rk4_step, linearize

# Reference (constant)
Q_REF = np.array([0.5, 0.4, 0.3])
Z_REF = np.concatenate([Q_REF, np.zeros(3)])

Q_E = Q_REF
A, B = linearize(Q_E)
DT_MPC = 0.05
A_d = np.eye(6) + A * DT_MPC
B_d = B * DT_MPC

N_horizon = 15
Q_mpc = np.diag([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
R_mpc = np.diag([0.5, 0.5, 0.5])
QN = np.diag([100.0, 100.0, 100.0, 20.0, 20.0, 20.0])

# Build prediction: Z = Phi z_0 + Psi U  (Z = [z_1; z_2; ...; z_N], U = [u_0; ...; u_{N-1}])
def build_psi(Ad, Bd, N):
    nz, nu = Bd.shape
    Psi = np.zeros((N * nz, N * nu))
    for i in range(N):
        for j in range(i + 1):
            Psi[i*nz:(i+1)*nz, j*nu:(j+1)*nu] = np.linalg.matrix_power(Ad, i - j) @ Bd
    return Psi

def build_phi(Ad, N):
    nz = Ad.shape[0]
    Phi = np.zeros((N * nz, nz))
    for i in range(N):
        Phi[i*nz:(i+1)*nz, :] = np.linalg.matrix_power(Ad, i + 1)
    return Phi

Psi = build_psi(A_d, B_d, N_horizon)
Phi = build_phi(A_d, N_horizon)
nz, nu = 6, 3
Q_bar = np.zeros((N_horizon * nz, N_horizon * nz))
for i in range(N_horizon - 1):
    Q_bar[i*nz:(i+1)*nz, i*nz:(i+1)*nz] = Q_mpc
Q_bar[(N_horizon-1)*nz:, (N_horizon-1)*nz:] = QN
R_bar = np.kron(np.eye(N_horizon), R_mpc)

H = 2 * (Psi.T @ Q_bar @ Psi + R_bar)
H = 0.5 * (H + H.T)

DT = 0.002
T_MAX = 4.0
N_sim = int(T_MAX / DT)
mpc_interval = max(1, int(DT_MPC / DT))

q = np.array([0.0, 0.0, 0.0])
qd = np.zeros(3)
t_hist = []
q_hist = []
tau_hist = []

for i in range(N_sim):
    t = i * DT
    z = np.concatenate([q, qd])
    Z_ref_stack = np.tile(Z_REF, N_horizon)
    c = 2 * Psi.T @ Q_bar @ (Phi @ z - Z_ref_stack)
    U = np.linalg.solve(H, -c)
    u0 = U[:nu]
    tau = G(Q_REF) + u0
    tau = np.clip(tau, -50.0, 50.0)

    t_hist.append(t)
    q_hist.append(q.copy())
    tau_hist.append(tau.copy())

    q, qd = rk4_step(q, qd, tau, DT)

q_hist = np.array(q_hist)
fig, ax = plt.subplots(figsize=(8, 4))
for j in range(3):
    ax.plot(t_hist, q_hist[:, j], label=r'$q_{}$'.format(j+1))
    ax.axhline(Q_REF[j], color='gray', linestyle='--')
ax.set_ylabel('q (rad)')
ax.set_xlabel('t (s)')
ax.legend()
ax.grid(True)
ax.set_title('MPC: joint angles')
plt.tight_layout()
plt.savefig('mpc_result.png', dpi=120)
plt.show()
