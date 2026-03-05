"""
MPC(Model Predictive Control) 시뮬레이션

개념: 선형 이산시간 z_{k+1}=A_d*z_k+B_d*u_k 로 N 스텝 예측,
      비용 sum(|z_k - z_r|_Q^2 + |u_k|_R^2) + |z_N - z_r|_QN^2 최소화하는 U=[u_0;..;u_{N-1}] 를
      제약 없는 QP로 풀고, 첫 입력 u_0 만 적용. 매 스텝 반복(재ceding horizon).

변수: Z = Phi*z_0 + Psi*U 로 예측 궤적을 U에 대해 선형 표현. 비용을 U에 대한 2차식으로 정리해
      H, c 로 QP min (1/2)U'*H*U + c'*U → U* = -H^{-1}*c, tau = G(q_r) + u_0.
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import G, rk4_step, linearize

# ========== 목표 상태 (상수) ==========
Q_REF = np.array([0.5, 0.4, 0.3])
Z_REF = np.concatenate([Q_REF, np.zeros(3)])  # [q_r; 0]

# ========== 선형화 및 이산화 ==========
Q_E = Q_REF
A, B = linearize(Q_E)
DT_MPC = 0.05   # MPC 예측용 이산화 시간
A_d = np.eye(6) + A * DT_MPC   # 오일러: A_d ≈ I + A*dt
B_d = B * DT_MPC

# ========== MPC 파라미터 ==========
N_horizon = 15   # 예측 구간 스텝 수
Q_mpc = np.diag([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])  # 상태 가중치 (q, qd)
R_mpc = np.diag([0.5, 0.5, 0.5])   # 입력 가중치
QN = np.diag([100.0, 100.0, 100.0, 20.0, 20.0, 20.0])  # 종단 가중치 (z_N)

# ========== 예측 행렬 구성: Z = Phi*z_0 + Psi*U ==========
# Z = [z_1; z_2; ...; z_N], U = [u_0; u_1; ...; u_{N-1}]
# z_{i+1} = A_d*z_i + B_d*u_i 이므로 z_i 에 u_0..u_{i-1} 이 선형으로 들어감 → Psi
def build_psi(Ad, Bd, N):
    """Psi: (N*nz, N*nu). Z = Phi*z_0 + Psi*U 에서 입력 U의 계수 행렬."""
    nz, nu = Bd.shape
    Psi = np.zeros((N * nz, N * nu))
    for i in range(N):
        for j in range(i + 1):
            # z_{i+1} 에서 u_j 의 계수 = A_d^{i-j} * B_d
            Psi[i*nz:(i+1)*nz, j*nu:(j+1)*nu] = np.linalg.matrix_power(Ad, i - j) @ Bd
    return Psi

def build_phi(Ad, N):
    """Phi: (N*nz, nz). Z = Phi*z_0 + Psi*U 에서 초기 상태 z_0 의 계수."""
    nz = Ad.shape[0]
    Phi = np.zeros((N * nz, nz))
    for i in range(N):
        Phi[i*nz:(i+1)*nz, :] = np.linalg.matrix_power(Ad, i + 1)  # z_{i+1} = A_d^{i+1}*z_0 + ...
    return Phi

Psi = build_psi(A_d, B_d, N_horizon)
Phi = build_phi(A_d, N_horizon)
nz, nu = 6, 3

# 비용: (Z - Z_ref)'*Q_bar*(Z - Z_ref) + U'*R_bar*U
# Q_bar: 블록 대각 diag(Q_mpc,...,Q_mpc, QN), R_bar: diag(R_mpc,...,R_mpc)
Q_bar = np.zeros((N_horizon * nz, N_horizon * nz))
for i in range(N_horizon - 1):
    Q_bar[i*nz:(i+1)*nz, i*nz:(i+1)*nz] = Q_mpc
Q_bar[(N_horizon-1)*nz:, (N_horizon-1)*nz:] = QN
R_bar = np.kron(np.eye(N_horizon), R_mpc)

# QP: min (1/2) U'*H*U + c'*U  →  H = 2*(Psi'*Q_bar*Psi + R_bar), c = 2*Psi'*Q_bar*(Phi*z_0 - Z_ref_stack)
H = 2 * (Psi.T @ Q_bar @ Psi + R_bar)
H = 0.5 * (H + H.T)  # 대칭 보정

# ========== 시뮬레이션 ==========
DT = 0.002
T_MAX = 4.0
N_sim = int(T_MAX / DT)
TAU_LIM = 50.0
T_DIST_START, T_DIST_END = 1.5, 2.5
TAU_DIST = np.array([5.0, -2.0, 1.0])


def simulate(use_disturbance):
    q = np.array([0.0, 0.0, 0.0])
    qd = np.zeros(3)
    t_hist, q_hist = [], []

    for i in range(N_sim):
        t = i * DT
        z = np.concatenate([q, qd])
        Z_ref_stack = np.tile(Z_REF, N_horizon)  # [z_r; z_r; ...; z_r]
        c = 2 * Psi.T @ Q_bar @ (Phi @ z - Z_ref_stack)
        U = np.linalg.solve(H, -c)   # U* = -H^{-1}*c
        u0 = U[:nu]                  # 첫 스텝 입력만 적용
        tau = G(Q_REF) + u0          # u = tau - G(q_r) 이므로 tau = G(q_r) + u0
        tau = np.clip(tau, -TAU_LIM, TAU_LIM)
        if use_disturbance and T_DIST_START <= t <= T_DIST_END:
            tau = tau + TAU_DIST
            tau = np.clip(tau, -TAU_LIM, TAU_LIM)
        t_hist.append(t)
        q_hist.append(q.copy())
        q, qd = rk4_step(q, qd, tau, DT)
    return t_hist, np.array(q_hist)


t_hist, q_no = simulate(False)
_, q_with = simulate(True)

fig, ax = plt.subplots(figsize=(8, 4))
for j in range(3):
    ax.plot(t_hist, q_no[:, j], '-', label='no dist' if j == 0 else None)
    ax.plot(t_hist, q_with[:, j], '--', label='with dist' if j == 0 else None)
    ax.axhline(Q_REF[j], color='gray', linestyle=':')
ax.axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
ax.set_ylabel('q (rad)')
ax.set_xlabel('t (s)')
ax.legend()
ax.grid(True)
ax.set_title('MPC: joint angles (red zone = disturbance)')
plt.tight_layout()
plt.savefig('mpc_result.png', dpi=120)
plt.show()
