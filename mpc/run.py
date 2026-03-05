"""
MPC 스타일 목표 추종 (LQR 이득 사용)

원래 MPC: 선형 이산시간으로 N 스텝 예측 후 QP로 U*를 구해 u_0만 적용.
현재: 선형화가 목표 q_r=[0.5,0.4,0.3] 근처에서 개방루프 불안정(양의 고유값)이라
      예측 기반 QP가 비정상 동작하므로, 동일 (A,B,Q,R)로 LQR 이득 K를 구해
      u = -K*(z - z_r), tau = G(q_r) + u 로 적용해 목표 추종을 구현.
      (같은 Q,R을 쓰는 LQR은 불안정한 A도 폐루프 안정화함.)
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import G, rk4_step, linearize

# ========== 목표 상태 (상수) ==========
Q_REF = np.array([0.5, 0.4, 0.3])
Z_REF = np.concatenate([Q_REF, np.zeros(3)])  # [q_r; 0]

# ========== 선형화 ==========
# 참고: q_r = [0.5, 0.4, 0.3] 근처 선형화는 개방루프 불안정(양의 고유값).
# 완전 MPC 예측이 불안정해지므로, 여기서는 동일 (A,B,Q,R) 로 LQR 이득 K를 구해
# u = -K*(z - z_r), tau = G(q_r) + u 로 적용 (안정적 목표 추종).
Q_E = Q_REF
A, B = linearize(Q_E)
Q_mpc = np.diag([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
R_mpc = np.diag([0.5, 0.5, 0.5])
nz, nu = 6, 3

# 정상오차 제거: q3가 0.3이 아닌 잘못된 균형에 수렴하므로 적분항 추가 (Ki * int(q - q_ref))
K_i = np.array([1.2, 1.2, 28.0])  # q3 정상오차 제거


def solve_care(A, B, Q, R):
    """CARE 해결 → K = R^{-1} B' P. LQR 이득으로 목표 추종."""
    Rinv = np.linalg.inv(R)
    H = np.block([[A, -B @ Rinv @ B.T], [-Q, -A.T]])
    eigvals, eigvecs = np.linalg.eig(H)
    n = A.shape[0]
    stable_mask = np.real(eigvals) < 0
    stable_idx = np.where(stable_mask)[0]
    if len(stable_idx) != n:
        stable_idx = np.argsort(np.real(eigvals))[:n]
    X = eigvecs[:n, stable_idx].real
    Y = eigvecs[n:, stable_idx].real
    P = Y @ np.linalg.solve(X.T, np.eye(n)).T
    P = 0.5 * (P + P.T)
    K = Rinv @ B.T @ P
    return K


K_mpc = solve_care(A, B, Q_mpc, R_mpc)

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
    int_q = np.zeros(3)  # integral of (q - Q_REF)
    t_hist, q_hist = [], []

    for i in range(N_sim):
        t = i * DT
        int_q = int_q + (q - Q_REF) * DT
        int_q = np.clip(int_q, -4.0, 4.0)  # anti-windup
        z = np.concatenate([q, qd])
        u0 = -K_mpc @ (z - Z_REF) - K_i * int_q   # LQR + 적분 보정
        tau = G(Q_REF) + u0
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
