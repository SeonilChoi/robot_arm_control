"""
LQR 제어 시뮬레이션

개념: 평형점 z_e = [q_e; 0], tau_e = G(q_e) 주변에서 선형화 dz/dt = A*z + B*u,
      u = tau - tau_e. LQR로 u = -K*(z - z_e) 이득 K 를 구함 (CARE 해결).
      tau = tau_e + u = G(q_e) - K*(z - z_e).

실험: 무외란 / 1.5~2.5초 구간 외란 두 궤적 비교.
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import G, rk4_step, linearize

# ========== 평형점 (목표로 유지하고 싶은 자세) ==========
Q_E = np.array([0.4, 0.3, 0.2])   # 평형 관절각 (rad)
TAU_E = G(Q_E)                     # 평형에서 필요한 토크 (중력 보상)
Z_E = np.concatenate([Q_E, np.zeros(3)])  # 평형 상태 벡터 [q_e; 0]

# ========== 선형화 및 LQR 이득 ==========
A, B = linearize(Q_E)  # dz/dt = A*z + B*u

# LQR 가중치: Q는 상태 오차 비용, R은 입력(토크) 비용. R이 너무 작으면 이득이 커져 속도 진동/불안정
Q_lqr = np.diag([30.0, 30.0, 30.0, 3.0, 3.0, 3.0])  # [q1,q2,q3, qd1,qd2,qd3]
R_lqr = np.diag([2.0, 2.0, 2.0])   # R을 키워 이득 완화 → 속도 진동 감소


def solve_care(A, B, Q, R):
    """
    연속시간 대수 Riccati 방정식(CARE) 해결: A'P + PA - P B R^{-1} B' P + Q = 0.
    해밀토니안 행렬의 고유벡터로 P를 구하고, K = R^{-1} B' P.

    반환: K (이득 3x6), P (Riccati 해 6x6)
    """
    Rinv = np.linalg.inv(R)
    H = np.block([
        [A, -B @ Rinv @ B.T],
        [-Q, -A.T]
    ])
    eigvals, eigvecs = np.linalg.eig(H)
    n = A.shape[0]
    # 안정 고유값(실부 < 0)만 선택. 해밀토니안은 실부<0 n개, 실부>0 n개 쌍으로 존재
    stable_mask = np.real(eigvals) < 0
    stable_idx = np.where(stable_mask)[0]
    if len(stable_idx) != n:
        # fallback: 실부가 작은 순으로 n개
        stable_idx = np.argsort(np.real(eigvals))[:n]
    X = eigvecs[:n, stable_idx].real
    Y = eigvecs[n:, stable_idx].real
    P = Y @ np.linalg.solve(X.T, np.eye(n)).T
    P = 0.5 * (P + P.T)  # 대칭 보정
    K = Rinv @ B.T @ P
    return K, P


K_lqr, _ = solve_care(A, B, Q_lqr, R_lqr)

# ========== 시뮬레이션 ==========
DT = 0.002
T_MAX = 4.0
N = int(T_MAX / DT)
TAU_LIM = 50.0
T_DIST_START, T_DIST_END = 1.5, 2.5
TAU_DIST = np.array([5.0, -2.0, 1.0])


def simulate(use_disturbance):
    """초기 상태는 평형에서 살짝 벗어남. LQR이 다시 평형으로 끌어당김."""
    q = np.array([0.5, 0.35, 0.25])
    qd = np.array([0.1, 0.0, 0.0])
    t_hist, q_hist, qd_hist = [], [], []

    for i in range(N):
        t = i * DT
        z = np.concatenate([q, qd])
        u = -K_lqr @ (z - Z_E)   # LQR: u = -K*(z - z_e)
        tau = TAU_E + u          # tau = tau_e + u
        tau = np.clip(tau, -TAU_LIM, TAU_LIM)
        if use_disturbance and T_DIST_START <= t <= T_DIST_END:
            tau = tau + TAU_DIST
            tau = np.clip(tau, -TAU_LIM, TAU_LIM)

        t_hist.append(t)
        q_hist.append(q.copy())
        qd_hist.append(qd.copy())
        q, qd = rk4_step(q, qd, tau, DT)

    return t_hist, np.array(q_hist), np.array(qd_hist)


t_hist, q_no, qd_no = simulate(False)
_, q_with, qd_with = simulate(True)

# 플롯
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for j in range(3):
    axes[0].plot(t_hist, q_no[:, j], '-', label='no dist' if j == 0 else None)
    axes[0].plot(t_hist, q_with[:, j], '--', label='with dist' if j == 0 else None)
    axes[0].axhline(Q_E[j], color='gray', linestyle=':')
axes[0].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
axes[0].set_ylabel('q (rad)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('LQR: joint angles (red zone = disturbance)')
for j in range(3):
    axes[1].plot(t_hist, qd_no[:, j], '-', label=r'$\dot{q}_%d$' % (j+1))
    axes[1].plot(t_hist, qd_with[:, j], '--')
axes[1].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
axes[1].set_ylabel(r'$\dot{q}$ (rad/s)')
axes[1].set_xlabel('t (s)')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('lqr_result.png', dpi=120)
plt.show()
