"""
QP 기반 관절각 추종 시뮬레이션

개념: PD형 목표 가속도 b = qdd_d + Kp*(q_d - q) + Kd*(qd_d - qd) 를 두고,
      QP min_a (1/2)|b - a|^2 + (lambda/2)|a|^2 로 a* 를 구함 (정규화로 a 완만하게).
      tau = M(q)*a* + C + G. 나중에 토크/가속도 제약을 넣으면 같은 QP 틀에 부등식 추가 가능.

실험: 무외란 / 1.5~2.5초 외란 두 궤적 비교.
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import M, C_vec, G, rk4_step

# ========== 시간 ==========
DT = 0.002
T_MAX = 5.0
N = int(T_MAX / DT)

# ========== 목표 ==========
Q_D = np.array([0.5, 0.4, 0.3])
QD_D = np.zeros(3)
QDD_D = np.zeros(3)

# ========== PD 이득 및 QP 정규화 ==========
# b = qdd_d + Kp*e + Kd*edot 에서 Kp, Kd. a* = (1+lambda)^{-1}*b 이므로 lambda 크면 a* 완만
Kp = np.diag([60.0, 50.0, 40.0])
Kd = np.diag([18.0, 14.0, 10.0])
lam = 0.1   # 정규화: min (1/2)|b-a|^2 + (lambda/2)|a|^2 → a* = b/(1+lambda)

TAU_LIM = 50.0
T_DIST_START, T_DIST_END = 1.5, 2.5
TAU_DIST = np.array([5.0, -2.0, 1.0])


def simulate(use_disturbance):
    """한 번 시뮬레이션. 반환: t_hist, q_hist, e_hist."""
    q = np.array([0.0, 0.0, 0.0])
    qd = np.zeros(3)
    t_hist, q_hist, e_hist = [], [], []

    for i in range(N):
        t = i * DT
        e = Q_D - q
        edot = QD_D - qd
        # PD형 목표 가속도
        b = QDD_D + Kp @ e + Kd @ edot

        # QP: min_a (1/2)(b-a)'(b-a) + (lambda/2)a'a = (1/2)a'Ha + c'a, H=(1+lambda)I, c=-b
        # 해: a* = -H^{-1}*c = b/(1+lambda)
        H_qp = (1.0 + lam) * np.eye(3)
        c_qp = -b
        a_star = np.linalg.solve(H_qp, -c_qp)

        tau = M(q) @ a_star + C_vec(q, qd) + G(q)
        tau = np.clip(tau, -TAU_LIM, TAU_LIM)
        if use_disturbance and T_DIST_START <= t <= T_DIST_END:
            tau = tau + TAU_DIST
            tau = np.clip(tau, -TAU_LIM, TAU_LIM)
        t_hist.append(t)
        q_hist.append(q.copy())
        e_hist.append(e.copy())
        q, qd = rk4_step(q, qd, tau, DT)
    return t_hist, np.array(q_hist), np.array(e_hist)


t_hist, q_no, e_no = simulate(False)
_, q_with, e_with = simulate(True)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for j in range(3):
    axes[0].plot(t_hist, q_no[:, j], '-', label='no dist' if j == 0 else None)
    axes[0].plot(t_hist, q_with[:, j], '--', label='with dist' if j == 0 else None)
    axes[0].axhline(Q_D[j], color='gray', linestyle=':')
axes[0].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
axes[0].set_ylabel('q (rad)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('QP: joint angles (red zone = disturbance)')
for j in range(3):
    axes[1].plot(t_hist, e_no[:, j], '-', label='no dist' if j == 0 else None)
    axes[1].plot(t_hist, e_with[:, j], '--', label='with dist' if j == 0 else None)
axes[1].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
axes[1].set_ylabel('error (rad)')
axes[1].set_xlabel('t (s)')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('qp_result.png', dpi=120)
plt.show()
