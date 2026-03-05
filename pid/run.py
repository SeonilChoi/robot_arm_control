"""
동역학 기반 PID(Computed Torque) 제어 시뮬레이션

공식: tau = M(q)*u + C(q,qd) + G(q),  u = qdd_d + Kp*e + Kd*edot + Ki*e_int
- e = q_d - q (각도 오차), edot = qd_d - qd (속도 오차), e_int = integral(e) (적분 오차)
- u를 "원하는 가속도"로 두고, 동역학을 이용해 그 가속도를 만들 토크 tau를 계산

실험: 외란 없음 / [1.5, 2.5]초 구간에 상수 토크 외란 추가 — 두 궤적을 비교
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import M, C_vec, G, rk4_step

# ========== 시뮬레이션 시간 설정 ==========
DT = 0.002   # 적분 시간 간격 (s). 작을수록 정확, 느려짐
T_MAX = 5.0  # 총 시뮬레이션 시간 (s)
N = int(T_MAX / DT)  # 스텝 수

# ========== 목표 궤적 (상수 setpoint) ==========
Q_D = np.array([0.5, 0.4, 0.3])   # 목표 관절각 (rad)
QD_D = np.zeros(3)                 # 목표 속도 (정지 목표이므로 0)
QDD_D = np.zeros(3)                # 목표 가속도 (0)

# ========== PID 이득 (대각행렬: 관절별 독립 튜닝) ==========
# Kp: 비례 — 오차가 크면 u가 커져 빠르게 수렴 (너무 크면 진동)
# Kd: 미분 — 속도 오차로 댐핑, 진동 억제
# Ki: 적분 — 정상상태 오차(외란) 제거
Kp = np.diag([80.0, 60.0, 40.0])
Kd = np.diag([20.0, 15.0, 10.0])
Ki = np.diag([5.0, 3.0, 2.0])

# ========== 제약 ==========
TAU_LIM = 50.0   # 관절 토크 한계 (N·m). 초과분은 clip
EINT_LIM = 2.0   # 적분항 e_int 의 각 성분 상한 (적분 wind-up 방지)

# ========== 외란 설정 ==========
# 아래 구간에서 tau에 TAU_DIST 를 더해 인가 (외란 실험)
T_DIST_START, T_DIST_END = 1.5, 2.5
TAU_DIST = np.array([5.0, -2.0, 1.0])  # (N·m) 관절별 상수 외란


def simulate(use_disturbance):
    """
    한 번의 시뮬레이션 수행.

    인자: use_disturbance — True 이면 [T_DIST_START, T_DIST_END] 구간에 TAU_DIST 추가
    반환: t_hist (시간 배열), q_hist (관절각 이력, (N,3))
    """
    # 초기 상태
    q = np.array([0.0, 0.0, 0.0])
    qd = np.zeros(3)
    e_int = np.zeros(3)  # 오차 적분 (적분기 상태)
    t_hist, q_hist = [], []

    for i in range(N):
        t = i * DT

        # ----- 1) 오차 계산 -----
        e = Q_D - q              # 각도 오차 e = q_d - q
        edot = QD_D - qd         # 속도 오차
        e_int = e_int + e * DT   # 적분 (오차 누적)
        e_int = np.clip(e_int, -EINT_LIM, EINT_LIM)  # wind-up 방지

        # ----- 2) 원하는 가속도 u (PID 출력) -----
        # u = qdd_d + Kp*e + Kd*edot + Ki*e_int
        u = QDD_D + Kp @ e + Kd @ edot + Ki @ e_int

        # ----- 3) Computed Torque: tau = M*u + C + G -----
        tau = M(q) @ u + C_vec(q, qd) + G(q)
        tau = np.clip(tau, -TAU_LIM, TAU_LIM)

        # ----- 4) 외란 실험: 구간 안이면 tau에 외란 가산 -----
        if use_disturbance and T_DIST_START <= t <= T_DIST_END:
            tau = tau + TAU_DIST
            tau = np.clip(tau, -TAU_LIM, TAU_LIM)

        t_hist.append(t)
        q_hist.append(q.copy())

        # ----- 5) 동역학 적분 (다음 스텝 q, qd) -----
        q, qd = rk4_step(q, qd, tau, DT)

    return t_hist, np.array(q_hist)


# 무외란 / 유외란 두 번 시뮬레이션
t_hist, q_no = simulate(False)
_, q_with = simulate(True)

# ========== 그래프 그리기 ==========
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for j in range(3):
    axes[j].plot(t_hist, q_no[:, j], '-', label='no dist' if j == 0 else None)
    axes[j].plot(t_hist, q_with[:, j], '--', label='with dist' if j == 0 else None)
    axes[j].axhline(Q_D[j], color='gray', linestyle=':', label=r'$q_{d,' + str(j+1) + '}$' if j == 0 else None)
    axes[j].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')  # 외란 구간 표시
    axes[j].set_ylabel(r'$q_{}$ (rad)'.format(j+1))
    if j == 0:
        axes[j].legend(loc='right')
    axes[j].grid(True)
axes[0].set_title('PID: joint angles (red zone = disturbance on)')
axes[-1].set_xlabel('t (s)')
plt.tight_layout()
plt.savefig('pid_result.png', dpi=120)
plt.show()
