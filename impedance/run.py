"""
임피던스 제어 시뮬레이션 (작업공간)

개념: 목표 EE 위치 x_d 에 대해 가상 스프링·댐퍼·질량(M_d, B_d, K_d)을 두고,
      원하는 EE 가속도 xdd_des = xdd_d + M_d^{-1}[ B_d*(xd_d - xd) + K_d*(x_d - x) ] 를 계산한 뒤,
      qdd_des = J†(xdd_des - J_dot*qd),  tau = M*qdd_des + C + G 로 토크 계산.

실험: 무외란 / 유외란(1.5~2.5초) 두 궤적 비교.
"""
import numpy as np
import matplotlib.pyplot as plt
from dynamics import (
    M, C_vec, G, forward_kinematics, jacobian, jacobian_dot,
    rk4_step
)

# 시간 설정
DT = 0.002
T_MAX = 5.0
N = int(T_MAX / DT)

# 목표: 엔드이펙터 위치·속도·가속도 (작업공간)
X_D = np.array([0.6, 0.3])   # 목표 [x, y] (m)
XD_D = np.zeros(2)
XDD_D = np.zeros(2)

# 임피던스 이득 (작업공간에서의 가상 질량·댐핑·강성). 대각으로 설정.
# M_d: 작을수록 반응 빠름. B_d: 댐핑(진동 억제). K_d: 위치 오차에 대한 강성
M_d = np.diag([2.0, 2.0])
B_d = np.diag([25.0, 25.0])
K_d = np.diag([120.0, 120.0])

TAU_LIM = 50.0
T_DIST_START, T_DIST_END = 1.5, 2.5
TAU_DIST = np.array([5.0, -2.0, 1.0])


def simulate(use_disturbance):
    """한 번 시뮬레이션. 반환: t_hist, x_hist (EE 위치), q_hist."""
    q = np.array([0.3, 0.2, 0.1])
    qd = np.zeros(3)
    t_hist, x_hist, q_hist = [], [], []

    for i in range(N):
        t = i * DT
        # 현재 EE 위치·속도
        x = forward_kinematics(q)
        xd = jacobian(q) @ qd
        J = jacobian(q)
        J_dot = jacobian_dot(q, qd)

        # 원하는 EE 가속도 (임피던스 법칙): xdd_des = xdd_d + M_d^{-1}[ B_d*(xd_d-xd) + K_d*(x_d-x) ]
        xdd_des = XDD_D + np.linalg.solve(M_d, B_d @ (XD_D - xd) + K_d @ (X_D - x))

        # 관절 가속도로 변환: J*qdd_des + J_dot*qd = xdd_des  =>  qdd_des = J†(xdd_des - J_dot*qd)
        J_pinv = np.linalg.pinv(J)  # 유사역행렬 (2x3 → 3x2)
        qdd_des = J_pinv @ (xdd_des - J_dot @ qd)

        # 역동역학: tau = M*qdd_des + C + G
        tau = M(q) @ qdd_des + C_vec(q, qd) + G(q)
        tau = np.clip(tau, -TAU_LIM, TAU_LIM)
        if use_disturbance and T_DIST_START <= t <= T_DIST_END:
            tau = tau + TAU_DIST
            tau = np.clip(tau, -TAU_LIM, TAU_LIM)

        t_hist.append(t)
        x_hist.append(x.copy())
        q_hist.append(q.copy())
        q, qd = rk4_step(q, qd, tau, DT)

    return t_hist, np.array(x_hist), np.array(q_hist)


t_hist, x_no, q_no = simulate(False)
_, x_with, q_with = simulate(True)

# 플롯: 위 EE (x,y), 아래 관절각
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(t_hist, x_no[:, 0], '-', label='x (no dist)')
axes[0].plot(t_hist, x_no[:, 1], '-', label='y (no dist)')
axes[0].plot(t_hist, x_with[:, 0], '--', label='x (with dist)')
axes[0].plot(t_hist, x_with[:, 1], '--', label='y (with dist)')
axes[0].axhline(X_D[0], color='gray', linestyle=':')
axes[0].axhline(X_D[1], color='gray', linestyle=':')
axes[0].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
axes[0].set_ylabel('EE position (m)')
axes[0].legend(loc='right')
axes[0].grid(True)
axes[0].set_title('Impedance: end-effector (red zone = disturbance)')
for j in range(3):
    axes[1].plot(t_hist, q_no[:, j], '-', label='no dist' if j == 0 else None)
    axes[1].plot(t_hist, q_with[:, j], '--', label='with dist' if j == 0 else None)
axes[1].axvspan(T_DIST_START, T_DIST_END, alpha=0.15, color='red')
axes[1].set_ylabel('Joint (rad)')
axes[1].set_xlabel('t (s)')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('impedance_result.png', dpi=120)
plt.show()
