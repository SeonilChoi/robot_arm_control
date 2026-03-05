"""
3-DOF 평면 로봇 팔 동역학 모듈 (numpy만 사용)

역할: 로봇의 운동방정식 M(q)q'' + C(q,q') + G(q) = tau 에 필요한
      관성행렬 M, 코리올리스/원심력 C, 중력 G 를 계산하고,
      주어진 tau로부터 가속도 q''를 구한 뒤 RK4로 시간적분합니다.

관절 각도 규약: q1은 수직 기준, q2는 링크1 기준, q3는 링크2 기준 (rad).
"""
import numpy as np

# ========== 로봇 물성 파라미터 ==========
# 링크 길이 (m)
L1, L2, L3 = 0.5, 0.4, 0.3
# 링크 질량 (kg)
M1, M2, M3 = 1.0, 0.8, 0.5
# 링크 관성모멘트 (kg·m^2, 회전축에 대한 것)
I1z, I2z, I3z = 0.01, 0.01, 0.005
# 중력가속도 (m/s^2)
GRAV = 9.81


def M(q):
    """
    관성 행렬 M(q) — 3x3 대칭 양정치 행렬.

    역할: 주어진 관절각 q에서 "가속도→힘" 변환의 질량 효과를 나타냄.
          tau = M*q'' + ... 에서 M*q'' 항에 해당.
    인자: q — (3,) 관절각 [q1, q2, q3] (rad)
    반환: (3,3) ndarray
    """
    q1, q2, q3 = q[0], q[1], q[2]
    # cos(q2), cos(q3), cos(q2+q3) — 수식 간소화용
    c2, c3 = np.cos(q2), np.cos(q3)
    c23 = np.cos(q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    i1, i2, i3 = I1z, I2z, I3z

    # 3링크 평면 로봇의 M(q) 성분 (라그랑주/뉴턴-오일러 유도식)
    I11 = (m1/4 + m2 + m3)*l1**2 + (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i1 + i2 + i3 \
          + (m2 + 2*m3)*l1*l2*c2 + m3*l3*(l1*c23 + l2*c3)
    I12 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 \
          + (m2/2 + m3)*l1*l2*c2 + (m3/2)*l3*(l1*c23 + 2*l2*c3)
    I13 = (m3/4)*l3**2 + i3 + (m3/2)*l3*(l1*c23 + l2*c3)
    I22 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 + m3*l2*l3*c3
    I23 = (m3/4)*l3**2 + i3 + (m3/2)*l2*l3*c3
    I33 = (m3/4)*l3**2 + i3

    return np.array([
        [I11, I12, I13],
        [I12, I22, I23],
        [I13, I23, I33]
    ], dtype=float)


def C_vec(q, qd):
    """
    코리올리스/원심력 벡터 C(q, qd).

    역할: 속도 qd에 의한 비선형 항. M*q'' + C_vec + G = tau 에서
          C_vec 은 q''가 아닌 q, qd의 함수로 관절 간 결합·원심력을 나타냄.
    인자: q  — (3,) 관절각
          qd — (3,) 관절 속도 (rad/s)
    반환: (3,) ndarray
    """
    q1, q2, q3 = q[0], q[1], q[2]
    qd1, qd2, qd3 = qd[0], qd[1], qd[2]
    s2, s3 = np.sin(q2), np.sin(q3)
    s23 = np.sin(q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m2, m3 = M2, M3

    c1 = -(m2/2 + m3)*l1*l2*s2*(2*qd1*qd2 + qd2**2) \
         - (m3/2)*l3*l1*s23*(qd2 + qd3)*(qd1 + qd2 + qd3) - (m3/2)*l3*l2*s3*qd3*(qd1 + qd2)
    c2 = (m2/2 + m3)*l1*l2*s2*qd1**2 + (m3/2)*l3*l1*s23*qd1**2 \
         - (m3/2)*l3*l2*s3*qd3*(2*qd1 + 2*qd2 + qd3)
    c3 = (m3/2)*l3*(l1*s23*qd1**2 + l2*s3*(qd1 + qd2)**2)

    return np.array([c1, c2, c3], dtype=float)


def G(q):
    """
    중력 벡터 G(q).

    역할: 중력에 의해 관절에 작용하는 토크. 평형에서 tau = G(q) 이면
          가속도 없이 그 자세를 유지할 수 있는 토크.
    인자: q — (3,) 관절각
    반환: (3,) ndarray (부호: tau = M*q'' + C + G 에서 G 항)
    """
    q1, q2, q3 = q[0], q[1], q[2]
    s1 = np.sin(q1)
    s12 = np.sin(q1 + q2)
    s123 = np.sin(q1 + q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    g = GRAV

    Gamma1 = (m1/2 + m2 + m3)*l1*s1 + (m2/2 + m3)*l2*s12 + (m3/2)*l3*s123
    Gamma2 = (m2/2 + m3)*l2*s12 + (m3/2)*l3*s123
    Gamma3 = (m3/2)*l3*s123
    return -g * np.array([Gamma1, Gamma2, Gamma3], dtype=float)


def forward_kinematics(q):
    """
    정기구학: 관절각 q → 엔드이펙터 위치 (2D).

    역할: 평면 3R 로봇에서 손끝 (x, y) 좌표 계산. 임피던스 제어 등에서 사용.
    인자: q — (3,) 관절각
    반환: (2,) ndarray [x, y] (m)
    """
    q1, q2, q3 = q[0], q[1], q[2]
    x = L1*np.cos(q1) + L2*np.cos(q1 + q2) + L3*np.cos(q1 + q2 + q3)
    y = L1*np.sin(q1) + L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3)
    return np.array([x, y], dtype=float)


def jacobian(q):
    """
    자코비안 J(q): 작업공간 속도와 관절 속도 연결.

    역할: dx/dt = J @ qd (엔드이펙터 속도 = J × 관절속도). 2x3 행렬.
    인자: q — (3,) 관절각
    반환: (2, 3) ndarray
    """
    q1, q2, q3 = q[0], q[1], q[2]
    s1, s12, s123 = np.sin(q1), np.sin(q1 + q2), np.sin(q1 + q2 + q3)
    c1, c12, c123 = np.cos(q1), np.cos(q1 + q2), np.cos(q1 + q2 + q3)
    J = np.array([
        [-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
        [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]
    ], dtype=float)
    return J


def jacobian_dot(q, qd):
    """
    자코비안 시간미분: dJ/dt = (dJ/dq)*qd.

    역할: 가속도 관계 d²x/dt² = J*q'' + (dJ/dt)*qd 에서 두 번째 항 계산.
    인자: q, qd — 관절각, 관절속도
    반환: (2, 3) ndarray
    """
    q1, q2, q3 = q[0], q[1], q[2]
    qd1, qd2, qd3 = qd[0], qd[1], qd[2]
    s1, s12, s123 = np.sin(q1), np.sin(q1 + q2), np.sin(q1 + q2 + q3)
    c1, c12, c123 = np.cos(q1), np.cos(q1 + q2), np.cos(q1 + q2 + q3)
    dJ_dq1 = np.array([
        [-L1*c1 - L2*c12 - L3*c123, -L2*c12 - L3*c123, -L3*c123],
        [-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123]
    ])
    dJ_dq2 = np.array([
        [-L2*c12 - L3*c123, -L2*c12 - L3*c123, -L3*c123],
        [-L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123]
    ])
    dJ_dq3 = np.array([
        [-L3*c123, -L3*c123, -L3*c123],
        [-L3*s123, -L3*s123, -L3*s123]
    ])
    J_dot = dJ_dq1*qd1 + dJ_dq2*qd2 + dJ_dq3*qd3
    return J_dot


def qdd_from_tau(q, qd, tau):
    """
    토크로부터 관절 가속도 계산: q'' = M^{-1}(tau - C - G).

    역할: 운동방정식을 q''에 대해 풀기. 시뮬레이션에서 한 스텝마다 호출.
    인자: q, qd — 현재 관절각·속도, tau — 인가 토크
    반환: (3,) ndarray 관절 가속도 (rad/s^2)
    """
    M_q = M(q)
    c_vec = C_vec(q, qd)
    g_vec = G(q)
    return np.linalg.solve(M_q, tau - c_vec - g_vec)


def rk4_step(q, qd, tau, dt):
    """
    RK4(4차 Runge-Kutta) 한 스텝: (q, qd) 와 tau, dt 로 다음 (q_new, qd_new) 계산.

    역할: 연속시간 동역학을 이산시간으로 적분. 상태 [q; qd] 의 미분을
          qd 와 qdd 로 나타내고, RK4로 dt 만큼 진행.
    인자: q, qd — (3,) 현재 관절각·속도
          tau  — (3,) 이 스텝 동안 일정하다고 가정하는 토크
          dt   — 시간 스텝 (s)
    반환: (q_new, qd_new) 각 (3,) ndarray
    """
    def deriv(state):
        q_now = state[:3]
        qd_now = state[3:6]
        qdd = qdd_from_tau(q_now, qd_now, tau)
        return np.concatenate([qd_now, qdd])

    state = np.concatenate([q, qd])
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    state_new = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return state_new[:3], state_new[3:6]
