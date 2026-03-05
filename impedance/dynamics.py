"""
3-DOF 평면 로봇 팔 동역학 + 기구학 모듈 (임피던스 제어용)

역할: M(q), C(q,qd), G(q) 계산, 정기구학(엔드이펙터 위치), 자코비안 J 및 J_dot,
      tau → q'' 역산, RK4 적분. 임피던스 제어는 작업공간(x,y)에서 목표를 주므로
      forward_kinematics, jacobian, jacobian_dot 가 필요합니다.
"""
import numpy as np

# 로봇 파라미터 (길이 m, 질량 kg, 관성모멘트 kg·m^2, 중력 m/s^2)
L1, L2, L3 = 0.5, 0.4, 0.3
M1, M2, M3 = 1.0, 0.8, 0.5
I1z, I2z, I3z = 0.01, 0.01, 0.005
GRAV = 9.81


def M(q):
    """관성 행렬 M(q) 3x3. tau = M*q'' + C + G 의 M 항."""
    q1, q2, q3 = q[0], q[1], q[2]
    c2, c3 = np.cos(q2), np.cos(q3)
    c23 = np.cos(q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    i1, i2, i3 = I1z, I2z, I3z
    I11 = (m1/4 + m2 + m3)*l1**2 + (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i1 + i2 + i3 \
          + (m2 + 2*m3)*l1*l2*c2 + m3*l3*(l1*c23 + l2*c3)
    I12 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 \
          + (m2/2 + m3)*l1*l2*c2 + (m3/2)*l3*(l1*c23 + 2*l2*c3)
    I13 = (m3/4)*l3**2 + i3 + (m3/2)*l3*(l1*c23 + l2*c3)
    I22 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 + m3*l2*l3*c3
    I23 = (m3/4)*l3**2 + i3 + (m3/2)*l2*l3*c3
    I33 = (m3/4)*l3**2 + i3
    return np.array([
        [I11, I12, I13], [I12, I22, I23], [I13, I23, I33]
    ], dtype=float)


def C_vec(q, qd):
    """코리올리스/원심력 벡터 (3,). M*q'' + C_vec + G = tau."""
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
    """중력 벡터 (3,). 평형에서 tau=G(q)면 유지."""
    q1, q2, q3 = q[0], q[1], q[2]
    s1, s12 = np.sin(q1), np.sin(q1 + q2)
    s123 = np.sin(q1 + q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    Gamma1 = (m1/2 + m2 + m3)*l1*s1 + (m2/2 + m3)*l2*s12 + (m3/2)*l3*s123
    Gamma2 = (m2/2 + m3)*l2*s12 + (m3/2)*l3*s123
    Gamma3 = (m3/2)*l3*s123
    return -GRAV * np.array([Gamma1, Gamma2, Gamma3], dtype=float)


def forward_kinematics(q):
    """
    정기구학: q → 엔드이펙터 위치 [x, y] (m).
    임피던스 제어에서 목표 x_d 와 현재 x 비교에 사용.
    """
    q1, q2, q3 = q[0], q[1], q[2]
    x = L1*np.cos(q1) + L2*np.cos(q1 + q2) + L3*np.cos(q1 + q2 + q3)
    y = L1*np.sin(q1) + L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3)
    return np.array([x, y], dtype=float)


def jacobian(q):
    """
    자코비안 J(q): 2x3. dx/dt = J @ qd.
    작업공간 속도와 관절 속도를 연결해, 원하는 xdd_des 를 qdd_des 로 바꿀 때 사용.
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
    """dJ/dt = (dJ/dq)*qd. 가속도 관계 d²x/dt² = J*q'' + J_dot*qd 에서 사용."""
    q1, q2, q3 = q[0], q[1], q[2]
    qd1, qd2, qd3 = qd[0], qd[1], qd[2]
    s1, s12, s123 = np.sin(q1), np.sin(q1 + q2), np.sin(q1 + q2 + q3)
    c1, c12, c123 = np.cos(q1), np.cos(q1 + q2), np.cos(q1 + q2 + q3)
    dJ_dq1 = np.array([[-L1*c1 - L2*c12 - L3*c123, -L2*c12 - L3*c123, -L3*c123],
                       [-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123]])
    dJ_dq2 = np.array([[-L2*c12 - L3*c123, -L2*c12 - L3*c123, -L3*c123],
                       [-L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123]])
    dJ_dq3 = np.array([[-L3*c123, -L3*c123, -L3*c123],
                       [-L3*s123, -L3*s123, -L3*s123]])
    return dJ_dq1*qd1 + dJ_dq2*qd2 + dJ_dq3*qd3


def qdd_from_tau(q, qd, tau):
    """q'' = M^{-1}(tau - C - G). 시뮬레이션 적분용."""
    return np.linalg.solve(M(q), tau - C_vec(q, qd) - G(q))


def rk4_step(q, qd, tau, dt):
    """RK4 한 스텝: (q,qd), tau, dt → (q_new, qd_new)."""
    def deriv(state):
        q_now, qd_now = state[:3], state[3:6]
        qdd = qdd_from_tau(q_now, qd_now, tau)
        return np.concatenate([qd_now, qdd])
    state = np.concatenate([q, qd])
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    state_new = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return state_new[:3], state_new[3:6]
