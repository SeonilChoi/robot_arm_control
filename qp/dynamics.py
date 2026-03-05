"""
3-DOF 평면 로봇 동역학 (QP 제어용)

역할: M(q), C(q,qd), G(q), q'' = M^{-1}(tau-C-G), RK4 적분.
      QP 제어기는 목표 가속도 a* 를 QP로 정한 뒤 tau = M*a* + C + G 로 토크를 냅니다.
"""
import numpy as np

L1, L2, L3 = 0.5, 0.4, 0.3
M1, M2, M3 = 1.0, 0.8, 0.5
I1z, I2z, I3z = 0.01, 0.01, 0.005
GRAV = 9.81


def M(q):
    """관성 행렬 3x3."""
    q1, q2, q3 = q[0], q[1], q[2]
    c2, c3 = np.cos(q2), np.cos(q3)
    c23 = np.cos(q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    i1, i2, i3 = I1z, I2z, I3z
    I11 = (m1/4 + m2 + m3)*l1**2 + (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i1 + i2 + i3 + (m2 + 2*m3)*l1*l2*c2 + m3*l3*(l1*c23 + l2*c3)
    I12 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 + (m2/2 + m3)*l1*l2*c2 + (m3/2)*l3*(l1*c23 + 2*l2*c3)
    I13 = (m3/4)*l3**2 + i3 + (m3/2)*l3*(l1*c23 + l2*c3)
    I22 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 + m3*l2*l3*c3
    I23 = (m3/4)*l3**2 + i3 + (m3/2)*l2*l3*c3
    I33 = (m3/4)*l3**2 + i3
    return np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]], dtype=float)


def C_vec(q, qd):
    """코리올리스/원심력 (3,)."""
    q1, q2, q3 = q[0], q[1], q[2]
    qd1, qd2, qd3 = qd[0], qd[1], qd[2]
    s2, s3 = np.sin(q2), np.sin(q3)
    s23 = np.sin(q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m2, m3 = M2, M3
    c1 = -(m2/2 + m3)*l1*l2*s2*(2*qd1*qd2 + qd2**2) - (m3/2)*l3*l1*s23*(qd2 + qd3)*(qd1 + qd2 + qd3) - (m3/2)*l3*l2*s3*qd3*(qd1 + qd2)
    c2 = (m2/2 + m3)*l1*l2*s2*qd1**2 + (m3/2)*l3*l1*s23*qd1**2 - (m3/2)*l3*l2*s3*qd3*(2*qd1 + 2*qd2 + qd3)
    c3 = (m3/2)*l3*(l1*s23*qd1**2 + l2*s3*(qd1 + qd2)**2)
    return np.array([c1, c2, c3], dtype=float)


def G(q):
    """중력 (3,)."""
    q1, q2, q3 = q[0], q[1], q[2]
    s1, s12 = np.sin(q1), np.sin(q1 + q2)
    s123 = np.sin(q1 + q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    Gamma1 = (m1/2 + m2 + m3)*l1*s1 + (m2/2 + m3)*l2*s12 + (m3/2)*l3*s123
    Gamma2 = (m2/2 + m3)*l2*s12 + (m3/2)*l3*s123
    Gamma3 = (m3/2)*l3*s123
    return -GRAV * np.array([Gamma1, Gamma2, Gamma3], dtype=float)


def qdd_from_tau(q, qd, tau):
    """q'' = M^{-1}(tau - C - G)."""
    return np.linalg.solve(M(q), tau - C_vec(q, qd) - G(q))


def rk4_step(q, qd, tau, dt):
    """RK4 한 스텝."""
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
