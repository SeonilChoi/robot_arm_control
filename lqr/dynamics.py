"""
3-DOF 평면 로봇 동역학 + 선형화 (LQR용)

역할: M, C, G, q'' = M^{-1}(tau-C-G), RK4 적분,
      평형점 q_e 에서 선형화하여 A, B 행렬 계산 (dz/dt = A*z + B*u).
      LQR는 이 선형 모델로 이득 K 를 구합니다.
"""
import numpy as np

L1, L2, L3 = 0.5, 0.4, 0.3
M1, M2, M3 = 1.0, 0.8, 0.5
I1z, I2z, I3z = 0.01, 0.01, 0.005
GRAV = 9.81


def M(q):
    """관성 행렬 M(q) 3x3."""
    q1, q2, q3 = q[0], q[1], q[2]
    c2, c3 = np.cos(q2), np.cos(q3)
    c23 = np.cos(q2 + q3)
    l1, l2, l3 = L1, L2, L3
    m1, m2, m3 = M1, M2, M3
    i1, i2, i3 = I1z, I2z, I3z
    I11 = (m1/4 + m2 + m3)*l1**2 + (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i1 + i2 + i3 \
          + (m2 + 2*m3)*l1*l2*c2 + m3*l3*(l1*c23 + l2*c3)
    I12 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 + (m2/2 + m3)*l1*l2*c2 + (m3/2)*l3*(l1*c23 + 2*l2*c3)
    I13 = (m3/4)*l3**2 + i3 + (m3/2)*l3*(l1*c23 + l2*c3)
    I22 = (m2/4 + m3)*l2**2 + (m3/4)*l3**2 + i2 + i3 + m3*l2*l3*c3
    I23 = (m3/4)*l3**2 + i3 + (m3/2)*l2*l3*c3
    I33 = (m3/4)*l3**2 + i3
    return np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]], dtype=float)


def C_vec(q, qd):
    """코리올리스/원심력 벡터 (3,)."""
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
    """중력 벡터 (3,). 평형에서 tau_e = G(q_e)."""
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
    """q'' = M^{-1}(tau - C - G). 선형화 시 수치미분에 사용."""
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


def linearize(q_e, eps=1e-7):
    """
    평형점 (z_e, tau_e) 에서 선형화: dz/dt = A*z + B*u.

    z = [q; qd] (6,), u = tau - tau_e (3,). 평형에서 tau_e = G(q_e), qd_e = 0.
    A: (6,6) — 상태에 대한 f의 야코비안 (수치 미분).
    B: (6,3) — B[0:3,:]=0 (q의 미분은 qd만), B[3:6,:]=M^{-1}(q_e).

    인자: q_e — 평형 관절각 (3,), eps — 수치 미분 시 증분
    반환: A, B
    """
    tau_e = G(q_e)
    A = np.zeros((6, 6))
    B = np.zeros((6, 3))
    B[3:6, :] = np.linalg.inv(M(q_e))  # d(qdd)/d(tau) = M^{-1}

    # A[3:6, 0:3]: d(qdd)/d(q) — q를 eps만큼 흔들어 qdd 변화량으로 근사
    for i in range(3):
        q_plus = q_e.copy()
        q_plus[i] += eps
        q_minus = q_e.copy()
        q_minus[i] -= eps
        A[3:6, i] = (qdd_from_tau(q_plus, np.zeros(3), tau_e) - qdd_from_tau(q_minus, np.zeros(3), tau_e)) / (2*eps)
    # A[3:6, 3:6]: d(qdd)/d(qd)
    for i in range(3):
        qd_eps = np.zeros(3)
        qd_eps[i] = eps
        A[3:6, 3+i] = (qdd_from_tau(q_e, qd_eps, tau_e) - qdd_from_tau(q_e, -qd_eps, tau_e)) / (2*eps)
    A[0:3, 3:6] = np.eye(3)  # d(q)/dt = qd
    return A, B
