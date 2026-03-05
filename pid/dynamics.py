"""
3-DOF planar robot arm dynamics (numpy only).
M(q) qdd + C(q,qd) + G(q) = tau
Convention: q1 relative to vertical, q2 relative to link1, q3 relative to link2.
"""
import numpy as np

# Robot parameters
L1, L2, L3 = 0.5, 0.4, 0.3
M1, M2, M3 = 1.0, 0.8, 0.5
I1z, I2z, I3z = 0.01, 0.01, 0.005
GRAV = 9.81


def M(q):
    """Inertia matrix M(q), 3x3 symmetric positive definite."""
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
        [I11, I12, I13],
        [I12, I22, I23],
        [I13, I23, I33]
    ], dtype=float)


def C_vec(q, qd):
    """Coriolis/centrifugal force vector C(q,qd) such that M*qdd + C_vec + G = tau."""
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
    """Gravity vector G(q). tau = M*qdd + C_vec + G."""
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
    """End-effector position (2D) x = [x, y] for planar 3R."""
    q1, q2, q3 = q[0], q[1], q[2]
    x = L1*np.cos(q1) + L2*np.cos(q1 + q2) + L3*np.cos(q1 + q2 + q3)
    y = L1*np.sin(q1) + L2*np.sin(q1 + q2) + L3*np.sin(q1 + q2 + q3)
    return np.array([x, y], dtype=float)


def jacobian(q):
    """Jacobian J such that dx/dt = J @ qd, 2x3."""
    q1, q2, q3 = q[0], q[1], q[2]
    s1, s12, s123 = np.sin(q1), np.sin(q1 + q2), np.sin(q1 + q2 + q3)
    c1, c12, c123 = np.cos(q1), np.cos(q1 + q2), np.cos(q1 + q2 + q3)
    J = np.array([
        [-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
        [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]
    ], dtype=float)
    return J


def jacobian_dot(q, qd):
    """J_dot such that d2x/dt2 = J @ qdd + J_dot @ qd. Returns 2x3."""
    q1, q2, q3 = q[0], q[1], q[2]
    qd1, qd2, qd3 = qd[0], qd[1], qd[2]
    s1, s12, s123 = np.sin(q1), np.sin(q1 + q2), np.sin(q1 + q2 + q3)
    c1, c12, c123 = np.cos(q1), np.cos(q1 + q2), np.cos(q1 + q2 + q3)
    # dJ/dq1, dJ/dq2, dJ/dq3
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
    """Compute qdd from tau: qdd = M^{-1}(tau - C_vec - G)."""
    M_q = M(q)
    c_vec = C_vec(q, qd)
    g_vec = G(q)
    return np.linalg.solve(M_q, tau - c_vec - g_vec)


def rk4_step(q, qd, tau, dt):
    """Single RK4 step for q, qd. Returns (q_new, qd_new)."""
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
