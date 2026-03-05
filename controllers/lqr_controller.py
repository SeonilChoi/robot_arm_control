"""
LQR controller: linearization around equilibrium (q_ref, 0) and constant gain u = -K @ (x - x_ref).
"""
import numpy as np
import mujoco
from scipy.linalg import solve_continuous_are
from controllers.base import ControllerBase
from robot.robot import RobotState


class LQRController(ControllerBase):
    """LQR on linearized dynamics. Independent of other controllers."""

    def __init__(
        self,
        robot,
        dt: float = 0.002,
        q_eq: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ):
        self._robot = robot
        self.dt = dt
        nq = robot.NQ
        self.q_eq = np.zeros(nq) if q_eq is None else np.asarray(q_eq)[:nq]
        Q = np.eye(2 * nq) * 10.0 if Q is None else Q
        R = np.eye(nq) * 0.1 if R is None else R
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.K = self._compute_gain()

    def _compute_gain(self) -> np.ndarray:
        nq = self._robot.NQ
        q_eq = self.q_eq
        # Linearize: x = [q, qd], xd = [qd, qdd], qdd = M^{-1}(tau - c)
        # At equilibrium qd=0: c = g(q). A = [0 I; -M^{-1} d(c)/dq 0], B = [0; M^{-1}]
        mujoco.mj_resetData(self._robot.model, self._robot.data)
        self._robot.set_state(q_eq, np.zeros(nq))
        mujoco.mj_forward(self._robot.model, self._robot.data)
        M = self._robot.get_M()
        c = self._robot.get_c()
        # Numerical Jacobian of -c w.r.t. q (minus because qdd = M^{-1}(tau - c), so d/dq of -M^{-1}c)
        eps = 1e-6
        dcdq = np.zeros((nq, nq))
        for j in range(nq):
            qp = q_eq.copy()
            qp[j] += eps
            self._robot.set_state(qp, np.zeros(nq))
            mujoco.mj_forward(self._robot.model, self._robot.data)
            cp = self._robot.get_c()
            qm = q_eq.copy()
            qm[j] -= eps
            self._robot.set_state(qm, np.zeros(nq))
            mujoco.mj_forward(self._robot.model, self._robot.data)
            cm = self._robot.get_c()
            dcdq[:, j] = (cp - cm) / (2 * eps)
        self._robot.set_state(q_eq, np.zeros(nq))
        mujoco.mj_forward(self._robot.model, self._robot.data)
        Minv = np.linalg.inv(M)
        # x = [q, qd], xd = A x + B u; qd_dot = qdd = M^{-1}(tau - c) ~ M^{-1} tau - M^{-1} c(q,0)
        # d(qdd)/dq = -M^{-1} dcdq (at eq), d(qdd)/d(qd) ~ 0 for coriolis at qd=0
        A21 = -Minv @ dcdq
        A = np.block([[np.zeros((nq, nq)), np.eye(nq)], [A21, np.zeros((nq, nq))]])
        B = np.block([[np.zeros((nq, nq))], [Minv]])
        # Solve continuous-time CARE: A'P + PA - P B R^{-1} B' P + Q = 0, K = R^{-1} B' P
        P = solve_continuous_are(A, B, self.Q, self.R)
        K = np.linalg.solve(self.R, B.T @ P)
        return K

    def compute_control(
        self,
        state: RobotState,
        reference: np.ndarray,
        t: float,
    ) -> np.ndarray:
        ref = np.atleast_1d(reference)
        q_ref = ref[:6] if ref.size >= 6 else self.q_eq.copy()
        qd_ref = ref[6:12] if ref.size >= 12 else np.zeros(6)
        x = np.concatenate([state.q[:6], state.qd[:6]])
        x_ref = np.concatenate([q_ref, qd_ref])
        u = -self.K @ (x - x_ref)
        return u.astype(np.float64)
