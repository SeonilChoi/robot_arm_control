"""
MPC: receding-horizon optimization with scipy; use first torque only each step.
Dynamics: M(q) qdd + c(q,qd) = tau, discretized with Euler.
"""
import numpy as np
import mujoco
from scipy.optimize import minimize
from controllers.base import ControllerBase
from robot.robot import RobotState


class MPCController(ControllerBase):
    """Scipy-based MPC; uses robot M, c for rollout. Independent of other controllers."""

    def __init__(
        self,
        robot,
        dt: float = 0.002,
        horizon: int = 10,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ):
        self._robot = robot
        self.dt = dt
        self.horizon = horizon
        nq = robot.NQ
        Q = np.eye(2 * nq) * 10.0 if Q is None else Q
        R = np.eye(nq) * 0.01 if R is None else R
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.nq = nq
        self._x_ref = np.zeros(2 * nq)

    def _dynamics_step(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray):
        """One Euler step: return (q_next, qd_next)."""
        self._robot.set_state(q, qd)
        mujoco.mj_forward(self._robot.model, self._robot.data)
        M = self._robot.get_M()
        c = self._robot.get_c()
        qdd = np.linalg.solve(M, tau - c)
        q_next = q + self.dt * qd
        qd_next = qd + self.dt * qdd
        return q_next, qd_next

    def _rollout_cost(self, u_flat: np.ndarray, x0: np.ndarray, x_ref: np.ndarray) -> float:
        """Cost over horizon: sum of (x-x_ref)'Q(x-x_ref) + u'Ru."""
        nq = self.nq
        N = self.horizon
        u = u_flat.reshape(N, nq)
        cost = 0.0
        q, qd = x0[:nq], x0[nq:]
        for k in range(N):
            tau = u[k]
            cost += (np.concatenate([q, qd]) - x_ref) @ self.Q @ (np.concatenate([q, qd]) - x_ref)
            cost += tau @ self.R @ tau
            q, qd = self._dynamics_step(q, qd, tau)
        return cost

    def compute_control(
        self,
        state: RobotState,
        reference: np.ndarray,
        t: float,
    ) -> np.ndarray:
        ref = np.atleast_1d(reference)
        q_ref = ref[:6] if ref.size >= 6 else np.zeros(6)
        qd_ref = ref[6:12] if ref.size >= 12 else np.zeros(6)
        self._x_ref = np.concatenate([q_ref, qd_ref])
        x0 = np.concatenate([state.q[:6], state.qd[:6]])
        N = self.horizon
        nq = self.nq
        # Initial guess: zero torques
        u0 = np.zeros(N * nq)
        result = minimize(
            self._rollout_cost,
            u0,
            args=(x0, self._x_ref),
            method="L-BFGS-B",
            bounds=[(-50.0, 50.0)] * (N * nq),  # match actuator limits
            options=dict(maxiter=50, disp=False),
        )
        u_opt = result.x.reshape(N, nq)
        return u_opt[0].astype(np.float64)
