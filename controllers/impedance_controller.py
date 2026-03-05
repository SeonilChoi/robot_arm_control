"""
Impedance controller (dynamics-based): M(q) u + c(q,qd) with
  u = qdd_ref + M_d^{-1} ( K_d (q_ref - q) + B_d (qd_ref - qd) ).
"""
import numpy as np
import mujoco
from controllers.base import ControllerBase
from robot.robot import RobotState


class ImpedanceController(ControllerBase):
    """Uses robot M(q), c(q,qd). Independent of other controllers."""

    def __init__(
        self,
        robot,
        dt: float = 0.002,
        Md: float | np.ndarray = 1.0,
        Bd: float | np.ndarray = 40.0,
        Kd: float | np.ndarray = 200.0,
    ):
        self._robot = robot
        self.dt = dt
        nq = robot.NQ
        if np.ndim(Md) == 0:
            self.Md = np.eye(nq) * float(Md)
        else:
            md = np.atleast_1d(Md)
            self.Md = np.diag(np.broadcast_to(md, nq).copy())
        self.Bd = np.diag(np.broadcast_to(np.atleast_1d(Bd), nq)) if np.ndim(Bd) < 2 else np.asarray(Bd)
        self.Kd = np.diag(np.broadcast_to(np.atleast_1d(Kd), nq)) if np.ndim(Kd) < 2 else np.asarray(Kd)

    def compute_control(
        self,
        state: RobotState,
        reference: np.ndarray,
        t: float,
    ) -> np.ndarray:
        ref = np.atleast_1d(reference)
        q_ref = ref[:6] if ref.size >= 6 else np.zeros(6)
        qd_ref = ref[6:12] if ref.size >= 12 else np.zeros(6)
        q = state.q[:6]
        qd = state.qd[:6]
        # Ensure dynamics are up to date
        mujoco.mj_forward(self._robot.model, self._robot.data)
        M = self._robot.get_M()
        c = self._robot.get_c()
        # Desired acceleration: impedance law
        err_q = q_ref - q
        err_qd = qd_ref - qd
        qdd_ref = np.zeros(6)  # could use finite diff of qd_ref if available
        u = qdd_ref + np.linalg.solve(self.Md, self.Kd @ err_q + self.Bd @ err_qd)
        tau = M @ u + c
        return tau.astype(np.float64)
