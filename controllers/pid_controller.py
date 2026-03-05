"""
Joint-space PID controller: tau = Kp*(q_ref - q) + Kd*(qd_ref - qd) + Ki*integral(q_ref - q).
"""
import numpy as np
from controllers.base import ControllerBase
from robot.robot import RobotState


class PIDController(ControllerBase):
    """Independent PID controller; no dependency on other controllers."""

    def __init__(
        self,
        dt: float = 0.002,
        Kp: float | np.ndarray = 80.0,
        Ki: float | np.ndarray = 2.0,
        Kd: float | np.ndarray = 15.0,
        nq: int = 6,
        integral_clip: float = 1.0,
        robot=None,
    ):
        self.dt = dt
        self.nq = nq
        self.Kp = np.broadcast_to(Kp, nq).copy()
        self.Ki = np.broadcast_to(Ki, nq).copy()
        self.Kd = np.broadcast_to(Kd, nq).copy()
        self.integral_clip = integral_clip
        self._integral = np.zeros(nq)

    def compute_control(
        self,
        state: RobotState,
        reference: np.ndarray,
        t: float,
    ) -> np.ndarray:
        # reference: (6,) or (12,) — if 12, use first 6 as q_ref, next 6 as qd_ref
        ref = np.atleast_1d(reference)
        if ref.size >= 12:
            q_ref = ref[:6]
            qd_ref = ref[6:12]
        else:
            q_ref = ref[:6] if ref.size >= 6 else np.zeros(6)
            qd_ref = np.zeros(6)
        q = state.q[: self.nq]
        qd = state.qd[: self.nq]
        err = q_ref - q
        self._integral += err * self.dt
        self._integral = np.clip(self._integral, -self.integral_clip, self.integral_clip)
        tau = self.Kp * err + self.Kd * (qd_ref - qd) + self.Ki * self._integral
        return tau.astype(np.float64)

    def reset_integral(self) -> None:
        self._integral.fill(0.0)
