"""
MuJoCo simulation environment: step, state, apply torque.
"""
import numpy as np
import mujoco
from robot.robot import RobotArm, RobotState


class SimulationEnv:
    """
    Wraps MuJoCo simulation: reset(), step(tau), get_state().
    Optional rendering via MuJoCo viewer.
    """
    def __init__(self, robot: RobotArm, render: bool = False):
        self._robot = robot
        self._render = render
        self._viewer = None
        if render:
            try:
                self._viewer = mujoco.viewer.launch_passive(
                    robot.model, robot.data, key_callback=None
                )
            except Exception:
                self._viewer = None
                self._render = False

    @classmethod
    def from_xml_path(cls, xml_path: str = "arm_6dof.xml", render: bool = False) -> "SimulationEnv":
        robot = RobotArm.from_xml_path(xml_path)
        return cls(robot, render=render)

    @property
    def robot(self) -> RobotArm:
        return self._robot

    def reset(self, q: np.ndarray | None = None, qd: np.ndarray | None = None) -> RobotState:
        """Reset simulation state. If q/qd not given, use zeros."""
        if q is None:
            q = np.zeros(self._robot.NQ)
        if qd is None:
            qd = np.zeros(self._robot.NQD)
        mujoco.mj_resetData(self._robot.model, self._robot.data)
        self._robot.set_state(q, qd)
        mujoco.mj_forward(self._robot.model, self._robot.data)
        if self._viewer is not None:
            self._viewer.sync()
        return self.get_state()

    def step(self, tau: np.ndarray) -> RobotState:
        """Apply joint torques and advance simulation by one step."""
        tau = np.atleast_1d(tau).astype(np.float64)
        n = min(tau.size, self._robot.model.nu)
        self._robot.data.ctrl[:n] = tau[:n]
        if n < self._robot.model.nu:
            self._robot.data.ctrl[n:] = 0.0
        mujoco.mj_step(self._robot.model, self._robot.data)
        if self._viewer is not None:
            self._viewer.sync()
        return self.get_state()

    def get_state(self) -> RobotState:
        """Return current state (q, qd)."""
        return self._robot.get_state()

    def get_time(self) -> float:
        return float(self._robot.data.time)

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
