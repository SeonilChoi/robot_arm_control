"""
6-DOF robot arm: model loading, state, and dynamics helpers for controllers.
"""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class RobotState:
    """Current state of the robot (joint space)."""
    q: np.ndarray      # joint positions (6,)
    qd: np.ndarray     # joint velocities (6,)


class RobotArm:
    """
    Loads MuJoCo 6-DOF arm model and provides state and dynamics helpers.
    Controllers use this for M(q), c(q,qd), and state; they do not depend on each other.
    """
    NQ = 6
    NQD = 6

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data
        self._qpos_idx = np.arange(self.NQ)
        self._qvel_idx = np.arange(self.NQ)
        self._actuator_idx = np.arange(self.NQ)
        # Ensure we have 6 DOF
        assert model.nq >= self.NQ and model.nv >= self.NQD and model.nu >= self.NQ, (
            "Model must have at least 6 position, 6 velocity, and 6 actuator dimensions."
        )

    @classmethod
    def from_xml_path(cls, path: str) -> "RobotArm":
        """Load robot from MJCF XML path."""
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / "assets" / path.name
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        return cls(model, data)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def get_state(self) -> RobotState:
        """Return current joint positions and velocities."""
        q = np.array(self._data.qpos[self._qpos_idx], copy=True)
        qd = np.array(self._data.qvel[self._qvel_idx], copy=True)
        return RobotState(q=q, qd=qd)

    def set_state(self, q: np.ndarray, qd: np.ndarray) -> None:
        """Set joint positions and velocities (e.g. for simulation reset)."""
        self._data.qpos[self._qpos_idx] = q
        self._data.qvel[self._qvel_idx] = qd

    def get_M(self) -> np.ndarray:
        """Return inertia matrix M(q) of shape (6, 6). Requires mj_forward to be called."""
        M = np.zeros((self.NQ, self.NQ), order="C")
        mujoco.mj_fullM(self._model, M, self._data.qM)
        return M

    def get_c(self) -> np.ndarray:
        """
        Return Coriolis + gravity vector c(q, qd) of shape (6,).
        MuJoCo's qfrc_inverse is the inverse dynamics with qacc=0, i.e. tau = c.
        Requires mj_forward to be called.
        """
        return np.array(self._data.qfrc_inverse[self._qvel_idx], copy=True)

    def get_Jacobian_ee(self) -> np.ndarray:
        """Return translational Jacobian of end-effector (3 x 6) in world frame."""
        mujoco.mj_forward(self._model, self._data)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, ee_id)
        return jacp[:, : self.NQ].copy()

    def get_ee_pos(self) -> np.ndarray:
        """Return end-effector position (3,) in world frame."""
        ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return np.array(self._data.site_xpos[ee_id], copy=True)
