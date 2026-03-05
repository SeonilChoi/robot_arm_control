"""
Common interface for all controllers: compute_control(state, reference, t) -> tau.
"""
from abc import ABC, abstractmethod
import numpy as np
from robot.robot import RobotState


class ControllerBase(ABC):
    """Base class for 6-DOF arm controllers. Each controller is independent."""

    @abstractmethod
    def compute_control(
        self,
        state: RobotState,
        reference: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Compute joint torques.

        Args:
            state: current q, qd
            reference: target (interpretation depends on controller; usually q_ref or [q_ref, qd_ref])
            t: current time in seconds

        Returns:
            tau: joint torques, shape (6,)
        """
        pass
