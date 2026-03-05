from controllers.base import ControllerBase

# Registry for run.py: name -> (module_name, class_name)
CONTROLLERS = {
    "pid": ("controllers.pid_controller", "PIDController"),
    "impedance": ("controllers.impedance_controller", "ImpedanceController"),
    "lqr": ("controllers.lqr_controller", "LQRController"),
    "mpc": ("controllers.mpc_controller", "MPCController"),
}

__all__ = ["ControllerBase", "CONTROLLERS"]
