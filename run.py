#!/usr/bin/env python3
"""
Run experiments: select controller and simulate in MuJoCo.
  python run.py --controller pid
  python run.py --controller impedance
  python run.py --controller lqr
  python run.py --controller mpc
"""
import argparse
import importlib
import sys
import numpy as np
from pathlib import Path

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot.robot import RobotArm
from sim.environment import SimulationEnv
from controllers import CONTROLLERS


def get_reference(t: float, nq: int = 6) -> np.ndarray:
    """Target: constant pose with optional small motion. Returns (6,) or (12,) for q_ref [, qd_ref]."""
    q_ref = np.zeros(nq)
    q_ref[0] = 0.5 * np.sin(0.5 * t)   # slow sine on joint 1
    q_ref[1] = 0.3
    q_ref[2] = -0.2
    q_ref[3] = 0.0
    q_ref[4] = 0.0
    q_ref[5] = 0.0
    qd_ref = np.zeros(nq)
    return np.concatenate([q_ref, qd_ref])


def make_controller(name: str, robot: RobotArm, dt: float, q0_ref: np.ndarray | None = None):
    """Instantiate controller by name. Each controller is independent."""
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {name}. Choose from {list(CONTROLLERS)}")
    mod_name, class_name = CONTROLLERS[name]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, class_name)
    q0_ref = q0_ref if q0_ref is not None else np.zeros(robot.NQ)
    if name == "lqr":
        return cls(robot=robot, dt=dt, q_eq=q0_ref)
    if name in ("impedance", "mpc"):
        return cls(robot=robot, dt=dt)
    return cls(dt=dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="6-DOF arm control experiment")
    parser.add_argument("--controller", type=str, default="pid", choices=list(CONTROLLERS))
    parser.add_argument("--model", type=str, default="arm_6dof.xml", help="MuJoCo model name in robot/assets")
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration (s)")
    parser.add_argument("--render", action="store_true", default=True, help="Open MuJoCo viewer (default: True)")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Disable MuJoCo viewer")
    args = parser.parse_args()

    xml_path = Path(__file__).resolve().parent / "robot" / "assets" / args.model
    env = SimulationEnv.from_xml_path(str(xml_path), render=args.render)
    robot = env.robot
    dt = float(robot.model.opt.timestep)

    # Initial reference for starting state and (for LQR) linearization point
    q0_ref = get_reference(0.0, robot.NQ)[: robot.NQ].copy()
    controller = make_controller(args.controller, robot, dt, q0_ref=q0_ref)
    if hasattr(controller, "reset_integral"):
        controller.reset_integral()

    # Start at initial reference so tracking error can decrease
    state = env.reset(q=q0_ref, qd=np.zeros(robot.NQ))
    t = 0.0
    n_steps = int(args.duration / dt)
    log_t = []
    log_q = []
    log_q_ref = []

    for _ in range(n_steps):
        ref = get_reference(t, robot.NQ)
        tau = controller.compute_control(state, ref, t)
        state = env.step(tau)
        t = env.get_time()
        log_t.append(t)
        log_q.append(state.q.copy())
        log_q_ref.append(ref[:6].copy())

    env.close()
    log_t = np.array(log_t)
    log_q = np.array(log_q)
    log_q_ref = np.array(log_q_ref)
    err = np.linalg.norm(log_q - log_q_ref, axis=1)
    print(f"Controller: {args.controller}, steps: {n_steps}, mean tracking error (L2): {err.mean():.6f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(log_t, log_q_ref[:, 0], "k--", label="q_ref")
        ax[0].plot(log_t, log_q[:, 0], label="q")
        ax[0].set_ylabel("q[0]")
        ax[0].legend()
        ax[0].grid(True)
        ax[1].plot(log_t, err, label="|q - q_ref|")
        ax[1].set_ylabel("error")
        ax[1].set_xlabel("t (s)")
        ax[1].legend()
        ax[1].grid(True)
        plt.savefig("run_result.png", dpi=120)
        print("Saved run_result.png")
    except Exception as e:
        print(f"Plot skip: {e}")


if __name__ == "__main__":
    main()
