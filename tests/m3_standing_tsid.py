#!/usr/bin/env python3
"""
M3 TSID standing balance, closed loop through MuJoCo.

TSID QP uses contact forces + PD task feedback to stabilise the robot.

Usage:
    python tests/m3_standing_tsid.py                    
    python tests/m3_standing_tsid.py --view             
    python tests/m3_standing_tsid.py --view --duration 20
"""

import argparse
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin
import tsid

sys.path.insert(0, "src")
from g1_wbc.mj_pin_bridge import build_joint_map, mj_to_pin_q, mj_to_pin_v

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"
MESH_DIR = "src/g1_wbc/models"

# constants
# measured in m3_setup_probe.py
PELVIS_H = 0.7729
STANDING_POSE = {"hip_pitch": -0.20, "knee": 0.42, "ankle_pitch": -0.23}

LF_FRAME = "left_ankle_roll_link"
RF_FRAME = "right_ankle_roll_link"

# four sole corners, expressed in the ankle_roll frame (from the MJCF spheres)
CONTACT_POINTS = np.array([
    [-0.05, -0.05,  0.12,  0.12],     # x : heel, heel, toe, toe
    [ 0.025, -0.025, 0.03, -0.03],    # y : left/right of each
    [-0.03, -0.03, -0.03, -0.03],     # z : the sole plane
])
CONTACT_NORMAL = np.array([0.0, 0.0, 1.0])

MU = 0.7          # friction coefficient
F_MIN = 1.0       # N, keeps the solver from letting a foot go weightless
F_MAX = 700.0     # N, robot weighs ~315 N so one foot can carry it all

# gains
KP_CONTACT = 30.0
KP_COM     = 20.0
KP_POSTURE = 20.0

# task weights
W_COM     = 1.0
W_POSTURE = 1e-3
W_FORCE   = 1e-5


class G1StandingController:
    def __init__(self, dt):
        self.dt = dt

        # TSID wraps Pinocchio and adds the actuation model
        self.robot = tsid.RobotWrapper(URDF, [MESH_DIR],
                                       pin.JointModelFreeFlyer(), False)
        self.model = self.robot.model()
        self.na = self.robot.na          # actuated dofs = 23

        self.formulation = tsid.InverseDynamicsFormulationAccForce(
            "tsid", self.robot, False)

    def setup(self, q0, v0):
        """Build the QP: contacts first (hard), then tasks (weighted)."""
        f = self.formulation
        f.computeProblemData(0.0, q0, v0)
        data = f.data()

        # contacts 
        
        self.contacts = {}
        for name, frame in [("lf", LF_FRAME), ("rf", RF_FRAME)]:
            c = tsid.Contact6d(f"contact_{name}", self.robot, frame,
                               CONTACT_POINTS, CONTACT_NORMAL,
                               MU, F_MIN, F_MAX)
            c.setKp(KP_CONTACT * np.ones(6))
            c.setKd(2.0 * np.sqrt(KP_CONTACT) * np.ones(6))
            H = self.robot.framePosition(data, self.model.getFrameId(frame))
            c.setReference(H)
            f.addRigidContact(c, W_FORCE)
            self.contacts[name] = c

        # CoM task
        # Holds the CoM at a fixed point.
        self.com_task = tsid.TaskComEquality("task-com", self.robot)
        self.com_task.setKp(KP_COM * np.ones(3))
        self.com_task.setKd(2.0 * np.sqrt(KP_COM) * np.ones(3))
        f.addMotionTask(self.com_task, W_COM, 1, 0.0)

        com0 = self.robot.com(data).copy()
        self.com_ref = com0.copy()
        self.com_sample = tsid.TrajectorySample(3)

        # posture task 
        
        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)
        self.posture_task.setKp(KP_POSTURE * np.ones(self.na))
        self.posture_task.setKd(2.0 * np.sqrt(KP_POSTURE) * np.ones(self.na))
        f.addMotionTask(self.posture_task, W_POSTURE, 1, 0.0)

        self.q_ref = q0[7:].copy()
        self.posture_sample = tsid.TrajectorySample(self.na)
        self.posture_sample.value(self.q_ref)
        self.posture_sample.derivative(np.zeros(self.na))
        self.posture_sample.second_derivative(np.zeros(self.na))

        # solver 
      
        self.solver = tsid.SolverHQuadProgFast("qp")
        self.solver.resize(f.nVar, f.nEq, f.nIn)

        return com0

    def compute(self, t, q, v):
        """One control cycle. Returns (tau_23, solver_status)."""
        self.com_sample.value(self.com_ref)
        self.com_sample.derivative(np.zeros(3))
        self.com_sample.second_derivative(np.zeros(3))
        self.com_task.setReference(self.com_sample)
        self.posture_task.setReference(self.posture_sample)

        HQPData = self.formulation.computeProblemData(t, q, v)
        sol = self.solver.solve(HQPData)
        if sol.status != 0:
            return None, sol.status
        return self.formulation.getActuatorForces(sol), 0


def build_actuated_ctrl_map(mj_model, mapping):
    """TSID's getActuatorForces returns 23 entries in ACTUATED order. Our pin_tau_to_mj_ctrl expects a 29-vector,
    so this needs its own map."""
    out = []
    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out.append((idx_v - 6, aid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--view", action="store_true",
                    help="open the MuJoCo viewer and run at real-time pace")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pm_data = pm.createData()
    mapping = build_joint_map(m, pm)
    ctrl_map = build_actuated_ctrl_map(m, mapping)

    # initial state: the standing pose measured in the probe
    mujoco.mj_resetData(m, d)
    d.qpos[2] = PELVIS_H
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, qadr, _, _, _ in mapping:
        for key, val in STANDING_POSE.items():
            if key in name:
                d.qpos[qadr] = val
    mujoco.mj_forward(m, d)
    print(f"initial contacts : {d.ncon}")

    q0 = mj_to_pin_q(d, mapping, pm)
    v0 = mj_to_pin_v(d, mapping, pm)

    ctrl = G1StandingController(m.opt.timestep)
    com0 = ctrl.setup(q0, v0)
    print(f"CoM reference: {com0}\n")

    n_steps = int(args.duration / m.opt.timestep)
    report = max(1, n_steps // 10)
    fails = 0

    print(f"  {'t':>6}  {'height':>7}  {'com_err':>9}  {'|tau|':>8}  {'ncon':>5}")
    print("  " + "-" * 44)

    #  viewer 
    
    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None
    wall_start = time.time()

    for i in range(n_steps):
        if viewer is not None and not viewer.is_running():
            print("\n  viewer closed by user")
            break

        q = mj_to_pin_q(d, mapping, pm)
        v = mj_to_pin_v(d, mapping, pm)

        tau, status = ctrl.compute(d.time, q, v)
        if tau is None:
            fails += 1
            d.ctrl[:] = 0.0
        else:
            c = np.zeros(m.nu)
            for i_act, aid in ctrl_map:
                c[aid] = tau[i_act]
            d.ctrl[:] = c

        mujoco.mj_step(m, d)

        if not np.all(np.isfinite(d.qpos)):
            print(f"\n  diverged at t={d.time:.3f}")
            if viewer is not None:
                viewer.close()
            return

        # throttle to wall 
        if viewer is not None:
            viewer.sync()
            lag = d.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

        if i % report == 0:
            com = pin.centerOfMass(pm, pm_data, q)
            err = np.linalg.norm(com - com0)
            tn = 0.0 if tau is None else np.linalg.norm(tau)
            print(f"  {d.time:6.3f}  {d.qpos[2]:7.4f}  {err:9.5f}  "
                  f"{tn:8.2f}  {d.ncon:5d}")

    if viewer is not None:
        viewer.close()

    com = pin.centerOfMass(pm, pm_data, mj_to_pin_q(d, mapping, pm))
    print()
    print(f"final height: {d.qpos[2]:.4f} m  (started {PELVIS_H})")
    print(f"final CoM error: {np.linalg.norm(com - com0):.5f} m")
    print(f"QP failures: {fails} / {n_steps}")
    print()
    upright = d.qpos[2] > 0.70
    if upright and fails == 0:
        print("M3 PASSED the robot stands.")
    else:
        print("M3 FAILED.")


if __name__ == "__main__":
    main()