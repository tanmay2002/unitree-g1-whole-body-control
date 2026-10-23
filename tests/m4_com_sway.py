#!/usr/bin/env python3
"""
M4 lateral CoM sway.

Move the CoM side to side over each foot before lifting the other foot.

Usage:
    python tests/m4_com_sway.py --view
    python tests/m4_com_sway.py --no-ff        # feedforward disabled
"""

import argparse
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin

sys.path.insert(0, "src")
from g1_wbc.mj_pin_bridge import build_joint_map, mj_to_pin_q, mj_to_pin_v
from g1_wbc.wbc_controller import (G1BalanceController, PELVIS_H,
                                   STANDING_POSE, LF_FRAME, RF_FRAME)

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"


def sway_reference(t, com0, amp, freq, t_ramp):
    """Position, velocity and acceleration of the swaying CoM target."""
    w = 2.0 * np.pi * freq
    r = min(1.0, t / t_ramp)
    rd = (1.0 / t_ramp) if t < t_ramp else 0.0

    s, c = np.sin(w * t), np.cos(w * t)
    y = amp * r * s
    yd = amp * (rd * s + r * w * c)
    ydd = amp * (2.0 * rd * w * c - r * w * w * s)

    pos = com0 + np.array([0.0, y, 0.0])
    vel = np.array([0.0, yd, 0.0])
    acc = np.array([0.0, ydd, 0.0])
    return pos, vel, acc


def build_actuated_ctrl_map(mj_model, mapping):
    """getActuatorForces returns 23 entries in ACTUATED order"""
    out = []
    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out.append((idx_v - 6, aid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--amp", type=float, default=0.05, help="sway amplitude [m]")
    ap.add_argument("--freq", type=float, default=0.25, help="sway frequency [Hz]")
    ap.add_argument("--ramp", type=float, default=2.0, help="ramp-in time [s]")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-ff", action="store_true",
                    help="disable velocity/acceleration feedforward")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pm_data = pm.createData()
    mapping = build_joint_map(m, pm)
    ctrl_map = build_actuated_ctrl_map(m, mapping)

    # initial standing state 
    mujoco.mj_resetData(m, d)
    d.qpos[2] = PELVIS_H
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, qadr, _, _, _ in mapping:
        for key, val in STANDING_POSE.items():
            if key in name:
                d.qpos[qadr] = val
    mujoco.mj_forward(m, d)

    q0 = mj_to_pin_q(d, mapping, pm)
    v0 = mj_to_pin_v(d, mapping, pm)
    pin.forwardKinematics(pm, pm_data, q0)
    pin.updateFramePlacements(pm, pm_data)

    # how far can the CoM travel sideways
    lf_y = pm_data.oMf[pm.getFrameId(LF_FRAME)].translation[1]
    rf_y = pm_data.oMf[pm.getFrameId(RF_FRAME)].translation[1]
    half_stance = abs(lf_y - rf_y) / 2.0

    ctrl = G1BalanceController()
    com0 = ctrl.setup(q0, v0)

    print(f"feet at y = {lf_y:+.4f} / {rf_y:+.4f}  half stance "
          f"{half_stance:.4f} m")
    print(f"sway amplitude {args.amp:.3f} m "
          f"({100*args.amp/half_stance:.0f}% of half stance)")
    print(f"feedforward: {'OFF' if args.no_ff else 'ON'}")
    print(f"CoM reference: {com0}\n")
    if args.amp > half_stance:
        print("amplitude exceeds half stance CoM will leave the support polygon ")
              

    n_steps = int(args.duration / m.opt.timestep)
    report = max(1, n_steps // 12)
    fails = 0

    log_t, log_ref, log_act = [], [], []

    print(f"  {'t':>6}  {'ref_y':>8}  {'com_y':>8}  {'err':>8}  "
          f"{'|tau|':>7}  {'ncon':>5}")
    print("  " + "-" * 52)

    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None
    wall_start = time.time()

    for i in range(n_steps):
        if viewer is not None and not viewer.is_running():
            print("\n  viewer closed by user")
            break

        q = mj_to_pin_q(d, mapping, pm)
        v = mj_to_pin_v(d, mapping, pm)

        pos, vel, acc = sway_reference(d.time, com0, args.amp,
                                       args.freq, args.ramp)
        if args.no_ff:
            ctrl.set_com_reference(pos)                 # zeros for vel, acc
        else:
            ctrl.set_com_reference(pos, vel, acc)

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
            print(f"\n  !! diverged at t={d.time:.3f}")
            if viewer is not None:
                viewer.close()
            return

        com = pin.centerOfMass(pm, pm_data, mj_to_pin_q(d, mapping, pm))
        
        if d.time > args.ramp:
            log_t.append(d.time)
            log_ref.append(pos[1])
            log_act.append(com[1])

        if viewer is not None:
            viewer.sync()
            lag = d.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

        if i % report == 0:
            tn = 0.0 if tau is None else np.linalg.norm(tau)
            print(f"  {d.time:6.3f}  {pos[1]:+8.4f}  {com[1]:+8.4f}  "
                  f"{com[1]-pos[1]:+8.4f}  {tn:7.2f}  {d.ncon:5d}")

    if viewer is not None:
        viewer.close()

    ref = np.array(log_ref)
    act = np.array(log_act)
    err = act - ref

    print()
    print(f"steady-state tracking (after {args.ramp}s ramp, "
          f"{len(ref)} samples):")
    print(f"RMS error: {np.sqrt(np.mean(err**2))*1000:7.2f} mm")
    print(f"max |error|: {np.abs(err).max()*1000:7.2f} mm")
    print(f"ref amplitude: {np.abs(ref).max()*1000:7.2f} mm")
    print(f"actual amplitude: {np.abs(act).max()*1000:7.2f} mm")
    print(f"amplitude ratio: {np.abs(act).max()/np.abs(ref).max():7.3f}")
    print(f"QP failures: {fails} / {n_steps}")
    print()

    ok = fails == 0 and np.sqrt(np.mean(err**2)) < 0.01
    print("  M4 PASSED the CoM tracks a moving reference."
          if ok else "  M4 needs work.")


if __name__ == "__main__":
    main()