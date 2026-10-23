#!/usr/bin/env python3
"""
M6a continuous walking quasi-static.

Repeat alternating SHIFT → SWING → plant cycles using quintic CoM shifts.
Carry over M5 contact handling, airborne foot task, landing bias and lift-time pose.

Usage:
    python tests/m6a_walk_quasistatic.py --steps 4 --view
    python tests/m6a_walk_quasistatic.py --steps 6 --step-length 0.12
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
                                   STANDING_POSE, FRAMES)

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"


def quintic(u):
    """Minimum-jerk profile and derivatives w.r.t. normalised time."""
    u = np.clip(u, 0.0, 1.0)
    return (10*u**3 - 15*u**4 + 6*u**5,
            30*u**2 - 60*u**3 + 30*u**4,
            60*u - 180*u**2 + 120*u**3)


def interp(p0, p1, u, T):
    """Quintic interpolation between two points with derivatives."""
    s, ds, dds = quintic(u)
    return (p0 + (p1 - p0) * s,
            (p1 - p0) * ds / T,
            (p1 - p0) * dds / (T * T))


def swing_pose(u, T, H_start, H_end, height):
    """SE3 pose, spatial velocity and acceleration of the swinging foot."""
    s, ds, dds = quintic(u)
    p0, p1 = H_start.translation, H_end.translation

    pos = (p0 + (p1 - p0) * s).copy()
    vel = ((p1 - p0) * ds / T).copy()
    acc = ((p1 - p0) * dds / (T * T)).copy()

    w = np.pi
    pos[2] += height * np.sin(w * u)
    vel[2] += height * w * np.cos(w * u) / T
    acc[2] += -height * w * w * np.sin(w * u) / (T * T)

    H = pin.SE3(H_start.rotation.copy(), pos)
    return H, np.concatenate([vel, np.zeros(3)]), np.concatenate([acc, np.zeros(3)])


def build_actuated_ctrl_map(mj_model, mapping):
    out = []
    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out.append((idx_v - 6, aid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--first-swing", default="lf", choices=["lf", "rf"])
    ap.add_argument("--step-length", type=float, default=0.10)
    ap.add_argument("--step-height", type=float, default=0.05)
    ap.add_argument("--t-shift", type=float, default=1.2)
    ap.add_argument("--t-swing", type=float, default=1.0)
    ap.add_argument("--t-settle", type=float, default=2.0)
    ap.add_argument("--shift-frac", type=float, default=0.85)
    ap.add_argument("--transition", type=float, default=0.05)
    ap.add_argument("--land-bias", type=float, default=0.003)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pm_data = pm.createData()
    mapping = build_joint_map(m, pm)
    ctrl_map = build_actuated_ctrl_map(m, mapping)

    #  initial standing state 
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

    ctrl = G1BalanceController()
    com0 = ctrl.setup(q0, v0)

    H0 = {s: ctrl.foot_placement(s) for s in ("lf", "rf")}
    foot_x0 = 0.5 * (H0["lf"].translation[0] + H0["rf"].translation[0])
    # keep the CoM the same distance in front of the stance foot as it
    # started in front of the feet
    com_x_offset = com0[0] - foot_x0

    T_step = args.t_shift + args.t_swing
    t_end = args.steps * T_step + args.t_settle

    print(f"steps: {args.steps}, first swing {args.first_swing}")
    print(f"step length: {args.step_length:.3f} m   "
          f"height {args.step_height:.3f} m")
    print(f"  cycle: shift {args.t_shift}s + swing {args.t_swing}s "
          f"= {T_step}s per step, then settle {args.t_settle}s")
    print(f"expected travel: {args.steps * args.step_length:.3f} m\n")

    n_steps = int(t_end / m.opt.timestep)
    report = max(1, n_steps // 24)
    fails = 0

    # gait bookkeeping 
    step_idx = -1                 # which step we are in
    phase = None
    swing = args.first_swing
    stance = "rf" if swing == "lf" else "lf"
    H_swing_0 = None
    H_swing_1 = None
    com_from = com0.copy()
    com_to = com0.copy()
    planted = True
    step_log = []

    # print(f"  {'t':>6}  {'st':>2} {'phase':>6}  {'com_x':>7}  {'com_y':>7}  "
    #       f"{'fz_sw':>6}  {'fz_st':>6}  {'roll':>6}  {'|tau|':>6}  {'ncon':>4}")
    # print("  " + "-" * 80)
    print(f"  {'t':>6}  {'st':>2} {'phase':>6}  {'com_x':>7}  {'com_y':>7}  "
          f"{'fz_sw':>6}  {'fz_st':>6}  {'roll':>6}  {'pitch':>6}  "
          f"{'nc':>5} {'|tau|':>6}  {'ncon':>4}")
    print("  " + "-" * 95)

    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None
    wall_start = time.time()

    for i in range(n_steps):
        if viewer is not None and not viewer.is_running():
            print("\n  viewer closed by user")
            break

        t = d.time
        q = mj_to_pin_q(d, mapping, pm)
        v = mj_to_pin_v(d, mapping, pm)

        # which phase now
        if t < args.steps * T_step:
            k = int(t / T_step)              # step index
            tau_in = t - k * T_step          # time within this step
            new_phase = "SHIFT" if tau_in < args.t_shift else "SWING"
        else:
            k = args.steps
            tau_in = t - args.steps * T_step
            new_phase = "SETTLE"

        # phase transitions
        if k != step_idx or new_phase != phase:

            # leaving a SWING means the foot has arrived plant it
            if phase == "SWING" and not planted:
                ctrl.make_contact(swing)
                planted = True
                H_land = ctrl.foot_placement(swing)
                step_log.append((step_idx, swing, H_land.translation[0]))
                print(f"      planted {swing} at x={H_land.translation[0]:+.4f}")
                # roles swap for the next step
                swing, stance = stance, swing

            if new_phase == "SHIFT":
                # CoM goes from wherever it is heading now
                com_from = com_to.copy()
                st_pos = ctrl.foot_placement(stance).translation
                com_to = np.array([st_pos[0] + com_x_offset,
                                   com0[1] + args.shift_frac
                                   * (st_pos[1] - com0[1]),
                                   com0[2]])

            elif new_phase == "SWING":
                H_swing_0 = ctrl.foot_placement(swing)
                st_x = ctrl.foot_placement(stance).translation[0]
                target_x = st_x + args.step_length
                H_swing_1 = pin.SE3(
                    H_swing_0.rotation.copy(),
                    np.array([target_x,
                              H_swing_0.translation[1],
                              H_swing_0.translation[2] - args.land_bias]))
                ctrl.break_contact(swing, args.transition)
                planted = False

            elif new_phase == "SETTLE":
                com_from = com_to.copy()
                lf_p = ctrl.foot_placement("lf").translation
                rf_p = ctrl.foot_placement("rf").translation
                com_to = np.array([0.5 * (lf_p[0] + rf_p[0]) + com_x_offset,
                                   com0[1], com0[2]])

            step_idx, phase = k, new_phase

        # set references
        if phase == "SHIFT":
            p, vel, acc = interp(com_from, com_to, tau_in / args.t_shift,
                                 args.t_shift)
            ctrl.set_com_reference(p, vel, acc)

        elif phase == "SWING":
            u = (tau_in - args.t_shift) / args.t_swing
            H, v6, a6 = swing_pose(u, args.t_swing, H_swing_0,
                                   H_swing_1, args.step_height)
            ctrl.set_foot_reference(swing, H, v6, a6)
            ctrl.set_com_reference(com_to)

        else:  # SETTLE
            p, vel, acc = interp(com_from, com_to, tau_in / args.t_settle,
                                 args.t_settle)
            ctrl.set_com_reference(p, vel, acc)

        # control
        tau, status = ctrl.compute(t, q, v)
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
            print(f"\n diverged at t={d.time:.3f} in {phase} step {step_idx}")
            if viewer is not None:
                viewer.close()
            return

        if d.qpos[2] < 0.55:
            print(f"\n fell at t={d.time:.3f} in {phase} step {step_idx} "
                  f"(pelvis {d.qpos[2]:.3f} m)")
            if viewer is not None:
                viewer.close()
            break

        if viewer is not None:
            viewer.sync()
            lag = d.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

        # if i % report == 0:
        #     qn = mj_to_pin_q(d, mapping, pm)
        #     com = pin.centerOfMass(pm, pm_data, qn)
        #     tn = 0.0 if tau is None else np.linalg.norm(tau)
        #     fz = ctrl.contact_normal_forces()
        #     w_, x_, y_, z_ = d.qpos[3:7]
        #     roll = np.degrees(np.arctan2(2*(w_*x_ + y_*z_),
        #                                  1 - 2*(x_*x_ + y_*y_)))
        #     print(f"  {d.time:6.3f}  {step_idx:2d} {phase:>6}  {com[0]:+7.4f}  "
        #           f"{com[1]:+7.4f}  {fz[swing]:6.1f}  {fz[stance]:6.1f}  "
        #           f"{roll:+6.2f}  {tn:6.2f}  {d.ncon:4d}")

        if i % report == 0:
            qn = mj_to_pin_q(d, mapping, pm)
            com = pin.centerOfMass(pm, pm_data, qn)
            tn = 0.0 if tau is None else np.linalg.norm(tau)
            fz = ctrl.contact_normal_forces()

            w_, x_, y_, z_ = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2*(w_*x_ + y_*z_),
                                         1 - 2*(x_*x_ + y_*y_)))
            pitch = np.degrees(np.arcsin(np.clip(2*(w_*y_ - z_*x_), -1, 1)))

            # how many MuJoCo contacts each foot actually has. Four spheres
            # per foot, so 4 means flat, 2 means resting on one edge.
            nc = {"lf": 0, "rf": 0}
            for s in ("lf", "rf"):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, FRAMES[s])
                for ci in range(d.ncon):
                    if bid in (m.geom_bodyid[d.contact[ci].geom1],
                               m.geom_bodyid[d.contact[ci].geom2]):
                        nc[s] += 1

            print(f"  {d.time:6.3f}  {step_idx:2d} {phase:>6}  {com[0]:+7.4f}  "
                  f"{com[1]:+7.4f}  {fz[swing]:6.1f}  {fz[stance]:6.1f}  "
                  f"{roll:+6.2f}  {pitch:+6.2f}  {nc[swing]:2d}/{nc[stance]:<2d} "
                  f"{tn:6.2f}  {d.ncon:4d}")

    if viewer is not None:
        viewer.close()

    qn = mj_to_pin_q(d, mapping, pm)
    com = pin.centerOfMass(pm, pm_data, qn)
    pin.forwardKinematics(pm, pm_data, qn)
    pin.updateFramePlacements(pm, pm_data)
    lf_x = pm_data.oMf[pm.getFrameId(FRAMES["lf"])].translation[0]
    rf_x = pm_data.oMf[pm.getFrameId(FRAMES["rf"])].translation[0]

    print()
    print(f"steps completed: {len(step_log)} / {args.steps}")
    for idx, side, x in step_log:
        print(f"step {idx}: {side} -> x={x:+.4f}")

    print(f"foot drift: ", end="")
    for idx, side, x_planted in step_log:
        x_now = pm_data.oMf[pm.getFrameId(FRAMES[side])].translation[0]
        print(f"{side}@{idx}: {(x_now - x_planted)*1000:+.1f}mm  ", end="")
    print()

    print(f"final pelvis h: {d.qpos[2]:.4f} m  (started {PELVIS_H})")
    print(f"final CoM x: {com[0]:+.4f}  (started {com0[0]:+.4f})")
    # print(f"  travelled       : {com[0]-com0[0]:+.4f} m  "
    #       f"(expected {args.steps*args.step_length:.3f})")
    expected = args.steps * args.step_length - 0.5 * args.step_length
    print(f"travelled: {com[0]-com0[0]:+.4f} m  "
          f"(expected {expected:.3f}, feet leapfrog so the body ends "
          f"half a step behind the lead foot)")
    print(f"foot x  lf/rf: {lf_x:+.4f} / {rf_x:+.4f}")
    print(f"contacts at end: {d.ncon}")
    print(f"QP failures: {fails} / {n_steps}")
    print()

    upright = d.qpos[2] > 0.70
    all_steps = len(step_log) == args.steps
    print(f"upright: {'YES' if upright else 'NO'}")
    print(f"all steps taken: {'YES' if all_steps else 'NO'}")
    if upright and all_steps and fails == 0:
        print("\n M6a PASSED continuous quasi-static walking.")
    else:
        print("\n  M6a FAILED.")


if __name__ == "__main__":
    main()