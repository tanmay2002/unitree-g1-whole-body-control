#!/usr/bin/env python3
"""
M5 single step.

Test mid-run contact switching: shift CoM, swing the foot, then settle.

Usage:
    python tests/m5_single_step.py --step-length 0 --view
    python tests/m5_single_step.py --step-length 0.10 --view
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
    """Minimum-jerk trajectory with zero velocity and acceleration at both ends for a smooth landing."""
    u = np.clip(u, 0.0, 1.0)
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = 30*u**2 - 60*u**3 + 30*u**4
    dds = 60*u - 180*u**2 + 120*u**3
    return s, ds, dds


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
    ap.add_argument("--swing", default="lf", choices=["lf", "rf"])
    ap.add_argument("--step-length", type=float, default=0.0)
    ap.add_argument("--step-height", type=float, default=0.05)
    ap.add_argument("--t-shift", type=float, default=1.5)
    ap.add_argument("--t-swing", type=float, default=1.2)
    ap.add_argument("--t-settle", type=float, default=2.0)
    ap.add_argument("--shift-frac", type=float, default=0.85,
                    help="how far toward the stance foot the CoM goes")
    ap.add_argument("--transition", type=float, default=0.05,
                    help="contact force ramp-down time [s]")
    ap.add_argument("--land-bias", type=float, default=0.003,
                    help="aim the SWING TARGET this far below the lift "
                         "height. Never applied to the contact reference.")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    stance = "rf" if args.swing == "lf" else "lf"

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
    H_stance = ctrl.foot_placement(stance)

    # captured at LIFT time
    H_swing_0 = None
    H_swing_1 = None

    com_shift = com0.copy()
    com_shift[1] = com0[1] + args.shift_frac * (H_stance.translation[1] - com0[1])

    t_lift = args.t_shift
    t_plant = t_lift + args.t_swing
    t_end = t_plant + args.t_settle

    print(f"swing foot: {args.swing}   stance: {stance}")
    print(f"step length: {args.step_length:.3f} m   "
          f"height {args.step_height:.3f} m   land bias {args.land_bias:.3f} m")
    print(f"CoM shift: y {com0[1]:+.4f} -> {com_shift[1]:+.4f} "
          f"(stance foot at {H_stance.translation[1]:+.4f})")
    print(f"phases: shift 0-{t_lift}s, swing {t_lift}-{t_plant}s, "
          f"settle {t_plant}-{t_end}s\n")

    n_steps = int(t_end / m.opt.timestep)
    report = max(1, n_steps // 16)
    fails = 0
    phase = "SHIFT"
    planted = False
    max_swing_z = 0.0

    print(f"{'t':>6}  {'phase':>7}  {'com_y':>8}  {'foot_z':>8}  "
          f"{'fz_sw':>7}  {'fz_st':>7}  {'roll':>7}  {'|tau|':>7}  {'ncon':>5}")
    print("  " + "-" * 78)

    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None
    wall_start = time.time()

    for i in range(n_steps):
        if viewer is not None and not viewer.is_running():
            print("\n  viewer closed by user")
            break

        t = d.time
        q = mj_to_pin_q(d, mapping, pm)
        v = mj_to_pin_v(d, mapping, pm)

        # phase logic
        if t < t_lift:
            phase = "SHIFT"
            u = t / args.t_shift
            s, ds, dds = quintic(u)
            ctrl.set_com_reference(
                com0 + (com_shift - com0) * s,
                (com_shift - com0) * ds / args.t_shift,
                (com_shift - com0) * dds / (args.t_shift ** 2))

        elif t < t_plant:
            phase = "SWING"
            if ctrl.in_contact[args.swing]:
                # read the foot where it actually is
                H_swing_0 = ctrl.foot_placement(args.swing)
                # swing target is biased below the ground on purpose
                H_swing_1 = pin.SE3(
                    H_swing_0.rotation.copy(),
                    H_swing_0.translation + np.array([args.step_length, 0.0,
                                                      -args.land_bias]))
                ctrl.break_contact(args.swing, args.transition)
                print(f"released  t={t:.3f}  z={H_swing_0.translation[2]:.4f}"
                      f"  x={H_swing_0.translation[0]:+.4f}")
            u = (t - t_lift) / args.t_swing
            H, v6, a6 = swing_pose(u, args.t_swing, H_swing_0,
                                   H_swing_1, args.step_height)
            ctrl.set_foot_reference(args.swing, H, v6, a6)
            ctrl.set_com_reference(com_shift)

        else:
            phase = "SETTLE"
            if not planted:
                # NO bias here: plant at the foot's true pose
                ctrl.make_contact(args.swing)
                H_land = ctrl.foot_placement(args.swing)
                c_chk = ctrl.contacts[args.swing]
                print(f"fMin={c_chk.getMinNormalForce:.3f}  "
                      f"fMax={c_chk.getMaxNormalForce:.3f}")
                planted = True
                print(f"planted   t={t:.3f}  z={H_land.translation[2]:.4f}"
                      f"x={H_land.translation[0]:+.4f}")
            u = (t - t_plant) / args.t_settle
            s, ds, dds = quintic(u)
            com_end = com0.copy()
            com_end[0] = com0[0] + 0.5 * args.step_length
            ctrl.set_com_reference(
                com_shift + (com_end - com_shift) * s,
                (com_end - com_shift) * ds / args.t_settle,
                (com_end - com_shift) * dds / (args.t_settle ** 2))

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
            print(f"\n diverged at t={d.time:.3f} in phase {phase}")
            if viewer is not None:
                viewer.close()
            return

        if viewer is not None:
            viewer.sync()
            lag = d.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

        if i % report == 0:
            qn = mj_to_pin_q(d, mapping, pm)
            com = pin.centerOfMass(pm, pm_data, qn)
            pin.forwardKinematics(pm, pm_data, qn)
            pin.updateFramePlacements(pm, pm_data)
            fp = pm_data.oMf[pm.getFrameId(FRAMES[args.swing])].translation
            max_swing_z = max(max_swing_z, fp[2])
            tn = 0.0 if tau is None else np.linalg.norm(tau)

            fz = ctrl.contact_normal_forces()      # order-aware
            w_, x_, y_, z_ = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2*(w_*x_ + y_*z_),
                                         1 - 2*(x_*x_ + y_*y_)))

            print(f"{d.time:6.3f}  {phase:>7}  {com[1]:+8.4f}  "
                  f"{fp[2]:8.4f}  {fz[args.swing]:7.1f}  {fz[stance]:7.1f}  "
                  f"{roll:+7.2f}  {tn:7.2f}  {d.ncon:5d}")

    if viewer is not None:
        viewer.close()

    qn = mj_to_pin_q(d, mapping, pm)
    pin.forwardKinematics(pm, pm_data, qn)
    pin.updateFramePlacements(pm, pm_data)
    fp = pm_data.oMf[pm.getFrameId(FRAMES[args.swing])].translation

    print()
    print(f"final pelvis height: {d.qpos[2]:.4f} m")
    if H_swing_1 is not None:
        target_x = H_swing_1.translation[0]
        print(f"swing foot x: {fp[0]:+.4f}  (target {target_x:+.4f}, "
              f"error {fp[0]-target_x:+.4f} m)")
        placed = abs(fp[0] - target_x) < 0.02
    else:
        print(f"swing foot x: {fp[0]:+.4f}  (never lifted)")
        placed = False
    print(f"swing foot z: {fp[2]:.4f} m")
    print(f"peak swing height: {max_swing_z:.4f} m")
    print(f"contacts at end: {d.ncon}")
    print(f"QP failures: {fails} / {n_steps}")
    print()

    upright = d.qpos[2] > 0.70
    landed = d.ncon >= 8
    print(f"upright: {'YES' if upright else 'NO'}")
    print(f"both feet down: {'YES' if landed else 'NO'}")
    print(f"foot placed: {'YES' if placed else 'NO'}")
    if upright and landed and placed and fails == 0:
        print("\n  M5 PASSED one step taken.")
    else:
        print("\n  M5 FAILED.")


if __name__ == "__main__":
    main()