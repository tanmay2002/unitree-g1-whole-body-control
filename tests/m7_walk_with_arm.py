#!/usr/bin/env python3
"""
M7 LIPM walking with a simultaneous upper-body task.

Walk with a sinusoidal hand task to test whether lower-priority arm motion affects balance.
Usage:
    python tests/m7_walk_with_arm.py --arm
    python tests/m7_walk_with_arm.py                  # baseline, no arm
    python tests/m7_walk_with_arm.py --arm --view
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
                                   STANDING_POSE, FRAMES, HAND_FRAMES)

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"
TRAJ = "results/g1_walking_traj_tsid.npz"
GRAVITY = 9.81

SWING_OF_PHASE = {0: None, 1: "lf", 2: "rf"}
AXES = {"x": 0, "y": 1, "z": 2}


def build_actuated_ctrl_map(mj_model, mapping):
    out = []
    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out.append((idx_v - 6, aid))
    return out


def arm_sinusoid(t, amp, freq, ramp, axis_i):
    """Offset, velocity and acceleration of the hand sinusoid.

  Ramp the amplitude linearly and include the ramp term when computing derivatives.
    """
    w = 2.0 * np.pi * freq
    r = min(1.0, t / ramp) if ramp > 0 else 1.0
    rd = (1.0 / ramp) if (ramp > 0 and t < ramp) else 0.0
    s, c = np.sin(w * t), np.cos(w * t)

    p = np.zeros(3); v = np.zeros(3); a = np.zeros(3)
    p[axis_i] = amp * r * s
    v[axis_i] = amp * (rd * s + r * w * c)
    a[axis_i] = amp * (2.0 * rd * w * c - r * w * w * s)
    return p, v, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=TRAJ)
    ap.add_argument("--arm", action="store_true",
                    help="enable the hand task (omit for the baseline)")
    ap.add_argument("--hand", default="lh", choices=["lh", "rh"])
    ap.add_argument("--arm-amp", type=float, default=0.10,
                    help="sinusoid amplitude [m]")
    ap.add_argument("--arm-freq", type=float, default=0.3,
                    help="sinusoid frequency [Hz]")
    ap.add_argument("--arm-axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--arm-ramp", type=float, default=2.0)
    ap.add_argument("--transition", type=float, default=0.05)
    ap.add_argument("--land-bias", type=float, default=0.0)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    axis_i = AXES[args.arm_axis]

    # trajectory 
    tr = np.load(args.traj)
    com_p = tr["com_traj"]
    com_v = tr["com_vel_traj"]
    foot_p = {"lf": tr["lf_traj"], "rf": tr["rf_traj"]}
    foot_v = {"lf": tr["lf_vel_traj"], "rf": tr["rf_vel_traj"]}
    foot_a = {"lf": tr["lf_acc_traj"], "rf": tr["rf_acc_traj"]}
    phase_seq = tr["contact_phase"]
    traj_dt = float(tr["dt"])
    N = com_p.shape[0]

    com_a = np.gradient(com_v, traj_dt, axis=0)
    h = float(com_p[0, 2])
    omega = np.sqrt(GRAVITY / h)

    # models 
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pm_data = pm.createData()
    mapping = build_joint_map(m, pm)
    ctrl_map = build_actuated_ctrl_map(m, mapping)

    if abs(traj_dt - m.opt.timestep) > 1e-9:
        print(f"  !! trajectory dt {traj_dt} != MuJoCo timestep "
              f"{m.opt.timestep}")
        return

    # initial state 
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

    com_offset = com0 - com_p[0]
    foot_offset = {s: ctrl.foot_placement(s).translation - foot_p[s][0]
                   for s in ("lf", "rf")}
    com_p_w = com_p + com_offset

    xi_ref = com_p_w + com_v / omega

    # hand setup
    hand0 = ctrl.hand_placement(args.hand).translation.copy()
    hand_offset = hand0 - com0          # hand pose relative to the CoM
    if args.arm:
        ctrl.add_hand_task(args.hand)

    mode = (f"walking + {args.hand} sinusoid, {args.arm_amp:.3f} m at "
            f"{args.arm_freq} Hz along {args.arm_axis}"
            if args.arm else "walking only (baseline, no arm task)")
    print(f"trajectory: {args.traj}")
    print(f"samples: {N}  ({N * traj_dt:.2f} s at dt={traj_dt})")
    print(f"mode: {mode}")
    if args.arm:
        print(f"hand frame: {HAND_FRAMES[args.hand]}")
        print(f"w_hand {ctrl.w_hand} vs w_com {ctrl.w_com} "
              f"vs w_posture {ctrl.w_posture}")
        print(f"hand offset from CoM : {hand_offset}")
    print(f"phase counts : 0={int((phase_seq==0).sum())} "
          f"1={int((phase_seq==1).sum())} 2={int((phase_seq==2).sum())}\n")

    fails = 0
    prev_phase = int(phase_seq[0])
    switches = []
    report = max(1, N // 22)
    max_dcm_err = 0.0
    hand_err_sum = 0.0
    hand_err_max = 0.0
    hand_n = 0

    hdr = (f"  {'t':>6}  {'ph':>2}  {'com_y':>7} {'ref_y':>7}  {'dcm_e':>6}  "
           f"{'fz_lf':>6} {'fz_rf':>6}  {'|tau|':>6}  {'ncon':>4}")
    if args.arm:
        hdr += f"  {'hand_x':>7} {'href_x':>7} {'h_err':>6}"
    print(hdr)
    print("  " + "-" * (100 if args.arm else 74))

    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None
    wall_start = time.time()
    i = 0

    for i in range(N):
        if viewer is not None and not viewer.is_running():
            print("\n  viewer closed by user")
            break

        ph = int(phase_seq[i])
        q = mj_to_pin_q(d, mapping, pm)
        v = mj_to_pin_v(d, mapping, pm)

        # contact switching
        if ph != prev_phase:
            leaving = SWING_OF_PHASE[prev_phase]
            entering = SWING_OF_PHASE[ph]
            if leaving is not None and leaving != entering:
                ctrl.make_contact(leaving)
                switches.append((d.time, f"plant {leaving}"))
            if entering is not None and entering != leaving:
                ctrl.break_contact(entering, args.transition)
                switches.append((d.time, f"lift  {entering}"))
            prev_phase = ph

        # CoM reference
        ctrl.set_com_reference(com_p_w[i], com_v[i], com_a[i])

        # foot reference
        sw = SWING_OF_PHASE[ph]
        if sw is not None:
            target = foot_p[sw][i] + foot_offset[sw]
            target = target - np.array([0.0, 0.0, args.land_bias])
            H = pin.SE3(ctrl.foot_placement(sw).rotation.copy(), target)
            ctrl.set_foot_reference(
                sw, H,
                np.concatenate([foot_v[sw][i], np.zeros(3)]),
                np.concatenate([foot_a[sw][i], np.zeros(3)]))

        # hand reference
        # anchored to the PLANNED CoM so it walks with the robot
        if args.arm:
            dp, dv, da = arm_sinusoid(d.time, args.arm_amp, args.arm_freq,
                                      args.arm_ramp, axis_i)
            hand_ref = com_p_w[i] + hand_offset + dp
            hand_vel = com_v[i] + dv
            hand_acc = com_a[i] + da
            ctrl.set_hand_reference(args.hand, hand_ref, hand_vel, hand_acc)

        # control
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
            print(f"\n  !! diverged at t={d.time:.3f}, phase {ph}")
            break
        if d.qpos[2] < 0.55:
            print(f"\n  !! fell at t={d.time:.3f}, phase {ph} "
                  f"(pelvis {d.qpos[2]:.3f} m)")
            break

        # metrics
        qn = mj_to_pin_q(d, mapping, pm)
        vn = mj_to_pin_v(d, mapping, pm)
        pin.centerOfMass(pm, pm_data, qn, vn)
        c_meas = np.array(pm_data.com[0]).flatten()
        cd_meas = np.array(pm_data.vcom[0]).flatten()
        xi = c_meas + cd_meas / omega
        dcm_err = np.linalg.norm((xi - xi_ref[i])[:2])
        max_dcm_err = max(max_dcm_err, dcm_err)

        h_err = np.nan
        if args.arm:
            pin.forwardKinematics(pm, pm_data, qn)
            pin.updateFramePlacements(pm, pm_data)
            hp = pm_data.oMf[pm.getFrameId(HAND_FRAMES[args.hand])].translation
            h_err = np.linalg.norm(hp - hand_ref)
            if d.time > args.arm_ramp:
                hand_err_sum += h_err ** 2
                hand_err_max = max(hand_err_max, h_err)
                hand_n += 1

        if viewer is not None:
            viewer.sync()
            lag = d.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

        if i % report == 0:
            fz = ctrl.contact_normal_forces()
            tn = 0.0 if tau is None else np.linalg.norm(tau)
            line = (f"{d.time:6.3f}  {ph:2d}  {c_meas[1]:+7.4f} "
                    f"{com_p_w[i,1]:+7.4f}  {dcm_err:6.4f}  {fz['lf']:6.1f} "
                    f"{fz['rf']:6.1f}  {tn:6.2f}  {d.ncon:4d}")
            if args.arm:
                line += (f"{hp[axis_i]:+7.4f} {hand_ref[axis_i]:+7.4f} "
                         f"{h_err:6.4f}")
            print(line)

    if viewer is not None:
        viewer.close()

    # summary
    n_done = i + 1
    qn = mj_to_pin_q(d, mapping, pm)
    pin.centerOfMass(pm, pm_data, qn)
    com = np.array(pm_data.com[0]).flatten()
    pin.forwardKinematics(pm, pm_data, qn)
    pin.updateFramePlacements(pm, pm_data)

    print("\n  contact switches:")
    for t_s, what in switches:
        print(f"{t_s:6.3f}  {what}")

    k = min(n_done - 1, N - 1)
    print()
    print(f"mode: {mode}")
    print(f"completed: {n_done} / {N} samples "
          f"({n_done*traj_dt:.2f} s)")
    print(f"final pelvis h: {d.qpos[2]:.4f} m  (started {PELVIS_H})")
    print(f"CoM  actual/ref: x {com[0]:+.4f} / {com_p_w[k,0]:+.4f}   "
          f"y {com[1]:+.4f} / {com_p_w[k,1]:+.4f}")
    print(f"travelled: {com[0]-com0[0]:+.4f} m")
    print(f"max DCM error: {max_dcm_err*1000:.1f} mm   "
          f"<-- compare with and without --arm")
    if args.arm and hand_n:
        print(f"hand RMS error: "
              f"{np.sqrt(hand_err_sum/hand_n)*1000:.1f} mm")
        print(f"hand max error: {hand_err_max*1000:.1f} mm")
    for s in ("lf", "rf"):
        p = pm_data.oMf[pm.getFrameId(FRAMES[s])].translation
        print(f"{s} final: x {p[0]:+.4f}  z {p[2]:.4f}")
    print(f"contacts at end: {d.ncon}")
    print(f"QP failures: {fails} / {n_done}")
    print()

    upright = d.qpos[2] > 0.70
    finished = n_done == N
    print(f"upright: {'YES' if upright else 'NO'}")
    print(f"ran to the end  : {'YES' if finished else 'NO'}")
    if upright and finished and fails == 0:
        label = "M7 PASSED walking with a simultaneous arm task." \
            if args.arm else "baseline OK (no arm task)."
        print(f"\n  {label}")
    else:
        print("\n  FAILED.")


if __name__ == "__main__":
    main()