#!/usr/bin/env python3
"""
M6b -- LIPM walking, with optional DCM feedback.
Use the precomputed LIPM trajectory and DCM feedback to control the unstable component.

Usage:
    python tests/m6b_walk_lipm.py --dcm
    python tests/m6b_walk_lipm.py            
    python tests/m6b_walk_lipm.py --dcm --view
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
TRAJ = "results/g1_walking_traj_tsid.npz"
GRAVITY = 9.81

SWING_OF_PHASE = {0: None, 1: "lf", 2: "rf"}


def build_actuated_ctrl_map(mj_model, mapping):
    out = []
    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out.append((idx_v - 6, aid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=TRAJ)
    ap.add_argument("--dcm", action="store_true",
                    help="use DCM feedback instead of plain CoM PD")
    ap.add_argument("--k-dcm", type=float, default=2.0,
                    help="DCM feedback gain; must exceed 1 for stability")
    ap.add_argument("--transition", type=float, default=0.05)
    ap.add_argument("--land-bias", type=float, default=0.0)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    if args.dcm and args.k_dcm <= 1.0:
        print(f"  !! k_dcm = {args.k_dcm} does not exceed 1; the DCM error "
              f"would diverge. Refusing to run.")
        return

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

    #  models 
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pm_data = pm.createData()
    mapping = build_joint_map(m, pm)
    ctrl_map = build_actuated_ctrl_map(m, mapping)

    if abs(traj_dt - m.opt.timestep) > 1e-9:
        print(f" trajectory dt {traj_dt} != MuJoCo timestep "
              f"{m.opt.timestep}")
        return

    #  initial state 
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

    # DCM reference, computed once from the plan 
    # xi_ref = c_ref + c_ref_dot / omega
    # z_ref  = xi_ref - xi_ref_dot / omega     (inverting xi_dot = w(xi - z))
    com_p_w = com_p + com_offset                    # plan in world frame
    xi_ref = com_p_w + com_v / omega
    xi_ref_dot = np.gradient(xi_ref, traj_dt, axis=0)
    z_ref = xi_ref - xi_ref_dot / omega

    mode = f"DCM feedback, k = {args.k_dcm}" if args.dcm else "plain CoM PD"
    print(f"trajectory: {args.traj}")
    print(f"samples: {N}  ({N * traj_dt:.2f} s at dt={traj_dt})")
    print(f"CoM height: {h:.4f} m   omega = {omega:.4f} rad/s")
    print(f"mode: {mode}")
    if args.dcm:
        decay = omega * (args.k_dcm - 1.0)
        print(f"DCM error decays as exp(-{decay:.3f} t), "
              f"time constant {1.0/decay:.3f} s")
    print(f"  phase counts : 0={int((phase_seq==0).sum())} "
          f"1={int((phase_seq==1).sum())} 2={int((phase_seq==2).sum())}\n")

    fails = 0
    prev_phase = int(phase_seq[0])
    switches = []
    report = max(1, N // 24)
    max_dcm_err = 0.0

    print(f" {'t':>6}  {'ph':>2}  {'com_y':>7} {'ref_y':>7}  "
          f"{'dcm_e':>6}  {'fz_lf':>6} {'fz_rf':>6}  {'roll':>6}  "
          f"{'|tau|':>6}  {'ncon':>4}")
    print("  " + "-" * 82)

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
        # measured CoM state
        pin.centerOfMass(pm, pm_data, q, v)
        c_meas = np.array(pm_data.com[0]).flatten()
        cd_meas = np.array(pm_data.vcom[0]).flatten()
        if abs(d.time - 1.500) < 0.0011:
            print(f"LIFTOFF  com_y {c_meas[1]:+.5f} ref {com_p_w[i,1]:+.5f}"
                  f"vy {cd_meas[1]:+.5f} ref {com_v[i,1]:+.5f}")

        xi = c_meas + cd_meas / omega
        dcm_err = np.linalg.norm((xi - xi_ref[i])[:2])
        max_dcm_err = max(max_dcm_err, dcm_err)

        if args.dcm:
            # z_cmd = z_ref + k (xi - xi_ref);  c_ddot = omega^2 (c - z_cmd)
            z_cmd = z_ref[i] + args.k_dcm * (xi - xi_ref[i])
            acc_cmd = omega ** 2 * (c_meas - z_cmd)

            # x,y: reference = measured, so the task PD contributes nothing
            # and acc_cmd alone acts. z: normal PD onto the planned height.
            pos_ref = np.array([c_meas[0], c_meas[1], com_p_w[i, 2]])
            vel_ref = np.array([cd_meas[0], cd_meas[1], 0.0])
            acc_ref = np.array([acc_cmd[0], acc_cmd[1], 0.0])
            ctrl.set_com_reference(pos_ref, vel_ref, acc_ref)
        else:
            ctrl.set_com_reference(com_p_w[i], com_v[i], com_a[i])

        #  foot reference
        sw = SWING_OF_PHASE[ph]
        if sw is not None:
            target = foot_p[sw][i] + foot_offset[sw]
            target = target - np.array([0.0, 0.0, args.land_bias])
            H = pin.SE3(ctrl.foot_placement(sw).rotation.copy(), target)
            ctrl.set_foot_reference(
                sw, H,
                np.concatenate([foot_v[sw][i], np.zeros(3)]),
                np.concatenate([foot_a[sw][i], np.zeros(3)]))

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
            print(f"\n diverged at t={d.time:.3f}, phase {ph}")
            break
        if d.qpos[2] < 0.55:
            print(f"\n fell at t={d.time:.3f}, phase {ph} "
                  f"(pelvis {d.qpos[2]:.3f} m)")
            break

        if viewer is not None:
            viewer.sync()
            lag = d.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

        if i % report == 0:
            fz = ctrl.contact_normal_forces()
            w_, x_, y_, z_ = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2*(w_*x_ + y_*z_),
                                         1 - 2*(x_*x_ + y_*y_)))
            tn = 0.0 if tau is None else np.linalg.norm(tau)
            cop = {}
            for s in ("lf", "rf"):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, FRAMES[s])
                num = 0.0; den = 0.0
                for ci in range(d.ncon):
                    con = d.contact[ci]
                    if bid not in (m.geom_bodyid[con.geom1],
                                   m.geom_bodyid[con.geom2]):
                        continue
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, f6)
                    fn = f6[0]
                    num += fn * con.pos[1]
                    den += fn
                cop[s] = num / den if den > 1e-6 else float('nan')

            # how close is the stance CoP to the outer edge of its foot
            st = "rf" if ph == 1 else ("lf" if ph == 2 else None)
            if st is not None and not np.isnan(cop[st]):
                foot_y = pm_data.oMf[pm.getFrameId(FRAMES[st])].translation[1]
                edge = foot_y - 0.03 if foot_y < 0 else foot_y + 0.03
                margin = abs(cop[st] - edge) * 1000.0
            else:
                margin = float('nan')
            print(f"  {d.time:6.3f}  {ph:2d}  {c_meas[1]:+7.4f} "
                  f"{com_p_w[i,1]:+7.4f}  {dcm_err:6.4f}  {fz['lf']:6.1f} "
                  f"{cop.get(st, float('nan')):+7.4f} {margin:6.1f}  "
                  f"{fz['rf']:6.1f}  {roll:+6.2f}  {tn:6.2f}  {d.ncon:4d}")

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
        print(f"      {t_s:6.3f}  {what}")

    k = min(n_done - 1, N - 1)
    print()
    print(f"mode: {mode}")
    print(f"completed: {n_done} / {N} samples "
          f"({n_done*traj_dt:.2f} s)")
    print(f"final pelvis h: {d.qpos[2]:.4f} m  (started {PELVIS_H})")
    print(f"CoM  actual/ref: x {com[0]:+.4f} / {com_p_w[k,0]:+.4f}   "
          f"y {com[1]:+.4f} / {com_p_w[k,1]:+.4f}")
    print(f"travelled: {com[0]-com0[0]:+.4f} m")
    print(f"max DCM error: {max_dcm_err*1000:.1f} mm")
    for s in ("lf", "rf"):
        p = pm_data.oMf[pm.getFrameId(FRAMES[s])].translation
        print(f"{s} final: x {p[0]:+.4f}  z {p[2]:.4f}")
    print(f"contacts at end : {d.ncon}")
    print(f"QP failures: {fails} / {n_done}")
    print()

    upright = d.qpos[2] > 0.70
    finished = n_done == N
    print(f"upright: {'YES' if upright else 'NO'}")
    print(f"ran to the end: {'YES' if finished else 'NO'}")
    if upright and finished and fails == 0:
        print("\n  M6b PASSED LIPM walking tracked in closed loop.")
    else:
        print("\n  M6b FAILED.")


if __name__ == "__main__":
    main()