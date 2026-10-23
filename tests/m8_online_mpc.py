#!/usr/bin/env python3
"""
M8 -- Online Herdt MPC walking, no arm task.

Re-solves the LIPM QP every 100ms from the measured CoM state
instead of an offline plan. Footsteps are decision variables
(Herdt 2010). Phase-lead fix: support-foot toggle fires one
MPC sample early, during double support.

Usage:
    python tests/m8_online_mpc.py --n-steps 12 --vx 0.2 --t-post 0
    python tests/m8_online_mpc.py --n-steps 12 --vx 0.2 --t-post 0 --view
"""

import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin
from quadprog import solve_qp
from herdt.automatic_lmpc import AutomaticLMPC

sys.path.insert(0, "src")

from g1_wbc.mj_pin_bridge import (
    build_joint_map,
    mj_to_pin_q,
    mj_to_pin_v,
)

from g1_wbc.wbc_controller import (
    G1BalanceController,
    PELVIS_H,
    STANDING_POSE,
    FRAMES,
    HAND_FRAMES,
)

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"

FOOT_Z = 0.035
DT = 0.002

def quintic(u):
    u = np.clip(u, 0.0, 1.0)

    return (
        10 * u**3 - 15 * u**4 + 6 * u**5,
        30 * u**2 - 60 * u**3 + 30 * u**4,
        60 * u - 180 * u**2 + 120 * u**3,
    )


def build_actuated_ctrl_map(mj_model, mapping):
    out = []

    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if aid >= 0:
            out.append((idx_v - 6, aid))

    return out


def triple_integrate(s, jerk, tau):
    """State [p, v, a] after time tau under constant jerk."""

    p, v, a = s

    return (
        p + v * tau + 0.5 * a * tau**2 + jerk * tau**3 / 6.0,
        v + a * tau + 0.5 * jerk * tau**2,
        a + jerk * tau,
    )

AXES = {"x": 0, "y": 1, "z": 2}

def main():
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--phase-lead", type=int, default=1,
                    help="MPC samples by which the planned ZMP switch leads the physical\n                    contact switch (1 = ramp during double support)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.00)
    ap.add_argument("--n-steps", type=int, default=6)

    ap.add_argument("--t-pre", type=float, default=2.5)
    ap.add_argument("--t-post", type=float, default=1.5)

    ap.add_argument(
        "--shift-frac",
        type=float,
        default=1.0,
        help="pre-phase CoM target as a fraction of the foot y",
    )

    ap.add_argument("--step-height", type=float, default=0.05)
    ap.add_argument("--transition", type=float, default=0.05)
    ap.add_argument("--land-bias", type=float, default=0.0)

    ap.add_argument("--view", action="store_true")

    
    ap.add_argument("--arm", action="store_true",
                    help="enable the hand task (omit for the baseline)")
    ap.add_argument("--hand", default="lh", choices=["lh", "rh"])
    ap.add_argument("--arm-amp", type=float, default=0.10,
                    help="hand sinusoid amplitude [m]; use 0.0 for a fixed trolley-push pose")
    ap.add_argument("--arm-freq", type=float, default=0.3)
    ap.add_argument("--arm-axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--arm-ramp", type=float, default=2.0)
    
    args = ap.parse_args()

    # MPC setup
    mpc = AutomaticLMPC()

    N = mpc.N
    m = mpc.m
    T = mpc.T
    T_step = mpc.T_step

    steps_per_foot = int(round(T_step / T))

    # Control steps per MPC sample
    n_mpc = int(round(T / DT))

    # Control steps per footstep
    n_step = int(round(T_step / DT))

    # Single-support duration in control steps
    n_ss = int(round(mpc.T_ss / DT))

    # Pre/post phase duration in control steps
    n_pre = int(round(args.t_pre / DT))
    n_post = int(round(args.t_post / DT))

    N_sim = (
        n_pre
        + args.n_steps * n_step
        + n_post
    )

    omega = np.sqrt(mpc.g / mpc.h)

    v_cmd = np.array([
        args.vx,
        args.vy,
    ])

    mpc.set_ref_velocity(v_cmd)
    
    print(f"  X_dot_ref_k[:3] = {mpc.X_dot_ref_k[:3]}")
    print(f"  Y_dot_ref_k[:3] = {mpc.Y_dot_ref_k[:3]}")


    # Models
    mj = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(mj)

    if abs(DT - mj.opt.timestep) > 1e-9:
        print(
            f"  !! DT {DT} != MuJoCo timestep "
            f"{mj.opt.timestep}"
        )
        return

    pm = pin.buildModelFromUrdf(
        URDF,
        pin.JointModelFreeFlyer(),
    )

    pm_data = pm.createData()

    mapping = build_joint_map(mj, pm)
    ctrl_map = build_actuated_ctrl_map(mj, mapping)


    # Initial robot configuration
    mujoco.mj_resetData(mj, d)

    d.qpos[2] = PELVIS_H
    d.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    for name, qadr, _, _, _ in mapping:
        for key, val in STANDING_POSE.items():
            if key in name:
                d.qpos[qadr] = val

    mujoco.mj_forward(mj, d)

    q0 = mj_to_pin_q(
        d,
        mapping,
        pm,
    )

    v0 = mj_to_pin_v(
        d,
        mapping,
        pm,
    )

    ctrl = G1BalanceController()

    com0 = ctrl.setup(
        q0,
        v0,
    )

    lf0 = ctrl.foot_placement(
        "lf"
    ).translation.copy()

    rf0 = ctrl.foot_placement(
        "rf"
    ).translation.copy()

    # Reference main.py assumes the right foot is the initial support
    x_fc = float(rf0[0])
    y_fc = float(rf0[1])

    # Pre-phase target:
    # CoM over the right foot, as the reference assumes
    com_pre_target = np.array([
        com0[0],
        args.shift_frac * rf0[1],
        com0[2],
    ])


    # Print 
    print(
        f"  planner          : AutomaticLMPC (reference), "
        f"N={N}, m={m}, T={T}, T_step={T_step} "
        f"(SS {mpc.T_ss} + DS {mpc.T_ds})"
    )

    print(
        f"  CoM height       : {mpc.h} m, "
        f"omega = {omega:.4f}"
    )

    print(
        f"  ZMP box half     : "
        f"x {mpc.foot_length / 2 - mpc.mpc_const.ZMP_TOLERANCE:.3f}"
        f"  y {mpc.foot_width / 2 - mpc.mpc_const.ZMP_TOLERANCE:.3f}"
    )

    print(
        f"  weights          : "
        f"alpha {mpc.mpc_const.ALPHA}  "
        f"beta {mpc.mpc_const.BETA}  "
        f"gamma {mpc.mpc_const.GAMMA}"
    )

    print(
        f"  v_cmd            : {v_cmd} m/s, "
        f"{args.n_steps} steps"
    )

    print(
        f"  initial feet     : "
        f"lf {lf0[:2].round(4)}  "
        f"rf {rf0[:2].round(4)}"
    )

    print(
        f"  pre-phase        : {args.t_pre} s, "
        f"CoM y -> {com_pre_target[1]:+.4f}\n"
    )


    # Online state
    k_mpc = -1
    u_opt = None
    a_pred = np.zeros(2)

    x_hat = np.array([
        com0[0],
        0.0,
        0.0,
    ])

    y_hat = np.array([
        com0[1],
        0.0,
        0.0,
    ])

    # States at the last solve, for the triple-integrator reference
    xs_at = None
    ys_at = None

    t_solve = 0.0

    jerk_x = 0.0
    jerk_y = 0.0

    swing = None
    swing_idx = 0
    swing_start = None
    i_lift = 0
    step_now = -1

    # (side, xy) of the most recent landing
    last_landed = None

    landed_log = []
    switches = []
    solve_times = []

    fails = 0
    max_dcm_err = 0.0

    report = max(
        1,
        N_sim // 26,
    )

    print(
        f"  {'t':>6}  {'st':>2} {'sw':>2}  "
        f"{'com_x':>7} {'com_y':>7}  "
        f"{'ref_y':>7}  {'dcm_e':>6}  "
        f"{'zmp_y':>7}  {'fz_lf':>6} "
        f"{'fz_rf':>6}  {'|tau|':>6} "
        f"{'ncon':>4}  {'next foot':>16}"
    )

    print(
        "  " + "-" * 110
    )

    viewer = (
        mujoco.viewer.launch_passive(mj, d)
        if args.view
        else None
    )

    wall_start = time.time()

    # Main loop
    for i in range(N_sim):

        if viewer is not None and not viewer.is_running():
            print("\n  viewer closed by user")
            break

        t = i * DT

        #current state
        q = mj_to_pin_q(
            d,
            mapping,
            pm,
        )

        v = mj_to_pin_v(
            d,
            mapping,
            pm,
        )

        pin.centerOfMass(
            pm,
            pm_data,
            q,
            v,
        )

        c_meas = np.array(
            pm_data.com[0]
        ).flatten()

        cd_meas = np.array(
            pm_data.vcom[0]
        ).flatten()

        walking = (
            n_pre
            <= i
            < n_pre + args.n_steps * n_step
        )

        # Gait machine
        if walking:
            s, r = divmod(
                i - n_pre,
                n_step,
            )

            if s != step_now:
                step_now = s

                swing = (
                    "lf"
                    if s % 2 == 0
                    else "rf"
                )

                swing_idx = s + 1

                swing_start = ctrl.foot_placement(
                    swing
                ).translation.copy()

                i_lift = i

                ctrl.break_contact(
                    swing,
                    args.transition,
                )

                switches.append(
                    (
                        t,
                        f"lift  {swing}  -> foot {swing_idx}",
                    )
                )

            if r == n_ss and swing is not None:
                ctrl.make_contact(swing)

                p_land = ctrl.foot_placement(
                    swing
                ).translation[:2].copy()

                planned = (
                    np.array([
                        u_opt[N],
                        u_opt[2 * N + m],
                    ])
                    if u_opt is not None
                    else None
                )

                last_landed = (
                    swing,
                    p_land,
                )

                landed_log.append(
                    (
                        swing_idx,
                        swing,
                        p_land,
                        planned,
                    )
                )

                switches.append(
                    (
                        t,
                        f"plant  {swing}  foot {swing_idx} "
                        f"at {p_land.round(4)}",
                    )
                )

                swing = None


        # CoM reference
        if i < n_pre:
            # Pre-phase: smoothstep weight shift.
            u = i / max(
                1,
                n_pre,
            )

            sq, dsq, ddsq = quintic(u)

            Tp = n_pre * DT

            ref_p = (
                com0
                + (com_pre_target - com0) * sq
            )

            ref_v = (
                (com_pre_target - com0)
                * dsq
                / Tp
            )

            ref_a = (
                (com_pre_target - com0)
                * ddsq
                / Tp**2
            )

            zmp_now = np.array([
                np.nan,
                np.nan,
            ])

        elif walking:
            k_now = (
                i - n_pre
            ) // n_mpc

            if k_now != k_mpc:

                if (
                    (k_now + args.phase_lead) % steps_per_foot == 0
                    and k_now + args.phase_lead > 0
                    and u_opt is not None
                ):
                    # Toggles support foot
                    mpc.get_next_footstep(u_opt)

                    # Commit the measured landing,
                    # not the planned one
                    if last_landed is not None:
                        x_fc = float(
                            last_landed[1][0]
                        )

                        y_fc = float(
                            last_landed[1][1]
                        )

                mpc.update_step_matrix(k_now + args.phase_lead)
                
                x_hat = np.array([c_meas[0], cd_meas[0], a_pred[0]])
                y_hat = np.array([c_meas[1], cd_meas[1], a_pred[1]])

                Q_k, p_k = mpc.compute_cost_function(
                    x_hat,
                    y_hat,
                    x_fc,
                    y_fc,
                )
                if k_now == 12:
                            eigenvalues = np.linalg.eigvalsh(Q_k)
                            print(f"  k=12 min_eig={eigenvalues.min():.4f} shift needed={max(0, -eigenvalues.min() + 1e-8):.4f}")
                            
                A_c, b_c = mpc.construct_constraints(
                    x_hat,
                    y_hat,
                    x_fc,
                    y_fc,
                )

                t0 = time.perf_counter()

                try:
                    u_opt = solve_qp(
                        Q_k,
                        -p_k,
                        A_c.T,
                        b_c,
                    )[0]

                except Exception as exc:
                    print(
                        f"  !! MPC failed at "
                        f"t={t:.3f}, k={k_now}: {exc}"
                    )

                    if u_opt is None:
                        return

                solve_times.append(
                    time.perf_counter() - t0
                )

                X_com, Y_com, Z_x, Z_y = (
                    mpc.compute_recursive_dynamics(
                        x_hat,
                        y_hat,
                        u_opt,
                    )
                )
                
                jerk_x = float(u_opt[0])
                jerk_y = float(u_opt[N + m])

                xs_at = x_hat.copy()
                ys_at = y_hat.copy()
                a_pred = np.array([X_com[0, 2], Y_com[0, 2]])

                t_solve = t
                k_mpc = k_now

                zmp_now = np.array([
                    Z_x[0],
                    Z_y[0],
                ])

            # Between samples
            tau = t - t_solve

            px, vx, ax = triple_integrate(xs_at, jerk_x, tau)
            py, vy, ay = triple_integrate(ys_at, jerk_y, tau)

            ref_p = np.array([px, py, com0[2]])
            ref_v = np.array([vx, vy, 0.0])
            ref_a = np.array([ax, ay, 0.0])
            
        else:
            
            # post-phase: shift CoM to midpoint between feet
            lf_pos = ctrl.foot_placement("lf").translation
            rf_pos = ctrl.foot_placement("rf").translation
            mid_y = (lf_pos[1] + rf_pos[1]) / 2.0
            mid_x = (lf_pos[0] + rf_pos[0]) / 2.0
            ref_p = np.array([mid_x, mid_y, com0[2]])
            ref_v = np.zeros(3)
            ref_a = np.zeros(3)
            zmp_now = np.array([mid_x, mid_y])
            
        ctrl.set_com_reference(
            ref_p,
            ref_v,
            ref_a,
        )

        # DCM error
        xi = (
            c_meas[:2]
            + cd_meas[:2] / omega
        )

        xi_ref = (
            ref_p[:2]
            + ref_v[:2] / omega
        )

        dcm_err = np.linalg.norm(
            xi - xi_ref
        )

        max_dcm_err = max(
            max_dcm_err,
            dcm_err,
        )

        # Swing reference
        if swing is not None and u_opt is not None:

            tgt_xy = np.array([
                u_opt[N],
                u_opt[2 * N + m],
            ])

            tgt = np.array([
                tgt_xy[0],
                tgt_xy[1],
                FOOT_Z - args.land_bias,
            ])

            u = (
                i - i_lift
            ) / n_ss

            sq, dsq, ddsq = quintic(u)

            T_sw = n_ss * DT

            pos = (
                swing_start
                + (tgt - swing_start) * sq
            ).copy()

            vel = (
                (tgt - swing_start)
                * dsq
                / T_sw
            ).copy()

            acc = (
                (tgt - swing_start)
                * ddsq
                / T_sw**2
            ).copy()

            w = np.pi

            pos[2] += (
                args.step_height
                * np.sin(w * u)**2
            )

            vel[2] += (
                args.step_height
                * w
                * np.sin(2 * w * u)
                / T_sw
            )

            acc[2] += (
                args.step_height
                * 2 * w * w
                * np.cos(2 * w * u)
                / T_sw**2
            )

            H = pin.SE3(
                ctrl.foot_placement(
                    swing
                ).rotation.copy(),
                pos,
            )

            ctrl.set_foot_reference(
                swing,
                H,
                np.concatenate([
                    vel,
                    np.zeros(3),
                ]),
                np.concatenate([
                    acc,
                    np.zeros(3),
                ]),
            )

        # Control
        tau_cmd, status = ctrl.compute(
            t,
            q,
            v,
        )

        if tau_cmd is None:
            fails += 1
            d.ctrl[:] = 0.0

        else:
            c = np.zeros(
                mj.nu
            )

            for i_act, aid in ctrl_map:
                c[aid] = tau_cmd[i_act]

            d.ctrl[:] = c


        # MuJoCo simulation
        mujoco.mj_step(
            mj,
            d,
        )

        if not np.all(
            np.isfinite(d.qpos)
        ):
            print(
                f"\n  !! diverged at t={t:.3f}"
            )
            break

        if d.qpos[2] < 0.55:
            print(
                f"\n  !! fell at t={t:.3f}, "
                f"step {step_now}, swing {swing} "
                f"(pelvis {d.qpos[2]:.3f} m)"
            )
            break

        # Viewer
        if viewer is not None:
            # ZMP box at the planned landing footstep
            with viewer.lock():
                scn = viewer.user_scn
                if swing is not None and u_opt is not None:
                    tol = mpc.mpc_const.ZMP_TOLERANCE
                    mujoco.mjv_initGeom(
                        scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([
                            mpc.foot_length / 2 - tol,
                            mpc.foot_width / 2 - tol,
                            0.002,
                        ]),
                        np.array([u_opt[N], u_opt[2 * N + m], 0.004]),
                        np.eye(3).flatten(),
                        np.array([0.1, 0.9, 0.2, 0.6], dtype=np.float32),
                    )
                    scn.ngeom = 1
                else:
                    scn.ngeom = 0

            viewer.sync()

            lag = (
                d.time
                - (time.time() - wall_start)
            )

            if lag > 0:
                time.sleep(lag)

        # Status output
        if i % report == 0:
            fz = ctrl.contact_normal_forces()

            tn = (
                0.0
                if tau_cmd is None
                else np.linalg.norm(tau_cmd)
            )

            nxt = (
                f"{swing_idx}: "
                f"[{u_opt[N]:+.3f} "
                f"{u_opt[2 * N + m]:+.3f}]"
                if swing is not None
                and u_opt is not None
                else "-"
            )

            print(
                f"  {t:6.3f}  "
                f"{step_now:2d} "
                f"{str(swing or '-'):>2}  "
                f"{c_meas[0]:+7.4f} "
                f"{c_meas[1]:+7.4f}  "
                f"{ref_p[1]:+7.4f}  "
                f"{dcm_err:6.4f}  "
                f"{zmp_now[1]:+7.4f}  "
                f"{fz['lf']:6.1f} "
                f"{fz['rf']:6.1f}  "
                f"{tn:6.2f} "
                f"{d.ncon:4d}  "
                f"{nxt:>16}"
            )
        

    # Close viewer
    if viewer is not None:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
        viewer.close()

    # Summary
    n_done = i + 1

    qn = mj_to_pin_q(
        d,
        mapping,
        pm,
    )

    pin.centerOfMass(
        pm,
        pm_data,
        qn,
    )

    com = np.array(
        pm_data.com[0]
    ).flatten()

    print("\n  contact switches:")

    for t_s, what in switches:
        print(
            f"      {t_s:6.3f}  {what}"
        )

    print(
        "\n  footsteps "
        "(index, foot, landed xy, MPC-planned xy, error mm):"
    )

    for f, side, land, plan in landed_log:
        e = (
            np.linalg.norm(land - plan) * 1000
            if plan is not None
            else np.nan
        )

        print(
            f"      {f:2d}  {side}  "
            f"{land.round(4)}  "
            f"{plan.round(4) if plan is not None else '-'}  "
            f"{e:6.1f}"
        )

    T_walk = args.n_steps * T_step

    print()

    print(
        f"  completed        : "
        f"{n_done} / {N_sim} samples "
        f"({n_done * DT:.2f} s)"
    )

    if solve_times:
        print(
            f"  MPC solves       : "
            f"{len(solve_times)}, mean "
            f"{np.mean(solve_times) * 1000:.2f} ms, max "
            f"{np.max(solve_times) * 1000:.2f} ms "
            f"(budget {T * 1000:.0f} ms)"
        )

    print(
        f"  final pelvis h   : "
        f"{d.qpos[2]:.4f} m  "
        f"(started {PELVIS_H})"
    )

    print(
        f"  travelled        : "
        f"{com[0] - com0[0]:+.4f} m   "
        f"(v_cmd x T_walk = "
        f"{v_cmd[0] * T_walk:.3f})"
    )

    print(
        f"  mean forward vel : "
        f"{(com[0] - com0[0]) / T_walk:+.4f} m/s  "
        f"(commanded {v_cmd[0]:+.3f})"
    )

    print(
        f"  max DCM error    : "
        f"{max_dcm_err * 1000:.1f} mm"
    )

    print(
        f"  QP failures      : "
        f"{fails} / {n_done}"
    )

    print()

    upright = d.qpos[2] > 0.70
    finished = n_done == N_sim

    print(
        f"  upright          : "
        f"{'YES' if upright else 'NO'}"
    )

    print(
        f"  ran to the end   : "
        f"{'YES' if finished else 'NO'}"
    )

    if upright and finished and fails == 0:
        print("\n  M8 PASSED -- online MPC replanning, fixed footsteps.")
    else:
        print(
            "\n  M8 FAILED."
        )


if __name__ == "__main__":
    main()