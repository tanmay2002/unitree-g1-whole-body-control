#!/usr/bin/env python3
"""
M2 -- gravity compensation.
torque crosses the bridge into MuJoCo.
"""

import argparse
import sys
import numpy as np
import mujoco
import pinocchio as pin

sys.path.insert(0, "src")
from g1_wbc.mj_pin_bridge import (build_joint_map, build_actuator_map,
                                  mj_to_pin_q, pin_tau_to_mj_ctrl)

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"


def run(sign, freeze_base, duration=2.0):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pd = pm.createData()

    mapping = build_joint_map(m, pm)
    act_map = build_actuator_map(m, mapping)

    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.2 if freeze_base else 0.80
    d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    # slight knee bend so gravity actually loads the joints
    for name, qadr, _, _, _ in mapping:
        if "knee" in name:
            d.qpos[qadr] = 0.3
        if "hip_pitch" in name:
            d.qpos[qadr] = -0.15
        if "ankle_pitch" in name:
            d.qpos[qadr] = -0.15
    mujoco.mj_forward(m, d)

    q0_joints = np.array([d.qpos[qadr] for _, qadr, _, _, _ in mapping])

    for _ in range(int(duration / m.opt.timestep)):
        q = mj_to_pin_q(d, mapping, pm)
        g = pin.computeGeneralizedGravity(pm, pd, q)
        d.ctrl[:] = pin_tau_to_mj_ctrl(sign * g, act_map, m)

        if freeze_base:
            d.qpos[0:3] = [0, 0, 1.2]
            d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            d.qvel[0:6] = 0.0

        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return None

    q1_joints = np.array([d.qpos[qadr] for _, qadr, _, _, _ in mapping])
    return np.abs(q1_joints - q0_joints).max()

def diagnose():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pd = pm.createData()
    mapping = build_joint_map(m, pm)
    act_map = build_actuator_map(m, mapping)
    print(f"  act_map entries  : {len(act_map)}")
    print(f"  act_map[:3]      : {act_map[:3]}")

    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.2
    d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    for name, qadr, _, _, _ in mapping:
        if "knee" in name:        d.qpos[qadr] = 0.3
        if "hip_pitch" in name:   d.qpos[qadr] = -0.15
        if "ankle_pitch" in name: d.qpos[qadr] = -0.15
    mujoco.mj_forward(m, d)

    q = mj_to_pin_q(d, mapping, pm)
    g = pin.computeGeneralizedGravity(pm, pd, q)

    print(f"|g| overall: {np.linalg.norm(g):.3f}")
    print(f"g[0:6]  (base): {g[0:6]}")
    print(f"|g[6:]| (joints): {np.linalg.norm(g[6:]):.3f}")
    print(f"max |g[6:]|: {np.abs(g[6:]).max():.3f} Nm")
    print(f"ctrlrange[0]: {m.actuator_ctrlrange[0]}")
    print(f"forcerange[0]: {m.actuator_forcerange[0]}\n")
    print(f"ctrllimited: {m.actuator_ctrllimited[:3]}")
    print(f"forcelimited: {m.actuator_forcelimited[:3]}")
    print(f"actuator_trnid[0]: {m.actuator_trnid[0]}")

    for sign, label in [(0.0, "tau=0"), (+1.0, "tau=+g"), (-1.0, "tau=-g")]:
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 1.2
        d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        for name, qadr, _, _, _ in mapping:
            if "knee" in name:        d.qpos[qadr] = 0.3
            if "hip_pitch" in name:   d.qpos[qadr] = -0.15
            if "ankle_pitch" in name: d.qpos[qadr] = -0.15
        d.qvel[:] = 0.0
        d.ctrl[:] = pin_tau_to_mj_ctrl(sign * g, act_map, m)
        print(f"  {label:<8} ctrl[:4] = {d.ctrl[:4]}  |ctrl| = {np.linalg.norm(d.ctrl):.4f}")
        mujoco.mj_step(m, d)
        print(f"  {label:<8} max |qacc[6:]| = {np.abs(d.qacc[6:]).max():9.4f} rad/s^2"
              f"max |qvel[6:]| = {np.abs(d.qvel[6:]).max():9.4f}")
def sign_check():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    mapping = build_joint_map(m, pm)
    act_map = build_actuator_map(m, mapping)

    target = "left_knee_joint"
    idx_v = [iv for n, _, _, _, iv in mapping if n == target][0]
    qadr  = [qa for n, qa, _, _, _ in mapping if n == target][0]

    for sign in (+1.0, -1.0):
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 1.2
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        tau = np.zeros(pm.nv)
        tau[idx_v] = sign * 5.0          # 5 Nm on one joint only
        d.ctrl[:] = pin_tau_to_mj_ctrl(tau, act_map, m)
        mujoco.mj_step(m, d)
        print(f"tau={sign*5.0:+.1f} Nm on {target}: "
              f"qvel = {d.qvel[idx_v]:+.6f} rad/s")

def standing_gravity_comp(duration=1.0):
    print("  Gravity compensation, feet on ground\n")
    m = mujoco.MjModel.from_xml_path(XML)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pd = pm.createData()
    mapping = build_joint_map(m, pm)
    act_map = build_actuator_map(m, mapping)

    for use_g in (False, True):
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.78
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for name, qadr, _, _, _ in mapping:
            if "knee" in name:        d.qpos[qadr] = 0.30
            if "hip_pitch" in name:   d.qpos[qadr] = -0.15
            if "ankle_pitch" in name: d.qpos[qadr] = -0.15
        mujoco.mj_forward(m, d)
        q0 = np.array([d.qpos[qa] for _, qa, _, _, _ in mapping])

        for _ in range(int(duration / m.opt.timestep)):
            if use_g:
                q = mj_to_pin_q(d, mapping, pm)
                g = pin.computeGeneralizedGravity(pm, pd, q)
                d.ctrl[:] = pin_tau_to_mj_ctrl(g, act_map, m)
            else:
                d.ctrl[:] = 0.0
            mujoco.mj_step(m, d)

        q1 = np.array([d.qpos[qa] for _, qa, _, _, _ in mapping])
        label = "with g(q)" if use_g else "zero torque"
        print(f"{label:<14} joint drift = {np.abs(q1-q0).max():.4f} rad   "
              f"height {d.qpos[2]:.3f} m   ncon {d.ncon}")


def verify_dynamics(n_trials=5):
    print("Dynamics agreement: Pinocchio vs MuJoCo\n")
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    pd = pm.createData()
    mapping = build_joint_map(m, pm)

    rng = np.random.default_rng(7)
    worst_g, worst_M = 0.0, 0.0

    for t in range(n_trials):
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = rng.uniform(-0.2, 0.2, 3) + np.array([0, 0, 0.9])
        quat = rng.normal(size=4)
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        for _, qadr, _, _, _ in mapping:
            d.qpos[qadr] = rng.uniform(-0.4, 0.4)
        d.qvel[:] = 0.0                     # kills the Coriolis term
        mujoco.mj_forward(m, d)

        q = mj_to_pin_q(d, mapping, pm)

        # gravity
        g_pin = pin.computeGeneralizedGravity(pm, pd, q)
        g_err = max(abs(g_pin[idx_v] - d.qfrc_bias[dadr])
                    for _, _, dadr, _, idx_v in mapping)

        # mass matrix
        M_pin = pin.crba(pm, pd, q)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M_mj)
        M_err = 0.0
        for _, _, dadr_i, _, iv_i in mapping:
            for _, _, dadr_j, _, iv_j in mapping:
                M_err = max(M_err, abs(M_pin[iv_i, iv_j] - M_mj[dadr_i, dadr_j]))

        worst_g = max(worst_g, g_err)
        worst_M = max(worst_M, M_err)
        print(f"trial {t}:  max |g| err = {g_err:.3e} Nm     "
              f"max |M| err = {M_err:.3e} kg m^2")

    print()
    print(f"worst gravity error: {worst_g:.3e} Nm")
    print(f"worst mass matrix error: {worst_M:.3e} kg m^2")
    ok = worst_g < 1e-4 and worst_M < 1e-5
    print(f"\n {'PASS the two dynamics models agree.' if ok else 'FAIL models disagree.'}")

def main():
    sign_check()
    standing_gravity_comp()
    verify_dynamics()
    # print("  Scenario A base frozen, testing the torque path only\n")
    # for sign, label in [(+1.0, "tau = +g(q)"), (-1.0, "tau = -g(q)"), (0.0, "tau = 0 (control)")]:
    #     drift = run(sign, freeze_base=True)
    #     if drift is None:
    #         print(f"{label:<20} DIVERGED")
    #     else:
    #         verdict = "  <-- correct sign" if drift < 0.05 else ""
    #         print(f"{label:<20} max joint drift = {drift:.4f} rad{verdict}")

    # print("\n  Scenario B base free, feet on ground\n")
    # for sign, label in [(+1.0, "tau = +g(q)"), (0.0, "tau = 0")]:
    #     drift = run(sign, freeze_base=False)
    #     if drift is None:
    #         print(f"{label:<20} DIVERGED")
    #     else:
    #         print(f"{label:<20} max joint drift = {drift:.4f} rad")
    diagnose()

if __name__ == "__main__":
    main()