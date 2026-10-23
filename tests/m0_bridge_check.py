#!/usr/bin/env python3
"""
M0 Bridge Check
"""

import argparse
import numpy as np
import mujoco
import pinocchio as pin
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, "src")
np.set_printoptions(precision=6, suppress=True, linewidth=140)

def load_models(xml_path, urdf_path):
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    pin_data = pin_model.createData()

    print("=" * 78)
    print("Model Dimensions")
    print("=" * 78)
    print(f" Mujoco : {mj_model.nq:3d} nv={mj_model.nv:3d} nu={mj_model.nu:3d}")
    print(f" Pinocchio : nq={pin_model.nq:3d} nv={pin_model.nv:3d}")
    if mj_model.nq != pin_model.nq or mj_model.nv != pin_model.nv:
        print("Mismatch")
    else:
        print("Match")
    print()
    return mj_model, mj_data, pin_model, pin_data

def build_joint_map(mj_model, pin_model):
    """Return each joint's name and its MuJoCo/Pinocchio indices."""
    print("=" * 78)
    print("TEST A JOINT ORDER")
    print("=" * 78)

    mj_joints = {}
    for jid in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        mj_joints[name] = (
            int(mj_model.jnt_qposadr[jid]),
            int(mj_model.jnt_dofadr[jid]),
            int(mj_model.jnt_type[jid]),
        )

    mapping = []
    for jid in range(1, pin_model.njoints):
        name = pin_model.names[jid]
        jmodel = pin_model.joints[jid]
        if jmodel.nq == 7:      # free flyer 
            continue
        if name not in mj_joints:
            print(f"Pinocchio joint '{name}' not found in MuJoCo model")
            continue
        qadr, dadr, _ = mj_joints[name]
        mapping.append((name, qadr, dadr, jmodel.idx_q, jmodel.idx_v))
        print(f"{name:<32} mj_qposadr={qadr:>3}  pin_idx_q={jmodel.idx_q:>3}")

    print(f"\n  mapped {len(mapping)} joints\n")

    return mapping

def mj_to_pin_q(mj_data, mapping, pin_model):
    """Convert MuJoCo qpos to Pinocchio q"""
    q = np.zeros(pin_model.nq)
    q[0:3] = mj_data.qpos[0:3]                 # base position
    w, x, y, z = mj_data.qpos[3:7]             # MuJoCo: quaternion is (w,x,y,z)
    q[3:7] = np.array([x, y, z, w])            # Pinocchio: quaternion is (x,y,z,w)
    for _, qadr, _, idx_q, _ in mapping:
        q[idx_q] = mj_data.qpos[qadr]
    return q

def check_kinematics(mj_model, mj_data, pin_model, pin_data, mapping, n_trials=5):
    print("=" * 78)
    print("TEST C CONFIGURATION ROUND TRIP")
    print("=" * 78)

    # pick a few bodies that exist in both models, to use as probes
    candidates = []
    for bid in range(1, mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and pin_model.existBodyName(name):
            candidates.append(name)
    probes = candidates[:: max(1, len(candidates) // 6)][:6]
    print(f"probing bodies: {probes}\n")

    rng = np.random.default_rng(0)
    worst = 0.0

    for trial in range(n_trials):
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[0:3] = rng.uniform(-0.3, 0.3, 3) + np.array([0, 0, 0.8])
        quat = rng.normal(size=4)
        mj_data.qpos[3:7] = quat / np.linalg.norm(quat)
        for _, qadr, _, _, _ in mapping:
            mj_data.qpos[qadr] = rng.uniform(-0.4, 0.4)
        mujoco.mj_forward(mj_model, mj_data)

        q = mj_to_pin_q(mj_data, mapping, pin_model)
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        trial_worst = 0.0
        for name in probes:
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            p_mj = mj_data.xpos[bid]
            fid = pin_model.getFrameId(name)
            p_pin = pin_data.oMf[fid].translation
            err = np.linalg.norm(p_mj - p_pin)
            trial_worst = max(trial_worst, err)

        worst = max(worst, trial_worst)
        print(f"trial {trial}: max body position error = {trial_worst:.3e} m")

    print()
    if worst < 1e-5:
        print(f" PASS worst error {worst:.3e} m. The bridge conversion is correct.")
    else:
        print(f" FAIL worst error {worst:.3e} m. Something in the conversion is wrong.")
    print()

def mj_to_pin_v(mj_data, mapping, pin_model, rotate_linear):
    """Convert MuJoCo qvel to Pinocchio v"""
    v = np.zeros(pin_model.nv)
    w, x, y, z = mj_data.qpos[3:7]
    R = pin.Quaternion(w, x, y, z).matrix()

    v_lin = mj_data.qvel[0:3]
    v[0:3] = R.T @ v_lin if rotate_linear else v_lin
    v[3:6] = mj_data.qvel[3:6]
    for _, _, dadr, _, idx_v in mapping:
        v[idx_v] = mj_data.qvel[dadr]
    return v


def check_velocity(mj_model, mj_data, pin_model, pin_data, mapping):
    print("=" * 78)
    print("TEST D VELOCITY CONVENTION")
    print("=" * 78)

    rng = np.random.default_rng(1)
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[2] = 0.8
    quat = rng.normal(size=4)
    mj_data.qpos[3:7] = quat / np.linalg.norm(quat)
    for _, qadr, _, _, _ in mapping:
        mj_data.qpos[qadr] = rng.uniform(-0.3, 0.3)
    mj_data.qvel[:] = rng.uniform(-0.5, 0.5, mj_model.nv)
    mujoco.mj_forward(mj_model, mj_data)

    # probe close to the base first
    probe = "pelvis"
    probe_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, probe)
    print(f"  probe body: {probe}\n")

    res = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, probe_bid, res, 0)
    v_mj_lin_world = res[3:6]

    print(f"full res (6,): {res}")
    print(f"qvel[0:3] (lin, world): {mj_data.qvel[0:3]}")
    print(f"qvel[3:6] (ang, body): {mj_data.qvel[3:6]}")
    print(f"res[0:3] vs qvel[0:3] diff: {np.linalg.norm(res[0:3] - mj_data.qvel[0:3]):.3e}")
    print(f"res[3:6] vs qvel[0:3] diff: {np.linalg.norm(res[3:6] - mj_data.qvel[0:3]):.3e}\n")
    q = mj_to_pin_q(mj_data, mapping, pin_model)
    fid = pin_model.getFrameId(probe)
    fid_check = pin_model.getFrameId(probe)
    placement = pin_model.frames[fid_check].placement
    print(f"  pelvis frame placement translation: {placement.translation}")
    print(f"  pelvis frame placement rotation:\n{placement.rotation}")

 # Probe the pelvis (base) and distal bodies
    probe_names = ["pelvis", "left_knee_link", "right_ankle_roll_link",
                   "left_elbow_link", "right_wrist_roll_link"]
    probe_names = [n for n in probe_names
                   if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
                   and pin_model.existBodyName(n)]

    for lin_label, rotate_lin in [("lin: world (no rotate)", False),
                                  ("lin: rotated R^T", True)]:
        # build candidate v
        v = np.zeros(pin_model.nv)
        w, x, y, z = mj_data.qpos[3:7]
        R = pin.Quaternion(w, x, y, z).matrix()
        v[0:3] = R.T @ mj_data.qvel[0:3] if rotate_lin else mj_data.qvel[0:3]
        v[3:6] = mj_data.qvel[3:6]
        for _, _, dadr, _, idx_v in mapping:
            v[idx_v] = mj_data.qvel[dadr]

        pin.forwardKinematics(pin_model, pin_data, q, v)
        pin.updateFramePlacements(pin_model, pin_data)

        worst = 0.0
        details = []
        for name in probe_names:
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

            # MuJoCo gives body velocity at the inertial frame (CoM) in world coordinates.
            res = np.zeros(6)
            mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY,
                                     bid, res, 0)
            v_mj_com = res[3:6]

            # Pinocchio gives body velocity at the body-frame origin
            fid = pin_model.getFrameId(name)
            vf = pin.getFrameVelocity(pin_model, pin_data, fid,
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            # Shift Pinocchio velocity from the frame origin to the CoM.
            #   v_com = v_frame + omega x r
            R_body = pin_data.oMf[fid].rotation
            r_local = mj_model.body_ipos[bid]          # CoM offset in body coords
            r_world = R_body @ r_local
            v_pin_com = vf.linear + np.cross(vf.angular, r_world)

            err = np.linalg.norm(v_pin_com - v_mj_com)
            worst = max(worst, err)
            details.append(f"      {name:<26} {err:.3e}")

        verdict = "  <-- USE THIS" if worst < 1e-6 else ""
        print(f"  {lin_label:<26} worst error = {worst:.3e} m/s{verdict}")
        for d in details:
            print(d)
        print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="src/g1_wbc/models/g1_23dof_rev_1_0.xml")
    ap.add_argument("--urdf", default="src/g1_wbc/models/g1_23dof_rev_1_0.urdf")
    args = ap.parse_args()

    mj_model, mj_data, pin_model, pin_data = load_models(args.xml, args.urdf)
    mapping = build_joint_map(mj_model, pin_model)
    check_kinematics(mj_model, mj_data, pin_model, pin_data, mapping)
    check_velocity(mj_model, mj_data, pin_model, pin_data, mapping)

if __name__ == "__main__":
    main()