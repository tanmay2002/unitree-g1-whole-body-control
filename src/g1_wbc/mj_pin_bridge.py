#!/usr/bin/env python3


import numpy as np
import mujoco
import pinocchio as pin


def build_joint_map(mj_model, pin_model):
    """Name-based joint mapping between the two models.

    Returns a list of tuples:
        (name, mj_qposadr, mj_dofadr, pin_idx_q, pin_idx_v)

    We map by NAME rather than assuming index alignment. For
    g1_23dof_rev_1_0 the indices happen to coincide exactly, but that
    is a property of this file pair, not a guarantee.
    """
    mj_joints = {}
    for jid in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        mj_joints[name] = (
            int(mj_model.jnt_qposadr[jid]),
            int(mj_model.jnt_dofadr[jid]),
        )

    mapping = []
    for jid in range(1, pin_model.njoints):
        name = pin_model.names[jid]
        jmodel = pin_model.joints[jid]
        if jmodel.nq == 7:          # the free-flyer, handled separately
            continue
        if name not in mj_joints:
            raise ValueError(f"Pinocchio joint '{name}' missing from MuJoCo model")
        qadr, dadr = mj_joints[name]
        mapping.append((name, qadr, dadr, jmodel.idx_q, jmodel.idx_v))

    return mapping


def mj_to_pin_q(mj_data, mapping, pin_model):
    """MuJoCo qpos -> Pinocchio q."""
    q = np.zeros(pin_model.nq)
    q[0:3] = mj_data.qpos[0:3]
    w, x, y, z = mj_data.qpos[3:7]      # MuJoCo: w first
    q[3:7] = np.array([x, y, z, w])     # Pinocchio: w last
    for _, qadr, _, idx_q, _ in mapping:
        q[idx_q] = mj_data.qpos[qadr]
    return q


def mj_to_pin_v(mj_data, mapping, pin_model):
    """MuJoCo qvel -> Pinocchio v.

    The linear part is rotated world -> local. The angular part is
    already local in both. Verified in M0 Test D.
    """
    v = np.zeros(pin_model.nv)
    w, x, y, z = mj_data.qpos[3:7]
    R = pin.Quaternion(w, x, y, z).matrix()

    v[0:3] = R.T @ mj_data.qvel[0:3]    # world -> local
    v[3:6] = mj_data.qvel[3:6]          # already local
    for _, _, dadr, _, idx_v in mapping:
        v[idx_v] = mj_data.qvel[dadr]
    return v


def mj_to_pin_state(mj_data, mapping, pin_model):
    """Convenience: read the full MuJoCo state as a Pinocchio (q, v) pair."""
    return (mj_to_pin_q(mj_data, mapping, pin_model),
            mj_to_pin_v(mj_data, mapping, pin_model))


def build_actuator_map(mj_model, mapping):
    """Map each joint to its MuJoCo actuator id.

    Returns list of (pin_idx_v, actuator_id) for joints that have an
    actuator. Joints without one are silently skipped -- they are
    unactuated and TSID must not try to command them.
    """
    act_map = []
    for name, _, _, _, idx_v in mapping:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            act_map.append((idx_v, aid))
    return act_map


def pin_tau_to_mj_ctrl(tau, act_map, mj_model):
    """Pinocchio tau (nv,) -> MuJoCo ctrl (nu,).

    Drops the first 6 entries: the floating base is unactuated, so
    tau[0:6] must be zero for any physically realisable controller.
    """
    ctrl = np.zeros(mj_model.nu)
    for idx_v, aid in act_map:
        ctrl[aid] = tau[idx_v]
    return ctrl