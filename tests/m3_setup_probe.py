#!/usr/bin/env python3
"""Generate the foot frames, sole geometry and standing pose."""
import sys
import numpy as np
import mujoco
import pinocchio as pin

sys.path.insert(0, "src")
from g1_wbc.mj_pin_bridge import build_joint_map, mj_to_pin_q

XML = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
pd = pm.createData()
mapping = build_joint_map(m, pm)

print("Foot geoms (collision shapes on the ankle roll links):\n")
for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, foot)
    print(f"{foot}  (body {bid})")
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == bid:
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
            print(f" geom {gid:3d} {str(gname):<28} type={m.geom_type[gid]} "
                  f"pos={m.geom_pos[gid]}  size={m.geom_size[gid]}")
    print()

# find a standing height by settling the robot with a reasonable pose
mujoco.mj_resetData(m, d)
d.qpos[2] = 0.80
d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
pose = {"hip_pitch": -0.20, "knee": 0.42, "ankle_pitch": -0.23}
for name, qadr, _, _, _ in mapping:
    for key, val in pose.items():
        if key in name:
            d.qpos[qadr] = val
mujoco.mj_forward(m, d)

q = mj_to_pin_q(d, mapping, pm)
pin.forwardKinematics(pm, pd, q)
pin.updateFramePlacements(pm, pd)
com = pin.centerOfMass(pm, pd, q)

print("Candidate standing pose (hip -0.20, knee 0.42, ankle -0.23):\n")
print(f"pelvis height: {d.qpos[2]:.4f} m")
print(f"CoM: {com}")
for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
    fid = pm.getFrameId(foot)
    print(f"{foot:<24} z = {pd.oMf[fid].translation[2]:+.4f} m")
print(f"contacts at this pose : {d.ncon}")

print("\n  Settling the pelvis so the soles touch:\n")
SOLE_OFFSET = 0.03      # sole plane below the ankle_roll frame origin

for name, qadr, _, _, _ in mapping:
    for key, val in pose.items():
        if key in name:
            d.qpos[qadr] = val

d.qpos[2] = 0.80
mujoco.mj_forward(m, d)
q = mj_to_pin_q(d, mapping, pm)
pin.forwardKinematics(pm, pd, q)
pin.updateFramePlacements(pm, pd)
foot_z = pd.oMf[pm.getFrameId("left_ankle_roll_link")].translation[2]
drop = foot_z - SOLE_OFFSET

d.qpos[2] = 0.80 - drop
mujoco.mj_forward(m, d)
q = mj_to_pin_q(d, mapping, pm)
pin.forwardKinematics(pm, pd, q)
pin.updateFramePlacements(pm, pd)
com = pin.centerOfMass(pm, pd, q)

print(f"corrected pelvis height: {d.qpos[2]:.4f} m")
print(f"CoM: {com}")
print(f"left sole z: "
      f"{pd.oMf[pm.getFrameId('left_ankle_roll_link')].translation[2] - SOLE_OFFSET:+.5f} m")
print(f"contacts: {d.ncon}")
print(f"\n CoM height for LIPM: {com[2]:.4f} m")