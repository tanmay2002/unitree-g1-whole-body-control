#!/usr/bin/env python3
"""
M1passive drop test.
"""

import argparse
import numpy as np
import mujoco

XML_DEFAULT = "src/g1_wbc/models/g1_23dof_rev_1_0.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML_DEFAULT)
    ap.add_argument("--height", type=float, default=0.90,
                    help="initial pelvis height in metres")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="seconds of simulated time")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    total_mass = m.body_mass.sum()
    print(f"total robot mass : {total_mass:.3f} kg")
    print(f"timestep: {m.opt.timestep} s")
    print(f"initial height: {args.height} m\n")

    # start upright at the requested height, everything else zero
    mujoco.mj_resetData(m, d)
    d.qpos[2] = args.height
    d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])   # identity quaternion (w,x,y,z)
    mujoco.mj_forward(m, d)

    n_steps = int(args.duration / m.opt.timestep)
    report_every = n_steps // 10

    print(f"  {'t [s]':>7}  {'height':>8}  {'vz':>8}  {'ncon':>5}  {'|qvel|':>9}")
    print("  " + "-" * 46)

    for i in range(n_steps):
        d.ctrl[:] = 0.0            # zero torque
        mujoco.mj_step(m, d)

        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            print(f"\n simulation blew up at step {i} (t={d.time:.4f})")
            return

        if i % report_every == 0:
            print(f"  {d.time:7.3f}  {d.qpos[2]:8.4f}  {d.qvel[2]:8.4f}  "
                  f"{d.ncon:5d}  {np.linalg.norm(d.qvel):9.4f}")

    print()
    print(f"final height: {d.qpos[2]:.4f} m")
    print(f"final |qvel|: {np.linalg.norm(d.qvel):.4f}")
    print(f"contacts: {d.ncon}")
    print()

    settled = np.linalg.norm(d.qvel) < 0.5
    on_ground = d.ncon > 0
    finite = np.all(np.isfinite(d.qpos))

    print(f"finite state: {'YES' if finite else 'NO'}")
    print(f"made contact: {'YES' if on_ground else 'NO'}")
    print(f"came to rest: {'YES' if settled else 'NO (still moving)'}")

    if finite and on_ground:
        print("\n  M1 PASSED physics works, robot falls and lands.")
    else:
        print("\n  M1 FAILED.")


if __name__ == "__main__":
    main()