"""
Converts raw LIPM/MPC output (g1_walking_traj_lipm.npz) into a TSID-ready
trajectory (g1_walking_traj_tsid.npz): resampled CoM trajectory, swing-foot
Cartesian trajectories, and a per-timestep contact-phase schedule.

NOTE: written directly against the TsidBiped interface (set_com_ref,
set_LF_3d_ref/RF_3d_ref, add/remove_contact_LF/RF) since the original Romeo
ex_4_LIPM_to_TSID.py was not available to adapt from. Review the footstep/
swing-assignment logic below carefully before trusting it.
"""
import numpy as np
import sys
sys.path.insert(0, "src")
from g1_wbc import lipm_conf as conf

# ---- load raw LIPM output ----
lipm = np.load(conf.DATA_FILE_LIPM)
com_state_x = lipm["com_state_x"]   # (N+1, 2) -> [pos, vel]
com_state_y = lipm["com_state_y"]   # (N+1, 2)
foot_steps = lipm["foot_steps"]     # (nb_steps, 2) -> [x, y] per phase

nb_steps = conf.nb_steps
T_step = conf.T_step
dt_mpc = conf.dt_mpc
nb_dt_per_step = round(T_step / dt_mpc)
N_mpc = nb_steps * nb_dt_per_step
assert com_state_x.shape[0] == N_mpc + 1

dt = conf.dt              # TSID timestep
T_pre = conf.T_pre
T_post = conf.T_post
h = conf.h                # constant CoM height
step_height = conf.step_height
FOOT_Z = 0.035

# ---- time grids ----
t_mpc = np.arange(N_mpc + 1) * dt_mpc
t_walk_end = t_mpc[-1]
n_pre = round(T_pre / dt)
n_walk = round(t_walk_end / dt)
n_post = round(T_post / dt)
N_total = n_pre + n_walk + n_post

# ---- resample CoM x,y (linear interp) onto TSID dt, hold pre/post ----
t_walk_fine = np.arange(n_walk) * dt
com_x_walk = np.interp(t_walk_fine, t_mpc, com_state_x[:, 0])
com_vx_walk = np.interp(t_walk_fine, t_mpc, com_state_x[:, 1])
com_y_walk = np.interp(t_walk_fine, t_mpc, com_state_y[:, 0])
com_vy_walk = np.interp(t_walk_fine, t_mpc, com_state_y[:, 1])

com_traj = np.zeros((N_total, 3))
com_vel_traj = np.zeros((N_total, 3))
com_traj[:n_pre, 0] = com_x_walk[0]
# pre-phase lateral weight shift: robot starts standing with CoM centered
# (y=0), but the LIPM plan starts with CoM already over the right foot
# (y = com_y_walk[0] = -0.096). Smoothly shift during the second half of
# T_pre so the plan's initial condition is actually met when walking starts.
n_shift = n_pre // 2
s_pre = np.linspace(0, 1, n_shift)
s_pre = 3 * s_pre**2 - 2 * s_pre**3  # smoothstep
com_traj[:n_pre - n_shift, 1] = 0.0
com_traj[n_pre - n_shift:n_pre, 1] = com_y_walk[0] * s_pre
com_traj[n_pre:n_pre + n_walk, 0] = com_x_walk
com_traj[n_pre:n_pre + n_walk, 1] = com_y_walk
com_traj[n_pre + n_walk:, 0] = com_x_walk[-1]
com_traj[n_pre + n_walk:, 1] = com_y_walk[-1]
com_traj[:, 2] = h
com_vel_traj[n_pre:n_pre + n_walk, 0] = com_vx_walk
com_vel_traj[n_pre:n_pre + n_walk, 1] = com_vy_walk

# ---- footstep / swing-foot assignment ----
# foot_steps[0] = initial stance (right foot, y<0, per foot_step_0).
# Convention: RIGHT foot starts in stance (phase 0 swings LEFT foot to
# foot_steps[1]); phases then alternate. Contact phase codes:
#   0 = double support (pre/post standing)
#   1 = right foot stance, LEFT foot swinging
#   2 = left foot stance, RIGHT foot swinging
assert foot_steps.shape[0] == nb_steps

lf_pos = np.zeros((N_total, 3))
rf_pos = np.zeros((N_total, 3))
contact_phase = np.zeros(N_total, dtype=int)

# static foot y-offsets (feet don't move laterally except at footstep events)
right_start = foot_steps[0]
left_start = np.array([foot_steps[0][0], -foot_steps[0][1]])  # mirror y

cur_rf = np.array([right_start[0], right_start[1], FOOT_Z])
cur_lf = np.array([left_start[0], left_start[1], FOOT_Z])
lf_pos[:n_pre] = cur_lf
rf_pos[:n_pre] = cur_rf
contact_phase[:n_pre] = 0

nb_dt_per_step_tsid = round(T_step / dt)  # step duration in TSID timesteps, not MPC steps
ds_frac = 0.30  # fraction of each step held in double support at the end, to let tracking error settle before the next swing
n_ds = round(nb_dt_per_step_tsid * ds_frac)
swing_is_left = True  # phase 0 swings left foot
for step_i in range(nb_steps):
    i0 = n_pre + step_i * nb_dt_per_step_tsid
    i1_full = n_pre + (step_i + 1) * nb_dt_per_step_tsid
    i1 = i1_full - n_ds  # swing ends here, double-support window follows
    target = foot_steps[step_i + 1] if step_i + 1 < nb_steps else \
    np.array([foot_steps[step_i][0], foot_steps[0][1] * (-1 if swing_is_left else 1)])
    #target = foot_steps[step_i + 1] if step_i + 1 < nb_steps else \
        #np.array([foot_steps[step_i][0], -foot_steps[step_i][1]])  # bring feet together on last phase

    if swing_is_left:
        start_pos = cur_lf.copy()
        end_pos = np.array([target[0], target[1], FOOT_Z])
        contact_phase[i0:i1] = 1  # right stance
    else:
        start_pos = cur_rf.copy()
        end_pos = np.array([target[0], target[1], FOOT_Z])
        contact_phase[i0:i1] = 2  # left stance

    n_phase = i1 - i0
    s = np.linspace(0, 1, n_phase, endpoint=False)
    # smoothstep easing (zero velocity AND well-behaved at both endpoints) instead
    # of raw linear interpolation, which had a velocity discontinuity at liftoff/
    # touchdown -- harmless for swing 1 (zero horizontal displacement) but a real
    # problem for swing 2 onward, which have actual forward displacement
    s_smooth = 3 * s**2 - 2 * s**3
    swing_xy = start_pos[:2, None].T * (1 - s_smooth[:, None]) + end_pos[:2, None].T * s_smooth[:, None]
    swing_z = FOOT_Z + step_height * np.sin(np.pi * s)**2  # zero value AND zero derivative at s=0,1

    if swing_is_left:
        lf_pos[i0:i1, :2] = swing_xy
        lf_pos[i0:i1, 2] = swing_z
        rf_pos[i0:i1] = cur_rf
        cur_lf = end_pos
        # double-support tail: both feet held at their now-settled positions
        lf_pos[i1:i1_full] = cur_lf
        rf_pos[i1:i1_full] = cur_rf
        contact_phase[i1:i1_full] = 0
        upcoming_stance_y = cur_lf[1]  # left just landed, becomes stance next
    else:
        rf_pos[i0:i1, :2] = swing_xy
        rf_pos[i0:i1, 2] = swing_z
        lf_pos[i0:i1] = cur_lf
        cur_rf = end_pos
        lf_pos[i1:i1_full] = cur_lf
        rf_pos[i1:i1_full] = cur_rf
        contact_phase[i1:i1_full] = 0
        upcoming_stance_y = cur_rf[1]  # right just landed, becomes stance next

    swing_is_left = not swing_is_left

lf_pos[n_pre + n_walk:] = cur_lf
rf_pos[n_pre + n_walk:] = cur_rf
contact_phase[n_pre + n_walk:] = 0

# recompute CoM velocity since com_traj[:,1] was reshaped above (post double-support blending)
com_vel_traj[:, 1] = np.gradient(com_traj[:, 1], dt)

# finite-difference velocity/acceleration for swing feet (zero feedforward
# was fighting the controller's own damping term during swing)
lf_vel = np.gradient(lf_pos, dt, axis=0)
rf_vel = np.gradient(rf_pos, dt, axis=0)
lf_acc = np.gradient(lf_vel, dt, axis=0)
rf_acc = np.gradient(rf_vel, dt, axis=0)
# zero out feedforward during stance (only meaningful during that foot's swing)
lf_vel[contact_phase != 1] = 0.0
lf_acc[contact_phase != 1] = 0.0
rf_vel[contact_phase != 2] = 0.0
rf_acc[contact_phase != 2] = 0.0

np.savez(
    conf.DATA_FILE_TSID,
    com_traj=com_traj,
    com_vel_traj=com_vel_traj,
    lf_traj=lf_pos,
    rf_traj=rf_pos,
    lf_vel_traj=lf_vel,
    rf_vel_traj=rf_vel,
    lf_acc_traj=lf_acc,
    rf_acc_traj=rf_acc,
    contact_phase=contact_phase,
    dt=dt,
)
print(f"Saved {conf.DATA_FILE_TSID}: N_total={N_total} "
      f"(pre={n_pre}, walk={n_walk}, post={n_post})")
