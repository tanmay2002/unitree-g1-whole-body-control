#!/usr/bin/env python3
"""
LIPM walking pattern generation via linear MPC (LMPC_walking), with a
double-support phase in the CoP reference.

Output: results/g1_walking_traj_lipm.npz, consumed by
        tests/ex_4_LIPM_to_TSID_g1.py.

WHY THE CoP REFERENCE HAS A DOUBLE-SUPPORT RAMP
-----------------------------------------------
LMPC_walking's create_CoP_trajectory holds the CoP at each foot centre
for a whole step and jumps instantaneously to the next foot. That is
the original preview-control formulation, which omits the double
support phase. It is a known deficiency: ignoring the ZMP during double
support makes the CoM acceleration discontinuous (Biomimetics 2025,
doi 10.3390/biomimetics10100654). A real CoP cannot jump 0.24 m across
the body -- it moves only as load transfers between the feet.

Measured consequence on the G1: the CoM tracked the plan through single
support with ~1 mm of DCM error and then lost 40 mm at every step
boundary, where the plan demanded a CoP the freshly planted foot could
not yet carry (it held 22 N of 315 N).

The standard reference shape, confirmed by two independent sources, is
CoP at the stance-foot centre during single support and linearly
interpolated between the feet during double support (Kajita-style
preview control; DCM-based NMPC, arXiv 2506.03856 eq. 4). That is what
create_cop_with_ds below implements.

THE TWO DS WINDOWS MUST AGREE
-----------------------------
ds_frac here (CoP reference) and ds_frac in ex_4_LIPM_to_TSID_g1.py
(contact schedule) must be the same value, both at the END of each
step. Otherwise TSID switches contacts at a different instant than the
plan assumed. 0.30 matches HRP-5P's 32% double support (arXiv
2207.12644).

HORIZON LIMIT
-------------
The LIPM preview matrices grow like sinh(omega t), so beyond roughly
omega * T_horizon = 18 the Hessian's condition number exceeds double
precision and quadprog fails with a misleading "not positive definite".
Keep nb_steps * T_step <= ~4.8 s. The script checks and refuses.
"""

import sys
sys.path.insert(0, "src")

import matplotlib
import numpy as np
from quadprog import solve_qp

try:
    import LMPC_walking.second_order.constraints as constraints
    import LMPC_walking.second_order.cost_function as cost_function
    import LMPC_walking.second_order.motion_model as motion_model
    import LMPC_walking.second_order.plot_utils as plot_utils
    import LMPC_walking.second_order.reference_trajectories as reference_trajectories
except ModuleNotFoundError as e:
    print("Please download LMPC_walking from "
          "https://github.com/machines-in-motion/lmpc_walking.")
    raise e

from g1_wbc import lipm_conf as conf
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = False

# must match ds_frac in ex_4_LIPM_to_TSID_g1.py
DS_FRAC = 0.30


def create_cop_with_ds(foot_steps, N, nb_dt_per_step, ds_frac):
    """CoP reference with a linear double-support ramp.

    Within each step of nb_dt_per_step samples:
        first (1 - ds_frac)   : hold at foot_steps[i]        (single support)
        last  ds_frac         : ramp to foot_steps[i+1]      (double support)
    The final step holds at its own foot for the whole step since there
    is no next foot to ramp toward.
    """
    n_ds = int(round(nb_dt_per_step * ds_frac))
    n_ss = nb_dt_per_step - n_ds
    Z = np.zeros((N, 2))
    nb = foot_steps.shape[0]
    for i in range(nb):
        i0 = i * nb_dt_per_step
        Z[i0:i0 + n_ss] = foot_steps[i]
        if i + 1 < nb:
            s = np.linspace(0.0, 1.0, n_ds + 1)[1:]            # (0, 1]
            Z[i0 + n_ss:i0 + nb_dt_per_step] = (
                foot_steps[i][None, :] * (1.0 - s[:, None])
                + foot_steps[i + 1][None, :] * s[:, None])
        else:
            Z[i0 + n_ss:i0 + nb_dt_per_step] = foot_steps[i]
    return Z, n_ss, n_ds


# ---------------------------------------------- inverted pendulum params
foot_length = conf.lxn + conf.lxp
foot_width = conf.lyn + conf.lyp
nb_dt_per_step = round(conf.T_step / conf.dt_mpc)
N = conf.nb_steps * nb_dt_per_step

omega = np.sqrt(conf.g / conf.h)
T_horizon = N * conf.dt_mpc

print("=" * 70)
print("LIPM PLAN")
print("=" * 70)
print(f"  h = {conf.h:.4f} m   omega = {omega:.4f} rad/s")
print(f"  T_step = {conf.T_step} s   dt_mpc = {conf.dt_mpc:.6f} s   "
      f"{nb_dt_per_step} samples/step")
print(f"  nb_steps = {conf.nb_steps}   N = {N}   "
      f"horizon = {T_horizon:.2f} s")
print(f"  omega * T_horizon = {omega * T_horizon:.2f}   (keep below ~18)")
print(f"  foot: length {foot_length:.3f} m, width {foot_width:.3f} m")
print(f"  step_length = {conf.step_length} m   "
      f"foot_step_0 = {conf.foot_step_0}")

# ---- CoM initial state ------------------------------------------------
x_0 = np.array([conf.foot_step_0[0], 0.0])
y_0 = np.array([conf.foot_step_0[1], 0.0])
step_width = 2 * np.absolute(y_0[0])

# ---- CoP reference: foot plan, then DS-interpolated ---------------------
foot_steps = reference_trajectories.manual_foot_placement(
    conf.foot_step_0, conf.step_length, conf.nb_steps
)
foot_steps[1:, 0] -= conf.step_length
cop_ref, n_ss, n_ds = create_cop_with_ds(foot_steps, N, nb_dt_per_step,
                                         DS_FRAC)
print(f"  double support : {n_ds} of {nb_dt_per_step} samples per step "
      f"({DS_FRAC:.0%}), {n_ds*conf.dt_mpc:.2f} s\n")

# ---- terminal constraints ---------------------------------------------
x_terminal = np.array([cop_ref[N - 1, 0], 0.0])
y_terminal = np.array([cop_ref[N - 1, 1], 0.0])
nb_terminal_constraints = 4
terminal_index = N - 1

# ---- preview system ----------------------------------------------------
[P_ps, P_vs, P_pu, P_vu] = motion_model.compute_recursive_matrices(
    conf.dt_mpc, conf.g, conf.h, N
)
[Q, p_k] = cost_function.compute_objective_terms(
    conf.alpha, conf.beta, conf.gamma, conf.T_step, nb_dt_per_step, N,
    conf.step_length, step_width, P_ps, P_pu, P_vs, P_vu, x_0, y_0, cop_ref,
)

# ---- conditioning check ------------------------------------------------
w_eig = np.linalg.eigvalsh(Q)
cond = abs(w_eig.max()) / max(abs(w_eig.min()), 1e-300)
print(f"  P_vu max entry      : {np.abs(P_vu).max():.3e}")
print(f"  Hessian eigenvalues : min {w_eig.min():.3e}   "
      f"max {w_eig.max():.3e}")
print(f"  condition number    : {cond:.3e}")

if w_eig.min() <= 0.0 or cond > 1e15:
    print()
    print("  !! Hessian unusable.")
    if w_eig.min() <= 0.0:
        print(f"     Min eigenvalue is {w_eig.min():.3e}, but analytically")
        print(f"     it must be at least alpha = {conf.alpha}. The small")
        print("     eigenvalues have been destroyed by round-off against")
        print(f"     a max of {w_eig.max():.3e}.")
    print()
    print(f"     Horizon: nb_steps * T_step = {conf.nb_steps} * "
          f"{conf.T_step} = {T_horizon:.2f} s, omega*T = {omega*T_horizon:.1f}")
    print("     Keep omega*T_horizon at or below ~18.")
    max_steps = int(18.0 / (omega * conf.T_step))
    max_tstep = 18.0 / (omega * conf.nb_steps)
    print(f"       nb_steps = {max_steps} at T_step = {conf.T_step}, or")
    print(f"       T_step   = {max_tstep:.2f} at nb_steps = {conf.nb_steps}")
    sys.exit(1)

print("  conditioning OK, solving\n")

# ---- constraints -------------------------------------------------------
# add_ZMP_constraints builds a box of +-foot dims around cop_ref at each
# sample. During the DS ramp that box slides between the feet, which is
# CONSERVATIVE (the true admissible region is the hull of both feet) --
# safe, and it keeps the CoP moving with the ramp rather than jumping.
[A_zmp, b_zmp] = constraints.add_ZMP_constraints(
    N, foot_length, foot_width, cop_ref, x_0, y_0
)
[A_terminal, b_terminal] = constraints.add_terminal_constraints(
    N, terminal_index, x_0, y_0, x_terminal, y_terminal,
    P_ps, P_vs, P_pu, P_vu
)
A = np.concatenate((A_terminal, A_zmp), axis=0)
b = np.concatenate((b_terminal, b_zmp), axis=0)

# ---- solve -------------------------------------------------------------
U = solve_qp(Q, -p_k, A.T, b, nb_terminal_constraints)[0]
cop_x = U[0:N]
cop_y = U[N:2 * N]

[com_state_x, com_state_y] = motion_model.compute_recursive_dynamics(
    P_ps, P_vs, P_pu, P_vu, N, x_0, y_0, U
)

# ---- report ------------------------------------------------------------
lo = cop_ref[:, 1] - conf.lyp
hi = cop_ref[:, 1] + conf.lyp
margin = min(np.min(cop_y - lo), np.min(hi - cop_y))
jumps = np.abs(np.diff(cop_y))

print(f"  solved. CoP y margin inside the box : {margin*1000:.1f} mm")
print(f"  largest CoP y step between samples  : {jumps.max()*1000:.1f} mm "
      f"(was ~237 mm with the jumping reference)")
if margin < 0:
    print("  !! CoP leaves the box -- plan is not executable")
elif margin < 0.005:
    print("  !! CoP margin under 5 mm -- at the physical limit")
print(f"  CoM y range: {com_state_y[:,0].min():+.4f} to "
      f"{com_state_y[:,0].max():+.4f}")
print(f"  CoM x range: {com_state_x[:,0].min():+.4f} to "
      f"{com_state_x[:,0].max():+.4f}")
print(f"  footsteps:\n{foot_steps}")

# ---- plots (optional) --------------------------------------------------
time = np.arange(N) * conf.dt_mpc
min_admissible_cop = cop_ref - np.tile([foot_length / 2, foot_width / 2], (N, 1))
max_admissible_cop = cop_ref + np.tile([foot_length / 2, foot_width / 2], (N, 1))
try:
    plot_utils.plot_x(time, N, min_admissible_cop, max_admissible_cop,
                      cop_x, com_state_x, cop_ref)
    plot_utils.plot_y(time, N, min_admissible_cop, max_admissible_cop,
                      cop_y, com_state_y, cop_ref)
except Exception as exc:
    print(f"  (plots skipped: {exc})")

# ---- save --------------------------------------------------------------
com_state_x = np.vstack((x_0, com_state_x))
com_state_y = np.vstack((y_0, com_state_y))

np.savez(
    conf.DATA_FILE_LIPM,
    com_state_x=com_state_x,
    com_state_y=com_state_y,
    cop_ref=cop_ref,
    cop_x=cop_x,
    cop_y=cop_y,
    foot_steps=foot_steps,
    ds_frac=DS_FRAC,
)
print(f"\n  saved {conf.DATA_FILE_LIPM}")

try:
    plt.show()
except Exception:
    pass