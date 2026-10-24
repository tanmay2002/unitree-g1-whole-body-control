## Pipeline Architecture

The pipeline has two layers:

*Planner (LIPM / Herdt MPC):* Treats the robot as a point mass and decides where the CoM should be and where the feet should go. Runs at 100ms.


*Controller (TSID):* Takes those references and solves a QP every 2ms over joint accelerations and contact forces to find the 23 joint torques that best achieve them.


Herdt MPC (100ms)
      |
      | CoM reference + footstep plan
      v
   TSID QP (2ms)
      |
      | 23 joint torques
      v
 MuJoCo contact physics (2ms)
      |
      | measured q, v
      v
   (feedback to both layers)


---

## Milestone Ladder

Each milestone gates the next. Each changes as close to one thing as possible.

| Milestone | Description | Owner | File |
|---|---|---|---|
| M0 | MuJoCo and Pinocchio bridge verification | Ashutosh | tests/m0_bridge_check.py |
| M1 | Zero torque drop test | Ashutosh | tests/m1_drop_test.py |
| M2 | Torque path and dynamics agreement | Ashutosh | tests/m2_gravity_comp.py |
| M3 | Closed loop standing balance with TSID | Ashutosh | tests/m3_standing_tsid.py |
| M4 | CoM tracking with feedforward comparison | Ashutosh | tests/m4_com_sway.py |
| M5 | Single step and contact switching | Ashutosh | tests/m5_single_step.py |
| M6a | Quasi static walking | Ashutosh | tests/m6a_walk_quasistatic.py |
| M6b | LIPM walking closed loop through MuJoCo | Ashutosh | tests/m6b_walk_lipm.py |
| M7 | Walking with simultaneous arm task | Ashutosh | tests/m7_walk_with_arm.py |
| M8 | Online Herdt MPC, no arm | Tanmay | tests/m8_online_mpc.py |
| M9 | Online Herdt MPC with trolley push arm pose | Tanmay | tests/m9_mpc_with_arm.py |

---

## Key Results

| Milestone | Key Result |
|---|---|
| M0 | Kinematics agree to 1.6e-7 m, velocity convention verified |
| M1 | Mass 32.107 kg, no NaNs, contacts engage correctly |
| M2 | +5 Nm gives +0.412 rad/s, dynamics agree to 3e-5 Nm |
| M3 | CoM error 2.89 mm, 0 QP failures over 2500 cycles |
| M4 | Feedforward: 0.43 mm RMS vs 22.06 mm without (51x improvement) |
| M5 | Placement error 0.9 mm, fMax bug found and fixed |
| M6a | 4 steps, 0.329 m travelled, 0 QP failures |
| M6b | DCM 32.7 mm (from 788 mm), 3 steps, 0 QP failures |
| M7 | DCM 29.6 mm with arm vs 32.1 mm without, hand RMS 9.5 mm |
| M8 | DCM 7.2 mm, 12 steps, 1.83 m at vx=0.2 m/s, 0.25 ms solve |
| M9 | DCM 9.1 mm, 12 steps, 1.85 m at vx=0.2 m/s, arm task active, 0 QP failures |

---

## Demo Commands

conda activate humanoid
cd ~/Humanoid_lab

# M8: online MPC walking, no arm
python tests/m8_online_mpc.py --n-steps 12 --vx 0.2 --t-post 0 --view

# M9: online MPC walking with trolley push arm pose
python tests/m9_mpc_with_arm.py --n-steps 12 --vx 0.2 --t-post 0 --view

# M9: 16 steps
python tests/m9_mpc_with_arm.py --n-steps 16 --vx 0.2 --t-post 0 --view

# M9: with sinusoidal moving arm
python tests/m9_mpc_with_arm.py --n-steps 12 --vx 0.2 --t-post 0 --no-trolley-pose --arm --arm-amp 0.1 --view


---

## Setup

# Create conda environment
conda create -n humanoid python=3.10
conda activate humanoid

# Install dependencies
pip install mujoco pinocchio tsid quadprog numpy matplotlib

# Clone and run
cd ~/Humanoid_lab
python tests/m9_mpc_with_arm.py --n-steps 12 --vx 0.2 --t-post 0 --view


---

## Repository Structure

src/

  g1_wbc/
  
    wbc_controller.py       # G1BalanceController: TSID formulation, tasks, contacts
    mj_pin_bridge.py        # MuJoCo to Pinocchio state conversion
    lipm_conf.py            # Robot and gait constants for LIPM planner
    models/                 # G1 URDF and MJCF files
    
  herdt/
  
    automatic_lmpc.py       # Herdt MPC reference class (adapted for G1)
    constants.py            # MPC parameters adapted for G1
    
tests/

  m0_bridge_check.py        # Bridge verification
  m1_drop_test.py           # Drop test
  m2_gravity_comp.py        # Torque path and dynamics
  m3_standing_tsid.py       # Standing balance
  m3_setup_probe.py         # Robot geometry measurement
  m4_com_sway.py            # CoM tracking
  m5_single_step.py         # Contact switching
  m6a_walk_quasistatic.py   # Quasi static walking
  m6b_walk_lipm.py          # LIPM walking
  m7_walk_with_arm.py       # Walking with arm task
  m8_online_mpc.py          # Online MPC, no arm
  m9_mpc_with_arm.py        # Online MPC with arm task


---

## Technical Details

### Why MuJoCo as ground truth

The earlier version of this pipeline was developed against Pinocchio's own integrator. When run in MuJoCo the robot swung instead of walking. From M0 onwards every result must work through MuJoCo's contact solver. Nothing counts as working until it does.

### Control authority bound

The maximum lateral acceleration available from the CoP on a 6 cm foot is:

a_max = omega^2 * l_y = 3.794^2 * 0.03 = 0.432 m/s^2


This bound determines which step durations are physically feasible, independent of gains. T_step must satisfy:

T_step >= sqrt(2 * lateral_traverse / a_max)


### Herdt MPC formulation

Decision vector per axis: [Z | F] where Z is N CoP samples and F is m footstep positions.

ZMP reference: ZMP_ref = U_c * f_current + U * F

The ZMP constraint is written relative to F so as footsteps move the feasible CoP region moves with them. This gives the robot recovery authority that CoP feedback alone cannot provide.

### Phase lead fix

The support foot toggle inside the MPC originally fired at foot plant, already in single support. We shift it one MPC sample (100ms) earlier so it fires during double support when bilateral contact provides authority to shift load. This single fix unlocked stable walking at vx=0.2 m/s.

### Wieber feedback

Every MPC tick the state is re-anchored to measurements:

x_hat = [c_meas, cd_meas, a_pred]


Position and velocity from Pinocchio forward kinematics. Acceleration from the previous solve's model prediction. This reduced DCM error from 32.7 mm (offline) to 7.2 mm (online).

---

## References

Herdt et al., 2010. Online Walking Motion Generation with Automatic Footstep Placement.
Wieber, 2006. Trajectory Free Linear Model Predictive Control for Stable Walking.
Kajita et al., 2003. Biped Walking Pattern Generation by using Preview Control.
Escande et al., 2014. Hierarchical Quadratic Programming.
