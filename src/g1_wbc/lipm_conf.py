from pathlib import Path
import numpy as np
import pinocchio as pin
import pinocchio.visualize

LINE_WIDTH = 60
DATA_FILE_LIPM = "results/g1_walking_traj_lipm.npz"
DATA_FILE_TSID = "results/g1_walking_traj_tsid.npz"

path = str(Path(__file__).resolve().parent / "models")
urdf = path + "/g1_23dof_rev_1_0.urdf"

#robot parameter
nv = 29
rf_frame_name = "right_ankle_roll_link"
lf_frame_name = "left_ankle_roll_link"
q0 = np.zeros(30)
q0[2] = 0.7729    # pelvis height
q0[6] = 1.0     # quaternion w — upright orientation
# left leg
q0[7]  = -0.20   # left hip pitch
q0[10] =  0.42   # left knee
q0[11] = -0.23   # left ankle pitch
# right leg
q0[13] = -0.20   # right hip pitch
q0[16] =  0.42   # right knee
q0[17] = -0.23   # right ankle pitch
foot_scaling = 1.0
lxp = foot_scaling * 0.12
lxn = foot_scaling * 0.05
lyp = foot_scaling * 0.03
lyn = foot_scaling * 0.03
lz = 0.03
mu = 0.7
fMin = 1.0
fMax = 700.0
contactNormal = np.matrix([0.0, 0.0, 1.0]).T

#lipm config
alpha = 10 ** (2)
beta = 0
gamma = 10 ** (-1)
h = 0.6816
g = 9.81
foot_step_0 = np.array([0.0, -0.1185])
dt_mpc = 0.133333333333333
T_step = 1.6
step_length = 0.20
step_height = 0.05
nb_steps = 3

#tsid config
dt = 0.002
T_pre = 2.5
T_post = 1.5
w_com = 1.0
w_cop = 0.0
w_am = 1e-4
w_foot = 1e2
w_waist = 1.0
w_contact = 1e2
w_posture = 1e-2
w_forceRef = 1e-5
w_torque_bounds = 0.0
w_joint_bounds = 0.0
tau_max_scaling = 1.45
v_max_scaling = 0.8
kp_contact = 30.0
kp_foot = 30.0
kp_com = 30.0
kp_am = 10.0
kp_posture = 10.0
kp_waist = 500.0
gain_vector = kp_posture * np.ones(nv - 6)
gain_vector[12] = kp_posture * 50.0
masks_posture = np.ones(nv - 6)

#viewer config
viewer = pin.visualize.MeshcatVisualizer 
PRINT_N = 500
DISPLAY_N =  20
CAMERA_TRANSFORM = [
    3.578777551651001,
    1.2937744855880737,
    0.8885031342506409,
    0.4116811454296112,
    0.5468055009841919,
    0.6109083890914917,
    0.3978860676288605,
]
