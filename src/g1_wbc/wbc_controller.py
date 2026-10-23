#!/usr/bin/env python3


import numpy as np
import pinocchio as pin
import tsid

# ------------------------------------------------------------ model paths
URDF = "src/g1_wbc/models/g1_23dof_rev_1_0.urdf"
MESH_DIR = "src/g1_wbc/models"

# --------------------------------------------------- standing configuration
# measured in tests/m3_setup_probe.py
PELVIS_H = 0.7729
STANDING_POSE = {"hip_pitch": -0.20, "knee": 0.42, "ankle_pitch": -0.23}

LF_FRAME = "left_ankle_roll_link"
RF_FRAME = "right_ankle_roll_link"
FRAMES = {"lf": LF_FRAME, "rf": RF_FRAME}

LH_FRAME = "left_wrist_roll_rubber_hand"
RH_FRAME = "right_wrist_roll_rubber_hand"
HAND_FRAMES = {"lh": LH_FRAME, "rh": RH_FRAME}

WAIST_FRAME = "pelvis"

# four sole corners in the ankle_roll frame. z is the sphere BOTTOM:
# centre at -0.03, radius 0.005.
CONTACT_POINTS = np.array([
    [-0.05, -0.05,  0.12,  0.12],
    [ 0.025, -0.025, 0.03, -0.03],
    [-0.035, -0.035, -0.035, -0.035],
])
CONTACT_NORMAL = np.array([0.0, 0.0, 1.0])

MU = 0.7
F_MIN = 1.0
F_MAX = 700.0


def se3_to_vec(H):
    """Pack an SE3 into TSID's 12-vector: translation, then rotation
    flattened column-major (Eigen's storage order)."""
    vec = np.zeros(12)
    vec[0:3] = H.translation
    vec[3:12] = H.rotation.flatten(order="F")
    return vec


class G1BalanceController:

    def __init__(self, kp_com=20.0, kp_posture=20.0, kp_contact=30.0,
                 kp_foot=100.0, kp_waist=500.0, kp_hand=100.0,
                 w_com=1.0, w_posture=1e-3, w_foot=1.0, w_waist=1.0,
                 w_hand=1e-1, w_force=1e-5,
                 w_arm_posture=0.0, kp_arm_posture=200.0):
        self.robot = tsid.RobotWrapper(URDF, [MESH_DIR],
                                       pin.JointModelFreeFlyer(), False)
        self.model = self.robot.model()
        self.na = self.robot.na

        self.formulation = tsid.InverseDynamicsFormulationAccForce(
            "tsid", self.robot, False)

        self.kp_com = kp_com
        self.kp_posture = kp_posture
        self.kp_contact = kp_contact
        self.kp_foot = kp_foot
        self.kp_waist = kp_waist
        self.kp_hand = kp_hand
        self.w_com = w_com
        self.w_posture = w_posture
        self.w_foot = w_foot
        self.w_waist = w_waist
        self.w_hand = w_hand
        self.w_force = w_force
        # optional stiff posture task on the arm joints only (0 = off)
        self.w_arm_posture = w_arm_posture
        self.kp_arm_posture = kp_arm_posture

        self.in_contact = {"lf": True, "rf": True}
        self.contact_order = ["lf", "rf"]
        self.hand_active = {"lh": False, "rh": False}

    # ------------------------------------------------------------ setup
    def setup(self, q0, v0):
        f = self.formulation
        f.computeProblemData(0.0, q0, v0)
        data = f.data()

        # ---- rigid contacts (hard constraints) ------------------------
        self.contacts = {}
        for side in self.contact_order:
            frame = FRAMES[side]
            c = tsid.Contact6d(f"contact_{side}", self.robot, frame,
                               CONTACT_POINTS, CONTACT_NORMAL,
                               MU, F_MIN, F_MAX)
            c.setKp(self.kp_contact * np.ones(6))
            c.setKd(2.0 * np.sqrt(self.kp_contact) * np.ones(6))
            c.setReference(
                self.robot.framePosition(data, self.model.getFrameId(frame)))
            f.addRigidContact(c, self.w_force)
            self.contacts[side] = c

        # ---- foot SE3 tasks (created, NOT registered) ------------------
        self.foot_tasks = {}
        self.foot_samples = {}
        for side, frame in FRAMES.items():
            task = tsid.TaskSE3Equality(f"task-{side}", self.robot, frame)
            task.setKp(self.kp_foot * np.ones(6))
            task.setKd(2.0 * np.sqrt(self.kp_foot) * np.ones(6))
            self.foot_tasks[side] = task
            self.foot_samples[side] = tsid.TrajectorySample(12, 6)
            H = self.robot.framePosition(data, self.model.getFrameId(frame))
            self.set_foot_reference(side, H)

        # ---- CoM task ---------------------------------------------------
        self.com_task = tsid.TaskComEquality("task-com", self.robot)
        self.com_task.setKp(self.kp_com * np.ones(3))
        self.com_task.setKd(2.0 * np.sqrt(self.kp_com) * np.ones(3))
        f.addMotionTask(self.com_task, self.w_com, 1, 0.0)

        self.com0 = self.robot.com(data).copy()
        self.com_sample = tsid.TrajectorySample(3)
        self.set_com_reference(self.com0)

        # ---- waist orientation ------------------------------------------
        # Required once one foot leaves the ground; see module docstring.
        self.waist_task = tsid.TaskSE3Equality("task-waist", self.robot,
                                               WAIST_FRAME)
        self.waist_task.setKp(self.kp_waist * np.ones(6))
        self.waist_task.setKd(2.0 * np.sqrt(self.kp_waist) * np.ones(6))
        mask = np.zeros(6)
        mask[3:6] = 1.0                       # orientation only
        self.waist_task.setMask(mask)
        f.addMotionTask(self.waist_task, self.w_waist, 1, 0.0)

        H_waist = self.robot.framePosition(
            data, self.model.getFrameId(WAIST_FRAME))
        self.waist_sample = tsid.TrajectorySample(12, 6)
        self.waist_sample.value(se3_to_vec(H_waist))
        self.waist_sample.derivative(np.zeros(6))
        self.waist_sample.second_derivative(np.zeros(6))
        self.waist_task.setReference(self.waist_sample)

        # ---- hand tasks (created, NOT registered -- see add_hand_task) --
        # Position only. The 23-dof G1 has 5 arm joints per side, which
        # cannot track a full 6D pose, so orientation is left free.
        self.hand_tasks = {}
        self.hand_samples = {}
        self.hand0 = {}
        for side, frame in HAND_FRAMES.items():
            task = tsid.TaskSE3Equality(f"task-{side}", self.robot, frame)
            task.setKp(self.kp_hand * np.ones(6))
            task.setKd(2.0 * np.sqrt(self.kp_hand) * np.ones(6))
            m = np.zeros(6)
            m[0:3] = 1.0                      # translation only
            task.setMask(m)
            self.hand_tasks[side] = task
            self.hand_samples[side] = tsid.TrajectorySample(12, 6)
            H = self.robot.framePosition(data, self.model.getFrameId(frame))
            self.hand0[side] = pin.SE3(H.rotation.copy(), H.translation.copy())
            self.set_hand_reference(side, H.translation)

        # ---- posture task (regulariser) ---------------------------------
        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)
        self.posture_task.setKp(self.kp_posture * np.ones(self.na))
        self.posture_task.setKd(2.0 * np.sqrt(self.kp_posture) * np.ones(self.na))
        f.addMotionTask(self.posture_task, self.w_posture, 1, 0.0)

        self.posture_sample = tsid.TrajectorySample(self.na)
        self.posture_sample.value(q0[7:].copy())
        self.posture_sample.derivative(np.zeros(self.na))
        self.posture_sample.second_derivative(np.zeros(self.na))
        self.posture_task.setReference(self.posture_sample)

        # ---- arm posture task (optional): holds shoulders/elbows/wrists
        # at the initial pose much more stiffly than the global regulariser
        if self.w_arm_posture > 0.0:
            names = [str(n) for n in self.model.names][2:]
            mask = np.array([1.0 if any(k in n for k in
                             ("shoulder", "elbow", "wrist")) else 0.0
                             for n in names])
            self.arm_posture_task = tsid.TaskJointPosture(
                "task-arm-posture", self.robot)
            self.arm_posture_task.setKp(self.kp_arm_posture * np.ones(self.na))
            self.arm_posture_task.setKd(
                2.0 * np.sqrt(self.kp_arm_posture) * np.ones(self.na))
            self.arm_posture_task.setMask(mask)
            f.addMotionTask(self.arm_posture_task, self.w_arm_posture, 1, 0.0)
            self.arm_posture_task.setReference(self.posture_sample)

        # ---- solver ------------------------------------------------------
        self.solver = tsid.SolverHQuadProgFast("qp")
        self.solver.resize(f.nVar, f.nEq, f.nIn)

        return self.com0

    # -------------------------------------------------------- references
    def set_com_reference(self, pos, vel=None, acc=None):
        self.com_sample.value(np.asarray(pos, float))
        self.com_sample.derivative(np.zeros(3) if vel is None
                                   else np.asarray(vel, float))
        self.com_sample.second_derivative(np.zeros(3) if acc is None
                                          else np.asarray(acc, float))
        self.com_task.setReference(self.com_sample)

    def set_foot_reference(self, side, H, vel=None, acc=None):
        s = self.foot_samples[side]
        s.value(se3_to_vec(H))
        s.derivative(np.zeros(6) if vel is None else np.asarray(vel, float))
        s.second_derivative(np.zeros(6) if acc is None else np.asarray(acc, float))
        self.foot_tasks[side].setReference(s)

    def set_hand_reference(self, side, pos, vel=None, acc=None):
        """Position-only hand reference. Orientation entries in the
        sample are kept at the initial pose and masked out anyway."""
        H = pin.SE3(self.hand0[side].rotation.copy(),
                    np.asarray(pos, float))
        s = self.hand_samples[side]
        s.value(se3_to_vec(H))
        v6 = np.zeros(6); a6 = np.zeros(6)
        if vel is not None:
            v6[0:3] = np.asarray(vel, float)
        if acc is not None:
            a6[0:3] = np.asarray(acc, float)
        s.derivative(v6)
        s.second_derivative(a6)
        self.hand_tasks[side].setReference(s)

    def add_hand_task(self, side):
        """Register a hand task. Weight sits an order below the CoM task
        so the arm can never compete with balance."""
        if self.hand_active[side]:
            return
        self.formulation.addMotionTask(self.hand_tasks[side],
                                       self.w_hand, 1, 0.0)
        self.hand_active[side] = True

    def remove_hand_task(self, side):
        if not self.hand_active[side]:
            return
        self.formulation.removeTask(f"task-{side}", 0.0)
        self.hand_active[side] = False

    def frame_position(self, frame_name):
        return self.robot.framePosition(
            self.formulation.data(), self.model.getFrameId(frame_name))

    def hand_placement(self, side):
        return self.frame_position(HAND_FRAMES[side])

    def foot_placement(self, side):
        return self.frame_position(FRAMES[side])

    # --------------------------------------------------- contact switching
    def break_contact(self, side, transition=0.0):
        """Release a foot. The motion task goes in FIRST so the foot is
        never briefly ungoverned."""
        if not self.in_contact[side]:
            return
        self.set_foot_reference(side, self.foot_placement(side))
        self.formulation.addMotionTask(self.foot_tasks[side],
                                       self.w_foot, 1, 0.0)
        self.formulation.removeRigidContact(f"contact_{side}", transition)
        self.in_contact[side] = False
        self.contact_order.remove(side)

    def make_contact(self, side, H=None):
        """Plant a foot. Contact goes in FIRST, then the motion task comes
        out. fMax MUST be restored -- see the module docstring."""
        if self.in_contact[side]:
            return
        c = self.contacts[side]
        c.setMaxNormalForce(F_MAX)
        c.setMinNormalForce(F_MIN)
        c.setReference(self.foot_placement(side) if H is None else H)
        self.formulation.addRigidContact(c, self.w_force)
        self.formulation.removeTask(f"task-{side}", 0.0)
        self.in_contact[side] = True
        self.contact_order.append(side)

    # ------------------------------------------------------------ solve
    def compute(self, t, q, v):
        """One control cycle. Returns (tau_23, status); tau is None on
        solver failure. Resized every cycle because adding or removing a
        contact or task changes the problem dimensions."""
        f = self.formulation
        HQPData = f.computeProblemData(t, q, v)
        self.solver.resize(f.nVar, f.nEq, f.nIn)
        sol = self.solver.solve(HQPData)
        self.last_sol = sol
        if sol.status != 0:
            return None, sol.status
        return f.getActuatorForces(sol), 0

    def contact_normal_forces(self):
        """Vertical contact force per foot, from the last solution.
        Uses contact_order because the layout changes after a switch."""
        fz = {"lf": 0.0, "rf": 0.0}
        if getattr(self, "last_sol", None) is None:
            return fz
        fall = np.asarray(
            self.formulation.getContactForces(self.last_sol)).ravel()
        for k, side in enumerate(self.contact_order):
            blk = fall[k * 12:(k + 1) * 12]
            if blk.size == 12:
                fz[side] = float(np.sum(blk[2::3]))
        return fz
