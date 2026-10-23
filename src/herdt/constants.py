"""
MPC constants for the Unitree G1, adapted from the reference constants.py
provided with AutomaticLMPC.

Every value that differs from the reference is marked and justified.
The reference file was written for a generic 60 kg, 0.80 m-CoM robot.
"""


class MPCConstants:
    # ------------------------------------------------------------------
    # Robot parameters
    # ------------------------------------------------------------------

    MASS = 32.107

    GRAVITY = 9.81

    COM_HEIGHT = 0.6816
    FOOT_LENGTH = 0.10
    FOOT_WIDTH = 0.06

    ZMP_TOLERANCE = 0.01

    FOOT_X_MAX_DISTANCE = 0.40
    
    FOOT_Y_MIN_DISTANCE = 0.237
    FOOT_Y_MAX_DISTANCE = 0.3
    
    HORIZON = 24

    SAMPLING_TIME = 0.1
    T_SINGLE_SUPPORT = 0.7
    T_DOUBLE_SUPPORT = 0.1

    ALPHA = 10e-3  
    GAMMA = 10e0  
    BETA = 10e1    
    BETA_Y = 10e1  
