import numpy as np
import numpy.random as random
from .constants import MPCConstants


class AutomaticLMPC:
    """Linear MPC with automatic footstep planning for ZMP control.

    Based on the paper:
    Herdt et al., "Online Walking Motion Generation with Automatic Foot Step Placement", 2010.

    Reference implementation provided with the project.

    Two edits only:
      1. Import path (config.constants -> .constants) for the src/herdt package.
      2. np.linalg.eigvals -> np.linalg.eigvalsh in compute_cost_function.
         Q_k is symmetric; eigvals returns a COMPLEX array and `> 0` on complex
         values does not test what was intended. eigvalsh returns real
         eigenvalues in ascending order.
    """

    def __init__(self):
        # General parameters
        self.mpc_const = MPCConstants()
        self.h = self.mpc_const.COM_HEIGHT
        self.g = self.mpc_const.GRAVITY
        self.foot_length = self.mpc_const.FOOT_LENGTH
        self.foot_width = self.mpc_const.FOOT_WIDTH

        # MPC Parameters
        self.T = self.mpc_const.SAMPLING_TIME
        self.N = self.mpc_const.HORIZON
        self.T_ss = self.mpc_const.T_SINGLE_SUPPORT
        self.T_ds = self.mpc_const.T_DOUBLE_SUPPORT
        self.T_step = self.T_ss + self.T_ds

        # Number of future footsteps in the MPC horizon
        self.m = int(round(self.N * self.T / self.T_step))

        # We assume the initial current support foot is right
        self.is_right_cnt = True

        # Pre-compute discrete dynamics and recursive matrices
        self.discrete_dynamics()
        self.compute_recursive_matrices()

        # References (to be set externally)
        self.X_dot_ref_k = np.zeros(self.N)
        self.Y_dot_ref_k = np.zeros(self.N)
        
        self.prev_u = None

    def discrete_dynamics(self):
        T, h, g = self.T, self.h, self.g

        self.A = np.array([
            [1, T, T**2 / 2],
            [0, 1, T],
            [0, 0, 1]
        ])

        self.B = np.array([
            [T**3 / 6],
            [T**2 / 2],
            [T]
        ])

        self.C = np.array([
            [1, 0, -h / g]
        ])

    def compute_recursive_matrices(self):
        A, B, C, N = self.A, self.B, self.C, self.N

        self.P_ps = np.zeros((N, 3))
        self.P_vs = np.zeros((N, 3))
        self.P_as = np.zeros((N, 3))
        self.P_zs = np.zeros((N, 3))

        self.P_pu = np.zeros((N, N))
        self.P_vu = np.zeros((N, N))
        self.P_au = np.zeros((N, N))
        self.P_zu = np.zeros((N, N))

        for k in range(N):
            P = np.linalg.matrix_power(A, k + 1)

            self.P_ps[k, :] = P[0, :]
            self.P_vs[k, :] = P[1, :]
            self.P_as[k, :] = P[2, :]
            self.P_zs[k, :] = C @ P

            for j in range(k + 1):
                PU = np.linalg.matrix_power(A, k - j)
                Pb = PU @ B

                self.P_pu[k, j] = Pb[0, 0]
                self.P_vu[k, j] = Pb[1, 0]
                self.P_au[k, j] = Pb[2, 0]
                self.P_zu[k, j] = (C @ Pb).item()

    def compute_recursive_dynamics(self, x_hat_k, y_hat_k, u_k):
        N = self.N
        m = self.m

        X_com = np.zeros((N, 3))
        Y_com = np.zeros((N, 3))

        X_jerk_k = u_k[0:N]

        X_com[0:N, 0] = (
            np.dot(self.P_ps, x_hat_k)
            + np.dot(self.P_pu, X_jerk_k)
        )

        X_com[0:N, 1] = (
            np.dot(self.P_vs, x_hat_k)
            + np.dot(self.P_vu, X_jerk_k)
        )

        X_com[0:N, 2] = (
            np.dot(self.P_as, x_hat_k)
            + np.dot(self.P_au, X_jerk_k)
        )

        Y_jerk_k = u_k[N + m:2 * N + m]

        Y_com[0:N, 0] = (
            np.dot(self.P_ps, y_hat_k)
            + np.dot(self.P_pu, Y_jerk_k)
        )

        Y_com[0:N, 1] = (
            np.dot(self.P_vs, y_hat_k)
            + np.dot(self.P_vu, Y_jerk_k)
        )

        Y_com[0:N, 2] = (
            np.dot(self.P_as, y_hat_k)
            + np.dot(self.P_au, Y_jerk_k)
        )

        Z_x = (
            np.dot(self.P_zs, x_hat_k)
            + np.dot(self.P_zu, X_jerk_k)
        )

        Z_y = (
            np.dot(self.P_zs, y_hat_k)
            + np.dot(self.P_zu, Y_jerk_k)
        )

        return X_com, Y_com, Z_x, Z_y

    def build_foot_sequence(self, is_right_cnt, m):
        seq = []

        next_is_right = not is_right_cnt

        for _ in range(m):
            seq.append('R' if next_is_right else 'L')
            next_is_right = not next_is_right

        return seq

    def update_step_matrix(self, current_step):
        N = self.N
        m = self.m

        steps_per_foot = int(round(self.T_step / self.T))
        phase_idx = current_step % steps_per_foot

        self.U_c_k = np.zeros(N)
        self.U_k = np.zeros((N, m))

        for i in range(N):
            step_ahead = (phase_idx + i) // steps_per_foot

            if step_ahead == 0:
                self.U_c_k[i] = 1.0

            s = step_ahead - 1

            if 0 <= s < m:
                self.U_k[i, s] = 1.0

    def compute_cost_function(
        self,
        x_hat_k,
        y_hat_k,
        x_fc_k,
        y_fc_k
    ):
        N = self.N
        U_k = self.U_k
        U_c_k = self.U_c_k

        P_vs = self.P_vs
        P_zs = self.P_zs
        P_vu = self.P_vu
        P_zu = self.P_zu

        alpha = self.mpc_const.ALPHA
        beta = self.mpc_const.BETA
        beta_y = getattr(self.mpc_const, "BETA_Y", beta)
        gamma = self.mpc_const.GAMMA

        def q_prime(b):
            Q11 = alpha * np.eye(N) + b * (P_vu.T @ P_vu) + gamma * (P_zu.T @ P_zu)
            Q12 = -gamma * (P_zu.T @ U_k)
            Q21 = -gamma * (U_k.T @ P_zu)
            Q22 = gamma * (U_k.T @ U_k)
            return np.vstack((np.hstack((Q11, Q12)), np.hstack((Q21, Q22))))

        Q_prime_x = q_prime(beta)
        Q_prime_y = q_prime(beta_y)

        nb = N + self.m
        Q_k = np.zeros((2 * nb, 2 * nb))
        Q_k[:nb, :nb] = Q_prime_x
        Q_k[nb:, nb:] = Q_prime_y

        pk_1 = (
            beta * P_vu.T @ (
                P_vs @ x_hat_k - self.X_dot_ref_k
            )
            + gamma * P_zu.T @ (
                P_zs @ x_hat_k - (U_c_k * x_fc_k)
            )
        )

        pk_1 = pk_1.flatten()

        pk_2 = (
            -gamma * U_k.T @ (
                P_zs @ x_hat_k - (U_c_k * x_fc_k)
            )
        )

        pk_2 = pk_2.flatten()

        pk_3 = (
            beta_y * P_vu.T @ (
                P_vs @ y_hat_k - self.Y_dot_ref_k
            )
            + gamma * P_zu.T @ (
                P_zs @ y_hat_k - (U_c_k * y_fc_k)
            )
        )

        pk_3 = pk_3.flatten()

        pk_4 = (
            -gamma * U_k.T @ (
                P_zs @ y_hat_k - (U_c_k * y_fc_k)
            )
        )

        pk_4 = pk_4.flatten()

        pk = np.hstack((
            pk_1,
            pk_2,
            pk_3,
            pk_4
        )).T

        rate_delta = 0
        if self.prev_u is not None:
            for idx in (N, 2*N+self.m):
                Q_k[idx, idx] += rate_delta
                pk[idx] += rate_delta * self.prev_u[idx]

        # Stance width regularization — penalise deviation from nominal
        # stance so the QP doesn't ride the lateral bounds.
        # (QP is min 1/2 u'Qu + p'u, so the linear term is -delta*y_nom.)
        delta = 1e2
        nominal_stance = 0.237
        foot_seq = self.build_foot_sequence(self.is_right_cnt, self.m)
        for j in range(self.m):
            sign = 1.0 if foot_seq[j] == 'L' else -1.0
            y_nom = sign * nominal_stance / 2
            ij = 2 * self.N + self.m + j
            Q_k[ij, ij] += delta
            pk[ij] -= delta * y_nom

        # Q_k is symmetric:
        # use eigvalsh (real, ascending), not eigvals (complex)
        eigenvalues = np.linalg.eigvalsh(Q_k)
        if not np.all(eigenvalues > 0):
            shift = -np.min(eigenvalues) + 1e-4
            Q_k = Q_k + shift * np.eye(Q_k.shape[0])

        return Q_k, pk

    def construct_constraints(
        self,
        x_hat_k,
        y_hat_k,
        x_fc_k,
        y_fc_k
    ):
        N = self.N
        m = self.m

        P_zs = self.P_zs
        P_zu = self.P_zu
        U_k = self.U_k
        U_c_k = self.U_c_k

        tol = self.mpc_const.ZMP_TOLERANCE

        l = self.foot_length / 2 - tol
        w = self.foot_width / 2 - tol

        l_vec = l * np.ones(N)
        w_vec = w * np.ones(N)

        # X-direction ZMP constraints
        A_x_1 = np.hstack((
            P_zu,
            -U_k,
            np.zeros((N, N)),
            np.zeros((N, m))
        ))

        b_x_1 = (
            -np.dot(P_zs, x_hat_k)
            + (U_c_k * x_fc_k)
        ).flatten() - l_vec

        A_x_2 = np.hstack((
            -P_zu,
            U_k,
            np.zeros((N, N)),
            np.zeros((N, m))
        ))

        b_x_2 = (
            np.dot(P_zs, x_hat_k)
            - (U_c_k * x_fc_k)
        ).flatten() - l_vec

        # Y-direction ZMP constraints
        A_y_1 = np.hstack((
            np.zeros((N, N)),
            np.zeros((N, m)),
            P_zu,
            -U_k
        ))

        b_y_1 = (
            -np.dot(P_zs, y_hat_k)
            + (U_c_k * y_fc_k)
        ).flatten() - w_vec

        A_y_2 = np.hstack((
            np.zeros((N, N)),
            np.zeros((N, m)),
            -P_zu,
            U_k
        ))

        b_y_2 = (
            np.dot(P_zs, y_hat_k)
            - (U_c_k * y_fc_k)
        ).flatten() - w_vec

        A_zmp = np.vstack((
            A_x_1,
            A_x_2,
            A_y_1,
            A_y_2
        ))

        b_zmp = np.hstack((
            b_x_1,
            b_x_2,
            b_y_1,
            b_y_2
        ))

        # Footstep constraints
        n_vars = 2 * (N + m)

        A_foot_x = np.zeros((m, n_vars))
        A_foot_y = np.zeros((m, n_vars))

        dim_y = 2 * N + m

        dx_max = self.mpc_const.FOOT_X_MAX_DISTANCE
        dy_min = self.mpc_const.FOOT_Y_MIN_DISTANCE
        dy_max = self.mpc_const.FOOT_Y_MAX_DISTANCE

        l_x, u_x = [], []
        l_y, u_y = [], []

        is_right = self.is_right_cnt
        foot_seq = self.build_foot_sequence(is_right, m)

        for i in range(m):

            # X position constraint
            if i == 0:
                A_foot_x[i, N + i] = 1

                l_x.append(x_fc_k - dx_max)
                u_x.append(x_fc_k + dx_max)

            else:
                A_foot_x[i, N + i - 1] = -1
                A_foot_x[i, N + i] = 1

                l_x.append(-dx_max)
                u_x.append(dx_max)

            # Y position constraint
            A_foot_y[i, dim_y + i] = 1

            if i > 0:
                A_foot_y[i, dim_y + i - 1] = -1

            if foot_seq[i] == 'L':
                l_y.append(
                    dy_min if i > 0 else y_fc_k + dy_min
                )

                u_y.append(
                    dy_max if i > 0 else y_fc_k + dy_max
                )

            else:
                l_y.append(
                    -dy_max if i > 0 else y_fc_k - dy_max
                )

                u_y.append(
                    -dy_min if i > 0 else y_fc_k - dy_min
                )

        l_foot_x = np.array(l_x)
        u_foot_x = np.array(u_x)

        l_foot_y = np.array(l_y)
        u_foot_y = np.array(u_y)

        A_constr = np.vstack((
            A_zmp,
            A_foot_x,
            -A_foot_x,
            A_foot_y,
            -A_foot_y
        ))

        b_constr = np.hstack((
            b_zmp,
            l_foot_x,
            -u_foot_x,
            l_foot_y,
            -u_foot_y
        ))

        return A_constr, b_constr

    def set_ref_velocity(self, v_com_ref):
        self.X_dot_ref_k = np.ones(self.N) * v_com_ref[0]
        self.Y_dot_ref_k = np.ones(self.N) * v_com_ref[1]

    def get_next_footstep(self, u_k):
        N = self.N
        m = self.m

        x_fc_k = u_k[N]
        y_fc_k = u_k[2 * N + m]

        self.is_right_cnt = not self.is_right_cnt

        return x_fc_k, y_fc_k

    def get_ref_zmp(self, x_fc_k, y_fc_k, u_k):
        N = self.N
        m = self.m

        Z_x_ref = (
            self.U_c_k * x_fc_k
            + np.dot(self.U_k, u_k[N:N + m])
        )

        Z_y_ref = (
            self.U_c_k * y_fc_k
            + np.dot(self.U_k, u_k[2 * N + m:])
        )

        return Z_x_ref, Z_y_ref