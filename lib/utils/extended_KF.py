import numpy as np
from scipy.spatial.transform import Rotation as R

class MEKFTrajectoryExtrapolator:
    def __init__(self, dt=1.0/15.0):
        """
        dt: Time step between frames (e.g., 1/15 for 15fps)
        """
        self.dt = dt
        
        # ==========================================
        # 1. POSITION KALMAN FILTER (Linear KF)
        # State: [x, y, z, vx, vy, vz]
        # ==========================================
        self.x_pos = np.zeros((6, 1))
        self.P_pos = np.eye(6) * 500.0 #Uncertainty for first step
        
        self.F_pos = np.eye(6)
        self.F_pos[0, 3] = self.dt
        self.F_pos[1, 4] = self.dt
        self.F_pos[2, 5] = self.dt
        
        self.H_pos = np.zeros((3, 6))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        self.H_pos[2, 2] = 1.0
        
        self.R_pos = np.eye(3) * 0.01  # ZED position noise
        self.Q_pos = np.eye(6) * 0.5   # Target maneuverability (process noise)
        
        # ==========================================
        # 2. ORIENTATION MEKF (Error-State KF)
        # Nominal State: r_est (Rotation), omega_est (Angular Velocity)
        # Error State: [delta_theta_x, delta_theta_y, delta_theta_z, delta_omega_x, delta_omega_y, delta_omega_z]
        # ==========================================
        self.r_est = R.identity()
        self.omega_est = np.zeros(3) # [wx, wy, wz]
        
        # Error State Covariance (6x6)
        self.P_ori = np.eye(6) * 10.0
        
        # Error State Transition Matrix
        self.F_ori = np.eye(6)
        self.F_ori[0:3, 3:6] = np.eye(3) * self.dt
        
        # Measurement Matrix (We only directly measure orientation, not angular velocity)
        self.H_ori = np.zeros((3, 6))
        self.H_ori[0:3, 0:3] = np.eye(3)
        
        # Orientation Tuning Parameters
        self.R_ori = np.eye(3) * 0.05 # Trust in ZED camera's orientation (lower = trust ZED more)
        
        self.Q_ori = np.eye(6)
        self.Q_ori[0:3, 0:3] *= 0.01  # Small noise on rotation kinematics
        self.Q_ori[3:6, 3:6] *= 2.0   # High noise on angular velocity (humans change turning speed fast)

        self.initialized = False

    def update(self, observed_position, observed_quat):
        """
        observed_position: (3,) [x, y, z]
        observed_quat: (4,) [x, y, z, w]
        """
        # --- Initialize states on first frame ---
        if not self.initialized:
            self.x_pos[0:3, 0] = observed_position
            self.r_est = R.from_quat(observed_quat)
            self.initialized = True
            return

        # ==========================================
        # 1. UPDATE POSITION (Standard Linear KF)
        # ==========================================
        z_pos = np.array(observed_position).reshape(3, 1)
        
        # Predict
        x_pred_pos = self.F_pos @ self.x_pos
        P_pred_pos = self.F_pos @ self.P_pos @ self.F_pos.T + self.Q_pos
        
        # Update
        y_pos = z_pos - (self.H_pos @ x_pred_pos)
        S_pos = self.H_pos @ P_pred_pos @ self.H_pos.T + self.R_pos
        K_pos = P_pred_pos @ self.H_pos.T @ np.linalg.inv(S_pos)
        
        self.x_pos = x_pred_pos + (K_pos @ y_pos)
        self.P_pos = (np.eye(6) - (K_pos @ self.H_pos)) @ P_pred_pos

        # ==========================================
        # 2. UPDATE ORIENTATION (MEKF)
        # ==========================================
        r_meas = R.from_quat(observed_quat)
        
        # Predict Nominal State
        # r_pred = r_prev * exp(omega * dt)
        step_rot = R.from_rotvec(self.omega_est * self.dt)
        r_pred = step_rot * self.r_est 
        omega_pred = self.omega_est.copy()
        
        # Predict Error Covariance
        P_pred_ori = self.F_ori @ self.P_ori @ self.F_ori.T + self.Q_ori
        
        # Calculate Measurement Residual (Error Rotation)
        # r_meas = r_err * r_pred  =>  r_err = r_meas * r_pred^-1
        r_err = r_meas * r_pred.inv()
        
        # Project quaternion error into the flat 3D tangent space (Rotation Vector)
        z_ori_err = r_err.as_rotvec().reshape(3, 1)
        
        # Calculate Kalman Gain
        S_ori = self.H_ori @ P_pred_ori @ self.H_ori.T + self.R_ori
        K_ori = P_pred_ori @ self.H_ori.T @ np.linalg.inv(S_ori)
        
        # Compute Error State Update: [delta_theta, delta_omega]
        delta_x = K_ori @ z_ori_err
        delta_theta = delta_x[0:3, 0]
        delta_omega = delta_x[3:6, 0]
        
        # Apply Error State to Nominal State (Multiplicative Update)
        update_rot = R.from_rotvec(delta_theta)
        self.r_est = update_rot * r_pred
        self.omega_est = omega_pred + delta_omega
        
        # Update Error Covariance
        self.P_ori = (np.eye(6) - (K_ori @ self.H_ori)) @ P_pred_ori

    def predict_future(self, num_frames, lock_upright=True, damping=0.9):
        """
        Extrapolates the trajectory `num_frames` into the future.
        lock_upright: If True, forces the angular velocity around X (Pitch) and Z (Roll) to zero. 
        damping: Friction applied to angular velocity (0.9 = loses 10% speed per frame).
        """
        future_positions = []
        future_quats = []
        
        # Clone states
        current_x_pos = self.x_pos.copy()
        current_r = R.from_quat(self.r_est.as_quat())
        current_omega = self.omega_est.copy()
        
        if lock_upright:
            current_omega[0] = 0.0 # Lock Pitch
            current_omega[2] = 0.0 # Lock Roll
            
        # Ignore micro-jitters
        if np.linalg.norm(current_omega) < 0.05:
            current_omega = np.zeros(3)
        
        for _ in range(num_frames):
            # Predict Position
            current_x_pos = self.F_pos @ current_x_pos
            future_positions.append(current_x_pos[:3, 0])
            
            # Predict Orientation
            current_omega = current_omega * damping # Apply friction
            step_rot = R.from_rotvec(current_omega * self.dt)
            current_r = step_rot * current_r
            future_quats.append(current_r.as_quat())
            
        return np.array(future_positions), np.array(future_quats)