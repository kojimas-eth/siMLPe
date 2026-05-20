import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation as R

class MEKFTrajectoryExtrapolator:
    def __init__(self, dt=1.0/15.0, ground_z=0.0):
        """
        dt: Time step between frames (1/15 for 15fps)
        ground_z: The assumed height of the floor (default 0.0)
        """
        self.dt = dt
        self.ground_z = ground_z
        
        # ==========================================
        # 1. POSITION KALMAN FILTER (2D Linear KF)
        # State: [x, y, vx, vy]
        # ==========================================
        self.x_pos = np.zeros((4, 1))
        self.P_pos = np.eye(4) * 500.0 # Uncertainty for first step
        
        self.F_pos = np.eye(4)
        self.F_pos[0, 2] = self.dt # x += vx * dt
        self.F_pos[1, 3] = self.dt # y += vy * dt
        
        self.H_pos = np.zeros((2, 4))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        
        self.R_pos = np.eye(2) * 0.01  # ZED position noise (X, Y)
        self.Q_pos = np.eye(4) * 0.5   # Target maneuverability
        
        # ==========================================
        # 2. ORIENTATION MEKF (Error-State KF)
        # ==========================================
        self.r_est = R.identity()
        self.omega_est = np.zeros(3)
        self.P_ori = np.eye(6) * 10.0
        
        self.F_ori = np.eye(6)
        self.F_ori[0:3, 3:6] = np.eye(3) * self.dt
        
        self.H_ori = np.zeros((3, 6))
        self.H_ori[0:3, 0:3] = np.eye(3)
        
        self.R_ori = np.eye(3) * 0.05
        self.Q_ori = np.eye(6)
        self.Q_ori[0:3, 0:3] *= 0.01
        self.Q_ori[3:6, 3:6] *= 2.0 

        self.initialized = False

    def update(self, observed_position, observed_quat):
        # --- Initialize states on first frame ---
        if not self.initialized:
            self.x_pos[0:2, 0] = observed_position[0:2] # Only take X and Y
            self.r_est = R.from_quat(observed_quat)
            self.initialized = True
            return

        # ==========================================
        # 1. UPDATE POSITION (2D Linear KF)
        # ==========================================
        # Only take X and Y from the observation
        z_pos = np.array(observed_position[0:2]).reshape(2, 1)
        
        # Predict
        x_pred_pos = self.F_pos @ self.x_pos
        P_pred_pos = self.F_pos @ self.P_pos @ self.F_pos.T + self.Q_pos
        
        # Update
        y_pos = z_pos - (self.H_pos @ x_pred_pos)
        S_pos = self.H_pos @ P_pred_pos @ self.H_pos.T + self.R_pos
        K_pos = P_pred_pos @ self.H_pos.T @ np.linalg.inv(S_pos)
        
        self.x_pos = x_pred_pos + (K_pos @ y_pos)
        self.P_pos = (np.eye(4) - (K_pos @ self.H_pos)) @ P_pred_pos

        # ==========================================
        # 2. UPDATE ORIENTATION (MEKF)
        # (Unchanged)
        # ==========================================
        r_meas = R.from_quat(observed_quat)
        step_rot = R.from_rotvec(self.omega_est * self.dt)
        r_pred = step_rot * self.r_est 
        omega_pred = self.omega_est.copy()
        
        P_pred_ori = self.F_ori @ self.P_ori @ self.F_ori.T + self.Q_ori
        r_err = r_meas * r_pred.inv()
        z_ori_err = r_err.as_rotvec().reshape(3, 1)
        
        S_ori = self.H_ori @ P_pred_ori @ self.H_ori.T + self.R_ori
        K_ori = P_pred_ori @ self.H_ori.T @ np.linalg.inv(S_ori)
        
        delta_x = K_ori @ z_ori_err
        delta_theta = delta_x[0:3, 0]
        delta_omega = delta_x[3:6, 0]
        
        update_rot = R.from_rotvec(delta_theta)
        self.r_est = update_rot * r_pred
        self.omega_est = omega_pred + delta_omega
        self.P_ori = (np.eye(6) - (K_ori @ self.H_ori)) @ P_pred_ori

    def predict_future(self, num_frames, lock_upright=True, damping=0.9):
        future_positions = []
        future_quats = []
        
        current_x_pos = self.x_pos.copy()
        current_r = R.from_quat(self.r_est.as_quat())
        current_omega = self.omega_est.copy()
        
        if lock_upright:
            current_omega[0] = 0.0 # Lock Pitch
            current_omega[2] = 0.0 # Lock Roll
            
        if np.linalg.norm(current_omega) < 0.05:
            current_omega = np.zeros(3)
        
        for _ in range(num_frames):
            # Predict Position (2D)
            current_x_pos = self.F_pos @ current_x_pos
            
            # Reconstruct the 3D position by injecting the locked Z value
            pos_3d = np.array([current_x_pos[0, 0], current_x_pos[1, 0], self.ground_z])
            future_positions.append(pos_3d)
            
            # Predict Orientation
            current_omega = current_omega * damping 
            step_rot = R.from_rotvec(current_omega * self.dt)
            current_r = step_rot * current_r
            future_quats.append(current_r.as_quat())
            
        return np.array(future_positions), np.array(future_quats)