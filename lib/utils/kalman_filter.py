import numpy as np
from scipy.spatial.transform import Rotation as R

class KFTrajectoryExtrapolator:
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
        # 2. ORIENTATION KALMAN FILTER (1D Linear KF)
        # State: [yaw, yaw_velocity] 
        # ==========================================
        self.x_ori = np.zeros((2, 1))
        self.P_ori = np.eye(2) * 10.0
        
        self.F_ori = np.eye(2)
        self.F_ori[0, 1] = self.dt 
        
        self.H_ori = np.zeros((1, 2))
        self.H_ori[0, 0] = 1.0 # We only measure yaw
        
        self.R_ori = np.array([[0.05]]) # ZED orientation noise
        self.Q_ori = np.array([[0.01, 0.0], 
                               [0.0, 2.0]]) # [Yaw noise, Yaw Velocity noise]

        self.initialized = False

    def update(self, observed_position, observed_quat):
        # Extract Yaw (Y-axis) from quaternion
        euler_angles = R.from_quat(observed_quat).as_euler('xyz')
        yaw_meas = euler_angles[1] 

        # --- Initialize states on first frame ---
        if not self.initialized:
            self.x_pos[0:2, 0] = observed_position[0:2] # Take X and Y
            self.x_ori[0, 0] = yaw_meas                 # Take Yaw
            self.initialized = True
            return

        # ==========================================
        # UPDATE POSITION (2D Linear KF)
        # ==========================================
        z_pos = np.array(observed_position[0:2]).reshape(2, 1)
        
        x_pred_pos = self.F_pos @ self.x_pos
        P_pred_pos = self.F_pos @ self.P_pos @ self.F_pos.T + self.Q_pos
        
        y_pos = z_pos - (self.H_pos @ x_pred_pos)
        S_pos = self.H_pos @ P_pred_pos @ self.H_pos.T + self.R_pos
        K_pos = P_pred_pos @ self.H_pos.T @ np.linalg.inv(S_pos)
        
        self.x_pos = x_pred_pos + (K_pos @ y_pos)
        self.P_pos = (np.eye(4) - (K_pos @ self.H_pos)) @ P_pred_pos

        # ==========================================
        # UPDATE ORIENTATION (1D Linear KF)
        # ==========================================
        z_ori = np.array([[yaw_meas]])
        
        x_pred_ori = self.F_ori @ self.x_ori
        P_pred_ori = self.F_ori @ self.P_ori @ self.F_ori.T + self.Q_ori
        
        y_ori = z_ori - (self.H_ori @ x_pred_ori)
        
        # Angle Wrapping: Force residual to be between -pi and pi
        # This prevents the filter from freaking out when crossing 180 degrees
        y_ori[0, 0] = (y_ori[0, 0] + np.pi) % (2 * np.pi) - np.pi
        
        S_ori = self.H_ori @ P_pred_ori @ self.H_ori.T + self.R_ori
        K_ori = P_pred_ori @ self.H_ori.T @ np.linalg.inv(S_ori)
        
        self.x_ori = x_pred_ori + (K_ori @ y_ori)
        self.P_ori = (np.eye(2) - (K_ori @ self.H_ori)) @ P_pred_ori

    def predict_future(self, num_frames, damping=0.9):
        future_positions = []
        future_quats = []
        
        current_x_pos = self.x_pos.copy()
        current_x_ori = self.x_ori.copy()
        
        # Ignore micro-jitters in rotational velocity
        if abs(current_x_ori[1, 0]) < 0.05:
            current_x_ori[1, 0] = 0.0
            
        for _ in range(num_frames):
            # --- Predict Position (2D) ---
            current_x_pos = self.F_pos @ current_x_pos
            
            # Reconstruct 3D position (inject locked ground_z)
            pos_3d = np.array([current_x_pos[0, 0], current_x_pos[1, 0], self.ground_z])
            future_positions.append(pos_3d)
            
            # --- Predict Orientation (1D Yaw) ---
            current_x_ori = self.F_ori @ current_x_ori
            current_x_ori[1, 0] *= damping # Apply friction to angular velocity
            
            # Reconstruct 3D quaternion (Pitch and Roll are locked to 0)
            step_quat = R.from_euler('xyz', [0.0, current_x_ori[0, 0], 0.0]).as_quat()
            future_quats.append(step_quat)
            
        return np.array(future_positions), np.array(future_quats)