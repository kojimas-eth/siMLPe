import numpy as np
from scipy.spatial.transform import Rotation as R

class GlobalTrajectoryExtrapolator:
    def __init__(self, dt=1.0/25.0):
        """
        dt: Time step between frames (e.g., 1/25 for 25fps data)
        """
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        
        # State Covariance matrix
        self.P = np.eye(6) * 500.0 
        
        # State Transition Matrix (Constant Velocity Model)
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        
        # Measurement Matrix (We only measure position [x, y, z])
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # Measurement Noise Covariance (How much we trust the ZED tracking)
        self.R_noise = np.eye(3) * 0.01 
        
        # Process Noise Covariance (How much we expect the velocity to randomly change)
        # Higher means the filter adapts faster to sudden stops/starts
        q = 0.7 
        self.Q = np.eye(6) * q
        
        # Orientation state
        self.last_quat = np.array([0, 0, 0, 1])
        self.angular_velocity = np.zeros(3) # Rotation vector representation
        
    def update(self, observed_position, observed_quat):
        """
        Feed the current frame's global root position and orientation to the filter.
        observed_position: shape (3,) -> [x, y, z]
        observed_quat: shape (4,) -> [x, y, z, w]
        """
        # --- 1. Update Position Kalman Filter ---
        z = np.array(observed_position).reshape(3, 1)
        
        # Predict Step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update Step
        y = z - (self.H @ x_pred) # Innovation
        S = self.H @ P_pred @ self.H.T + self.R_noise
        K = P_pred @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = x_pred + (K @ y)
        self.P = (np.eye(6) - (K @ self.H)) @ P_pred
        
        # --- 2. Update Orientation (Angular Velocity) ---
        # Calculate rotation difference between current and last frame
        r_curr = R.from_quat(observed_quat)
        r_last = R.from_quat(self.last_quat)
        
        # r_diff represents the rotation from last frame to current frame
        r_diff = r_curr * r_last.inv()
        
        # Convert to rotation vector (direction is axis, magnitude is angle)
        # Divide by dt to get angular velocity
        current_angular_vel = r_diff.as_rotvec() / self.dt
        
        # Smooth the angular velocity (Simple Exponential Smoothing)
        alpha = 0.3 # Smoothing factor (0.0 = completely ignore new, 1.0 = completely trust new)
        self.angular_velocity = (alpha * current_angular_vel) + ((1 - alpha) * self.angular_velocity)
        self.last_quat = observed_quat

    def predict_future(self, num_frames):
        """
        Extrapolates the trajectory `num_frames` into the future.
        Returns:
            future_positions: (num_frames, 3)
            future_quats: (num_frames, 4)
        """
        future_positions = []
        future_quats = []
        
        # Clone current state so we don't mess up the actual filter
        current_x = self.x.copy()
        current_r = R.from_quat(self.last_quat)
        
        for _ in range(num_frames):
            # Predict next position using KF transition matrix
            current_x = self.F @ current_x
            future_positions.append(current_x[:3, 0])
            
            # Predict next orientation using smoothed angular velocity
            # rotation to apply = angular_velocity * dt
            step_rot = R.from_rotvec(self.angular_velocity * self.dt)
            current_r = step_rot * current_r
            future_quats.append(current_r.as_quat())
            
        return np.array(future_positions), np.array(future_quats)