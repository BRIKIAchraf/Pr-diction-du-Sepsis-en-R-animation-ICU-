import numpy as np

class ClinicalKalmanFilter:
    """
    A linear Kalman Filter customized for ICU continuous monitoring.
    Used to smooth noisy sensor data and impute missing values (NaNs).
    State vector: [SpO2, HR, Temp, MAP, Lactate]
    """
    def __init__(self, n_states=5):
        self.n_states = n_states
        # Initial State (SpO2, HR, Temp, MAP, Lactate)
        self.x = np.array([98.0, 70.0, 37.0, 90.0, 1.0])
        # Initial Uncertainty (Covariance)
        self.P = np.eye(n_states) * 10.0
        
        # State Transition Matrix (Assuming constant state model for short windows)
        self.F = np.eye(n_states)
        
        # Measurement Matrix (We measure all states directly when available)
        self.H = np.eye(n_states)
        
        # Process Noise Covariance (How much we expect the patient to naturally change)
        self.Q = np.diag([0.1, 1.0, 0.05, 1.0, 0.05])
        
        # Measurement Noise Covariance (Sensor reliability)
        self.R = np.diag([2.0, 5.0, 0.5, 5.0, 0.2])
        
    def predict(self):
        """
        Predicts the next state.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
        
    def update(self, z):
        """
        Updates the state with a new measurement z.
        Handles NaNs in measurement by dynamically adjusting the H matrix.
        """
        # Create a dynamic H matrix that ignores missing values
        H_dyn = np.copy(self.H)
        z_dyn = np.copy(z)
        
        for i in range(self.n_states):
            if np.isnan(z[i]):
                H_dyn[i, i] = 0.0 # Ignore this measurement
                z_dyn[i] = 0.0    # Dummy value, will be multiplied by 0
                
        # Innovation (Residual)
        y = z_dyn - np.dot(H_dyn, self.x)
        
        # Innovation Covariance
        S = np.dot(np.dot(H_dyn, self.P), H_dyn.T) + self.R
        
        # Kalman Gain
        # S can be singular if we have 0s, use pseudo-inverse
        K = np.dot(np.dot(self.P, H_dyn.T), np.linalg.pinv(S))
        
        # State Update
        self.x = self.x + np.dot(K, y)
        
        # Covariance Update
        I = np.eye(self.n_states)
        self.P = np.dot(I - np.dot(K, H_dyn), self.P)
        
        return self.x
