import numpy as np
from scipy.stats import wasserstein_distance

class DataDriftDetector:
    """
    MLOps component: Data Drift Detection.
    Computes Wasserstein distance (Earth Mover's Distance) to detect if the patient
    population distribution has shifted compared to the training distribution.
    """
    def __init__(self, reference_data: np.ndarray):
        """
        reference_data: Matrix of historical data (m_ref x n_features) used for training.
        """
        self.reference_data = reference_data
        self.n_features = reference_data.shape[1]
        self.feature_names = ["SpO2", "HeartRate", "Temperature", "MAP", "Lactate"]
        
    def check_drift(self, incoming_data: np.ndarray, threshold=1.5):
        """
        incoming_data: Matrix of recent data (m_new x n_features).
        Returns a dictionary with drift alerts per feature.
        """
        drift_report = {}
        is_drifting = False
        
        for i in range(self.n_features):
            ref_col = self.reference_data[:, i]
            # Drop NaNs from incoming
            inc_col = incoming_data[:, i]
            inc_col = inc_col[~np.isnan(inc_col)]
            
            if len(inc_col) == 0:
                continue
                
            distance = wasserstein_distance(ref_col, inc_col)
            
            # Simple heuristic threshold for demo purposes
            feature_drift = distance > threshold
            if feature_drift:
                is_drifting = True
                
            drift_report[self.feature_names[i]] = {
                "wasserstein_distance": distance,
                "drift_detected": feature_drift
            }
            
        return {
            "global_drift_alert": is_drifting,
            "details": drift_report
        }
