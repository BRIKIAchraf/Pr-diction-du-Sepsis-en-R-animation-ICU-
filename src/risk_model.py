import numpy as np
from src.math_engine import ICUMathEngine

class ExponentialRiskModel:
    """
    Linearized Exponential Risk Model.
    Risque = a * e^(b*x)
    Transformed to: ln(Risque) = ln(a) + b*x
    Let Y = ln(Risque), A = ln(a), B = b.
    Then Y = A + B*x.
    """
    def __init__(self):
        self.intercept = 0.0 # ln(a)
        self.coefficients = None # B
        
    def fit(self, X: np.ndarray, y_risk: np.ndarray):
        """
        Fits the model using QR Householder for maximum numerical stability.
        X: Feature matrix (m samples, n features)
        y_risk: Actual risk scores (e.g. SOFA proxy, strictly positive)
        """
        # Ensure strictly positive values for log
        y_safe = np.maximum(y_risk, 1e-5)
        Y = np.log(y_safe)
        
        m, n = X.shape
        
        # We need to add a column of 1s for the intercept (A)
        # Equation: Y = 1*A + X_1*B_1 + X_2*B_2 ...
        X_design = np.hstack([np.ones((m, 1)), X])
        
        # Solve using our stable QR Householder method
        # X_design * params = Y
        params = ICUMathEngine.solve_qr(X_design, Y)
        
        self.intercept = params[0]
        self.coefficients = params[1:]
        
        # We can also compute a simple covariance matrix for Mahalanobis training
        self.mu = np.mean(X, axis=0)
        self.cov = np.cov(X, rowvar=False)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the raw exponential risk score.
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction.")
            
        # ln(Risque) = ln(a) + b*x
        Y_pred = self.intercept + np.dot(X, self.coefficients)
        
        # Risque = e^Y
        return np.exp(Y_pred)
        
    def get_anomaly_score(self, x: np.ndarray) -> float:
        """
        Returns Mahalanobis distance squared for a single observation.
        """
        return ICUMathEngine.mahalanobis_distance(x, self.mu, self.cov)
