import numpy as np

class LinearSHAPExplainer:
    """
    Custom Explainable AI (XAI) implementation for our Linearized Risk Model.
    Since Y = A + BX, the exact SHAP value for feature i is simply:
    phi_i(x) = B_i * (x_i - E[x_i])
    
    This provides 100% accurate, fast, exact explanations without needing the heavy 
    kernel explainer from the `shap` package, perfectly suited for our architecture.
    """
    def __init__(self, model):
        self.coefficients = model.coefficients
        self.base_value = model.intercept + np.dot(model.mu, model.coefficients)
        self.mu = model.mu
        self.feature_names = ["SpO2", "HeartRate", "Temperature", "MAP", "Lactate"]
        
    def explain(self, x: np.ndarray):
        """
        Returns a dictionary explaining the contribution of each feature to the risk score.
        """
        # Calculate exact SHAP values for the linear model
        contributions = self.coefficients * (x - self.mu)
        
        explanation = {
            "base_risk_log": self.base_value,
            "total_risk_log": self.base_value + np.sum(contributions),
            "features": {}
        }
        
        for i, name in enumerate(self.feature_names):
            explanation["features"][name] = {
                "value": x[i],
                "contribution_log": contributions[i],
                "impact": "Increases Risk" if contributions[i] > 0 else "Decreases Risk"
            }
            
        return explanation
