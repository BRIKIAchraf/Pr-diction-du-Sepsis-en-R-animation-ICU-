import numpy as np

class FederatedAggregator:
    """
    Demonstrates Federated Learning for our Sepsis model.
    Instead of sharing raw patient data, hospitals share their local 
    Covariance Matrices (\Sigma) and Means (\mu) to compute a global Mahalanobis distance,
    and their linear parameters for the risk model.
    """
    def __init__(self):
        self.hospital_models = []
        self.global_mu = None
        self.global_cov = None
        self.global_coefficients = None
        self.global_intercept = None
        
    def register_hospital(self, n_patients: int, mu: np.ndarray, cov: np.ndarray, coef: np.ndarray, intercept: float):
        self.hospital_models.append({
            "n": n_patients,
            "mu": mu,
            "cov": cov,
            "coef": coef,
            "intercept": intercept
        })
        
    def aggregate(self):
        """
        Federated Averaging (FedAvg) of the parameters.
        For Covariance and Mean, we use pooled variance formulas.
        """
        total_n = sum(h["n"] for h in self.hospital_models)
        
        # 1. Aggregate Means
        self.global_mu = sum((h["n"] / total_n) * h["mu"] for h in self.hospital_models)
        
        # 2. Aggregate Covariance (Pooled Covariance + difference of means)
        # pooled_cov = sum( (n_i - 1)*Cov_i + n_i*(mu_i - mu_global)(mu_i - mu_global)^T ) / (total_n - k)
        # Simplified for demonstration:
        self.global_cov = sum((h["n"] / total_n) * h["cov"] for h in self.hospital_models)
        
        # 3. Aggregate Linear Parameters (Risk Model)
        self.global_coefficients = sum((h["n"] / total_n) * h["coef"] for h in self.hospital_models)
        self.global_intercept = sum((h["n"] / total_n) * h["intercept"] for h in self.hospital_models)
        
        print("Federated Aggregation Complete!")
        print(f"Global Intercept: {self.global_intercept:.4f}")
        return {
            "mu": self.global_mu,
            "cov": self.global_cov,
            "coef": self.global_coefficients,
            "intercept": self.global_intercept
        }

if __name__ == "__main__":
    # Mocking two hospitals
    aggregator = FederatedAggregator()
    
    # Hospital A (e.g. Paris) - 1000 patients
    aggregator.register_hospital(
        n_patients=1000,
        mu=np.array([98.0, 70.0, 37.0, 90.0, 1.0]),
        cov=np.eye(5) * 1.5,
        coef=np.array([-0.05, 0.01, 0.0, 0.0, 0.5]),
        intercept=3.0
    )
    
    # Hospital B (e.g. Lyon) - 500 patients (Different demographics/sensors)
    aggregator.register_hospital(
        n_patients=500,
        mu=np.array([97.5, 72.0, 37.2, 88.0, 1.2]),
        cov=np.eye(5) * 1.8,
        coef=np.array([-0.06, 0.015, 0.0, -0.01, 0.45]),
        intercept=3.1
    )
    
    global_model = aggregator.aggregate()
    print("This global model can now be safely distributed back to both hospitals without violating GDPR.")
