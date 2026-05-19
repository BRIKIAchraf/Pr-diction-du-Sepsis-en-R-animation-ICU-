import numpy as np
from scipy import stats
from typing import Tuple

class ICUMathEngine:
    """
    Core mathematical engine for Sepsis Prediction.
    Implements QR Householder, Cholesky, and Mahalanobis from scratch/low-level
    to demonstrate numerical stability and precision.
    """

    @staticmethod
    def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the QR decomposition of matrix A using Householder reflections.
        A = QR where Q is orthogonal and R is upper triangular.
        Provides better numerical stability (condition number) than normal equations.
        """
        m, n = A.shape
        R = A.copy().astype(float)
        Q = np.eye(m)

        for i in range(min(m, n)):
            x = R[i:, i]
            e1 = np.zeros_like(x)
            e1[0] = 1.0

            # Compute Householder vector v = x +/- ||x|| * e1
            # We choose sign to avoid catastrophic cancellation
            norm_x = np.linalg.norm(x)
            if norm_x == 0:
                continue

            sign = -1.0 if x[0] < 0 else 1.0
            u = x + sign * norm_x * e1
            
            norm_u = np.linalg.norm(u)
            if norm_u == 0:
                continue
                
            v = u / norm_u

            # H = I - 2 * v * v.T
            # Apply H to R: R = H * R
            R[i:, :] = R[i:, :] - 2.0 * np.outer(v, np.dot(v.T, R[i:, :]))
            # Apply H to Q: Q = Q * H.T (since H is symmetric, H.T = H)
            Q[:, i:] = Q[:, i:] - 2.0 * np.outer(np.dot(Q[:, i:], v), v)

        return Q, R

    @staticmethod
    def solve_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solves the overdetermined system Ax = b using QR Householder decomposition.
        Minimizes ||Ax - b||_2.
        """
        Q, R = ICUMathEngine.householder_qr(A)
        # Rx = Q.T * b
        y = np.dot(Q.T, b)
        
        n = R.shape[1]
        x = np.zeros(n)
        
        # Back substitution for upper triangular matrix R
        for i in range(n - 1, -1, -1):
            if R[i, i] == 0:
                x[i] = 0
            else:
                x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
        return x

    @staticmethod
    def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
        """
        Computes the Cholesky decomposition of a Symmetric Positive Definite matrix A.
        A = L L^T. Returns the lower triangular matrix L.
        Useful for inverting covariance matrices safely.
        """
        n = A.shape[0]
        L = np.zeros_like(A, dtype=float)

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i, k] * L[j, k] for k in range(j))
                if i == j:
                    val = A[i, i] - s
                    L[i, j] = np.sqrt(max(val, 0.0)) # Max to avoid floating point neg
                else:
                    if L[j, j] > 0:
                        L[i, j] = (A[i, j] - s) / L[j, j]
                    else:
                        L[i, j] = 0.0
        return L

    @staticmethod
    def invert_cholesky(L: np.ndarray) -> np.ndarray:
        """
        Inverts a matrix A given its Cholesky decomposition L.
        A^-1 = (L^-1)^T L^-1
        """
        n = L.shape[0]
        # Invert lower triangular matrix L
        L_inv = np.zeros_like(L, dtype=float)
        for i in range(n):
            L_inv[i, i] = 1.0 / (L[i, i] + 1e-12)
            for j in range(i):
                s = sum(L[i, k] * L_inv[k, j] for k in range(j, i))
                L_inv[i, j] = -s / (L[i, i] + 1e-12)
                
        return np.dot(L_inv.T, L_inv)

    @staticmethod
    def mahalanobis_distance(x: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Computes the squared Mahalanobis distance d^2(x).
        d^2(x) = (x - mu)^T * Sigma^-1 * (x - mu)
        Utilizes Cholesky for safe inversion.
        """
        delta = x - mu
        # Use Cholesky to invert covariance matrix
        L = ICUMathEngine.cholesky_decomposition(cov_matrix)
        cov_inv = ICUMathEngine.invert_cholesky(L)
        
        return float(np.dot(np.dot(delta.T, cov_inv), delta))

    @staticmethod
    def is_anomaly(d_squared: float, df: int, p_value: float = 0.99) -> bool:
        """
        Checks if the computed distance is an anomaly.
        d^2 > chi2(df, p)
        """
        threshold = stats.chi2.ppf(p_value, df)
        return d_squared > threshold
