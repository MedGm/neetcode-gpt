import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

        z = x @ w.T + b
        if activation == "sigmoid":
            e_nz = np.exp(-z)
            e_z = np.exp(z)
            return np.round(np.where(z>=0, 1/ (1 + e_nz), e_z / (1 + e_z)), 5)
        else:
            return np.round(max(0.0,z), 5)

        return 0.0
