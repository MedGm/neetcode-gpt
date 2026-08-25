import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        z_clipped = np.clip(z,-30,30)
        
        s = np.where(z>=0, 1 / (1 + np.exp(-z_clipped)), np.exp(z_clipped) / (1 + np.exp(z_clipped)))
        return np.round(s, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        relu = []
        for z_elements in z:
            if z_elements < 0:
                relu.append(0.0)
            else:
                relu.append(z_elements)
        return relu
