import numpy as np
import pandas as pd
from skimage import color


class ColorDetector:
    def __init__(self):
        self.color_system = pd.read_excel('asset/file/iscc-nbs-colour-system.xlsx')
        self.color_system = self.color_system.dropna(subset=['r', 'g', 'b'])
        self.color_system.reset_index(drop=True, inplace=True)

        self.iscc_color = self.color_system[["color"]].values
        self.iscc_category = self.color_system[["category"]].values
        self.iscc_rgb = self.color_system[['r', 'g', 'b']]
        self.iscc_rgb = self.iscc_rgb.to_numpy()
        self.iscc_rgb = self.iscc_rgb / 255
        self.iscc_lab = color.rgb2lab(self.iscc_rgb)

    @staticmethod
    def manhattan_distance(a, b):
        """Calculates the Manhattan distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            float: The Manhattan distance between the vectors.

        Raises:
            ValueError: If the input vectors have different shapes.
        """

        if a.shape != b.shape:
            raise ValueError("Input vectors must have the same shape.")
        return np.sum(np.abs(a - b))

    @staticmethod
    def chebyshev_distance(a, b):
        """Calculates the Chebyshev distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            float: The Chebyshev distance between the vectors.

        Raises:
            ValueError: If the input vectors have different shapes.
        """

        if a.shape != b.shape:
            raise ValueError("Input vectors must have the same shape.")
        return np.max(np.abs(a - b))

    @staticmethod
    def minkowski_distance(a, b, p):
        """Calculates the Minkowski distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.
            p (float): The power parameter (must be greater than 0).

        Returns:
            float: The Minkowski distance between the vectors.

        Raises:
            ValueError: If the input vectors have different shapes or p is not a positive value.
        """

        if a.shape != b.shape:
            raise ValueError("Input vectors must have the same shape.")
        if p <= 0:
            raise ValueError("p must be a positive value.")
        return np.sum(np.abs(a - b) ** p) ** (1 / p)

    @staticmethod
    def euclidean_distance(a, b, axis=None):
        """Calculates the Euclidean distance between vectors or the corresponding rows or columns in matrices.

        Args:
            a (np.ndarray): The first input array.
            b (np.ndarray): The second input array, with the same shape as `a` or compatible broadcasting.
            axis (int, optional): The axis along which to compute the distance.
                - None (default): Computes the distance between vectors or entire matrices based on their shapes.
                - 0: Computes the distance between rows.
                - 1: Computes the distance between columns.

        Returns:
            np.ndarray: An array containing the Euclidean distances.

        Raises:
            ValueError: If the input arrays have incompatible shapes for broadcasting except for the specified axis.
        """

        if len(a.shape) != len(b.shape) and any(d != 1 for d in np.subtract(a.shape, b.shape) if d is not None):
            raise ValueError("Input arrays must have compatible shapes, except for the specified axis.")

        return np.linalg.norm(a - b, axis=axis)

    def predict(self, rgb):
        rgb = np.array(rgb)
        rgb = rgb / 255
        lab = color.rgb2lab(rgb)

        # Find the closest color in LUT1
        t = []
        for i in range(len(self.iscc_lab)):
            distances = self.euclidean_distance(self.iscc_lab[i], lab)
            t.append(distances)
        distances = np.array(t)
        closest_color_index = np.argmin(distances)

        # Get the dominant hue from LUT1
        dominant_hue_color = str(self.iscc_color[closest_color_index][0])
        dominant_hue_category = str(self.iscc_category[closest_color_index][0])

        return dominant_hue_color, dominant_hue_category