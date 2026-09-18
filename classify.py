import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class PatchClassifier:
    """Reduces spatial feature vectors via PCA and classifies patches with KNN."""

    def __init__(self, n_components: int = 8, n_neighbors: int = 3):
        self.target_size = (32, 32)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False

    def _extract_vector(self, patch: np.ndarray) -> np.ndarray:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        resized = cv2.resize(gray_patch, self.target_size, interpolation=cv2.INTER_AREA)
        return resized.flatten() / 255.0

    def fit_synthetic_baseline(self):
        """Creates a baseline dataset of synthetic primitives (circles, bars, blocks)."""
        x_train, y_train = [], []
        labels = ["circular_component", "rectangular_bar", "square_block"]

        for _ in range(15):
            # Circle
            img_c = np.zeros(self.target_size, dtype=np.uint8)
            cv2.circle(img_c, (16, 16), 10, 255, -1)
            x_train.append(img_c.flatten() / 255.0)
            y_train.append("circular_component")

            # Horizontal bar
            img_b = np.zeros(self.target_size, dtype=np.uint8)
            cv2.rectangle(img_b, (4, 12), (28, 20), 255, -1)
            x_train.append(img_b.flatten() / 255.0)
            y_train.append("rectangular_bar")

            # Square block
            img_s = np.zeros(self.target_size, dtype=np.uint8)
            cv2.rectangle(img_s, (8, 8), (24, 24), 255, -1)
            x_train.append(img_s.flatten() / 255.0)
            y_train.append("square_block")

        x_train = np.array(x_train)
        x_pca = self.pca.fit_transform(x_train)
        self.knn.fit(x_pca, y_train)
        self.is_trained = True

    def predict(self, patch: np.ndarray) -> str:
        if not self.is_trained:
            self.fit_synthetic_baseline()
        vec = self._extract_vector(patch).reshape(1, -1)
        vec_pca = self.pca.transform(vec)
        return self.knn.predict(vec_pca)[0]