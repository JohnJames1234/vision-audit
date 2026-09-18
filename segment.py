import cv2
import numpy as np
from sklearn.cluster import KMeans

class Segmenter:
    """Segments foreground objects using K-Means and morphological refinement."""

    @staticmethod
    def cluster_pixels(image_bgr: np.ndarray, k: int = 2) -> np.ndarray:
        pixels = image_bgr.reshape((-1, 3)).astype(np.float32)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels)
        segmented = labels.reshape(image_bgr.shape[:2]).astype(np.uint8)
        return segmented

    @staticmethod
    def clean_mask(binary_mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def extract_bounding_boxes(self, image_bgr: np.ndarray, min_area: int = 400) -> list:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cleaned = self.clean_mask(thresh)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                crop = image_bgr[y:y+h, x:x+w]
                bounding_boxes.append({
                    "box": (x, y, w, h),
                    "crop": crop,
                    "area": cv2.contourArea(cnt)
                })
        return bounding_boxes