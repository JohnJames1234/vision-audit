import cv2
import numpy as np

class GeometricAnalyzer:
    """Extracts structural primitives: Canny edges, Harris corners, and Hough lines."""

    @staticmethod
    def detect_edges(image_gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
        blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
        return cv2.Canny(blurred, low_thresh, high_thresh)

    @staticmethod
    def detect_corners(image_gray: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04) -> np.ndarray:
        gray_float = np.float32(image_gray)
        dst = cv2.cornerHarris(gray_float, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        corner_mask = np.zeros_like(image_gray, dtype=np.uint8)
        corner_mask[dst > 0.01 * dst.max()] = 255
        return corner_mask

    @staticmethod
    def detect_lines(edges: np.ndarray, min_line_length: int = 30, max_line_gap: int = 10) -> list:
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi / 180, 
            threshold=50, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        return lines if lines is not None else []

    def analyze(self, image_gray: np.ndarray) -> dict:
        edges = self.detect_edges(image_gray)
        corners = self.detect_corners(image_gray)
        lines = self.detect_lines(edges)
        return {
            "edges": edges,
            "corner_mask": corners,
            "lines": lines,
            "line_count": len(lines),
            "corner_count": int(np.sum(corners > 0))
        }