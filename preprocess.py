import cv2
import numpy as np

class ImagePreprocessor:
    """Handles color conversions, noise reduction, and contrast enhancement."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return img

    @staticmethod
    def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def split_channels(image_bgr: np.ndarray) -> tuple:
        b, g, r = cv2.split(image_bgr)
        return b, g, r

    @staticmethod
    def remove_noise(image_gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        # Median filter to eliminate salt-and-pepper noise while preserving edges
        return cv2.medianBlur(image_gray, kernel_size)

    @staticmethod
    def equalize_contrast(image_gray: np.ndarray) -> np.ndarray:
        # Global histogram equalization to expand dynamic range
        return cv2.equalizeHist(image_gray)

    def process(self, image_path: str) -> dict:
        bgr = self.load_image(image_path)
        gray = self.to_grayscale(bgr)
        denoised = self.remove_noise(gray)
        enhanced = self.equalize_contrast(denoised)
        return {
            "original_bgr": bgr,
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced
        }