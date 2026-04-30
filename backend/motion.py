"""
motion.py — Motion detection via frame differencing.

Computes a normalised motion_score (0.0–1.0) by comparing
consecutive grayscale frames.
"""

import cv2
import numpy as np


class MotionDetector:
    """Simple frame-differencing motion detector."""

    def __init__(self, blur_kernel: tuple = (21, 21), normalise_factor: float = 30.0):
        """
        Parameters
        ----------
        blur_kernel : tuple
            Gaussian blur kernel size applied before differencing.
        normalise_factor : float
            The mean pixel difference is divided by this to produce
            a 0-1 score.  Increase to reduce sensitivity.
        """
        self.blur_kernel = blur_kernel
        self.normalise_factor = normalise_factor
        self._prev_gray: np.ndarray | None = None

    def set_sensitivity(self, sensitivity: float):
        """
        Adjust motion detection sensitivity.

        Parameters
        ----------
        sensitivity : float
            Value in [0.0, 1.0].
            0.0 = least sensitive (normalise_factor=60, needs large motion)
            1.0 = most sensitive  (normalise_factor=5, reacts to tiny motion)
        """
        # Map sensitivity 0→1 to normalise_factor 60→5
        sensitivity = max(0.0, min(1.0, sensitivity))
        self.normalise_factor = 60.0 - (55.0 * sensitivity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, frame: np.ndarray) -> float:
        """
        Compute motion score for the given BGR frame.

        Returns
        -------
        float
            Motion score in [0.0, 1.0].
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        motion_score = 0.0

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self._prev_gray, gray)
            motion_score = float(np.mean(diff)) / self.normalise_factor
            motion_score = min(motion_score, 1.0)

        self._prev_gray = gray
        return motion_score

    def reset(self):
        """Reset internal state (e.g. when switching video sources)."""
        self._prev_gray = None
