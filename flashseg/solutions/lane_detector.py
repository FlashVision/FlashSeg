"""Lane detection using segmentation."""

from typing import List, Tuple

import cv2
import numpy as np

from flashseg.engine.predictor import Predictor


class LaneDetector:
    """Detect lane markings using semantic segmentation."""

    def __init__(self, predictor: Predictor, lane_class_id: int = 1):
        self.predictor = predictor
        self.lane_class_id = lane_class_id

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect lanes and return binary lane mask."""
        mask = self.predictor.predict(image)
        lane_mask = (mask == self.lane_class_id).astype(np.uint8) * 255
        return lane_mask

    def get_lane_points(self, image: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Extract lane line polypoints from segmentation."""
        lane_mask = self.detect(image)

        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lanes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                lane_points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                lanes.append(lane_points)

        return lanes

    def visualize(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Overlay detected lanes on image."""
        lane_mask = self.detect(image)
        overlay = image.copy()
        overlay[lane_mask > 0] = color
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
