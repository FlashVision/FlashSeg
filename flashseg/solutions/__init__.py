"""High-level segmentation solutions."""

from flashseg.solutions.area_calculator import AreaCalculator
from flashseg.solutions.background_remover import BackgroundRemover
from flashseg.solutions.lane_detector import LaneDetector
from flashseg.solutions.scene_parser import SceneParser

__all__ = ["SceneParser", "LaneDetector", "BackgroundRemover", "AreaCalculator"]
