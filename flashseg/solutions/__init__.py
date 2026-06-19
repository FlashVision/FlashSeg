"""High-level segmentation solutions."""

from flashseg.solutions.scene_parser import SceneParser
from flashseg.solutions.lane_detector import LaneDetector
from flashseg.solutions.background_remover import BackgroundRemover
from flashseg.solutions.area_calculator import AreaCalculator

__all__ = ["SceneParser", "LaneDetector", "BackgroundRemover", "AreaCalculator"]
