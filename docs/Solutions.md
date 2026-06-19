# Solutions

## Background Remover

```python
from flashseg.solutions import BackgroundRemover

remover = BackgroundRemover(predictor, foreground_classes=[15])
result = remover.remove(image, background_color=(255, 255, 255))
alpha = remover.get_alpha_matte(image)
```

## Lane Detector

```python
from flashseg.solutions import LaneDetector

detector = LaneDetector(predictor, lane_class_id=1)
lane_mask = detector.detect(image)
lanes = detector.get_lane_points(image)
```

## Scene Parser

```python
from flashseg.solutions import SceneParser

parser = SceneParser(predictor, class_names=["bg", "road", "building"])
stats = parser.parse(image)  # {"road": 35.2%, "building": 12.1%}
```

## Area Calculator

```python
from flashseg.solutions import AreaCalculator

calc = AreaCalculator(predictor, pixels_per_meter=10.0)
areas = calc.calculate(image)  # {class_id: area_m2}
```
