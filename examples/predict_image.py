"""Run segmentation on an image."""

import cv2

from flashseg import Predictor

predictor = Predictor(
    model_path="workspace/best.pth",
    model_size="m",
    num_classes=21,
    input_size=512,
    device="cuda",
)

image = cv2.imread("test.jpg")
mask = predictor.predict(image)

overlay = predictor.visualize(image, mask, alpha=0.5)
cv2.imwrite("result.png", overlay)
print(f"Segmentation saved to result.png (unique classes: {len(set(mask.flatten()))})")
