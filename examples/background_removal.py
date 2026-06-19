"""Remove background from an image."""

import cv2

from flashseg import Predictor
from flashseg.solutions import BackgroundRemover

predictor = Predictor(model_path="workspace/best.pth", num_classes=21, device="cuda")
remover = BackgroundRemover(predictor, foreground_classes=[15])  # person class

image = cv2.imread("person.jpg")
result = remover.remove(image, background_color=(255, 255, 255))
cv2.imwrite("no_background.png", result)
print("Background removed!")
