"""Export FlashSeg to ONNX."""

from flashseg import Exporter

exporter = Exporter(
    model_path="workspace/best.pth",
    model_size="m",
    num_classes=21,
    input_size=512,
)

exporter.export(output="flashseg.onnx", simplify=True)
print("Exported to flashseg.onnx")
