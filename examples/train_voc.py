"""Train FlashSeg on Pascal VOC."""

from flashseg import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/VOC2012/JPEGImages",
    train_masks="data/VOC2012/SegmentationClass",
    val_images="data/VOC2012/JPEGImages",
    val_masks="data/VOC2012/SegmentationClass",
    num_classes=21,
    input_size=512,
    epochs=100,
    batch_size=16,
    device="cuda",
    pretrained=True,
)

trainer.train()
