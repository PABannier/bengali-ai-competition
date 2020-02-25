import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34,
    'se_resnext_50_32x4d': models.SE_ResNeXt50_32x4d,
    'efficientnet-b3': models.EfficientNetB3
}