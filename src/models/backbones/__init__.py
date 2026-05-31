from src.models.backbones.mobilenet import MobileNetV3Encoder
from src.models.backbones.resnet import ResNetEncoder


def build_backbone(name: str, in_channels: int = 3, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        return ResNetEncoder(in_channels=in_channels, pretrained=pretrained)
    if name == "mobilenet_v3_small":
        return MobileNetV3Encoder(in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {name}")
