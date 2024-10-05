import torch.nn as nn
import torchvision
import timm
from torchvision.models import mobilenet_v2, resnet50, MobileNet_V2_Weights, ResNet50_Weights


class Backbone(nn.Module):
    def __init__(self, backbone="xception"):
        super().__init__()

        if backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.low_level_channels = 256
            self.high_level_channels = 2048

        elif backbone == "mobilenetv2":
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.low_level_channels = 24
            self.high_level_channels = 1280

        elif backbone == "xception":
            self.backbone = timm.create_model("xception65", pretrained=True, features_only=True)
            self.low_level_channels = self.backbone.feature_info.channels()[1]
            self.high_level_channels = self.backbone.feature_info.channels()[-1]

        else:
            raise ValueError("Unsupported backbone: Choose either 'xception', 'resnet50' or 'mobilenetv2'")

    def forward(self, x):
        if isinstance(self.backbone, torchvision.models.ResNet):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            # Low level features
            low_level_features = self.backbone.layer1(x)

            # High level features
            x = self.backbone.layer2(low_level_features)
            x = self.backbone.layer3(x)
            high_level_features = self.backbone.layer4(x)

        elif isinstance(self.backbone, torchvision.models.MobileNetV2):
            features = self.backbone.features
            low_level_features = features[:4](x) # 4th inverted residual block
            high_level_features = features[4:](low_level_features)  # 17th inverted residual block

        else:
            # Xception
            features = self.backbone(x)
            low_level_features = features[1]
            high_level_features = features[-1]
            
        return low_level_features, high_level_features
