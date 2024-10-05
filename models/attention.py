import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        # Pushes output to 0 or 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply GAP and GMP and resize to become 2D
        gap = self.global_avg_pool(x).view(x.size(0), -1)
        gmp = self.global_max_pool(x).view(x.size(0), -1)

        # Feed through shared MLP
        mlp_gap = self.mlp(gap)
        mlp_gmp = self.mlp(gmp)

        # Combine and get attention weights
        combined = mlp_gap + mlp_gmp
        attention_weights = self.sigmoid(combined).view(x.size(0), x.size(1), 1, 1)

        # Element-wise multiplication with input to get final feature map
        return attention_weights * x


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # 7x7 convolution and sigmoid activation
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply GAP and GMP across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along the channel dimension
        combined = torch.cat([avg_pool, max_pool], dim=1)

        # Get spatial attention weights
        attention_map = self.conv(combined)
        attention_weights = self.sigmoid(attention_map)

        return attention_weights * x
