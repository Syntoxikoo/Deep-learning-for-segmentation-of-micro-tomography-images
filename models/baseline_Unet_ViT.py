#baseline_Unet_ViT.py hybrid U-Net with transformer bottleneck

import torch
from torch import nn
import torch.nn.functional as F


class PrintSize(nn.Module): 
    """Utility module to print current shape of a Tensor in Sequential, only at the first pass."""

    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x




class ViTBottleneck(nn.Module):       # wrapper around PyTorch's nn.TransformerEncoder, operating on flattened (H*W) spatial tokens
    """
    ViT-style Transformer bottleneck used inside U_net_ViT.

    Operates on a feature map of shape (B, C, H, W) by treating each spatial
    location as a token of dimension C, applying a stack of TransformerEncoder
    layers, and reshaping back to the same 4D shape.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        d_model = channels
        dim_feedforward = 4 * d_model if mlp_dim is None else mlp_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, N, C)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape

        # Flatten spatial dims into a sequence of tokens: (B, H*W, C)
        tokens = x.view(b, c, h * w).permute(0, 2, 1)

        # Apply Transformer encoder
        tokens = self.encoder(tokens)

        # Reshape back to (B, C, H, W)
        x_out = tokens.permute(0, 2, 1).view(b, c, h, w)
        return x_out


class BottleneckViT(nn.Module):
    """
    Replacement for the original convolutional bottleneck.

    Includes ADAPTIVE downsampling before the ViT:
        - Repeatedly downsample until H*W <= max_tokens
        - Conv → BN/ReLU → Conv → BN/ReLU
        - Transformer (ViTBottleneck)
        - Upsample back to original bottleneck resolution

    This keeps memory under control even for large inputs like 1270x1350.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        normalize: bool = False,
        vit_num_layers: int = 2,
        vit_num_heads: int = 4,
        vit_mlp_dim: int = None,
        vit_dropout: float = 0.2,
        filter_size: int = 3,
        max_tokens: int = 2048,  # <--- safe upper bound for H*W seen by ViT
    ):
        super().__init__()

        self.max_tokens = max_tokens

        # Conv part (similar to original bottleneck)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, filter_size)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, filter_size)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity()
        self.act = nn.ReLU(True)

        # ViT bottleneck
        self.vit = ViTBottleneck(
            channels=bottleneck_channels,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            mlp_dim=vit_mlp_dim,
            dropout=vit_dropout,
        )

    def forward(self, x):
        # Remember original spatial size at the bottleneck
        b, c, h0, w0 = x.shape

        # 1) Adaptively downsample until token count is safe
        x_down = x
        down_factor = 1
        while (
            x_down.shape[2] * x_down.shape[3] > self.max_tokens
            and x_down.shape[2] >= 4
            and x_down.shape[3] >= 4
        ):
            x_down = F.max_pool2d(x_down, kernel_size=2, stride=2)
            down_factor *= 2

        # 2) Conv block at reduced resolution
        x_down = self.conv1(x_down)
        x_down = self.bn1(x_down)
        x_down = self.act(x_down)
        x_down = self.conv2(x_down)
        x_down = self.bn2(x_down)
        x_down = self.act(x_down)

        # 3) Transformer on reduced-resolution feature map
        x_down = self.vit(x_down)

        # 4) Upsample back to the original bottleneck spatial size
        x_up = F.interpolate(
            x_down,
            size=(h0, w0),  # explicitly restore original H,W
            mode="bilinear",
            align_corners=False,
        )

        # print("[DEBUG] Bottleneck output:",
        #     "min =", x_up.min().item(),
        #     "max =", x_up.max().item(),
        #     "mean =", x_up.mean().item(),
        #     "shape =", tuple(x_up.shape))

        return x_up


import torch
from torch import nn
import torch.nn.functional as F
from baseline_Unet import DoubleConv, Down, Up  # baseline components

class U_net_ViT(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,  # 1 channel for BCE+Dice
        features=(64, 128, 256, 512),
        bilinear=False,
        normalize=True,
        filter_size=3,
        dropout=0.2,
        max_tokens=2048,
    ):
        super().__init__()
        self.features = list(features)

        # === Baseline encoder path ===
        self.inc = DoubleConv(in_channels, features[0], filter_size, normalize, dropout)

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                Down(
                    features[i],
                    features[i + 1],
                    filter_size=filter_size,
                    normalize=normalize,
                    dropout=dropout,
                )
            )

        # === ViT Bottleneck ===
        self.bottleneck = BottleneckViT(
            in_channels=features[-1],
            bottleneck_channels=features[-1] * 2,
            normalize=normalize,
            vit_num_layers=2,
            vit_num_heads=4,
            vit_dropout=0.2,
            filter_size=filter_size,
            max_tokens=max_tokens,
        )

        # === Baseline decoder path ===
        self.ups = nn.ModuleList()
        prev_channels = features[-1] * 2
        for feat in reversed(features):
            self.ups.append(
                Up(
                    prev_channels,
                    feat,
                    filter_size=filter_size,
                    normalize=normalize,
                    upsampling=("Ctranspose" if not bilinear else "bilinear"),
                    dropout=dropout,
                )
            )
            prev_channels = feat

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

        # === Deep supervision heads (mirroring later decoder sizes) ===
        ds_in = (512, 256, 128, 64)
        self.deep_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in ds_in])

    def forward(self, x):
        # ---- Baseline encoder + skip collection ----
        skips = []

        x0 = self.inc(x)
        for down in self.downs:
            conv_out, x0 = down(x0)
            skips.append(conv_out)

        # ---- Bottleneck ----
        x = self.bottleneck(x0)

        # ---- Decode + deep supervision ----
        deepest = skips[-1] if skips else None
        skips = list(reversed(skips[:-1]))

        deep_preds = []
        for i, up in enumerate(self.ups):
            if i == 0:
                skip = deepest
            else:
                skip = skips[i - 1] if i - 1 < len(skips) else None
            x = up(x, skip)

            if i < len(self.deep_heads):
                dp = self.deep_heads[i](x)
                dp = F.interpolate(dp, size=x.shape[2:], mode="bilinear", align_corners=False)
                deep_preds.append(dp)

        logits = self.outc(x)
        return logits, deep_preds

if __name__ == "__main__": # dummy test
    model = U_net_ViT()
    x = torch.randn(1, 1, 572, 572)
    x_out = model(x)
    print("Out shape: ", x_out.shape)
