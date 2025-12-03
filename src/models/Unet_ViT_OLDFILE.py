import torch
import torch.nn.functional as F
from torch import nn

from ..utils import init_weights
from .baseline_Unet import DoubleConv, Down, Up


class ViTBottleneck(nn.Module):
    """Vision Transformer bottleneck using standard PyTorch TransformerEncoder.

    Converts spatial features to tokens, processes with transformer attention,
    and converts back to spatial representation.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int | None = None,
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
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).permute(0, 2, 1)
        tokens = self.encoder(tokens)
        x_out = tokens.permute(0, 2, 1).view(b, c, h, w)
        return x_out


class BottleneckViT(nn.Module):
    """Bottleneck block combining convolutions with Vision Transformer.

    Features:
    - Preserves spatial dimensions via padding
    - Optionally downsamples if input exceeds max_tokens for efficiency
    - Applies transformer attention for global context
    - Upsamples back to original resolution
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        normalize: bool = False,
        vit_num_layers: int = 2,
        vit_num_heads: int = 4,
        vit_mlp_dim: int | None = None,
        vit_dropout: float = 0.2,
        filter_size: int = 3,
        max_tokens: int = 2048,
    ):
        super().__init__()

        self.max_tokens = max_tokens

        self.conv = DoubleConv(
            in_channels,
            bottleneck_channels,
            filter_size=filter_size,
            normalize=normalize,
        )

        # Vision Transformer for global context
        self.vit = ViTBottleneck(
            channels=bottleneck_channels,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            mlp_dim=vit_mlp_dim,
            dropout=vit_dropout,
        )

    def forward(self, x):
        b, c, h0, w0 = x.shape

        x_down = x
        down_factor = 1
        while (
            x_down.shape[2] * x_down.shape[3] > self.max_tokens
            and x_down.shape[2] >= 4
            and x_down.shape[3] >= 4
        ):
            x_down = F.max_pool2d(x_down, kernel_size=2, stride=2)
            down_factor *= 2

        x_down = self.conv(x_down)

        x_down = self.vit(x_down)

        x_up = F.interpolate(
            x_down,
            size=(h0, w0),
            mode="bilinear",
            align_corners=False,
        )

        return x_up


class UNetViT(nn.Module):
    """U-Net with Vision Transformer Bottleneck.

    Architecture:
    - Encoder: series of DoubleConv + MaxPool blocks
    - Bottleneck: BottleneckViT with transformer attention
    - Decoder: series of Up blocks with skip connections

    Args:
        in_channels: number of input channels (default: 1 for grayscale)
        num_classes: number of output classes (default: 2 for binary segmentation)
        features: tuple of feature dimensions at each level (default: (64, 128, 256, 512))
        up_method: upsampling method - "Ctranspose" for ConvTranspose2d or "bilinear" for bilinear interpolation
        normalize: if True, use BatchNorm; else use GroupNorm for small batch sizes
        filter_size: kernel size for convolutions (default: 3)
        dropout: dropout rate for regularization (default: 0.0)
        vit_num_layers: number of transformer encoder layers (default: 2)
        vit_num_heads: number of attention heads (default: 4)
        vit_mlp_dim: MLP hidden dimension for transformer (default: 4*channels)
        vit_dropout: dropout rate for transformer (default: 0.2)
        max_tokens: maximum number of tokens for ViT (default: 2048)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        features: tuple = (64, 128, 256, 512),
        up_method: str = "Ctranspose",
        normalize: bool = True,
        filter_size: int = 3,
        dropout: float = 0.0,
        vit_num_layers: int = 2,
        vit_num_heads: int = 4,
        vit_mlp_dim: int | None = None,
        vit_dropout: float = 0.2,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = list(features)
        self.up_method = up_method
        self.normalize = normalize
        self.filter_size = filter_size

        batch_norm = kwargs.get("batchsize", 8) >= 8
        kwargs["batch_norm"] = batch_norm

        # ============ ENCODER ============
        self.inc = DoubleConv(
            in_channels,
            self.features[0],
            filter_size=filter_size,
            normalize=normalize,
            **kwargs,
        )
        self.downs = nn.ModuleList()
        for i in range(len(self.features) - 1):
            self.downs.append(
                Down(
                    self.features[i],
                    self.features[i + 1],
                    filter_size=filter_size,
                    normalize=normalize,
                    dropout=dropout,
                    **kwargs,
                )
            )

        # ============ BOTTLENECK ============
        bottleneck_channels = self.features[-1] * 2
        self.bottleneck = BottleneckViT(
            in_channels=self.features[-1],
            bottleneck_channels=bottleneck_channels,
            normalize=normalize,
            vit_num_layers=vit_num_layers,
            vit_num_heads=vit_num_heads,
            vit_mlp_dim=vit_mlp_dim,
            vit_dropout=vit_dropout,
            filter_size=filter_size,
            max_tokens=max_tokens,
        )

        # ============ DECODER ============
        self.ups = nn.ModuleList()
        prev_channels = bottleneck_channels
        for feat in reversed(self.features):
            self.ups.append(
                Up(
                    prev_channels,
                    feat,
                    filter_size=filter_size,
                    normalize=normalize,
                    upsampling=up_method,
                    dropout=dropout,
                    **kwargs,
                )
            )
            prev_channels = feat

        # ============ OUTPUT ============
        self.outc = nn.Conv2d(self.features[0], num_classes, kernel_size=1)

        # ============ DEEP SUPERVISION ============
        # Auxiliary prediction heads for intermediate decoder outputs
        # Match decoder outputs: (512, 256, 128, 64)
        ds_in = tuple(reversed(self.features))
        self.deep_heads = nn.ModuleList([nn.Conv2d(ch, num_classes, 1) for ch in ds_in])

    def forward(self, x):
        """Forward pass through encoder, bottleneck, and decoder.

        Args:
            x: (B, C_in, H, W) input image

        Returns:
            logits: (B, num_classes, H, W) segmentation logits
            deep_preds: List of (B, num_classes, H, W) predictions from intermediate decoders
        """
        # ============ ENCODER ============
        skips = []

        x0 = self.inc(x)
        skips.append(x0)
        x_cur = x0

        # Downsampling path: store skip connections
        for down in self.downs:
            conv_out, pooled = down(x_cur)
            skips.append(conv_out)
            x_cur = pooled

        # ============ BOTTLENECK ============
        deepest = skips.pop()  # Last encoder feature before bottleneck
        x = self.bottleneck(x_cur)

        # ============ DECODER ============
        skips = skips[::-1]

        deep_preds = []
        for i, up in enumerate(self.ups):
            if i == 0:
                skip = deepest
            else:
                skip = skips[i - 1] if i - 1 < len(skips) else None
            x = up(x, skip)

            # Collect deep supervision predictions
            if i < len(self.deep_heads):
                dp = self.deep_heads[i](x)
                dp = F.interpolate(
                    dp, size=x.shape[2:], mode="bilinear", align_corners=False
                )
                deep_preds.append(dp)

        # ============ OUTPUT ============
        logits = self.outc(x)
        return logits, deep_preds


# ============ SMOKE TEST ============
if __name__ == "__main__":
    model = UNetViT(
        in_channels=1,
        num_classes=2,
        features=(64, 128, 256, 512),
        up_method="Ctranspose",
        normalize=True,
        vit_num_layers=2,
        vit_num_heads=4,
    )
    model.apply(init_weights)

    x = torch.randn(1, 1, 388, 388)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Expected shape: torch.Size([1, 2, 388, 388])")
