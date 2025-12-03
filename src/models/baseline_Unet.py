import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv => BN? => ReLU) * 2 with same padding"""

    def __init__(
        self, in_ch, out_ch, filter_size=3, normalize=True, dropout=0.0, **kwargs
    ):
        super().__init__()
        pad = kwargs.get("padding", filter_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch, out_ch, kernel_size=filter_size, padding=pad, bias=not normalize
            ),
        ]
        if normalize:
            if kwargs.get("batch_norm", True):
                layers.append(nn.BatchNorm2d(out_ch))
            else:
                layers.append(
                    nn.GroupNorm(num_groups=32, num_channels=out_ch)
                )  # TODO : 32 might not be opti
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                out_ch, out_ch, kernel_size=filter_size, padding=pad, bias=not normalize
            )
        )
        if normalize:
            if kwargs.get("batch_norm", True):
                layers.append(nn.BatchNorm2d(out_ch))
            else:
                layers.append(nn.GroupNorm(num_groups=32, num_channels=out_ch))
        layers.append(nn.ReLU(inplace=True))

        if dropout and dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """DoubleConv then MaxPool (return conv output for skip, and pooled for deeper)."""

    def __init__(
        self, in_ch, out_ch, filter_size=3, normalize=True, dropout=0.0, **kwargs
    ):
        super().__init__()
        self.conv = DoubleConv(
            in_ch,
            out_ch,
            filter_size=filter_size,
            normalize=normalize,
            dropout=dropout,
            **kwargs,
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        pooled = self.pool(out)
        return out, pooled


class Up(nn.Module):
    """Upscaling then double conv. Supports transposed conv or bilinear upsampling.
    Robust center-cropping: crop whichever tensor is larger so the two match
    before concatenation.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        filter_size=3,
        normalize=True,
        upsampling="Ctranspose",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.upsampling = upsampling
        self.filter_size = filter_size

        up_out_ch = in_ch // 2 if in_ch // 2 >= 1 else in_ch
        if upsampling == "Ctranspose":
            self.up = nn.ConvTranspose2d(in_ch, up_out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode=upsampling,
                    align_corners=(upsampling == "bilinear"),
                ),
                nn.Conv2d(in_ch, up_out_ch, kernel_size=1),
            )

        self.conv = DoubleConv(
            up_out_ch + out_ch,
            out_ch,
            filter_size=filter_size,
            normalize=normalize,
            dropout=dropout,
            **kwargs,
        )

    def _center_crop(self, tensor, target_h, target_w):
        """Center-crop tensor to (target_h, target_w)."""
        _, _, h, w = tensor.shape
        if h == target_h and w == target_w:
            return tensor
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return tensor[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

    def forward(self, x, skip):
        x = self.up(x)

        if skip is None:
            return self.conv(x)

        # Make shapes equal by cropping the larger tensor (robust to both possibilities)
        x_h, x_w = x.shape[2], x.shape[3]
        s_h, s_w = skip.shape[2], skip.shape[3]
        target_h = min(x_h, s_h)
        target_w = min(x_w, s_w)

        if s_h != target_h or s_w != target_w:
            skip = self._center_crop(skip, target_h, target_w)
        if x_h != target_h or x_w != target_w:
            x = self._center_crop(x, target_h, target_w)

        # Now concat and conv
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Flexible U-Net.

    Args:
        in_channels: input image channels (e.g. 1 for grayscale)
        num_classes: output channels (for CrossEntropyLoss use number of classes, e.g. 2)
        features: list of feature sizes at each level, e.g. [64,128,256,512]
        up_method: upsampling method - "Ctranspose" for ConvTranspose2d or "bilinear" for bilinear interpolation
        normalize: whether to use BatchNorm
        filter_size: kernel size for convs (3 recommended)
        dropout: dropout rate for regularization
        padding: padding size for convolutions (default: filter_size//2)
        batchsize: if <=8 -> use GroupNorm instead of BatchNorm
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        features=(64, 128, 256, 512),
        up_method="Ctranspose",
        normalize=True,
        filter_size=3,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = list(features)
        self.up_method = up_method
        self.normalize = normalize
        self.filter_size = filter_size
        batch_norm = kwargs.get("batchsize", 6) >= 6
        kwargs["batch_norm"] = batch_norm
        # Encoder path
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

        # Bottleneck
        bottleneck_channels = self.features[-1] * 2
        self.bottleneck = DoubleConv(
            self.features[-1],
            bottleneck_channels,
            filter_size=filter_size,
            normalize=normalize,
            dropout=dropout,
            **kwargs,
        )

        # Decoder path: build ups mirroring encoder (features reversed)
        decoder_features = list(reversed(self.features))
        self.ups = nn.ModuleList()
        prev_channels = bottleneck_channels
        for feat in decoder_features:
            # prev_channels -> up block expecting in_ch=prev_channels, out_ch=feat
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
            prev_channels = feat  # after Up, output channels are 'feat'

        # Final 1x1 conv to get desired classes
        self.outc = nn.Conv2d(self.features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward with storage of skip connections (store pre-pool conv outputs)
        skips = []
        x0 = self.inc(x)  # full-resolution conv
        skips.append(x0)
        x_cur = x0
        # For each Down we get (conv_out, pooled) and we append conv_out (skip) and set x_cur = pooled
        for down in self.downs:
            conv_out, pooled = down(x_cur)
            skips.append(conv_out)
            x_cur = pooled

        # deepest skip is the last conv_out (before the deepest pooling)
        deepest = (
            skips.pop()
        )  # this corresponds to the deepest encoder feature (pre-last-pool)
        # x_cur is the pooled tensor fed into bottleneck
        x = self.bottleneck(x_cur)

        # Now decode using reversed skips
        skips = skips[::-1]  # shallower skips in reverse order

        for i, up in enumerate(self.ups):
            if i == 0:
                skip = deepest
            else:
                skip = skips[i - 1] if i - 1 < len(skips) else None
            x = up(x, skip)

        logits = self.outc(x)
        return logits


def init_weights(m):
    """Initialize Conv and BatchNorm weights sensibly."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


# quick smoke test when running file directly
if __name__ == "__main__":
    model = UNet(
        in_channels=1,
        num_classes=2,
        features=(64, 128, 256, 512),
        up_method="Ctranspose",
        normalize=True,
    )
    model.apply(init_weights)
    x = torch.randn(1, 1, 388, 388)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # should be [1, num_classes, 388, 388]
