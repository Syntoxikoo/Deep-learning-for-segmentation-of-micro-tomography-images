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


class Convblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int = 3,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        stride: int = kwargs.get("stride", 1)
        dilation: int = kwargs.get("dilation", 1)
        normalize: bool = kwargs.get("normalize", False)

        # Preserve resolution with padding=1 for 3×3 kernels
        padding = 1 if filter_size == 3 else 0

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, filter_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bNorm1 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, filter_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bNorm2 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()

        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bNorm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bNorm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class EncodeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual_channels: int,
        pool_size: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.conv_block = Convblock(in_channels, out_channels, *args, **kwargs)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.resample = None
        normalize: bool = kwargs.get("normalize", False)

        if residual_channels != out_channels:
            self.resample = nn.Sequential(
                nn.Conv2d(out_channels, residual_channels, 1),
                nn.BatchNorm2d(residual_channels) if normalize else nn.Identity(),
            )

    def forward(self, x):
        out = self.conv_block(x)
        pooled = self.pool(out)
        if self.resample is not None:
            residual = self.resample(out)
        else:
            residual = out.clone()
        return pooled, residual


class DecodeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_size: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.residual_meth = kwargs.get("residual", "interpolate")
        upsampling = kwargs.get("upsampling", "nearest")

        self.conv_block = Convblock(in_channels, out_channels, *args, **kwargs)

        if upsampling == "Ctranspose":
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, up_size, stride=up_size
            )
        else:
            self.up = nn.Upsample(scale_factor=up_size, mode=upsampling)
        self.p1 = PrintSize()
        self.p2 = PrintSize()
        self.p3 = PrintSize()
        self.resample = None
        if up_size > 1:
            features = int(in_channels / up_size)
            self.resample = nn.Sequential(
                nn.Conv2d(in_channels, features, 1),
                nn.BatchNorm2d(features),
            )

    def forward(self, x, residual=None):
        upsampled = self.up(x)
        self.p1(upsampled)
        if self.resample is not None:
            upsampled = self.resample(upsampled)

        self.p2(upsampled)
        if residual is not None:
            if self.residual_meth.lower() == "interpolate":
                residual = F.interpolate(
                    residual, upsampled.shape[2:], mode="bilinear", align_corners=False
                )
            else:
                if residual.shape[2:] != upsampled.shape[2:]:
                    diff_h = residual.shape[2] - upsampled.shape[2]
                    diff_w = residual.shape[3] - upsampled.shape[3]
                    crop_h = diff_h // 2
                    crop_w = diff_w // 2
                    residual = residual[
                        :,
                        :,
                        crop_h : crop_h + upsampled.shape[2],
                        crop_w : crop_w + upsampled.shape[3],
                    ]
            concat = torch.cat((residual, upsampled), dim=1)
        else:
            concat = upsampled
        self.p3(concat)
        out = self.conv_block(concat)
        return out


class BottleneckConv(nn.Module):
    """
    Plain convolutional bottleneck (no ViT). Mimics a classic U-Net bottleneck:
    Conv -> BN/ReLU -> Conv -> BN/ReLU, preserving spatial size.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        filter_size: int = 3,
        normalize: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        padding = 1 if filter_size == 3 else 0
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=filter_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity()
        self.act = nn.ReLU(True)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=filter_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class U_net(nn.Module):
    def __init__(
        self,
        encode_in: tuple = (1,),
        encode_out: tuple = (64,),
        decode_in: tuple = (128,),
        decode_out: tuple = (64,),
        filter_size: int = 3,
        **kwargs,
    ) -> None:
        """Plain U-Net (convolutional bottleneck).

        Args same as before; ViT-specific args are ignored.
        """
        super().__init__()
        assert len(encode_in) == len(
            decode_in
        ), "U-net should have the same number of encode and decode layers"

        normalize: bool = kwargs.get("normalize", False)
        self.N_layers = len(encode_in)
        self.encode = nn.ModuleList()
        for ii in range(self.N_layers):
            self.encode.append(
                EncodeBlock(
                    in_channels=encode_in[ii],
                    out_channels=encode_out[ii],
                    residual_channels=int(decode_in[self.N_layers - 1 - ii] / 2),
                    filter_size=filter_size,
                    **kwargs,
                )
            )

        last_encode = encode_out[-1]
        bottleneck_channels = kwargs.get("bottleneck_channels", last_encode * 2)

        # Replace ViT bottleneck with plain conv bottleneck
        self.bottleneck = BottleneckConv(
            in_channels=last_encode,
            bottleneck_channels=bottleneck_channels,
            filter_size=filter_size,
            normalize=normalize,
            dropout_rate=kwargs.get("bottleneck_dropout", 0.0),
        )

        self.decode = nn.ModuleList()
        for ii in range(self.N_layers):
            self.decode.append(
                DecodeBlock(
                    in_channels=decode_in[ii],
                    out_channels=decode_out[ii],
                    filter_size=filter_size,
                    **kwargs,
                )
            )

        self.segment_conv = nn.Conv2d(decode_out[-1], 1, kernel_size=1)

        # build deep_heads to match decode_out channels (flexible for any architecture)
        self.deep_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in decode_out])

    def forward(self, x):
        residuals: list = []

        # Encode with residual conn
        for ii in range(self.N_layers):
            x, residual = self.encode[ii](x)
            residuals.append(residual)

        # Bottleneck (plain conv)
        x = self.bottleneck(x)

        # Decode with concat
        residuals = residuals[::-1]
        deep_preds = []

        for ii in range(self.N_layers):
            x = self.decode[ii](x, residuals[ii])

            # grab deep supervision predictions
            if ii < len(self.deep_heads):
                dp = self.deep_heads[ii](x)
                # upsample deep head prediction to current spatial size (keeps behavior consistent)
                dp = F.interpolate(dp, size=x.shape[2:], mode="bilinear", align_corners=False)
                deep_preds.append(dp)

        # Final prediction
        final_pred = self.segment_conv(x)
        return final_pred, deep_preds


if __name__ == "__main__":
    encode_in  = (1, 64, 128, 256)
    encode_out = (64, 128, 256, 512)

    # decode_in = upsampled channels BEFORE concat
    decode_in  = (1024, 512, 256, 128)
    decode_out = (512, 256, 128, 64)


    model = U_net(
        encode_in=encode_in,
        encode_out=encode_out,
        decode_in=decode_in,
        decode_out=decode_out,
        filter_size=3,
        normalize=True,
    )

    x = torch.randn(1, 1, 572, 572)
    final_pred, deep_preds = model(x)
    print("Final out shape:", final_pred.shape)
