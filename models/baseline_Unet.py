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
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self._stride: int = kwargs.get("stride", 1)
        self._padding: int = kwargs.get("padding", 0)
        self._dilation: int = kwargs.get("dilation", 1)
        normalize: bool = kwargs.get("normalize", False)
        self.p1 = PrintSize()
        self.p2 = PrintSize()
        self.p3 = PrintSize()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            filter_size,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )
        if normalize:
            self.bNorm1 = nn.BatchNorm2d(out_channels)
        else:
            self.bNorm1 = nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            filter_size,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )
        if normalize:
            self.bNorm2 = nn.BatchNorm2d(out_channels)
        else:
            self.bNorm2 = nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        self.p1(x)
        out = self.conv1(x)
        self.p2(out)
        out = self.bNorm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        self.p3(out)
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
                    residual, upsampled.shape[2], mode="bilinear", align_corners=False
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
        """Own implementation of U-net, more or less close to the original paper

        Args:
            encode_in (tuple, optional): n_channels per layer of encoding (entry). Defaults to (1,).
            encode_out (tuple, optional):n_channels per layer of encoding (out). Defaults to (64,).
            decode_in (tuple, optional): n_channels per layer of decoding (entry). Defaults to (128,).
            decode_out (tuple, optional): n_channels per layer of decoding (entry). Defaults to (64,).
            filter_size (int, optional):  Defaults to 3.
            kwargs:
                - normalize (bool): perform normalization after each conv
                - stride (int): ..
                - padding (int): ..
                - dilation (int): ..
                - upsampling (str): "Ctranspose" or "bilinear"
                - residual (str): "interpolate" or "crop"
        """
        super().__init__()
        assert len(encode_in) == len(
            decode_in
        ), "U-net should have the same number of encode and decode layer"
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
        self.bottleneck = nn.Sequential(
            nn.Conv2d(last_encode, bottleneck_channels, filter_size),
            nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, filter_size),
            nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity(),
            nn.ReLU(True),
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
        self.segment_conv = nn.Sequential(nn.Conv2d(decode_out[-1], 2, 1))

    def forward(self, x):
        residuals: list = []

        # Encode with residual conn
        for ii in range(self.N_layers):
            x, residual = self.encode[ii](x)
            residuals.append(residual)

        x = self.bottleneck(x)

        # Decode with concat
        residuals = residuals[::-1]
        for ii in range(self.N_layers):
            x = self.decode[ii](x, residuals[ii])
        x = self.segment_conv(x)
        return x


if __name__ == "__main__":
    # Dummy test
    # model = EncodeBlock(in_channels=1, out_channels=64, residual_channels=64)
    # x = torch.randn(1, 1, 572, 572)
    # pooled, residual = model(x)
    # print("pooled shape:", pooled.shape)
    # residual = F.interpolate(residual, 392, mode="bilinear", align_corners=False)
    # print("residual shape:", residual.shape)
    model = U_net()
    x = torch.randn(1, 1, 572, 572)
    x_out = model(x)
    print("Out shape: ", x_out.shape)
