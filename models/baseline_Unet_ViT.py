#hybrid U-Net with transformer bottleneck

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
        print("[DEBUG] Encode block:",
            "min =", pooled.min().item(),
            "max =", pooled.max().item(),
            "mean =", pooled.mean().item(),
            "shape =", tuple(pooled.shape))

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

        print("[DEBUG] Decode block:",
            "min =", out.min().item(),
            "max =", out.max().item(),
            "mean =", out.mean().item(),
            "shape =", tuple(out.shape))

        return out


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
        vit_dropout: float = 0.1,
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

        print("[DEBUG] Bottleneck output:",
            "min =", x_up.min().item(),
            "max =", x_up.max().item(),
            "mean =", x_up.mean().item(),
            "shape =", tuple(x_up.shape))

        return x_up

class U_net_ViT(nn.Module): #vit_num_layers, vit_num_heads, vit_mlp_dim, vit_dropout, are new variables, 
                            # default to small values so behavior matches simple Unet when ViT is shallow
    def __init__(
        self,
        encode_in: tuple = (1,),
        encode_out: tuple = (64,),
        decode_in: tuple = (128,),
        decode_out: tuple = (64,),
        filter_size: int = 3,
        vit_num_layers: int = 2,
        vit_num_heads: int = 4,
        vit_mlp_dim: int = None,
        vit_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """U-net with Transformer bottleneck, closely following baseline_Unet.py.

        Args:
            encode_in (tuple, optional): n_channels per layer of encoding (entry).
            encode_out (tuple, optional): n_channels per layer of encoding (out).
            decode_in (tuple, optional): n_channels per layer of decoding (entry).
            decode_out (tuple, optional): n_channels per layer of decoding (out).
            filter_size (int, optional): kernel size for convs.
            vit_num_layers (int): number of TransformerEncoder layers in the bottleneck.
            vit_num_heads (int): number of attention heads in each Transformer layer.
            vit_mlp_dim (int or None): hidden dimension of the MLP inside Transformer.
            vit_dropout (float): dropout probability inside Transformer.
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

        # main change compared to the simple Unet :
        self.bottleneck = BottleneckViT(
            in_channels=last_encode,
            bottleneck_channels=bottleneck_channels,
            normalize=normalize,
            vit_num_layers=vit_num_layers,
            vit_num_heads=vit_num_heads,
            vit_mlp_dim=vit_mlp_dim,
            vit_dropout=vit_dropout,
            filter_size=filter_size,
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
        #self.segment_conv = nn.Sequential(nn.Conv2d(decode_out[-1], 2, 1))
        self.segment_conv = nn.Conv2d(decode_out[-1], 1, kernel_size=1)    #change that makes U-Net output 1 channel, not 2 since pred: (B, 1, H, W) and mask: (B, 1, H, W)

    def forward(self, x):
        residuals: list = []

        # Encode with residual conn
        for ii in range(self.N_layers):
            x, residual = self.encode[ii](x)
            residuals.append(residual)

        # Bottleneck (conv + Transformer)
        x = self.bottleneck(x)
        print("[DEBUG] After bottleneck:",
            "min =", x.min().item(),
            "max =", x.max().item(),
            "mean =", x.mean().item(),
            "shape =", tuple(x.shape))

        # Decode with concat
        residuals = residuals[::-1]
        for ii in range(self.N_layers):
            x = self.decode[ii](x, residuals[ii])
        x = self.segment_conv(x)

        # === DEBUG: Final logits stats ===
        print("\n[DEBUG] Final logits:",
            "min =", x.min().item(),
            "max =", x.max().item(),
            "mean =", x.mean().item(),
            "shape =", tuple(x.shape))

        return x

if __name__ == "__main__": # dummy test
    model = U_net_ViT()
    x = torch.randn(1, 1, 572, 572)
    x_out = model(x)
    print("Out shape: ", x_out.shape)
