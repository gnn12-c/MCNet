from typing import Optional
import torch.nn as nn
import torch
from models.utils import calculate_upsample_padding
import torch.nn.functional as F


class ChannelReduction(nn.Module):
    def __init__(
            self,
            spatial_dims,
            in_channels: int,
            r_channels,
            norm_factory=None,
            act: Optional[nn.Module] = None,
    ):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        if norm_factory is None:
            norm = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        else:
            norm = norm_factory
        act = act if act is not None else nn.GELU()
        self.conv_main = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=r_channels,
                kernel_size=3,
                padding='same'
            ),
            norm(r_channels),
            act,
            conv(
                in_channels=r_channels,
                out_channels=r_channels,
                kernel_size=3,
                padding='same'
            ),
            norm(r_channels),
            act
        )
        self.res_proj = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=r_channels,
                kernel_size=1,
            ),
            norm(r_channels),
        )
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res_proj(x)
        x = self.conv_main(x)
        out = self.act(x + res)
        return out

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()


class DecoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels_skip: int,
        in_channels: int,
        r_channels: int,
        kernel_size: int,
        upsample_kernel_size: int,
        dropout: float = 0.,
        norm_factory=None,
        act: Optional[nn.Module] = None,
        remove_high_res: bool = False,
    ):
        super().__init__()
        if remove_high_res:
            self.remove_high_res = True
            return
        else:
            self.remove_high_res = False
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        tpconv = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        if norm_factory is None:
            norm = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        else:
            norm = norm_factory
        act = act if act is not None else nn.GELU()
        self.cr_skip = ChannelReduction(
            spatial_dims=spatial_dims,
            in_channels=in_channels_skip,
            r_channels=r_channels,
            norm_factory=norm_factory,
            act=act,
        )

        pad = calculate_upsample_padding(2, upsample_kernel_size)
        self.upsample = tpconv(
            in_channels=in_channels,
            out_channels=r_channels,
            kernel_size=upsample_kernel_size,
            stride=2,
            padding=pad['padding'],
            output_padding=pad['output_padding'],
        )

        self.fusion = nn.Sequential(
            conv(
                in_channels=2 * r_channels,
                out_channels=r_channels,
                kernel_size=kernel_size,
                padding='same',
            ),
            norm(r_channels),
            act,
            conv(
                in_channels=r_channels,
                out_channels=r_channels,
                kernel_size=kernel_size,
                padding='same',
            ),
            norm(r_channels),
            act
        )

    def forward(self, x, skip) -> torch.Tensor:
        # print('input: ', x.shape)
        # print('input_skip: ', skip.shape)
        if self.remove_high_res:
            x = F.interpolate(x, scale_factor=2, mode='trilinear')
            return x
        x = self.upsample(x)
        skip = self.cr_skip(skip)
        residual = x
        x = torch.cat((x, skip), dim=1)
        out = self.fusion(x)
        out = out + residual
        return out
