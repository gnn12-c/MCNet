from typing import Optional, Sequence, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.man.ssm import SS2DConv_K1, SS3DConv
from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetResBlock, UnetUpBlock
from models.utils import calculate_upsample_padding, calculate_downsample_padding
from models.man.decoder import DecoderBlock


def get_unet_small(spatial_dims):
    # 最后一个ks还没用上
    # [input, downsamples * 5, output]
    kernel_size = [3, 5, 5, 5, 5, 5, 5]
    # [downsamples * 5]
    d_conv = [3, 3, 3, 3, 3]
    # [upsample * 5]
    upsample_kernel_size = [2, 2, 2, 2, 2]
    # [upsample * 5]
    reduction_filters = [64, 64, 64, 32, 32]
    if spatial_dims == 2:
        filters = [32, 64, 128, 256, 512, 728]
        depths_encoder = [1, 1, 1, 1, 1]
    else:
        # [input, downsamples * 5]
        filters = [32, 64, 128, 256, 320, 320]
        # [upsample * 5]
        depths_encoder = [1, 1, 1, 1, 1]
    in_channels = 3 if spatial_dims == 2 else 1
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=1,
        kernel_size=kernel_size,
        d_conv=d_conv,
        reduction_filters=reduction_filters,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        depths_encoder=depths_encoder,
    )

class ConvPath(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        dim_expand_factor: int,
        kernel_size: Union[int, tuple],
        in_channels: int,
        out_channels: int,
        num_layers: int,
        norm,
        act,
    ):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        conv_blocks = []
        for i in range(0, num_layers - 1):
            block = nn.Sequential(
                conv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    padding='same',
                    groups=in_channels,
                ),
                norm(in_channels),
                conv(
                    in_channels=in_channels,
                    out_channels=dim_expand_factor * in_channels,
                    kernel_size=1,
                ),
                act,
                conv(
                    in_channels=dim_expand_factor * in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                ),
            )
            conv_blocks.append(block)
        last_block = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding='same',
                groups=in_channels,
            ),
            norm(in_channels),
            conv(
                in_channels=in_channels,
                out_channels=dim_expand_factor * in_channels,
                kernel_size=1,
            ),
            act,
            conv(
                in_channels=dim_expand_factor * in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            norm(out_channels),
        )
        conv_blocks.append(last_block)
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        return x


class ManBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        d_conv,
        norm_factory=None,
        act=None,
        dropout=0.,
        residual: bool = False,
        slow_sample: bool = False,
        down_sample=False,
    ):
        super().__init__()

        self.residual = residual
        self.slow_sample = slow_sample
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        # tran_conv = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        if norm_factory is None:
            norm = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        else:
            norm = norm_factory

        if spatial_dims == 2:
            self.ssm = SS2DConv_K1(d_model=in_channels, d_conv=d_conv)
        else:
            self.ssm = SS3DConv(d_model=in_channels, d_conv=d_conv)

        self.conv = ConvPath(
            spatial_dims=spatial_dims,
            dim_expand_factor=4,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=2,
            norm=norm,
            act=act,
        )

        self.fc_man = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
            ),
            norm(in_channels),
        )
        self.fc_residual = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
            ),
            norm(in_channels),
        )

        self.out_norm = norm(in_channels)

        self.downsample = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=calculate_downsample_padding(3),
            ),
            act
        ) if down_sample else act

    def forward(self, x):
        residual = self.fc_residual(x)
        o_conv = self.conv(x)
        i_ssm = torch.moveaxis(x, 1, -1).contiguous()
        o_ssm = self.ssm(i_ssm)
        o_ssm = torch.moveaxis(o_ssm, -1, 1).contiguous()
        o_ssm = self.fc_man(o_ssm)
        out = torch.mul(o_conv, F.silu(o_ssm))
        out = self.out_norm(out)
        if self.residual:
            out = out + residual
        out = self.downsample(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, upsample, add_block=None):
        super().__init__()
        self.upsample = upsample
        self.add_block = add_block if add_block is not None else nn.Identity()

    def forward(self, x, skip):
        x = self.upsample(x, skip)
        return self.add_block(x)

class UNet(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[Union[Sequence[int], int]] = None,
            d_conv: Sequence[Union[Sequence[int], int]] = None,
            reduction_filters:Sequence[Union[Sequence[int], int]] = None,
            upsample_kernel_size: Sequence[Union[Sequence[int], int]] = None,
            filters: Optional[Sequence[int]] = None,
            depths_encoder: Optional[Sequence[int]] = None,
            norm_factory=None,
            act: Optional = None,
            dropout: Optional[Union[Tuple, str, float]] = 0.,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.reduction_filters = reduction_filters
        self.upsample_kernel_size = upsample_kernel_size
        self.filters = filters
        self.d_conv = d_conv
        self.depths_encoder = depths_encoder
        self.act = nn.GELU() if act is None else act
        self.dropout = dropout
        if norm_factory is None:
            self.norm_factory = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        else:
            self.norm_factory = norm_factory

        # init blocks
        self.downsample_stage = ManBlock
        self.upsample_stage = UnetUpBlock
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsample_stages()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsample_stages()
        self.output_block = self.get_output_block()

        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.input_block(x)
        # print(f'in: {x.shape}')
        d_out = [x]
        for i, stage in enumerate(self.downsamples):
            x = stage(x)
            if i < len(self.downsamples) - 1:
                d_out.append(x)
        d_out = d_out[::-1]

        x = self.bottleneck(x)

        for i, stage in enumerate(self.upsamples):
            x = stage(x, d_out[i])
        out = self.output_block(x)
        return out

    def get_input_block(self):
        return UnetResBlock(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            1,
            norm_name=("INSTANCE", {"affine": True}),
            act_name="gelu",
            dropout=self.dropout,
        )

    def get_downsample_stages(self):
        inp, out = self.filters[:-1], self.filters[1:]
        kernel_size = self.kernel_size[1:-1]
        layers = []
        for idx, (i, o, k) in enumerate(zip(inp, out, kernel_size)):
            stage = []
            for _ in range(0, self.depths_encoder[idx] - 1):
                stage.append(self.downsample_stage(
                    self.spatial_dims,
                    i, o, k,
                    self.d_conv[idx],
                    down_sample=False,
                    residual=True,
                    norm_factory=self.norm_factory,
                    act=self.act,
                    dropout=self.dropout
                ))
            stage.append(self.downsample_stage(
                    self.spatial_dims,
                    i, o, k,
                    self.d_conv[idx],
                    down_sample=True,
                    residual=True,
                    norm_factory=self.norm_factory,
                    act=self.act,
                    dropout=self.dropout
            ))
            stage = nn.Sequential(*stage)
            layers.append(stage)
        return nn.ModuleList(layers)

    def get_bottleneck(self):
        return ManBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.filters[-1],
            out_channels=self.filters[-1],
            kernel_size=self.kernel_size[len(self.kernel_size) // 2],
            d_conv=self.d_conv[-1],
            norm_factory=self.norm_factory,
            act=self.act,
        )

    def get_upsample_stages(self):
        inp = [self.filters[-1]] + [x for x in self.reduction_filters[:-1]]
        in_channels_skip = self.filters[:-1][::-1]
        kernel_size = [5 for _ in range(len(self.kernel_size) - 2)]
        upsample_kernel_size = self.upsample_kernel_size
        layers = []
        for idx, (i, ins, k, r, u) in enumerate(zip(inp, in_channels_skip, kernel_size, self.reduction_filters, upsample_kernel_size)):
            stage = DecoderBlock(
                spatial_dims=self.spatial_dims,
                in_channels_skip=ins,
                in_channels=i,
                r_channels=r,
                kernel_size=k,
                upsample_kernel_size=u,
                dropout=0.,
                norm_factory=self.norm_factory,
                act=self.act
            )
            layers.append(stage)
        return nn.ModuleList(layers)

    def get_output_block(self):
        return UnetOutBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.filters[0],
            out_channels=self.out_channels,
            dropout=self.dropout,
        )

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    from thop import profile
    import torch
    def print_model_parameters(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("--- 模型参数统计 ---")
        print(f"总参数量 (Total Parameters): {total_params:,}")
        print(f"可训练参数量 (Trainable Parameters): {trainable_params:,}")
        print(f"总参数量 (M): {total_params / 1_000_000:.2f}M")
        print(f"可训练参数量 (M): {trainable_params / 1_000_000:.2f}M")
        print("--------------------")


    model = get_unet_small(3)
    print(model)
    model.to(torch.device("cuda"))
    print_model_parameters(model)
    tensor = torch.randn(1, 1, 64, 64, 64).to(torch.device("cuda"))
    outputs = model(tensor)
    print(outputs.shape)

    dummy_input = torch.randn(1, 1, 128, 128, 128).to(torch.device("cuda"))

    flops, _ = profile(model, inputs=(dummy_input, ))

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"总计算量 (FLOPs): {flops / 1e9:.2f} G")