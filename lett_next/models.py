from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

MEDNEXTV2_VARIANTS: dict[str, dict[str, object]] = {
    "mednextv2_f32": {
        "base_channels": 32,
        "stage_depths": (2, 2, 4, 4, 2),
        "decoder_depths": (4, 4, 2, 2),
        "expansion_ratio": 2,
        "depthwise_kernel_size": 3,
        "use_checkpoint": True,
    },
}


class GRN3d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class MedNeXtV2Block3D(nn.Module):
    def __init__(self, dim: int, expansion_ratio: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = dim * expansion_ratio
        self.depthwise = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = nn.InstanceNorm3d(dim, affine=True)
        self.expand = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN3d(hidden_dim)
        self.compress = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.compress(x)
        return x + residual


class MedNeXtV2DownBlock3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, expansion_ratio: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        hidden_dim = out_dim * expansion_ratio
        self.depthwise = nn.Conv3d(in_dim, in_dim, kernel_size=kernel_size, stride=2, padding=padding, groups=in_dim)
        self.norm = nn.InstanceNorm3d(in_dim, affine=True)
        self.expand = nn.Conv3d(in_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN3d(hidden_dim)
        self.compress = nn.Conv3d(hidden_dim, out_dim, kernel_size=1)
        self.residual = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.compress(x)
        return x + residual


class MedNeXtV2UpBlock3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, expansion_ratio: int) -> None:
        super().__init__()
        hidden_dim = out_dim * expansion_ratio
        self.depthwise = nn.ConvTranspose3d(in_dim, in_dim, kernel_size=2, stride=2, groups=in_dim)
        self.norm = nn.InstanceNorm3d(in_dim, affine=True)
        self.expand = nn.Conv3d(in_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN3d(hidden_dim)
        self.compress = nn.Conv3d(hidden_dim, out_dim, kernel_size=1)
        self.residual = nn.Conv3d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.compress(x)
        return x + residual


class MedNeXtV2UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int,
        stage_depths: tuple[int, int, int, int, int],
        decoder_depths: tuple[int, int, int, int],
        expansion_ratio: int,
        depthwise_kernel_size: int = 3,
        deep_supervision: bool = False,
        aux_output_classes: int | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.deep_supervision = deep_supervision
        self.aux_output_classes = None if aux_output_classes is None else int(aux_output_classes)
        self.depthwise_kernel_size = depthwise_kernel_size
        dims = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.stem = nn.Conv3d(in_channels, dims[0], kernel_size=1)
        self.stage0 = self._make_stage(dims[0], stage_depths[0], expansion_ratio, depthwise_kernel_size)
        self.down1 = MedNeXtV2DownBlock3D(
            dims[0], dims[1], expansion_ratio=expansion_ratio, kernel_size=depthwise_kernel_size
        )
        self.stage1 = self._make_stage(dims[1], stage_depths[1], expansion_ratio, depthwise_kernel_size)
        self.down2 = MedNeXtV2DownBlock3D(
            dims[1], dims[2], expansion_ratio=expansion_ratio, kernel_size=depthwise_kernel_size
        )
        self.stage2 = self._make_stage(dims[2], stage_depths[2], expansion_ratio, depthwise_kernel_size)
        self.down3 = MedNeXtV2DownBlock3D(
            dims[2], dims[3], expansion_ratio=expansion_ratio, kernel_size=depthwise_kernel_size
        )
        self.stage3 = self._make_stage(dims[3], stage_depths[3], expansion_ratio, depthwise_kernel_size)
        self.down4 = MedNeXtV2DownBlock3D(
            dims[3], dims[4], expansion_ratio=expansion_ratio, kernel_size=depthwise_kernel_size
        )
        self.stage4 = self._make_stage(dims[4], stage_depths[4], expansion_ratio, depthwise_kernel_size)
        self.up3 = MedNeXtV2UpBlock3D(dims[4], dims[3], expansion_ratio=expansion_ratio)
        self.dec3 = self._make_stage(dims[3], decoder_depths[0], expansion_ratio, depthwise_kernel_size)
        self.up2 = MedNeXtV2UpBlock3D(dims[3], dims[2], expansion_ratio=expansion_ratio)
        self.dec2 = self._make_stage(dims[2], decoder_depths[1], expansion_ratio, depthwise_kernel_size)
        self.up1 = MedNeXtV2UpBlock3D(dims[2], dims[1], expansion_ratio=expansion_ratio)
        self.dec1 = self._make_stage(dims[1], decoder_depths[2], expansion_ratio, depthwise_kernel_size)
        self.up0 = MedNeXtV2UpBlock3D(dims[1], dims[0], expansion_ratio=expansion_ratio)
        self.dec0 = self._make_stage(dims[0], decoder_depths[3], expansion_ratio, depthwise_kernel_size)
        self.out = nn.Conv3d(dims[0], out_channels, kernel_size=1)
        self.aux_out = None if self.aux_output_classes is None else nn.Conv3d(dims[0], self.aux_output_classes, kernel_size=1)
        if deep_supervision:
            self.aux1 = nn.Conv3d(dims[1], out_channels, kernel_size=1)
            self.aux2 = nn.Conv3d(dims[2], out_channels, kernel_size=1)
            self.aux3 = nn.Conv3d(dims[3], out_channels, kernel_size=1)

    @staticmethod
    def _make_stage(dim: int, depth: int, expansion_ratio: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            *(MedNeXtV2Block3D(dim, expansion_ratio=expansion_ratio, kernel_size=kernel_size) for _ in range(depth))
        )

    def _run_module(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and any(input.requires_grad for input in inputs):
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    @staticmethod
    def _align_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            return F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return x

    @staticmethod
    def _upsample_logits(logits: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
        if logits.shape[2:] != spatial_shape:
            return F.interpolate(logits, size=spatial_shape, mode="trilinear", align_corners=False)
        return logits

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        input_shape = x.shape[2:]
        x0 = self._run_module(self.stage0, self.stem(x))
        x1 = self._run_module(self.stage1, self._run_module(self.down1, x0))
        x2 = self._run_module(self.stage2, self._run_module(self.down2, x1))
        x3 = self._run_module(self.stage3, self._run_module(self.down3, x2))
        x4 = self._run_module(self.stage4, self._run_module(self.down4, x3))
        d3 = self._align_to_skip(self._run_module(self.up3, x4), x3) + x3
        d3 = self._run_module(self.dec3, d3)
        d2 = self._align_to_skip(self._run_module(self.up2, d3), x2) + x2
        d2 = self._run_module(self.dec2, d2)
        d1 = self._align_to_skip(self._run_module(self.up1, d2), x1) + x1
        d1 = self._run_module(self.dec1, d1)
        d0 = self._align_to_skip(self._run_module(self.up0, d1), x0) + x0
        d0 = self._run_module(self.dec0, d0)
        logits = self._upsample_logits(self.out(d0), input_shape)
        aux_logits = None if self.aux_out is None else self._upsample_logits(self.aux_out(d0), input_shape)
        if not (self.deep_supervision and self.training):
            if aux_logits is None:
                return logits
            return {"main": logits, "aux": aux_logits}
        aux1 = self._upsample_logits(self.aux1(d1), input_shape)
        aux2 = self._upsample_logits(self.aux2(d2), input_shape)
        aux3 = self._upsample_logits(self.aux3(d3), input_shape)
        main_outputs = (logits, aux1, aux2, aux3)
        if aux_logits is None:
            return main_outputs
        return {"main": main_outputs, "aux": aux_logits}


def _final_logits(output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, object]) -> torch.Tensor:
    if isinstance(output, dict):
        if "main" not in output:
            raise ValueError("Model output dict is missing 'main'")
        return _final_logits(output["main"])  # type: ignore[arg-type]
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model output tuple/list is empty")
        return output[0]
    return output


def _build_mednextv2(
    model_name: str,
    in_channels: int,
    out_channels: int,
    *,
    deep_supervision: bool = False,
    aux_output_classes: int | None = None,
) -> MedNeXtV2UNet3D:
    variant = MEDNEXTV2_VARIANTS[model_name]
    return MedNeXtV2UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=int(variant["base_channels"]),
        stage_depths=variant["stage_depths"],  # type: ignore[arg-type]
        decoder_depths=variant["decoder_depths"],  # type: ignore[arg-type]
        expansion_ratio=int(variant["expansion_ratio"]),
        depthwise_kernel_size=int(variant["depthwise_kernel_size"]),
        deep_supervision=deep_supervision,
        aux_output_classes=aux_output_classes,
        use_checkpoint=bool(variant.get("use_checkpoint", False)),
    )


def build_model(
    model_name: str,
    in_channels: int,
    out_channels: int = 1,
    deep_supervision: bool = False,
    aux_output_classes: int | None = None,
):
    if model_name in MEDNEXTV2_VARIANTS:
        return _build_mednextv2(
            model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            deep_supervision=deep_supervision,
            aux_output_classes=aux_output_classes,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
