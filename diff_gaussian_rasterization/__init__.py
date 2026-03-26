"""
Additive 2D Gaussian Rasterizer — Python interface.

Drop-in replacement for the original diff-gaussian-rasterization, adapted for
additive blending on 2D Gaussians (no 3D projection, no alpha compositing).

Keeps the original class names: GaussianRasterizationSettings, GaussianRasterizer,
_RasterizeGaussians, rasterize_gaussians.
"""

from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means2D,
    conics,
    weights,
    colors,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means2D,
        conics,
        weights,
        colors,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2D, conics, weights, colors, raster_settings):
        args = (
            means2D,
            conics,
            weights,
            colors,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means2D, conics, weights, colors, radii,
                              geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        means2D, conics, weights, colors, radii, \
            geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        args = (
            means2D,
            conics,
            weights,
            colors,
            radii,
            grad_out_color,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_conics, grad_weights, grad_colors = \
                    _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_conics, grad_weights, grad_colors = \
                _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means2D,
            grad_conics,
            grad_weights,
            grad_colors,
            None,  # raster_settings
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    debug: bool = False


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means2D, conics, weights, colors):
        """Rasterize 2D Gaussians with additive blending.

        Args:
            means2D: (N, 2) pixel-space positions
            conics:  (N, 3) precision matrix in pixel space (p00, p01, p11)
            weights: (N,)   combined weight (opacity * temporal_weight)
            colors:  (N, 3) per-Gaussian RGB

        Returns:
            color: (3, H, W)
            radii: (N,) int
        """
        return rasterize_gaussians(
            means2D,
            conics,
            weights,
            colors,
            self.raster_settings,
        )
