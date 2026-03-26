/*
 * Additive 2D Gaussian Rasterizer — pybind11 module.
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeAdditiveForwardCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeAdditiveBackwardCUDA);
}
