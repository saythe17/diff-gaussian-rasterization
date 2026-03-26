/*
 * Additive 2D Gaussian Rasterizer — PyTorch C++ binding declarations.
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeAdditiveForwardCUDA(
	const torch::Tensor& means2D,     // (P, 2) pixel coords
	const torch::Tensor& conics,      // (P, 3) precision matrix
	const torch::Tensor& weights,     // (P,)
	const torch::Tensor& colors,      // (P, 3)
	const int image_height,
	const int image_width,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeAdditiveBackwardCUDA(
	const torch::Tensor& means2D,
	const torch::Tensor& conics,
	const torch::Tensor& weights,
	const torch::Tensor& colors,
	const torch::Tensor& radii,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int image_height,
	const int image_width,
	const bool debug);
