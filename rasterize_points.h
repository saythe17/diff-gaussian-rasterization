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

// ---- Batched versions (T frames in one call) ----

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>, std::vector<int>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
RasterizeAdditiveBatchForwardCUDA(
	const torch::Tensor& means2D_batch,   // (T, P, 2)
	const torch::Tensor& conics_batch,    // (T, P, 3)
	const torch::Tensor& weights,         // (P,)
	const torch::Tensor& colors_batch,    // (T, P, 3) per-frame colors
	const torch::Tensor& temporal_weights, // (T, P) per-frame temporal weight
	const int image_height,
	const int image_width,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeAdditiveBatchBackwardCUDA(
	const torch::Tensor& means2D_batch,   // (T, P, 2)
	const torch::Tensor& conics_batch,    // (T, P, 3)
	const torch::Tensor& weights,         // (P,)
	const torch::Tensor& colors_batch,    // (T, P, 3) per-frame colors
	const torch::Tensor& radii_batch,     // (T, P)
	const torch::Tensor& dL_dout_color,   // (T, 3, H, W)
	const std::vector<torch::Tensor>& geomBuffers,
	const std::vector<int>& num_rendered,
	const std::vector<torch::Tensor>& binningBuffers,
	const std::vector<torch::Tensor>& imgBuffers,
	const int image_height,
	const int image_width,
	const bool debug);
