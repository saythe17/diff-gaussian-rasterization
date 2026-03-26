/*
 * Additive 2D Gaussian Rasterizer — PyTorch C++ binding implementation.
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeAdditiveForwardCUDA(
	const torch::Tensor& means2D,
	const torch::Tensor& conics,
	const torch::Tensor& weights,
	const torch::Tensor& colors,
	const int image_height,
	const int image_width,
	const bool debug)
{
	if (means2D.ndimension() != 2 || means2D.size(1) != 2)
		AT_ERROR("means2D must have dimensions (num_points, 2)");

	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means2D.options().dtype(torch::kFloat32);
	auto int_opts = means2D.options().dtype(torch::kInt32);

	torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
	torch::Tensor radii = torch::full({P}, 0, int_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc, binningFunc, imgFunc,
			P, W, H,
			means2D.contiguous().data<float>(),
			conics.contiguous().data<float>(),
			weights.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			out_color.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug);
	}
	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const bool debug)
{
	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;

	torch::Tensor dL_dmeans2D = torch::zeros({P, 2}, means2D.options());
	torch::Tensor dL_dconics = torch::zeros({P, 3}, means2D.options());
	torch::Tensor dL_dweights = torch::zeros({P}, means2D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means2D.options());

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(
			P, R, W, H,
			means2D.contiguous().data<float>(),
			conics.contiguous().data<float>(),
			weights.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
			dL_dmeans2D.contiguous().data<float>(),
			dL_dconics.contiguous().data<float>(),
			dL_dweights.contiguous().data<float>(),
			dL_dcolors.contiguous().data<float>(),
			debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dconics, dL_dweights, dL_dcolors);
}
