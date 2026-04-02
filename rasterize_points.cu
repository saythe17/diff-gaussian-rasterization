/*
 * Additive 2D Gaussian Rasterizer — PyTorch C++ binding implementation.
 *
 * Optimizations in batch path:
 *   Forward (B):  two-phase pipeline with multi-stream parallelism
 *   Backward (E): per-frame gradient buffers for shared params (no cross-stream atomics)
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
#include <fstream>
#include <string>
#include <functional>
#include <cooperative_groups.h>
// CUB must be included BEFORE config.h: CUB headers use NUM_CHANNELS as a
// template parameter name, and config.h #defines it to 3, which breaks parsing.
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/backward.h"
#include "cuda_rasterizer/auxiliary.h"
namespace cg = cooperative_groups;

// ---- Helpers ----

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

// CPU helper: next power-of-two bit count for tile ID (same as rasterizer_impl.cu)
static uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// duplicateWithKeys kernel (same as rasterizer_impl.cu, needed for streamed forward)
__global__ void duplicateWithKeysStreamed(
	int P,
	const float2* points_xy,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= (uint32_t)idx;
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// identifyTileRanges kernel (same as rasterizer_impl.cu)
__global__ void identifyTileRangesStreamed(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Simple reduction kernel: sum T per-frame buffers into one output buffer
__global__ void reduceGradients(
	const float* __restrict__ per_frame,  // (T, size)
	float* __restrict__ output,           // (size,)
	int T, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	float sum = 0.0f;
	for (int t = 0; t < T; t++)
		sum += per_frame[t * size + idx];
	output[idx] = sum;
}

// ===================================================================
// Single-frame Forward
// ===================================================================
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

// ===================================================================
// Single-frame Backward
// ===================================================================
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

// ===================================================================
// Batched Forward: T frames, two-phase streamed pipeline (B)
// ===================================================================
std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>, std::vector<int>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
RasterizeAdditiveBatchForwardCUDA(
	const torch::Tensor& means2D_batch,   // (T, P, 2)
	const torch::Tensor& conics_batch,    // (T, P, 3)
	const torch::Tensor& weights,         // (P,)
	const torch::Tensor& colors_batch,    // (T, P, 3) per-frame colors
	const int image_height,
	const int image_width,
	const bool debug)
{
	const int T = means2D_batch.size(0);
	const int P = means2D_batch.size(1);
	const int H = image_height;
	const int W = image_width;

	auto float_opts = means2D_batch.options().dtype(torch::kFloat32);
	auto int_opts = means2D_batch.options().dtype(torch::kInt32);

	// Pre-allocate all output buffers
	torch::Tensor out_colors = torch::zeros({T, NUM_CHANNELS, H, W}, float_opts);
	torch::Tensor radii_batch = torch::full({T, P}, 0, int_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions byte_opts(torch::kByte);
	byte_opts = byte_opts.device(device);

	std::vector<torch::Tensor> geomBuffers(T);
	std::vector<int> num_rendered_vec(T, 0);
	std::vector<torch::Tensor> binningBuffers(T);
	std::vector<torch::Tensor> imgBuffers(T);

	if (P == 0)
	{
		for (int t = 0; t < T; t++)
		{
			geomBuffers[t] = torch::empty({0}, byte_opts);
			binningBuffers[t] = torch::empty({0}, byte_opts);
			imgBuffers[t] = torch::empty({0}, byte_opts);
		}
		return std::make_tuple(out_colors, radii_batch,
		                       geomBuffers, num_rendered_vec, binningBuffers, imgBuffers);
	}

	const float* weights_ptr = weights.contiguous().data<float>();

	dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	size_t num_tiles = tile_grid.x * tile_grid.y;

	// Static stream pool
	// NOTE: Use cudaStreamDefault (0) instead of cudaStreamNonBlocking.
	// cudaStreamNonBlocking streams have NO implicit sync relationship with the
	// default (null) stream that PyTorch's caching allocator uses for memory
	// lifecycle tracking.  Concurrent launches on non-blocking streams can race
	// with allocator recycling, causing cudaErrorIllegalAddress on larger
	// frame counts / resolutions.  Regular streams have the required implicit
	// sync points with the null stream so the allocator sees correct ordering.
	static std::vector<cudaStream_t> fwd_stream_pool;
	static std::vector<cudaEvent_t> fwd_event_pool;
	while ((int)fwd_stream_pool.size() < T)
	{
		cudaStream_t s;
		cudaStreamCreate(&s);  // cudaStreamDefault (blocking w.r.t. null stream)
		fwd_stream_pool.push_back(s);
		cudaEvent_t e;
		cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
		fwd_event_pool.push_back(e);
	}

	// Static pinned host memory for async D2H of num_rendered
	static int* h_num_rendered = nullptr;
	static int h_num_rendered_cap = 0;
	if (h_num_rendered_cap < T)
	{
		if (h_num_rendered) cudaFreeHost(h_num_rendered);
		cudaMallocHost(&h_num_rendered, T * sizeof(int));
		h_num_rendered_cap = T;
	}

	// Per-frame geometry state (allocated from geomBuffers)
	using GeometryState = CudaRasterizer::GeometryState;
	using ImageState = CudaRasterizer::ImageState;
	using BinningState = CudaRasterizer::BinningState;

	std::vector<GeometryState> geomStates(T);
	std::vector<ImageState> imgStates(T);

	// ---- Phase 1: preprocess + prefix sum (streamed) ----
	for (int t = 0; t < T; t++)
	{
		const float* means2D_ptr = means2D_batch[t].contiguous().data<float>();
		const float* conics_ptr = conics_batch[t].contiguous().data<float>();
		int* radii_ptr = radii_batch[t].data<int>();

		// Allocate geometry buffer
		geomBuffers[t] = torch::empty({0}, byte_opts);
		size_t geom_chunk_size = CudaRasterizer::required<GeometryState>(P);
		geomBuffers[t].resize_({(long long)geom_chunk_size});
		char* geom_chunkptr = reinterpret_cast<char*>(geomBuffers[t].data_ptr());
		geomStates[t] = GeometryState::fromChunk(geom_chunkptr, P);

		// Allocate image buffer
		imgBuffers[t] = torch::empty({0}, byte_opts);
		size_t img_chunk_size = CudaRasterizer::required<ImageState>(num_tiles);
		imgBuffers[t].resize_({(long long)img_chunk_size});
		char* img_chunkptr = reinterpret_cast<char*>(imgBuffers[t].data_ptr());
		imgStates[t] = ImageState::fromChunk(img_chunkptr, num_tiles);

		cudaStream_t stream = fwd_stream_pool[t];

		// 1. Preprocess
		FORWARD::preprocess(
			P, means2D_ptr, conics_ptr, weights_ptr,
			W, H, radii_ptr, geomStates[t].means2D, geomStates[t].conic_weight,
			tile_grid, geomStates[t].tiles_touched, stream);

		// 2. Prefix sum
		cub::DeviceScan::InclusiveSum(
			geomStates[t].scanning_space, geomStates[t].scan_size,
			geomStates[t].tiles_touched, geomStates[t].point_offsets, P,
			stream);

		// 3. Async copy num_rendered to pinned host memory
		cudaMemcpyAsync(&h_num_rendered[t], geomStates[t].point_offsets + P - 1,
			sizeof(int), cudaMemcpyDeviceToHost, stream);
	}

	// ---- Sync point: wait for all Phase 1 to finish ----
	for (int t = 0; t < T; t++)
		cudaStreamSynchronize(fwd_stream_pool[t]);

	// ---- Phase 2: duplicate + sort + render (streamed) ----
	for (int t = 0; t < T; t++)
	{
		int num_rendered = h_num_rendered[t];
		num_rendered_vec[t] = num_rendered;

		if (num_rendered == 0)
		{
			binningBuffers[t] = torch::empty({0}, byte_opts);
			continue;
		}

		cudaStream_t stream = fwd_stream_pool[t];
		int* radii_ptr = radii_batch[t].data<int>();
		float* out_ptr = out_colors[t].data<float>();

		// Allocate binning buffer
		binningBuffers[t] = torch::empty({0}, byte_opts);
		size_t binning_chunk_size = CudaRasterizer::required<BinningState>(num_rendered);
		binningBuffers[t].resize_({(long long)binning_chunk_size});
		char* binning_chunkptr = reinterpret_cast<char*>(binningBuffers[t].data_ptr());
		BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

		// 4. Duplicate Gaussians into per-tile lists
		duplicateWithKeysStreamed<<<(P + 255) / 256, 256, 0, stream>>>(
			P,
			geomStates[t].means2D,
			geomStates[t].point_offsets,
			binningState.point_list_keys_unsorted,
			binningState.point_list_unsorted,
			radii_ptr,
			tile_grid);

		// 5. Sort by tile ID
		int bit = getHigherMsb(tile_grid.x * tile_grid.y);
		cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space,
			binningState.sorting_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			binningState.point_list_unsorted, binningState.point_list,
			num_rendered, 0, 32 + bit, stream);

		// 6. Identify per-tile ranges
		cudaMemsetAsync(imgStates[t].ranges, 0, num_tiles * sizeof(uint2), stream);
		identifyTileRangesStreamed<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
			num_rendered,
			binningState.point_list_keys,
			imgStates[t].ranges);

		// 7. Render (with per-frame colors)
		const float* colors_t_ptr = colors_batch[t].contiguous().data<float>();
		FORWARD::render(
			tile_grid, block,
			imgStates[t].ranges,
			binningState.point_list,
			W, H,
			geomStates[t].means2D,
			colors_t_ptr,
			geomStates[t].conic_weight,
			out_ptr, stream);
	}

	// Non-blocking sync: make default stream wait for all work streams
	for (int t = 0; t < T; t++)
	{
		cudaEventRecord(fwd_event_pool[t], fwd_stream_pool[t]);
		cudaStreamWaitEvent(nullptr, fwd_event_pool[t], 0);
	}

	return std::make_tuple(out_colors, radii_batch,
	                       geomBuffers, num_rendered_vec, binningBuffers, imgBuffers);
}

// ===================================================================
// Batched Backward: per-frame gradient separation (E) + multi-stream
// ===================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeAdditiveBatchBackwardCUDA(
	const torch::Tensor& means2D_batch,
	const torch::Tensor& conics_batch,
	const torch::Tensor& weights,
	const torch::Tensor& colors_batch,    // (T, P, 3) per-frame colors
	const torch::Tensor& radii_batch,
	const torch::Tensor& dL_dout_color,    // (T, 3, H, W)
	const std::vector<torch::Tensor>& geomBuffers,
	const std::vector<int>& num_rendered_vec,
	const std::vector<torch::Tensor>& binningBuffers,
	const std::vector<torch::Tensor>& imgBuffers,
	const int image_height,
	const int image_width,
	const bool debug)
{
	const int T = means2D_batch.size(0);
	const int P = means2D_batch.size(1);
	const int H = image_height;
	const int W = image_width;

	// Per-frame gradients for means2D, conics, and colors (all per-frame)
	torch::Tensor dL_dmeans2D_batch = torch::zeros({T, P, 2}, means2D_batch.options());
	torch::Tensor dL_dconics_batch = torch::zeros({T, P, 3}, means2D_batch.options());
	torch::Tensor dL_dcolors_batch = torch::zeros({T, P, NUM_CHANNELS}, means2D_batch.options());

	// Per-frame gradient buffers for weights (shared param) — reduce after
	torch::Tensor dL_dweights_per_frame = torch::zeros({T, P}, means2D_batch.options());
	torch::Tensor dL_dweights = torch::zeros({P}, means2D_batch.options());

	const float* weights_ptr = weights.contiguous().data<float>();

	if (P == 0)
		return std::make_tuple(dL_dmeans2D_batch, dL_dconics_batch, dL_dweights, dL_dcolors_batch);

	// Static stream pool (same rationale as forward: use blocking streams)
	static std::vector<cudaStream_t> bwd_stream_pool;
	static std::vector<cudaEvent_t> bwd_event_pool;
	while ((int)bwd_stream_pool.size() < T)
	{
		cudaStream_t s;
		cudaStreamCreate(&s);  // cudaStreamDefault (blocking w.r.t. null stream)
		bwd_stream_pool.push_back(s);
		cudaEvent_t e;
		cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
		bwd_event_pool.push_back(e);
	}

	// Launch backward kernels on separate streams, each writing to its own buffer
	for (int t = 0; t < T; t++)
	{
		if (num_rendered_vec[t] == 0)
			continue;

		const float* means2D_ptr = means2D_batch[t].contiguous().data<float>();
		const float* conics_ptr = conics_batch[t].contiguous().data<float>();
		const float* colors_t_ptr = colors_batch[t].contiguous().data<float>();
		const int* radii_ptr = radii_batch[t].contiguous().data<int>();
		const float* dL_dpix_ptr = dL_dout_color[t].contiguous().data<float>();

		float* dL_dmeans2D_ptr = dL_dmeans2D_batch[t].data<float>();
		float* dL_dconics_ptr = dL_dconics_batch[t].data<float>();
		float* dL_dweights_ptr = dL_dweights_per_frame[t].data<float>();
		float* dL_dcolors_ptr = dL_dcolors_batch[t].data<float>();

		CudaRasterizer::Rasterizer::backward(
			P, num_rendered_vec[t], W, H,
			means2D_ptr, conics_ptr, weights_ptr, colors_t_ptr,
			radii_ptr,
			reinterpret_cast<char*>(geomBuffers[t].contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffers[t].contiguous().data_ptr()),
			reinterpret_cast<char*>(imgBuffers[t].contiguous().data_ptr()),
			dL_dpix_ptr,
			dL_dmeans2D_ptr,
			dL_dconics_ptr,
			dL_dweights_ptr,
			dL_dcolors_ptr,
			debug,
			bwd_stream_pool[t]);
	}

	// Wait for all backward kernels to finish before reduction
	for (int t = 0; t < T; t++)
	{
		cudaEventRecord(bwd_event_pool[t], bwd_stream_pool[t]);
		cudaStreamWaitEvent(nullptr, bwd_event_pool[t], 0);
	}

	// Reduce per-frame weights gradients into final buffer (on default stream)
	// Colors are per-frame now, so no reduction needed for them.
	{
		int weights_size = P;
		int block_size = 256;
		int grid_w = (weights_size + block_size - 1) / block_size;

		reduceGradients<<<grid_w, block_size>>>(
			dL_dweights_per_frame.data<float>(),
			dL_dweights.data<float>(),
			T, weights_size);
	}

	return std::make_tuple(dL_dmeans2D_batch, dL_dconics_batch, dL_dweights, dL_dcolors_batch);
}
