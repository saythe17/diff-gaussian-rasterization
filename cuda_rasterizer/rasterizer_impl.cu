/*
 * Additive 2D Gaussian Rasterizer — orchestrator.
 *
 * Pipeline:
 *   1. Preprocess (radius, tile overlap, pack data)
 *   2. Prefix sum on tiles_touched
 *   3. Duplicate Gaussians into per-tile lists, sort by tile
 *   4. Identify per-tile ranges
 *   5. Render (additive blending)
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// ---- CPU helper: next power-of-two bit count for tile ID ----
uint32_t getHigherMsb(uint32_t n)
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

// ---- Duplicate each Gaussian into every tile it touches ----
// Key = tile_id (upper 32 bits) | gaussian_idx (lower 32 bits)
// No depth sorting needed for additive blending.
__global__ void duplicateWithKeys(
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
				key |= (uint32_t)idx;   // Gaussian index (deterministic order)
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// ---- Identify per-tile ranges in sorted list ----
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
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

// ---- State allocation ----

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.conic_weight, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// ===================================================================
// FORWARD
// ===================================================================
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P,
	const int width, int height,
	const float* means2D,
	const float* conics,
	const float* weights,
	const float* colors,
	float* out_color,
	int* radii,
	bool debug)
{
	// Allocate geometry buffer
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
		radii = geomState.internal_radii;

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Allocate image buffer (just ranges, no accum_alpha/n_contrib)
	size_t img_chunk_size = required<ImageState>(tile_grid.x * tile_grid.y);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, tile_grid.x * tile_grid.y);

	// 1. Preprocess: compute radius, tile overlap, pack data
	CHECK_CUDA(FORWARD::preprocess(
		P, means2D, conics, weights,
		width, height,
		radii, geomState.means2D, geomState.conic_weight,
		tile_grid, geomState.tiles_touched
	), debug)

	// 2. Prefix sum over tiles_touched
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P), debug)

	// 3. Get total number of tile-Gaussian pairs
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1,
		sizeof(int), cudaMemcpyDeviceToHost), debug);

	if (num_rendered == 0)
		return 0;

	// Allocate binning buffer
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 4. Duplicate Gaussians into per-tile lists
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid);
	CHECK_CUDA(, debug)

	// 5. Sort by tile ID
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// 6. Identify per-tile ranges
	CHECK_CUDA(cudaMemset(imgState.ranges, 0,
		tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
		num_rendered,
		binningState.point_list_keys,
		imgState.ranges);
	CHECK_CUDA(, debug)

	// 7. Render (additive blending)
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		colors,
		geomState.conic_weight,
		out_color), debug)

	return num_rendered;
}

// ===================================================================
// BACKWARD
// ===================================================================
void CudaRasterizer::Rasterizer::backward(
	const int P, int R,
	const int width, int height,
	const float* means2D,
	const float* conics,
	const float* weights,
	const float* colors,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* image_buffer,
	const float* dL_dpix,
	float* dL_dmeans2D,
	float* dL_dconics,
	float* dL_dweights,
	float* dL_dcolors,
	bool debug)
{
	// Recover state from buffers
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	ImageState imgState = ImageState::fromChunk(image_buffer, tile_grid.x * tile_grid.y);

	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Single backward kernel: gradients for means2D, conics, weights, colors
	CHECK_CUDA(BACKWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		geomState.conic_weight,
		colors,
		dL_dpix,
		dL_dmeans2D,
		dL_dconics,
		dL_dweights,
		dL_dcolors), debug)

	// No BACKWARD::preprocess needed — no 3D projection to backprop through.
	// Gradients w.r.t. original model params (scaling, rotation, xyz) are
	// handled by PyTorch autograd in the Python wrapper.
}
