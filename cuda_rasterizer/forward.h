/*
 * Additive 2D Gaussian Rasterizer — forward declarations.
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace FORWARD
{
	// Per-Gaussian preprocessing: compute radius, tile overlap, pack data.
	void preprocess(int P,
		const float* means2D,       // (P, 2) pixel coords
		const float* conics,        // (P, 3) precision matrix (p00, p01, p11)
		const float* weights,       // (P,)   combined weight
		int W, int H,
		int* radii,
		float2* means2D_packed,     // output: float2 packed
		float4* conic_weight,       // output: (p00, p01, p11, w) packed
		const dim3 grid,
		uint32_t* tiles_touched,
		const float* temporal_weights = nullptr,  // (P,) per-frame temporal weight (nullable)
		cudaStream_t stream = nullptr);

	// Tile-based additive rendering.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float* colors,
		const float4* conic_weight,
		float* out_color,
		cudaStream_t stream = nullptr);
}

#endif
