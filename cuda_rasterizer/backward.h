/*
 * Additive 2D Gaussian Rasterizer — backward declarations.
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float4* conic_weight,
		const float* colors,
		const float* dL_dpixels,
		float* dL_dmeans2D,
		float* dL_dconics,
		float* dL_dweights,
		float* dL_dcolors);
}

#endif
