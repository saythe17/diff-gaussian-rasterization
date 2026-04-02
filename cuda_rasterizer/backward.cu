/*
 * Additive 2D Gaussian Rasterizer — backward pass.
 *
 * Much simpler than original 3DGS backward because additive blending means
 * each Gaussian's gradient is independent (no transmittance coupling).
 * No reverse traversal or T-restoration needed.
 *
 * Optimizations over naive implementation:
 *   - Warp-level reduction before atomicAdd (~32x fewer atomics)
 *   - Colors cached in shared memory (avoid global memory reads in inner loop)
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ means2D,
	const float4* __restrict__ conic_weight,
	const float* __restrict__ colors,
	const float* __restrict__ dL_dpixels,    // (C, H, W)
	float* __restrict__ dL_dmeans2D,          // (P, 2)
	float* __restrict__ dL_dconics,           // (P, 3)
	float* __restrict__ dL_dweights,          // (P,)
	float* __restrict__ dL_dcolors)           // (P, C)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;

	// Warp info for reduction
	const unsigned int lane = block.thread_rank() & 31;

	// Load per-pixel output gradient
	float dL_dC[CHANNELS] = { 0 };
	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			dL_dC[ch] = dL_dpixels[ch * H * W + pix_id];
	}

	// Tile range
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Shared memory — batch loading pattern as forward, plus colors (D)
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_cw[BLOCK_SIZE];
	__shared__ float collected_colors[BLOCK_SIZE * CHANNELS];

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Cooperatively load batch
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = means2D[coll_id];
			collected_cw[block.thread_rank()] = conic_weight[coll_id];
			// Cache colors in shared memory
			for (int ch = 0; ch < CHANNELS; ch++)
				collected_colors[block.thread_rank() * CHANNELS + ch] = colors[coll_id * CHANNELS + ch];
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			// NOTE: no 'continue' for !inside — all threads must participate
			// in warp shuffles. Invalid threads contribute 0.

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 cw = collected_cw[j];
			float p00 = cw.x, p01 = cw.y, p11 = cw.z, w = cw.w;

			float power = -0.5f * (p00 * d.x * d.x + p11 * d.y * d.y) - p01 * d.x * d.y;

			bool valid = inside && (power <= 0.0f);
			float G = valid ? __expf(power) : 0.0f;

			// dot(color, dL_dC) using shared memory colors
			float dot_c_dLdC = 0.0f;
			if (valid)
			{
				for (int ch = 0; ch < CHANNELS; ch++)
					dot_c_dLdC += collected_colors[j * CHANNELS + ch] * dL_dC[ch];
			}

			float dL_dG = w * dot_c_dLdC;
			float dL_dpower = dL_dG * G;

			// Per-thread gradient contributions (0 when !valid)
			float g_c0 = dL_dpower * (-0.5f) * d.x * d.x;
			float g_c1 = dL_dpower * (-1.0f) * d.x * d.y;
			float g_c2 = dL_dpower * (-0.5f) * d.y * d.y;
			float g_m0 = dL_dpower * -(p00 * d.x + p01 * d.y);
			float g_m1 = dL_dpower * -(p01 * d.x + p11 * d.y);
			float g_w  = G * dot_c_dLdC;
			float wG   = w * G;

			// Warp-level reduction: sum across 32 lanes, ~32x fewer atomicAdd
			#pragma unroll
			for (int offset = 16; offset > 0; offset >>= 1)
			{
				g_c0 += __shfl_down_sync(0xFFFFFFFF, g_c0, offset);
				g_c1 += __shfl_down_sync(0xFFFFFFFF, g_c1, offset);
				g_c2 += __shfl_down_sync(0xFFFFFFFF, g_c2, offset);
				g_m0 += __shfl_down_sync(0xFFFFFFFF, g_m0, offset);
				g_m1 += __shfl_down_sync(0xFFFFFFFF, g_m1, offset);
				g_w  += __shfl_down_sync(0xFFFFFFFF, g_w, offset);
			}

			int gid = collected_id[j];
			if (lane == 0)
			{
				atomicAdd(&dL_dconics[gid * 3 + 0], g_c0);
				atomicAdd(&dL_dconics[gid * 3 + 1], g_c1);
				atomicAdd(&dL_dconics[gid * 3 + 2], g_c2);
				atomicAdd(&dL_dmeans2D[gid * 2 + 0], g_m0);
				atomicAdd(&dL_dmeans2D[gid * 2 + 1], g_m1);
				atomicAdd(&dL_dweights[gid], g_w);
			}

			// Color gradients: warp reduce per channel
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				float g_color = wG * dL_dC[ch];
				#pragma unroll
				for (int offset = 16; offset > 0; offset >>= 1)
					g_color += __shfl_down_sync(0xFFFFFFFF, g_color, offset);
				if (lane == 0)
					atomicAdd(&dL_dcolors[gid * CHANNELS + ch], g_color);
			}
		}
		block.sync();
	}
}

void BACKWARD::render(
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
	float* dL_dcolors,
	cudaStream_t stream)
{
	renderBackwardCUDA<NUM_CHANNELS><<<grid, block, 0, stream>>>(
		ranges, point_list,
		W, H, means2D, conic_weight, colors,
		dL_dpixels,
		dL_dmeans2D, dL_dconics, dL_dweights, dL_dcolors);
}
