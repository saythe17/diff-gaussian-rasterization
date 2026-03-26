/*
 * Additive 2D Gaussian Rasterizer — backward pass.
 *
 * Much simpler than original 3DGS backward because additive blending means
 * each Gaussian's gradient is independent (no transmittance coupling).
 * No reverse traversal or T-restoration needed.
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

	// Shared memory — same batch loading pattern as forward
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_cw[BLOCK_SIZE];

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
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			if (!inside)
				continue;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 cw = collected_cw[j];
			float p00 = cw.x, p01 = cw.y, p11 = cw.z, w = cw.w;

			float power = -0.5f * (p00 * d.x * d.x + p11 * d.y * d.y) - p01 * d.x * d.y;
			if (power > 0.0f)
				continue;

			float G = __expf(power);
			int gid = collected_id[j];

			// dot(color, dL_dC)
			float dot_c_dLdC = 0.0f;
			for (int ch = 0; ch < CHANNELS; ch++)
				dot_c_dLdC += colors[gid * CHANNELS + ch] * dL_dC[ch];

			// dL/dG = w * dot(c, dL_dC)
			float dL_dG = w * dot_c_dLdC;
			// dL/dpower = dL_dG * G  (since G = exp(power), dG/dpower = G)
			float dL_dpower = dL_dG * G;

			// ---- Gradients w.r.t. precision matrix (conic) ----
			// power = -0.5*(p00*dx^2 + 2*p01*dx*dy + p11*dy^2)
			atomicAdd(&dL_dconics[gid * 3 + 0], dL_dpower * (-0.5f) * d.x * d.x);
			atomicAdd(&dL_dconics[gid * 3 + 1], dL_dpower * (-1.0f) * d.x * d.y);
			atomicAdd(&dL_dconics[gid * 3 + 2], dL_dpower * (-0.5f) * d.y * d.y);

			// ---- Gradients w.r.t. 2D mean position ----
			// d = mean - pixel, so dpower/d(mean_x) = -(p00*dx + p01*dy)
			atomicAdd(&dL_dmeans2D[gid * 2 + 0], dL_dpower * -(p00 * d.x + p01 * d.y));
			atomicAdd(&dL_dmeans2D[gid * 2 + 1], dL_dpower * -(p01 * d.x + p11 * d.y));

			// ---- Gradient w.r.t. combined weight ----
			// C[ch] += color[ch] * w * G  =>  dL/dw = G * dot(c, dL_dC)
			atomicAdd(&dL_dweights[gid], G * dot_c_dLdC);

			// ---- Gradients w.r.t. color ----
			// C[ch] += color[ch] * w * G  =>  dL/dcolor[ch] = w * G * dL_dC[ch]
			float wG = w * G;
			for (int ch = 0; ch < CHANNELS; ch++)
				atomicAdd(&dL_dcolors[gid * CHANNELS + ch], wG * dL_dC[ch]);
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
	float* dL_dcolors)
{
	renderBackwardCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges, point_list,
		W, H, means2D, conic_weight, colors,
		dL_dpixels,
		dL_dmeans2D, dL_dconics, dL_dweights, dL_dcolors);
}
