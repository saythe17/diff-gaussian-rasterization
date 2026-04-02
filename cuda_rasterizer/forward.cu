/*
 * Additive 2D Gaussian Rasterizer — forward pass.
 *
 * Two kernels:
 *   1) preprocessCUDA  — compute screen radius, tile overlap, pack data
 *   2) renderCUDA      — tile-based additive blending (no alpha compositing)
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Preprocess: compute radius from precision matrix, tile overlap, pack data
// ---------------------------------------------------------------------------
__global__ void preprocessCUDA(
	int P,
	const float* __restrict__ means2D,   // (P, 2) pixel coords
	const float* __restrict__ conics,     // (P, 3) precision (p00, p01, p11)
	const float* __restrict__ weights,    // (P,)
	int W, int H,
	int* __restrict__ radii,
	float2* __restrict__ means2D_packed,
	float4* __restrict__ conic_weight,
	const dim3 grid,
	uint32_t* __restrict__ tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize to invisible
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Read pixel-space position
	float px = means2D[idx * 2 + 0];
	float py = means2D[idx * 2 + 1];

	// Read precision matrix in pixel space
	float p00 = conics[idx * 3 + 0];
	float p01 = conics[idx * 3 + 1];
	float p11 = conics[idx * 3 + 2];

	// Compute screen-space radius from precision matrix.
	// Covariance = inverse(precision):
	//   det_P = p00*p11 - p01^2
	//   cov_xx = p11/det_P,  cov_yy = p00/det_P
	// Eigenvalues of covariance give the extent.
	float det_p = p00 * p11 - p01 * p01;
	if (det_p <= 1e-12f)
		return;

	float inv_det_p = 1.0f / det_p;
	float cov_xx = p11 * inv_det_p;
	float cov_yy = p00 * inv_det_p;

	float mid = 0.5f * (cov_xx + cov_yy);
	float disc = max(0.1f, mid * mid - inv_det_p);
	float lambda_max = mid + sqrtf(disc);
	int my_radius = (int)ceilf(3.0f * sqrtf(max(0.0f, lambda_max)));

	if (my_radius <= 0)
		return;

	// Bounding rectangle in tile coordinates
	float2 point_image = { px, py };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store results
	radii[idx] = my_radius;
	means2D_packed[idx] = point_image;
	conic_weight[idx] = { p00, p01, p11, weights[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// ---------------------------------------------------------------------------
// Render: tile-based additive blending
// ---------------------------------------------------------------------------
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,       // (P, C) colors
	const float4* __restrict__ conic_weight,   // (p00, p01, p11, w)
	float* __restrict__ out_color)
{
	// Identify current tile and pixel
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;

	// Load tile range
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Shared memory for batch loading
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_cw[BLOCK_SIZE];

	// Accumulator
	float C[CHANNELS] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Cooperatively load a batch of Gaussians into shared memory
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_cw[block.thread_rank()] = conic_weight[coll_id];
		}
		block.sync();

		// Process each Gaussian in the batch
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			if (!inside)
				continue;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 cw = collected_cw[j];

			// Mahalanobis distance: -0.5*(p00*dx^2 + 2*p01*dx*dy + p11*dy^2)
			float power = -0.5f * (cw.x * d.x * d.x + cw.z * d.y * d.y) - cw.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float G = __expf(power);
			float wG = cw.w * G;  // weight * Gaussian

			// Additive blending: C += color * w * G
			int gid = collected_id[j];
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[gid * CHANNELS + ch] * wG;
		}
		block.sync();
	}

	// Write output
	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch];
	}
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void FORWARD::preprocess(int P,
	const float* means2D,
	const float* conics,
	const float* weights,
	int W, int H,
	int* radii,
	float2* means2D_packed,
	float4* conic_weight,
	const dim3 grid,
	uint32_t* tiles_touched,
	cudaStream_t stream)
{
	preprocessCUDA<<<(P + 255) / 256, 256, 0, stream>>>(
		P, means2D, conics, weights,
		W, H, radii, means2D_packed, conic_weight,
		grid, tiles_touched);
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_weight,
	float* out_color,
	cudaStream_t stream)
{
	renderCUDA<NUM_CHANNELS><<<grid, block, 0, stream>>>(
		ranges, point_list,
		W, H, means2D, colors, conic_weight,
		out_color);
}
