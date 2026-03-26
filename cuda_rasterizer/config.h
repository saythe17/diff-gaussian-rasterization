/*
 * Additive 2D Gaussian Rasterizer (based on diff-gaussian-rasterization).
 * Modified for additive blending (no alpha compositing).
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#endif
