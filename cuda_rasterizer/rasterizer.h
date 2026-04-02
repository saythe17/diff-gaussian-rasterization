/*
 * Additive 2D Gaussian Rasterizer — public C++ API.
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static int forward(
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
			bool debug);

		static void backward(
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
			bool debug,
			cudaStream_t stream = nullptr);
	};
};

#endif
