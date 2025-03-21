
#define CUB_DISABLE_BF16_SUPPORT

#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000

namespace std{
	using size_t = ::size_t;
};


#include <curand_kernel.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

#include "../utils.cuh"

#include "../HostDeviceInterface.h"

constexpr float PI = 3.1415f;

extern "C" __global__
void kernel_drawLine(CommonLaunchArgs args, RenderTarget target, vec3 start, vec3 end, uint32_t color){

	auto grid = cg::this_grid();

	float u = float(grid.thread_rank()) / float(grid.size());

	vec3 pos = (1.0f - u) * start + u * end;

	mat4 transform = target.proj * target.view;

	vec4 ndc = transform * vec4(pos, 1.0f);
	// ndc.x = ndc.x / ndc.w;
	// ndc.y = ndc.y / ndc.w;
	// ndc.z = ndc.z / ndc.w;

	if(ndc.w < 0.0f) return;
	// if(ndc.x < -1.0f  || ndc.x > 1.0f) return;
	// if(ndc.y < -1.0f  || ndc.y > 1.0f) return;

	// if(grid.thread_rank() == 10000){
	// 	printf("%f, %f, %f \n", pos.x, pos.y, pos.z);
	// }

	ivec2 imgCoords = ivec2(
		(0.5f * (ndc.x / ndc.w) + 0.5f) * target.width,
		(0.5f * (ndc.y / ndc.w) + 0.5f) * target.height
	);

	if(imgCoords.x < 0 || imgCoords.x >= target.width) return;
	if(imgCoords.y < 0 || imgCoords.y >= target.height) return;

	int pixelID = int(imgCoords.x) + int(imgCoords.y) * target.width;

	if(pixelID >= target.width * target.height) return;
	if(pixelID < 0) return;

	float depth = ndc.w;
	uint64_t udepth = *((uint32_t*)&depth);
	uint64_t pixel = (udepth << 32ull) | color;

	atomicMin(&target.framebuffer[pixelID], pixel);

}

extern "C" __global__
void kernel_applyEyeDomeLighting(CommonLaunchArgs args, RenderTarget target){

	auto grid = cg::this_grid();

	auto uniforms = args.uniforms;

	float r = 1.5f;
	float edlStrength = 0.37f;
	float numSamples = 5;

	Pixel* framebuffer = ((Pixel*)target.framebuffer);

	int pixelID = grid.thread_rank();

	if(pixelID >= target.width * target.height) return;

	// process(target.width * target.height, [&](int pixelID){
		Pixel pixel = framebuffer[pixelID];

		// framebuffer[pixelID].color = p.color * 1;

		float sum = 0.0f;

		for(float u : {0.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f}){
			int dx = r * sin(u);
			int dy = r * cos(u);

			int neighborIndex = pixelID + dx + target.width * dy;
			neighborIndex = max(neighborIndex, 0);
			neighborIndex = min(neighborIndex, target.width * target.height - 1);

			Pixel neighbor = framebuffer[neighborIndex];

			sum = sum + max(log2(pixel.depth) - log2(neighbor.depth), 0.0f);
		}

		float response = sum / 4.0f;
		float shade = __expf(-response * 300.0f * edlStrength);

		uint32_t R = shade * ((pixel.color >>  0) & 0xff);
		uint32_t G = shade * ((pixel.color >>  8) & 0xff);
		uint32_t B = shade * ((pixel.color >> 16) & 0xff);
		uint32_t color = R | (G << 8) | (B << 16) | (255u << 24);

		framebuffer[pixelID].color = color;
	// });


}
