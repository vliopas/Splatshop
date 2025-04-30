#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000
namespace std{
	using size_t = ::size_t;
};
// =======================

#include <curand_kernel.h>
#include <cooperative_groups.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

#include "../utils.cuh"

#include "../HostDeviceInterface.h"


constexpr int TRIANGLES_PER_SWEEP = 32;
constexpr int MAX_VERYLARGE_TRIANGLES = 10 * 1024;

__device__ uint32_t numProcessedTriangles;
__device__ uint32_t veryLargeTriangleIndices[MAX_VERYLARGE_TRIANGLES];
__device__ uint32_t veryLargeTriangleCounter;

#define RGBA(r, g, b) ((uint32_t(255) << 24) | (uint32_t(r) << 0) | (uint32_t(g) << 8) | (uint32_t(b) << 16))

constexpr uint32_t SPECTRAL[11] = {
	RGBA(158, 1, 66), 
	RGBA(213, 62, 79), 
	RGBA(244, 109, 67),
	RGBA(253, 174, 97), 
	RGBA(254, 224, 139), 
	RGBA(255, 255, 191),
	RGBA(230, 245, 152), 
	RGBA(171, 221, 164), 
	RGBA(102, 194, 165),
	RGBA(50, 136, 189), 
	RGBA(94, 79, 162)
};



uint32_t sampleSpectral(float u){

	// u = clamp(u, 0.0f, 1.0f);
	u = fmodf(u, 1.0f);
	float i = u * 10.0f;

	int i_l = int(floor(i)) % 11;
	int i_u = int(ceil(i)) % 11;
	float w = fmodf(i, 1.0f);

	uint32_t a = SPECTRAL[i_l];
	uint32_t b = SPECTRAL[i_u];
	uint32_t sample = 0;

	uint8_t* a_rgba = (uint8_t*)&a;
	uint8_t* b_rgba = (uint8_t*)&b;
	uint8_t* s_rgba = (uint8_t*)&sample;

	s_rgba[0] = (1.0f - w) * float(a_rgba[0]) + w * float(b_rgba[0]);
	s_rgba[1] = (1.0f - w) * float(a_rgba[1]) + w * float(b_rgba[1]);
	s_rgba[2] = (1.0f - w) * float(a_rgba[2]) + w * float(b_rgba[2]);
	s_rgba[3] = 255;

	return sample;
}

inline vec4 toScreenCoord(vec3 p, mat4& transform, int width, int height){
	vec4 pos = transform * vec4{p.x, p.y, p.z, 1.0f};

	pos.x = pos.x / pos.w;
	pos.y = pos.y / pos.w;

	vec4 imgPos = {
		(pos.x * 0.5f + 0.5f) * width, 
		(pos.y * 0.5f + 0.5f) * height,
		pos.z, 
		pos.w
	};

	return imgPos;
}

inline uint32_t computeColor(
	int triangleIndex, 
	TriangleData triangles, 
	TriangleMaterial material, 
	Texture texture,
	float s, float t, float v
){

	uint32_t color;
	uint8_t* rgb = (uint8_t*)&color;

	color = triangleIndex * 123456;
	// color = 0x0000ff00;

	// material.mode = MATERIAL_MODE_UVS;


	if(material.mode == MATERIAL_MODE_COLOR){
		rgb[0] = 255.0f * material.color.r;
		rgb[1] = 255.0f * material.color.g;
		rgb[2] = 255.0f * material.color.b;
		rgb[3] = 255.0f * material.color.a;
	}else if(material.mode == MATERIAL_MODE_VERTEXCOLOR && triangles.colors != nullptr){
		uint8_t rgba_0[4];
		uint8_t rgba_1[4];
		uint8_t rgba_2[4];
		memcpy(rgba_0, &triangles.colors[3 * triangleIndex + 0], 4);
		memcpy(rgba_1, &triangles.colors[3 * triangleIndex + 1], 4);
		memcpy(rgba_2, &triangles.colors[3 * triangleIndex + 2], 4);

		vec3 c0 = {rgba_0[0], rgba_0[1], rgba_0[2]};
		vec3 c1 = {rgba_1[0], rgba_1[1], rgba_1[2]};
		vec3 c2 = {rgba_2[0], rgba_2[1], rgba_2[2]};

		vec3 c = v * c0 + s * c1 + t * c2;
		color = (int(c.x) << 0) | (int(c.y) << 8) | (int(c.z) << 16);

	}else if(material.mode == MATERIAL_MODE_UVS && triangles.uv != nullptr){
		uint8_t rgba_0[4];
		uint8_t rgba_1[4];
		uint8_t rgba_2[4];

		vec2 uv0 = {
			triangles.uv[3 * triangleIndex + 0].s,
			triangles.uv[3 * triangleIndex + 0].t,
		};
		vec2 uv1 = {
			triangles.uv[3 * triangleIndex + 1].s,
			triangles.uv[3 * triangleIndex + 1].t,
		};
		vec2 uv2 = {
			triangles.uv[3 * triangleIndex + 2].s,
			triangles.uv[3 * triangleIndex + 2].t,
		};

		vec2 uv = v * uv0 + s * uv1 + t * uv2;
		uv = uv / material.uv_scale + material.uv_offset;
		uv.x = clamp(uv.x, 0.0f, 1.0f);
		uv.y = clamp(uv.y, 0.0f, 1.0f);

	}else if(material.mode == MATERIAL_MODE_TEXTURED && triangles.uv != nullptr){

		uint8_t rgba_0[4];
		uint8_t rgba_1[4];
		uint8_t rgba_2[4];

		vec2 uv0 = {
			triangles.uv[3 * triangleIndex + 0].s,
			triangles.uv[3 * triangleIndex + 0].t,
		};
		vec2 uv1 = {
			triangles.uv[3 * triangleIndex + 1].s,
			triangles.uv[3 * triangleIndex + 1].t,
		};
		vec2 uv2 = {
			triangles.uv[3 * triangleIndex + 2].s,
			triangles.uv[3 * triangleIndex + 2].t,
		};

		vec2 uv = v * uv0 + s * uv1 + t * uv2;
		uv = uv / material.uv_scale + material.uv_offset;
		uv.x = clamp(uv.x, 0.0f, 1.0f);
		uv.y = clamp(uv.y, 0.0f, 1.0f);

		if(texture.data){
			
			auto sampleTexture = [&](vec2 uv, Texture texture){
				int tx = int(uv.x * texture.width) % texture.width;
				int ty = int(uv.y * texture.height) % texture.height;
				// ty = texture.height - ty;

				int texelID = tx + texture.width * ty;

	

				if(texelID < 0){
					printf("uv:  %.2f, %.2f\n", uv.x, uv.y);
					printf("texture %d, %d\n", texture.width, texture.height);
					printf("test %d\n", texelID);
				}

				// if(triangleIndex == 0){
				// 	printf("texture.width: %d \n", texture.width);
				// }

				if(texelID < 0) return 0xff0000ff;

				uint32_t r = texture.data[4 * texelID + 0];
				uint32_t g = texture.data[4 * texelID + 1];
				uint32_t b = texture.data[4 * texelID + 2];
				uint32_t a = texture.data[4 * texelID + 3];

				uint32_t color = (r << 0) | (g << 8) | (b << 16) | (a << 24);

				

				return color;
			};

			color = sampleTexture(uv, texture);
		}else if(texture.surface != -1){
			
			auto sampleTexture = [&](vec2 uv, Texture texture){
				int tx = int(uv.x * texture.width) % texture.width;
				int ty = int(uv.y * texture.height) % texture.height;

				uint32_t color = surf2Dread<uint32_t>(texture.surface, tx * 4, ty, cudaBoundaryModeClamp);
				
				return color;
			};


			color = sampleTexture(uv, texture);
		}else if(texture.cutexture != -1){
			
			int tx = int(uv.x * texture.width) % texture.width;
			int ty = int(uv.y * texture.height) % texture.height;

			float x = clamp(uv.x * texture.width, 0.0f, texture.width - 1.0f);
			float y = clamp(uv.y * texture.height, 0.0f, texture.height - 1.0f);

			float dx = 1.0f / texture.width;
			float dy = 1.0f / texture.height;


			float4 values;
			tex2D(&values, texture.cutexture, x, y);

			rgb[0] = 255.0f * values.x;
			rgb[1] = 255.0f * values.y;
			rgb[2] = 255.0f * values.z;
			rgb[3] = 255.0f * values.w;

		}else{
			rgb[0] = 255.0f * uv.s;
			rgb[1] = 255.0f * uv.t;
		}

		// constexpr float3 SPECTRAL[11] = {
		// 	float3{158,1,66},
		// 	float3{213,62,79},
		// 	float3{244,109,67},
		// 	float3{253,174,97},
		// 	float3{254,224,139},
		// 	float3{255,255,191},
		// 	float3{230,245,152},
		// 	float3{171,221,164},
		// 	float3{102,194,165},
		// 	float3{50,136,189},
		// 	float3{94,79,162},
		// };



		// rgb[0] = SPECTRAL[5].x;
		// rgb[1] = SPECTRAL[5].y;
		// rgb[2] = SPECTRAL[5].z;

		// int n = 2'700'000;
		// int n = 48'700'000;
		// int n = triangles.count;
		// // int i = clamp(11.0f * (float(triangleIndex) / float(n)), 0.0f, 10.0f);
		// // rgb[0] = SPECTRAL[i].x;
		// // rgb[1] = SPECTRAL[i].y;
		// // rgb[2] = SPECTRAL[i].z;

		// if(triangleIndex % 50 != 0) color = 0;

		// if(triangleIndex < 48'700'000){
		// 	color = 0xff00ff00;
		// }

	}else{
		// color = 0xff0000ff;
	}

	// if(s < 0.1f) color = 0;
	// if(t < 0.1f) color = 0;
	// if(v < 0.1f) color = 0;

	// color = triangleIndex * 123456;

	return color;
}



void rasterizeTriangles_block(
	TriangleData geometry,
	TriangleMaterial material,
	uint32_t triangleOffset,
	CommonLaunchArgs args,
	RenderTarget target
){
	auto block = cg::this_thread_block();

	// mat4 rot = glm::rotate(3.1415f * 0.5f, vec3{0.0f, 1.0f, 0.0f});
	mat4 transform = target.proj * target.view * geometry.transform;

	__shared__ vec3 sh_positions[3 * TRIANGLES_PER_SWEEP];
	__shared__ vec2 sh_uvs[3 * TRIANGLES_PER_SWEEP];

	int numTrianglesInBlock = min(int(geometry.count) - triangleOffset, TRIANGLES_PER_SWEEP);

	if(numTrianglesInBlock <= 0) return;

	// load triangles into shared memory
	for(
		int i = block.thread_rank();
		i < numTrianglesInBlock;
		i += block.size()
	){
		int triangleIndex = triangleOffset + i;
		sh_positions[3 * i + 0] = geometry.position[3 * triangleIndex + 0];
		sh_positions[3 * i + 1] = geometry.position[3 * triangleIndex + 1];
		sh_positions[3 * i + 2] = geometry.position[3 * triangleIndex + 2];

		sh_uvs[3 * i + 0] = geometry.uv[3 * triangleIndex + 0];
		sh_uvs[3 * i + 1] = geometry.uv[3 * triangleIndex + 1];
		sh_uvs[3 * i + 2] = geometry.uv[3 * triangleIndex + 2];
	}

	block.sync();

	// draw triangles
	for(
		int i = block.thread_rank();
		i < numTrianglesInBlock;
		i += block.size()
	){
		int triangleIndex = triangleOffset + i;
		
		vec3 v_0 = sh_positions[3 * i + 0];
		vec3 v_1 = sh_positions[3 * i + 1];
		vec3 v_2 = sh_positions[3 * i + 2];

		vec4 p_0 = toScreenCoord(v_0, transform, target.width, target.height);
		vec4 p_1 = toScreenCoord(v_1, transform, target.width, target.height);
		vec4 p_2 = toScreenCoord(v_2, transform, target.width, target.height);

		if(p_0.w < 0.0f || p_1.w < 0.0f || p_2.w < 0.0f) continue;

		vec2 v_01 = {p_1.x - p_0.x, p_1.y - p_0.y};
		vec2 v_02 = {p_2.x - p_0.x, p_2.y - p_0.y};

		auto cross = [](vec2 a, vec2 b){ return a.x * b.y - a.y * b.x; };

		{// backface culling
			float w = cross(v_01, v_02);
			if(w < 0.0) continue;
		}

		// compute screen-space bounding rectangle
		float min_x = min(min(p_0.x, p_1.x), p_2.x);
		float min_y = min(min(p_0.y, p_1.y), p_2.y);
		float max_x = max(max(p_0.x, p_1.x), p_2.x);
		float max_y = max(max(p_0.y, p_1.y), p_2.y);

		// clamp to screen
		min_x = clamp(min_x, 0.0f, (float)target.width);
		min_y = clamp(min_y, 0.0f, (float)target.height);
		max_x = clamp(max_x, 0.0f, (float)target.width);
		max_y = clamp(max_y, 0.0f, (float)target.height);

		int size_x = ceil(max_x) - floor(min_x);
		int size_y = ceil(max_y) - floor(min_y);
		int numFragments = size_x * size_y;

		if(numFragments > 40'000){
			// uint32_t index = atomicAdd(&veryLargeTriangleCounter, 1);
			// veryLargeTriangleIndices[index] = triangleIndex;
			continue;
		}else if(numFragments > 4024){
			// TODO: schedule block-wise rasterization
			// uint32_t index = atomicAdd(&largeTriangleSchedule.numTriangles, 1);
			// largeTriangleSchedule.indices[index] = triangleIndex;
			continue;
		}

		int numProcessedSamples = 0;
		for(int fragOffset = 0; fragOffset < numFragments; fragOffset += 1){

			// safety mechanism: don't draw more than <x> pixels per thread
			if(numProcessedSamples > 4000) break;

			int fragID = fragOffset; // + block.thread_rank();
			int fragX = fragID % size_x;
			int fragY = fragID / size_x;

			vec2 pFrag = {
				floor(min_x) + float(fragX), 
				floor(min_y) + float(fragY)
			};
			vec2 sample = {pFrag.x - p_0.x, pFrag.y - p_0.y};

			// v: vertex[0], s: vertex[1], t: vertex[2]
			float s = cross(sample, v_02) / cross(v_01, v_02);
			float t = cross(v_01, sample) / cross(v_01, v_02);
			float v = 1.0f - (s + t);

			int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
			int pixelID = pixelCoords.x + pixelCoords.y * target.width;
			pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

			bool isInsideTriangle = (s >= 0.0f) && (t >= 0.0f) && (v >= 0.0f);
			// if(numFragments == 1) isInsideTriangle = true;

			// Draw every xth triangle as a point instead.
			// if(triangleIndex % 100 != 0) continue;
			// if(triangleIndex % 100 > 10) continue;
			// #define PROTOTYPE_SUBSAMPLING
			// #if defined(PROTOTYPE_SUBSAMPLING)
			// if(numFragments == 1){
			// 	// PROTOTYPING/DEBUGGING
			// 	uint32_t color = computeColor(triangleIndex, geometry, material, material.texture, s, t, v);
			// 	if((color & 0xff000000) == 0) continue;
				
			// 	float depth = p_0.w;
			// 	uint64_t udepth = *((uint32_t*)&depth);
			// 	uint64_t pixel = (udepth << 32ull) | color;

			// 	atomicMin(&target.framebuffer[pixelID + 0], pixel);
			// 	atomicMin(&target.framebuffer[pixelID + 1], pixel);
			// 	atomicMin(&target.framebuffer[pixelID + target.width + 0], pixel);
			// 	atomicMin(&target.framebuffer[pixelID + target.width + 1], pixel);
			// }
			// #endif

			if(isInsideTriangle){
				uint32_t color = computeColor(triangleIndex, geometry, material, material.texture, s, t, v);
				uint8_t* rgb = (uint8_t*)&color;

				// color = sampleSpectral(float(2 * triangleIndex) / float(geometry.count));
				// color = sampleSpectral(floor(11.0f * float(2 * triangleIndex) / float(geometry.count)) / 11.0f);

				float depth = v * p_0.w + s * p_1.w + t * p_2.w;
				uint64_t udepth = *((uint32_t*)&depth);
				uint64_t pixel = (udepth << 32ull) | color;

				atomicMin(&target.framebuffer[pixelID], pixel);
			}
			

			numProcessedSamples++;
		}
	}

	block.sync();

	
}

inline void rasterizeVeryLargeTriangles(
	TriangleData triangles, 
	TriangleMaterial material, 
	Texture texture,
	RenderTarget target, 
	mat4& transform
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	for(int i = 0; i < min(veryLargeTriangleCounter, MAX_VERYLARGE_TRIANGLES); i++){

		int triangleIndex = veryLargeTriangleIndices[i];

		vec3 v0 = triangles.position[3 * triangleIndex + 0];
		vec3 v1 = triangles.position[3 * triangleIndex + 1];
		vec3 v2 = triangles.position[3 * triangleIndex + 2];

		vec4 p0 = toScreenCoord(v0, transform, target.width, target.height);
		vec4 p1 = toScreenCoord(v1, transform, target.width, target.height);
		vec4 p2 = toScreenCoord(v2, transform, target.width, target.height);

		if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) continue;

		vec2 v01 = {p1.x - p0.x, p1.y - p0.y};
		vec2 v02 = {p2.x - p0.x, p2.y - p0.y};

		auto cross = [](vec2 a, vec2 b){ return a.x * b.y - a.y * b.x; };

		{// backface culling
			float w = cross(v01, v02);
			if(w < 0.0) continue;
		}

		// compute screen-space bounding rectangle
		float min_x = min(min(p0.x, p1.x), p2.x);
		float min_y = min(min(p0.y, p1.y), p2.y);
		float max_x = max(max(p0.x, p1.x), p2.x);
		float max_y = max(max(p0.y, p1.y), p2.y);

		// clamp to screen
		min_x = clamp(min_x, 0.0f, (float)target.width);
		min_y = clamp(min_y, 0.0f, (float)target.height);
		max_x = clamp(max_x, 0.0f, (float)target.width);
		max_y = clamp(max_y, 0.0f, (float)target.height);

		int size_x = ceil(max_x) - floor(min_x);
		int size_y = ceil(max_y) - floor(min_y);
		int numFragments = size_x * size_y;

		int fragsPerBlock = numFragments / (grid.num_blocks() - 1) + 1;

		int startFrag = (grid.block_rank() + 0) * fragsPerBlock;
		int endFrag =   (grid.block_rank() + 1) * fragsPerBlock;

		endFrag = min(endFrag, numFragments);

		int numProcessedSamples = 0;
		for(int fragOffset = startFrag; fragOffset < endFrag; fragOffset += block.num_threads()){
			// safety mechanism: don't draw more than <x> pixels per thread
			if(numProcessedSamples > 10'000) break;

			int fragID = fragOffset + block.thread_rank();
			int fragX = fragID % size_x;
			int fragY = fragID / size_x;

			vec2 pFrag = {
				floor(min_x) + float(fragX), 
				floor(min_y) + float(fragY)
			};
			vec2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

			// v: vertex[0], s: vertex[1], t: vertex[2]
			float s = cross(sample, v02) / cross(v01, v02);
			float t = cross(v01, sample) / cross(v01, v02);
			float v = 1.0f - (s + t);

			int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
			int pixelID = pixelCoords.x + pixelCoords.y * target.width;
			pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

			if(s >= 0.0f)
			if(t >= 0.0f)
			if(v >= 0.0f)
			{
				uint32_t color = computeColor(triangleIndex, triangles, material, texture, s, t, v);

				float depth = v * p0.w + s * p1.w + t * p2.w;
				uint64_t udepth = *((uint32_t*)&depth);
				uint64_t pixel = (udepth << 32ull) | color;

				atomicMin(&target.framebuffer[pixelID], pixel);
			}

			numProcessedSamples++;
		}

	}

}

inline void rasterizeLargeTriangles(
	TriangleData triangles, 
	TriangleMaterial material, 
	Texture texture,
	RenderTarget target, 
	int* triangleIndices, 
	uint32_t numTriangles, 
	mat4& transform
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	for(int i = 0; i < numTriangles; i++){

		int triangleIndex = triangleIndices[i];

		vec3 v0 = triangles.position[3 * triangleIndex + 0];
		vec3 v1 = triangles.position[3 * triangleIndex + 1];
		vec3 v2 = triangles.position[3 * triangleIndex + 2];

		vec4 p0 = toScreenCoord(v0, transform, target.width, target.height);
		vec4 p1 = toScreenCoord(v1, transform, target.width, target.height);
		vec4 p2 = toScreenCoord(v2, transform, target.width, target.height);

		// cull a triangle if one of its vertices is closer than depth 0
		if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) continue;

		vec2 v01 = {p1.x - p0.x, p1.y - p0.y};
		vec2 v02 = {p2.x - p0.x, p2.y - p0.y};

		auto cross = [](vec2 a, vec2 b){ return a.x * b.y - a.y * b.x; };

		{// backface culling
			float w = cross(v01, v02);
			if(w < 0.0) continue;
		}

		// compute screen-space bounding rectangle
		float min_x = min(min(p0.x, p1.x), p2.x);
		float min_y = min(min(p0.y, p1.y), p2.y);
		float max_x = max(max(p0.x, p1.x), p2.x);
		float max_y = max(max(p0.y, p1.y), p2.y);

		// clamp to screen
		min_x = clamp(min_x, 0.0f, (float)target.width);
		min_y = clamp(min_y, 0.0f, (float)target.height);
		max_x = clamp(max_x, 0.0f, (float)target.width);
		max_y = clamp(max_y, 0.0f, (float)target.height);

		int size_x = ceil(max_x) - floor(min_x);
		int size_y = ceil(max_y) - floor(min_y);
		int numFragments = size_x * size_y;

		// iterate through fragments in bounding rectangle and draw if within triangle
		int numProcessedSamples = 0;
		for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

			// safety mechanism: don't draw more than <x> pixels per thread
			if(numProcessedSamples > 20'000) break;

			int fragID = fragOffset + block.thread_rank();
			int fragX = fragID % size_x;
			int fragY = fragID / size_x;

			vec2 pFrag = {
				floor(min_x) + float(fragX), 
				floor(min_y) + float(fragY)
			};
			vec2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

			// v: vertex[0], s: vertex[1], t: vertex[2]
			float s = cross(sample, v02) / cross(v01, v02);
			float t = cross(v01, sample) / cross(v01, v02);
			float v = 1.0 - (s + t);

			int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
			int pixelID = pixelCoords.x + pixelCoords.y * target.width;
			pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

			if(s >= 0.0f)
			if(t >= 0.0f)
			if(v >= 0.0f)
			{
				uint32_t color = computeColor(triangleIndex, triangles, material, texture, s, t, v);

				float depth = v * p0.w + s * p1.w + t * p2.w;
				uint64_t udepth = *((uint32_t*)&depth);
				uint64_t pixel = (udepth << 32ull) | color;

				atomicMin(&target.framebuffer[pixelID], pixel);
			}

			numProcessedSamples++;
		}

	}
	
}


void rasterizeTriangles(
	TriangleData triangles, 
	CommonLaunchArgs args, 
	TriangleMaterial material, 
	Texture texture, 
	RenderTarget target
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto uniforms = args.uniforms;
	
	mat4 transform = target.proj * target.view * triangles.transform;

	__shared__ vec3 sh_positions[3 * TRIANGLES_PER_SWEEP];
	__shared__ vec2 sh_uvs[3 * TRIANGLES_PER_SWEEP];
	__shared__ int sh_blockTriangleOffset;

	__shared__ struct {
		int numTriangles;
		int indices[TRIANGLES_PER_SWEEP];
	} largeTriangleSchedule;

	while(true){


		block.sync();
		if(block.thread_rank() == 0){
			sh_blockTriangleOffset = atomicAdd(&numProcessedTriangles, TRIANGLES_PER_SWEEP);
			largeTriangleSchedule.numTriangles = 0;
		}
		block.sync();
		
		int numTrianglesInBlock = min(int(triangles.count) - sh_blockTriangleOffset, TRIANGLES_PER_SWEEP);

		if(numTrianglesInBlock <= 0) break;


		// load triangles into shared memory
		for(
			int i = block.thread_rank();
			i < numTrianglesInBlock;
			i += block.size()
		){
			int triangleIndex = sh_blockTriangleOffset + i;
			sh_positions[3 * i + 0] = triangles.position[3 * triangleIndex + 0];
			sh_positions[3 * i + 1] = triangles.position[3 * triangleIndex + 1];
			sh_positions[3 * i + 2] = triangles.position[3 * triangleIndex + 2];

			sh_uvs[3 * i + 0] = triangles.uv[3 * triangleIndex + 0];
			sh_uvs[3 * i + 1] = triangles.uv[3 * triangleIndex + 1];
			sh_uvs[3 * i + 2] = triangles.uv[3 * triangleIndex + 2];
		}

		block.sync();

		// draw triangles
		for(
			int i = block.thread_rank();
			i < numTrianglesInBlock;
			i += block.size()
		){
			int triangleIndex = sh_blockTriangleOffset + i;
			
			vec3 v_0 = sh_positions[3 * i + 0];
			vec3 v_1 = sh_positions[3 * i + 1];
			vec3 v_2 = sh_positions[3 * i + 2];

			vec4 p_0 = toScreenCoord(v_0, transform, target.width, target.height);
			vec4 p_1 = toScreenCoord(v_1, transform, target.width, target.height);
			vec4 p_2 = toScreenCoord(v_2, transform, target.width, target.height);

			if(p_0.w < 0.0f || p_1.w < 0.0f || p_2.w < 0.0f) continue;

			vec2 v_01 = {p_1.x - p_0.x, p_1.y - p_0.y};
			vec2 v_02 = {p_2.x - p_0.x, p_2.y - p_0.y};

			auto cross = [](vec2 a, vec2 b){ return a.x * b.y - a.y * b.x; };

			{// backface culling
				float w = cross(v_01, v_02);
				if(w < 0.0) continue;
			}

			// compute screen-space bounding rectangle
			float min_x = min(min(p_0.x, p_1.x), p_2.x);
			float min_y = min(min(p_0.y, p_1.y), p_2.y);
			float max_x = max(max(p_0.x, p_1.x), p_2.x);
			float max_y = max(max(p_0.y, p_1.y), p_2.y);

			// clamp to screen
			min_x = clamp(min_x, 0.0f, (float)target.width);
			min_y = clamp(min_y, 0.0f, (float)target.height);
			max_x = clamp(max_x, 0.0f, (float)target.width);
			max_y = clamp(max_y, 0.0f, (float)target.height);

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			if(numFragments > 60'000){
				uint32_t index = atomicAdd(&veryLargeTriangleCounter, 1);
				veryLargeTriangleIndices[index] = triangleIndex;
				continue;
			}else if(numFragments > 1024){
				// TODO: schedule block-wise rasterization
				uint32_t index = atomicAdd(&largeTriangleSchedule.numTriangles, 1);
				largeTriangleSchedule.indices[index] = triangleIndex;
				continue;
			}

			int numProcessedSamples = 0;
			for(int fragOffset = 0; fragOffset < numFragments; fragOffset += 1){

				// safety mechanism: don't draw more than <x> pixels per thread
				if(numProcessedSamples > 2000) break;

				int fragID = fragOffset; // + block.thread_rank();
				int fragX = fragID % size_x;
				int fragY = fragID / size_x;

				vec2 pFrag = {
					floor(min_x) + float(fragX), 
					floor(min_y) + float(fragY)
				};
				vec2 sample = {pFrag.x - p_0.x, pFrag.y - p_0.y};

				// v: vertex[0], s: vertex[1], t: vertex[2]
				float s = cross(sample, v_02) / cross(v_01, v_02);
				float t = cross(v_01, sample) / cross(v_01, v_02);
				float v = 1.0f - (s + t);

				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = pixelCoords.x + pixelCoords.y * target.width;
				pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

				if(s >= 0.0f)
				if(t >= 0.0f)
				if(v >= 0.0f)
				{
					uint32_t color = computeColor(triangleIndex, triangles, material, texture, s, t, v);

					// if((color >> 24) == 0){
					// 	color = target.framebuffer[pixelID] & 0xffffffff;
					// }

					float depth = v * p_0.w + s * p_1.w + t * p_2.w;
					uint64_t udepth = *((uint32_t*)&depth);
					uint64_t pixel = (udepth << 32ull) | color;

					atomicMin(&target.framebuffer[pixelID], pixel);
				}

				numProcessedSamples++;
			}
		}

		block.sync();

		// do blockwise rasterization for triangles that were too large for thread-wise rasterization
		rasterizeLargeTriangles(triangles, material, texture, target, largeTriangleSchedule.indices, largeTriangleSchedule.numTriangles, transform);
	}

	grid.sync();

	// rasterizeVeryLargeTriangles(triangles, material, texture, target, transform);
}

extern "C" __global__
void kernel_drawTriangles(CommonLaunchArgs args, TriangleData triangles, TriangleMaterial material, RenderTarget target){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// material.mode = MATERIAL_MODE_UVS;

	if(grid.thread_rank() == 0){
		numProcessedTriangles = 0;
		veryLargeTriangleCounter = 0;
	}

	grid.sync();

	Texture texture = material.texture;

	rasterizeTriangles(triangles, args, material, texture, target);
}


extern "C" __global__
void kernel_drawTriangleQueue(CommonLaunchArgs args, TriangleModelQueue queue, RenderTarget target){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0){
		numProcessedTriangles = 0;
		veryLargeTriangleCounter = 0;
	}


	__shared__ int sh_blockTriangleOffset;       // the global offset to the set of triangles that this block should render
	__shared__ int sh_blockLocalTriangleOffset;  // the "local" offset relative to the current triangle queue element
	__shared__ int sh_triangleQueueIndex;        // the index of the current triangle queue element
	__shared__ TriangleData sh_geometry;
	__shared__ TriangleMaterial sh_material;
	


	if(block.thread_rank() == 0){
		sh_blockTriangleOffset      = 0;
		sh_blockLocalTriangleOffset = 0;
		sh_triangleQueueIndex       = 0;
		sh_geometry                 = queue.geometries[0];
		sh_material                 = queue.materials[0];
	}

	grid.sync();

	while(true){

		// Check which batch of triangles this block should render next.
		block.sync();
		if(block.thread_rank() == 0){
			uint32_t next = atomicAdd(&numProcessedTriangles, TRIANGLES_PER_SWEEP);
			uint32_t diff = next - sh_blockTriangleOffset;

			sh_blockTriangleOffset = next;
			sh_blockLocalTriangleOffset += diff;

			// if((numProcessedTriangles / TRIANGLES_PER_SWEEP) % 1000 == 0){
			// 	printf("%8u, %8u \n", sh_blockTriangleOffset, sh_blockLocalTriangleOffset);
			// }

			// The next global triangle index may be multiple queued geometries ahead.
			// Let this block advance to the correct geometry
			while(sh_blockLocalTriangleOffset > sh_geometry.count){
				sh_triangleQueueIndex++;

				if(sh_triangleQueueIndex >= queue.count) break;

				sh_blockLocalTriangleOffset -= sh_geometry.count;

				sh_geometry = queue.geometries[sh_triangleQueueIndex];
				sh_material = queue.materials[sh_triangleQueueIndex];

			}
		}
		block.sync();

		if(sh_triangleQueueIndex >= queue.count) break;

		rasterizeTriangles_block(sh_geometry, sh_material, sh_blockLocalTriangleOffset, args, target);

	}





}

extern "C" __global__
void kernel_compute_boundingbox(CommonLaunchArgs args, TriangleData model, vec3& min, vec3& max){

	int index = cg::this_grid().thread_rank();

	if(index >= model.count) return;

	vec3 pos = vec3(model.transform * vec4(model.position[index], 1.0f));

	if(index == 0){
		float* floats = &model.transform[0].x;
		mat4 t = model.transform;
		printf("%.1f, %.1f, %.1f, %.1f \n", t[0].x, t[0].y, t[0].z, t[0].w);
		printf("%.1f, %.1f, %.1f, %.1f \n", t[1].x, t[1].y, t[1].z, t[1].w);
		printf("%.1f, %.1f, %.1f, %.1f \n", t[2].x, t[2].y, t[2].z, t[2].w);
		printf("%.1f, %.1f, %.1f, %.1f \n", t[3].x, t[3].y, t[3].z, t[3].w);
	}

	atomicMinFloat(&min.x, pos.x);
	atomicMinFloat(&min.y, pos.y);
	atomicMinFloat(&min.z, pos.z);
	atomicMaxFloat(&max.x, pos.x);
	atomicMaxFloat(&max.y, pos.y);
	atomicMaxFloat(&max.z, pos.z);
}