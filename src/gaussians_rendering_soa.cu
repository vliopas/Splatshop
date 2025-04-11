// used to compare gaussian_rendering.cu to a struct-of-arrays variation, but not intended for regular use. 

#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>

#define CUB_DISABLE_BF16_SUPPORT

#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000

namespace std{
	using size_t = ::size_t;
};

using namespace std;

#include <cuda_fp16.h>

#include "./libs/glm/glm/glm.hpp"

#include "utils.cuh"
#include "HostDeviceInterface.h"
#include "math.cuh"
#include "./libs/glm/glm/gtc/quaternion.hpp"

namespace cg = cooperative_groups;

using glm::i16vec2;
using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::quat;
using glm::dot;
using glm::transpose;
using glm::inverse;

constexpr int VIEWMODE_DESKTOP = 0;
constexpr int VIEWMODE_DESKTOP_VR = 1;
constexpr int VIEWMODE_IMMERSIVE_VR = 2;
// constexpr uint32_t BACKGROUND_COLOR = 0xff000000;
// constexpr uint32_t BACKGROUND_COLOR = 0xffffffff;
constexpr uint32_t BACKGROUND_COLOR = 0xff443322;
constexpr uint64_t DEFAULT_PIXEL = (uint64_t(Infinity) << 32) | BACKGROUND_COLOR;

constexpr float3 SPECTRAL[11] = {
	float3{158,1,66},
	float3{213,62,79},
	float3{244,109,67},
	float3{253,174,97},
	float3{254,224,139},
	float3{255,255,191},
	float3{230,245,152},
	float3{171,221,164},
	float3{102,194,165},
	float3{50,136,189},
	float3{94,79,162},
};

constexpr int dbgtile_x = 60;
constexpr int dbgtile_y = 50;
constexpr float basisvector_encoding_factor = 20.0f;

glm::i16vec2 encode_basisvector_i16vec2(vec2 basisvector){

	float length = glm::length(basisvector);
	float angle = atan2(basisvector.y, basisvector.x);

	int16_t ilength = clamp(length * basisvector_encoding_factor, 0.0f, 60'000.0f);
	int16_t iangle = angle * 10'000.0f;

	return {iangle, ilength};
}

vec2 decode_basisvector_i16vec2(glm::i16vec2 encoded){

	float length = float(encoded.y) / basisvector_encoding_factor;
	float angle = float(encoded.x) / 10'000.0f;

	float x = cos(angle);
	float y = sin(angle);

	return vec2{x, y} * length;
}

__half2 decode_basisvector_i16vec2_half2(glm::i16vec2 encoded){

	float length = float(encoded.y) / basisvector_encoding_factor;
	float angle = float(encoded.x) / 10'000.0f;

	float x = cos(angle) * length;
	float y = sin(angle) * length;

	return __half2{x, y};
}

#if defined(STAGEDATA_16BYTE)

	void encode_stagedata(StageData& stagedata, vec2 a, vec2 b, float depth){
		stagedata.basisvector1_encoded = encode_basisvector_i16vec2(a);
		// stagedata.basisvector2_encoded = encode_basisvector_i16vec2(b);

		vec2 b_encoded = encode_basisvector_i16vec2(b);
		stagedata.basisvector2_encoded = b_encoded.y;
		
		// stagedata.depth = depth;

		__half hdepth = depth;
		memcpy(&stagedata.depth_encoded, &hdepth, 2);
	}

	void decode_stagedata(StageData stagedata, vec2& a, vec2& b, float& depth){
		a = decode_basisvector_i16vec2(stagedata.basisvector1_encoded);
		// b = decode_basisvector_i16vec2(stagedata.basisvector2_encoded);

		float b_length = float(stagedata.basisvector2_encoded) / basisvector_encoding_factor;
		b = normalize(vec2{-a.y, a.x}) * b_length;

		// depth = stagedata.depth;

		__half hdepth;
		memcpy(&hdepth, &stagedata.depth_encoded, 2);
		depth = hdepth;
	}

#elif defined(STAGEDATA_20BYTE)

	void encode_stagedata(StageData& stagedata, vec2 a, vec2 b, float depth){
		stagedata.basisvector1_encoded = encode_basisvector_i16vec2(a);
		stagedata.basisvector2_encoded = encode_basisvector_i16vec2(b);

		stagedata.depth = depth;
	}

	void decode_stagedata(StageData stagedata, vec2& a, vec2& b, float& depth){
		a = decode_basisvector_i16vec2(stagedata.basisvector1_encoded);
		b = decode_basisvector_i16vec2(stagedata.basisvector2_encoded);

		depth = stagedata.depth;
	}

#elif defined(STAGEDATA_24BYTE)

	void encode_stagedata(StageData& stagedata, vec2 a, vec2 b, float depth){
		stagedata.basisvector1_encoded = encode_basisvector_i16vec2(a);
		stagedata.basisvector2_encoded = encode_basisvector_i16vec2(b);

		stagedata.depth = depth;
	}

	void decode_stagedata(StageData stagedata, vec2& a, vec2& b, float& depth){
		a = decode_basisvector_i16vec2(stagedata.basisvector1_encoded);
		b = decode_basisvector_i16vec2(stagedata.basisvector2_encoded);

		depth = stagedata.depth;
	}

#endif

// We need the same iteration logic for "touched" tiles in multiple kernels. 
template <typename Function>
void forEachTouchedTile(vec2 splatCoord, vec2 basisVector1, vec2 basisVector2, RenderTarget target, Function f){

	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	ivec2 tile_start = {
		(splatCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(splatCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));
	vec2 tileCoord = {
		splatCoord.x / TILE_SIZE_3DGS,
		splatCoord.y / TILE_SIZE_3DGS
	};

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	int tileFrags = 0;

	// float diag = 22.627416997969522f;  // sqrt(16.0f * 16.0f + 16.0f * 16.0f);
	float tileRadius = 11.313708498984761f; // sqrt(8.0f * 8.0f + 8.0f * 8.0f);
	for(int tile_x = tile_start.x; tile_x <= tile_end.x; tile_x++)
	for(int tile_y = tile_start.y; tile_y <= tile_end.y; tile_y++)
	{
		vec2 tilePos = {tile_x * 16.0f + 8.0f, tile_y * 16.0f + 8.0f}; 

		bool intersectsTile = intersection_circle_splat(
			tilePos, tileRadius, 
			splatCoord, 
			basisVector1, 
			basisVector2
		);

		intersectsTile = true;

		if(intersectsTile){
			f(tile_x, tile_y);
		}
	}
}

// stages the model for rasterization, meaning it creates tile fragments for each tile the splat overlaps, and adds it to the staging buffer.
// Much of the math here is from https://github.com/mkkellogg/GaussianSplats3D
extern "C" __global__
void kernel_stageSplats(
	CommonLaunchArgs args,
	RenderTarget target,
	ColorCorrection colorCorrection,
	GaussianData model,
	// out
	uint32_t* visibleSplatCounter,
	uint32_t* numTilefragments,
	uint32_t* numTilefragments_splatwise,
	float* staging_depth,

	// StageData* staging_data,
	i16vec2*  sd_basisvector1_encoded,
	i16vec2*  sd_basisvector2_encoded,
	i16vec2*  sd_imgPos_encoded,
	uint32_t* sd_color,
	uint32_t* sd_flags,
	float*    sd_depth,

	uint32_t* ordering
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	mat4 world = model.transform;
	mat4 view = target.view;
	mat4 proj = target.proj;
	mat4 worldView = view * world;
	mat4 worldViewProj = proj * view * world;

	vec3 splatPos = model.position[splatIndex];
	vec4 worldPos = world * vec4(splatPos, 1.0f);
	vec4 viewPos = view * worldPos;
	vec4 ndc = proj * viewPos;

	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;

	if(ndc.w <= 0.0f) return;
	if(ndc.x < -1.1f || ndc.x >  1.1f) return;
	if(ndc.y < -1.1f || ndc.y >  1.1f) return;

	vec4 color = model.color[splatIndex].normalized();

	vec4 quat = model.quaternion[splatIndex];

	// int8_t qw = quat.w * 127.0f;

	// // __half qx = __half(quat.x);
	// // __half qy = __half(quat.y);
	// // __half qz = __half(quat.z);
	// // __half qw = __half(quat.w);

	mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));

	mat3 scale = mat3(1.0f);
	scale[0][0] = model.scale[splatIndex].x;
	scale[1][1] = model.scale[splatIndex].y;
	scale[2][2] = model.scale[splatIndex].z;

	mat3 cov3D = rotation * scale * scale * transpose(rotation);

	// check lower bitrates
	// float* floats = (float*)&cov3D;
	// for(int i = 0; i < 9; i++){
	// 	floats[i] = __half(floats[i]);
	// 	// floats[i] = clamp(floats[i], -10.0f, 10.0f);
	// }

	// model.cov3d[index].m11 = cov3D[0][0];
	// model.cov3d[index].m12 = cov3D[0][1];
	// model.cov3d[index].m13 = cov3D[0][2];
	// model.cov3d[index].m22 = cov3D[1][1];
	// model.cov3d[index].m23 = cov3D[1][2];
	// model.cov3d[index].m33 = cov3D[2][2];

	// auto cov3Del = model.cov3d[index];
	// mat3 cov3D;
	// cov3D[0][0] = cov3Del.m11;
	// cov3D[0][1] = cov3Del.m12;
	// cov3D[0][2] = cov3Del.m13;

	// cov3D[1][0] = cov3Del.m12;
	// cov3D[1][1] = cov3Del.m22;
	// cov3D[1][2] = cov3Del.m23;

	// cov3D[2][0] = cov3Del.m13;
	// cov3D[2][1] = cov3Del.m23;
	// cov3D[2][2] = cov3Del.m33;

	vec2 focal = vec2(
		proj[0][0] * target.width * 0.5f,
		proj[1][1] * target.height * 0.5f
	);
	float s = 1.0f / (viewPos.z * viewPos.z);
	mat3 J = mat3(
		focal.x / viewPos.z   , 0.0f                  , -(focal.x * viewPos.x) * s,
		0.0f                  , focal.y / viewPos.z   , -(focal.y * viewPos.y) * s,
		0.0f                  , 0.0f                  , 0.0f
	);

	mat3 W = transpose(mat3(worldView));
	mat3 T = W * J;

	mat3 cov2Dm = transpose(T) * cov3D * T;
	cov2Dm[0][0] += 0.2f;
	cov2Dm[1][1] += 0.2f;

	vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

	float a = cov2Dv.x;
	float d = cov2Dv.z;
	float b = cov2Dv.y;
	float D = a * d - b * b;
	float trace = a + d;
	float traceOver2 = 0.5f * trace;
	float term2 = sqrt(max(0.05f, traceOver2 * traceOver2 - D));
	float eigenValue1 = traceOver2 + term2;
	float eigenValue2 = traceOver2 - term2;

	if(args.uniforms.cullSmallSplats){
		// clip tiny gaussians
		if(eigenValue1 < 0.05f) return;
		if(eigenValue2 < 0.05f) return;
	}else{
		eigenValue1 = max(eigenValue1, 0.0f);
		eigenValue2 = max(eigenValue2, 0.0f);
	}

	if(args.uniforms.makePoints){
		if(eigenValue1 < 0.105f) return;
		if(eigenValue2 < 0.105f) return;
	}

	vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
	vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

	float splatScale = args.uniforms.splatSize;
	const float sqrt8 = sqrt(8.0f);

	vec2 _basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), MAX_SCREENSPACE_SPLATSIZE);
	vec2 _basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), MAX_SCREENSPACE_SPLATSIZE);

	// We are using quantized basisvectors elsewhere, 
	// so we must make sure to also use them in this kernel to obtain matching results
	vec2 basisVector1, basisVector2;
	StageData tmp;
	float tmpdepth;
	encode_stagedata(tmp, _basisVector1, _basisVector2, tmpdepth);
	decode_stagedata(tmp, basisVector1, basisVector2, tmpdepth);

	// discard small splats with low opacity
	if(args.uniforms.cullSmallSplats)
	if(length(_basisVector1) < 5.0f)
	if(length(_basisVector2) < 5.0f)
	{
		if(color.a < 0.2f) return;
	}

	float depth = ndc.w;
	vec2 _pixelCoord = {
		((ndc.x) * 0.5f + 0.5f) * target.width,
		((ndc.y) * 0.5f + 0.5f) * target.height
	};

	glm::i16vec2 pixelCoord_encoded = _pixelCoord * 10.0f;
	vec2 pixelCoord = vec2(pixelCoord_encoded) / 10.0f;

	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	ivec2 tile_start = {
		(pixelCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(pixelCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_size = (tile_end - tile_start) + 1;

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));
	vec2 tileCoord = {
		pixelCoord.x / TILE_SIZE_3DGS,
		pixelCoord.y / TILE_SIZE_3DGS
	};

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	color = applyColorCorrection(color, colorCorrection);

	uint32_t flags = model.flags[splatIndex];

	{ // HANDLE FLAGS

		bool isSelected           = (flags & FLAGS_SELECTED) != 0;
		bool isDeleted            = (flags & FLAGS_DELETED) != 0;
		bool isHighlighted        = (flags & FLAGS_HIGHLIGHTED) != 0;
		bool isHighlightedNeg     = (flags & FLAGS_HIGHLIGHTED_NEGATIVE) != 0;

		auto rect = args.rectselect;
		if(isDeleted && isHighlighted && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			color.b = min(color.b * 2.0f + 0.4f, 1.0f);
			color.a = color.a / 4.0f;
		}else if(isDeleted){
			color.a = 0;
		}else if(isSelected && !isHighlightedNeg){
			color.r = min(color.r * 2.0f + 0.4f, 1.0f);
		}

		if(isHighlighted && (args.brush.mode == BRUSHMODE::ERASE)){
			color.r = min(color.r /  1.0f, 1.0f);
			color.g = min(color.g /  1.0f, 1.0f);
			color.b = min(color.b /  1.0f, 1.0f);
			color.a = min(color.a / 10.0f, 1.0f);
		}else if(isHighlighted && (args.brush.mode == BRUSHMODE::SELECT || rect.active)){
			color.r = min(color.r * 2.0f + 0.4f, 1.0f);
			color.g = min(color.g * 0.5f + 0.1f, 1.0f);
			color.b = min(color.b * 0.5f + 0.1f, 1.0f);
		}else if(isHighlighted && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			color.r = min(color.r * 0.5f + 0.1f, 1.0f);
			color.g = min(color.g * 0.5f + 0.1f, 1.0f);
			color.b = min(color.b * 2.0f + 0.4f, 1.0f);
		}
	}

	if(!model.writeDepth){
		flags = flags | FLAGS_DISABLE_DEPTHWRITE;
	}

	if((flags & FLAGS_DELETED) != 0) return;

	// TODO: check if its more efficient to run a separate kernel that 
	// initializes <numVisibleSplats> elements with ordering[index] = index;
	uint32_t visibleSplatID = atomicAdd(visibleSplatCounter, 1);
	ordering[visibleSplatID] = visibleSplatID;

	// StageData stuff;
	// stuff.imgPos_encoded       = pixelCoord_encoded;
	// stuff.flags                = flags;
	// stuff.color                = to32BitColor(color);
	
	// encode_stagedata(stuff, _basisVector1, _basisVector2, depth);

	staging_depth[visibleSplatID] = depth;
	// staging_data[visibleSplatID] = stuff;
	// i16vec2*  sd_basisvector1_encoded,
	// i16vec2*  sd_basisvector2_encoded,
	// i16vec2*  sd_imgPos_encoded,
	// uint32_t* sd_color,
	// uint32_t* sd_flags,
	// float*    sd_depth,
	sd_basisvector1_encoded[visibleSplatID] = encode_basisvector_i16vec2(_basisVector1);
	sd_basisvector2_encoded[visibleSplatID] = encode_basisvector_i16vec2(_basisVector2);
	sd_imgPos_encoded[visibleSplatID] = pixelCoord_encoded;
	sd_color[visibleSplatID] = to32BitColor(color);
	sd_flags[visibleSplatID] = flags;
	sd_depth[visibleSplatID] = depth;

	// Count tile fragments that each splat produces
	uint32_t tileFrags = 0;

	forEachTouchedTile(pixelCoord, basisVector1, basisVector2, target, [&](uint32_t tile_x, uint32_t tile_y){
		tileFrags++;
	});

	numTilefragments_splatwise[visibleSplatID] = tileFrags;
	atomicAdd(numTilefragments, tileFrags);
}


extern "C" __global__
void kernel_createTilefragmentArray(
	// input
	CommonLaunchArgs args,
	RenderTarget target,
	uint32_t* ordering, 
	uint32_t numStagedSplats,

	// StageData* stageDataArray,
	i16vec2*  sd_basisvector1_encoded,
	i16vec2*  sd_basisvector2_encoded,
	i16vec2*  sd_imgPos_encoded,
	uint32_t* sd_color,
	uint32_t* sd_flags,
	float*    sd_depth,

	// uint32_t* dbg_numTilefragments_ordered,
	uint32_t* prefixsum,
	uint32_t tileFragmentsCounter,
	// output
	uint32_t* tileIDs,
	uint32_t* splatIDs
){
	
	// index of unsorted, staged splats
	uint32_t index = cg::this_grid().thread_rank();

	if(index >= numStagedSplats) return;

	// index of depth-sorted splats
	uint32_t order = ordering[index];

	// load stagedata of splats in depth-sorted order
	// StageData stageData = stageDataArray[order];

	// vec2 basisVector1, basisVector2;
	vec2 basisVector1 = decode_basisvector_i16vec2(sd_basisvector1_encoded[order]);
	vec2 basisVector2 = decode_basisvector_i16vec2(sd_basisvector2_encoded[order]);
	vec2 imgPos_encoded = sd_imgPos_encoded[order];
	float depth = sd_depth[order];
	// decode_stagedata(stageData, basisVector1, basisVector2, depth);

	vec2 pixelCoord = vec2(imgPos_encoded) / 10.0f;

	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));

	ivec2 tile_start = {
		(pixelCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(pixelCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	int ltiles_x = (tile_end.x - tile_start.x) + 1;
	int ltiles_y = (tile_end.y - tile_start.y) + 1;
	int numTiles = ltiles_x * ltiles_y;

	uint32_t fragmentOffset = prefixsum[index];

	forEachTouchedTile(pixelCoord, basisVector1, basisVector2, target, [&](uint32_t tile_x, uint32_t tile_y){
		uint32_t tileID = tile_x + tile_y * tiles_x;

		tileIDs[fragmentOffset] = tileID;
		splatIDs[fragmentOffset] = order;

		fragmentOffset++;
	});
	
}

// this method uses the same approach as INRIA / Kerbl & Kopanas et al.
// rasterizer_impl method identifyTileRanges()
// https://github.com/graphdeco-inria/gaussian-splatting
extern "C" __global__
void kernel_computeTiles_method1(
	CommonLaunchArgs args, RenderTarget target, 
	uint32_t* tileIDs, uint32_t numFragments,
	int tile_size,
	// output
	Tile* tiles
){
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index >= numFragments) return;

	int tiles_x = (target.width + tile_size - 1) / tile_size;
	int tiles_y = (target.height + tile_size - 1) / tile_size;
	uint32_t numTiles = tiles_x * tiles_y;


	uint32_t tileID = tileIDs[index];

	if(tileID >= numTiles) return;

	if(index == 0){
		tiles[tileID].firstIndex = 0;
	}else{
		uint32_t prevTileID = tileIDs[index - 1];

		if(tileID != prevTileID){
			tiles[prevTileID].lastIndex = index - 1;
			tiles[tileID].firstIndex = index;
		}
	}

	if(index == numFragments - 1){
		tiles[tileID].lastIndex = index;
	}
}

constexpr int PREFETCH_COUNT = 128;

// #define DEBUGING
// #define DEBUG_MEASURE
// #define DEBUG_HEATMAP

// Padded to 64 bytes to avoid bank-conflicts. (16 banks of 4 byte words -> 64 bytes)
struct RasterizationData{
	vec2 dir1;
	vec2 dir2;
	vec2 imgpos;
	float depth;
	uint32_t flags;
	vec4 colors;
	uint32_t padding1;
	uint32_t padding2;
	uint32_t padding3;
	uint32_t padding4;
	// uint32_t padding5;
};

extern "C" __global__
void kernel_render_gaussians(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, 
	
	// StageData* stagedatas,
	i16vec2*  sd_basisvector1_encoded,
	i16vec2*  sd_basisvector2_encoded,
	i16vec2*  sd_imgPos_encoded,
	uint32_t* sd_color,
	uint32_t* sd_flags,
	float*    sd_depth,
	

	uint32_t pointsInTileThreshold
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();

	// constexpr int PREFETCH_COUNT = 128;

	uint64_t t_start = nanotime();

	int width     = target.width;
	int height    = target.height;
	int numPixels = width * height;
	int tiles_x   = int(width + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
	int tiles_y   = int(height + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
	int numTiles  = tiles_x * tiles_y;

	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];
	__shared__ float sh_remainingTranslucency[256];

	int tileID = grid.block_rank();
	
	Tile tile = tiles[tileID];

	// tile coordinates
	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;

	// this thread's pixel coordinate within tile
	int tilePixelIndex = threadIdx.x;
	int tile_pixel_x = tilePixelIndex % TILE_SIZE_3DGS;
	int tile_pixel_y = tilePixelIndex / TILE_SIZE_3DGS;
	// int tile_pixel_x = 0;
	// int tile_pixel_y = 0;
	// if(tilePixelIndex < 64){
	// 	tile_pixel_x = 0 + (tilePixelIndex -   0) % 8;
	// 	tile_pixel_y = 0 + (tilePixelIndex -   0) / 8;
	// }else if(tilePixelIndex < 128){
	// 	tile_pixel_x = 0 + (tilePixelIndex -  64) % 8;
	// 	tile_pixel_y = 8 + (tilePixelIndex -  64) / 8;
	// }else if(tilePixelIndex < 192){
	// 	tile_pixel_x = 8 + (tilePixelIndex - 128) % 8;
	// 	tile_pixel_y = 0 + (tilePixelIndex - 128) / 8;
	// }else{
	// 	tile_pixel_x = 8 + (tilePixelIndex - 192) % 8;
	// 	tile_pixel_y = 8 + (tilePixelIndex - 192) / 8;
	// }

	// This thread's pixel coordinates within framebuffer
	int pixel_x = tile_x * TILE_SIZE_3DGS + tile_pixel_x;
	int pixel_y = tile_y * TILE_SIZE_3DGS + tile_pixel_y;
	float fpixel_x = pixel_x;
	float fpixel_y = pixel_y;
	int pixelID = int(pixel_x + pixel_y * width);

	int numPointsInTile = clamp(tile.lastIndex - tile.firstIndex + 1, 0u, 1'000'000u);

	// if(numPointsInTile > pointsInTileThreshold) return;

	// __shared__ uint32_t dbg_numAccepted;
	// __shared__ uint32_t dbg_numRejected;

	// if(block.thread_rank() == 0){
	// 	atomicMax(&args.state->numSplatsInLargestTile, numPointsInTile);
	// 	dbg_numAccepted = 0;
	// 	dbg_numRejected = 0;
	// }

	int iterations = ((numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT);
	uint32_t numProcessed = 0;

	vec4 pixel = vec4{0.0f, 0.0f, 0.0f, 0.0f};
	float depth = Infinity;
	if(pixel_x < width && pixel_y < height){
		depth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}
	float remainingTranslucency = 1.0f;
	sh_remainingTranslucency[block.thread_rank()] = 1.0f;

	// iterate through all the splats
	// - 16x16 pixels, 256 threads (1 thread per pixel)
	// - <PREFETCH_COUNT> splats at a time
	for(int iteration = 0; iteration < iterations; iteration++){


		// if(block.thread_rank() == 0){
		// 	sh_numRejected = 0;
		// 	memset(&sh_numRejected_warpwise, 0, 4 * 8);
		// }
		// if(block.thread_rank() % 32 == 0){
		// 	sh_numRejected_warpwise[block.thread_rank() / 32] = 0;
		// }

		// block.sync();

		// if(iteration > 100) continue;
		// if(iteration > 10){
		// 	if(iteration % 2 == 0) continue;
		// }

		//-------------------------------------------
		//   MULTIWARP-START
		//-------------------------------------------

		// - load splats into shared memory
		// - each thread of the block loads one splat
		// - precompute stuff that is later used by all pixels/threads in tile
		int index = PREFETCH_COUNT * iteration + block.thread_rank();
		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT){


			#if defined(FRAGWISE_ORDERING)
				StageData stageData = stagedatas[tile.firstIndex + index];
			#else
				int splatIndex = indices[tile.firstIndex + index];
				// StageData stageData = stagedatas[splatIndex];
				StageData stageData;
				stageData.basisvector1_encoded  = sd_basisvector1_encoded[splatIndex];
				stageData.basisvector2_encoded  = sd_basisvector2_encoded[splatIndex];
				stageData.imgPos_encoded        = sd_imgPos_encoded[splatIndex];
				stageData.color                 = sd_color[splatIndex];
				stageData.flags                 = sd_flags[splatIndex];
				stageData.depth                 = sd_depth[splatIndex];


			#endif
			
			uint32_t C = stageData.color;

			vec2 basisvector1, basisvector2;
			float depth;
			decode_stagedata(stageData, basisvector1, basisvector2, depth);

			// TODO: not a normalized direction vector...rename?
			RasterizationData data;
			data.dir1 = basisvector1 / dot(basisvector1, basisvector1);
			data.dir2 = basisvector2 / dot(basisvector2, basisvector2);
			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
			data.depth = depth;
			data.flags = stageData.flags;
			data.colors = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				((C >> 24) & 0xff) / 255.0f,
			};

			sh_rasterdata[threadIdx.x] = data;
		}

		int splatsInSharedMemory = min(numPointsInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);
		
		block.sync();

		// now iterate with all threads of block through all splats.
		// i.e., all 256 threads process first splat, then all 256 threads process second, ...
		// uint64_t t_start_draw = nanotime();
		for(int i = 0; i < splatsInSharedMemory; i++)
		{
			auto rasterdata = sh_rasterdata[i];
			vec4 splatColor = rasterdata.colors;
			vec2 imgCoords = rasterdata.imgpos;

			numProcessed++;

			// if(splatColor.a < 0.003f) continue;

			vec2 pFrag = {fpixel_x - imgCoords.x, fpixel_y - imgCoords.y};

			float sT = dot(rasterdata.dir1, pFrag); 
			float sB = dot(rasterdata.dir2, pFrag); 
			float w = (sT * sT + sB * sB);

			if(rasterdata.depth > depth) break;
			if(pixel_x >= width || pixel_y >= height) break;
			
			// Splat boundary at w = 1.0. The remaining part of the gaussian is discarded.
			if(w < 1.0f)
			{
				// vec4 splatColor = rasterdata.colors;

				w = exp(-4.0f * w);
				
				// remainingTranslucency = 1.0f - pixel.w;

				if(args.uniforms.showRing && w < 0.025f){
					splatColor.a = 1.0f;
					w += 0.5f;
				}

				float alpha = w * (1.0f - pixel.w) * splatColor.a;

				pixel.r += alpha * splatColor.r;
				pixel.g += alpha * splatColor.g;
				pixel.b += alpha * splatColor.b;
				pixel.w += alpha;

				remainingTranslucency = 1.0f - pixel.w;
				
				// update depth buffer
				// if(remainingTranslucency < 8.0f / 255.0f)
				// if(remainingTranslucency < 0.030f)
				if(remainingTranslucency < 8.0f / 255.0f && (rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0)
				{
					depth = rasterdata.depth;
					uint64_t udepth = __float_as_uint(depth);
					uint64_t pixel = (udepth << 32) | 0x00000000;
					
					target.framebuffer[pixelID] = pixel;
					
					break;
				}
			}
		}

		sh_remainingTranslucency[block.thread_rank()] = remainingTranslucency;

		block.sync();
	}

	block.sync();

	// if(isDebugTile(tile_x, tile_y)){
	// 	// args.state->dbg.numAccepted = dbg_numAccepted;
	// 	// args.state->dbg.numRejected = dbg_numRejected;
	// 	if(block.thread_rank() == 0){
	// 		printf("numPointsInTile: %d\n", numPointsInTile);
	// 	}
	// }

	#if defined(DEBUG_MEASURE)
	float millies = float(nanotime() - t_start) / 1'000'000.0f;
	#endif

	// Transfer colors from register "pixel" to global memory. 
	if(pixel_x < width && pixel_y < height){
		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;

		if(args.uniforms.rendermode == RENDERMODE_COLOR){

			// color = target.framebuffer[pixelID] & 0xffffffff;

			if(args.uniforms.showTiles)
			if(tile_pixel_x == 15 || tile_pixel_y == 15)
			{
				pixel = pixel * 0.5f;
				pixel.w = 255.0f;
			}

			uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
			vec4 oldPixel = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				(C >> 24) & 0xff,
			};

			rgba[0] = clamp(pixel.r + remainingTranslucency * oldPixel.r, 0.0f, 255.0f) ;
			rgba[1] = clamp(pixel.g + remainingTranslucency * oldPixel.g, 0.0f, 255.0f) ;
			rgba[2] = clamp(pixel.b + remainingTranslucency * oldPixel.b, 0.0f, 255.0f) ;
			rgba[3] = 255;

			// if(millies > 2.0f){
			// 	color = 0xff0000ff;
			// }

			#if defined(DEBUG_HEATMAP)
			float w = log2(pow(float(numProcessed), 0.7f));
			uint32_t spectralIndex = clamp(int(w), 0, 10);

			// if(numProcessed > 1000)
			{
				float3 cf = SPECTRAL[10 - spectralIndex];
				rgba[0] = cf.x;
				rgba[1] = cf.y;
				rgba[2] = cf.z;
			}
			#endif

			uint64_t pixelValue = target.framebuffer[pixelID];
			pixelValue = pixelValue & 0xffffffff'00000000;
			pixelValue = pixelValue | color;


			target.framebuffer[pixelID] = pixelValue;
		}else if(args.uniforms.rendermode == RENDERMODE_DEPTH){

			vec3 SPECTRAL[11] = {
				vec3{158,1,66},
				vec3{213,62,79},
				vec3{244,109,67},
				vec3{253,174,97},
				vec3{254,224,139},
				vec3{255,255,191},
				vec3{230,245,152},
				vec3{171,221,164},
				vec3{102,194,165},
				vec3{50,136,189},
				vec3{94,79,162},
			};

			float w = depth / 3.0f;
			float u = w - floor(w);

			int i0 = w;

			vec3 C0 = SPECTRAL[clamp((i0 + 0), 0, 10) % 11];
			vec3 C1 = SPECTRAL[clamp((i0 + 1), 0, 10) % 11];

			vec3 C = (1.0f - u) * C0 + u * C1;

			rgba[0] = C.x;
			rgba[1] = C.y;
			rgba[2] = C.z;

			uint32_t udepth = __float_as_uint(depth);

			uint64_t pixel = (uint64_t(udepth) << 32) | color;

			target.framebuffer[pixelID] = pixel;
		}
	}

	// uint64_t nanos = nanotime() - t_start;
	// if(nanos > 5'500'000){
	// 	float millies = float(nanos) / 1'000'000.0f;

	// 	if(block.thread_rank() == 0){
	// 		printf("millies: %.1f ms\n", millies);
	// 	}
	// }
}


