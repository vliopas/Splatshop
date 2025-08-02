#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000
namespace std
{
    using size_t = ::size_t;
};
// =======================

#include <curand_kernel.h>
#include <cooperative_groups.h>

#include "libs/glm/glm/glm.hpp"
#include "libs/glm/glm/gtc/matrix_transform.hpp"
#include "libs/glm/glm/gtc/matrix_access.hpp"
#include "libs/glm/glm/gtx/transform.hpp"
#include "libs/glm/glm/gtc/quaternion.hpp"

#include "utils.cuh"

#include "HostDeviceInterface.h"
#include "raster_utils.cuh"

__device__ uint32_t numProcessedTriangles;

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
    RGBA(94, 79, 162)};

uint32_t sampleSpectral(float u)
{

    // u = clamp(u, 0.0f, 1.0f);
    u = fmodf(u, 1.0f);
    float i = u * 10.0f;

    int i_l = int(floor(i)) % 11;
    int i_u = int(ceil(i)) % 11;
    float w = fmodf(i, 1.0f);

    uint32_t a = SPECTRAL[i_l];
    uint32_t b = SPECTRAL[i_u];
    uint32_t sample = 0;

    uint8_t *a_rgba = (uint8_t *)&a;
    uint8_t *b_rgba = (uint8_t *)&b;
    uint8_t *s_rgba = (uint8_t *)&sample;

    s_rgba[0] = (1.0f - w) * float(a_rgba[0]) + w * float(b_rgba[0]);
    s_rgba[1] = (1.0f - w) * float(a_rgba[1]) + w * float(b_rgba[1]);
    s_rgba[2] = (1.0f - w) * float(a_rgba[2]) + w * float(b_rgba[2]);
    s_rgba[3] = 255;

    return sample;
}

inline vec4 toScreenCoord(vec3 p, mat4 &transform, int width, int height)
{
    vec4 pos = transform * vec4{p.x, p.y, p.z, 1.0f};

    pos.x = pos.x / pos.w;
    pos.y = pos.y / pos.w;

    vec4 imgPos = {
        (pos.x * 0.5f + 0.5f) * width,
        (pos.y * 0.5f + 0.5f) * height,
        pos.z,
        pos.w};

    return imgPos;
}

inline uint32_t computeColor(
    int triangleIndex,
    TriangleData triangles,
    TriangleMaterial material,
    Texture texture,
    float s, float t, float v)
{

    uint32_t color;
    uint8_t *rgb = (uint8_t *)&color;

    color = triangleIndex * 123456;
    // color = 0x0000ff00;

    // material.mode = MATERIAL_MODE_UVS;

    if (material.mode == MATERIAL_MODE_COLOR)
    {
        rgb[0] = 255.0f * material.color.r;
        rgb[1] = 255.0f * material.color.g;
        rgb[2] = 255.0f * material.color.b;
        rgb[3] = 255.0f * material.color.a;
    }
    else if (material.mode == MATERIAL_MODE_VERTEXCOLOR && triangles.colors != nullptr)
    {
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
    }
    else if (material.mode == MATERIAL_MODE_UVS && triangles.uv != nullptr)
    {
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
    }
    else if (material.mode == MATERIAL_MODE_TEXTURED && triangles.uv != nullptr)
    {

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

        if (texture.data)
        {

            auto sampleTexture = [&](vec2 uv, Texture texture)
            {
                int tx = int(uv.x * texture.width) % texture.width;
                int ty = int(uv.y * texture.height) % texture.height;
                // ty = texture.height - ty;

                int texelID = tx + texture.width * ty;

                if (texelID < 0)
                {
                    printf("uv:  %.2f, %.2f\n", uv.x, uv.y);
                    printf("texture %d, %d\n", texture.width, texture.height);
                    printf("test %d\n", texelID);
                }

                // if(triangleIndex == 0){
                // 	printf("texture.width: %d \n", texture.width);
                // }

                if (texelID < 0)
                    return 0xff0000ff;

                uint32_t r = texture.data[4 * texelID + 0];
                uint32_t g = texture.data[4 * texelID + 1];
                uint32_t b = texture.data[4 * texelID + 2];
                uint32_t a = texture.data[4 * texelID + 3];

                uint32_t color = (r << 0) | (g << 8) | (b << 16) | (a << 24);

                return color;
            };

            color = sampleTexture(uv, texture);
        }
        else if (texture.surface != -1)
        {

            auto sampleTexture = [&](vec2 uv, Texture texture)
            {
                int tx = int(uv.x * texture.width) % texture.width;
                int ty = int(uv.y * texture.height) % texture.height;

                uint32_t color = surf2Dread<uint32_t>(texture.surface, tx * 4, ty, cudaBoundaryModeClamp);

                return color;
            };

            color = sampleTexture(uv, texture);
        }
        else if (texture.cutexture != -1)
        {

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
        }
        else
        {
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
    }
    else
    {
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
    RenderTarget target)
{
    auto block = cg::this_thread_block();

    // mat4 rot = glm::rotate(3.1415f * 0.5f, vec3{0.0f, 1.0f, 0.0f});
    mat4 transform = target.proj * target.view * geometry.transform;

    __shared__ vec3 sh_positions[3 * TRIANGLES_PER_SWEEP];
    __shared__ vec2 sh_uvs[3 * TRIANGLES_PER_SWEEP];

    int numTrianglesInBlock = min(int(geometry.count) - triangleOffset, TRIANGLES_PER_SWEEP);

    if (numTrianglesInBlock <= 0)
        return;

    // load triangles into shared memory
    for (
        int i = block.thread_rank();
        i < numTrianglesInBlock;
        i += block.size())
    {
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
    for (
        int i = block.thread_rank();
        i < numTrianglesInBlock;
        i += block.size())
    {
        int triangleIndex = triangleOffset + i;

        vec3 v_0 = sh_positions[3 * i + 0];
        vec3 v_1 = sh_positions[3 * i + 1];
        vec3 v_2 = sh_positions[3 * i + 2];

        vec4 p_0 = toScreenCoord(v_0, transform, target.width, target.height);
        vec4 p_1 = toScreenCoord(v_1, transform, target.width, target.height);
        vec4 p_2 = toScreenCoord(v_2, transform, target.width, target.height);

        if (p_0.w < 0.0f || p_1.w < 0.0f || p_2.w < 0.0f)
            continue;

        vec2 v_01 = {p_1.x - p_0.x, p_1.y - p_0.y};
        vec2 v_02 = {p_2.x - p_0.x, p_2.y - p_0.y};

        auto cross = [](vec2 a, vec2 b)
        { return a.x * b.y - a.y * b.x; };

        { // backface culling
            float w = cross(v_01, v_02);
            if (w < 0.0)
                continue;
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

        if (numFragments > 40'000)
        {
            // uint32_t index = atomicAdd(&veryLargeTriangleCounter, 1);
            // veryLargeTriangleIndices[index] = triangleIndex;
            continue;
        }
        else if (numFragments > 4024)
        {
            // TODO: schedule block-wise rasterization
            // uint32_t index = atomicAdd(&largeTriangleSchedule.numTriangles, 1);
            // largeTriangleSchedule.indices[index] = triangleIndex;
            continue;
        }

        int numProcessedSamples = 0;
        for (int fragOffset = 0; fragOffset < numFragments; fragOffset += 1)
        {

            // safety mechanism: don't draw more than <x> pixels per thread
            if (numProcessedSamples > 4000)
                break;

            int fragID = fragOffset; // + block.thread_rank();
            int fragX = fragID % size_x;
            int fragY = fragID / size_x;

            vec2 pFrag = {
                floor(min_x) + float(fragX),
                floor(min_y) + float(fragY)};
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

            if (isInsideTriangle)
            {
                uint32_t color = computeColor(triangleIndex, geometry, material, material.texture, s, t, v);
                uint8_t *rgb = (uint8_t *)&color;

                // color = sampleSpectral(float(2 * triangleIndex) / float(geometry.count));
                // color = sampleSpectral(floor(11.0f * float(2 * triangleIndex) / float(geometry.count)) / 11.0f);

                float depth = v * p_0.w + s * p_1.w + t * p_2.w;
                uint64_t udepth = *((uint32_t *)&depth);
                uint64_t pixel = (udepth << 32ull) | color;

                atomicMin(&target.framebuffer[pixelID], pixel);
            }

            numProcessedSamples++;
        }
    }

    block.sync();
}

extern "C" __global__ void kernel_drawTriangleQueue(CommonLaunchArgs args, TriangleModelQueue queue, RenderTarget target)
{

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (grid.thread_rank() == 0)
    {
        numProcessedTriangles = 0;
    }

    __shared__ int sh_blockTriangleOffset;      // the global offset to the set of triangles that this block should render
    __shared__ int sh_blockLocalTriangleOffset; // the "local" offset relative to the current triangle queue element
    __shared__ int sh_triangleQueueIndex;       // the index of the current triangle queue element
    __shared__ TriangleData sh_geometry;
    __shared__ TriangleMaterial sh_material;

    if (block.thread_rank() == 0)
    {
        sh_blockTriangleOffset = 0;
        sh_blockLocalTriangleOffset = 0;
        sh_triangleQueueIndex = 0;
        sh_geometry = queue.geometries[0];
        sh_material = queue.materials[0];
    }

    grid.sync();

    while (true)
    {

        // Check which batch of triangles this block should render next.
        block.sync();
        if (block.thread_rank() == 0)
        {
            uint32_t next = atomicAdd(&numProcessedTriangles, TRIANGLES_PER_SWEEP);
            uint32_t diff = next - sh_blockTriangleOffset;

            sh_blockTriangleOffset = next;
            sh_blockLocalTriangleOffset += diff;

            // if((numProcessedTriangles / TRIANGLES_PER_SWEEP) % 1000 == 0){
            // 	printf("%8u, %8u \n", sh_blockTriangleOffset, sh_blockLocalTriangleOffset);
            // }

            // The next global triangle index may be multiple queued geometries ahead.
            // Let this block advance to the correct geometry
            while (sh_blockLocalTriangleOffset > sh_geometry.count)
            {
                sh_triangleQueueIndex++;

                if (sh_triangleQueueIndex >= queue.count)
                    break;

                sh_blockLocalTriangleOffset -= sh_geometry.count;

                sh_geometry = queue.geometries[sh_triangleQueueIndex];
                sh_material = queue.materials[sh_triangleQueueIndex];
            }
        }
        block.sync();

        if (sh_triangleQueueIndex >= queue.count)
            break;

        rasterizeTriangles_block(sh_geometry, sh_material, sh_blockLocalTriangleOffset, args, target);
    }
}


inline bool fetchNextTriangleBatch(
    TriangleModelQueue queue,
    TriangleBatchInfo &state)
{
    uint32_t next = atomicAdd(&numProcessedTriangles, TRIANGLES_PER_SWEEP);
    uint32_t diff = next - state.blockTriangleOffset;

    state.blockTriangleOffset = next;
    state.blockLocalTriangleOffset += diff;

    while (state.blockLocalTriangleOffset >= state.geometry.count)
    {
        state.triangleQueueIndex++;

        if (state.triangleQueueIndex >= queue.count)
            return false; // no more geometry to process

        state.blockLocalTriangleOffset -= state.geometry.count;

        state.geometry = queue.geometries[state.triangleQueueIndex];
        state.material = queue.materials[state.triangleQueueIndex];
    }

    return true;
}

inline Stage chooseStage(
    TriangleModelQueue &queue,
    TriangleBatchInfo &state,
    int2 binRange,
    int &rasterBinID)
{
    // if enough work for rasterization -> prioritize this
    uint32_t maxOccupancy = 0;

    for (int binID = binRange.x; binID < binRange.y; ++binID)
    {
        uint32_t occupancy = binQueues[binID].size();
        if (occupancy > maxOccupancy)
        {
            maxOccupancy = occupancy;
            rasterBinID = binID;
        }
    }

    // for multiple bins per block, select the one with most triangles enqueued
    if (maxOccupancy > TRIANGLES_PER_SWEEP)
        return Stage::RASTERIZATION; // enough work gathered -> rasterize

    // we can now move to binning stage
    // if reprocessing a previous batch is needed we will specify that
    if (state.reprocessTriangleBatch)
        return Stage::BINNING;

    // otherwise we will fetch the next triangle batch
    // while also checking if there is more geometry to process
    bool geometryWorkLeft = fetchNextTriangleBatch(queue, state);
    if (geometryWorkLeft)
        return Stage::BINNING;

    // if no geometry left to process, we can finish
    // rasterizing whatever triangles are left in the bin queues
    if (maxOccupancy > 0)
        return Stage::RASTERIZATION;

    return Stage::DONE;
}

inline void trianglesToBinQueues(
    TriangleBatchInfo &state,
    CommonLaunchArgs args,
    RenderTarget target)
{
    auto block = cg::this_thread_block();
    auto grid = cg::this_grid();

    auto &geometry = state.geometry;
    int triangleOffset = state.blockLocalTriangleOffset;

    mat4 transform = target.proj * target.view * geometry.transform;

    __shared__ vec3 sh_positions[3 * TRIANGLES_PER_SWEEP];
    __shared__ struct{ int count; BinGroup buf[TRIANGLES_PER_SWEEP]; } binGroups;
    __shared__ uint16_t writeIndicesPerBin[MAX_NUMBER_OF_BINS]; // indices for each bin
    __shared__ uint32_t reprocessTriangleBatch;                 // flag (bitset) to for tris in batch to reprocess
    __shared__ uint32_t reprocessTriangleBatchOld;

    int numTrianglesInBlock = min(int(geometry.count) - triangleOffset, TRIANGLES_PER_SWEEP);

    if (numTrianglesInBlock <= 0)
        return;

    if (block.thread_rank() == 0)
    {
        binGroups.count = state.reprocessTriangleBatch ? binGroups.count : 0;
        reprocessTriangleBatchOld = state.reprocessTriangleBatch ? reprocessTriangleBatch : 0;
        reprocessTriangleBatch = 0; // reset reprocess flag
    }

    // load triangles into shared memory
    if (!state.reprocessTriangleBatch) // if we are working with the same batch - no need to fetch again
        for (
            int i = block.thread_rank();
            i < numTrianglesInBlock;
            i += block.size())
        {
            int triangleIndex = triangleOffset + i;
            sh_positions[3 * i + 0] = geometry.position[3 * triangleIndex + 0];
            sh_positions[3 * i + 1] = geometry.position[3 * triangleIndex + 1];
            sh_positions[3 * i + 2] = geometry.position[3 * triangleIndex + 2];
        }

    for (int i = block.thread_rank(); i < totalBins; i += block.size())
        writeIndicesPerBin[i] = 0xffffu; // reset write indices for each bin

    block.sync();
    int binsPerRow = (target.width + BIN_SIZE - 1) / BIN_SIZE;
    int binsPerCol = (target.height + BIN_SIZE - 1) / BIN_SIZE;

    if (!state.reprocessTriangleBatch)
        for (
            int i = block.thread_rank();
            i < numTrianglesInBlock;
            i += block.size())
        {
            int triangleIndex = triangleOffset + i;

            vec3 v_0 = sh_positions[3 * i + 0];
            vec3 v_1 = sh_positions[3 * i + 1];
            vec3 v_2 = sh_positions[3 * i + 2];

            vec4 p_0 = toScreenCoord(v_0, transform, target.width, target.height);
            vec4 p_1 = toScreenCoord(v_1, transform, target.width, target.height);
            vec4 p_2 = toScreenCoord(v_2, transform, target.width, target.height);

            if (p_0.w < 0.0f || p_1.w < 0.0f || p_2.w < 0.0f)
                continue;

            vec2 v_01 = {p_1.x - p_0.x, p_1.y - p_0.y};
            vec2 v_02 = {p_2.x - p_0.x, p_2.y - p_0.y};

            auto cross = [](vec2 a, vec2 b)
            { return a.x * b.y - a.y * b.x; };

            { // backface culling
                float w = cross(v_01, v_02);
                if (w < 0.0)
                    continue;
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

            if (numFragments <= 4024)
                continue;
            else // large triangles between 4024 and 40000 fragments^2
            {
                // take a grid of bins that covers the triangle Bounding Box
                // each bin is of size BIN_SIZE x BIN_SIZE pixels
                // bins are stored in binGroups - a grid of Height x Width bins
                int bx0 = int(min_x) / BIN_SIZE;
                int by0 = int(min_y) / BIN_SIZE;
                int bx1 = int(max_x) / BIN_SIZE;
                int by1 = int(max_y) / BIN_SIZE;

                // update bin queue
                uint32_t slot = atomicAdd(&binGroups.count, 1);
                uint8_t bw = uint8_t(min(bx1 - bx0 + 1, binsPerRow - bx0));
                uint8_t bh = uint8_t(min(by1 - by0 + 1, binsPerCol - by0));

                binGroups.buf[slot] = {(uint16_t)i,
                                    (uint16_t)bx0, (uint16_t)by0,
                                    bw, bh};
            }
        }

    block.sync(); // wait for bin groups to be fully populated

    for (uint8_t i = 0; i < binGroups.count; ++i)
    {
        const BinGroup &bg = binGroups.buf[i];
        int numOfBins = bg.binW * bg.binH;

        bool reprocessFlag = (reprocessTriangleBatchOld & (1u << bg.triangleIndex)) != 0;
        if (state.reprocessTriangleBatch && !reprocessFlag)
            continue;

        TriangleRef triRef;
        triRef.pack(
            static_cast<uint32_t>(state.triangleQueueIndex),
            static_cast<uint32_t>(triangleOffset + bg.triangleIndex));

        for (uint16_t binIdx = block.thread_rank(); binIdx < numOfBins; binIdx += block.size())
        {
            int binX = bg.binX + binIdx % bg.binW;
            int binY = bg.binY + binIdx / bg.binW;
            int globalBinID = binX + binY * binsPerRow;
            int index = binQueues[globalBinID].push(triRef);
            if (index < 0)
            {
                // bin is full, we need to reprocess this batch
                atomicOr(&reprocessTriangleBatch, 1u << bg.triangleIndex);
                break;
            }
            writeIndicesPerBin[globalBinID] = (uint16_t)index; // if -1 it will become 0xffff so its fine
        }
        block.sync();

        for (uint16_t binIdx = block.thread_rank(); binIdx < totalBins; binIdx += block.size())
        {
            auto writeIdx = writeIndicesPerBin[binIdx];
            if (writeIdx != 0xffffu)
            {
                binQueues[binIdx][writeIdx].setState(((reprocessTriangleBatch & (1u << bg.triangleIndex)) != 0)
                ? TriangleRef::INVALID : TriangleRef::VALID);
                assert(!binQueues[binIdx][writeIdx].isTentative());
                writeIndicesPerBin[binIdx] = 0xffffu; // reset write index for this bin to ready for next iteration
            }
        }
        
        block.sync();
    }

    if (block.thread_rank() == 0)
        state.reprocessTriangleBatch = reprocessTriangleBatch > 0; // set the reprocess flag if any triangle needs reprocessing
}

inline void rasterizeBin(
    TriangleModelQueue &queue,
    CommonLaunchArgs args,
    RenderTarget target,
    int binID)
{
    auto block = cg::this_thread_block();
    auto grid = cg::this_grid();

    mat4 transform = target.proj * target.view; // * geometry.transform

    auto binPixelOrigin = [=](int binID)
    {
        int binsPerRow = (target.width + BIN_SIZE - 1) / BIN_SIZE;
        int binX = binID % binsPerRow;
        int binY = binID / binsPerRow;

        return int2{binX * BIN_SIZE, binY * BIN_SIZE};
    };

    __shared__ TriangleRef triRefs[TRIANGLES_PER_SWEEP];
    __shared__ vec4 sh_positions[3 * TRIANGLES_PER_SWEEP];
    __shared__ vec2 sh_uvs[3 * TRIANGLES_PER_SWEEP];
    __shared__ uint64_t tileMasks[TRIANGLES_PER_SWEEP];
    __shared__ int numTrianglesToRasterize;
    __shared__ int numOfTilesToRasterize;
    __shared__ uint8_t tilesToRasterize[TRIANGLES_PER_SWEEP * TILE_SIZE * TILE_SIZE];
    __shared__ uint8_t tileMaskCounts[TRIANGLES_PER_SWEEP];
    __shared__ int tileMaskCountPrefixSum[TRIANGLES_PER_SWEEP];
    __shared__ int binPixX, binPixY; // pixel coordinates of the bin's top-left corner

    if (block.thread_rank() == 0)
    {
        int2 binOrigin = binPixelOrigin(binID);
        binPixX = binOrigin.x;
        binPixY = binOrigin.y;

        numTrianglesToRasterize = 0; // reset triangle count
        numOfTilesToRasterize = 0; // reset tile count
        
        uint32_t queueSize = binQueues[binID].size();
        uint32_t readIdx = binQueues[binID].getReadIndex();

        int i = TRIANGLES_PER_SWEEP;
        for (; numTrianglesToRasterize < queueSize && numTrianglesToRasterize < TRIANGLES_PER_SWEEP; ++numTrianglesToRasterize)
        {
            const TriangleRef& tri = binQueues[binID][readIdx + numTrianglesToRasterize];
            if (tri.isTentative()) break;
        }
    }

    block.sync(); // ensure all threads have the bin pixel origin

    if (numTrianglesToRasterize <= 0)
        return; // no triangles to rasterize in this bin

    for (uint16_t i = block.thread_rank(); i < numTrianglesToRasterize; i += block.size())
    {
        TriangleRef triRef = binQueues[binID].pop();
        triRefs[i] = triRef; // store the triangle reference for later use

        uint32_t triangleIndex = triRef.getTriangleId();
        uint32_t geometryId = triRef.getGeometryId();

        if (!triRef.isValid())
        {
            tileMasks[i] = 0ull; // triangle invalid, set mask to 0 - no tiles to rasterize
            continue;            // skip invalid triangles
        }

        TriangleData &geometry = queue.geometries[geometryId];
        TriangleMaterial &material = queue.materials[geometryId];

        sh_positions[3 * i + 0] = vec4(geometry.position[3 * triangleIndex + 0], 1.0f);
        sh_positions[3 * i + 1] = vec4(geometry.position[3 * triangleIndex + 1], 1.0f);
        sh_positions[3 * i + 2] = vec4(geometry.position[3 * triangleIndex + 2], 1.0f);

        // store screen space positions in shared mem since they're being reused later
        mat4 geomTransform = transform * geometry.transform;
        sh_positions[3 * i + 0] = toScreenCoord((vec3)sh_positions[3 * i + 0], geomTransform, target.width, target.height);
        sh_positions[3 * i + 1] = toScreenCoord((vec3)sh_positions[3 * i + 1], geomTransform, target.width, target.height);
        sh_positions[3 * i + 2] = toScreenCoord((vec3)sh_positions[3 * i + 2], geomTransform, target.width, target.height);

        sh_uvs[3 * i + 0] = geometry.uv[3 * triangleIndex + 0];
        sh_uvs[3 * i + 1] = geometry.uv[3 * triangleIndex + 1];
        sh_uvs[3 * i + 2] = geometry.uv[3 * triangleIndex + 2];

        const vec4 *tri = &sh_positions[3 * i];
        vec3 A, B, C;
        tri_edge_coeffs_conservative(tri, A, B, C);

        uint64_t mask = 0ull;
        for (uint8_t ty = 0; ty < TILES_PER_BIN; ++ty)
        {
            mask |= uint64_t(row_span_mask(A, B, C, binPixX, binPixY, ty)) << (ty * 8);
        }

        tileMasks[i] = mask;
    }
    block.sync(); // ensure all triangles are loaded and masks computed

    constexpr uint16_t lanesPerMask = TILE_SIZE * TILE_SIZE;
    const uint8_t warpId = (block.thread_rank() >> 5) & 1;
    const uint8_t warpLane = block.thread_rank() & 31; // 0..31
    const uint16_t masksPerBlock = block.size() / lanesPerMask;
    for (uint16_t maskIdx = 0; maskIdx < numTrianglesToRasterize; maskIdx += masksPerBlock)
    {
        uint64_t mask = __shfl_sync(0xFFFFFFFFu, tileMasks[maskIdx], 0);

        uint8_t bitIndex = warpLane + warpId * 32; // bits per mask
        bool isBitSet = mask & (1ull << bitIndex);

        if (isBitSet)
        {
            int slot = atomicAdd(&numOfTilesToRasterize, 1); // increment the number of tiles to rasterize
            tilesToRasterize[slot] = bitIndex;               // store the tile index
        }

        block.sync();
    }

    for (uint16_t i = block.thread_rank(); i < numTrianglesToRasterize; i += block.size())
    {
        uint64_t mask = tileMasks[i];
        uint8_t count = __popcll(mask); // count the number of bits set in the mask
        tileMaskCounts[i] = count;
    }
    block.sync();

    // compute (exclusive) prefix sum of tile mask counts
    prefixSum(tileMaskCounts, numTrianglesToRasterize, tileMaskCountPrefixSum);

    block.sync();
    int binsPerRow = target.width / BIN_SIZE;

    // rasterize triangles
    for (uint16_t i = block.thread_rank(); i < numOfTilesToRasterize; i += block.size())
    {
        using bool3 = glm::bvec3;
        using float3 = glm::vec3;

        uint16_t triSweepIdx = find_bin_group_idx(i, tileMaskCountPrefixSum, numTrianglesToRasterize);
        TriangleRef triRef = triRefs[triSweepIdx];

        uint32_t triangleIndex = triRef.getTriangleId();
        uint32_t geometryId = triRef.getGeometryId();
        TriangleData &geometry = queue.geometries[geometryId];
        TriangleMaterial &material = queue.materials[geometryId];

        uint8_t tileIndex = tilesToRasterize[i];
        uint8_t tileX = tileIndex % TILE_SIZE;
        uint8_t tileY = tileIndex / TILE_SIZE;

        // compute the screen-space bounding rectangle of the tile
        int min_y = clamp(binPixY + tileY * TILE_SIZE, 0, target.height);
        int min_x = clamp(binPixX + tileX * TILE_SIZE, 0, target.width);
        int max_y = clamp(min_y + TILE_SIZE, 0, target.height);
        int max_x = clamp(min_x + TILE_SIZE, 0, target.width);

        const vec4 *tri = &sh_positions[3 * triSweepIdx];
        vec3 A, B, C;
        tri_edge_coeffs(tri, A, B, C);

        bool3 horiz = glm::equal(A, float3(0.0f));
        bool3 pos = glm::greaterThan(A, float3(0.0f));
        bool3 neg = glm::lessThan(A, float3(0.0f));

        float3 invA = {A.x == 0 ? 1e8 : 1.0f / A.x, A.y == 0 ? 1e8 : 1.0f / A.y, A.z == 0 ? 1e8 : 1.0f / A.z};
        float invArea = 1.0f / (B.x * A.z - A.x * B.z);

        vec2 sample = {(float)min_x - tri[0].x, (float)min_y - tri[0].y};
        float3 CY = B * float(min_y) + C;

        // precompute these once outside the loop
        float ds_dx = A.z * invArea;
        float dt_dx = A.x * invArea;

        float ds_dy = B.z * invArea;
        float dt_dy = B.x * invArea;

        float s = sample.y * ds_dy;
        float t = sample.y * dt_dy;

        for (int y = min_y; y < max_y; y++, CY += B, sample.y++, s += ds_dy, t += dt_dy)
        {
            bool3 outsideH = bool3(horiz.x && (CY.x < 0.0f), horiz.y && (CY.y < 0.0f), horiz.z && (CY.z < 0.0f));
            if (glm::any(outsideH))
                continue; // skip rows outside the triangle

            float3 cross = -(CY * invA);

            float lower = glm::compMax(glm::mix(float3(-FLT_MAX), cross, pos));
            float upper = glm::compMin(glm::mix(float3(FLT_MAX), cross, neg));
            if (lower > upper)
                continue; // skip rows outside the triangle

            int ix0 = int(floorf(lower));
            int ix1 = int(ceilf(upper));

            ix0 = clamp(ix0, (int)min_x, (int)max_x); // clamp within tile width
            ix1 = clamp(ix1, (int)min_x, (int)max_x);

            sample.x = ix0 - tri[0].x;
            s += sample.x * ds_dx;
            t += sample.x * dt_dx;

            for (int x = ix0; x < ix1; x++, sample.x++, s += ds_dx, t += dt_dx)
            {
                float v = 1.0f - s - t;

                uint32_t color = computeColor(triangleIndex, geometry, material, material.texture, s, t, v);
                uint8_t *rgb = (uint8_t *)&color;

                int2 pixelCoords = make_int2(x, y);
                int pixelID = pixelCoords.x + pixelCoords.y * target.width;

                float depth = v * tri[0].w + s * tri[1].w + t * tri[2].w;
                uint64_t udepth = *((uint32_t *)&depth);
                uint64_t pixel = (udepth << 32ull) | color;

                atomicMin(&target.framebuffer[pixelID], pixel);
            }

            s -= sample.x * ds_dx;
            t -= sample.x * dt_dx;
        }
    }
    block.sync();
}

inline void cleanup(int2 binRange, TriangleModelQueue &queue, CommonLaunchArgs args, RenderTarget target)
{
    auto grid = cg::this_grid();
    grid.sync();

    // After all triangles have been processed, we can rasterize any remaining triangles in the bin queues
    // This is done because, due to contention, a block might detect no work to do after geometry depletes,
    // while other blocks might still be processing triangles to send to the respective bin queues
    bool anyWorkDone;
    do 
    {
        anyWorkDone = false;

        for (int i = binRange.x; i < binRange.y; i++) {
            rasterizeBin(queue, args, target, i);  // modify to return true if work was done
            anyWorkDone |= binQueues[i].size() > 0; // check if there are still triangles to rasterize in this bin
        }
    } 
    while (anyWorkDone);
}

extern "C" __global__ void kernel_persistentDraw(CommonLaunchArgs args, TriangleModelQueue queue, RenderTarget target)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (grid.thread_rank() == 0)
    {
        numProcessedTriangles = 0;

        int binsX = (target.width + BIN_SIZE - 1) / BIN_SIZE;
        int binsY = (target.height + BIN_SIZE - 1) / BIN_SIZE;
        totalBins = binsX * binsY;
    }
    grid.sync();

    __shared__ TriangleBatchInfo sh_state;
    __shared__ int sh_rasterBinID;
    __shared__ int2 sh_binRange;
    __shared__ Stage sh_stage;

    if (block.thread_rank() == 0)
    {
        sh_state.blockTriangleOffset = 0;
        sh_state.blockLocalTriangleOffset = 0;
        sh_state.triangleQueueIndex = 0;
        sh_state.reprocessTriangleBatch = false;
        sh_state.geometry = queue.geometries[0];
        sh_state.material = queue.materials[0];

        sh_binRange = computeBinRange();

        for (int i = sh_binRange.x; i < sh_binRange.y; i++)
            binQueues[i].clear();
    }
    grid.sync();

    while (true)
    {
        if (block.thread_rank() == 0)
        {
            sh_rasterBinID = -1;
            sh_stage = chooseStage(queue, sh_state, sh_binRange, sh_rasterBinID);
        }

        block.sync();
        switch (sh_stage)
        {
            case (Stage::BINNING):
            {
                trianglesToBinQueues(sh_state, args, target);
                continue;
            }
            case (Stage::RASTERIZATION):
            {
                rasterizeBin(queue, args, target, sh_rasterBinID);
                continue;
            }
            case (Stage::DONE):
            {
                cleanup(sh_binRange, queue, args, target); // cleanup necessary for contention reasons in bin queues
                return;
            }
        }
    }
}

extern "C" __global__ void kernel_compute_boundingbox(CommonLaunchArgs args, TriangleData model, vec3 &min, vec3 &max)
{

    int index = cg::this_grid().thread_rank();

    if (index >= model.count)
        return;

    vec3 pos = vec3(model.transform * vec4(model.position[index], 1.0f));

    if (index == 0)
    {
        float *floats = &model.transform[0].x;
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

inline void trianglesToBins(
    TriangleBatchInfo &state,
    CommonLaunchArgs args,
    RenderTarget target,
    BinElement* binElementQueue,
    uint32_t* binElementQueueCounter)
{
    auto block = cg::this_thread_block();
    auto grid = cg::this_grid();

    auto &geometry = state.geometry;
    int triangleOffset = state.blockLocalTriangleOffset;
    mat4 transform = target.proj * target.view * geometry.transform;

    __shared__ vec3 sh_positions[3 * TRIANGLES_PER_SWEEP];
    __shared__ struct{ int count; BinGroup buf[TRIANGLES_PER_SWEEP]; } binGroups;
    __shared__ int binPrefixSum[TRIANGLES_PER_SWEEP];
    __shared__ int binsPerRow;
    __shared__ int binsPerCol;
    
    int numTrianglesInBlock = min(int(geometry.count) - triangleOffset, TRIANGLES_PER_SWEEP);

    if (numTrianglesInBlock <= 0)
        return;

    if(block.thread_rank() == 0)
    {
        binsPerRow = (target.width + BIN_SIZE - 1) / BIN_SIZE;
        binsPerCol = (target.height + BIN_SIZE - 1) / BIN_SIZE;
        binGroups.count = 0;
    }

    for (
    int i = block.thread_rank();
    i < numTrianglesInBlock;
    i += block.size())
    {
        int triangleIndex = triangleOffset + i;
        sh_positions[3 * i + 0] = geometry.position[3 * triangleIndex + 0];
        sh_positions[3 * i + 1] = geometry.position[3 * triangleIndex + 1];
        sh_positions[3 * i + 2] = geometry.position[3 * triangleIndex + 2];
    }
    block.sync();

    for (
    int i = block.thread_rank();
    i < numTrianglesInBlock;
    i += block.size())
    {
        int triangleIndex = triangleOffset + i;

        vec3 v_0 = sh_positions[3 * i + 0];
        vec3 v_1 = sh_positions[3 * i + 1];
        vec3 v_2 = sh_positions[3 * i + 2];

        vec4 p_0 = toScreenCoord(v_0, transform, target.width, target.height);
        vec4 p_1 = toScreenCoord(v_1, transform, target.width, target.height);
        vec4 p_2 = toScreenCoord(v_2, transform, target.width, target.height);

        if (p_0.w < 0.0f || p_1.w < 0.0f || p_2.w < 0.0f)
            continue;

        vec2 v_01 = {p_1.x - p_0.x, p_1.y - p_0.y};
        vec2 v_02 = {p_2.x - p_0.x, p_2.y - p_0.y};

        auto cross = [](vec2 a, vec2 b)
        { return a.x * b.y - a.y * b.x; };

        { // backface culling
            float w = cross(v_01, v_02);
            if (w < 0.0)
                continue;
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

        if (numFragments <= 4024)
            continue;
        else // large triangles between 4024 and 40000 fragments^2
        {
            // take a grid of bins that covers the triangle Bounding Box
            // each bin is of size BIN_SIZE x BIN_SIZE pixels
            // bins are stored in binGroups - a grid of Height x Width bins
            int bx0 = int(min_x) / BIN_SIZE;
            int by0 = int(min_y) / BIN_SIZE;
            int bx1 = int(max_x) / BIN_SIZE;
            int by1 = int(max_y) / BIN_SIZE;

            // update bin queue
            uint32_t slot = atomicAdd(&binGroups.count, 1);
            uint8_t bw = uint8_t(min(bx1 - bx0 + 1, binsPerRow - bx0));
            uint8_t bh = uint8_t(min(by1 - by0 + 1, binsPerCol - by0));

            binGroups.buf[slot] = {(uint16_t)i,
                                (uint16_t)bx0, (uint16_t)by0,
                                bw, bh};
        }
    }

    block.sync();

    // prefix sum on binGroups so that each bin can be processed by a single thread 
    // needed for bin processing later 
    prefixSum(binGroups.buf, binGroups.count, binPrefixSum); // exclusive prefix sum
	 
	int binShift = 0; 
    int numOfBins = 0; 
    if (binGroups.count > 0)  
    { 
        const BinGroup& lastGroup = binGroups.buf[binGroups.count - 1]; 
        numOfBins = binPrefixSum[binGroups.count - 1] + lastGroup.binH * lastGroup.binW; 
    }
    
    __shared__ uint32_t slot;
    if(block.thread_rank() == 0) slot = atomicAdd(binElementQueueCounter, numOfBins);
    block.sync();

    for(uint16_t binIdx = block.thread_rank(); binIdx < numOfBins; binIdx += block.size())
    {
        uint16_t binGroupIndex = find_bin_group_idx(binIdx, binPrefixSum, binGroups.count); 
        uint16_t localBinIndex = binIdx - binPrefixSum[binGroupIndex]; 
        const BinGroup bg = binGroups.buf[binGroupIndex]; // 4‑byte read

        uint16_t binX = bg.binX + localBinIndex % bg.binW;
        uint16_t binY = bg.binY + localBinIndex / bg.binW;
        uint16_t globalBinID = binX + binY * binsPerRow;

        TriangleRef triRef;
        triRef.pack(
            static_cast<uint32_t>(state.triangleQueueIndex),
            static_cast<uint32_t>(triangleOffset + bg.triangleIndex),
            TriangleRef::VALID);

        binElementQueue[slot + binIdx] = BinElement{triRef, globalBinID};
    }
}

extern "C" __global__ void kernel_storeBins(
    CommonLaunchArgs args, 
    TriangleModelQueue queue, 
    RenderTarget target,
    BinElement* binElementQueue,
    uint32_t* binElementQueueCounter
){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    if (grid.thread_rank() == 0) numProcessedTriangles = 0;
    grid.sync();

    __shared__ TriangleBatchInfo sh_state;
    __shared__ bool sh_geometryLeft;

    if(block.thread_rank() == 0)
    {
        sh_state.blockTriangleOffset = 0;
        sh_state.blockLocalTriangleOffset = 0;
        sh_state.triangleQueueIndex = 0;
        sh_state.reprocessTriangleBatch = false;
        sh_state.geometry = queue.geometries[0];
        sh_state.material = queue.materials[0];

        sh_geometryLeft = true;
    }

    while(true)
    {
        block.sync();
        
        if(block.thread_rank() == 0)
            sh_geometryLeft = fetchNextTriangleBatch(queue, sh_state);

        block.sync();

        if(!sh_geometryLeft) break;
        
        trianglesToBins(sh_state, args, target, binElementQueue, binElementQueueCounter);
    }

}

extern "C" __global__ void kernel_rasterizeBins(
    CommonLaunchArgs args, 
    TriangleModelQueue queue, 
    RenderTarget target,
    BinElement* binElementQueue,
    uint32_t* binElementQueueCounter,
    uint32_t* binElementProcessedCounter
){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    constexpr int BINS_PER_SWEEP = 32;

    mat4 transform = target.proj * target.view; // * geometry.transform

    auto binPixelOrigin = [=](int binID)
    {
        int binsPerRow = (target.width + BIN_SIZE - 1) / BIN_SIZE;
        int binX = binID % binsPerRow;
        int binY = binID / binsPerRow;

        return int2{binX * BIN_SIZE, binY * BIN_SIZE};
    };

    __shared__ int binsToRasterize;
    __shared__ int readStartIdx;
    
    __shared__ vec4 sh_positions[3 * BINS_PER_SWEEP];
    __shared__ vec2 sh_uvs[3 * BINS_PER_SWEEP];
    __shared__ BinElement binElements[BINS_PER_SWEEP];
    __shared__ uint64_t tileMasks[BINS_PER_SWEEP];
    __shared__ uint8_t tileMaskCounts[BINS_PER_SWEEP];

    __shared__ int tilesToRasterize;
    __shared__ uint8_t tileElements[BINS_PER_SWEEP * TILE_SIZE * TILE_SIZE];

    while(true)
    {
        if(block.thread_rank() == 0)
        {
            readStartIdx = atomicAdd(binElementProcessedCounter, BINS_PER_SWEEP);
            binsToRasterize = min(BINS_PER_SWEEP, (int)(*binElementQueueCounter) - readStartIdx);
            tilesToRasterize = 0;
        }

        block.sync();
        
        if (binsToRasterize <= 0) break;
        
        for (uint16_t i = block.thread_rank(); i < binsToRasterize; i += block.size())
        {
            binElements[i] = binElementQueue[readStartIdx + i];
            auto& binElement = binElements[i];

            uint32_t triangleIndex = binElement.triangleRef.getTriangleId();
            uint32_t geometryId = binElement.triangleRef.getGeometryId();
            
            TriangleData &geometry = queue.geometries[geometryId];
            TriangleMaterial &material = queue.materials[geometryId];
            
            sh_positions[3 * i + 0] = vec4(geometry.position[3 * triangleIndex + 0], 1.0f);
            sh_positions[3 * i + 1] = vec4(geometry.position[3 * triangleIndex + 1], 1.0f);
            sh_positions[3 * i + 2] = vec4(geometry.position[3 * triangleIndex + 2], 1.0f);
            
            // store screen space positions in shared mem since they're being reused later
            mat4 geomTransform = transform * geometry.transform;
            sh_positions[3 * i + 0] = toScreenCoord((vec3)sh_positions[3 * i + 0], geomTransform, target.width, target.height);
            sh_positions[3 * i + 1] = toScreenCoord((vec3)sh_positions[3 * i + 1], geomTransform, target.width, target.height);
            sh_positions[3 * i + 2] = toScreenCoord((vec3)sh_positions[3 * i + 2], geomTransform, target.width, target.height);
            
            sh_uvs[3 * i + 0] = geometry.uv[3 * triangleIndex + 0];
            sh_uvs[3 * i + 1] = geometry.uv[3 * triangleIndex + 1];
            sh_uvs[3 * i + 2] = geometry.uv[3 * triangleIndex + 2];
            
            const vec4 *tri = &sh_positions[3 * i];
            vec3 A, B, C;
            tri_edge_coeffs_conservative(tri, A, B, C);
            
            int2 binOrigin = binPixelOrigin(binElement.binID);
            
            uint64_t mask = 0ull;
            
            for (uint8_t ty = 0; ty < TILES_PER_BIN; ++ty)
            {
                mask |= uint64_t(row_span_mask(A, B, C, binOrigin.x, binOrigin.y, ty)) << (ty * 8);
            }

            tileMasks[i] = mask;
            tileMaskCounts[i] = __popcll(mask);
        }
        
        block.sync();
        
        constexpr uint16_t lanesPerMask = TILE_SIZE * TILE_SIZE;
        const uint8_t warpId = (block.thread_rank() >> 5) & 1;
        const uint8_t warpLane = block.thread_rank() & 31; // 0..31
        const uint16_t masksPerBlock = block.size() / lanesPerMask;
        for (uint16_t maskIdx = 0; maskIdx < binsToRasterize; maskIdx += masksPerBlock)
        {
            uint64_t mask = __shfl_sync(0xFFFFFFFFu, tileMasks[maskIdx], 0);
            
            uint8_t bitIndex = warpLane + warpId * 32; // bits per mask
            bool isBitSet = mask & (1ull << bitIndex);
            
            if (isBitSet)
            {
                int slot = atomicAdd(&tilesToRasterize, 1); // increment the number of tiles to rasterize
                tileElements[slot] = bitIndex;               // store the tile index
            }
            
            block.sync();
        }
        
        // compute (exclusive) prefix sum of tile mask counts
        __shared__ int tileMaskCountPrefixSum[BINS_PER_SWEEP];
        prefixSum(tileMaskCounts, binsToRasterize, tileMaskCountPrefixSum);
        
        int binsPerRow = target.width / BIN_SIZE;
        
        // rasterize triangles
        for (uint16_t i = block.thread_rank(); i < tilesToRasterize; i += block.size())
        {
            using bool3 = glm::bvec3;
            using float3 = glm::vec3;
            
            uint16_t binIdx = find_bin_group_idx(i, tileMaskCountPrefixSum, binsToRasterize);
            TriangleRef triRef = binElements[binIdx].triangleRef;
            uint16_t binID = binElements[binIdx].binID;
            
            uint32_t triangleIndex = triRef.getTriangleId();
            uint32_t geometryId = triRef.getGeometryId();
            TriangleData &geometry = queue.geometries[geometryId];
            TriangleMaterial &material = queue.materials[geometryId];
            
            uint8_t tileIndex = tileElements[i];
            uint8_t tileX = tileIndex % TILE_SIZE;
            uint8_t tileY = tileIndex / TILE_SIZE;
            
            int2 binOrigin = binPixelOrigin(binID);
            
            // compute the screen-space bounding rectangle of the tile
            int min_y = clamp(binOrigin.y + tileY * TILE_SIZE, 0, target.height);
            int min_x = clamp(binOrigin.x + tileX * TILE_SIZE, 0, target.width);
            int max_y = clamp(min_y + TILE_SIZE, 0, target.height);
            int max_x = clamp(min_x + TILE_SIZE, 0, target.width);
            
            const vec4 *tri = &sh_positions[3 * binIdx];
            vec3 A, B, C;
            tri_edge_coeffs(tri, A, B, C);
            
            bool3 horiz = glm::equal(A, float3(0.0f));
            bool3 pos = glm::greaterThan(A, float3(0.0f));
            bool3 neg = glm::lessThan(A, float3(0.0f));
            
            float3 invA = {A.x == 0 ? 1e8 : 1.0f / A.x, A.y == 0 ? 1e8 : 1.0f / A.y, A.z == 0 ? 1e8 : 1.0f / A.z};
            float invArea = 1.0f / (B.x * A.z - A.x * B.z);
            
            vec2 sample = {(float)min_x - tri[0].x, (float)min_y - tri[0].y};
            float3 CY = B * float(min_y) + C;
            
            // precompute these once outside the loop
            float ds_dx = A.z * invArea;
            float dt_dx = A.x * invArea;
            
            float ds_dy = B.z * invArea;
            float dt_dy = B.x * invArea;
            
            float s = sample.y * ds_dy;
            float t = sample.y * dt_dy;
            
            for (int y = min_y; y < max_y; y++, CY += B, sample.y++, s += ds_dy, t += dt_dy)
            {
                bool3 outsideH = bool3(horiz.x && (CY.x < 0.0f), horiz.y && (CY.y < 0.0f), horiz.z && (CY.z < 0.0f));
                if (glm::any(outsideH))
                continue; // skip rows outside the triangle
                
                float3 cross = -(CY * invA);
                
                float lower = glm::compMax(glm::mix(float3(-FLT_MAX), cross, pos));
                float upper = glm::compMin(glm::mix(float3(FLT_MAX), cross, neg));
                if (lower > upper)
                continue; // skip rows outside the triangle
                
                int ix0 = int(floorf(lower));
                int ix1 = int(ceilf(upper));
                
                ix0 = clamp(ix0, (int)min_x, (int)max_x); // clamp within tile width
                ix1 = clamp(ix1, (int)min_x, (int)max_x);
                
                sample.x = ix0 - tri[0].x;
                s += sample.x * ds_dx;
                t += sample.x * dt_dx;
                
                for (int x = ix0; x < ix1; x++, sample.x++, s += ds_dx, t += dt_dx)
                {
                    float v = 1.0f - s - t;
                    
                    uint32_t color = computeColor(triangleIndex, geometry, material, material.texture, s, t, v);
                    uint8_t *rgb = (uint8_t *)&color;
                    
                    int2 pixelCoords = make_int2(x, y);
                    int pixelID = pixelCoords.x + pixelCoords.y * target.width;
                    
                    float depth = v * tri[0].w + s * tri[1].w + t * tri[2].w;
                    uint64_t udepth = *((uint32_t *)&depth);
                    uint64_t pixel = (udepth << 32ull) | color;
                    
                    atomicMin(&target.framebuffer[pixelID], pixel);
                }
                
                s -= sample.x * ds_dx;
                t -= sample.x * dt_dx;
            }
        }

        block.sync();            
    }   
    
}