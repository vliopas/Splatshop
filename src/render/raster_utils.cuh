#include <cooperative_groups.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtx/component_wise.hpp"
#include "../HostDeviceInterface.h"

namespace cg = cooperative_groups;

constexpr int BIN_SIZE = 64; 
constexpr int TILE_SIZE = 8; 
constexpr int TILES_PER_BIN = BIN_SIZE / TILE_SIZE;

#define FLT_MAX 3.402823466e+38F
 
// fixed-size work records 
struct BinGroup  { 
    uint32_t triangleIndex;    // triangle ID 
    uint16_t binX, binY;       // top-left bin (in BIN_SIZE units) 
    uint8_t  binW, binH;       // width & height in bins 
}; 
 
struct BinElement 
{ 
    uint32_t triangleIndex;    // triangle ID 
    uint16_t binX, binY;       // top-left bin (in BIN_SIZE units) 
    uint64_t mask;             // tile coverage mask 
};

struct TileElement 
{ 
    uint32_t triangleIndex;    // triangle ID 
    uint16_t binX, binY;       // top-left bin (in pixel units) 
};

void prefixSum(
    const BinGroup *input_array, 
    int size, // it must hold that size <= block.size()
    int *prefixSum, // output array
    int&  inclusiveOffset // inclusive prefix sum offset for this thread [output parameter]
)
{
    auto block = cg::this_thread_block();
    const int tid = block.thread_rank();

    if (tid < size)
        prefixSum[tid] = input_array[tid].binH * input_array[tid].binW;
    block.sync();

    // Inclusive scan (Hillis-Steele)
    for (int stride = 1; stride < size; stride *= 2)
    {
        int val = 0;
        if (tid >= stride  && tid < size)
            val = prefixSum[tid - stride];
        block.sync();
        if (tid < size)
            prefixSum[tid] += val;
        block.sync();
    }

    // Compute exclusive prefix sum offset for this thread
    if (tid < size)
        inclusiveOffset = (tid == 0) ? 0 : prefixSum[tid - 1];
}

// conservative rectangle test  
// Akenine-Möller, Tomas, and Timo Aila. "Conservative and tiled rasterization using a modified triangle set-up."  
// Journal of graphics tools 10.3 (2005): 1-8.  
__device__ bool rect_outside_tri(const float (&A)[3], const float (&B)[3],
                                 const float (&C)[3], int x, int y,
                                 int w, int h)
{ 
    #pragma unroll 
    for (int k = 0; k < 3; ++k) { 
        int cx = (A[k] >= 0.f) ? w : 0; 
        int cy = (B[k] >= 0.f) ? h : 0; 
        if (A[k]*(x+cx) + B[k]*(y+cy) + C[k] < 0.f) 
            return true;                 // fully outside this edge 
    } 
    return false;                        // partly or fully inside 
};

// binary‑search prefix sum to map global‑bin -> group index
__device__ __forceinline__ int find_bin_group_idx(int bin,
                                                 const int* __restrict__ prefix,
                                                 int group_count)
{
    int hi = 0, lo = group_count - 1;
    while (hi <= lo) {
        int mid = (hi + lo) >> 1;
        if (prefix[mid] <= bin)
            hi = mid + 1;
        else
            lo = mid - 1;
    }
    return lo;
}

// compute edge coefficients (A,B,C) for 3 edges; bias for conservative raster
__device__ void tri_edge_coeffs(const vec4* __restrict__ tri,
    vec3& A, vec3& B, vec3& C)
{
    vec2 v0 = vec2(tri[0]);          // (x0,y0)
    vec2 v1 = vec2(tri[1]);          // (x1,y1)
    vec2 v2 = vec2(tri[2]);          // (x2,y2)

    A = vec3(v0.y - v1.y, v1.y - v2.y, v2.y - v0.y);
    B = vec3(v1.x - v0.x, v2.x - v1.x, v0.x - v2.x);
    C = -(A * vec3(v0.x, v1.x, v2.x) + B * vec3(v0.y, v1.y, v2.y));

    // bias for conservative rasterization
    // vec3 invLen2 = A*A + B*B; // length² per edge
    // vec3 invLen  = vec3(rsqrtf(invLen2.x),
    //                     rsqrtf(invLen2.y),
    //                     rsqrtf(invLen2.z));
    // C -= 0.5f * invLen;                          // bias outward 
}

// bit‑span mask for a single 8‑tile row (ty) inside the bin
__device__ __forceinline__ uint8_t row_span_mask(const vec3& A, const vec3& B, const vec3& C, 
    int bin_x_pix, int bin_y_pix, int ty)
{
    using float3 = glm::vec3;
    using bool3 = glm::bvec3;
    
    constexpr int TILE_COUNT = BIN_SIZE / TILE_SIZE;
    
    float sampleY = float(bin_y_pix + ty * TILE_SIZE + 0.5f * TILE_SIZE);
    float3 CY = B * sampleY + C;

    // horizontal‑edge cull: A == 0 && CY < 0 -> we are in negative half-space -> whole row outside
    bool3 horiz = glm::equal(A, float3(0.0f));
    bool3 outsideH = bool3( horiz.x && (CY.x < 0.0f), horiz.y && (CY.y < 0.0f), horiz.z && (CY.z < 0.0f));

    if (glm::any(outsideH)) return 0x00;

    // find x where each (non‑horizontal) edge hits the scan‑line
    // cross = -(B*y + C)/A — we’ll never use the horizontal components
    float3 cross = -(CY / A);   // A==0 gives ±INF or NaN, but masked out below

    // merge positive half‑spaces
    // A > 0  -> lower bound  (x ≥ cross)
    // A < 0  -> upper bound  (x ≤ cross)
    bool3  pos = glm::greaterThan(A, float3(0.0f));
    bool3  neg = glm::lessThan   (A, float3(0.0f));

    float lower = glm::compMax(glm::mix(float3(-FLT_MAX), cross, pos));
    float upper = glm::compMin(glm::mix(float3( FLT_MAX), cross, neg));

    if (lower > upper) return 0x00; // no overlap with the row

    // snap to pixel centres, convert to tile indices, build mask (same as before)
    float xStart = ceilf (lower);
    float xEnd   = floorf(upper);

    int left  = max(0,              int((xStart - bin_x_pix) / TILE_SIZE));
    int right = min(TILE_COUNT - 1, int((xEnd   - bin_x_pix) / TILE_SIZE));

    if (right < left) return 0x00;

    uint8_t mask = uint8_t(0xFFu >> (8 - (right - left + 1))) << left;
    return mask;
}