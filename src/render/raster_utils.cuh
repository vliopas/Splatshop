#include <cooperative_groups.h>

#include "libs/glm/glm/glm.hpp"
#include "libs/glm/glm/gtx/component_wise.hpp"
#include "HostDeviceInterface.h"

namespace cg = cooperative_groups;

constexpr int BIN_SIZE = 64; 
constexpr int TILE_SIZE = 8; 
constexpr int TILES_PER_BIN = BIN_SIZE / TILE_SIZE;

#define FLT_MAX 3.402823466e+38F
 
// fixed-size work records 
struct BinGroup  { 
    uint16_t triangleIndex;    // local triangle ID 
    uint16_t binX, binY;       // top-left bin (in BIN_SIZE units) 
    uint8_t  binW, binH;       // width & height in bins 
}; 
 
struct BinElement // packed data to 4-byte alignment
{
    uint32_t triBinPacked; // packed: [local triangle ID(8 bits) | binX(12 bits) | binY(12 bits)]
    uint32_t maskLo; // lower 32 bits of mask
    uint32_t maskHi; // upper 32 bits of mask

    __host__ __device__ inline void pack(uint8_t triangleIndex, uint16_t binX, uint16_t binY)
    {
        triBinPacked = ((uint32_t)(triangleIndex & 0xFF) << 24) 
                     | ((uint32_t)(binX & 0xFFF) << 12) 
                     | (binY & 0xFFF);
    }

    __host__ __device__ inline uint8_t getTriangleIndex() const { return (triBinPacked >> 24) & 0xFF; }
    __host__ __device__ inline uint16_t getBinX() const { return (triBinPacked >> 12) & 0xFFF; }
    __host__ __device__ inline uint16_t getBinY() const { return triBinPacked & 0xFFF; }
    __host__ __device__ inline uint64_t getMask() const { return ((uint64_t)maskHi << 32) | (uint64_t)maskLo; }

    __host__ __device__ inline void setMask(uint64_t mask)
    {
        maskLo = (uint32_t)(mask & 0xFFFFFFFF);
        maskHi = (uint32_t)(mask >> 32);
    }
};

struct TileElement 
{ 
    uint16_t triangleIndex;    // local triangle ID 
    uint16_t binX, binY;       // top-left bin (in pixel units) 
};

void prefixSum(
    const BinGroup *input_array, 
    int size, // it must hold that size <= block.size()
    int *prefixSum, // output array
    int&  exclusiveOffset // exclusive prefix sum offset for this thread [output parameter]
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
        if (tid >= stride && tid < size)
            val = prefixSum[tid - stride];
        block.sync();
        if (tid < size)
            prefixSum[tid] += val;
        block.sync();
    }

    // Compute exclusive prefix sum offset for this thread
    if (tid < size)
        exclusiveOffset = (tid == 0) ? 0 : prefixSum[tid - 1];
}

__device__ void prefixSumSmall(
        const BinGroup* input_array,
        int  size,
        int* prefixSum,
        int& exclusiveOffset)            // output per thread (same for all)
{
    auto block = cg::this_thread_block();

    /* ---- Let exactly one thread do the whole job ------------------------ */
    if (block.thread_rank() == 0)
    {
        int running = 0;
        for (int i = 0; i < size; ++i)
        {
            running += input_array[i].binH * input_array[i].binW;
            prefixSum[i] = running;                // inclusive so far
        }
    }

    block.sync();                                  // make data visible

    /* ---- Convert to per‑thread exclusive offset ------------------------- */
    if (block.thread_rank() < size)
        exclusiveOffset = (block.thread_rank() == 0)
                          ? 0
                          : prefixSum[block.thread_rank() - 1];
    block.sync();
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
__device__ __forceinline__ uint16_t find_bin_group_idx(uint16_t bin,
                                                 const int* __restrict__ prefix,
                                                 int group_count)
{
    uint16_t hi = 0, lo = group_count - 1;
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
}

__device__ void tri_edge_coeffs_conservative(const vec4* __restrict__ tri,
    vec3& A, vec3& B, vec3& C)
{
    vec2 v0 = vec2(tri[0]);          // (x0,y0)
    vec2 v1 = vec2(tri[1]);          // (x1,y1)
    vec2 v2 = vec2(tri[2]);          // (x2,y2)

    A = vec3(v0.y - v1.y, v1.y - v2.y, v2.y - v0.y);
    B = vec3(v1.x - v0.x, v2.x - v1.x, v0.x - v2.x);
    C = -(A * vec3(v0.x, v1.x, v2.x) + B * vec3(v0.y, v1.y, v2.y));

    // bias for conservative rasterization
    vec3 len = A*A + B*B; // length² per edge
    len  =         vec3(sqrt(len.x),
                        sqrt(len.y),
                        sqrt(len.z));
    C += 0.5f * TILE_SIZE* len;  
}

// bit‑span mask for a single 8‑tile row (ty) inside the bin
__device__  uint8_t row_span_mask(const vec3& A, const vec3& B, const vec3& C, 
    int bin_x_pix, int bin_y_pix, int ty)
{
    using float3 = glm::vec3;
    using bool3 = glm::bvec3;
    constexpr float epsilon = 1e-6;
    
    // sample Y at the top and bottom of the tile row
    float y_top    = float(bin_y_pix + ty * TILE_SIZE);
    float y_bottom = float(bin_y_pix + (ty + 1) * TILE_SIZE);
    vec3 len = A*A + B*B; // length² per edge
    len  =         vec3(sqrt(len.x),
                        sqrt(len.y),
                        sqrt(len.z));
    float3 Cnew = C + 0.5f * TILE_SIZE* len;   
    // compute CY at top and bottom sample lines
    float3 CY_top    = B * y_top + Cnew;
    float3 CY_bottom = B * y_bottom + Cnew;

    // horizontal-edge cull for top and bottom sample lines
    bool3 horiz = {abs(A.x) < epsilon, abs(A.y) < epsilon, abs(A.z) < epsilon};
    bool3 outsideH_top = bool3(horiz.x && (CY_top.x < 0.0f), horiz.y && (CY_top.y < 0.0f), horiz.z && (CY_top.z < 0.0f));
    bool3 outsideH_bottom = bool3(horiz.x && (CY_bottom.x < 0.0f), horiz.y && (CY_bottom.y < 0.0f), horiz.z && (CY_bottom.z < 0.0f));

    // if both top and bottom are outside for any edge, the whole row is outside
    if (glm::any(outsideH_top & outsideH_bottom)) return 0x00;

    float3 invA = glm::mix(1.0f / A, float3(1e8f), horiz);

    // compute intersection points for top and bottom lines
    auto compute_bounds = [&](float3 CY) -> float2 {
        // avoid division by zero, masked out later by pos/neg masks
        float3 cross = - CY * invA;

        bool3 pos = glm::greaterThan(A, float3(epsilon));
        bool3 neg = glm::lessThan(A, float3(epsilon));

        float lower = glm::compMax(glm::mix(float3(-FLT_MAX), cross, pos));
        float upper = glm::compMin(glm::mix(float3( FLT_MAX), cross, neg));

        return {lower, upper};
    };

    auto [lower_top, upper_top]       = compute_bounds(CY_top);
    auto [lower_bottom, upper_bottom] = compute_bounds(CY_bottom);

    // combine bounds to conservatively cover whole vertical span
    float lower = fminf(lower_top, lower_bottom);
    float upper = fmaxf(upper_top, upper_bottom);

    if (lower > upper) return 0x00; // no overlap with the row

    // snap to pixel centers and convert to tile indices
    // using floorf for start and ceilf for end to cull tiles conservatively
    float xStart = floorf(lower);
    float xEnd   = ceilf(upper);

    int left  = max(0, int((xStart - bin_x_pix) / TILE_SIZE)); // +1 and -1 for conservative culling
    int right = min(TILES_PER_BIN - 1, int((xEnd   - bin_x_pix) / TILE_SIZE));

    if (right < left) return 0x00;

    // Build mask: set bits for tiles covered between left and right
    uint8_t mask = uint8_t(0xFFu >> (8 - (right - left + 1))) << left;

    return mask;
}