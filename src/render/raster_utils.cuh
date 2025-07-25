#include <cooperative_groups.h>

#include "libs/glm/glm/glm.hpp"
#include "libs/glm/glm/gtx/component_wise.hpp"
#include "HostDeviceInterface.h"

namespace cg = cooperative_groups;

constexpr int BIN_SIZE = 64;
constexpr int TILE_SIZE = 8;
constexpr int TILES_PER_BIN = BIN_SIZE / TILE_SIZE;
constexpr int MAX_NUMBER_OF_BINS = 2040;

#define FLT_MAX 3.402823466e+38F // Removed to avoid redefinition error

constexpr int TRIANGLES_PER_SWEEP = 32;
constexpr int MAX_VERYLARGE_TRIANGLES = 10 * 1024;
constexpr int MAX_TRIANGLES_PER_BIN = MAX_NUMBER_OF_BINS * TRIANGLES_PER_SWEEP; // max triangles per bin

// for shared memory allocation
constexpr int MAX_BIN_SIZE = 32;
constexpr int MAX_TILE_SIZE = TILE_SIZE * TILE_SIZE * MAX_BIN_SIZE;

__device__ uint32_t totalBins;

enum Stage
{
    BINNING,
    RASTERIZATION,
    DONE
};

struct TriangleRef
{
    // Bit layout: [31:30] state (2 bits), [29:20] geometryId (10 bits), [19:0] triangleId (20 bits)
    static constexpr uint32_t STATE_BITS = 2;
    static constexpr uint32_t GEOM_ID_BITS = 10;
    static constexpr uint32_t TRI_ID_BITS = 20;

    static constexpr uint32_t TRI_ID_MASK = (1u << TRI_ID_BITS) - 1;                  // 0x000FFFFF
    static constexpr uint32_t GEOM_ID_MASK = (1u << GEOM_ID_BITS) - 1;                // 0x000003FF
    static constexpr uint32_t STATE_MASK = (1u << STATE_BITS) - 1;                    // 0x00000003

    static constexpr uint32_t TRI_ID_SHIFT = 0;
    static constexpr uint32_t GEOM_ID_SHIFT = TRI_ID_BITS;
    static constexpr uint32_t STATE_SHIFT = TRI_ID_BITS + GEOM_ID_BITS;

    enum State : uint32_t
    {
        INVALID   = 0b00,
        TENTATIVE = 0b01,
        VALID     = 0b10
        // 0b11 is unused
    };

    uint32_t packed;

    inline void pack(uint32_t geometryId, uint32_t triangleId, State state = TENTATIVE)
    {
        packed = ((state & STATE_MASK)     << STATE_SHIFT) |
                 ((geometryId & GEOM_ID_MASK) << GEOM_ID_SHIFT) |
                 ((triangleId & TRI_ID_MASK) << TRI_ID_SHIFT);
    }

    inline void setState(State state)
    {
        packed = (packed & ~(STATE_MASK << STATE_SHIFT)) |
                 ((state & STATE_MASK) << STATE_SHIFT);
    }
    
    inline bool isValid() const     { return getState() == VALID; }
    inline bool isTentative() const { return getState() == TENTATIVE; }
    inline bool isInvalid() const   { return getState() == INVALID; }
    
    inline State getState() const { return static_cast<State>((packed >> STATE_SHIFT) & STATE_MASK); }
    inline uint32_t getTriangleId() const { return (packed >> TRI_ID_SHIFT) & TRI_ID_MASK;}
    inline uint32_t getGeometryId() const { return (packed >> GEOM_ID_SHIFT) & GEOM_ID_MASK; }
};

struct BinQueue
{
private:
    // atomic indices for circular (ring) buffer
    uint32_t readIdx;                                   // Points to the next element to be read
    uint32_t writeIdx;                                  // Points to the next slot to write
    TriangleRef triangleIndices[MAX_TRIANGLES_PER_BIN]; // circular buffer

public:
    static uint32_t capacity() { return MAX_TRIANGLES_PER_BIN; }

    uint32_t size() const
    {
        uint32_t r = atomicAdd((uint32_t *)&readIdx, 0);
        uint32_t w = atomicAdd((uint32_t *)&writeIdx, 0);
        return (w + capacity() - r) % capacity();
    }

    int push(TriangleRef val)
    {
        while (true)
        {
            uint32_t r = atomicAdd((uint32_t *)&readIdx, 0);
            uint32_t w = atomicAdd((uint32_t *)&writeIdx, 0);
            uint32_t size = (w + capacity() - r) % capacity();

            if (size >= capacity() - 1) // queue full
                return -1;

            uint32_t newWrite = (w + 1) % capacity();

            if (atomicCAS(&writeIdx, w, newWrite) == w)
            {
                triangleIndices[w] = val; // already wrapped
                return w;
            }
        }
    }

    TriangleRef pop()
    {
        while (true)
        {
            uint32_t r = atomicAdd((uint32_t *)&readIdx, 0);
            uint32_t w = atomicAdd((uint32_t *)&writeIdx, 0);

            // Assert if the queue is empty
            assert(r != w && "BinQueue::pop(): attempted to pop from an empty queue");

            uint32_t newRead = (r + 1) % capacity();

            if (atomicCAS(&readIdx, r, newRead) == r)
                return triangleIndices[r];
        }
    }

    TriangleRef &operator[](uint32_t index) { return triangleIndices[index % capacity()]; }
    const TriangleRef &operator[](uint32_t index) const { return triangleIndices[index % capacity()]; }
    uint32_t getReadIndex() const { return readIdx; }

    void clear()
    {
        readIdx = 0;
        writeIdx = 0;
    }
};

__device__ BinQueue binQueues[MAX_NUMBER_OF_BINS];

// fixed-size work records
struct BinGroup
{
    uint16_t triangleIndex; // local triangle ID
    uint16_t binX, binY;    // top-left bin (in BIN_SIZE units)
    uint8_t binW, binH;     // width & height in bins
};

struct TriangleBatchInfo
{
    int blockTriangleOffset;      // global offset in full triangle list
    int blockLocalTriangleOffset; // local offset within current geometry chunk
    int triangleQueueIndex;       // current geometry block index

    TriangleData geometry;
    TriangleMaterial material;
    bool reprocessTriangleBatch;
};

// if screen resolution higher than 1080p
// it is not possible to have 1 block - 1 bin mapping - too many blocks necessary
// perform triangle striping -> multiple specific bins - 1 block mapping
// returns the range [first bin index, last bin index)
inline int2 computeBinRange()
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    int firstBinID;
    int lastBinID;
    // if(grid.thread_rank() == 0) printf("total bins is %d \n", totalBins);
    int binsPerBlock = totalBins / grid.num_blocks();
    int extraBins = totalBins % grid.num_blocks();

    int blockID = grid.block_rank();
    firstBinID = blockID * binsPerBlock + min(blockID, extraBins);
    int numBins = binsPerBlock + (blockID < extraBins ? 1 : 0);
    lastBinID = firstBinID + numBins;

    return make_int2(firstBinID, lastBinID);
}

template <typename T>
void prefixSum(
    const T *input_array,
    int size,
    int *prefixSum) // output - exclusive prefix sum array
{
    auto block = cg::this_thread_block();

    // input arrays not expected to be more than a few tens of elements
    // so let one thread do the whole job
    if (block.thread_rank() == 0)
    {
        int running = 0;
        for (int i = 0; i < size; ++i)
        {
            prefixSum[i] = running;
            running += input_array[i];
        }
    }

    block.sync(); // make data visible
}

template <>
void prefixSum<BinGroup>(
    const BinGroup *input_array,
    int size,
    int *prefixSum) // output - exclusive prefix sum array
{
    auto block = cg::this_thread_block();

    // input arrays not expected to be more than a few tens of elements
    // so let one thread do the whole job
    if (block.thread_rank() == 0)
    {
        int running = 0;
        for (int i = 0; i < size; ++i)
        {
            prefixSum[i] = running;
            running += input_array[i].binH * input_array[i].binW;
        }
    }

    block.sync(); // make data visible
}

// conservative rectangle test
// Akenine-Möller, Tomas, and Timo Aila. "Conservative and tiled rasterization using a modified triangle set-up."
// Journal of graphics tools 10.3 (2005): 1-8.
__device__ bool rect_outside_tri(const float (&A)[3], const float (&B)[3],
                                 const float (&C)[3], int x, int y,
                                 int w, int h)
{
#pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        int cx = (A[k] >= 0.f) ? w : 0;
        int cy = (B[k] >= 0.f) ? h : 0;
        if (A[k] * (x + cx) + B[k] * (y + cy) + C[k] < 0.f)
            return true; // fully outside this edge
    }
    return false; // partly or fully inside
};

// binary‑search prefix sum to map global‑bin -> group index
__device__ __forceinline__ uint16_t find_bin_group_idx(uint16_t bin,
                                                       const int *__restrict__ prefix,
                                                       int group_count)
{
    int lo = 0, hi = group_count - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        if (prefix[mid] <= bin)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return static_cast<uint16_t>(hi); // largest index such that prefix[idx] <= bin
}

// compute edge coefficients (A,B,C) for 3 edges; bias for conservative raster
__device__ void tri_edge_coeffs(const vec4 *__restrict__ tri,
                                vec3 &A, vec3 &B, vec3 &C)
{
    vec2 v0 = vec2(tri[0]); // (x0,y0)
    vec2 v1 = vec2(tri[1]); // (x1,y1)
    vec2 v2 = vec2(tri[2]); // (x2,y2)

    A = vec3(v0.y - v1.y, v1.y - v2.y, v2.y - v0.y);
    B = vec3(v1.x - v0.x, v2.x - v1.x, v0.x - v2.x);
    C = -(A * vec3(v0.x, v1.x, v2.x) + B * vec3(v0.y, v1.y, v2.y));
}

__device__ void tri_edge_coeffs_conservative(const vec4 *__restrict__ tri,
                                             vec3 &A, vec3 &B, vec3 &C)
{
    vec2 v0 = vec2(tri[0]); // (x0,y0)
    vec2 v1 = vec2(tri[1]); // (x1,y1)
    vec2 v2 = vec2(tri[2]); // (x2,y2)

    A = vec3(v0.y - v1.y, v1.y - v2.y, v2.y - v0.y);
    B = vec3(v1.x - v0.x, v2.x - v1.x, v0.x - v2.x);
    C = -(A * vec3(v0.x, v1.x, v2.x) + B * vec3(v0.y, v1.y, v2.y));

    // bias for conservative rasterization
    vec3 len = A * A + B * B; // length² per edge
    len = vec3(sqrt(len.x),
               sqrt(len.y),
               sqrt(len.z));
    C += 0.5f * TILE_SIZE * len;
}

// bit‑span mask for a single 8‑tile row (ty) inside the bin
__device__ uint8_t row_span_mask(const vec3 &A, const vec3 &B, const vec3 &C,
                                 int bin_x_pix, int bin_y_pix, int ty)
{
    using float3 = glm::vec3;
    using bool3 = glm::bvec3;
    constexpr float epsilon = 1e-6;

    // Sample Y at the middle of the tile row
    float y_row = float(bin_y_pix + (ty + 0.5f) * TILE_SIZE);

    vec3 len = A * A + B * B; // length² per edge
    len = vec3(sqrt(len.x), sqrt(len.y), sqrt(len.z));
    float3 Cnew = C + 0.5f * TILE_SIZE * len;

    // Compute CY at mid sample line
    float3 CY = B * y_row + Cnew;

    // Horizontal-edge cull at mid sample line
    bool3 horiz = {abs(A.x) < epsilon, abs(A.y) < epsilon, abs(A.z) < epsilon};
    bool3 outsideH_mid = bool3(horiz.x && (CY.x < 0.0f),
                               horiz.y && (CY.y < 0.0f),
                               horiz.z && (CY.z < 0.0f));

    // If any edge is outside at mid line, cull the row
    if (glm::any(outsideH_mid))
        return 0x00;

    float3 invA = glm::mix(1.0f / A, float3(1e8f), horiz);

    // Compute intersection bounds at mid sample line
    float3 cross = -CY * invA;
    bool3 pos = glm::greaterThan(A, float3(epsilon));
    bool3 neg = glm::lessThan(A, float3(-epsilon));

    float lower = glm::compMax(glm::mix(float3(-FLT_MAX), cross, pos));
    float upper = glm::compMin(glm::mix(float3(FLT_MAX), cross, neg));

    if (lower > upper)
        return 0x00; // no overlap with row

    // Snap to pixel centers and convert to tile indices
    float xStart = floorf(lower);
    float xEnd = ceilf(upper);

    int left = max(0, int((xStart - bin_x_pix) / TILE_SIZE));
    int right = min(TILES_PER_BIN - 1, int((xEnd - bin_x_pix) / TILE_SIZE));

    if (right < left)
        return 0x00;

    // Build mask: set bits for tiles covered between left and right
    uint8_t mask = uint8_t(0xFFu >> (8 - (right - left + 1))) << left;

    return mask;
}