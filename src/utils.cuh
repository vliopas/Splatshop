
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FALSE 0
#define TRUE 1

typedef unsigned int uint32_t;
typedef int int32_t;
// typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define Infinity 0x7f800000
constexpr uint32_t MAX_STRING_LENGTH = 1'000;

inline uint32_t strlen(const char* str){

	uint32_t length = 0;

	for(int i = 0; i < MAX_STRING_LENGTH; i++){
		if(str[i] != 0){
			length++;
		}else{
			break;
		}
	}


	return length;
}

inline bool strequals(const char* str1, const char* str2){

	int i = 0;
	while(true){

		if(str1[i] == 0 && str2[i] == 0) return true;
		if(str1[i] != str2[i]) return false;

		i++;

		if(i > 1000) return false;
	}

	return true;
}

template<typename Function>
void process(int size, Function&& f){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = block_offset + thread_offset + i;

		if(index >= size){
			break;
		}

		f(index);
	}
}

// Loops through [0, size), but blockwise instead of threadwise.
// That is, all threads of block 0 are called with index 0, block 1 with index 1, etc.
// Intented for when <size> is larger than the number of blocks,
// e.g., size 10'000 but #blocks only 100, then the blocks will keep looping until all indices are processed.
inline int for_blockwise_counter;
template<typename Function>
inline void for_blockwise(int size, Function&& f){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	__shared__ int sh_index;
	sh_index = 0;
	for_blockwise_counter = 0;

	grid.sync();

	while(true){

		if(block.thread_rank() == 0){
			uint32_t index = atomicAdd(&for_blockwise_counter, 1);
			sh_index = index;
		}

		block.sync();

		if(sh_index >= size) break;

		f(sh_index);

		block.sync();
	}
}


template<typename Function>
void processTiles(int tiles_x, int tiles_y, const int tileSize, uint32_t& tileCounter, Function&& f){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int numTiles = tiles_x * tiles_y;

	if(grid.thread_rank() == 0){
		tileCounter = 0;
	}

	grid.sync();

	__shared__ uint32_t sh_tileID;
	while(true){

		int t_tileID = 0;
		if(block.thread_rank() == 0){
			t_tileID = atomicAdd(&tileCounter, 1);
		}
		sh_tileID = t_tileID;

		block.sync();

		if(sh_tileID >= numTiles) break;

		int tileX = sh_tileID % tiles_x;
		int tileY = sh_tileID / tiles_x;

		f(tileX, tileY);

		block.sync();
	}

}

// void printNumber(int64_t number, int leftPad = 0);

inline uint64_t nanotime(){

	uint64_t nanotime;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));

	return nanotime;
}


template<typename T>
T clamp(T value, T min, T max){

	if(value < min) return min;
	if(value > max) return max;

	return value;
}

// from https://stackoverflow.com/a/51549250, by user Xiaojing An; license: CC BY-SA 4.0
__device__ __forceinline__ float atomicMinFloat(float * addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

// from https://stackoverflow.com/a/51549250, by user Xiaojing An; license: CC BY-SA 4.0
__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
			__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

// from: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
template<typename T>
__device__ __inline__
T warpReduceMax(T val) {

	for (int offset = warpSize / 2; offset > 0; offset /= 2){
		val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset, 32));
	}

	return 
	val;
}
// from: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
template<typename T>
__device__ __inline__
T warpReduceMin(T val) {

	for (int offset = warpSize / 2; offset > 0; offset /= 2){
		val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset, 32));
	}

	return val;
}

// from: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
template<typename T>
__inline__ __device__
T blockReduceMax(T val) {

	static __shared__ T shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMax(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceMax(val); //Final reduce within first warp

	return val;
}

template<typename T>
__inline__ __device__
T blockReduceMin(T val) {

	static __shared__ T shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMin(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceMin(val); //Final reduce within first warp

	return val;
}