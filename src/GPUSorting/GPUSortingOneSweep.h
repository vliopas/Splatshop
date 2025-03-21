/******************************************************************************
 * GPUSorting
 * 
 * Adapted from:
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 2/21/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/

#pragma once

namespace GPUSortingOneSweep{

// SORTING
const int k_maxSize = 50'000'000;
const uint32_t k_radix = 256;
const uint32_t k_radixPasses = 4;
const uint32_t k_partitionSize = 7680;
const uint32_t k_globalHistPartitionSize = 65536;
const uint32_t k_globalHistThreads = 128;
const uint32_t k_binningThreads = 512;
const uint32_t k_valPartSize = 4096;

CUdeviceptr m_alt;
CUdeviceptr m_altPayload;
CUdeviceptr m_index;
CUdeviceptr m_globalHistogram;
CUdeviceptr m_firstPassHistogram;
CUdeviceptr m_secPassHistogram;
CUdeviceptr m_thirdPassHistogram;
CUdeviceptr m_fourthPassHistogram;

CudaModularProgram* prog = nullptr;

bool initialized = false;

static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
{ 
	return (x + y - 1) / y;
}

void initSorting(){

	prog = new CudaModularProgram({"./src/GPUSorting/OneSweep.cu"});


	const uint32_t maxBinningThreadblocks = divRoundUp(k_maxSize, k_partitionSize);

	m_alt                 = CURuntime::alloc("m_alt", k_maxSize * sizeof(uint32_t));
	m_index               = CURuntime::alloc("m_index", k_radixPasses * sizeof(uint32_t));
	m_globalHistogram     = CURuntime::alloc("m_globalHistogram", k_radixPasses * k_radix * sizeof(uint32_t));
	m_firstPassHistogram  = CURuntime::alloc("m_firstPassHistogram", maxBinningThreadblocks * k_radix * sizeof(uint32_t));
	m_secPassHistogram    = CURuntime::alloc("m_secPassHistogram", maxBinningThreadblocks * k_radix * sizeof(uint32_t));
	m_thirdPassHistogram  = CURuntime::alloc("m_thirdPassHistogram", maxBinningThreadblocks * k_radix * sizeof(uint32_t));
	m_fourthPassHistogram = CURuntime::alloc("m_fourthPassHistogram", maxBinningThreadblocks * k_radix * sizeof(uint32_t));
	m_altPayload          = CURuntime::alloc("m_altPayload", k_maxSize * sizeof(uint32_t));

	initialized = true;
}



void sort_32bit_keyvalue(uint32_t numElements, CUdeviceptr keys, CUdeviceptr values){

	if(!initialized){
		initSorting();
	}

	uint32_t size = numElements;

	// UPDATE INDEX AND DEPTH BUFFERS
	CUevent ce_start, ce_end;
	cuEventCreate(&ce_start, CU_EVENT_DEFAULT);
	cuEventCreate(&ce_end, CU_EVENT_DEFAULT);

	if(size == 0) return;

	cuEventRecord(ce_start, 0);

	// uint32_t threadblocks  = (size + k_partitionSize - 1) / k_partitionSize;
	const uint32_t globalHistThreadBlocks = divRoundUp(size, k_globalHistPartitionSize);
	const uint32_t binningThreadBlocks = divRoundUp(size, k_partitionSize);

	// clear memory
	cuMemsetD8(m_index, 0, k_radixPasses * sizeof(uint32_t));
	cuMemsetD8(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));
	cuMemsetD8(m_firstPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
	cuMemsetD8(m_secPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
	cuMemsetD8(m_thirdPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
	cuMemsetD8(m_fourthPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));

	auto m_sort = keys;
	auto m_sortPayload = values;

	auto launch = [&](string kernelName, void** args, int gridSize, int blockSize){
		auto res_launch = cuLaunchKernel(prog->kernels[kernelName],
			gridSize, 1, 1,
			blockSize, 1, 1,
			0, 0, args, nullptr);

		if (res_launch != CUDA_SUCCESS) {
			const char* str;
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
			cout << __FILE__ << " - " << __LINE__ << endl;

			println("kernel: {}", kernelName);

			exit(5243);
		}
	};

	cuCtxSynchronize();


	void* argsHist[] = {&m_sort, &m_globalHistogram, &size};
	launch("GlobalHistogram", argsHist, globalHistThreadBlocks, k_globalHistThreads);
	// cuCtxSynchronize();

	void* argsScan[] = {&m_sort, &m_globalHistogram, &m_firstPassHistogram, &m_secPassHistogram, &m_thirdPassHistogram, &m_fourthPassHistogram};
	launch("Scan", argsScan, k_radixPasses, k_radix);
	// cuCtxSynchronize();

	uint32_t shift0 = 0;
	void* args1[] = {&m_sort, &m_sortPayload, &m_alt, &m_altPayload, &m_firstPassHistogram, &m_index, &size, &shift0};
	launch("DigitBinningPassPairs", args1, binningThreadBlocks, k_binningThreads);
	// cuCtxSynchronize();

	uint32_t shift1 = 8;
	void* args2[] = {&m_alt, &m_altPayload, &m_sort, &m_sortPayload, &m_secPassHistogram, &m_index, &size, &shift1};
	launch("DigitBinningPassPairs", args2, binningThreadBlocks, k_binningThreads);
	// cuCtxSynchronize();

	uint32_t shift2 = 16;
	void* args3[] = {&m_sort, &m_sortPayload, &m_alt,  &m_altPayload, &m_thirdPassHistogram, &m_index, &size, &shift2};
	launch("DigitBinningPassPairs", args3, binningThreadBlocks, k_binningThreads);
	//// cuCtxSynchronize();

	uint32_t shift3 = 24;
	void* args4[] = {&m_alt, &m_altPayload, &m_sort, &m_sortPayload, &m_fourthPassHistogram, &m_index, &size, &shift3};
	launch("DigitBinningPassPairs", args4, binningThreadBlocks, k_binningThreads);
	//// cuCtxSynchronize();

	cuEventRecord(ce_end, 0);

	if(Runtime::measureTimings){
		cuEventSynchronize(ce_end);
		float millies;
		cuEventElapsedTime(&millies, ce_start, ce_end);

		//Runtime::Timing timing = { "onesweep-sort, 32bit key, 32bit value", millies };
		//Runtime::timings.emplace_back(timing);
		Runtime::timings.add("onesweep-sort, 32bit key, 32bit value", millies);
	}
}



};