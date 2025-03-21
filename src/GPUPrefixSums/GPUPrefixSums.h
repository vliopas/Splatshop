/******************************************************************************
 * GPUPrefixSums
 * 
 * Adapted from: https://github.com/b0nes164/GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2024 Thomas Smith
 * https://github.com/b0nes164/GPUPrefixSums/blob/main/LICENSE
 *
 ******************************************************************************/

#pragma once

namespace GPUPrefixSums{

	static inline uint32_t divRoundUp(uint32_t x, uint32_t y){
		return (x + y - 1) / y;
	}

	static inline uint32_t align16(uint32_t x){
		return divRoundUp(x, 4) * 4;
	}

	static inline uint32_t vectorizeAlignedSize(uint32_t alignedSize){
		return alignedSize / 4;
	}

	const uint32_t k_maxSize = 50'000'000;
	const uint32_t k_partitionSize = 3072;
	const uint32_t k_rtsThreads = 256;

	// CUdeviceptr m_scan;
	static inline CUdeviceptr m_threadBlockReduction;
	static inline CUdeviceptr m_errCount;

	static inline CudaModularProgram* prog = nullptr;

	static inline CUevent ce_start, ce_end;

	void init(){

		static bool initialized = false;

		if(!initialized){
			prog = new CudaModularProgram({"./src/GPUPrefixSums/ReduceThenScan.cu"});

			const uint32_t maxThreadBlocks = divRoundUp(k_maxSize, k_partitionSize);

			m_threadBlockReduction = CURuntime::alloc("m_threadBlockReduction", maxThreadBlocks * sizeof(uint32_t));
			m_errCount = CURuntime::alloc("m_errCount", sizeof(uint32_t));

			cuEventCreate(&ce_start, CU_EVENT_DEFAULT);
			cuEventCreate(&ce_end, CU_EVENT_DEFAULT);

			initialized = true;
		}
	}

	void dispatch(uint32_t size, CUdeviceptr m_scan){

		init();

		if (size == 0) return;

		uint32_t alignedSize = align16(size);
		uint32_t vectorizedSize = vectorizeAlignedSize(alignedSize);
		uint32_t threadBlocks = divRoundUp(alignedSize, k_partitionSize);

		cuEventRecord(ce_start, 0);

		// uint32_t* m_scan = (uint32_t*)values;

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

				exit(76334565);
			}
		};

		// ReduceThenScan::Reduce<<<threadBlocks, k_rtsThreads>>>(m_scan, m_threadBlockReduction, vectorizedSize);
		void* argsReduce[] = {&m_scan, &m_threadBlockReduction, &vectorizedSize};
		launch("Reduce", argsReduce, threadBlocks, k_rtsThreads);

		// ReduceThenScan::Scan<<<1,k_rtsThreads>>>(m_threadBlockReduction, threadBlocks);
		void* argsScan[] = {&m_threadBlockReduction, &threadBlocks};
		launch("Scan", argsScan, 1, k_rtsThreads);

		// ReduceThenScan::DownSweepExclusive<<<threadBlocks, k_rtsThreads>>>(m_scan, m_threadBlockReduction, vectorizedSize);
		void* argsDownsweep[] = {&m_scan, &m_threadBlockReduction, &vectorizedSize};
		launch("DownSweepExclusive", argsDownsweep, threadBlocks, k_rtsThreads);

		cuEventRecord(ce_end, 0);

		if(Runtime::measureTimings){
			cuEventSynchronize(ce_end);
			float millies;
			cuEventElapsedTime(&millies, ce_start, ce_end);

			//Runtime::Timing timing = { "prefix sum", millies };
			//Runtime::timings.emplace_back(timing);
			Runtime::timings.add("prefix sum", millies);
		}

	}

};