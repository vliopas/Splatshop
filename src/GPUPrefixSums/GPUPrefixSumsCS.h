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

#include <print>
#include <source_location>

using namespace std;

namespace GPUPrefixSumsCS{

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
	const uint32_t k_csdlThreads = 256;

	static inline CudaModularProgram* prog = nullptr;

	struct State{
		CUdeviceptr m_index;
		CUdeviceptr m_threadBlockReduction;
		double t_start;
	};

	unordered_map<CUstream, State> streamState;

	State getState(CUstream stream){
		if(!streamState.contains(stream)){
			const uint32_t maxThreadBlocks = divRoundUp(k_maxSize, k_partitionSize);
			
			State state;
			
			state.m_index = CURuntime::alloc("m_index", sizeof(uint32_t));
			state.m_threadBlockReduction = CURuntime::alloc("m_threadBlockReduction", maxThreadBlocks * sizeof(uint32_t));

			streamState[stream] = state;
		}

		return streamState[stream];
	}

	void init(){
		static bool initialized = false;

		if(!initialized){
			prog = new CudaModularProgram({"./src/GPUPrefixSums/ChainedScanDecoupledLookback.cu"});
			initialized = true;
		}
	}

	void dispatch(uint32_t size, CUdeviceptr m_scan, CUstream stream = 0, source_location location = source_location::current()){

		init();

		if (size == 0) return;

		uint32_t alignedSize = align16(size);
		uint32_t vectorizedSize = vectorizeAlignedSize(alignedSize);
		uint32_t threadBlocks = divRoundUp(alignedSize, k_partitionSize);

		State state = getState(stream);

		CudaModularProgram::Timing timing;
		if(Runtime::measureTimings){
			state.t_start = now();
			timing.cuStart = CudaModularProgram::eventPool.acquire();
			cuEventRecord(timing.cuStart, stream);
		}

		// uint32_t* m_scan = (uint32_t*)values;

		auto launch = [&](string kernelName, void** args, int gridSize, int blockSize){
			auto res_launch = cuLaunchKernel(prog->kernels[kernelName],
				gridSize, 1, 1,
				blockSize, 1, 1,
				0, stream, args, nullptr);

			if (res_launch != CUDA_SUCCESS) {
				const char* errorName;
				cuGetErrorName(res_launch, &errorName);

				const char* errorString;
				cuGetErrorString(res_launch, &errorString);

				println("ERROR: failed to launch kernel \"{}\". gridSize: {}, blocksize: {}", kernelName, gridSize, blockSize);

				uint32_t code = res_launch;
				string filename = location.file_name();
				uint32_t line = location.line();
				string functionName = location.function_name();

				std::println("ERROR(CUDA): code: {}, name: '{}', string: '{}'", code, errorName, errorString);
				std::println("    at file: {}, line: {}, function: {}", filename, line, functionName);

				exit(16334565);
			}
		};

		CURuntime::check(cuMemsetD8Async(state.m_index, 0, sizeof(uint32_t), stream));
		CURuntime::check(cuMemsetD8Async(state.m_threadBlockReduction, 0, threadBlocks * sizeof(uint32_t), stream));


		void* args[] = { &m_scan, &state.m_threadBlockReduction, &state.m_index, &vectorizedSize };
		launch("CSDLExclusive", args, threadBlocks, k_csdlThreads);
		
		if(Runtime::measureTimings){
			timing.cuEnd = CudaModularProgram::eventPool.acquire();
			cuEventRecord(timing.cuEnd, stream);

			timing.host_duration = now() - state.t_start;
			timing.device_duration = 0.0;
			timing.kernelName = "prefix sum";
			CudaModularProgram::addTiming(timing);
		}

	}

};