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

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"

namespace GPUSorting{

	extern CudaModularProgram* prog_radix;

	void initSorting();

	void sort_32bit_keyvalue(uint32_t numElements, CUdeviceptr keys, CUdeviceptr values, CUdeviceptr alt_key = 0, CUdeviceptr alt_payload = 0, CUstream stream = 0);

	// note that keys are still 32 bit, but only the first 16 bit are sorted
	void sort_16bitkey_32bitvalue(uint32_t numElements, CUdeviceptr keys, CUdeviceptr values, CUstream stream);

};