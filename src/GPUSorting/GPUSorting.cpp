
#include "GPUSorting.h"
#include "../CudaVirtualMemory.h"
#include "CURuntime.h"
#include "Runtime.h"

namespace GPUSorting{

const bool k_keysOnly = false;
const uint32_t k_radix = 256;
const uint32_t k_radixPasses = 4;
const uint32_t k_partitionSize = 7680;
const uint32_t k_upsweepThreads = 128;
const uint32_t k_scanThreads = 128;
const uint32_t k_downsweepThreads = 512;
const uint32_t k_valPartSize = 4096;

struct State{
	shared_ptr<CudaVirtualMemory> m_alt;
	shared_ptr<CudaVirtualMemory> m_altPayload;
	CUdeviceptr m_globalHistogram;
	shared_ptr<CudaVirtualMemory> m_passHistogram;
	double t_start;
};

unordered_map<CUstream, State> streamState;

CudaModularProgram* prog_radix = nullptr;


void initSorting(){
	static bool initialized = false;

	if(!initialized){
		prog_radix = new CudaModularProgram({"./src/GPUSorting/RadixSort.cu"});

		initialized = true;
	}
}

State getState(CUstream stream){
	if(!streamState.contains(stream)){
		State state;
		state.m_globalHistogram = CURuntime::alloc("m_globalHistogram", k_radix * k_radixPasses * sizeof(uint32_t));
		state.m_passHistogram   = CURuntime::allocVirtual("m_passHistogram");
		state.m_alt             = CURuntime::allocVirtual("m_alt");
		state.m_altPayload      = CURuntime::allocVirtual("m_altPayload");

		streamState[stream] = state;
	}

	return streamState[stream];
}

void sort_32bit_keyvalue(
	uint32_t numElements, 
	CUdeviceptr keys, CUdeviceptr values, 
	CUdeviceptr alt_key, CUdeviceptr alt_payload, 
	CUstream stream
){
	initSorting();

	if(numElements == 0) return;

	bool measureTimings_old = CudaModularProgram::measureTimings;
	CudaModularProgram::measureTimings = false;

	State state = getState(stream);

	CudaModularProgram::Timing timing;
	if(Runtime::measureTimings){
		timing.cuStart = CudaModularProgram::eventPool.acquire();
		cuEventRecord(timing.cuStart, stream);
		state.t_start = now();
	}

	uint32_t threadblocks  = (numElements + k_partitionSize - 1) / k_partitionSize;

	auto m_sort = keys;
	auto m_sortPayload = values;

	CUdeviceptr l_alt, l_alt_payload;
	if(alt_key != 0 && alt_payload != 0){
		// use provided intermediate buffers
		l_alt = alt_key;
		l_alt_payload = alt_payload;
	}else{
		// commit as much memory as needed for intermediate buffers

		state.m_alt->commit(4 * numElements);
		state.m_altPayload->commit(4 * numElements);

		l_alt = state.m_alt->cptr;
		l_alt_payload = state.m_altPayload->cptr;
	}

	// For some reason we need the * 2, even though that wasnt the case in the original source
	state.m_passHistogram->commit(threadblocks * k_radix * sizeof(uint32_t));

	CURuntime::check(cuMemsetD8Async(state.m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t), stream));

	// cuCtxSynchronize();

	auto launch = [&](string kernelName, void** args, uint32_t gridSize, uint32_t blockSize){
		prog_radix->launch(kernelName, args, {.gridsize = gridSize, .blocksize = blockSize, .stream = stream});
	};

	{ // PASS 1
		uint32_t radixShift = 0;

		void* argsUpsweep[] = {&m_sort, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("Upsweep", argsUpsweep, threadblocks, k_upsweepThreads);

		void* argsScan[] = {&state.m_passHistogram->cptr, &threadblocks};
		launch("Scan", argsScan, k_radix, k_scanThreads);

		void* argsDownsweep[] = {&m_sort, &m_sortPayload, &l_alt, &l_alt_payload, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("DownsweepPairs", argsDownsweep, threadblocks, k_downsweepThreads);
	}

	{ // PASS 2
		uint32_t radixShift = 8;

		void* argsUpsweep[] = {&l_alt, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("Upsweep", argsUpsweep, threadblocks, k_upsweepThreads);

		void* argsScan[] = {&state.m_passHistogram->cptr, &threadblocks};
		launch("Scan", argsScan, k_radix, k_scanThreads);

		void* argsDownsweep[] = {&l_alt, &l_alt_payload, &m_sort, &m_sortPayload, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("DownsweepPairs", argsDownsweep, threadblocks, k_downsweepThreads);
	}

	{ // PASS 3
		uint32_t radixShift = 16;

		void* argsUpsweep[] = {&m_sort, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("Upsweep", argsUpsweep, threadblocks, k_upsweepThreads);

		void* argsScan[] = {&state.m_passHistogram->cptr, &threadblocks};
		launch("Scan", argsScan, k_radix, k_scanThreads);

		void* argsDownsweep[] = {&m_sort, &m_sortPayload, &l_alt, &l_alt_payload, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("DownsweepPairs", argsDownsweep, threadblocks, k_downsweepThreads);
	}

	{ // PASS 4
		uint32_t radixShift = 24;

		void* argsUpsweep[] = {&l_alt, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("Upsweep", argsUpsweep, threadblocks, k_upsweepThreads);

		void* argsScan[] = {&state.m_passHistogram->cptr, &threadblocks};
		launch("Scan", argsScan, k_radix, k_scanThreads);

		void* argsDownsweep[] = {&l_alt, &l_alt_payload, &m_sort, &m_sortPayload, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("DownsweepPairs", argsDownsweep, threadblocks, k_downsweepThreads);
	}


	if(Runtime::measureTimings){
		timing.cuEnd = CudaModularProgram::eventPool.acquire();
		cuEventRecord(timing.cuEnd, stream);

		timing.host_duration = now() - state.t_start;
		timing.device_duration = 0.0;
		timing.kernelName = "[32bit radix sort]";
		CudaModularProgram::addTiming(timing);
	}

	CudaModularProgram::measureTimings = measureTimings_old;
}


// note that keys are still 32 bit, but only the first 16 bit are sorted
void sort_16bitkey_32bitvalue(uint32_t numElements, CUdeviceptr keys, CUdeviceptr values, CUstream stream){

	initSorting();

	if(numElements == 0) return;

	bool measureTimings_old = CudaModularProgram::measureTimings;
	CudaModularProgram::measureTimings = false;

	State state = getState(stream);

	CudaModularProgram::Timing timing;
	if(Runtime::measureTimings){
		timing.cuStart = CudaModularProgram::eventPool.acquire();
		cuEventRecord(timing.cuStart, stream);
		state.t_start = now();
	}

	uint32_t threadblocks  = (numElements + k_partitionSize - 1) / k_partitionSize;

	auto m_sort = keys;
	auto m_sortPayload = values;


	CUdeviceptr l_alt, l_alt_payload;
	{
		// commit as much memory as needed for intermediate buffers
		state.m_alt->commit(4 * numElements);
		state.m_altPayload->commit(4 * numElements);

		l_alt = state.m_alt->cptr;
		l_alt_payload = state.m_altPayload->cptr;
	}

	state.m_passHistogram->commit(threadblocks * k_radix * sizeof(uint32_t));

	CURuntime::check(cuMemsetD8Async(state.m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t), stream));

	auto launch = [&](string kernelName, void** args, uint32_t gridSize, uint32_t blockSize){
		prog_radix->launch(kernelName, args, {.gridsize = gridSize, .blocksize = blockSize, .stream = stream});
	};

	{ // PASS 1
		uint32_t radixShift = 0;

		void* argsUpsweep[] = {&m_sort, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("Upsweep", argsUpsweep, threadblocks, k_upsweepThreads);
		// prog_radix->launch("Upsweep", argsUpsweep, {.gridsize = threadblocks, .blocksize = k_upsweepThreads});

		void* argsScan[] = {&state.m_passHistogram->cptr, &threadblocks};
		launch("Scan", argsScan, k_radix, k_scanThreads);
		// prog_radix->launch("Scan", argsScan, {.gridsize = k_radix, .blocksize = k_scanThreads});

		void* argsDownsweep[] = {&m_sort, &m_sortPayload, &l_alt, &l_alt_payload, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("DownsweepPairs", argsDownsweep, threadblocks, k_downsweepThreads);
		// prog_radix->launch("DownsweepPairs", argsDownsweep, {.gridsize = threadblocks, .blocksize = k_downsweepThreads});
	}

	{ // PASS 2
		uint32_t radixShift = 8;

		void* argsUpsweep[] = {&l_alt, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("Upsweep", argsUpsweep, threadblocks, k_upsweepThreads);
		// prog_radix->launch("Upsweep", argsUpsweep, {.gridsize = threadblocks, .blocksize = k_upsweepThreads});

		void* argsScan[] = {&state.m_passHistogram->cptr, &threadblocks};
		launch("Scan", argsScan, k_radix, k_scanThreads);
		// prog_radix->launch("Scan", argsScan, {.gridsize = k_radix, .blocksize = k_scanThreads});

		void* argsDownsweep[] = {&l_alt, &l_alt_payload, &m_sort, &m_sortPayload, &state.m_globalHistogram, &state.m_passHistogram->cptr, &numElements, &radixShift};
		launch("DownsweepPairs", argsDownsweep, threadblocks, k_downsweepThreads);
		// prog_radix->launch("DownsweepPairs", argsDownsweep, {.gridsize = threadblocks, .blocksize = k_downsweepThreads});
	}


	if(Runtime::measureTimings){
		timing.cuEnd = CudaModularProgram::eventPool.acquire();
		cuEventRecord(timing.cuEnd, stream);

		timing.host_duration = now() - state.t_start;
		timing.device_duration = 0.0;
		timing.kernelName = "[16bit radix sort]";
		prog_radix->addTiming(timing);
	}

	CudaModularProgram::measureTimings = measureTimings_old;
}

};