#pragma once

#include <print>
#include <vector>

#include "cuda.h"
#include "unsuck.hpp"

using std::println;
using std::vector;

// Physical allocation granularity on an RTX 4090: 2'097'152 bytes
constexpr uint64_t DEFAULT_VIRTUAL_SIZE = 2'097'152;

// see https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
struct CudaVirtualMemory{

	struct PhysicalAllocation{
		CUmemGenericAllocationHandle handle;

		// cuMemMap parameters
		uint64_t offset = 0;
		uint64_t size = 0;
	};

	uint64_t virtualSize = 0;   // currently reserved virtual address range
	uint64_t comitted = 0;      // actuall amount of physically comitted memory
	uint64_t granularity = 0;
	uint64_t granularity_recommended = 0;
	CUdeviceptr cptr = 0;

	// Keeping track of allocated physical memory, so we can remap or free
	vector<PhysicalAllocation> allocations;

	CudaVirtualMemory(){

	}

	~CudaVirtualMemory(){
		destroy();
	}

	void destroy(){

		// cuMemCreate          ->  cuMemRelease 
		// cuMemMap             ->  cuMemUnmap
		// cuMemAddressReserve  ->  cuMemAddressFree 

		if(cptr == 0){
			// println("WARNING: tried to destroy virtual memory that was already destroyed.");
			return;
		}
		
		for(auto allocation : allocations){
			cuMemUnmap(cptr + allocation.offset, allocation.size);
			cuMemRelease(allocation.handle);
		}
		// allocHandles.clear();
		allocations.clear();

		// TODO: do we also need to revert cuMemSetAccess()?

		cuMemAddressFree(cptr, virtualSize);

		cptr = 0;
	}

	// allocate potentially large amounts of virtual memory
	static shared_ptr<CudaVirtualMemory> create(uint64_t virtualSize = DEFAULT_VIRTUAL_SIZE){

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);
		
		CUmemAllocationProp prop = {};
		prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id   = cuDevice;

		uint64_t granularity_minimum = 0;
		uint64_t granularity_recommended = 0;
		cuMemGetAllocationGranularity(&granularity_minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		cuMemGetAllocationGranularity(&granularity_recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

		uint64_t padded_size = roundUp(virtualSize, granularity_minimum);

		// reserve virtual memory
		CUdeviceptr cptr = 0;
		auto result = cuMemAddressReserve(&cptr, padded_size, 0, 0, 0);

		if(result != CUDA_SUCCESS){
			println("error {} while trying to reserve virtual memory. Requested virtual size: {}. Attempted to reserve {}", int(result), virtualSize, padded_size);
			exit(52457);
		}
		
		auto memory = std::make_shared<CudaVirtualMemory>();
		memory->virtualSize = padded_size;
		memory->granularity = granularity_minimum;
		memory->granularity_recommended = granularity_recommended;
		memory->cptr = cptr;
		memory->comitted = 0;

		return memory;
	}

	void print(){
		println("cptr: {}", uint64_t(cptr));
		println("size: {}", virtualSize);
		println("physical allocations: {}", allocations.size());
		for(auto allocation : allocations){
			println("    offset: {:12}, size: {:12}, handle: {}", allocation.offset, allocation.size, uint64_t(allocation.handle));
		}
	}

	// commits <size> physical memory. 
	void commit(uint64_t requested_size, bool shrinkIfLarger = false){

		int64_t padded_requested_size = roundUp(requested_size, granularity);

		// remove unneeded physical allocations
		while(shrinkIfLarger && comitted > padded_requested_size){

			PhysicalAllocation allocation = allocations[allocations.size() - 1];

			cuMemUnmap(cptr + allocation.offset, allocation.size);
			cuMemRelease(allocation.handle);
			comitted -= allocation.size;

			println("Reducing physically comitted memory by {}", allocation.size);

			allocations.erase(allocations.begin() + allocations.size() - 1);
			
			// TODO: do we also need to revert cuMemSetAccess()?
		}


		int64_t required_additional_size = padded_requested_size - comitted;

		// Do we already have enough comitted memory?
		if(required_additional_size <= 0) return;

		bool hasRemappedAddressRange = false;
		if(padded_requested_size > virtualSize){
			// println("INFO: requested physical memory is larger than reserved virtual memory.");
			// println("Reserving a new virtual memory range.");
			// println("old state: ");
			// this->print();

			// first, unmap physical allocations and release virtual memory
			for(auto allocation : allocations){
				CUresult result = cuMemUnmap(cptr + allocation.offset, allocation.size);

				if (result != CUDA_SUCCESS) {
					println("ERROR: failed to cuMemUnmap() old physical mappings while trying to reserve larger virtual address range.");
					exit(634264);
				}
			}
			if (cuMemAddressFree(this->cptr, virtualSize) != CUDA_SUCCESS) {
				println("ERROR: failed to cuMemAddressFree while trying to reserve larger virtual address range");
				exit(634264);
			}
			this->cptr = 0;

			// now try to reserve a larger virtual address range
			int64_t requestedPhysicalWithExtraVirtualCapacity = 10 * padded_requested_size;
			auto result = cuMemAddressReserve(&this->cptr, requestedPhysicalWithExtraVirtualCapacity, 0, 0, 0);

			if(result != CUDA_SUCCESS){
				println("error {} while trying to reserve virtual memory. Requested virtual size: {}. Attempted to reserve {}", 
					int(result), requestedPhysicalWithExtraVirtualCapacity, padded_requested_size);
				exit(52457);
			}

			this->virtualSize = requestedPhysicalWithExtraVirtualCapacity;

			CUdevice cuDevice;
			cuDeviceGet(&cuDevice, 0);

			// now re-map the physical allocations to the new virtual address range
			for(PhysicalAllocation allocation : allocations){
				auto result = cuMemMap(cptr + allocation.offset, allocation.size, 0, allocation.handle, 0);

				if (result != CUDA_SUCCESS) {
					println("ERROR: failed to cuMemMap() physical mappings to new virtual address range while trying to reserve larger virtual address range.");
					exit(634264);
				}

				// make the new mapped range accessible
				CUmemAccessDesc accessDesc = {};
				accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
				accessDesc.location.id = cuDevice;
				accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
				if(cuMemSetAccess(cptr + allocation.offset, allocation.size, &accessDesc, 1) != CUDA_SUCCESS){
					println("ERROR: failed to make newly mapped memory accessible.");
					exit(63641);
				}
			}

			hasRemappedAddressRange = true;
		}


		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		CUmemAllocationProp prop = {};
		prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id   = cuDevice;

		while(comitted < padded_requested_size){

			int64_t allocationSize = std::min(uint64_t(required_additional_size), granularity_recommended);

			// create a little bit of physical memory
			CUmemGenericAllocationHandle allocHandle;
			auto result = cuMemCreate(&allocHandle, allocationSize, &prop, 0);
			
			if(result != CUDA_SUCCESS){
				println("error {} while trying to allocate physical memory.", int(result));
				exit(52458);
			}

			PhysicalAllocation allocation;
			allocation.offset = comitted;
			allocation.handle = allocHandle;
			allocation.size = allocationSize;

			allocations.push_back(allocation);

			// and map the physical memory
			if(cuMemMap(cptr + allocation.offset, allocation.size, 0, allocation.handle, 0) != CUDA_SUCCESS){
				println("ERROR: failed to map physical memory to virtual address range.");
				exit(63641);
			}

			// make the new memory accessible
			CUmemAccessDesc accessDesc = {};
			accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
			accessDesc.location.id = cuDevice;
			accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
			if(cuMemSetAccess(cptr + allocation.offset, allocation.size, &accessDesc, 1) != CUDA_SUCCESS){
				println("ERROR: failed to make newly mapped memory accessible.");
				exit(63641);
			}

			comitted += allocationSize;
		}

		// if(hasRemappedAddressRange){
		// 	println("new state: ");
		// 	this->print();
		// }
	}

};

