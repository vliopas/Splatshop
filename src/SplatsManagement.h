#pragma once

#include "HostDeviceInterface.h"
#include "CudaVirtualMemory.h"
#include "CURuntime.h"



struct GaussianDataManager{

	string name;

	shared_ptr<CudaVirtualMemory> vm_position            = nullptr;
	shared_ptr<CudaVirtualMemory> vm_scale               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_quaternion          = nullptr;
	shared_ptr<CudaVirtualMemory> vm_color               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_depth               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_flags               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_sphericalHarmonics  = nullptr;
	// shared_ptr<CudaVirtualMemory> vm_cov3d         = nullptr;
	// shared_ptr<CudaVirtualMemory> vm_batches       = nullptr;

	GaussianData data;

	GaussianDataManager(string name){

		vm_position            = CURuntime::allocVirtual(format("[{}] position",   name)); 
		vm_scale               = CURuntime::allocVirtual(format("[{}] scale",      name)); 
		vm_quaternion          = CURuntime::allocVirtual(format("[{}] quaternion", name)); 
		vm_color               = CURuntime::allocVirtual(format("[{}] color",      name)); 
		vm_depth               = CURuntime::allocVirtual(format("[{}] depth",      name)); 
		vm_flags               = CURuntime::allocVirtual(format("[{}] flags",      name)); 
		vm_sphericalHarmonics  = CURuntime::allocVirtual(format("[{}] spherical harmonics",name)); 
		// vm_cov3d      = CURuntime::allocVirtual(format("[{}] cov3d",      name)); 
		// vm_batches    = CURuntime::allocVirtual(format("[{}] batches",    name)); 

		data.position           = (decltype(data.position))    vm_position->cptr;
		data.scale              = (decltype(data.scale))       vm_scale->cptr;
		data.quaternion         = (decltype(data.quaternion))  vm_quaternion->cptr;
		data.color              = (decltype(data.color))       vm_color->cptr;
		data.depth              = (decltype(data.depth))       vm_depth->cptr;
		data.flags              = (decltype(data.flags))       vm_flags->cptr;
		data.sphericalHarmonics = (decltype(data.sphericalHarmonics))       vm_sphericalHarmonics->cptr;
		// data.cov3d         = (decltype(data.cov3d))       vm_cov3d->cptr;
		// data.batches        = (decltype(data.batches))     vm_batches->cptr;

		this->name = name;
	}

	~GaussianDataManager(){
		// TODO: free memory...or not
	}

	void destroy(){

		CURuntime::free(vm_position);
		CURuntime::free(vm_scale);
		CURuntime::free(vm_quaternion);
		CURuntime::free(vm_color);
		CURuntime::free(vm_depth);
		CURuntime::free(vm_flags);
		CURuntime::free(vm_sphericalHarmonics);
		// CURuntime::free(vm_cov3d);
		// CURuntime::free(vm_batches);

		vm_position            = nullptr;
		vm_scale               = nullptr;
		vm_quaternion          = nullptr;
		vm_color               = nullptr;
		vm_depth               = nullptr;
		vm_flags               = nullptr;
		vm_sphericalHarmonics  = nullptr;
		// vm_cov3d          = nullptr;
		// vm_batches        = nullptr;

		data.count = 0;
	}

	// commit sufficient physical memory for the given amount of splats
	void commit(uint64_t numSplats){
		vm_position            ->commit(sizeof(*data.position  ) * numSplats, true);
		vm_scale               ->commit(sizeof(*data.scale     ) * numSplats, true);
		vm_quaternion          ->commit(sizeof(*data.quaternion) * numSplats, true);
		vm_color               ->commit(sizeof(*data.color     ) * numSplats, true);
		vm_depth               ->commit(sizeof(*data.depth     ) * numSplats, true);
		vm_flags               ->commit(sizeof(*data.flags     ) * numSplats, true);
		vm_sphericalHarmonics  ->commit(data.numSHCoefficients * sizeof(float) * numSplats, true);
		// vm_cov3d     ->commit(sizeof(*data.cov3d     ) * numSplats, true);

		// int numBatches = (numSplats + SPLATLET_SIZE - 1) / SPLATLET_SIZE;
		// vm_batches   ->commit(sizeof(*data.batches   ) * numBatches, true);

		// may need to update pointers, in case virtual address range changed.
		data.position            = (decltype(data.position))             vm_position->cptr;
		data.scale               = (decltype(data.scale))                vm_scale->cptr;
		data.quaternion          = (decltype(data.quaternion))           vm_quaternion->cptr;
		data.color               = (decltype(data.color))                vm_color->cptr;
		data.depth               = (decltype(data.depth))                vm_depth->cptr;
		data.flags               = (decltype(data.flags))                vm_flags->cptr;
		data.sphericalHarmonics  = (decltype(data.sphericalHarmonics))   vm_sphericalHarmonics->cptr;
		// data.cov3d         = (decltype(data.cov3d))         vm_cov3d->cptr;
		// data.batches       = (decltype(data.batches))       vm_batches->cptr;
	}

	uint64_t getGpuMemoryUsage(){
		uint64_t usage = 0;
		
		usage += vm_position->comitted;
		usage += vm_scale->comitted;
		usage += vm_quaternion->comitted;
		usage += vm_color->comitted;
		usage += vm_depth->comitted;
		usage += vm_flags->comitted;
		usage += vm_sphericalHarmonics->comitted;
		// usage += vm_cov3d->comitted;
		// usage += vm_batches->comitted;
		
		return usage;
	}

};

