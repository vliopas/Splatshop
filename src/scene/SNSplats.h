#pragma once

#include <string>

#include "SceneNode.h"
#include "HostDeviceInterface.h"
#include "Splats.h"
#include "SplatsManagement.h"

using std::string;

struct SNSplats : public SceneNode{

	shared_ptr<Splats> splats = nullptr;
	GaussianDataManager dmng;

	SNSplats(string name, shared_ptr<Splats> splats)
		: SceneNode(name), dmng(name)
	{
		this->splats = splats;
		this->dmng.data.numSHCoefficients = splats->numSHCoefficients;
		this->dmng.data.shDegree = splats->shDegree;
	}

	SNSplats(string name)
		: SceneNode(name), dmng(name)
	{
		this->splats = nullptr;
	}

	~SNSplats(){
		println("Destroying SNSPlats node {}", name);
		dmng.destroy();
	}

	uint64_t getGpuMemoryUsage(){
		return dmng.getGpuMemoryUsage();
	}

	Box3 getBoundingBox(){
		return aabb;
	}

	

};