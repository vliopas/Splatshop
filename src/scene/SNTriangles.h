#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "HostDeviceInterface.h"

using std::string;
using std::vector;
using glm::ivec2;

struct SNTriangles : public SceneNode{

	TriangleData data;
	TriangleMaterial material;
	// Texture texture;
	
	SNTriangles(string name) : SceneNode(name){
		
	}

	void set(vector<vec3> positions, vector<vec2> uvs){

		if(data.position != nullptr) cuMemFree((CUdeviceptr)data.position);
		if(data.uv != nullptr)       cuMemFree((CUdeviceptr)data.uv);

		uint64_t size_triangles = sizeof(vec3) * positions.size();
		uint64_t size_uvs       = sizeof(vec2) * uvs.size();

		data.position = (vec3*)CURuntime::alloc("SNTriangles.position", size_triangles + 4 * positions.size());
		data.uv       = (vec2*)CURuntime::alloc("SNTriangles.uv", size_uvs);

		cuMemcpyHtoD((CUdeviceptr)data.position, &positions[0], size_triangles);
		cuMemcpyHtoD((CUdeviceptr)data.uv, &uvs[0], size_uvs);

		data.count = positions.size() / 3;

		for(int i = 0; i < positions.size(); i++){
			vec3 pos = positions[i];
			aabb.min.x = min(aabb.min.x, pos.x);
			aabb.min.y = min(aabb.min.y, pos.y);
			aabb.min.z = min(aabb.min.z, pos.z);
			aabb.max.x = max(aabb.max.x, pos.x);
			aabb.max.y = max(aabb.max.y, pos.y);
			aabb.max.z = max(aabb.max.z, pos.z);
		}
	}

	void set(vector<vec3> positions, vector<vec2> uvs, vector<uint32_t> colors){

		CURuntime::free((CUdeviceptr)data.position);
		CURuntime::free((CUdeviceptr)data.uv);
		CURuntime::free((CUdeviceptr)data.colors);

		uint64_t size_triangles = sizeof(vec3) * positions.size();
		uint64_t size_uvs       = sizeof(vec2) * uvs.size();
		uint64_t size_colors    = sizeof(uint32_t) * uvs.size();

		data.position = (vec3*)CURuntime::alloc("SNTriangles.position", size_triangles + 4 * positions.size());
		data.uv       = (vec2*)CURuntime::alloc("SNTriangles.uv", size_uvs);
		data.colors   = (uint32_t*)CURuntime::alloc("SNTriangles.colors", size_colors);

		cuMemcpyHtoD((CUdeviceptr)data.position, &positions[0], size_triangles);
		cuMemcpyHtoD((CUdeviceptr)data.uv, &uvs[0], size_uvs);
		cuMemcpyHtoD((CUdeviceptr)data.colors, &colors[0], size_colors);

		data.count = positions.size() / 3;
	}

	void setTexture(ivec2 size, void* data){

		int width = size.x;
		int height= size.y;
		
		CUdeviceptr cptr = CURuntime::alloc("SNTriangles.texture", width * height * 4);
		cuMemcpyHtoD(cptr, data, width * height * 4);

		material.texture.width = width;
		material.texture.height = height;
		material.texture.data = (uint8_t*)cptr;
		material.mode = MATERIAL_MODE_TEXTURED;
	}

	uint64_t getGpuMemoryUsage(){
		return 0;
	}

};