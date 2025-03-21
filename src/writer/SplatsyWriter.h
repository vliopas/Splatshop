#pragma once


#include <cmath>
#include <iostream>
#include <print>
#include <format>
#include <memory>
#include <string>
#include <sstream>

#include "unsuck.hpp"

#include <glm/gtx/quaternion.hpp>
#include "json/json.hpp"

#include "Splats.h"
#include "./scene/SceneNode.h"
#include "./scene/SNSplats.h"
#include "./scene/Scene.h"
#include "AssetLibrary.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace std;


struct SplatsyWriter{

	static void write(SNSplats* node, json& j_scene, fstream& fout){

		GaussianData& data = node->dmng.data;

		Buffer tmpbuffer(16 * data.count);

		json j_attributes = json::object();

		// Let's really remove flats flagged as "deleted" upon saving.
		vector<uint32_t> acceptedSplatIndices;
		{
			vector<uint32_t> flags(data.count, 0);
			cuMemcpyDtoH(flags.data(), (CUdeviceptr)data.flags, 4 * data.count);

			for(int i = 0; i < data.count; i++){
				bool isDeleted = flags[i] & FLAGS_DELETED;

				if(!isDeleted){
					acceptedSplatIndices.push_back(i);
				}
			}
		}
		
		{ // POSITION
			auto source = data.position;

			int64_t allByteLength = sizeof(decltype(*source)) * data.count;
			int64_t acceptedByteLength = sizeof(decltype(*source)) * acceptedSplatIndices.size();

			cuMemcpyDtoH(tmpbuffer.data, (CUdeviceptr)source, allByteLength);
			j_attributes["position"] = {
				{"byteOffset", int64_t(fout.tellg())},
				{"byteLength", acceptedByteLength},
			};

			// There might be gaps between accepted splats.
			// Close these gaps by shifting the accepted splats
			for(int targetIndex = 0; targetIndex < acceptedSplatIndices.size(); targetIndex++){
				int sourceIndex = acceptedSplatIndices[targetIndex];

				// shouldnt memcpy if source and target are same
				if(targetIndex != sourceIndex){
					uint8_t* sourceSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * sourceIndex;
					uint8_t* targetSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * targetIndex;

					memcpy(targetSplat, sourceSplat, sizeof(decltype(*source)));
				}
			}

			fout.write(tmpbuffer.data_char, acceptedByteLength);
		}

		{ // SCALE
			auto source = data.scale;

			int64_t allByteLength = sizeof(decltype(*source)) * data.count;
			int64_t acceptedByteLength = sizeof(decltype(*source)) * acceptedSplatIndices.size();

			cuMemcpyDtoH(tmpbuffer.data, (CUdeviceptr)source, allByteLength);
			j_attributes["scale"] = {
				{"byteOffset", int64_t(fout.tellg())},
				{"byteLength", acceptedByteLength},
			};

			// There might be gaps between accepted splats.
			// Close these gaps by shifting the accepted splats
			for(int targetIndex = 0; targetIndex < acceptedSplatIndices.size(); targetIndex++){
				int sourceIndex = acceptedSplatIndices[targetIndex];

				// shouldnt memcpy if source and target are same
				if(targetIndex != sourceIndex){
					uint8_t* sourceSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * sourceIndex;
					uint8_t* targetSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * targetIndex;

					memcpy(targetSplat, sourceSplat, sizeof(decltype(*source)));
				}
			}

			fout.write(tmpbuffer.data_char, acceptedByteLength);
		}

		{ // ROTATION
			auto source = data.quaternion;

			int64_t allByteLength = sizeof(decltype(*source)) * data.count;
			int64_t acceptedByteLength = sizeof(decltype(*source)) * acceptedSplatIndices.size();

			cuMemcpyDtoH(tmpbuffer.data, (CUdeviceptr)source, allByteLength);
			j_attributes["rotation"] = {
				{"byteOffset", int64_t(fout.tellg())},
				{"byteLength", acceptedByteLength},
			};

			// There might be gaps between accepted splats.
			// Close these gaps by shifting the accepted splats
			for(int targetIndex = 0; targetIndex < acceptedSplatIndices.size(); targetIndex++){
				int sourceIndex = acceptedSplatIndices[targetIndex];

				// shouldnt memcpy if source and target are same
				if(targetIndex != sourceIndex){
					uint8_t* sourceSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * sourceIndex;
					uint8_t* targetSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * targetIndex;

					memcpy(targetSplat, sourceSplat, sizeof(decltype(*source)));
				}
			}

			fout.write(tmpbuffer.data_char, acceptedByteLength);
		}

		{ // COLOR
			auto source = data.color;

			int64_t allByteLength = sizeof(decltype(*source)) * data.count;
			int64_t acceptedByteLength = sizeof(decltype(*source)) * acceptedSplatIndices.size();

			cuMemcpyDtoH(tmpbuffer.data, (CUdeviceptr)source, allByteLength);
			j_attributes["color"] = {
				{"byteOffset", int64_t(fout.tellg())},
				{"byteLength", acceptedByteLength},
			};

			// There might be gaps between accepted splats.
			// Close these gaps by shifting the accepted splats
			for(int targetIndex = 0; targetIndex < acceptedSplatIndices.size(); targetIndex++){
				int sourceIndex = acceptedSplatIndices[targetIndex];

				// shouldnt memcpy if source and target are same
				if(targetIndex != sourceIndex){
					uint8_t* sourceSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * sourceIndex;
					uint8_t* targetSplat = tmpbuffer.ptr + sizeof(decltype(*source)) * targetIndex;

					memcpy(targetSplat, sourceSplat, sizeof(decltype(*source)));
				}
			}

			fout.write(tmpbuffer.data_char, acceptedByteLength);
		}

		json j_transform = json::array();
		for(int i = 0; i < 16; i++){
			float value = ((float*)&data.transform)[i];
			j_transform.push_back(value);
		}

		json j_node = {
			{"name", node->name},
			{"count", acceptedSplatIndices.size()},
			{"type", "SNSplats"},
			{"transform", j_transform},
			{"attributes", j_attributes},
		};

		j_scene.push_back(j_node);
	}

	static void write(string path, Scene& scene, OrbitControls controls){
		
		println("=============================");
		println("=== Saving Splatsy format ===");
		println("=============================");

		vector<SceneNode*> nodes;
		scene.forEach<SceneNode>([&](SceneNode* node){
			if(!node->hidden && node != scene.root.get()){
				nodes.push_back(node);
			}
		});


		auto fout = fstream(path, ios::out | ios::binary);

		uint32_t headerSize = 0;
		fout.write(reinterpret_cast<char*>(&headerSize), 4);
		
		json j_scene;
		for(SceneNode* node : nodes){
			if(types_match<SNSplats*>(node)){
				write((SNSplats*)node, j_scene, fout);
			}
		}

		json j_assets;
		for(shared_ptr<SceneNode> node : AssetLibrary::assets){
			if(types_match<SNSplats*>(node.get())){
				write((SNSplats*)node.get(), j_assets, fout);
			}
		}

		json j_target = {controls.target.x, controls.target.y, controls.target.z};
		json j_camera = {
			{"yaw", controls.yaw},
			{"pitch", controls.pitch},
			{"radius", controls.radius},
			{"target", j_target},
		};

		json j;
		j["scene"] = j_scene;
		j["assets"] = j_assets;
		j["camera"] = j_camera;



		string strJson = j.dump(4);
		println("{}", strJson);

		fout.write(strJson.c_str(), strJson.size());

		uint32_t realHeaderSize = strJson.size();
		fout.seekg(0);
		fout.write(reinterpret_cast<char*>(&realHeaderSize), 4);
		fout.close();



		println("=============================");


	}

};