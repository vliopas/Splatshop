#pragma once

#include <cmath>
#include <iostream>
#include <print>
#include <format>
#include <memory>
#include <string>
#include <sstream>
#include <thread>
#include <set>

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

// Merges all nodes into a single plky
namespace SplatsyPlyWriter{

	inline string createHeader(int64_t numSplats){

		stringstream ss;

		println(ss, "ply");
		println(ss, "format binary_little_endian 1.0");
		println(ss, "element vertex {:012}", numSplats); 

		println(ss, "property float x");
		println(ss, "property float y");
		println(ss, "property float z");
		println(ss, "property float nx");
		println(ss, "property float ny");
		println(ss, "property float nz");
		println(ss, "property float f_dc_0");
		println(ss, "property float f_dc_1");
		println(ss, "property float f_dc_2");
		println(ss, "property float opacity");
		println(ss, "property float scale_0");
		println(ss, "property float scale_1");
		println(ss, "property float scale_2");
		println(ss, "property float rot_0");
		println(ss, "property float rot_1");
		println(ss, "property float rot_2");
		println(ss, "property float rot_3");
		println(ss, "end_header");

		return ss.str();
	}

	inline void write_batched(SNSplats* node, json& j_scene, string path){

		CUevent event_memcopied;
		cuEventCreate(&event_memcopied, CU_EVENT_DEFAULT);

		GaussianData& data = node->dmng.data;

		int64_t MAX_BATCHSIZE = 100'000;

		println("## Writing node {}", node->name);

		auto t_start = now();

		int64_t numSplats = data.count;
		string header = createHeader(numSplats);

		// uint64_t bytesPerSplat = 62 * 4; // with degree 3 SHs
		uint64_t bytesPerSplat = 17 * 4; // with no SHs
		uint64_t splatsByteSize = bytesPerSplat * numSplats;
		uint64_t filesize = header.size() + splatsByteSize;
		shared_ptr<Buffer> buffer = make_shared<Buffer>(filesize);
		
		memset(buffer->data, 0, buffer->size);
		memcpy(buffer->data, header.c_str(), header.size());

		uint64_t splatsStart = header.size();
		// uint64_t numSHBytes = 45 * 4;

		uint64_t OFFSETS_POSITION   =  0 * 4;
		uint64_t OFFSETS_DC         =  6 * 4;
		uint64_t OFFSETS_SHs        =  9 * 4;
		// with degree 3 SHs
		// uint64_t OFFSETS_OPACITY    = 54 * 4;
		// uint64_t OFFSETS_SCALE      = 55 * 4;
		// uint64_t OFFSETS_ROTATION   = 58 * 4;
		// with no SHs
		uint64_t OFFSETS_OPACITY    = 9 * 4;
		uint64_t OFFSETS_SCALE      = 10 * 4;
		uint64_t OFFSETS_ROTATION   = 13 * 4;

		shared_ptr<Buffer> position_loading = make_shared<Buffer>(12 * MAX_BATCHSIZE);
		shared_ptr<Buffer> scale_loading    = make_shared<Buffer>(12 * MAX_BATCHSIZE);
		shared_ptr<Buffer> rotation_loading = make_shared<Buffer>(16 * MAX_BATCHSIZE);
		shared_ptr<Buffer> color_loading    = make_shared<Buffer>( 8 * MAX_BATCHSIZE);
		shared_ptr<Buffer> flags_loading    = make_shared<Buffer>( 4 * MAX_BATCHSIZE);

		shared_ptr<Buffer> position_processing = make_shared<Buffer>(12 * MAX_BATCHSIZE);
		shared_ptr<Buffer> scale_processing    = make_shared<Buffer>(12 * MAX_BATCHSIZE);
		shared_ptr<Buffer> rotation_processing = make_shared<Buffer>(16 * MAX_BATCHSIZE);
		shared_ptr<Buffer> color_processing    = make_shared<Buffer>( 8 * MAX_BATCHSIZE);
		shared_ptr<Buffer> flags_processing    = make_shared<Buffer>( 4 * MAX_BATCHSIZE);

		vec3 t_dscale;
		quat t_rotation;
		vec3 t_translation;
		vec3 t_skew;
		vec4 t_perspective;
		glm::decompose(node->transform, t_dscale, t_rotation, t_translation, t_skew, t_perspective);
		t_rotation = glm::conjugate(t_rotation);

		auto downloadBatch = [&](int64_t firstSplat, int64_t numSplats){
			cuMemcpyDtoHAsync(position_loading->data , (CUdeviceptr)(data.position   + firstSplat), 12 * numSplats, 0);
			cuMemcpyDtoHAsync(scale_loading->data    , (CUdeviceptr)(data.scale      + firstSplat), 12 * numSplats, 0);
			cuMemcpyDtoHAsync(rotation_loading->data , (CUdeviceptr)(data.quaternion + firstSplat), 16 * numSplats, 0);
			cuMemcpyDtoHAsync(color_loading->data    , (CUdeviceptr)(data.color      + firstSplat),  8 * numSplats, 0);
			cuMemcpyDtoHAsync(flags_loading->data    , (CUdeviceptr)(data.flags      + firstSplat),  4 * numSplats, 0);
			cuEventRecord(event_memcopied, 0);
		};

		// Start downloading the first batch
		downloadBatch(0, min(MAX_BATCHSIZE, int64_t(data.count)));

		int64_t numBatches = (data.count + MAX_BATCHSIZE - 1) / MAX_BATCHSIZE;
		int64_t numWritten = 0;
		for(int batchIndex = 0; batchIndex < numBatches; batchIndex++){

			// wait until current batch is downloaded
			cuStreamWaitEvent(0, event_memcopied, CU_EVENT_WAIT_DEFAULT);
			
			std::swap(position_loading , position_processing);
			std::swap(scale_loading    , scale_processing);
			std::swap(rotation_loading , rotation_processing);
			std::swap(color_loading    , color_processing);
			std::swap(flags_loading    , flags_processing);

			{ // then start downloading next batch before processing current batch
				int64_t nextBatchStart = (batchIndex + 1) * MAX_BATCHSIZE;
				int64_t nextBatchSize = min(data.count - nextBatchStart, MAX_BATCHSIZE);

				if(nextBatchSize > 0){
					downloadBatch(nextBatchStart, nextBatchSize);
				}
			}

			// now process current batch
			int64_t batchStart = batchIndex * MAX_BATCHSIZE;
			int64_t batchSize = min(data.count - batchStart, MAX_BATCHSIZE);

			for(int i = 0; i < batchSize; i++){

				constexpr uint32_t FLAGS_DELETED = 1 <<  3;
				uint32_t flags = flags_processing->get<uint32_t>(4 * i);
				bool isDeleted = flags & FLAGS_DELETED;

				if(isDeleted) continue;
				
				// POSITION
				auto pos = glm::vec4(
					position_processing->get<float>(12 * i + 0),
					position_processing->get<float>(12 * i + 4),
					position_processing->get<float>(12 * i + 8),
					1.0
				);
				pos = node->transform * pos;
				buffer->set<float>(pos.x, splatsStart + numWritten * bytesPerSplat + OFFSETS_POSITION + 0);
				buffer->set<float>(pos.y, splatsStart + numWritten * bytesPerSplat + OFFSETS_POSITION + 4);
				buffer->set<float>(pos.z, splatsStart + numWritten * bytesPerSplat + OFFSETS_POSITION + 8);

				// SCALE
				vec3 s = {
					t_dscale.x * scale_processing->get<float>(12 * i + 0),
					t_dscale.y * scale_processing->get<float>(12 * i + 4),
					t_dscale.z * scale_processing->get<float>(12 * i + 8),
				};
				vec3 s_out = {
					log(s.x),
					log(s.y),
					log(s.z),
				};
				buffer->set<float>(s_out.x, splatsStart + numWritten * bytesPerSplat + OFFSETS_SCALE + 0);
				buffer->set<float>(s_out.y, splatsStart + numWritten * bytesPerSplat + OFFSETS_SCALE + 4);
				buffer->set<float>(s_out.z, splatsStart + numWritten * bytesPerSplat + OFFSETS_SCALE + 8);

				// ROTATION
				auto q = glm::quat(
					rotation_processing->get<float>(16 * i +  0),
					rotation_processing->get<float>(16 * i +  4),
					rotation_processing->get<float>(16 * i +  8),
					rotation_processing->get<float>(16 * i + 12)
				);
				q = t_rotation * q;
				// float l = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
				// l = 1.0f;

				buffer->set<float>(q.w, splatsStart + numWritten * bytesPerSplat + OFFSETS_ROTATION +  0);
				buffer->set<float>(q.x, splatsStart + numWritten * bytesPerSplat + OFFSETS_ROTATION +  4);
				buffer->set<float>(q.y, splatsStart + numWritten * bytesPerSplat + OFFSETS_ROTATION +  8);
				buffer->set<float>(q.z, splatsStart + numWritten * bytesPerSplat + OFFSETS_ROTATION + 12);

				// COLOR
				float r       = float(color_processing->get<uint16_t>(8 * i + 0)) / 65536.0f;
				float g       = float(color_processing->get<uint16_t>(8 * i + 2)) / 65536.0f;
				float b       = float(color_processing->get<uint16_t>(8 * i + 4)) / 65536.0f;
				float opacity = float(color_processing->get<uint16_t>(8 * i + 6)) / 65536.0f;
				float dc0 = (r - 0.5) / 0.28209479177387814;
				float dc1 = (g - 0.5) / 0.28209479177387814;
				float dc2 = (b - 0.5) / 0.28209479177387814;

				buffer->set<float>(dc0, splatsStart + numWritten * bytesPerSplat + OFFSETS_DC +  0);
				buffer->set<float>(dc1, splatsStart + numWritten * bytesPerSplat + OFFSETS_DC +  4);
				buffer->set<float>(dc2, splatsStart + numWritten * bytesPerSplat + OFFSETS_DC +  8);

				// OPACITY
				float opacity_out = -log(1.0 / opacity - 1.0);
				buffer->set<float>(opacity_out, splatsStart + numWritten * bytesPerSplat + OFFSETS_OPACITY);

				numWritten++;

				if(numWritten % 100'000 == 0){
					int index = batchIndex * MAX_BATCHSIZE + i;
					println("# Debug Splat, index {}", index);
					println("    position:     {:.3f}, {:.3f}, {:.3f}           ", pos.x, pos.y, pos.z);
					println("    scale:        {:.3f}, {:.3f}, {:.3f}           -> {:.3f}, {:.3f}, {:.3f}", s.x, s.y, s.z, s_out.x, s_out.y, s_out.z);
					println("    quaternion:   {:.3f}, {:.3f}, {:.3f}, {:.3f}   -> (x, y, z, w) stored as (w, x, y, z)", q.x, q.y, q.z, q.w);
					println("    color(rgb):   {:3}, {:3}, {:3}                 -> {:3}, {:3}, {:3}", r, g, b, dc0, dc1, dc2);
					println("    opacity:      {:.3f}                           -> {:.3f}", opacity, opacity_out);

				}
			}
		}

		// auto t_createdBinary = now();
		printElapsedTime("Created binary - Time for saving: {:.1f}", t_start);

		// We allocate memory for all splats, including those that are flagged as deleted.
		// Reduce reported size of the buffer object to the actual amount of bytes used. 
		buffer->size = splatsStart + numWritten * bytesPerSplat;

		shared_ptr<jthread> t = make_shared<jthread>([path, buffer, numWritten](){
			shared_ptr<Buffer> _buffer = buffer;

			// update header with actual amount of points
			string header = createHeader(numWritten);
			memcpy(buffer->data, header.c_str(), header.size());

			// println("writing to path: {}; bytes: {:L}", path, _buffer->size);
			writeBinaryFile(path, *_buffer);

		});
		// t.detach();

		// auto t_wroteFile = now();
		printElapsedTime("Write file to disk - Time for saving: {:.1f}", t_start);

		// json j_transform = json::array();
		// for(int i = 0; i < 16; i++){
		// 	float value = ((float*)&data.transform)[i];
		// 	j_transform.push_back(value);
		// }

		// json j_node = {
		// 	{"name", node->name},
		// 	// {"count", numWritten},
		// 	{"type", "SNSplats"},
		// 	{"transform", j_transform},
		// 	//{"attributes", j_attributes},
		// };

		// j_scene.push_back(j_node);

	}

	inline void write(string path, Scene& scene, OrbitControls controls){
		
		if(!iEndsWith(path, ".ply")){
			path = path + ".ply";
		}

		vector<shared_ptr<SNSplats>> nodes;
		scene.process<SNSplats>([&](shared_ptr<SNSplats> node){
			if(node->hidden) return;
			if(!node->visible) return;

			nodes.push_back(node);
		});

		int64_t numTotalSplats = 0;
		for(auto node : nodes){
			
		}

	}

};