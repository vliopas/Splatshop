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

#include "Splats.h"

using namespace std;


struct GSPlyWriter{

	static string createHeader(Splats* splats){

		stringstream ss;

		println(ss, "ply");
		println(ss, "format binary_little_endian 1.0");
		println(ss, "element vertex {}", splats->numSplats);

		println(ss, "property float x");
		println(ss, "property float y");
		println(ss, "property float z");
		println(ss, "property float nx");
		println(ss, "property float ny");
		println(ss, "property float nz");
		println(ss, "property float f_dc_0");
		println(ss, "property float f_dc_1");
		println(ss, "property float f_dc_2");

		for(int i = 0; i < 45; i++){
			println(ss, "property float f_rest_{}", i);
		}

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

	static void write(string path, Splats* splats){

		string header = GSPlyWriter::createHeader(splats);

		uint64_t bytesPerSplat = 62 * 4;
		uint64_t splatsByteSize = bytesPerSplat * splats->numSplats;
		uint64_t filesize = header.size() + splatsByteSize;
		Buffer buffer = Buffer(filesize);
		
		memset(buffer.data, 0, buffer.size);
		memcpy(buffer.data, header.c_str(), header.size());

		uint64_t splatsStart = header.size();
		uint64_t numSHBytes = 45 * 4;

		uint64_t OFFSETS_POSITION   =  0 * 4;
		uint64_t OFFSETS_DC         =  6 * 4;
		uint64_t OFFSETS_SHs        =  9 * 4;
		uint64_t OFFSETS_OPACITY    = 54 * 4;
		uint64_t OFFSETS_SCALE      = 55 * 4;
		uint64_t OFFSETS_ROTATION   = 58 * 4;

		for (int i = 0; i < splats->numSplats; i++) {

			// POSITION
			auto pos = glm::vec4(
				splats->position->get<float>(12 * i + 0),
				splats->position->get<float>(12 * i + 4),
				splats->position->get<float>(12 * i + 8),
				1.0
			);
			pos = splats->world * pos;
			buffer.set<float>(pos.x, splatsStart + i * bytesPerSplat + OFFSETS_POSITION + 0);
			buffer.set<float>(pos.y, splatsStart + i * bytesPerSplat + OFFSETS_POSITION + 4);
			buffer.set<float>(pos.z, splatsStart + i * bytesPerSplat + OFFSETS_POSITION + 8);

			// SCALE
			float sx = log(splats->scale->get<float>(12 * i + 0));
			float sy = log(splats->scale->get<float>(12 * i + 4));
			float sz = log(splats->scale->get<float>(12 * i + 8));
			buffer.set<float>(sx, splatsStart + i * bytesPerSplat + OFFSETS_SCALE + 0);
			buffer.set<float>(sy, splatsStart + i * bytesPerSplat + OFFSETS_SCALE + 4);
			buffer.set<float>(sz, splatsStart + i * bytesPerSplat + OFFSETS_SCALE + 8);

			// ROTATION
			auto q = glm::quat(
				splats->rotation->get<float>(16 * i +  0),
				splats->rotation->get<float>(16 * i +  4),
				splats->rotation->get<float>(16 * i +  8),
				splats->rotation->get<float>(16 * i + 12)
			);
			auto q_world = glm::toQuat(splats->world);
			q = q_world * q;

			buffer.set<float>(q.w, splatsStart + i * bytesPerSplat + OFFSETS_ROTATION +  0);
			buffer.set<float>(q.x, splatsStart + i * bytesPerSplat + OFFSETS_ROTATION +  4);
			buffer.set<float>(q.y, splatsStart + i * bytesPerSplat + OFFSETS_ROTATION +  8);
			buffer.set<float>(q.z, splatsStart + i * bytesPerSplat + OFFSETS_ROTATION + 12);

			// COLOR
			float r = float(splats->color->get<uint8_t>(4 * i + 0));
			float g = float(splats->color->get<uint8_t>(4 * i + 1));
			float b = float(splats->color->get<uint8_t>(4 * i + 2));
			float opacity = float(splats->color->get<uint8_t>(4 * i + 3));
			r = (r / 255.0 - 0.5) / 0.28209479177387814;
			g = (g / 255.0 - 0.5) / 0.28209479177387814;
			b = (b / 255.0 - 0.5) / 0.28209479177387814;

			buffer.set<float>(r, splatsStart + i * bytesPerSplat + OFFSETS_DC +  0);
			buffer.set<float>(g, splatsStart + i * bytesPerSplat + OFFSETS_DC +  4);
			buffer.set<float>(b, splatsStart + i * bytesPerSplat + OFFSETS_DC +  8);
			// OPACITY
			opacity = -log(255.0 / opacity - 1.0);
			buffer.set<float>(opacity, splatsStart + i * bytesPerSplat + OFFSETS_OPACITY);
		}

		writeBinaryFile(path, buffer);
	}

};