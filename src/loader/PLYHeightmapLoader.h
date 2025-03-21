#pragma once


#include <cmath>
#include <iostream>
#include <filesystem>
#include <print>
#include <format>
#include <memory>
#include <string>
#include <thread>

#include "unsuck.hpp"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

using namespace std;
using glm::vec3;

struct PlyHeightmapHeader{
	int64_t bytesPerRecord   = 0;
	int64_t numVertices      = 0;
	int64_t headerSize       = 0;
	int64_t OFFSETS_POSITION = 0;
	int64_t OFFSETS_COLOR    = 0;
};

struct PlyHeightmap{
	int64_t numVertices = 0;
	vector<vec3> positions;
	vector<uint32_t> colors;
};

struct PLYHeightmapLoader{

	static PlyHeightmapHeader readHeader(string path){

		// AttributesIndices attributes;
		int numSplats = 0;

		auto potentialHeaderData = readBinaryFile(path, 0, 10'000);
		std::string strPotentialHeader(potentialHeaderData->data_char, potentialHeaderData->size);

		uint64_t posStartEndHeader = strPotentialHeader.find("end_header");
		uint64_t posEndHeader = strPotentialHeader.find('\n', posStartEndHeader);
		
		auto headerData = readBinaryFile(path, 0, posEndHeader);
		string strHeader(headerData->data_char, headerData->size);

		vector<string> lines = split(strHeader, '\n');
		int numVertices = 0;
		int byteOffset = 0;
		int OFFSETS_POSITION = 0;
		int OFFSETS_COLOR    = 0;

		for(string line : lines){

			vector<string> tokens = split(line, ' ');

			if(tokens[0] == "element" && tokens[1] == "vertex"){
				numVertices = std::stoi(tokens[2]);
			}else if(tokens[0] == "property"){
				string type = tokens[1];
				string name = tokens[2];

				if(name == "x"){
					// attributes.x = numAttributesProcessed;
					OFFSETS_POSITION = byteOffset;
				}else if(name == "y"){
					// attributes.y = numAttributesProcessed;
				}else if(name == "z"){
					// attributes.z = numAttributesProcessed;
				}else if(name == "red"){
					OFFSETS_COLOR = byteOffset;
				}

				if(tokens[1] == "float"){
					byteOffset += 4;
				}else if(tokens[1] == "uchar"){
					byteOffset += 1;
				}else if(tokens[1] == "list"){
					byteOffset += 0;
				}else{
					cout << format("type not implemented: {} \n", type);
					exit(123);
				}

				// numAttributesProcessed++;
			}

		}


		PlyHeightmapHeader header;
		header.numVertices       = numVertices;
		header.headerSize        = posEndHeader;
		header.OFFSETS_POSITION  = OFFSETS_POSITION;
		header.OFFSETS_COLOR     = OFFSETS_COLOR;
		header.bytesPerRecord    = byteOffset;

		return header;
	}

	static PlyHeightmap load(string path){

		auto header = readHeader(path);

		auto buffer = readBinaryFile(path, header.headerSize + 1);

		PlyHeightmap map;

		for(int i = 0; i < header.numVertices; i++){
			float x = buffer->get<float>(i * header.bytesPerRecord + 0);
			float y = buffer->get<float>(i * header.bytesPerRecord + 4);
			float z = buffer->get<float>(i * header.bytesPerRecord + 8);
			uint32_t rgba = buffer->get<uint32_t>(i * header.bytesPerRecord + header.OFFSETS_COLOR);

			map.positions.push_back({x, y, z});
			map.colors.push_back(rgba);
			map.numVertices++;
		}

		return map;
	}


};