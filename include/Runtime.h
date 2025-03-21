
#pragma once

#include <string>
#include <unordered_map>
#include <map>

#include "GLFW/glfw3.h"
#include "glm/common.hpp"

#include "OrbitControls.h"
#include "MouseEvents.h"

using namespace std;

struct Timings{

	int averageCount = 50;
	uint64_t counter = 0;

	unordered_map<string, vector<float>> entries;

	void add(string label, float milliseconds){

		entries[label].resize(averageCount);

		int entryPos = counter % averageCount;

		entries[label][entryPos] += milliseconds;
	}

	void newFrame(){

		counter++;
		int entryPos = counter % averageCount;

		for(auto& [label, list] : entries){
			list[entryPos] = 0.0f;
		}
	}

	float getAverage(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float sum = 0.0f;
		for(float value : entries[label]){
			sum += value;
		}

		float avg = sum / averageCount;

		return avg;
	}

};

struct StartStop{
	uint64_t t_start;
	uint64_t t_end;
};

struct Runtime{

	struct GuiItem{
		uint32_t type = 0;
		float min = 0.0;
		float max = 1.0;
		float oldValue = 0.5;
		float value = 0.5;
		string label = "";
	};

	inline static vector<int> keyStates = vector<int>(65536, 0);
	inline static int mods = 0;
	inline static vector<int> frame_keys = vector<int>();
	inline static vector<int> frame_scancodes = vector<int>();
	inline static vector<int> frame_actions = vector<int>();
	inline static vector<int> frame_mods = vector<int>();
	inline static OrbitControls* controls = new OrbitControls();
	inline static MouseEvents mouseEvents;
	inline static unordered_map<string, string> debugValues;
	inline static vector<std::pair<string, string>> debugValueList;
	inline static int totalTileFragmentCount;

	inline static glm::dvec2 mousePosition = {0.0, 0.0};
	inline static int mouseButtons = 0;

	inline static int64_t numVisibleSplats = 0;
	inline static int64_t numVisibleFragments = 0;
	inline static int64_t numSelectedSplats = 0;
	inline static int64_t numRenderedTriangles = 0;

	struct Timing{
		string label;
		float milliseconds;
	};
	inline static bool measureTimings;
	inline static Timings timings;
	inline static vector<StartStop> profileTimings;

	// textures
	inline static uint32_t gltex_symbols;
	inline static int gltex_symbols_width;
	inline static int gltex_symbols_height;
	inline static uint32_t gltex_symbols_32x32;
	inline static int gltex_symbols_32x32_width;
	inline static int gltex_symbols_32x32_height;

	Runtime(){
		
	}

	static Runtime* getInstance(){
		static Runtime* instance = new Runtime();

		return instance;
	}

	static int getKeyAction(int key){
		for(int i = 0; i < frame_keys.size(); i++){
			if(frame_keys[i] == key){
				return frame_actions[i];
			}
		}

		return -1;
	}

	static int getKeyAction(char key){

		for(int i = 0; i < frame_keys.size(); i++){

			const char* keyname = glfwGetKeyName(frame_keys[i], frame_scancodes[i]);
			// auto length = strnlen_s(keyname, 4); // strnlen_s not supported on linux

			if(keyname != nullptr && keyname[0] != 0 && keyname[0] == key){
				return frame_actions[i];
			}

			// if(length == 1 && keyname[0] == key){
			// 	return frame_actions[i];
			// }

			// if(frame_keys[i] == key){
			// 	return frame_actions[i];
			// }
		}

		return -1;
	}

};