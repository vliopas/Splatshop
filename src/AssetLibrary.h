#pragma once

#include <vector>
#include <string>
#include <memory>

#include "scene/SceneNode.h"

using std::vector;
using std::string;
using std::shared_ptr;


struct AssetLibrary{

	inline static vector<shared_ptr<SceneNode>> assets;

};