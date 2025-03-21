#pragma once

#include <vector>
#include <memory>

#include "scene/SceneNode.h"

#include "glm/common.hpp"

using namespace std;
using glm::vec2;
using glm::vec3;
using glm::vec4;

struct Mesh : public SceneNode{

	vector<vec3> position;
	vector<vec2> uv;
	vector<uint32_t> color;

	Mesh(){

	}
	string toString(){
		return "Mesh";
	}

	// // splits every triangle into three smaller triangles
	// void subdivide(){

	// }

	static shared_ptr<Mesh> createPlane(int divisions = 64){

		vector<vec3> positions;
		vector<vec2> uvs;
		vector<uint32_t> colors;

		auto sample = [&](float s, float t) -> vec3{
			return {
				s * 2.0f - 1.0f,
				t * 2.0f - 1.0f, 
				0.0f
			};
		};

		auto toUV = [&](int i, int j) -> vec2{
			return {
				float(i) / float(divisions),
				float(j) / float(divisions),
			};
		};

		for(int i = 0; i < divisions; i++)
		for(int j = 0; j < divisions; j++)
		{

			vec2 uv_00 = toUV(i + 0, j + 0);
			vec2 uv_10 = toUV(i + 1, j + 0);
			vec2 uv_11 = toUV(i + 1, j + 1);
			vec2 uv_01 = toUV(i + 0, j + 1);

			vec3 p_00 = sample(uv_00.s, uv_00.t);
			vec3 p_10 = sample(uv_10.s, uv_10.t);
			vec3 p_11 = sample(uv_11.s, uv_11.t);
			vec3 p_01 = sample(uv_01.s, uv_01.t);

			positions.push_back(p_00);
			positions.push_back(p_10);
			positions.push_back(p_11);
			positions.push_back(p_00);
			positions.push_back(p_11);
			positions.push_back(p_01);

			uvs.push_back(uv_00);
			uvs.push_back(uv_10);
			uvs.push_back(uv_11);
			uvs.push_back(uv_00);
			uvs.push_back(uv_11);
			uvs.push_back(uv_01);

			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);


		}

		auto mesh = make_shared<Mesh>();
		mesh->position = positions;
		mesh->uv = uvs;
		mesh->color = colors;

		return mesh;
	}

	static shared_ptr<Mesh> createSphere(int segments = 36){

		auto sample = [&](float s, float t){

			float u = s * 2.0f * 3.1415f;
			float v = t * 2.0f * 3.1415f;
			vec3 circle = {cos(u), 0.0f, sin(u)};

			mat4 rot = glm::rotate(v, vec3{0.0f, 0.0f, 1.0f});

			vec3 pos = vec3(rot * vec4(circle, 1.0f));

			return pos;
		};

		auto toUV = [&](int i, int j){
			vec2 uv = {
				float(i) / float(segments),
				float(j) / float(segments),
			};

			return uv;
		};

		vector<vec3> positions;
		vector<vec2> uvs;
		vector<uint32_t> colors;

		for(int i = 0; i < segments; i++)
		for(int j = 0; j < segments; j++)
		{

			vec2 uv_00 = toUV(i + 0, j + 0);
			vec2 uv_10 = toUV(i + 1, j + 0);
			vec2 uv_11 = toUV(i + 1, j + 1);
			vec2 uv_01 = toUV(i + 0, j + 1);

			vec3 p_00 = sample(uv_00.s, uv_00.t);
			vec3 p_10 = sample(uv_10.s, uv_10.t);
			vec3 p_11 = sample(uv_11.s, uv_11.t);
			vec3 p_01 = sample(uv_01.s, uv_01.t);

			positions.push_back(p_00);
			positions.push_back(p_10);
			positions.push_back(p_11);
			positions.push_back(p_00);
			positions.push_back(p_11);
			positions.push_back(p_01);

			uvs.push_back(uv_00);
			uvs.push_back(uv_10);
			uvs.push_back(uv_11);
			uvs.push_back(uv_00);
			uvs.push_back(uv_11);
			uvs.push_back(uv_01);

			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
		}

		auto mesh = make_shared<Mesh>();
		mesh->position = positions;
		mesh->uv = uvs;
		mesh->color = colors;

		return mesh;
	}

	static shared_ptr<Mesh> createTorus(float radius, float innerRadius, int segments = 36){

		auto sample = [&](float s, float t){

			float outerAngle = s * 2.0f * 3.1415f;
			float innerAngle = t * 2.0f * 3.1415f;
			vec3 inner = {cos(innerAngle), 0.0f, sin(innerAngle)};
			inner = inner * innerRadius + vec3{radius, 0.0f, 0.0f};

			mat4 rot = glm::rotate(outerAngle, vec3{0.0f, 0.0f, 1.0f});

			vec3 pos = vec3(rot * vec4(inner, 1.0f));

			return pos;
		};

		auto toUV = [&](int i, int j){
			vec2 uv = {
				float(i) / float(segments),
				float(j) / float(segments),
			};

			return uv;
		};

		vector<vec3> positions;
		vector<vec2> uvs;
		vector<uint32_t> colors;

		for(int i = 0; i < segments; i++)
		for(int j = 0; j < segments; j++)
		{

			vec2 uv_00 = toUV(i + 0, j + 0);
			vec2 uv_10 = toUV(i + 1, j + 0);
			vec2 uv_11 = toUV(i + 1, j + 1);
			vec2 uv_01 = toUV(i + 0, j + 1);

			vec3 p_00 = sample(uv_00.s, uv_00.t);
			vec3 p_10 = sample(uv_10.s, uv_10.t);
			vec3 p_11 = sample(uv_11.s, uv_11.t);
			vec3 p_01 = sample(uv_01.s, uv_01.t);

			positions.push_back(p_00);
			positions.push_back(p_10);
			positions.push_back(p_11);
			positions.push_back(p_00);
			positions.push_back(p_11);
			positions.push_back(p_01);

			uvs.push_back(uv_00);
			uvs.push_back(uv_10);
			uvs.push_back(uv_11);
			uvs.push_back(uv_00);
			uvs.push_back(uv_11);
			uvs.push_back(uv_01);

			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);
			colors.push_back(0xff00ff00);


		}

		auto mesh = make_shared<Mesh>();
		mesh->position = positions;
		mesh->uv = uvs;
		mesh->color = colors;

		return mesh;
	}

	static shared_ptr<Mesh> createBox(){

		vector<vec3> position = {
			// BOTTOM
			{-1.0f, -1.0f, -1.0f},
			{ 1.0f, -1.0f, -1.0f},
			{ 1.0f,  1.0f, -1.0f},
			{-1.0f, -1.0f, -1.0f},
			{ 1.0f,  1.0f, -1.0f},
			{-1.0f,  1.0f, -1.0f},

			// TOP
			{-1.0f, -1.0f,  1.0f},
			{ 1.0f, -1.0f,  1.0f},
			{ 1.0f,  1.0f,  1.0f},
			{-1.0f, -1.0f,  1.0f},
			{ 1.0f,  1.0f,  1.0f},
			{-1.0f,  1.0f,  1.0f},

			// LEFT
			{-1.0f, -1.0f, -1.0f},
			{-1.0f,  1.0f, -1.0f},
			{-1.0f,  1.0f,  1.0f},
			{-1.0f, -1.0f, -1.0f},
			{-1.0f,  1.0f,  1.0f},
			{-1.0f, -1.0f,  1.0f},

			// RIGHT
			{ 1.0f, -1.0f, -1.0f},
			{ 1.0f,  1.0f, -1.0f},
			{ 1.0f,  1.0f,  1.0f},
			{ 1.0f, -1.0f, -1.0f},
			{ 1.0f,  1.0f,  1.0f},
			{ 1.0f, -1.0f,  1.0f},

			// FRONT
			{-1.0f, -1.0f, -1.0f},
			{ 1.0f, -1.0f, -1.0f},
			{ 1.0f, -1.0f,  1.0f},
			{-1.0f, -1.0f, -1.0f},
			{ 1.0f, -1.0f,  1.0f},
			{-1.0f, -1.0f,  1.0f},

			// BACK
			{-1.0f,  1.0f, -1.0f},
			{ 1.0f,  1.0f, -1.0f},
			{ 1.0f,  1.0f,  1.0f},
			{-1.0f,  1.0f, -1.0f},
			{ 1.0f,  1.0f,  1.0f},
			{-1.0f,  1.0f,  1.0f},

		};

		vector<vec2> uv = {
			// BOTTOM
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f},

			// TOP
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f},

			// LEFT
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f},

			// RIGHT
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f},

			// FRONT
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f},

			// BACK
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f},
		};

		vector<uint32_t> color = {
			// BOTTOM
			0xffff0000,
			0xffff0000,
			0xffff0000,
			0xffff0000,
			0xffff0000,
			0xffff0000,

			// TOP
			0xffaa0000,
			0xffaa0000,
			0xffaa0000,
			0xffaa0000,
			0xffaa0000,
			0xffaa0000,

			// LEFT
			0xff0000ff,
			0xff0000ff,
			0xff0000ff,
			0xff0000ff,
			0xff0000ff,
			0xff0000ff,

			// RIGHT
			0xff0000aa,
			0xff0000aa,
			0xff0000aa,
			0xff0000aa,
			0xff0000aa,
			0xff0000aa,

			// FRONT
			0xff00ff00,
			0xff00ff00,
			0xff00ff00,
			0xff00ff00,
			0xff00ff00,
			0xff00ff00,

			// BACK
			0xff00aa00,
			0xff00aa00,
			0xff00aa00,
			0xff00aa00,
			0xff00aa00,
			0xff00aa00,
		};

		auto mesh = make_shared<Mesh>();
		mesh->position = position;
		mesh->uv = uv;
		mesh->color = color;

		return mesh;
	}

	// static shared_ptr<Mesh> createSphereCloud(){

	// 	auto mesh = make_shared<Mesh>();

	// 	int lines_vertical = 5;
	// 	int lines_horizontal = 10;
	// 	constexpr double PI = 3.1415;
	// 	constexpr double v_angle_start = -PI / 2.0;
	// 	constexpr double v_angle_end   = PI / 2.0;
	// 	constexpr double h_angle_start = 0;
	// 	constexpr double h_angle_end   = 2.0 * PI;

	// 	auto sample = [](double v, double h){
			
	// 		double x = sin(h) * cos(v);
	// 		double y =          cos(h);
	// 		double z = sin(h) * sin(v);

	// 		glm::vec3 xyz(x, y, z);

	// 		return xyz;
	// 	};

	// 	for(int v = 0; v < lines_vertical; v++)
	// 	for(float h = 0.0; h < 2.0 * PI; h += 0.01)
	// 	{
	// 		double v_norm = double(v) / double(lines_vertical);
	// 		double v_rad = (1.0 - v_norm) * v_angle_start + v_norm * v_angle_end;

	// 		auto p00 = sample(v_rad, h);

	// 		mesh->position.push_back(p00.x);
	// 		mesh->position.push_back(p00.y);
	// 		mesh->position.push_back(p00.z);
	// 	}

	// 	for(int h = 0; h < lines_horizontal; h++)
	// 	for(float v = -PI / 2.0; v < PI / 2.0; v += 0.01)
	// 	{
	// 		double h_norm = double(h) / double(lines_horizontal);
	// 		double h_rad = (1.0 - h_norm) * h_angle_start + h_norm * h_angle_end;

	// 		auto p00 = sample(v, h_rad);

	// 		mesh->position.push_back(p00.x);
	// 		mesh->position.push_back(p00.y);
	// 		mesh->position.push_back(p00.z);
	// 	}

	// 	return mesh;
	// };

	// static shared_ptr<Mesh> createLineCloud(vec3 start, vec3 end){

	// 	auto mesh = make_shared<Mesh>();

	// 	int steps = 1000;

	// 	for(int i = 0; i < steps; i++){
	// 		float u = float(i) / (float(steps) - 1.0);

	// 		float x = (1.0 - u) * start.x + u * end.x;
	// 		float y = (1.0 - u) * start.y + u * end.y;
	// 		float z = (1.0 - u) * start.z + u * end.z;

	// 		mesh->position.push_back(x);
	// 		mesh->position.push_back(y);
	// 		mesh->position.push_back(z);
	// 	}

	// 	glCreateBuffers(1, &mesh->ssbo_position);
	// 	glNamedBufferData(mesh->ssbo_position, sizeof(float) * mesh->position.size(), mesh->position.data(), GL_DYNAMIC_DRAW);

	// 	return mesh;
	// };

	// static shared_ptr<Mesh> createLineCirce(){

	// 	auto mesh = make_shared<Mesh>();

	// 	int steps = 2000;

	// 	for(int i = 0; i < steps; i++){
	// 		float u = float(i) / (float(steps) - 1.0);
	// 		u = 2.0 * 3.1415 * u;

	// 		float x = sin(u);
	// 		float y = cos(u);
	// 		float z = 0.0f;

	// 		mesh->position.push_back(x);
	// 		mesh->position.push_back(y);
	// 		mesh->position.push_back(z);
	// 	}

	// 	glCreateBuffers(1, &mesh->ssbo_position);
	// 	glNamedBufferData(mesh->ssbo_position, sizeof(float) * mesh->position.size(), mesh->position.data(), GL_DYNAMIC_DRAW);

	// 	return mesh;
	// };

	// static shared_ptr<Mesh> createBox(){

	// 	auto mesh = make_shared<Mesh>();

	// 	mesh->position = vector<float>{
	// 		// BOTTOM
	// 		-1.0, -1.0, -1.0,
	// 		 1.0, -1.0, -1.0,
	// 		 1.0,  1.0, -1.0,
	// 		-1.0, -1.0, -1.0, 
	// 		 1.0,  1.0, -1.0,
	// 		-1.0,  1.0, -1.0,

	// 		// TOP
	// 		-1.0, -1.0,  1.0,
	// 		 1.0, -1.0,  1.0,
	// 		 1.0,  1.0,  1.0,
	// 		-1.0, -1.0,  1.0, 
	// 		 1.0,  1.0,  1.0,
	// 		-1.0,  1.0,  1.0,

	// 		// LEFT
	// 		-1.0, -1.0, -1.0, 
	// 		-1.0,  1.0, -1.0, 
	// 		-1.0,  1.0,  1.0, 
	// 		-1.0, -1.0, -1.0, 
	// 		-1.0,  1.0,  1.0, 
	// 		-1.0, -1.0,  1.0, 

	// 		// RIGHT
	// 		 1.0, -1.0, -1.0, 
	// 		 1.0,  1.0, -1.0, 
	// 		 1.0,  1.0,  1.0, 
	// 		 1.0, -1.0, -1.0, 
	// 		 1.0,  1.0,  1.0, 
	// 		 1.0, -1.0,  1.0, 

	// 		 // FRONT
	// 		-1.0, -1.0, -1.0, 
	// 		 1.0, -1.0, -1.0, 
	// 		 1.0, -1.0,  1.0, 
	// 		-1.0, -1.0, -1.0, 
	// 		 1.0, -1.0,  1.0, 
	// 		-1.0, -1.0,  1.0, 

	// 		 // BACK
	// 		-1.0,  1.0, -1.0, 
	// 		 1.0,  1.0, -1.0, 
	// 		 1.0,  1.0,  1.0, 
	// 		-1.0,  1.0, -1.0, 
	// 		 1.0,  1.0,  1.0, 
	// 		-1.0,  1.0,  1.0, 
	// 	};

	// 	glCreateBuffers(1, &mesh->ssbo_position);
	// 	glNamedBufferData(mesh->ssbo_position, sizeof(float) * mesh->position.size(), mesh->position.data(), GL_DYNAMIC_DRAW);

	// 	return mesh;
	// }

};