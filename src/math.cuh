#pragma once

using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::quat;
using glm::dot;
using glm::transpose;
using glm::inverse;

struct Plane{
	vec3 normal;
	float constant;
};

float distanceToPoint(vec3 point, Plane plane){
	return dot(plane.normal, point) + plane.constant;
}

float distanceToPlane(vec3 origin, vec3 direction, Plane plane){

	float denominator = dot(plane.normal, direction);

	if(denominator < 0.0f){
		return Infinity;
	}

	if(denominator == 0.0f){

		// line is coplanar, return origin
		if(distanceToPoint(origin, plane) == 0.0f){
			return 0.0f;
		}

		// Null is preferable to undefined since undefined means.... it is undefined
		return Infinity;
	}

	float t = -(dot(origin, plane.normal) + plane.constant) / denominator;

	if(t >= 0.0){
		return t;
	}else{
		return Infinity;
	}
}

Plane createPlane(float x, float y, float z, float w){

	float nLength = length(vec3{x, y, z});

	Plane plane;
	plane.normal = vec3{x, y, z} / nLength;
	plane.constant = w / nLength;

	return plane;
}

struct Frustum{
	Plane planes[6];

	static Frustum fromWorldViewProj(mat4 worldViewProj){
		float* values = reinterpret_cast<float*>(&worldViewProj);

		float m_0  = values[ 0];
		float m_1  = values[ 1];
		float m_2  = values[ 2];
		float m_3  = values[ 3];
		float m_4  = values[ 4];
		float m_5  = values[ 5];
		float m_6  = values[ 6];
		float m_7  = values[ 7];
		float m_8  = values[ 8];
		float m_9  = values[ 9];
		float m_10 = values[10];
		float m_11 = values[11];
		float m_12 = values[12];
		float m_13 = values[13];
		float m_14 = values[14];
		float m_15 = values[15];

		Plane planes[6] = {
			createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12),
			createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12),
			createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13),
			createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13),
			createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14),
			createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14),
		};

		Frustum frustum;

		frustum.planes[0] = createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12);
		frustum.planes[1] = createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12);
		frustum.planes[2] = createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13);
		frustum.planes[3] = createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13);
		frustum.planes[4] = createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14);
		frustum.planes[5] = createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14);
		
		return frustum;
	}

	vec3 intersectRay(vec3 origin, vec3 direction){

		float closest = Infinity;
		float farthest = -Infinity;

		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			float d = distanceToPlane(origin, direction, plane);

			if(d > 0){
				closest = min(closest, d);
			}
			if(d > 0 && d != Infinity){
				farthest = max(farthest, d);
			}
		}

		if(farthest == -Infinity) return {farthest, farthest, farthest};

		vec3 intersection = {
			origin.x + direction.x * farthest,
			origin.y + direction.y * farthest,
			origin.z + direction.z * farthest
		};

		return intersection;
	}

	float intersectRayDistance(vec3 origin, vec3 direction){

		float closest = Infinity;
		float farthest = -Infinity;

		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			float d = distanceToPlane(origin, direction, plane);

			if(d > 0){
				closest = min(closest, d);
			}
			if(d > 0 && d != Infinity){
				farthest = max(farthest, d);
			}
		}

		return farthest == -Infinity ? Infinity : farthest;
	}

	bool contains(vec3 point){
		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			float d = distanceToPoint(point, plane);

			if(d < 0){
				return false;
			}
		}

		return true;
	}
};

bool intersectsFrustum(mat4 worldViewProj, Box3 box){

	float* values = reinterpret_cast<float*>(&worldViewProj);

	float m_0  = values[ 0];
	float m_1  = values[ 1];
	float m_2  = values[ 2];
	float m_3  = values[ 3];
	float m_4  = values[ 4];
	float m_5  = values[ 5];
	float m_6  = values[ 6];
	float m_7  = values[ 7];
	float m_8  = values[ 8];
	float m_9  = values[ 9];
	float m_10 = values[10];
	float m_11 = values[11];
	float m_12 = values[12];
	float m_13 = values[13];
	float m_14 = values[14];
	float m_15 = values[15];

	Plane planes[6] = {
		createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12),
		createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12),
		createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13),
		createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13),
		createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14),
		createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14),
	};
	
	for(int i = 0; i < 6; i++){

		Plane plane = planes[i];

		vec3 vector;
		vector.x = plane.normal.x > 0.0 ? box.max.x : box.min.x;
		vector.y = plane.normal.y > 0.0 ? box.max.y : box.min.y;
		vector.z = plane.normal.z > 0.0 ? box.max.z : box.min.z;

		float d = distanceToPoint(vector, plane);

		if(d < 0){
			return false;
		}
	}

	return true;
}

bool isPointInBox(Box3 box, vec3 point){

	if(point.x < box.min.x) return false;
	if(point.y < box.min.y) return false;
	if(point.z < box.min.z) return false;
	if(point.x > box.max.x) return false;
	if(point.y > box.max.y) return false;
	if(point.z > box.max.z) return false;

	return true;
}


// see https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
inline static float ray_plane_intersection(vec3 origin, vec3 direction, vec3 N, float d) {
	float t = -(dot(origin, N) + d) / dot(direction, N);

	return t;
}

// from https://github.com/mrdoob/three.js/blob/4f067f0f4dc9e81f7fb3484962f2e973f71fab60/src/math/Ray.js#L220
// LICENSE: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
inline static float ray_sphere_intersection(vec3 origin, vec3 direction, vec3 spherePos, float radius){
	vec3 OtoS = spherePos - origin;

	float tca = dot(OtoS, direction);
	float d2 = dot(OtoS, OtoS) - tca * tca;
	float radius2 = radius * radius;

	if(d2 > radius2) return Infinity;

	float thc = sqrt(radius2 - d2);

	float t0 = tca - thc;
	float t1 = tca + thc;

	if(t1 < 0) return Infinity;
	if(t0 < 0) return Infinity;

	return t0;
}

inline bool intersection_point_splat(vec2 point, vec2 splatPos, vec2 basisVector1, vec2 basisVector2){

	vec2 pFrag = point - splatPos;
	float sT = dot(normalize(basisVector1), pFrag) / length(basisVector1);
	float sB = dot(normalize(basisVector2), pFrag) / length(basisVector2);

	float w = sqrt(sT * sT + sB * sB);

	return w < 0.999f;
}


// inline bool intersection_point_splat(
// 	vec2 point, vec2 splatPos, 
// 	vec2 a, vec2 b
// ){
// 	vec2 pFrag = point - splatPos;

// 	float sA = dot(a, pFrag) / dot(a, a);
// 	float sB = dot(b, pFrag) / dot(b, b);

// 	float w = sqrt(sA * sA + sB * sB);

// 	return w < 1.0f;
// }

inline bool intersection_circle_splat(
	vec2 circlePos, float radius,
	vec2 splatPos, 
	vec2 a, vec2 b
){
	vec2 pFrag = circlePos - splatPos;

	vec2 a_large = a * (length(a) + radius) / length(a);
	vec2 b_large = b * (length(b) + radius) / length(b);

	float sA = dot(a_large, pFrag) / dot(a_large, a_large);
	float sB = dot(b_large, pFrag) / dot(b_large, b_large);

	float w = sqrt(sA * sA + sB * sB);

	return w < 1.0f;
}

inline uint32_t to32BitColor(uint64_t c64){
	uint16_t* rgba64 = (uint16_t*)&c64;

	uint32_t c32;
	uint8_t* rgba32 = (uint8_t*)&c32;
	rgba32[0] = rgba64[0] / 256;
	rgba32[1] = rgba64[1] / 256;
	rgba32[2] = rgba64[2] / 256;
	rgba32[3] = rgba64[3] / 256;

	return c32;
}

inline uint32_t to32BitColor(vec4 color){

	uint32_t c32;
	uint8_t* rgba32 = (uint8_t*)&c32;
	rgba32[0] = color.r * 255.0f;
	rgba32[1] = color.g * 255.0f;
	rgba32[2] = color.b * 255.0f;
	rgba32[3] = color.a * 255.0f;
	// rgba32[3] = 255;

	return c32;
}