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




// SH rotation code from: https://github.com/andrewwillmott/sh-lib/blob/8821cba4acc2273ab20417388df16bd0012f0760/SHLib.cpp#L1090
// LICENSE: https://github.com/andrewwillmott/sh-lib/blob/8821cba4acc2273ab20417388df16bd0012f0760/LICENSE.md (public domain)
constexpr float kSqrt02_01  = 1.4142135623730951;
constexpr float kSqrt01_02  = 0.7071067811865476;
constexpr float kSqrt03_02  = 1.224744871391589;
constexpr float kSqrt01_03  = 0.5773502691896257;
constexpr float kSqrt02_03  = 0.816496580927726;
constexpr float kSqrt04_03  = 1.1547005383792515;
constexpr float kSqrt01_04  = 0.5;
constexpr float kSqrt03_04  = 0.8660254037844386;
constexpr float kSqrt05_04  = 1.118033988749895;
constexpr float kSqrt01_05  = 0.4472135954999579;
constexpr float kSqrt02_05  = 0.6324555320336759;
constexpr float kSqrt03_05  = 0.7745966692414834;
constexpr float kSqrt04_05  = 0.8944271909999159;
constexpr float kSqrt06_05  = 1.0954451150103321;
constexpr float kSqrt08_05  = 1.2649110640673518;
constexpr float kSqrt09_05  = 1.3416407864998738;
constexpr float kSqrt01_06  = 0.408248290463863;
constexpr float kSqrt05_06  = 0.9128709291752769;
constexpr float kSqrt07_06  = 1.0801234497346435;
constexpr float kSqrt02_07  = 0.5345224838248488;
constexpr float kSqrt06_07  = 0.9258200997725514;
constexpr float kSqrt10_07  = 1.1952286093343936;
constexpr float kSqrt12_07  = 1.3093073414159542;
constexpr float kSqrt15_07  = 1.4638501094227998;
constexpr float kSqrt16_07  = 1.5118578920369088;
constexpr float kSqrt01_08  = 0.3535533905932738;
constexpr float kSqrt03_08  = 0.6123724356957945;
constexpr float kSqrt05_08  = 0.7905694150420949;
constexpr float kSqrt07_08  = 0.9354143466934853;
constexpr float kSqrt09_08  = 1.0606601717798212;
constexpr float kSqrt05_09  = 0.7453559924999299;
constexpr float kSqrt08_09  = 0.9428090415820634;
constexpr float kSqrt01_10  = 0.31622776601683794;
constexpr float kSqrt03_10  = 0.5477225575051661;
constexpr float kSqrt07_10  = 0.8366600265340756;
constexpr float kSqrt09_10  = 0.9486832980505138;
constexpr float kSqrt01_12  = 0.28867513459481287;
constexpr float kSqrt07_12  = 0.7637626158259734;
constexpr float kSqrt11_12  = 0.9574271077563381;
constexpr float kSqrt01_14  = 0.2672612419124244;
constexpr float kSqrt03_14  = 0.4629100498862757;
constexpr float kSqrt15_14  = 1.0350983390135313;
constexpr float kSqrt04_15  = 0.5163977794943222;
constexpr float kSqrt07_15  = 0.8366600265340756;
constexpr float kSqrt14_15  = 0.9660917830792959;
constexpr float kSqrt16_15  = 1.0327955589886444;
constexpr float kSqrt01_16  = 0.25;
constexpr float kSqrt03_16  = 0.4330127018922193;
constexpr float kSqrt07_16  = 0.6614378277661477;
constexpr float kSqrt15_16  = 0.9682458365518543;
constexpr float kSqrt01_18  = 0.23570226039551584;
constexpr float kSqrt01_24  = 0.2041241452319315;
constexpr float kSqrt03_25  = 0.34641016151377546;
constexpr float kSqrt09_25  = 0.6;
constexpr float kSqrt14_25  = 0.7483314773547883;
constexpr float kSqrt16_25  = 0.8;
constexpr float kSqrt18_25  = 0.848528137423857;
constexpr float kSqrt21_25  = 0.916515138991168;
constexpr float kSqrt24_25  = 0.9797958971132712;
constexpr float kSqrt03_28  = 0.32732683535398854;
constexpr float kSqrt05_28  = 0.4225771273642583;
constexpr float kSqrt01_30  = 0.18257418583505536;
constexpr float kSqrt01_32  = 0.1767766952966369;
constexpr float kSqrt03_32  = 0.30618621784789724;
constexpr float kSqrt15_32  = 0.6846531968814576;
constexpr float kSqrt21_32  = 0.8100925873009825;
constexpr float kSqrt11_36  = 0.5527707983925667;
constexpr float kSqrt35_36  = 0.9860132971832694;
constexpr float kSqrt01_50  = 0.1414213562373095;
constexpr float kSqrt03_50  = 0.2449489742783178;
constexpr float kSqrt21_50  = 0.648074069840786;
constexpr float kSqrt15_56  = 0.5175491695067657;
constexpr float kSqrt01_60  = 0.12909944487358055;
constexpr float kSqrt01_112 = 0.0944911182523068;
constexpr float kSqrt03_112 = 0.16366341767699427;
constexpr float kSqrt15_112 = 0.36596252735569995;

static int shDegreeToNumCoefficients(int degree){

	if(degree == 0) return 1;
	if(degree == 1) return 4;
	if(degree == 2) return 9;
	if(degree == 3) return 16;

	return 0;
}

static int numCoefficientsToShDegree(int numCoefficients){
	if(numCoefficients == 1) return 0;
	if(numCoefficients == 4) return 1;
	if(numCoefficients == 9) return 2;
	if(numCoefficients == 16) return 3;

	return 0;
}

// SH rotation code from: https://github.com/andrewwillmott/sh-lib/blob/8821cba4acc2273ab20417388df16bd0012f0760/SHLib.cpp#L1090
// LICENSE: https://github.com/andrewwillmott/sh-lib/blob/8821cba4acc2273ab20417388df16bd0012f0760/LICENSE.md (public domain)
void rotateSH(int degree, vec3* shs, vec3* shs_transformed, mat3 rotation){
	
	auto dp = [](int n, const vec3* a, const float* b){
		vec3 sum = {0.0f, 0.0f, 0.0f};

		for(int i = 0; i < n; i++){
			sum = sum + a[i] * b[i];
		}
		
		return sum;
	};

	// After struggling for a while with the correct rotation of sh-lib, 
	// Supersplat's source helped us figure out the correct re-ordering of the rotation, 
	// specifically the minus signs that were absent in sh-lib
	// see: https://github.com/playcanvas/supersplat/blob/6f01d82114faaef33fe9c9bf86a746c71d1f6dbf/src/sh-utils.ts#L53
	float sh1[3][3] = {
		{ rotation[1].y, -rotation[2].y,  rotation[0].y},
		{-rotation[1].z,  rotation[2].z, -rotation[0].z},
		{ rotation[1].x, -rotation[2].x,  rotation[0].x},
	};

	// vec3 shs_transformed[15];

	// DEGREE 1
	if(degree > 0){ 
		shs_transformed[0] = dp(3, shs + 0, sh1[0]);
		shs_transformed[1] = dp(3, shs + 0, sh1[1]);
		shs_transformed[2] = dp(3, shs + 0, sh1[2]);
	}

	// DEGREE 2
	float sh2[5][5];
	if(degree > 1){

		sh2[0][0] = kSqrt01_04 * ((sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2]) + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2]));
		sh2[0][1] =               (sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0]);
		sh2[0][2] = kSqrt03_04 *  (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1]);
		sh2[0][3] =               (sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2]);
		sh2[0][4] = kSqrt01_04 * ((sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0]) + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]));

		shs_transformed[3] = dp(5, shs + 3, sh2[0]);
		
		sh2[1][0] = kSqrt01_04 * ((sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2]) + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2]));
		sh2[1][1] =                sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0];
		sh2[1][2] = kSqrt03_04 *  (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1]);
		sh2[1][3] =                sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2];
		sh2[1][4] = kSqrt01_04 * ((sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0]) + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]));

		shs_transformed[4] = dp(5, shs + 3, sh2[1]);

		sh2[2][0] = kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) - kSqrt01_12 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) + (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
		sh2[2][1] = kSqrt04_03 *  sh1[1][1] * sh1[1][0] - kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0]);
		sh2[2][2] =               sh1[1][1] * sh1[1][1] - kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1]);
		sh2[2][3] = kSqrt04_03 *  sh1[1][1] * sh1[1][2] - kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2]);
		sh2[2][4] = kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) - kSqrt01_12 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) + (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

		shs_transformed[5] = dp(5, shs + 3, sh2[2]);

		sh2[3][0] = kSqrt01_04 * ((sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2]) + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2]));
		sh2[3][1] =                sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0];
		sh2[3][2] = kSqrt03_04 *  (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1]);
		sh2[3][3] =                sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2];
		sh2[3][4] = kSqrt01_04 * ((sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0]) + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]));

		shs_transformed[6] = dp(5, shs + 3, sh2[3]);

		sh2[4][0] = kSqrt01_04 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
		sh2[4][1] =               (sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0]);
		sh2[4][2] = kSqrt03_04 *  (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1]);
		sh2[4][3] =               (sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2]);
		sh2[4][4] = kSqrt01_04 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

		shs_transformed[7] = dp(5, shs + 3, sh2[4]);
	}

	// DEGREE 3
	if(degree > 2){
		float sh3[7][7];

		sh3[0][0] = kSqrt01_04 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
		sh3[0][1] = kSqrt03_02 *  (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0]);
		sh3[0][2] = kSqrt15_16 *  (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1]);
		sh3[0][3] = kSqrt05_06 *  (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2]);
		sh3[0][4] = kSqrt15_16 *  (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3]);
		sh3[0][5] = kSqrt03_02 *  (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4]);
		sh3[0][6] = kSqrt01_04 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

		shs_transformed[8] = dp(7, shs + 3 + 5, sh3[0]);

		sh3[1][0] = kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][0] + sh1[2][0] * sh2[1][4]) + (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4]));
		sh3[1][1] =               sh1[1][1] * sh2[0][0]                          +               (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0]);
		sh3[1][2] = kSqrt05_08 *  sh1[1][1] * sh2[0][1]                          + kSqrt05_08 *  (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1]);
		sh3[1][3] = kSqrt05_09 *  sh1[1][1] * sh2[0][2]                          + kSqrt05_09 *  (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2]);
		sh3[1][4] = kSqrt05_08 *  sh1[1][1] * sh2[0][3]                          + kSqrt05_08 *  (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3]);
		sh3[1][5] =               sh1[1][1] * sh2[0][4]                          +               (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4]);
		sh3[1][6] = kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][4] - sh1[2][0] * sh2[1][0]) + (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]));

		shs_transformed[9] = dp(7, shs + 3 + 5, sh3[1]);

		sh3[2][0] = kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) + kSqrt01_05 * (sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) - kSqrt01_60 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) - (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
		sh3[2][1] = kSqrt08_05 *  sh1[1][1] * sh2[1][0]                          + kSqrt06_05 *  sh1[0][1] * sh2[2][0] - kSqrt01_10 * (sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0]);
		sh3[2][2] =               sh1[1][1] * sh2[1][1]                          + kSqrt03_04 *  sh1[0][1] * sh2[2][1] - kSqrt01_16 * (sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1]);
		sh3[2][3] = kSqrt08_09 *  sh1[1][1] * sh2[1][2]                          + kSqrt02_03 *  sh1[0][1] * sh2[2][2] - kSqrt01_18 * (sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2]);
		sh3[2][4] =               sh1[1][1] * sh2[1][3]                          + kSqrt03_04 *  sh1[0][1] * sh2[2][3] - kSqrt01_16 * (sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3]);
		sh3[2][5] = kSqrt08_05 *  sh1[1][1] * sh2[1][4]                          + kSqrt06_05 *  sh1[0][1] * sh2[2][4] - kSqrt01_10 * (sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4]);
		sh3[2][6] = kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) + kSqrt01_05 * (sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) - kSqrt01_60 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) - (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

		shs_transformed[10] = dp(7, shs + 3 + 5, sh3[2]);

		sh3[3][0] = kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) - kSqrt01_10 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) + (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
		sh3[3][1] = kSqrt09_05 *  sh1[1][1] * sh2[2][0]                          - kSqrt03_05 *  (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0]);
		sh3[3][2] = kSqrt09_08 *  sh1[1][1] * sh2[2][1]                          - kSqrt03_08 *  (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1]);
		sh3[3][3] =               sh1[1][1] * sh2[2][2]                          - kSqrt01_03 *  (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2]);
		sh3[3][4] = kSqrt09_08 *  sh1[1][1] * sh2[2][3]                          - kSqrt03_08 *  (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3]);
		sh3[3][5] = kSqrt09_05 *  sh1[1][1] * sh2[2][4]                          - kSqrt03_05 *  (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4]);
		sh3[3][6] = kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) - kSqrt01_10 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) + (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

		shs_transformed[11] = dp(7, shs + 3 + 5, sh3[3]);

		sh3[4][0] = kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) + kSqrt01_05 * (sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) - kSqrt01_60 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) + (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
		sh3[4][1] = kSqrt08_05 *  sh1[1][1] * sh2[3][0]                          + kSqrt06_05 *  sh1[2][1] * sh2[2][0] - kSqrt01_10 * (sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0]);
		sh3[4][2] =               sh1[1][1] * sh2[3][1]                          + kSqrt03_04 *  sh1[2][1] * sh2[2][1] - kSqrt01_16 * (sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1]);
		sh3[4][3] = kSqrt08_09 *  sh1[1][1] * sh2[3][2]                          + kSqrt02_03 *  sh1[2][1] * sh2[2][2] - kSqrt01_18 * (sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2]);
		sh3[4][4] =               sh1[1][1] * sh2[3][3]                          + kSqrt03_04 *  sh1[2][1] * sh2[2][3] - kSqrt01_16 * (sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3]);
		sh3[4][5] = kSqrt08_05 *  sh1[1][1] * sh2[3][4]                          + kSqrt06_05 *  sh1[2][1] * sh2[2][4] - kSqrt01_10 * (sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4]);
		sh3[4][6] = kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) + kSqrt01_05 * (sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) - kSqrt01_60 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) + (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));

		shs_transformed[12] = dp(7, shs + 3 + 5, sh3[4]);

		sh3[5][0] = kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) - (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
		sh3[5][1] =               sh1[1][1] * sh2[4][0]                          +               (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0]);
		sh3[5][2] = kSqrt05_08 *  sh1[1][1] * sh2[4][1]                          + kSqrt05_08 *  (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1]);
		sh3[5][3] = kSqrt05_09 *  sh1[1][1] * sh2[4][2]                          + kSqrt05_09 *  (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2]);
		sh3[5][4] = kSqrt05_08 *  sh1[1][1] * sh2[4][3]                          + kSqrt05_08 *  (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3]);
		sh3[5][5] =               sh1[1][1] * sh2[4][4]                          +               (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4]);
		sh3[5][6] = kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) - (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

		shs_transformed[13] = dp(7, shs + 3 + 5, sh3[5]);

		sh3[6][0] = kSqrt01_04 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
		sh3[6][1] = kSqrt03_02 *  (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0]);
		sh3[6][2] = kSqrt15_16 *  (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1]);
		sh3[6][3] = kSqrt05_06 *  (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2]);
		sh3[6][4] = kSqrt15_16 *  (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3]);
		sh3[6][5] = kSqrt03_02 *  (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4]);
		sh3[6][6] = kSqrt01_04 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));

		shs_transformed[14] = dp(7, shs + 3 + 5, sh3[6]);
	}
}