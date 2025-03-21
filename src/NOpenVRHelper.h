#pragma once

// A No-Op interface of the OpenVRHelper. Used when OpenVR can not be included in the build.

#include <string>
#include <vector>
#include <memory>

using namespace std;

#include "openvr.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"

using std::string;
using std::vector;
using glm::dmat4;
using std::shared_ptr;

using namespace vr;

struct Pose {
	bool valid;
	dmat4 transform;
};

struct Controller{
	bool valid;
	dmat4 transform;
	VRControllerState_t state;

	struct {
		bool valid;
		dmat4 transform;
		VRControllerState_t state;
	} previous;

	bool isTriggerPressed(){
		return state.ulButtonPressed == (1llu << k_EButton_SteamVR_Trigger) != 0;;
	}

	bool wasTriggerPressed(){
		return previous.state.ulButtonPressed == (1llu << k_EButton_SteamVR_Trigger) != 0;
	}

	bool triggered(){
		return isTriggerPressed() && !wasTriggerPressed();
	}

	bool untriggered(){
		return !isTriggerPressed() && wasTriggerPressed();
	}

	bool isBPressed(){
		return state.ulButtonPressed & 2;
	}

	bool wasBPressed(){
		return previous.state.ulButtonPressed & 2;
	}

	bool isAPressed(){
		return state.ulButtonPressed & 4;
	}
	
	bool wasAPressed(){
		return previous.state.ulButtonPressed & 4;
	}
};

// struct UV {
// 	float u;
// 	float v;
// };

// struct DistortionMap {
// 	vector<UV> red;
// 	vector<UV> green;
// 	vector<UV> blue;
// };

struct Keyframe{};
struct OpenVrRecording{};

class OpenVRHelper {

public:

	shared_ptr<OpenVrRecording> currentRecording = nullptr;
	shared_ptr<OpenVrRecording> currentPlayback = nullptr;
	Keyframe frame;
	double t_start_playing = 0.0;
	IVRSystem *system = nullptr;
	string driver = "No Driver";
	string display = "No Display";
	uint64_t buttonMap[k_unMaxTrackedDeviceCount];
	TrackedDevicePose_t trackedDevicePose[k_unMaxTrackedDeviceCount];
	vector<dmat4> previousDevicePose = vector<dmat4>(k_unMaxTrackedDeviceCount);
	vector<dmat4> devicePose = vector<dmat4>(k_unMaxTrackedDeviceCount);
	vector<vr::VRControllerState_t> controllerStates = vector<vr::VRControllerState_t>(k_unMaxTrackedDeviceCount);
	dmat4 hmdPose;

	const dmat4 flip = dmat4(
		1.0,  0.0, 0.0, 0.0,
		0.0,  0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 0.0,
		0.0,  0.0, 0.0, 1.0
	);

	// static OpenVRHelper *_instance;

	OpenVRHelper() {

	}

	static OpenVRHelper *instance() {
		static OpenVRHelper* _instance = nullptr;

		if(_instance == nullptr){
			_instance = new OpenVRHelper();
		}

		return _instance;
	}

	shared_ptr<OpenVrRecording> startRecording();
	void stopRecording(){}
	void playRecording(shared_ptr<OpenVrRecording> recording);
	void stopPlaying(){}
	bool start(){return false;}
	void stop(){}
	bool isActive(){return false;}
	void processEvents(){}
	void updatePose(bool enableSync = true){

	}
	void postPresentHandoff(){}
	Pose getPose(int deviceID){return Pose();}
	inline Pose getLeftControllerPose(){return Pose();}
	inline Pose getRightControllerPose(){return Pose();}
	inline Controller getLeftController(){return Controller();}
	inline Controller getRightController(){return Controller();}
	std::pair<RenderModel_t*, RenderModel_TextureMap_t*> getRenderModel(int deviceID){return {nullptr, nullptr};}
	inline vr::VRControllerState_t getLeftControllerState(){return {0};}
	inline vr::VRControllerState_t getRightControllerState(){return {0};}
	vector<unsigned int> getRecommmendedRenderTargetSize(){return {128, 128};}
	void submit(unsigned int left, unsigned int right){}
	void submit(unsigned int texture, EVREye eye, VRTextureBounds_t bounds){}
	void submitDistortionApplied(unsigned int left, unsigned int right){}
	void submitDistortionApplied(unsigned int texture, EVREye eye){}
	// DistortionMap computeDistortionMap(EVREye eye, int width, int height){}
	string getTrackedDeviceString(TrackedDeviceIndex_t device, TrackedDeviceProperty prop, TrackedPropertyError* peError = nullptr) { return ""; }
	dmat4 getProjection(EVREye eye, float nearClip, float farClip){return dmat4(1.0f);}
	dmat4 getProjectionVP(EVREye eye, float nearClip, float farClip, int width, int height){return dmat4(1.0f);}
	float getFOV() { return 0.0f; }
	dmat4 getEyePose(Hmd_Eye nEye){return dmat4(1.0f);}
	dmat4 steamToGLM(const HmdMatrix34_t &mSteam){return dmat4(1.0f);}
	dmat4 getHmdPose(){return dmat4(1.0f);}
};