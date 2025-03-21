
#pragma once

#include <iostream>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Runtime.h"

using glm::mat4;

// using namespace std;

struct ProxyControls{

	float wheel_x = 0;
	float wheel_y = 0;
	float pos_x = 0;
	float pos_y = 0;
	int button = 0;
	int action = 0;
	int mods = 0;

	bool buttonPressed = false;
	bool mouseMoved = false;
	bool mouseScrolled = false;

	bool isLeftDown = false;
	bool isMiddleDown = false;
	bool isRightDown = false;

	mat4 world = mat4(1.0f);


	ProxyControls(){
	
	}

	void onMouseButton(int button, int action, int mods){

		this->button = this->button | button;
		this->action = this->action | action;
		this->mods   = this->mods | mods;

		if(button == 0 && action == 1){
			isLeftDown = true;
		}else if(action == 0){
			isLeftDown = false;
		}

		if(button == 2 && action == 1){
			isMiddleDown = true;
		}else if(action == 0){
			isMiddleDown = false;
		}

		if(button == 1 && action == 1){
			isRightDown = true;
		}else if(action == 0){
			isRightDown = false;
		}

		buttonPressed = true;
	}

	void onMouseMove(double xpos, double ypos){

		this->pos_x = xpos;
		this->pos_y = ypos;

		mouseMoved = true;
	}

	void onMouseScroll(double xoffset, double yoffset){

		this->wheel_x += xoffset;
		this->wheel_y += yoffset;

		mouseScrolled = true;
	}

	void update(){

	}

	void clear(){
		wheel_x = 0;
		wheel_y = 0;
		// pos_x = 0;
		// pos_y = 0;
		button = 0;
		action = 0;
		mods = 0;

		buttonPressed = false;
		mouseMoved = false;
		mouseScrolled = false;
	}

};