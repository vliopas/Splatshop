
#pragma once


using glm::mat4;
using glm::vec4;
using glm::vec3;

struct DesktopVRControls{

	mat4 pose_left = mat4(1.0f);
	mat4 pose_right = mat4(1.0f);

	mat4 pose_left_previous = mat4(1.0f);
	mat4 pose_right_previous = mat4(1.0f);

	mat4 pose_left_diff = mat4(1.0f);
	mat4 pose_right_diff = mat4(1.0f);

	vec3 diff_left;
	vec3 diff_right;


	DesktopVRControls(){
	
	}


	void update(){
		
		{
			vec4 pos = pose_right * vec4(0.0f, 0.0f, 0.0f, 1.0f);
			vec4 posPrev = pose_right_previous * vec4(0.0f, 0.0f, 0.0f, 1.0f);

			diff_right = vec3(pos - posPrev);
		}

		{
			vec4 pos = pose_left * vec4(0.0f, 0.0f, 0.0f, 1.0f);
			vec4 posPrev = pose_left_previous * vec4(0.0f, 0.0f, 0.0f, 1.0f);

			diff_left = vec3(pos - posPrev);
		}


	}

};