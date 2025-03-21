#pragma once

#include "actions/VR_TransformSelectionAction.h"

static ImguiNode* intersectedGuiNode = nullptr;
static ImguiNode* grabbedGuiNode = nullptr;
static inline shared_ptr<VR_TransformSelectionAction> transformSelectionAction = nullptr;

// Handle intersections of the controller with VR GUI panels.
// - Check intersection
// - Each backed by an imgui context. 
// - Injects controller state as imgui mouse input
//     - Intersection with gui panel = mouse pos
//     - Right controller trigger maps to left mouse button
void handleControllerImguiNodeIntersection(Controller controller){

	auto editor = SplatEditor::instance;
	auto ovr = editor->ovr;

	intersectedGuiNode = nullptr;
	editor->state_vr.menu_intersects = false;

	// Don't do menu stuff when we're currently draging a menu.
	if(grabbedGuiNode != nullptr) return;

	editor->scene.process<ImguiNode>([&](ImguiNode* guinode){
		mat4 mRot = glm::rotate(-140.0f * 3.1415f / 180.0f, vec3{ 1.0f, 0.0f, 0.0f });
		mat4 transform = mat4(ovr->flip * controller.transform) * mRot;

		vec3 origin = vec3(transform * vec4{0.0f, 0.0f, 0.0f, 1.0f});
		vec3 p1 = vec3(transform * vec4{0.0f, 1.0f, 0.0f, 1.0f});
		vec3 dir = normalize(p1 - origin);

		vec3 start = origin;
		vec3 end = origin + 2.0f * dir;
		
		vec3 plane_pos = vec3(guinode->transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 plane_N = vec3(guinode->transform * vec4(0.0f, 0.0f, 1.0f, 0.0f));
		float plane_D = -dot(plane_pos, plane_N);

		float d = ray_plane_intersection(origin, dir, plane_N, plane_D);
		vec3 I = origin + d * dir;

		bool intersectsMenuPlane = d > 0 && d != Infinity;

		if(intersectsMenuPlane){
			float sx = float(guinode->page->width) / 1000.0f;
			float sy = float(guinode->page->height) / 1000.0f;

			vec3 I_local = I - plane_pos;
			vec3 right = normalize(vec3(guinode->transform * vec4(1.0f, 0.0f, 0.0f, 1.0f)) - plane_pos);
			vec3 up = normalize(vec3(guinode->transform * vec4(0.0f, 1.0f, 0.0f, 1.0f)) - plane_pos);

			float factor = 0.2f;

			float u = (dot(right, I_local) / factor) / sx;
			float v = (dot(up, I_local) / factor) / sy;

			bool intersectsMenuQuad = abs(u) < 1.0f && abs(v) < 1.0f;
			if(intersectsMenuQuad){

				if (controller.isBPressed() && !controller.wasBPressed()) {
					grabbedGuiNode = guinode;
					return;
				}

				editor->state_vr.menu_intersects = true;
				editor->state_vr.menu_intersection = I;
				mat4 t_sphere = glm::translate(I) * glm::scale(vec3{1.0f, 1.0f, 1.0f} * 0.005f);
				guinode->page->context->IO.MousePos = ImVec2(
					(u * 0.5f + 0.5f) * float(guinode->page->width), 
					float(guinode->page->height) - (v * 0.5f + 0.5f) * float(guinode->page->height)
				);

				if(controller.isTriggerPressed() && controller.valid){
					guinode->page->context->IO.MouseDown[0] = true;
				} else {
					guinode->page->context->IO.MouseDown[0] = false;
				}
				guinode->page->context->IO.MouseWheel = controller.state.rAxis[0].y;
				
				intersectedGuiNode = guinode;
			}

			end = I;
		}
	});
}



void SplatEditor::inputHandlingVR(){

	auto editor = SplatEditor::instance;
	Scene& scene = editor->scene;
	auto ovr = editor->ovr;
	auto& state = editor->state;

	#if !defined(NO_OPENVR)

	if(!ovr->isActive()){
		editor->sn_brushsphere->visible = false;
		return;
	}

	//if(transformSelectionAction == nullptr){
	//	transformSelectionAction = make_shared<VR_TransformSelectionAction>();
	//}

	ImGuiIO& io = ImGui::GetIO();

	// "reset" imgui inputs. Then map VR controller input to mouse input latter on
	io.MouseDown[0] = false;
	io.MousePos = ImVec2(0.0f, 0.0f);
	io.MouseWheel = 0.0f;

	Controller left = ovr->getLeftController();
	Controller right = ovr->getRightController();

	// TRANSFORM IMGUI NODE
	if (grabbedGuiNode && right.isBPressed()) {
		transformSelectionAction = nullptr;

		float sx = float(grabbedGuiNode->page->width) / 1000.0f;
		float sy = float(grabbedGuiNode->page->height) / 1000.0f;

		mat4 mRot = glm::rotate(-3.1415f * 0.5f, vec3{ 1.0f, 0.0f, 0.0f });
		mat4 mScale = glm::scale(vec3{sx, sy, 1.0f} * 0.2f);

		grabbedGuiNode->transform = mat4(ovr->flip * right.transform) * mRot * mScale;
	} else if (grabbedGuiNode && !right.isBPressed()) {
		transformSelectionAction = nullptr;

		grabbedGuiNode = nullptr;
	}else if(right.isBPressed()){

		if(transformSelectionAction == nullptr){
			transformSelectionAction = make_shared<VR_TransformSelectionAction>();
		}

		transformSelectionAction->update();

	}else{
		transformSelectionAction = nullptr;
	}

	// CONTROLLER-MENU INTERSECTION
	handleControllerImguiNodeIntersection(right);

	// Update current action, or cancel with left trigger
	// Actions typically utilize the right trigger buttons.
	if(state.currentAction){
		if(left.triggered()){
			editor->setAction(nullptr);
		}else{
			state.currentAction->update();
		}
	}

	{ // UNDO HANDLING with left joystick
		float value = left.state.rAxis[0].x;

		constexpr int STATE_NEUTRAL = 0;
		constexpr int STATE_LEFT = 1;
		constexpr int STATE_RIGHT = 2;
		constexpr float THRESHOLD_LEFT_RIGHT = 0.7f; // EXCEEDING +- THRESHOLD will set state to left or right
		constexpr float THRESHOLD_NEUTRAL = 0.1f; // if value is under this threshold, state will be set to neutral
		static int state = STATE_NEUTRAL;

		int newState = STATE_NEUTRAL;
		if(value < -THRESHOLD_LEFT_RIGHT) newState = STATE_LEFT;
		if(value > +THRESHOLD_LEFT_RIGHT) newState = STATE_RIGHT;

		if(state == STATE_NEUTRAL && newState == STATE_LEFT){
			editor->undo();
		}else if(state == STATE_NEUTRAL && newState == STATE_RIGHT){
			editor->redo();
		}
		
		state = newState;
	}

	// Transform "world" with A button, which unfortunately is equal to grip button in OpenVR's legacy input handling.
	if(left.isAPressed() && right.isAPressed()){
		// GRIP SCALE ROTATE TRANSLATE

		vec3 left_pos = vec3(ovr->flip * left.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 left_previousPos = vec3(ovr->flip * left.previous.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 left_delta = left_pos - left_previousPos;

		vec3 right_pos = vec3(ovr->flip * right.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 right_previousPos = vec3(ovr->flip * right.previous.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 right_delta = right_pos - right_previousPos;

		float amount = length(left_pos - right_pos);
		float amountPrevious = length(left_previousPos - right_previousPos);
		float scale = amount / amountPrevious;

		vec3 center = (left_pos + right_pos) * 0.5f;
		vec3 center_prev = (left_previousPos + right_previousPos) * 0.5f;

		vec2 dirxy = normalize(vec2((right_pos - left_pos)));
		vec2 dirxy_prev = normalize(vec2((right_previousPos - left_previousPos)));
		

		float angle_prev = atan2(dirxy_prev.y, dirxy_prev.x);
		float angle = atan2(dirxy.y, dirxy.x);
		float angle_diff = angle - angle_prev;

		mat4 rot = rotate(angle_diff, vec3{0.0f, 0.0f, 1.0f});
		mat4 trans = translate(center - center_prev);
		
		mat4 transform = trans * translate(center) * rot * glm::scale(vec3{ scale, scale, scale }) * translate(-center);

		scene.world->transform = transform * scene.world->transform;

	}else if(right.isAPressed()){
		// GRIP TRANSLATE

		vec3 right_pos = vec3(ovr->flip * right.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 right_previousPos = vec3(ovr->flip * right.previous.transform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		vec3 right_delta = right_pos - right_previousPos;

		scene.world->transform = translate(right_delta) * scene.world->transform;
	}

	// FLY-THROUGH USING RIGHT JOYSTICK
	if(!editor->state_vr.menu_intersects){ 

		mat4 mRot = glm::rotate(-140.0f * 3.1415f / 180.0f, vec3{ 1.0f, 0.0f, 0.0f });
		mat4 transform_controller = mat4(ovr->flip * right.transform) * mRot;

		vec3 origin = vec3(transform_controller * vec4{0.0f, 0.0f, 0.0f, 1.0f});
		vec3 p_forward = vec3(transform_controller * vec4{0.0f, 1.0f, 0.0f, 1.0f});
		vec3 p_right = vec3(transform_controller * vec4{1.0f, 0.0f, 0.0f, 1.0f});
		vec3 forward = -normalize(p_forward - origin);
		vec3 vright = -normalize(p_right - origin);

		vec3 delta = forward * float(GLRenderer::timeSinceLastFrame) * right.state.rAxis[0].y;
		delta = delta + vright * float(GLRenderer::timeSinceLastFrame) * right.state.rAxis[0].x;

		mat4 transform = translate(delta);

		scene.world->transform = transform * scene.world->transform;
	}

	#endif
}