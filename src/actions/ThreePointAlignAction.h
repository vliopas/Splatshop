#pragma once 

#include "InputAction.h"

struct ThreePointAlignUndoAction : public Action{

	shared_ptr<SceneNode> node = nullptr;
	mat4 before;
	mat4 after;

	void undo(){
		if(node) node->transform = before;
	}

	void redo(){
		if(node) node->transform = after;
	}

	int64_t byteSize(){
		return 0; // Not enough to matter
	}

};

struct ThreePointAlignAction : public InputAction{

	vector<vec3> points;

	void drawCircle_tpa(ImVec2 position, float radius, ImU32 color){
		auto drawlist = ImGui::GetForegroundDrawList();

		drawlist->AddCircle(position, radius, color, 32, 3.0f);
	}

	void start(){
		// auto editor = SplatEditor::instance;
	}

	void update(){

		auto editor = SplatEditor::instance;
		bool isLeftClicked = Runtime::mouseEvents.button == 0 && Runtime::mouseEvents.action == 1;

		if (editor->deviceState.hovered_depth != Infinity) {
			vec3 pos = editor->deviceState.hovered_pos;

			ImVec2 impos;
			impos.x = Runtime::mousePosition.x; 
			impos.y = Runtime::mousePosition.y;

			drawCircle_tpa(impos, 5, 0xffff0000);

			if(isLeftClicked){
				points.push_back(pos);
			}

			// draw up-vector
			if(points.size() == 2){
				vec3 v01 = points[1] - points[0];
				vec3 v02 = pos - points[0];
				vec3 up = glm::normalize(glm::cross(v02, v01));

				vec3 center = (points[0] + points[1] + pos) / 3.0f;
				vec3 target = center + up;

				editor->drawLine(center, target, 0xffff0000);
			}
		}
		mat4 view = glm::inverse(Runtime::controls->world);

		vec3 campos = vec3{view * vec4{0.0f, 0.0f, 0.0f, 1.0f}};

		for(vec3 point : points){

			float distance = glm::length(campos - point);

			float size = 20.0f * distance / float(GLRenderer::width);
			editor->drawSphere({.pos = point, .scale = vec3{size}});
		}

		// after setting the third point, we compute and apply the transformation and stop.
		if(points.size() == 3){

			vec3 v01 = points[1] - points[0];
			vec3 v02 = points[2] - points[0];
			vec3 up = glm::normalize(glm::cross(v02, v01));

			vec3 center = (points[0] + points[1] + points[2]) / 3.0f;
			vec3 target = center + up;
			vec3 right = v01;

			mat4 transform = glm::lookAt(center, target, v01);

			shared_ptr<SceneNode> selected = editor->getSelectedNode();

			if(selected){

				struct Decomposition{
					vec3 scale;
					quat rotation;
					vec3 translation;
					vec3 skew;
					vec4 perspective;
				};

				Decomposition start, end;

				decompose(selected->transform, start.scale, start.rotation, start.translation, start.skew, start.perspective);
				decompose(transform * selected->transform, end.scale, end.rotation, end.translation, end.skew, end.perspective);
				// selected->transform = transform * selected->transform;

				// mat4 start = selected->transform;
				// mat4 end = transform * selected->transform;

				shared_ptr<ThreePointAlignUndoAction> undoAction = make_shared<ThreePointAlignUndoAction>();
				undoAction->node = selected;
				undoAction->before = selected->transform;
				undoAction->after = transform * selected->transform;
				editor->addAction(undoAction);
				
				selected->transform = transform * selected->transform;

				// TWEEN::animate(0.1f, [=](double u){
				// 	float a = (1.0 - u);
				// 	float b = u;

				// 	mat4 scaleMatrix = glm::scale(a * start.scale + b * end.scale);
				// 	mat4 rotationMatrix = glm::toMat4(a * start.rotation + b * end.rotation);
				// 	mat4 translationMatrix = glm::translate(a * start.translation + b * end.translation);

				// 	selected->transform = translationMatrix * rotationMatrix * scaleMatrix;
				// });

				
			}



			editor->setAction(nullptr);
		}
		
	}

	void stop(){
		auto editor = SplatEditor::instance;


	}

};