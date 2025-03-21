#pragma once

#include "InputAction.h"

struct SphereSelectUndoAction : public Action{

	unordered_map<SNSplats*, CUdeviceptr> splatmasks;

	~SphereSelectUndoAction(){
		println("Freeing SphereSelectUndoAction::Undodata");
		for(auto [node, cptr] : splatmasks){
			CURuntime::free(cptr);
		}
	}

	void undo(){
		auto editor = SplatEditor::instance;

		for(auto [node, mask] : splatmasks){
			void* args[] = {&node->dmng.data, &mask};
			editor->prog_gaussians_editing->launch("kernel_deselect_masked", args, node->dmng.data.count);
		}
	}

	void redo(){
		auto editor = SplatEditor::instance;

		for(auto [node, mask] : splatmasks){
			void* args[] = {&node->dmng.data, &mask};
			editor->prog_gaussians_editing->launch("kernel_select_masked", args, node->dmng.data.count);
		}
	}

	int64_t byteSize(){
		int64_t totalSize = 0;

		for(auto [node, cptr] : splatmasks){
			if(cptr) totalSize += CURuntime::getSize(cptr);
		}

		return totalSize;
	}

};

struct SphereSelectAction : public InputAction{

	shared_ptr<SphereSelectUndoAction> currentUndoAction = nullptr;
	static const int SPACE_WORLD  = 0;
	static const int SPACE_SCREEN = 1;

	inline static int space = SPACE_SCREEN;
	inline static float size_world = 1.0f;
	inline static float size_screen = 0.3f;

	void start(){
		auto editor = SplatEditor::instance;
	}

	void update(){

		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		static bool wasLeftDown = false;
		bool isLeftDown = Runtime::mouseEvents.isLeftDown;
		bool leftReleased = wasLeftDown && !isLeftDown;
		bool leftPressed = wasLeftDown == false && isLeftDown == true;
		wasLeftDown = isLeftDown;

		if(leftPressed){
			beginUndoable();
		}else if (leftReleased) {
			endUndoable();
		}

		if(editor->deviceState.hovered_depth < Infinity){
			vec3 spherePos = editor->deviceState.hovered_pos;
			float radius = 2.0f;

			if(space == SPACE_WORLD){
				radius = size_world;
			}else{
				radius = editor->deviceState.hovered_depth * size_screen / 2.0f;
			}

			editor->scene.process<SNSplats>([&](SNSplats* node) {

				if(!node->visible) return;
				if(node->locked) return;

				auto data = node->dmng.data;
				data.transform = node->transform_global;

				CUdeviceptr cptr_splatmask = 0;
				if(currentUndoAction && currentUndoAction->splatmasks.contains(node)){
					cptr_splatmask = currentUndoAction->splatmasks[node];
				}

				void* args[] = {&editor->launchArgs, &data, &spherePos, &radius, &cptr_splatmask};
				editor->prog_gaussians_editing->launch("kernel_select_sphere", args, data.count);
			});

			auto sample = [&](float u){
				vec3 pos;
				pos.x = radius * cos(u);
				pos.y = radius * sin(u);
				pos.z = 0.0f;

				return pos;
			};

			int segments = 32;
			vec3 center = editor->deviceState.hovered_pos;
			for(int i = 0; i < segments; i++){

				float u  = 2.0f * 3.1415f * float(i) / float(segments);
				float u1 = 2.0f * 3.1415f * float(i + 1) / float(segments);

				vec3 p0 = sample(u);
				vec3 p1 = sample(u1);

				editor->drawLine(center + p0, center + p1, 0xffff0000);
				editor->drawLine(center + vec3{p0.z, p0.x, p0.y}, center + vec3{p1.z, p1.x, p1.y}, 0xff0000ff);
				editor->drawLine(center + vec3{p0.x, p0.z, p0.y}, center + vec3{p1.x, p1.z, p1.y}, 0xff00ff00);


			}
		}

		
		
	}

	void stop(){
		
		auto editor = SplatEditor::instance;

		// TODO: is this still necessary after switching to "dirty" handling?
		editor->scene.process<SNSplats>([&](SNSplats* node) {

			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			editor->prog_gaussians_editing->launch("kernel_clear_highlighting", {&editor->launchArgs, &data}, data.count);
		});

		// editor->settings.brush.mode = BRUSHMODE::NONE; 

		endUndoable();
	}

	void makeToolbarSettings(){
		
		ImGui::Text("Space: ");

		ImGui::SameLine();
		ImGui::RadioButton("World", &space, SPACE_WORLD); 

		ImGui::SameLine();
		ImGui::RadioButton("Screen", &space, SPACE_SCREEN); 

		ImGui::SameLine();
		ImGui::Text("Size: ");

		if(space == SPACE_WORLD){

			ImGui::SameLine();
			ImGui::SetNextItemWidth(100.0f);
			ImGui::SliderFloat(" ", &size_world, 0.01f, 10.0f, "%.2f");

		}else if(space == SPACE_SCREEN){

			ImGui::SameLine();
			ImGui::SetNextItemWidth(100.0f);
			ImGui::SliderFloat(" ", &size_screen, 0.01f, 1.0f, "%.2f");

		}

	}

	void beginUndoable(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		// Create new undo action
		currentUndoAction = make_shared<SphereSelectUndoAction>();

		// Create splatmasks
		editor->scene.process<SNSplats>([&](SNSplats* node) {
			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			uint32_t requiredBits = node->dmng.data.count;
			uint32_t requiredBytes = (requiredBits / 8) + 4;
			CUdeviceptr cptr_splatmask = CURuntime::alloc("SphereSelectUndoAction splatmask", requiredBytes);
			cuMemsetD8(cptr_splatmask, 0, requiredBytes);
			
			currentUndoAction->splatmasks[node] = cptr_splatmask;
		});
	}

	void endUndoable(){
		auto editor = SplatEditor::instance;

		editor->scene.process<SNSplats>([&](SNSplats* node) {
			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			editor->prog_gaussians_editing->launch("kernel_clear_highlighting", {&editor->launchArgs, &data}, data.count);
		});

		if(currentUndoAction){
			editor->addAction(currentUndoAction);
			currentUndoAction = nullptr;
		}
	}

};