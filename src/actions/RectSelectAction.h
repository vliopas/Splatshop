#pragma once

#include "InputAction.h"

struct RectSelectUndoAction : public Action{

	unordered_map<SNSplats*, CUdeviceptr> splatmasks;

	~RectSelectUndoAction(){
		println("Freeing RectSelectUndoAction undodata");
		for(auto [node, cptr] : splatmasks){
			CURuntime::free(cptr);
		}
	}

	void undo(){
		auto editor = SplatEditor::instance;

		for(auto [node, mask] : splatmasks){
			void* args[] = {&node->dmng.data, &mask};
			editor->prog_gaussians_editing->launch("kernel_swap_selection", args, node->dmng.data.count);
		}
	}

	void redo(){
		auto editor = SplatEditor::instance;

		for(auto [node, mask] : splatmasks){
			void* args[] = {&node->dmng.data, &mask};
			editor->prog_gaussians_editing->launch("kernel_swap_selection", args, node->dmng.data.count);
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

struct RectSelectAction : public InputAction{

	shared_ptr<RectSelectUndoAction> currentUndoAction = nullptr;

	void beginUndoable(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		// Create new undo action
		currentUndoAction = make_shared<RectSelectUndoAction>();

		// Create splatmasks
		editor->scene.process<SNSplats>([&](SNSplats* node) {
			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			uint32_t requiredBits = node->dmng.data.count;
			uint32_t requiredBytes = (requiredBits / 8) + 4;
			CUdeviceptr cptr_splatmask = CURuntime::alloc("RectSelectUndoAction splatmask", requiredBytes);
			//cuMemsetD8(cptr_splatmask, 0, requiredBytes);
			editor->prog_gaussians_editing->launch("kernel_get_selectionmask", {&node->dmng.data, &cptr_splatmask}, node->dmng.data.count);
			
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

	void applySelection(){
		auto editor = SplatEditor::instance;

		RenderTarget target;
		target.width = GLRenderer::width;
		target.height = GLRenderer::height;
		target.framebuffer = (uint64_t*)editor->virt_framebuffer->cptr;
		target.indexbuffer = nullptr;
		target.view = mat4(GLRenderer::camera->view); // * scene.transform;
		target.proj = GLRenderer::camera->proj;

		editor->scene.process<SNSplats>([&](SNSplats* node) {
			auto data = node->dmng.data;

			if (!node->visible) return;
			if (node->locked) return;

			data.transform = node->transform_global;

			CUdeviceptr cptr_splatmask = 0;
			if(currentUndoAction && currentUndoAction->splatmasks.contains(node)){
				cptr_splatmask = currentUndoAction->splatmasks[node];
			}

			bool apply = true;
			editor->prog_gaussians_editing->launch("kernel_rectselect", {&editor->launchArgs, &target, &data, &editor->settings.rectselect, &apply, &cptr_splatmask}, data.count);
		});

		endUndoable();
	}

	void start(){

		// auto editor = SplatEditor::instance;

	}

	void update(){

		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;
		auto& scene = editor->scene;
		auto& launchArgs = editor->launchArgs;

		RenderTarget target;
		target.width = GLRenderer::width;
		target.height = GLRenderer::height;
		target.framebuffer = (uint64_t*)editor->virt_framebuffer->cptr;
		target.indexbuffer = nullptr;
		target.view = mat4(GLRenderer::camera->view); // * scene.transform;
		target.proj = GLRenderer::camera->proj;

		bool isCtrlDown  = Runtime::keyStates[341] != 0;
		bool isAltDown   = Runtime::keyStates[342] != 0;
		bool isShiftDown = Runtime::keyStates[340] != 0;
		bool isLeftClicked = Runtime::mouseEvents.button == 0 && Runtime::mouseEvents.action == 1;
		bool isLeftDown = Runtime::mouseEvents.isLeftDown;
		bool isRightClicked = Runtime::mouseEvents.button == 1 && Runtime::mouseEvents.action == 1;
		static bool wasLeftDown = false;
		bool leftReleased = wasLeftDown && !isLeftDown;
		bool leftPressed = wasLeftDown == false && isLeftDown == true;
		wasLeftDown = isLeftDown;

		if(Runtime::mouseEvents.isLeftDownEvent()){
			settings.rectselect.start = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
			settings.rectselect.end = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
			settings.rectselect.startpos_specified = true;
			settings.rectselect.unselecting = isAltDown;

			beginUndoable();

			if(!isShiftDown && !isAltDown){
				editor->deselectAll();
			}
		}

		if(isLeftDown){
			settings.rectselect.end = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
		}

		if(Runtime::mouseEvents.isLeftUpEvent()){
			applySelection();
		}

		if(settings.rectselect.startpos_specified){
			auto drawlist = ImGui::GetForegroundDrawList();

			ImU32 white = IM_COL32(255, 255, 255, 127);
			ImU32 color = 0;
			color = IM_COL32(255,   0,   0, 255);
			// color = IM_COL32(  0, 255, 255, 255);
			// color = IM_COL32(  0,   0,   0, 255);

			float thickness = 3.0f;

			float height = GLRenderer::height;
			ImVec2 imstart = ImVec2(settings.rectselect.start.x, height - settings.rectselect.start.y);
			ImVec2 imend = ImVec2(settings.rectselect.end.x, height - settings.rectselect.end.y);
			
			drawlist->AddLine(
				ImVec2(imstart.x, imstart.y), 
				ImVec2(imend.x, imstart.y), 
				white, thickness);
			drawlist->AddLine(
				ImVec2(imend.x, imstart.y), 
				ImVec2(imend.x, imend.y), 
				white, thickness);
			drawlist->AddLine(
				ImVec2(imend.x, imend.y), 
				ImVec2(imstart.x, imend.y), 
				white, thickness);
			drawlist->AddLine(
				ImVec2(imstart.x, imstart.y), 
				ImVec2(imstart.x, imend.y), 
				white, thickness);
		}

		bool apply = false;
		scene.process<SNSplats>([&](SNSplats* node) {
			auto data = node->dmng.data;

			if (!node->visible) return;
			if (node->locked) return;

			data.transform = node->transform_global;

			void* splatmask = nullptr;
			editor->prog_gaussians_editing->launch("kernel_rectselect", {&launchArgs, &target, &data, &settings.rectselect, &apply, &splatmask}, data.count);
		});
		
	}

	void stop(){
		auto editor = SplatEditor::instance;

		applySelection();

		editor->settings.rectselect.active = false;
	}

	void makeToolbarSettings(){

		auto editor = SplatEditor::instance;

		int intersectionmode = editor->settings.brush.intersectionmode;
		ImGui::Text("Brush intersects splat's"); 

		ImGui::SameLine();
		ImGui::RadioButton("Center", &intersectionmode, BRUSH_INTERSECTION_CENTER); 

		ImGui::SameLine();
		ImGui::RadioButton("Border", &intersectionmode, BRUSH_INTERSECTION_BORDER); 
		editor->settings.brush.intersectionmode = intersectionmode;

	}
};