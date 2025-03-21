#pragma once

#include "actions/VR_BrushSelectAction.h"
#include "actions/VR_BrushPaintAction.h"
#include "actions/VR_PlaceAssetAction.h"

const ImVec2 VR_BUTTON_SIZE = ImVec2(0, 60);
constexpr int row_min_height_vr = 43;

namespace GUIVR{
	bool Button(const char* label, const ImVec2& size){

		auto cursor_start = ImGui::GetCursorPos();

		bool result = ImGui::Button(label, size);

		auto drawlist = ImGui::GetForegroundDrawList();
		auto windowSize = ImGui::GetWindowSize();
		auto windowPos = ImGui::GetWindowPos();
		auto cursor_end = ImGui::GetCursorPos();

		ImVec2 min = { 
			windowPos.x + cursor_start.x, 
			windowPos.y + cursor_start.y,
		};
		ImVec2 max = {
			windowPos.x + cursor_start.x + size.x, 
			windowPos.y + cursor_start.y + size.y,
		};
		drawlist->AddRect(min, max, IM_COL32_BLACK, 5.0f, ImDrawCornerFlags_All, 5.0f);

		return result;
	}

	bool ImageButton(
		const char* str_id,
		ImTextureID user_texture_id, 
		const ImVec2& image_size, 
		const ImVec2& uv0, 
		const ImVec2& uv1, 
		const ImVec4& bg_col, 
		const ImVec4& tint_col
	){

		auto cursor_start = ImGui::GetCursorPos();

		bool result = ImGui::ImageButton(str_id, user_texture_id, image_size, uv0, uv1, bg_col, tint_col);

		auto drawlist = ImGui::GetForegroundDrawList();
		auto windowSize = ImGui::GetWindowSize();
		auto windowPos = ImGui::GetWindowPos();
		auto cursor_end = ImGui::GetCursorPos();

		ImVec2 min = { 
			windowPos.x + cursor_start.x + 2.0f, 
			windowPos.y + cursor_start.y + 2.0f,
		};
		ImVec2 max = {
			windowPos.x + cursor_start.x + image_size.x + 6.0f,
			windowPos.y + cursor_start.y + image_size.y + 4.0f,
		};
		
		auto color = IM_COL32_BLACK;
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
			color = ImGui::GetColorU32(ImGuiCol_ButtonHovered, 1.0f);
		}
		

		drawlist->AddRect(min, max, color, 5.0f, ImDrawCornerFlags_All, 5.0f);

		return result;
	}
}




void make_layeritem_vr(SceneNode* node, int i){

	auto editor = SplatEditor::instance;

	ImGui::TableNextRow(ImGuiTableRowFlags_None, row_min_height_vr);
	ImGui::TableNextColumn();

	{ // COLUMN SELECTABLE
		ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
		string id = format("  ##vr_{}_{}", node->name, i);
		if (ImGui::Selectable(id.c_str(), node->selected, selectable_flags, VR_BUTTON_SIZE)) {
			editor->scene.deselectAllNodes();
			node->selected = true;

			if (ImGui::GetIO().KeyCtrl){
				// TODO: on ctrl+click, all primitives of the node should be selected
				// setSelected(gd);
			}

			// updateBoundingBox(node);
		}

		if (node->selected) {
			ImGui::SetItemDefaultFocus();
		}

		ImGui::SameLine();
		float cy = ImGui::GetCursorPosY();
		//ImGui::SetCursorPosY(cy + 10.0f);
		ImGui::Text(node->name.c_str());
		//ImGui::SetCursorPosY(cy);
	}

	{ // draw selectable outline
		auto drawlist = ImGui::GetForegroundDrawList();
		auto windowSize = ImGui::GetWindowSize();
		auto windowPos = ImGui::GetWindowPos();
		auto cursor_end = ImGui::GetCursorPos();

		ImVec2 min = { windowPos.x + 10.0f, windowPos.y + cursor_end.y - row_min_height_vr - 24.0f};
		ImVec2 max = { windowPos.x + windowSize.x - 10.0f, windowPos.y + cursor_end.y + 20.0f - 24.0f};
		drawlist->AddRect(min, max, IM_COL32_BLACK, 5.0f, ImDrawCornerFlags_All, 5.0f);
	}

	{ // COLUMN ACTIONS
		ImGui::TableNextColumn();

		ImVec2 buttonSize = ImVec2(32.0f, 32.0f);
		float symbolSize = 32.0f;
		float cy = ImGui::GetCursorPosY();
		ImGui::SetCursorPosY(cy + 10.0f);

		ImTextureID my_tex_id = (void*)(intptr_t)Runtime::gltex_symbols_32x32;
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
		
		ImVec2 uv_eye_open_0   = ImVec2(2.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 0.0f * symbolSize);
		ImVec2 uv_eye_open_1   = ImVec2(3.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 1.0f * symbolSize / Runtime::gltex_symbols_32x32_height);
		ImVec2 uv_eye_closed_0 = ImVec2(3.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 0.0f * symbolSize);
		ImVec2 uv_eye_closed_1 = ImVec2(4.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 1.0f * symbolSize / Runtime::gltex_symbols_32x32_height);

		ImVec2 uv_lock_open_0   = ImVec2(0.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 0.0f * symbolSize);
		ImVec2 uv_lock_open_1   = ImVec2(1.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 1.0f * symbolSize / Runtime::gltex_symbols_32x32_height);
		ImVec2 uv_lock_closed_0 = ImVec2(1.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 0.0f * symbolSize);
		ImVec2 uv_lock_closed_1 = ImVec2(2.0f * symbolSize / Runtime::gltex_symbols_32x32_width, 1.0f * symbolSize / Runtime::gltex_symbols_32x32_height);

		ImVec2 uv_eye_0 = uv_eye_open_0;
		ImVec2 uv_eye_1 = uv_eye_open_1;
		if(!node->visible){
			uv_eye_0 = uv_eye_closed_0;
			uv_eye_1 = uv_eye_closed_1;
		}

		ImVec2 uv_lock_0 = uv_lock_open_0;
		ImVec2 uv_lock_1 = uv_lock_open_1;
		if(node->locked){
			uv_lock_0 = uv_lock_closed_0;
			uv_lock_1 = uv_lock_closed_1;
		}

		string label_eye = format("toggle_visibility##{}", i);
		string label_lock = format("toggle_lock##{}", i);

		ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(0, 0, 0, 0));
		// ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(0, 0, 0, 255));
		// ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(0, 0, 0, 255));
		

		float cx = ImGui::GetCursorPosX();

		// SHOW / HIDE
		if(ImGui::ImageButton(label_eye.c_str(), my_tex_id, buttonSize, uv_eye_0, uv_eye_1, bg_col, tint_col)){
			node->visible = !node->visible;
		}
		// if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
		// 	ImGui::SetTooltip("Show/Hide this layer.");
		// }


		// LOCK / UNLOCK
		ImGui::SameLine();
		ImGui::SetCursorPosX(cx + buttonSize.x + 5);
		if(ImGui::ImageButton(label_lock.c_str(), my_tex_id, buttonSize, uv_lock_0, uv_lock_1, bg_col, tint_col)){
			node->locked = !node->locked;
		}
		// if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
		// 	ImGui::SetTooltip("Lock to prevent interacting with this layer.");
		// }

		ImGui::PopStyleColor(1);
		// ImGui::PopStyleColor();
		// ImGui::PopStyleColor();
	}
}

void makeLayersVR(ImguiPage* page){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;

	page->width = 440;
	page->height = 700;

	ImGui::SetCurrentContext(page->context);
	ImGuiIO& io = page->context->IO;
	auto drawlist = ImGui::GetForegroundDrawList();

	auto framebuffer = page->framebuffer;
	framebuffer->setSize(page->width, page->height);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->handle);
	glViewport(0, 0, framebuffer->width, framebuffer->height);
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// We need to override DisplaySize if we want to draw imgui to a texture.
	io.DisplaySize.x = framebuffer->width;
	io.DisplaySize.y = framebuffer->height;

	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	editor->imguiStyleVR();

	ImU32 white = IM_COL32(255, 255, 255, 255);
	ImU32 black = IM_COL32(  0,   0,   0, 255);
	ImU32 color = 0;

	ImGuiWindowFlags window_flags = 0;
	window_flags = window_flags | ImGuiWindowFlags_NoTitleBar;
	//window_flags = window_flags | ImGuiWindowFlags_MenuBar;
	window_flags = window_flags | ImGuiWindowFlags_NoResize;
	window_flags = window_flags | ImGuiWindowFlags_NoMove;
	// window_flags = window_flags | ImGuiWindowFlags_NoBackground;

	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 255));

	auto size_layers = ImVec2(page->width, page->height);
	auto windowPos = ImVec2(0, 0);
	auto buttonSize = ImVec2(0, 60);

	ImGui::SetNextWindowPos(windowPos);
	ImGui::SetNextWindowSize(size_layers);
	ImGui::SetNextWindowBgAlpha(1.0f);

	ImGui::Begin("VR Layers", nullptr, window_flags);
	ImGui::PushFont(editor->font_vr_title);

	ImGui::Text("Layers");
	// ImGui::NewLine();

	float cy = ImGui::GetCursorPosY();
	ImGui::SetCursorPosY(cy + 20.0f);

	
	ImGui::PopFont();
	ImGui::PushFont(editor->font_vr_text);

	static ImGuiTableFlags flags =
		ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
		// | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_NoBordersInBody
		| ImGuiTableFlags_NoBordersInBody
		// | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
		| ImGuiTableFlags_SizingFixedFit 
		// | ImGuiTableFlags_SizingStretchProp
		// | ImGuiTableFlags_SizingStretchSame
		| ImGuiTableFlags_NoSavedSettings;

	ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(10.0f, 10.0f));

	static int selectedIndex = -1;

	// ImVec2 cell_padding = {20.0f, 10.0f};
	// ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, cell_padding);

	if(ImGui::BeginTable("Models##listOfModels", 2, flags)){

		ImGui::TableSetupColumn("Model",   ImGuiTableColumnFlags_WidthStretch, 0.0f);
		// ImGui::TableSetupColumn("Mem",     ImGuiTableColumnFlags_WidthFixed, 50.0f);
		ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 80.0f);

		int i = 0;
		editor->scene.root->traverse([&](SceneNode* node){

			if(node->hidden) return;

			
			if(dynamic_cast<SNSplats*>(node) != nullptr){
				make_layeritem_vr((SNSplats*)node, i);
				i++;
			}else if(dynamic_cast<SNPoints*>(node) != nullptr){
				make_layeritem_vr((SNPoints*)node, i);
				i++;
			}else if(dynamic_cast<SNTriangles*>(node) != nullptr){
				make_layeritem_vr((SNTriangles*)node, i);
				i++;
			}
			
		});

		ImGui::EndTable();
	}

	// float cy = ImGui::GetCursorPosY();
	ImGui::SetCursorPosY(size_layers.y - 80.0f);
	if(ImGui::Button("Add Layer##VR", {-1.0f, buttonSize.y})){
		editor->scene.deselectAllNodes();

		string name = "New Splat Layer";
		shared_ptr<Splats> splats = make_shared<Splats>();
		shared_ptr<SNSplats> node = make_shared<SNSplats>(name, splats);
		editor->scene.world->children.push_back(node);

		node->selected = true;
	}

	{
		auto cursor_end = ImGui::GetCursorPos();
		ImVec2 min = { windowPos.x + 10.0f                 , windowPos.y + cursor_end.y - row_min_height_vr - 24.0f};
		ImVec2 max = { windowPos.x + size_layers.x - 10.0f , windowPos.y + cursor_end.y + 20.0f - 24.0f};
		drawlist->AddRect(min, max, black, 5.0f, ImDrawCornerFlags_All, 5.0f);
	}

	ImGui::PopStyleVar();
	ImGui::PopFont();
	ImGui::PopStyleColor();

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}



void makeAssetsVR(ImguiPage* page){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;

	int itemsPerRow = 5;
	page->width = 720;
	page->height = 700;

	ImGui::SetCurrentContext(page->context);
	ImGuiIO& io = page->context->IO;
	auto drawlist = ImGui::GetForegroundDrawList();
	
	auto framebuffer = page->framebuffer;
	framebuffer->setSize(page->width, page->height);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->handle);
	glViewport(0, 0, framebuffer->width, framebuffer->height);
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// We need to override DisplaySize if we want to draw imgui to a texture.
	io.DisplaySize.x = framebuffer->width;
	io.DisplaySize.y = framebuffer->height;

	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	editor->imguiStyleVR();

	ImU32 white = IM_COL32(255, 255, 255, 255);
	ImU32 black = IM_COL32(  0,   0,   0, 255);
	ImU32 color = 0;

	ImGuiWindowFlags window_flags = 0;
	window_flags = window_flags | ImGuiWindowFlags_NoTitleBar;
	//window_flags = window_flags | ImGuiWindowFlags_MenuBar;
	window_flags = window_flags | ImGuiWindowFlags_NoResize;
	window_flags = window_flags | ImGuiWindowFlags_NoMove;
	// window_flags = window_flags | ImGuiWindowFlags_NoBackground;

	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 255));

	auto size_layers = ImVec2(page->width, page->height);
	auto windowPos = ImVec2(0, 0);

	ImGui::SetNextWindowPos(windowPos);
	ImGui::SetNextWindowSize(size_layers);
	ImGui::SetNextWindowBgAlpha(1.0f);

	ImGui::Begin("VR Prototype Assets", nullptr, window_flags);
	ImGui::PushFont(editor->font_vr_title);

	ImGui::Text("Assets");
	//ImGui::Text(u8"Assets \u2122");

	float cy = ImGui::GetCursorPosY();
	ImGui::SetCursorPosY(cy + 20.0f);

	ImGui::PopFont();
	ImGui::PushFont(editor->font_vr_text);

	ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

	ImVec2 uv0 = ImVec2(0.0f, 1.0f);
	ImVec2 uv1 = ImVec2(1.0f, 0.0f);

	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();
	// ImVec2 thumbSize = ImVec2(128.0f, 128.0f);
	ImVec2 thumbSize = {
		(vMax.x - vMin.x) / float(itemsPerRow) - 15.0f,
		(vMax.x - vMin.x) / float(itemsPerRow) - 15.0f,
	};

	int i = 0;
	for(auto node : AssetLibrary::assets){
		ImTextureID handle = (void*)(intptr_t)Runtime::gltex_symbols_32x32;

		if(node->thumbnail){
			handle = (void*)(intptr_t)node->thumbnail->handle;
		}

		string id = format("{}##{}", node->name, i);
		if(i > 0 && (i % itemsPerRow) != 0){
			ImGui::SameLine();
		}

		if(GUIVR::ImageButton(id.c_str(), handle, thumbSize, uv0, uv1, bg_col, tint_col)){
			// editor->state.action = ACTION_PLACING;
			// editor->setPlacingItem(node);

			// TODO: VR Placing Item action
			shared_ptr<VR_PlaceAssetAction> action = make_shared<VR_PlaceAssetAction>();
			action->placingItem = node;
			editor->setAction(action);
		}
		
		i++;
	}

	// ImGui::PopStyleVar();
	ImGui::PopFont();
	ImGui::PopStyleColor();

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void makeBrushesVR(ImguiPage* page){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;

	page->width = 440;
	page->height = 500;

	ImGui::SetCurrentContext(page->context);
	ImGuiIO& io = page->context->IO;
	auto drawlist = ImGui::GetForegroundDrawList();
	
	auto framebuffer = page->framebuffer;
	framebuffer->setSize(page->width, page->height);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->handle);
	glViewport(0, 0, framebuffer->width, framebuffer->height);
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// We need to override DisplaySize if we want to draw imgui to a texture.
	io.DisplaySize.x = framebuffer->width;
	io.DisplaySize.y = framebuffer->height;

	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	editor->imguiStyleVR();

	ImU32 white = IM_COL32(255, 255, 255, 255);
	ImU32 black = IM_COL32(  0,   0,   0, 255);
	ImU32 color = 0;


	ImGuiWindowFlags window_flags = 0;
	window_flags = window_flags | ImGuiWindowFlags_NoTitleBar;
	//window_flags = window_flags | ImGuiWindowFlags_MenuBar;
	window_flags = window_flags | ImGuiWindowFlags_NoResize;
	window_flags = window_flags | ImGuiWindowFlags_NoMove;
	// window_flags = window_flags | ImGuiWindowFlags_NoBackground;

	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 255));

	auto size_layers = ImVec2(page->width, page->height);
	auto windowPos = ImVec2(0, 0);

	ImGui::SetNextWindowPos(windowPos);
	ImGui::SetNextWindowSize(size_layers);
	ImGui::SetNextWindowBgAlpha(1.0f);

	ImGui::Begin("VR Prototype Select & Delete", nullptr, window_flags);
	ImGui::PushFont(editor->font_vr_title);

	 ImGui::Text("Select & Delete");
	//ImGui::Text(u8"Assets \u2122");

	float cy = ImGui::GetCursorPosY();
	ImGui::SetCursorPosY(cy + 20.0f);

	ImGui::PopFont();
	ImGui::PushFont(editor->font_vr_text);

	ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

	ImVec2 uv0 = ImVec2(0.0f, 1.0f);
	ImVec2 uv1 = ImVec2(1.0f, 0.0f);

	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();

	ImVec2 buttonSize = {
		(vMax.x - vMin.x) / 3.0f - 12.0f,
		(vMax.x - vMin.x) / 3.0f - 12.0f,
	};


	if(GUIVR::Button("Select", buttonSize)){
		editor->settings.brush.mode = BRUSHMODE::SELECT;

		shared_ptr<VR_BrushSelectAction> action = make_shared<VR_BrushSelectAction>();
		editor->setAction(action);
	}
	
	ImGui::SameLine(); 
	if(GUIVR::Button("Delete", buttonSize)){
		editor->settings.brush.mode = BRUSHMODE::ERASE;

		shared_ptr<VR_BrushSelectAction> action = make_shared<VR_BrushSelectAction>();
		editor->setAction(action);
	}

	ImGui::SameLine();
	if(GUIVR::Button("Duplicate", buttonSize)){
		// editor->settings.brush.mode = BRUSHMODE::REMOVE_FLAGS;

		auto numSelected = editor->getNumSelectedSplats();
		if(numSelected > 0){
			// Duplicate selected splats
			FilterRules rules;
			rules.selection = FILTER_SELECTION_SELECTED;
			editor->filterToNewLayer_undoable(rules);
		}
	}

	ImGui::Text(" ");

	if(GUIVR::Button("Invert", buttonSize)){
		editor->invertSelection_undoable();
	}

	ImGui::SameLine(); 
	if(GUIVR::Button("Deselect\nAll", buttonSize)){
		editor->deselectAll_undoable();
	}

	ImGui::SameLine(); 
	if(GUIVR::Button("Delete\nSelected", buttonSize)){
		// editor->deleteSelection();
		editor->deleteSelection_undoable();
	}
	


	ImGui::PopFont();
	ImGui::PopStyleColor();

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void makePaintingVR(ImguiPage* page){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;

	page->width = 440;
	page->height = 500;

	ImGui::SetCurrentContext(page->context);
	ImGuiIO& io = page->context->IO;
	auto drawlist = ImGui::GetForegroundDrawList();
	
	auto framebuffer = page->framebuffer;
	framebuffer->setSize(page->width, page->height);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->handle);
	glViewport(0, 0, framebuffer->width, framebuffer->height);
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// We need to override DisplaySize if we want to draw imgui to a texture.
	io.DisplaySize.x = framebuffer->width;
	io.DisplaySize.y = framebuffer->height;

	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	editor->imguiStyleVR();

	ImU32 white = IM_COL32(255, 255, 255, 255);
	ImU32 black = IM_COL32(  0,   0,   0, 255);
	ImU32 color = 0;


	ImGuiWindowFlags window_flags = 0;
	window_flags = window_flags | ImGuiWindowFlags_NoTitleBar;
	//window_flags = window_flags | ImGuiWindowFlags_MenuBar;
	window_flags = window_flags | ImGuiWindowFlags_NoResize;
	window_flags = window_flags | ImGuiWindowFlags_NoMove;
	// window_flags = window_flags | ImGuiWindowFlags_NoBackground;

	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 255));

	auto size_layers = ImVec2(page->width, page->height);
	auto windowPos = ImVec2(0, 0);

	ImGui::SetNextWindowPos(windowPos);
	ImGui::SetNextWindowSize(size_layers);
	ImGui::SetNextWindowBgAlpha(1.0f);

	ImGui::Begin("VR Painting", nullptr, window_flags);
	ImGui::PushFont(editor->font_vr_title);

	 ImGui::Text("Painting");
	//ImGui::Text(u8"Assets \u2122");

	float cy = ImGui::GetCursorPosY();
	ImGui::SetCursorPosY(cy + 20.0f);

	ImGui::PopFont();
	ImGui::PushFont(editor->font_vr_text);

	ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
	ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

	ImVec2 uv0 = ImVec2(0.0f, 1.0f);
	ImVec2 uv1 = ImVec2(1.0f, 0.0f);

	ImVec2 vMin = ImGui::GetWindowContentRegionMin();
	ImVec2 vMax = ImGui::GetWindowContentRegionMax();

	ImVec2 buttonSize = {
		(vMax.x - vMin.x) / 3.0f - 12.0f,
		(vMax.x - vMin.x) / 3.0f - 12.0f,
	};


	if(GUIVR::Button("Paint", buttonSize)){
		shared_ptr<VR_BrushPaintAction> action = make_shared<VR_BrushPaintAction>();
		editor->setAction(action);
	}

	static ImVec4 pickedColor(1.0f, 0.0f, 0.0f, 1.0f);
	ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoInputs;
	flags |= ImGuiColorEditFlags_NoSmallPreview;
	flags |= ImGuiColorEditFlags_NoSidePreview;
	flags |= ImGuiColorEditFlags_AlphaBar;
	ImGui::ColorPicker4("##picker", (float*)&pickedColor, flags);

	editor->settings.brush.color.r = pickedColor.x;
	editor->settings.brush.color.g = pickedColor.y;
	editor->settings.brush.color.b = pickedColor.z;
	editor->settings.brush.color.a = pickedColor.w;

	ImGui::PopFont();
	ImGui::PopStyleColor();

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}