
#pragma once

#include "imutils.h"

constexpr int freeze_cols = 1;
constexpr int freeze_rows = 1;
constexpr int row_min_height = 43;

string formatShortMemMsg(uint64_t memory){
	string msg = "-";
	if(memory <= 100'000'000llu){
		msg = format("{:.1f} MB", double(memory) / 1'000'000.0);
	}else if(memory <= 1'000'000'000llu){
		msg = format("{:.1f} GB", double(memory) / 1'000'000'000.0);
	}else{
		msg = format("{:.1f} GB", double(memory) / 1'000'000'000.0);
	}

	return msg;
}

void make_layeritem_triangles(shared_ptr<SNTriangles> node, int i){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;
	auto& scene = editor->scene;


	string name = node->name;

	ImGui::TableNextRow(ImGuiTableRowFlags_None, row_min_height);

	float offsetToCenterY = 12.0f;

	{ // COLUMN - SELECTABLE
		ImGui::TableNextColumn();

		ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
		string id = format("{}##{}", name, uint64_t(node.get()));
		if (ImGui::Selectable(id.c_str(), node->selected, selectable_flags, ImVec2(0, row_min_height))) {
			scene.deselectAllNodes();
			node->selected = true;
		}

		bool isHovered = ImGui::IsItemHovered();
		bool isDoubleClicked = ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0);

		if (isDoubleClicked) {
			Runtime::controls->focus(node->aabb.min, node->aabb.max, 1.0f);
		}

		if (node->selected) {
			ImGui::SetItemDefaultFocus();
		}
	}

	{ // COLUMN MEMORY
		ImGui::TableNextColumn();

		

		double millions = double(node->data.count) / 1'000'000.0;
		uint64_t memory = node->getGpuMemoryUsage();
		string msg = formatShortMemMsg(memory);

		msg = format("{:4.1f} M \n{:>7} ", millions, msg);

		// println("{}", msg);

		float cy = ImGui::GetCursorPosY();
		ImGui::SetCursorPosY(cy + offsetToCenterY);
		ImGui::Text(msg.c_str());
		
		// uint64_t memory = node->getGpuMemoryUsage();
		// string msg = formatShortMemMsg(memory);
		// ImGui::Text(msg.c_str());
	}

	
	{ // COLUMN ACTIONS
		ImGui::TableNextColumn();

		// ImGui::Text("DEF");

		ImTextureID my_tex_id = (void*)(intptr_t)Runtime::gltex_symbols;
		ImVec2 size = ImVec2(16.0f, 16.0f);
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		float iconSize = 16.0f;

		ImVec2 uv_eye_open_0   = ImVec2(2.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_eye_open_1   = ImVec2(3.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);
		ImVec2 uv_eye_closed_0 = ImVec2(3.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_eye_closed_1 = ImVec2(4.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);

		ImVec2 uv_lock_open_0   = ImVec2(0.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_lock_open_1   = ImVec2(1.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);
		ImVec2 uv_lock_closed_0 = ImVec2(1.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_lock_closed_1 = ImVec2(2.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);

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

		// SHOW / HIDE
		if(ImGui::ImageButton(label_eye.c_str(), my_tex_id, size, uv_eye_0, uv_eye_1, bg_col, tint_col)){
			node->visible = !node->visible;
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("Show/Hide this layer.");
		}

		// LOCK / UNLOCK
		ImGui::SameLine();
		if(ImGui::ImageButton(label_lock.c_str(), my_tex_id, size, uv_lock_0, uv_lock_1, bg_col, tint_col)){
			node->locked = !node->locked;
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("Lock to prevent interacting with this layer.");
		}

	}
}

void make_layeritem_splats(shared_ptr<SNSplats> node, int i){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;
	auto& scene = editor->scene;

	shared_ptr<Splats> splats = node->splats;
	GaussianData& gd = node->dmng.data;

	string name = node->name;

	ImGui::TableNextRow(ImGuiTableRowFlags_None, row_min_height);

	float offsetToCenterY = 12.0f;

	{ // COLUMN - SELECTABLE
		ImGui::TableNextColumn();

		ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
		string id = format("##{}_{}", name, node->ID);

		float cy = ImGui::GetCursorPosY();
		// ImGui::SetCursorPosY(cy + 10);
		if (ImGui::Selectable(id.c_str(), node->selected, selectable_flags, ImVec2(0, row_min_height))) {
			scene.deselectAllNodes();
			node->selected = true;

			if (ImGui::GetIO().KeyCtrl){
				editor->selectAllInNode_undoable(node);
			}

			editor->updateBoundingBox(node.get());
		}

		// POPUP MENU 
		if (ImGui::BeginPopupContextItem()){

			editor->temporarilyDisableShortcuts();

			ImGui::MenuItem("(Context Menu)", NULL, false, false);
			
			static char strName[256] = "";
			ImGui::Text("Name: ");
			ImGui::SameLine();
			if(ImGui::InputTextWithHint("", "Enter New Name", strName, IM_ARRAYSIZE(strName)));
			ImGui::SameLine();
			if(ImGui::Button("Set Name")){
				name = string(strName);
				node->name = name;

				memset(strName, 0, sizeof(strName));
			}

			if(ImGui::MenuItem("Create Asset From Layer", NULL)){
				
				auto newNode = editor->clone(node.get());

				// bakeTransformation(newNode->dmng.data);
				// bake transformation into splats
				editor->applyTransformation(newNode->dmng.data, newNode->dmng.data.transform);
				newNode->dmng.data.transform = mat4(1.0f);

				newNode->transform = mat4(1.0f);
				newNode->dmng.data.transform = mat4(1.0f);
				newNode->dmng.data.writeDepth = false;

				editor->updateBoundingBox(newNode.get());

				newNode->aabb.min = newNode->dmng.data.min;
				newNode->aabb.max = newNode->dmng.data.max;

				editor->createOrUpdateThumbnail(newNode.get());

				AssetLibrary::assets.push_back(newNode);
			}

			if(ImGui::MenuItem("Merge Into Layer Below (ctrl + e)", NULL)){

				vector<shared_ptr<SceneNode>> layers = editor->getLayers();
				for(int i = 0; i < layers.size(); i++){
					bool isLast = i == layers.size() - 1;
					if(layers[i] == node && !isLast){
						shared_ptr<SceneNode> next = layers[i + 1];

						editor->merge(node, next);
					}
				}
			}
			
			ImGui::EndPopup();
		}else{
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
				ImGui::SetTooltip("- Ctrl+Click to select this layer's splats.\n- Double Click to focus viewpoint on node.");
			}
		}

		bool isHovered = ImGui::IsItemHovered();
		bool isDoubleClicked = ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0);

		if (isHovered) {
			state.hoveredObjectIndex = i;
			editor->updateBoundingBox(node.get());
			// println("id: {}", id);
		}

		node->hovered = isHovered;

		if (isDoubleClicked) {
			state.doubleClickedObjectIndex = i;
			Runtime::controls->focus(node->aabb.min, node->aabb.max, 0.6f);
		}

		if (node->selected) {
			ImGui::SetItemDefaultFocus();
		}

		ImGui::SetCursorPosY(cy + offsetToCenterY);
		ImGui::Text(name.c_str());
		// ImGui::SetCursorPosY(cy);
	}



	{ // COLUMN MEMORY
		ImGui::TableNextColumn();
		
		double millions = double(gd.count) / 1'000'000.0;
		uint64_t memory = node->getGpuMemoryUsage();
		string msg = formatShortMemMsg(memory);

		msg = format("{:4.1f} M \n{:>7} ", millions, msg);

		// println("{}", msg);

		float cy = ImGui::GetCursorPosY();
		ImGui::SetCursorPosY(cy + offsetToCenterY);
		ImGui::Text(msg.c_str());
		// ImGui::SetCursorPosY(cy);

	}

	
	{ // COLUMN ACTIONS
		ImGui::TableNextColumn();

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.0f, 0.0f, 0.0f, 0.0f});

		// ImGui::Text("DEF");
		float cy = ImGui::GetCursorPosY();
		ImGui::SetCursorPosY(cy + offsetToCenterY - 4);

		ImTextureID my_tex_id = (void*)(intptr_t)Runtime::gltex_symbols;
		ImVec2 size = ImVec2(16.0f, 16.0f);
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		float iconSize = 16.0f;

		ImVec2 uv_eye_open_0   = ImVec2(2.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_eye_open_1   = ImVec2(3.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);
		ImVec2 uv_eye_closed_0 = ImVec2(3.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_eye_closed_1 = ImVec2(4.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);

		ImVec2 uv_lock_open_0   = ImVec2(0.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_lock_open_1   = ImVec2(1.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);
		ImVec2 uv_lock_closed_0 = ImVec2(1.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_lock_closed_1 = ImVec2(2.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);

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

		// SHOW / HIDE
		if(ImGui::ImageButton(label_eye.c_str(), my_tex_id, size, uv_eye_0, uv_eye_1, bg_col, tint_col)){
			node->visible = !node->visible;
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("Show/Hide this layer.");
		}

		// LOCK / UNLOCK
		ImGui::SameLine();
		if(ImGui::ImageButton(label_lock.c_str(), my_tex_id, size, uv_lock_0, uv_lock_1, bg_col, tint_col)){
			node->locked = !node->locked;
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("Lock to prevent interacting with this layer.");
		}

		ImGui::PopStyleColor(1);

	}
}

void make_layeritem_points(SNPoints* node, int i){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;
	auto& scene = editor->scene;

	shared_ptr<Points> points = node->points;
	PointData& pd = node->manager.data;

	string name = node->name;

	ImGui::TableNextRow(ImGuiTableRowFlags_None, row_min_height);

	{ // COLUMN SELECTABLE
		ImGui::TableNextColumn();
		
		ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap;
		string id = format("{}##{}", name, uint64_t(points.get()));
		if (ImGui::Selectable(id.c_str(), node->selected, selectable_flags, ImVec2(0, row_min_height))) {
			scene.deselectAllNodes();
			node->selected = true;

			editor->updateBoundingBox(pd);
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("- Ctrl+Click to select this layer's points.\n- Double Click to focus viewpoint on node.");
		}

		bool isHovered = ImGui::IsItemHovered();
		bool isDoubleClicked = ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0);


		if (isHovered) {
			state.hoveredObjectIndex = i;
			editor->updateBoundingBox(pd);
		}
		if (isDoubleClicked) {
			state.doubleClickedObjectIndex = i;
			editor->prog_gaussians_editing->launch("kernel_focus_onto_aabb", {&editor->launchArgs, &pd.min, &pd.max}, 1);
		}
		
		if (node->selected) {
			ImGui::SetItemDefaultFocus();
		}
	}

	{ // COLUMN MEMORY
		ImGui::TableNextColumn();

		uint64_t memory = node->getGpuMemoryUsage();
		string msg = formatShortMemMsg(memory);
		ImGui::Text(msg.c_str());
	}

	{ // COLUMN ACTIONS
		ImGui::TableNextColumn();

		// ImGui::Text("DEF");

		ImTextureID my_tex_id = (void*)(intptr_t)Runtime::gltex_symbols;
		ImVec2 size = ImVec2(16.0f, 16.0f);
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		float iconSize = 16.0f;

		ImVec2 uv_eye_open_0   = ImVec2(2.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_eye_open_1   = ImVec2(3.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);
		ImVec2 uv_eye_closed_0 = ImVec2(3.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_eye_closed_1 = ImVec2(4.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);

		ImVec2 uv_lock_open_0   = ImVec2(0.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_lock_open_1   = ImVec2(1.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);
		ImVec2 uv_lock_closed_0 = ImVec2(1.0f * iconSize / Runtime::gltex_symbols_width, 0.0f * iconSize);
		ImVec2 uv_lock_closed_1 = ImVec2(2.0f * iconSize / Runtime::gltex_symbols_width, 1.0f * iconSize / Runtime::gltex_symbols_height);

		ImVec2 uv_eye_0 = uv_eye_open_0;
		ImVec2 uv_eye_1 = uv_eye_open_1;
		if(!pd.visible){
			uv_eye_0 = uv_eye_closed_0;
			uv_eye_1 = uv_eye_closed_1;
		}

		ImVec2 uv_lock_0 = uv_lock_open_0;
		ImVec2 uv_lock_1 = uv_lock_open_1;
		if(pd.locked){
			uv_lock_0 = uv_lock_closed_0;
			uv_lock_1 = uv_lock_closed_1;
		}

		string label_eye = format("toggle_visibility##{}", i);
		string label_lock = format("toggle_lock##{}", i);

		// SHOW / HIDE
		if(ImGui::ImageButton(label_eye.c_str(), my_tex_id, size, uv_eye_0, uv_eye_1, bg_col, tint_col)){
			// pd.visible = !pd.visible;
			node->visible = !node->visible;
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("Show/Hide this layer.");
		}

		// LOCK / UNLOCK
		ImGui::SameLine();
		if(ImGui::ImageButton(label_lock.c_str(), my_tex_id, size, uv_lock_0, uv_lock_1, bg_col, tint_col)){
			pd.locked = !pd.locked;
		}
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			ImGui::SetTooltip("Lock to prevent interacting with this layer.");
		}

	}
}

void makeTable(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;
	auto& scene = editor->scene;
	
	static ImGuiTableFlags flags =
		ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
		| ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_NoBordersInBody
		// | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY
		| ImGuiTableFlags_SizingFixedFit 
		// | ImGuiTableFlags_SizingStretchProp
		// | ImGuiTableFlags_SizingStretchSame
		| ImGuiTableFlags_NoSavedSettings;
	
	

	if(ImGui::BeginTable("Models##listOfModels", 3, flags))
	{
		ImGui::TableSetupColumn("Model",   ImGuiTableColumnFlags_WidthStretch, 0.0f);
		ImGui::TableSetupColumn("Mem",     ImGuiTableColumnFlags_WidthFixed, 50.0f);
		ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 50.0f);
		ImGui::TableSetupScrollFreeze(freeze_cols, freeze_rows);

		ImGui::TableHeadersRow();

		int i = 0; 
		vector<shared_ptr<SceneNode>> layers = editor->getLayers();
		for(shared_ptr<SceneNode> node : layers){
			if(node->hidden) return;

			onTypeMatch<SNSplats>(node, [&](shared_ptr<SNSplats> node) {
				make_layeritem_splats(node, i);
				i++;
			});

			onTypeMatch<SNTriangles>(node, [&](shared_ptr<SNTriangles> node) {
				make_layeritem_triangles(node, i);
				i++;
			});

			//if(dynamic_cast<SNSplats*>(node) != nullptr){
			//	make_layeritem_splats((SNSplats*)node, i);
			//	i++;
			//}
			//else if(dynamic_cast<SNPoints*>(node) != nullptr){
			//	make_layeritem_points((SNPoints*)node, i);
			//	i++;
			//}else if(dynamic_cast<SNTriangles*>(node) != nullptr){
			//	make_layeritem_triangles((SNTriangles*)node, i);
			//	i++;
			//}
		}

		ImGui::EndTable();
	}

	

	if(ImGui::Button("Add Splat Layer")){
		scene.deselectAllNodes();

		string name = "New Splat Layer";
		shared_ptr<Splats> splats = make_shared<Splats>();
		shared_ptr<SNSplats> node = make_shared<SNSplats>(name, splats);
		scene.world->children.push_back(node);

		node->selected = true;
	}

	ImGui::SameLine();
	ImUtils::alignRight("Delete Layer");
	if(ImGui::Button("Delete Layer")){
		shared_ptr<SceneNode> node = editor->getSelectedNode();
		editor->deleteNode_undoable(node);
		// scene.erase([](SceneNode* node){
		// 	return node->selected;
		// });
	}

	// SceneNode* selected = scene.find([](SceneNode* node){return node->selected;});

}

void SplatEditor::makeLayers(){
	// LAYERS
	if (ImGui::CollapsingHeader("Layers", ImGuiTreeNodeFlags_DefaultOpen)) {

		// static int item_current_idx = 0;

		ImGui::Separator();

		makeTable();
	}
}