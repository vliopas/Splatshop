#include "actions/PlaceAssetAction.h"

void SplatEditor::makeAssetGUI(){

	auto editor = SplatEditor::instance;

	if (ImGui::CollapsingHeader("Assets", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("Asset Library");
		
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

		//float iconSize = 256.0f;
		//ImVec2 size = ImVec2(iconSize, iconSize);

		ImVec2 uv0 = ImVec2(0.0f, 1.0f);
		ImVec2 uv1 = ImVec2(1.0f, 0.0f);

		ImVec2 vMin = ImGui::GetWindowContentRegionMin();
		ImVec2 vMax = ImGui::GetWindowContentRegionMax();
		// ImVec2 thumbSize = ImVec2(128.0f, 128.0f);
		ImVec2 thumbSize = {
			(vMax.x - vMin.x) / 4.0f - 10.0f,
			(vMax.x - vMin.x) / 4.0f - 10.0f,
		};

		int deletingAssetIndex = -1;
		int i = 0;
		for(auto node : AssetLibrary::assets){

			ImTextureID handle = (void*)(intptr_t)Runtime::gltex_symbols;

			if(node->thumbnail){
				handle = (void*)(intptr_t)node->thumbnail->handle;
			}

			string id = format("{}##{}", node->name, i);
			if(i > 0 && (i % 4) != 0){
				ImGui::SameLine();
			}
			if(ImGui::ImageButton(id.c_str(), handle, thumbSize, uv0, uv1, bg_col, tint_col)){
				shared_ptr<PlaceAssetAction> action = make_shared<PlaceAssetAction>();
				action->placingItem = node;
				setAction(action);
			}
			if (ImGui::BeginPopupContextItem()){

				ImGui::MenuItem("(Context Menu)", NULL, false, false);

				bool deleteAsset = false;

				ImGui::MenuItem("Delete Asset", NULL, &deleteAsset);

				if(deleteAsset){
					deletingAssetIndex = i;
				}
				
				ImGui::EndPopup();
			}else {
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip(node->name.c_str());

					// updateBoundingBox(node->dmng.data);
					// createOrUpdateThumbnail(node.get());
				}
			}
			
			i++;
		}

		if(deletingAssetIndex >= 0){
			AssetLibrary::assets.erase(AssetLibrary::assets.begin() + deletingAssetIndex);
		}



		// if(ImGui::ImageButton("Asset 1", my_tex_id, size, uv0, uv1, bg_col, tint_col)){
			
		// }
		// ImGui::SameLine();
		// if(ImGui::ImageButton("Asset 2", my_tex_id, size, uv0, uv1, bg_col, tint_col)){
			
		// }

		// ImGui::SameLine();
		// if(ImGui::ImageButton("Asset 3", my_tex_id, size, uv0, uv1, bg_col, tint_col)){
			
		// }

		// if(ImGui::ImageButton("Asset 4", my_tex_id, size, uv0, uv1, bg_col, tint_col)){
			
		// }
		// ImGui::SameLine();
		// if(ImGui::ImageButton("Asset 5", my_tex_id, size, uv0, uv1, bg_col, tint_col)){
			
		// }
	}


}