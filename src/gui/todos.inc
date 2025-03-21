
void SplatEditor::makeTODOs(){
	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;

	// TODO: should be true if there are pending operations/TODOs
	// bool showTODOs = true;
	// if(showTODOs){

	// 	int numTasks = 0;
	// 	if(editor->state.hasPendingDeletionTask) numTasks++;
	// 	if(editor->state.hasPendingPaintingTask) numTasks++;

	// 	if(numTasks == 0) return;

	// 	ImVec2 kernelWindowSize = {300, 80};
	// 	ImGui::SetNextWindowPos({
	// 		0, 
	// 		(GLRenderer::height - kernelWindowSize.y), });
	// 	ImGui::SetNextWindowSize(kernelWindowSize);

	// 	if(ImGui::Begin("Tasks")){

	// 		if(editor->state.hasPendingDeletionTask)
	// 		{ // DELETED SPLATS
	// 			ImGui::Text("Deleted Splats ");

	// 			ImGui::SameLine();
	// 			if(ImGui::Button("Apply##deletion")){
	// 				editor->applyDeletion();
	// 			}
	// 			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
	// 				ImGui::SetTooltip("Temove the splats that are flagged as \"deleted\" for real.");
	// 			}

	// 			ImGui::SameLine();
	// 			if(ImGui::Button("Revert##deletion")){
	// 				editor->revertDeletion();
	// 			}
	// 			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
	// 				ImGui::SetTooltip("Restore the splats that are flagged as \"deleted\".");
	// 			}
	// 		}

	// 		if(editor->state.hasPendingPaintingTask)
	// 		{ // BRUSH PAINTING
	// 			ImGui::Text("Paint Brushing ");

	// 			ImGui::SameLine();
	// 			if(ImGui::Button("Apply##painting")){
	// 				editor->applyPainting();
	// 			}
	// 			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
	// 				ImGui::SetTooltip("Bake the painted colors into the color buffer.");
	// 			}

	// 			ImGui::SameLine();
	// 			if(ImGui::Button("Revert##painting")){
	// 				editor->revertPainting();
	// 			}
	// 			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
	// 				ImGui::SetTooltip("Revert modifications to colors that happened as a result of paint brushing.");
	// 			}
	// 		}
	// 	}

	// 	ImGui::End();
	
	// }

}