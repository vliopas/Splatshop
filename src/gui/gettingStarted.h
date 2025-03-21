
void SplatEditor::makeGettingStarted(){
	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;

	if(settings.showGettingStarted){

		ImVec2 kernelWindowSize = {720, 660};
		ImGui::SetNextWindowPos({10, 100}, ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(kernelWindowSize, ImGuiCond_FirstUseEver);

		if(ImGui::Begin("Getting Started")){

			
			ImGui::Text("What to do");
			ImGui::Text("- Drag&Drop ply files to load. (Windows)");
			ImGui::Text("- Adapt main.cpp initScene() to load files (Linux)");

			// ImGui::Text("");

			ImGui::Text("Limitations");
			ImGui::Text("- Academic Prototype -> expect Bugs");
			ImGui::Text("- No Spherical Harmonics.");
			ImGui::Text("- No Drag&Drop in Linux, yet.");

			ImGui::Text("");

			ImGui::Text("Alternatives");
			ImGui::Text("- Check out Supersplat!");

		}

		ImGui::End();
	
	}

}