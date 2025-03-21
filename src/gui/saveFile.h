#pragma once

inline static ImVec2 windowSize = {1000, 130};

void makeSavefileProject(){

	auto editor = SplatEditor::instance;
	Scene& scene = editor->scene;
	auto& settings = editor->settings;

	static string defaultPath = fs::weakly_canonical(".").string() + "/splatmodels";
	constexpr int pathBufferSize = 1024;
	static char pathBuffer[pathBufferSize] = {0};
	static bool initialized = false;
	if(!initialized && defaultPath.size() < pathBufferSize){
		memset(pathBuffer, 0, pathBufferSize);
		strcpy(pathBuffer, defaultPath.c_str());
		pathBuffer[pathBufferSize - 1] = 0;
		initialized = true;
	}

	ImGui::Text("Path (Directory): ");
	ImGui::SameLine();
	ImGui::SetNextItemWidth(1000.0f);
	ImGui::InputText("", pathBuffer, 1024);

	ImGui::Text("Specify the folder in which the scene will be stored. For each node, a separate ply file will be created.");

	ImVec2 textSize = ImGui::CalcTextSize("Cancel");

	ImGui::SetCursorPosY(windowSize.y - textSize.y - 20);
	if(ImGui::Button("Save")){
		string path = string(pathBuffer);
		SplatsyFilesWriter::write(path, scene, *Runtime::controls);
		settings.showFileSaveDialog = false;
	}
	ImGui::SameLine();

	ImGui::SetCursorPosX(windowSize.x - textSize.x - 20);
	ImGui::SetCursorPosY(windowSize.y - textSize.y - 20);
	if(ImGui::Button("Cancel")){
		settings.showFileSaveDialog = false;
	}
}


void makeSavefileSinglePly(){
	auto editor = SplatEditor::instance;
	Scene& scene = editor->scene;
	auto& settings = editor->settings;

	static string defaultPath = fs::weakly_canonical(".").string() + "/splatmodels";
	constexpr int pathBufferSize = 1024;
	static char pathBuffer[pathBufferSize] = {0};
	static bool initialized = false;
	if(!initialized && defaultPath.size() < pathBufferSize){
		memset(pathBuffer, 0, pathBufferSize);
		strcpy(pathBuffer, defaultPath.c_str());
		pathBuffer[pathBufferSize - 1] = 0;
		initialized = true;
	}

	ImGui::Text("Path: ");
	ImGui::SameLine();
	ImGui::SetNextItemWidth(1000.0f);
	ImGui::InputText("", pathBuffer, 1024);

	ImGui::Text("Specify the file in which the scene will be stored. All nodes will be merged into a single ply.");

	ImVec2 textSize = ImGui::CalcTextSize("Cancel");

	ImGui::SetCursorPosY(windowSize.y - textSize.y - 20);
	if(ImGui::Button("Save")){
		string path = string(pathBuffer);
		SplatsyFilesWriter::write(path, scene, *Runtime::controls);
		settings.showFileSaveDialog = false;
	}
	ImGui::SameLine();

	ImGui::SetCursorPosX(windowSize.x - textSize.x - 20);
	ImGui::SetCursorPosY(windowSize.y - textSize.y - 20);
	if(ImGui::Button("Cancel")){
		settings.showFileSaveDialog = false;
	}
}

void SplatEditor::makeSaveFileGUI(){

	if(!settings.showFileSaveDialog) return;

	ImGui::SetNextWindowPos({
		GLRenderer::width / 2 - windowSize.x / 2,
		(GLRenderer::height - windowSize.y) / 2, }, 
		ImGuiCond_Once);
	ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

	temporarilyDisableShortcuts();

	static bool open = false;
	if(ImGui::Begin("Save File", &open)){

		windowSize = ImGui::GetWindowSize();

		// const int FILETYPE_PROJECT = 0;
		// const int FILETYPE_SINGLE_PLY = 1;
		// static int filetype = FILETYPE_PROJECT;
		// ImGui::RadioButton("Separate PLY's", &filetype, FILETYPE_PROJECT); ImGui::SameLine();
		// ImGui::RadioButton("Single PLY", &filetype, FILETYPE_SINGLE_PLY);

		// if(filetype == FILETYPE_PROJECT){
			makeSavefileProject();
		// }else if(filetype == FILETYPE_SINGLE_PLY){
		// 	makeSavefileSinglePly();
		// }

		

		
	}

	ImGui::End();

}