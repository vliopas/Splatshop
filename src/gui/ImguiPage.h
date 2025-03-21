#pragma once

struct ImguiPage{

	ImGuiContext* context = nullptr;
	shared_ptr<Framebuffer> framebuffer = nullptr;
	int width = 1024;
	int height = 1024;

	ImguiPage(){
		context = ImGui::CreateContext(ImGui::GetIO().Fonts);
		// ImGui::SetCurrentContext(context);
		// io = &ImGui::GetIO();
		framebuffer = Framebuffer::create();;
	}
};
