
#pragma once

#include "imgui.h"

namespace ImUtils{

	void alignRight(string text) {
		float rightBorder = ImGui::GetCursorPosX() + ImGui::GetColumnWidth();
		float width = ImGui::CalcTextSize(text.c_str()).x;
		ImGui::SetCursorPosX(rightBorder - width);
	}


};