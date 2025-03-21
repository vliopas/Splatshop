


void SplatEditor::makePerf(){
	//PERFORMANCE
	if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)){
		string strFPS = format("FPS: {:5.1f}", GLRenderer::fps);
		ImGui::Text(strFPS.c_str());

		static float history = 2.0f;
		static ScrollingBuffer sFrames;
		static ScrollingBuffer s60fps;
		static ScrollingBuffer s120fps;
		float t = now();

		sFrames.AddPoint(t, 1000.0f * GLRenderer::timeSinceLastFrame);

		// sFrames.AddPoint(t, 1000.0f * timeSinceLastFrame);
		s60fps.AddPoint(t, 1000.0f / 60.0f);
		s120fps.AddPoint(t, 1000.0f / 120.0f);
		static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
		ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
		ImPlot::SetNextPlotLimitsY(0, 30, ImGuiCond_Always);

		if (ImPlot::BeginPlot("Timings", nullptr, nullptr, ImVec2(-1, 200))){

			auto x = &sFrames.Data[0].x;
			auto y = &sFrames.Data[0].y;
			ImPlot::PlotShaded("frame time(ms)", x, y, sFrames.Data.size(), -Infinity, sFrames.Offset, 2 * sizeof(float));

			ImPlot::PlotLine("16.6ms (60 FPS)", &s60fps.Data[0].x, &s60fps.Data[0].y, s60fps.Data.size(), s60fps.Offset, 2 * sizeof(float));
			ImPlot::PlotLine(" 8.3ms (120 FPS)", &s120fps.Data[0].x, &s120fps.Data[0].y, s120fps.Data.size(), s120fps.Offset, 2 * sizeof(float));

			ImPlot::EndPlot();
		}

		static float maxy = 1.0f;
		ImPlot::SetNextPlotLimitsX(0.0, 10.0, ImGuiCond_Always);
		ImPlot::SetNextPlotLimitsY(0, maxy, ImGuiCond_Always);
		if (ImPlot::BeginPlot("Timings - Profile", "ms", "utilization", ImVec2(-1.0f, 200.0f))){

			static bool initialized = false;
			static vector<float> x(100);
			static vector<float> y(100);

			if(!initialized){
				for(int i = 0; i < x.size(); i++){
					x[i] = float(i) / 10.0f;
					y[i] = i;
				}
				initialized = true;
			}

			for(int i = 0; i < x.size(); i++){
				y[i] = 0;
			}

			vector<StartStop>& timings = Runtime::profileTimings;
			uint64_t tmin = -1;
			uint64_t tmax = 0;
			for(int i = 0; i < timings.size(); i++){
				StartStop timing = timings[i];
				tmin = min(tmin, timing.t_start);
				tmax = max(tmax, timing.t_end);
			}

			for(int i = 0; i < timings.size(); i++){
				StartStop timing = timings[i];

				uint64_t t = (timing.t_start - tmin);

				uint64_t STEP = 1'000'000 / 10; // 0.1 ms
				
				if(timing.t_end > timing.t_start)
				for(uint64_t t = timing.t_start; t < timing.t_end; t += STEP){
					int msSinceStart = (t - tmin) / STEP;

					if(msSinceStart >= 0 && msSinceStart < y.size()){
						y[msSinceStart]++;
					}
				}
			}

			for(int i = 0; i < x.size(); i++){
				maxy = max(maxy, y[i]); 
			}

			ImPlot::PlotShaded("render gaussians - utilization", x.data(), y.data(), x.size(), -Infinity, 0, sizeof(float));
			ImPlot::PlotLine("render gaussians - utilization line", x.data(), y.data(), x.size(), 0, sizeof(float));

			// ImPlot::PlotLine(" 8.3ms (120 FPS)", &s120fps.Data[0].x, &s120fps.Data[0].y, s120fps.Data.size(), s120fps.Offset, 2 * sizeof(float));

			ImPlot::EndPlot();
		}

		{ // AVAILABLE GPU MEMORY

			size_t availableMem = 0;
			size_t totalMem = 0;
			cuMemGetInfo(&availableMem, &totalMem);

			float available = double(availableMem) / 1'000'000'000.0;
			float total = double(totalMem) / 1'000'000'000.0;

			float progress = available / total;

			string msg = format("{:.1f} GB / {:.1f} GB", available, total);
			ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), msg.c_str());
			ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
			ImGui::Text("GPU Memory");
		}
	}
}