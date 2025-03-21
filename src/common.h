#pragma once

constexpr int64_t MAX_SPLATS = 100'000'000;
constexpr int64_t MAX_LINES = 10'000'000;

constexpr int VIEWMODE_DESKTOP = 0;
constexpr int VIEWMODE_DESKTOP_VR = 1;
constexpr int VIEWMODE_IMMERSIVE_VR = 2;

static inline int viewmode = VIEWMODE_DESKTOP;


