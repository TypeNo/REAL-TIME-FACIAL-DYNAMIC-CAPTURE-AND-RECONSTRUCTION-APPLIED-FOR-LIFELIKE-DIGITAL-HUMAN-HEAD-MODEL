#define PROGRESS_SHARED_EXPORTS
#include "progress_shared.hpp"
#include <iostream>
#include <windows.h>


int current_progress = 0;
int total_progress = 0;
std::string animation_model_path = "";

extern "C" __declspec(dllexport) void update_progress(int current, int total) {
    current_progress = current;
    total_progress = total;
    printf("[C++] update_progress called: %d / %d\n", current, total);
}

extern "C" __declspec(dllexport) int get_current_progress() {
    return current_progress;
}

extern "C" __declspec(dllexport) int get_total_progress() {
    return total_progress;
}

// Sets the model path manually — you can change this logic
extern "C" __declspec(dllexport) void update_model_path(std::string path) {
    animation_model_path = path;
    printf("[C++] update_model_path called: %s\n", animation_model_path.c_str());
}

// Returns a C-style string for external (e.g., Python) use
extern "C" __declspec(dllexport) const char* get_model_path() {
    return animation_model_path.c_str();
}
