#define PROGRESS_SHARED_EXPORTS
#include "progress_shared.hpp"
#include <iostream>
#include <windows.h>


int current_progress = 0;
int total_progress = 0;

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
