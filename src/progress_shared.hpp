#pragma once

#ifdef _WIN32
  #ifdef PROGRESS_SHARED_EXPORTS
    #define PROGRESS_API __declspec(dllexport)
  #else
    #define PROGRESS_API __declspec(dllimport)
  #endif
#else
  #define PROGRESS_API
#endif

extern PROGRESS_API int current_progress;
extern PROGRESS_API int total_progress;

extern "C" {
    PROGRESS_API void update_progress(int current, int total);
    PROGRESS_API int get_current_progress();
    PROGRESS_API int get_total_progress();
}