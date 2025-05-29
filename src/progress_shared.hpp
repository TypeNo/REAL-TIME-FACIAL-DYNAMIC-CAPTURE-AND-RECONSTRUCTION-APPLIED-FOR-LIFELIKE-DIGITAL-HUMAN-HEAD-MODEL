#pragma once

#include <string>

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
extern PROGRESS_API std::string animation_model_path;
extern PROGRESS_API std::string expression_model_path;
extern PROGRESS_API float FPS;



extern "C" {
    PROGRESS_API void update_progress(int current, int total);
    PROGRESS_API int get_current_progress();
    PROGRESS_API int get_total_progress();
    PROGRESS_API void update_model_path(std::string path);
    PROGRESS_API const char* get_model_path();  // C-compatible
    PROGRESS_API float get_FPS();  // C-compatible
    PROGRESS_API void update_FPS(float framerate);  // C-compatible
    PROGRESS_API void update_expression_path(std::string path);
    PROGRESS_API const char* get_expression_path();




}