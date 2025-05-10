#include <iostream>
#include <filesystem>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "Dwmapi.lib")
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "ImGuiFileDialog.h"
#include <cstdlib>
#include <string>
#include <pybind11/embed.h> // Everything needed for embedding
#include <Python.h>
namespace py = pybind11;
#include <thread>
#include <atomic>
using namespace std::chrono_literals;
#include <chrono>
//#include "progress.hpp"
#include <sstream> // for std::stringstream
#include <cmath> // For sin() and M_PI
#include "progress_shared.hpp"
#include <windows.h>
#include <algorithm>  // for std::clamp


typedef void (__cdecl *update_progress_t)(int, int);
typedef int (__cdecl *get_progress_t)();
typedef int (*GetProgressFunc)();

// Global or class-level flag and thread
std::thread pythonThread;
std::atomic<bool> shouldRunPython(false);
std::string pendingPythonPath;
bool pythonThreadRunning = false;

// Global variables for progress
int cur = get_current_progress();
int previous_cur = cur;
int total = get_total_progress();
std::string progressLabel = "Waiting...";

//Default Input Dir
std::string GetDesktopPath() {
    #ifdef _WIN32
        const char* userProfile = std::getenv("USERPROFILE");
        if (userProfile)
            return std::string(userProfile) + "\\Desktop";
    #elif __APPLE__ || __linux__
        const char* home = std::getenv("HOME");
        if (home)
            return std::string(home) + "/Desktop";
    #endif
        return "."; // fallback to current directory
    }

//Application Dark Theme Setup
void EnableDarkTitleBar(HWND hwnd) {
    BOOL value = TRUE;
    DwmSetWindowAttribute(hwnd, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/, &value, sizeof(value));
}

//GLFW Error Debug
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

//Functions
//GUI of Facial Tracking
void RenderFacialTrackingTab();

//Loading Images as Texture
GLuint LoadTextureFromFile(const char* filename);

//Running Face Reconstruction Task in Python
void runPythonConstruct(const std::string& selectedFilePath);

int main()
{
    //Loading dll
    HMODULE dll = LoadLibraryW(L"progress_shared.dll");
    if (!dll) return 1;
    //Loading dll function
    auto get_current = (GetProgressFunc)GetProcAddress(dll, "get_current_progress");
    auto get_total = (GetProgressFunc)GetProcAddress(dll, "get_total_progress");

    //Python interpretator Configuration
    std::string newPath = 
    "E:/anaconda3/envs/pytorch3d;" 
    "E:/anaconda3/envs/pytorch3d/Library/bin;" + 
    std::string(std::getenv("PATH"));
    _putenv_s("PATH", newPath.c_str());
    _putenv_s("PYTHONHOME", "E:/anaconda3/envs/pytorch3d");
    _putenv_s("PYTHONPATH", "E:/Project/DECA3/DECA/src");

    try {
        py::scoped_interpreter guard{}; // Initialize interpreter

        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return -1;

        // Windowed mode (NOT fullscreen), but large size
        int width = 1920;
        int height = 1080;

        // Decide GL+GLSL versions
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        // Get primary monitor and its video mode
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        // Set window hints to match the monitor's settings (optional but safer)
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        // Optional: set some window hints
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // Allow resizing
        glfwWindowHint(GLFW_DECORATED, GLFW_TRUE); // Show minimize/maximize/close buttons
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API); // Use OpenGL
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE); // start maximized
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Window starts hidden

        // Create windowed mode window
        GLFWwindow* window = glfwCreateWindow(width, height, "FaceRec Studio", NULL, NULL);
        if (window == NULL) {
            glfwTerminate();
            return -1;
        }

        HWND hwnd = glfwGetWin32Window(window);
        EnableDarkTitleBar(hwnd);

        // Force Windows to redraw the title bar immediately
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);

        // Now show the window
        glfwShowWindow(window);
        
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        // Set dark background color
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        // Initialize OpenGL loader + ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Main loop
        while (!glfwWindowShouldClose(window))
        {
             //Polling the progress
            if (shouldRunPython){
               
                cur = get_current_progress();
                total = get_total_progress();
            }

            std::cout << "C++ Progress: " << cur << " / " << total << std::endl;

            py::gil_scoped_release release; // This releases the GIL for this scope   
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Your main UI
            {
                // Before ImGui::Begin()
                int display_w, display_h;
                glfwGetFramebufferSize(window, &display_w, &display_h); // get current window size

                ImGui::SetNextWindowPos(ImVec2(0, 0)); // always top-left
                ImGui::SetNextWindowSize(ImVec2((float)display_w, (float)display_h)); // always fill the window
                
                ImGui::Begin("Facial Reconstruction App", nullptr,
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoTitleBar
                );

                if (ImGui::BeginTabBar("MainTabs"))
                {
                    if (ImGui::BeginTabItem("Facial Capture"))
                    {
                        RenderFacialTrackingTab();
                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Facial Reconstruction"))
                    {
                        ImGui::Text("Facial Capture content goes here...");
                        ImGui::EndTabItem();;
                    }

                    ImGui::EndTabBar();
                }

                ImGui::End();

                // After ImGui::Render() or inside your main loop:
                static std::thread pythonThread;
                if (shouldRunPython && !pythonThreadRunning) {
                    pythonThreadRunning = true;
                    std::string capturedPath = pendingPythonPath;

                    //Running Python Backend in Background
                    pythonThread = std::thread([capturedPath]() {
                        try {
                            py::gil_scoped_acquire gil;
                            std::cout << "[Thread] Started lambda.\n";
                
                            if (!Py_IsInitialized()) {
                                std::cerr << "Python is not initialized!\n";
                            }
                            std::cout << "[Thread] Py_IsInitialized: " << Py_IsInitialized() << std::endl;
                            std::cout << "[Thread] PyGILState_Check: " << PyGILState_Check() << std::endl;
                
                            {
                                //py::gil_scoped_acquire acquire; // Always acquire GIL before running Python
                                std::cout << "[Thread] GIL acquired.\n";
                
                                std::cout << "Running in thread with file: " << capturedPath << std::endl;
                                runPythonConstruct(capturedPath);
                                std::cout << "Successfully constructed the face model.\n";
                                std::cout << "All done!" << std::endl;
                            }
                
                        } catch (const std::exception& e) {
                            std::cerr << "C++ exception in thread: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "Unknown exception in Python thread!\n";
                        }
                        
                        pythonThreadRunning = false;
                        shouldRunPython = false;
                    });
                
                    pythonThread.detach();
                    
                }                
            }

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            //glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
        FreeLibrary(dll);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Python initialization error: " << e.what() << std::endl;
        return -1;
    }
}

void RenderFacialTrackingTab()
{
     //py::gil_scoped_acquire acquire;

    //Loading Screen------------------------------------------------------------------------
    // Constants
    const int numSegments = 20;
    const float barWidth = 300.0f;
    const float barHeight = 20.0f;
    const float spacing = 2.0f;
    const float segmentWidth = (barWidth - spacing * (numSegments - 1)) / numSegments;
    const float speed = 2.0f;  // how fast the animation loops

    // Get draw position
    ImVec2 barSize(barWidth, barHeight);
    float availWidth = ImGui::GetContentRegionAvail().x;
    float offsetX = (availWidth - barSize.x) * 0.5f;
    if (offsetX > 0)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

    // Reserve space for the bar
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(barSize);

    // Animate
    float t = ImGui::GetTime();
    float offset = fmod(t * speed, numSegments);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    //---------------------------------------------------------------------------------------

    //Input Selection------------------------------------------------------------------------
    static std::string selectedFilePath;
    static std::string PreviewFilePath;
    static std::string LandmarkFilePath;
   
    // Static textures: loaded only once
    static GLuint previewTexture = 0;
    static GLuint trackingTexture = 0;
    static GLuint defaultTexture = 0;

    // Log current texture IDs
    std::cout << "[DEBUG] Texture States:" << std::endl;
    std::cout << "  previewTexture ID:  " << previewTexture << std::endl;
    std::cout << "  trackingTexture ID: " << trackingTexture << std::endl;
    std::cout << "  defaultTexture ID:  " << defaultTexture << std::endl;

    static std::string lastPreviewPath = "";
    static std::string lastTrackingPath = "";

    //---------------------------------------------------------------------------------------

    // Load default "no image" texture once
    if (defaultTexture == 0) {
        std::cout << "[INFO] Creating default black texture." << std::endl;

        const int width = 256;
        const int height = 256;
        unsigned char blackPixels[width * height * 3];
        std::fill_n(blackPixels, width * height * 3, 0);

        glGenTextures(1, &defaultTexture);
        glBindTexture(GL_TEXTURE_2D, defaultTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, blackPixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        std::cout << "[INFO] Default texture ID: " << defaultTexture << std::endl;
    }

    // Setting the Path of outputs and input
    if (!selectedFilePath.empty()) {
        std::cout << "[INFO] Selected file path: " << selectedFilePath << std::endl;

        std::filesystem::path selectedPath(selectedFilePath);
        std::string filename = selectedPath.stem().string();

        std::string baseOutputPath = "E:/Project/DECA3/DECA/output/";

        PreviewFilePath = selectedFilePath; // Direct preview of selected image
        //PreviewFilePath = baseOutputPath + filename + "/inputs/" + filename + "_inputs.jpg";

        std::cout << "[INFO] Preview path: " << PreviewFilePath << std::endl;
        std::cout << "[INFO] Landmark path: " << LandmarkFilePath << std::endl;

        // Reload preview texture only if path changes
        if (PreviewFilePath != lastPreviewPath) {
            if (previewTexture != 0) {
                std::cout << "[INFO] Deleting old preview texture ID: " << previewTexture << std::endl;
                glDeleteTextures(1, &previewTexture);
                std::cout << "[DEBUG] - AFTER DELETE ORIGINAL defaultTexture - ID: " << defaultTexture << ", previewTexture ID: " << previewTexture << std::endl;
                previewTexture = 0;
            }
            previewTexture = LoadTextureFromFile(PreviewFilePath.c_str());
            std::cout << "[INFO] Loaded preview texture: " << PreviewFilePath << " -> ID: " << previewTexture << std::endl;
            std::cout << "[DEBUG] defaultTexture ID: " << defaultTexture << ", previewTexture ID: " << previewTexture << std::endl;
            lastPreviewPath = PreviewFilePath;
        }

        // Reload tracking texture only if path changes
        if (LandmarkFilePath != lastTrackingPath) {
            LandmarkFilePath = baseOutputPath + filename + "/landmarks2d/" + filename + "_landmarks2d.jpg";

            if (trackingTexture != 0) {
                std::cout << "[INFO] Deleting old tracking texture ID: " << trackingTexture << std::endl;
                glDeleteTextures(1, &trackingTexture);
                trackingTexture = 0;
            }
            trackingTexture = LoadTextureFromFile(LandmarkFilePath.c_str());
            if (trackingTexture != 0) {
                std::cout << "[INFO] Loaded tracking texture: " << LandmarkFilePath << " -> ID: " << trackingTexture << std::endl;
            }
            lastTrackingPath = LandmarkFilePath;
        }
    } else {
        std::cout << "[INFO] No file selected. Using default black texture." << std::endl;
    }

    // Split vertically: Left big panel + Right control panel
    ImGui::BeginChild("LeftSection", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0), false);
    {
        // Top Part: Image Preview and Facial Tracking Panels
        ImGui::BeginChild("LeftTop", ImVec2(0, ImGui::GetContentRegionAvail().y - 60), true);
        {
            // Two children side-by-side without extra nesting
            ImGui::BeginChild("LeftTab", ImVec2(ImGui::GetContentRegionAvail().x * 0.5f, 0), true);
            {
                if (ImGui::BeginTabBar("LeftTabBar"))
                {
                    if (ImGui::BeginTabItem("Image Preview"))
                    {
                        ImGui::Text("Image Preview Content");
                        ImVec2 availableSize = ImGui::GetContentRegionAvail();
                        if(!selectedFilePath.empty()){
                            ImGui::Image((ImTextureID)(intptr_t)previewTexture, availableSize);
                        }
                        else{
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, availableSize);
                        }
                        
                        //ImGui::Image((ImTextureID)123, ImVec2(-1, -1)); // Placeholder image
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("RightTab", ImVec2(0, 0), true);
            {
                if (ImGui::BeginTabBar("RightTabBar"))
                {
                    if (ImGui::BeginTabItem("Facial Tracking"))
                    {
                        if (shouldRunPython) {
                            previous_cur = cur;
                            cur = get_current_progress();
                            total = get_total_progress();
                    
                            std::cout << "C++ Progress: " << cur << " / " << total << std::endl;
                    
                            // Label
                            std::string progressLabel = "Refreshing facial tracking data...";
                            ImGui::Text("%s", progressLabel.c_str());
                    
                            // Dimensions
                            ImVec2 barSize(300.0f, 20.0f); // bar width & height
                            float spacing = 3.0f;
                            int numSegments = 20;
                            float segmentWidth = (barSize.x - spacing * (numSegments - 1)) / numSegments;
                            float barHeight = barSize.y;
                    
                            float availWidth = ImGui::GetContentRegionAvail().x;
                            float offsetX = (availWidth - barSize.x) * 0.5f;
                            if (offsetX > 0)
                                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
                    
                            if (total > 0) {
                                // Actual progress bar
                                float progress = static_cast<float>(cur) / static_cast<float>(total);
                                ImGui::ProgressBar(progress, barSize);
                            } else {
                                // Reserve space and get drawing area
                                ImVec2 pos = ImGui::GetCursorScreenPos();
                                ImGui::Dummy(barSize);  // Reserves the area for custom drawing
                    
                                ImDrawList* drawList = ImGui::GetWindowDrawList();
                    
                                // Animation phase
                                float t = ImGui::GetTime();
                                float speed = 2.0f;
                                float offset = fmod(t * speed, static_cast<float>(numSegments));
                    
                                // Draw segments
                                for (int i = 0; i < numSegments; ++i) {
                                    float normIndex = (i + offset);
                                    float alpha = 1.0f - fabsf(fmodf(normIndex, numSegments) - numSegments / 2.0f) / (numSegments / 2.0f);
                                    alpha = std::clamp(alpha, 0.3f, 1.0f);
                    
                                    ImVec2 segMin = ImVec2(pos.x + i * (segmentWidth + spacing), pos.y);
                                    ImVec2 segMax = ImVec2(segMin.x + segmentWidth, segMin.y + barHeight);
                    
                                    drawList->AddRectFilled(segMin, segMax, ImGui::GetColorU32(ImVec4(0.2f, 0.7f, 1.0f, alpha)), 3.0f);
                                }
                            }
                        }
                    
                        // Placeholder image centered
                        ImVec2 imageSize(200, 200);
                        float availWidth = ImGui::GetContentRegionAvail().x;
                        float offsetX = (availWidth - imageSize.x) * 0.5f;
                        if (offsetX > 0)
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
                    
                        if(!selectedFilePath.empty()){
                            ImGui::Image((ImTextureID)(intptr_t)trackingTexture, imageSize);
                        }
                        else{
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, imageSize);
                        }
                    
                        ImGui::EndTabItem();
                    }
                    
                    
                    
                    
                    ImGui::EndTabBar();
                }
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();

        // Bottom Part: Timeline
        ImGui::BeginChild("LeftBottom", ImVec2(0, 0), true);
        {
            static float currentFrame = 0.0f;
            static int endFrame = 464;
            ImGui::SliderFloat("Timeline", &currentFrame, 0.0f, (float)endFrame, "%.0f");
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Right Control Panel
    ImGui::BeginChild("RightSection", ImVec2(0, 0), true);
    {
        ImGui::Text("[Capture Source]");
    
        // Begin horizontal group: Button + Selected Path
        ImGui::BeginGroup();
        if (ImGui::Button("Choose File")) {
            IGFD::FileDialogConfig config;
            config.path = GetDesktopPath(); // Default to Desktop
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_Modal;
    
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCapture",
                "Select Capture Source",
                "Images and Videos{.png,.jpg,.jpeg,.bmp,.gif,.mp4,.avi,.mov,.mkv}",
                config
            );
        }
        ImGui::SameLine();
    
        // Limit max display width for the selected path
        float availableWidth = ImGui::GetContentRegionAvail().x;
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + availableWidth);
        ImGui::TextWrapped("%s", selectedFilePath.empty() ? "<No file selected>" : selectedFilePath.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    
        if (ImGuiFileDialog::Instance()->Display("ChooseCapture", ImGuiWindowFlags_NoCollapse, ImVec2(700, 400))) {
            ImGui::SetNextWindowFocus();
            if (ImGuiFileDialog::Instance()->IsOk()) {
                selectedFilePath = ImGuiFileDialog::Instance()->GetFilePathName();
                shouldRunPython = true;
                pendingPythonPath = selectedFilePath;
            }
            ImGuiFileDialog::Instance()->Close();
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    
        // Frame range input
        static int startFrame = 0, endFrame = 464;
        ImGui::InputInt("Start Frame to Process", &startFrame);
        ImGui::InputInt("End Frame to Process", &endFrame);
    }
    
    ImGui::EndChild();
}

void runPythonConstruct(const std::string& selectedFilePath) {
    try {
        //py::gil_scoped_acquire acquire;
        std::cout << "Working directory: " << std::filesystem::current_path() << std::endl;
        py::module sys = py::module::import("sys");

        // Dynamically compute the absolute path to /src
        std::filesystem::path scriptPath = std::filesystem::current_path() / "src";
        std::string scriptDir = scriptPath.string();

        std::cout << "Adding to sys.path: " << scriptDir << std::endl;
        sys.attr("path").attr("insert")(0, scriptDir);

        // Print the current sys.path for diagnostics
        std::cout << "Python sys.path:" << std::endl;
        for (auto item : sys.attr("path")) {
            std::cout << "  - " << std::string(py::str(item)) << std::endl;
        }

        // Attempt to import and run face_reconstruct
        std::cout << "Attempting to import face_reconstruct..." << std::endl;
        py::module face_reconstruct = py::module::import("face_reconstruct");
        std::cout << "Successfully imported face_reconstruct.py!" << std::endl;
        

        py::list args;
        args.append("src/face_reconstruct.py"); // Emulate `python src/face_reconstruct.py`
        //std::string argument_I = "-i "+ selectedFilePath;
        //args.append("-i C:\\Users\\acer\\Desktop\\IMG_0392_inputs.jpg");
        //args.append(argument_I);
        args.append("-i");
        //args.append("C:\\Users\\acer\\Desktop\\IMG_0392_inputs.jpg");
        args.append(selectedFilePath);
        sys.attr("argv") = args;
        

        // Call main() if defined
        if (py::hasattr(face_reconstruct, "main")) {
            std::cerr << "face_reconstruct.py does define a 'main' function." << std::endl;
            //face_reconstruct.attr("main")(args);
            // Evaluate the script — this runs it as a top-level script
            py::eval_file("src/face_reconstruct.py");
        } else {
            std::cerr << "face_reconstruct.py does not define a 'main' function." << std::endl;
        }

    } catch (py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        
    
        // Force full traceback to stderr
        PyErr_Print();
    }catch (const std::exception& e) {
        std::cerr << "C++ exception: " << e.what() << std::endl;
    }
    
}

//LoadTextureFromFile
GLuint LoadTextureFromFile(const char* filename)
{
    if (!std::filesystem::exists(filename)) {
        std::cout << "[ERROR] File does not exist: " << filename << std::endl;
    }
    
    std::cout << "[DEBUG] Attempting to load texture from: " << filename << std::endl;

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 4); // Force RGBA
    if (data == nullptr)
    {
        std::cout << "[ERROR] Failed to load image: " << filename << std::endl;
        return 0;
    }

    std::cout << "[INFO] Image loaded: " << filename << " (" << width << "x" << height 
              << ", channels requested: 4, original channels: " << channels << ")" << std::endl;

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    if (textureID == 0)
    {
        std::cout << "[ERROR] glGenTextures failed!" << std::endl;
        stbi_image_free(data);
        return 0;
    }

    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "[ERROR] OpenGL error after glTexImage2D: " << err << std::endl;
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0); // Unbind for safety
    stbi_image_free(data);

    std::cout << "[DEBUG] Texture loaded successfully with ID: " << textureID << std::endl;

    return textureID;
}

