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
#include <filesystem>
namespace fs = std::filesystem;

typedef void (__cdecl *update_progress_t)(int, int);
typedef int (__cdecl *get_progress_t)();
typedef int (*GetProgressFunc)();

// Global or class-level flag and thread
std::thread pythonThread;
std::atomic<bool> shouldRunPython(false);
std::string pendingPythonPath;
bool pythonThreadRunning = false;

// Global variables for progress
static bool hasShownNoFileInfo = false;
int cur = get_current_progress();
int prevCur = -1;
int prevTotal = -1;
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

//IMGUI Texture Rendering------------------------------------------------------------------------
//Frames Playback-----
float playbackFPS = 30.0f;            // Target playback rate
static size_t currentFrameIndex = 0;
static double frameDuration = 1.0 / playbackFPS;
//--------------------
//Input Directory Configuration-----
static std::string selectedFilePath="";
static std::string Last_selectedFilePath="";
//---------------------
//Output Direcotry Configuration-----
static std::string targetDir;
static std::string PreviewFolderPath;
static std::string LandmarkFolderPath;
static std::string lastTrackingPath = "";
static std::string lastPreviewPath = "";
//---------------------
static std::map<std::string, GLuint> trackingTextures;  // Store all tracking textures
static std::vector<GLuint> sortedTrackingKeys;
static std::map<std::string, GLuint> previewTextures;  // Store all preview textures
static std::vector<GLuint> sortedPreviewKeys;

//Configuring Working Directory
void SetAppWorkingDirectory();

//Updating Tracking Images
void UpdateTexture(const std::string& FolderPath, std::map<std::string, GLuint> &Textures,std::vector<GLuint> &SortedKeys,std::string &LastFile);

//Loading Single Images as Texture
GLuint LoadTextureFromFile(const char* filename);

//Delete Tracking Images
void DeleteAllTextures(std::map<std::string, GLuint>& Textures, std::vector<GLuint>& sortedKeys); 
//----------------------------------------------------------------------------------------------

//Running Face Reconstruction Task in Python
void runPythonConstruct(const std::string& selectedFilePath);


int main()
{
    SetAppWorkingDirectory();
    ///Loading dll
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

            if (cur != prevCur || total != prevTotal) {
                std::cout << "C++ Progress: " << cur << " / " << total << std::endl;
                prevCur = cur;
                prevTotal = total;
            }

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
                        
                    update_progress(0,0);
                    //Delete Previous Cache
                    DeleteAllTextures(previewTextures, sortedPreviewKeys);
                    DeleteAllTextures(trackingTextures, sortedTrackingKeys);
                    lastPreviewPath = "";
                    lastTrackingPath = "";
                    try {
                        if (std::filesystem::exists(targetDir)) {
                            std::filesystem::remove_all(targetDir);
                            std::cout << "[INFO] Deleted directory: " << targetDir << std::endl;
                        } else {
                            std::cout << "[INFO] Directory does not exist: " << targetDir << std::endl;
                        }
                    } catch (const std::filesystem::filesystem_error& e) {
                        std::cerr << "[ERROR] Failed to delete directory: " << targetDir << "\n"
                                  << e.what() << std::endl;
                    }

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
                // ---------- Overlay Progress Bar ----------
                if (shouldRunPython) {
                    //std::cout <<"Overlay block running!\n";
                    ImGui::SetNextWindowPos(ImVec2(0, 0));
                    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
                    ImGui::SetNextWindowBgAlpha(0.8f); // semi-transparent background
                    ImGui::Begin("Overlay", nullptr,
                        ImGuiWindowFlags_NoDecoration |
                        ImGuiWindowFlags_NoMove |
                        ImGuiWindowFlags_NoSavedSettings |
                        ImGuiWindowFlags_AlwaysAutoResize |
                        ImGuiWindowFlags_NoFocusOnAppearing |
                        //ImGuiWindowFlags_NoInputs |
                        //ImGuiWindowFlags_NoBackground |
                        //ImGuiWindowFlags_NoBringToFrontOnFocus |
                        ImGuiWindowFlags_NoNav);
                
                    ImVec2 displaySize = ImGui::GetIO().DisplaySize;
                    ImVec2 center = ImVec2(displaySize.x * 0.5f, displaySize.y * 0.5f);
                    ImVec2 barSize(400.0f, 24.0f);
                    ImVec2 barMin = ImVec2(center.x - barSize.x * 0.5f, center.y);
                    ImVec2 barMax = ImVec2(barMin.x + barSize.x, barMin.y + barSize.y);
                
                    if (total > 0) {
                        float progress = static_cast<float>(cur) / static_cast<float>(total);
                        ImGui::GetWindowDrawList()->AddRectFilled(barMin, barMax, IM_COL32(80, 80, 80, 200), 6.0f);
                        ImGui::GetWindowDrawList()->AddRectFilled(barMin, ImVec2(barMin.x + progress * barSize.x, barMax.y),
                            IM_COL32(50, 160, 255, 255), 6.0f);
                    } else {
                        // Indeterminate animation bar
                        int numSegments = 15;
                        float spacing = 4.0f;
                        float segmentWidth = (barSize.x - spacing * (numSegments - 1)) / numSegments;
                        float t = ImGui::GetTime();
                        float speed = 2.5f;
                        float offset = fmod(t * speed, static_cast<float>(numSegments));
                
                        for (int i = 0; i < numSegments; ++i) {
                            float alpha = 1.0f - fabsf(fmodf(i + offset, numSegments) - numSegments / 2.0f) / (numSegments / 2.0f);
                            alpha = std::clamp(alpha, 0.3f, 1.0f);
                            ImVec2 segMin = ImVec2(barMin.x + i * (segmentWidth + spacing), barMin.y);
                            ImVec2 segMax = ImVec2(segMin.x + segmentWidth, segMin.y + barSize.y);
                            ImGui::GetWindowDrawList()->AddRectFilled(segMin, segMax,
                                ImGui::GetColorU32(ImVec4(0.2f, 0.7f, 1.0f, alpha)), 4.0f);
                        }
                    }
                
                    std::string label = "Processing facial reconstruction...";
                    ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
                    ImVec2 textPos = ImVec2(center.x - textSize.x * 0.5f, barMin.y - textSize.y - 12.0f);
                    ImGui::GetWindowDrawList()->AddText(textPos, IM_COL32_WHITE, label.c_str());
                
                    ImGui::End();
                }
                                
            }

            // === Then finish the frame

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
    static double lastTime = ImGui::GetTime();
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

    //Default Texture------------------------------------------------------------------------
    
    // Static textures: loaded only once
    static GLuint defaultTexture = 0;

    // Log current texture IDs
    //std::cout << "[DEBUG] Texture States:" << std::endl;
    //std::cout << "  defaultTexture ID:  " << defaultTexture << std::endl;

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

     //---------------------------------------------------------------------------------------


    // Setting the Path of outputs and input
    if (!selectedFilePath.empty()) {
        /*std::cout << "[INFO] Selected file path: " << selectedFilePath << std::endl;

        std::filesystem::path selectedPath(selectedFilePath);
        std::string filename = selectedPath.stem().string();

        std::string baseOutputPath = "E:/Project/DECA3/DECA/output/";

        LandmarkFolderPath = baseOutputPath + filename + "/landmarks2d";
        PreviewFolderPath = baseOutputPath + filename + "/inputs";*/

        // Reload preview texture only if path changes
        /*if (selectedFilePath != Last_selectedFilePath) {
            std::cout << "[DEBUG] Last_selectedFilePath BEFORE: " << Last_selectedFilePath << std::endl;
            std::cout << "[DEBUG] selectedFilePath: " << selectedFilePath << std::endl;

            DeleteAllTextures(previewTextures, sortedPreviewKeys);
            DeleteAllTextures(trackingTextures, sortedTrackingKeys);

            Last_selectedFilePath = selectedFilePath;

            std::cout << "[DEBUG] Last_selectedFilePath AFTER: " << Last_selectedFilePath << std::endl;
        }*/
       
        hasShownNoFileInfo = false; // Reset the flag when a file is selected
        UpdateTexture(PreviewFolderPath, previewTextures, sortedPreviewKeys,lastPreviewPath);
        UpdateTexture(LandmarkFolderPath, trackingTextures, sortedTrackingKeys,lastTrackingPath);
        
    } else if (!hasShownNoFileInfo) {
        std::cout << "[INFO] No file selected. Using default black texture." << std::endl;
        hasShownNoFileInfo = true;
    }

    // Split vertically: Left big panel + Right control panel
    ImGui::BeginChild("LeftSection", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0), false);
    {
        // Top Part: Image Preview and Facial Tracking Panels
        ImGui::BeginChild("LeftTop", ImVec2(0, ImGui::GetContentRegionAvail().y - 60), true);
        {
            ImVec2 leftTopMin = ImGui::GetCursorScreenPos(); // Top-left of "LeftTop"
            // Two children side-by-side without extra nesting
            ImGui::BeginChild("LeftTab", ImVec2(ImGui::GetContentRegionAvail().x * 0.5f, 0), true);
            {
                if (ImGui::BeginTabBar("LeftTabBar"))
                {
                    if (ImGui::BeginTabItem("Image Preview"))
                    {
                        //ImGui::Text("Image Preview Content");
                        ImVec2 availableSize = ImGui::GetContentRegionAvail();
                        if (!selectedFilePath.empty() && !sortedPreviewKeys.empty()) {
                            double currentTime = ImGui::GetTime();
                            if (currentTime - lastTime >= frameDuration) {
                                currentFrameIndex = (currentFrameIndex + 1) % previewTextures.size();
                            }
                            const GLuint& frameKey = sortedPreviewKeys[currentFrameIndex];
                            GLuint texID = frameKey;
                            ImGui::Image((ImTextureID)(intptr_t)texID, availableSize);
                        } else {
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
                        // Placeholder image centered
                        ImVec2 imageSize = ImGui::GetContentRegionAvail();
                        float padding = 10.0f;
                        if (imageSize.x > padding * 2 && imageSize.y > padding * 2)
                            imageSize = ImVec2(imageSize.x - padding * 2, imageSize.y - padding * 2);
                    
                        float availWidth = ImGui::GetContentRegionAvail().x;
                        float offsetX = (availWidth - imageSize.x) * 0.5f;
                        if (offsetX > 0)
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
                    
                        if (!selectedFilePath.empty() && !sortedTrackingKeys.empty()) {
                            double currentTime = ImGui::GetTime();
                            if (currentTime - lastTime >= frameDuration) {
                                currentFrameIndex = (currentFrameIndex + 1) % trackingTextures.size();
                                lastTime = currentTime;
                            }
                    
                            const GLuint& frameKey = sortedTrackingKeys[currentFrameIndex];
                            GLuint texID = frameKey;
                    
                            ImVec2 imagePos = ImGui::GetCursorScreenPos();
                            ImGui::Image((ImTextureID)(intptr_t)texID, imageSize);
                    
                            // Debug output
                           /* ImGui::Text("sortedTrackingKeys size: %d", (int)sortedTrackingKeys.size());
                            ImGui::Text("Current frame index: %d", currentFrameIndex);
                            ImGui::Text("texID (GLuint): %u", texID);*/
                        } else {
                            ImVec2 imagePos = ImGui::GetCursorScreenPos();
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, imageSize);
                    
                            // Debug output
                            /*ImGui::Text("No valid texture to show.");
                            ImGui::Text("sortedTrackingKeys size: %d", (int)sortedTrackingKeys.size());
                            ImGui::Text("selectedFilePath: %s", selectedFilePath.c_str());*/
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
                std::cout << "[DEBUG] Last_selectedFilePath AFTER: " << Last_selectedFilePath << std::endl;
                selectedFilePath = ImGuiFileDialog::Instance()->GetFilePathName();
                Last_selectedFilePath = selectedFilePath;
                shouldRunPython = true;
                pendingPythonPath = selectedFilePath;
                std::cout << "[INFO] Selected file path: " << selectedFilePath << std::endl;

                std::filesystem::path selectedPath(selectedFilePath);
                std::string filename = selectedPath.stem().string();
                //Setting up directory for selected file
                std::string baseOutputPath = "E:/Project/DECA3/DECA/output/";
                LandmarkFolderPath = baseOutputPath + filename + "/landmarks2d";
                PreviewFolderPath = baseOutputPath + filename + "/inputs";
                targetDir = baseOutputPath + filename;
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

void UpdateTexture(const std::string& FolderPath, std::map<std::string, GLuint> &Textures, std::vector<GLuint> &sortedKeys, std::string &LastFile) {
    namespace fs = std::filesystem;

    // Static state per folder
    static std::map<std::string, bool> shownFolderMissing;
    static std::map<std::string, bool> shownNoImage;
    static std::map<std::string, std::string> lastDebuggedFile;

    if (!fs::exists(FolderPath)) {
        if (!shownFolderMissing[FolderPath]) {
            std::cout << "[INFO] Folder does not exist yet: " << FolderPath << std::endl;
            shownFolderMissing[FolderPath] = true;
        }
        return;
    }
    shownFolderMissing[FolderPath] = false; // Reset when folder exists

    // Find latest .jpg file in folder
    std::string latestFile;
    fs::file_time_type latestTime;

    for (const auto& entry : fs::directory_iterator(FolderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            auto writeTime = fs::last_write_time(entry);
            if (writeTime > latestTime) {
                latestTime = writeTime;
                latestFile = entry.path().string();
            }
        }
    }

    if (latestFile.empty()) {
        if (!shownNoImage[FolderPath]) {
            std::cout << "[INFO] No image files found in: " << FolderPath << std::endl;
            shownNoImage[FolderPath] = true;
        }
        return;
    }
    shownNoImage[FolderPath] = false; // Reset when an image is found

    if (latestFile != LastFile) {
        // New file detected — load texture and keep old ones
        GLuint newTexture = LoadTextureFromFile(latestFile.c_str());
        if (newTexture != 0) {
            Textures[latestFile] = newTexture;

            std::cout << "[INFO] Loaded previous last texture path: " << LastFile << std::endl;
            LastFile = latestFile;
            std::cout << "[INFO] Loaded current texture path: " << LastFile << std::endl;
            std::cout << "[INFO] Loaded new texture: " << latestFile << " -> ID: " << newTexture << std::endl;

            sortedKeys.push_back(newTexture);
            std::sort(sortedKeys.begin(), sortedKeys.end()); // Optional sort
        } else {
            std::cout << "[WARN] Failed to load new texture: " << latestFile << std::endl;
        }

        lastDebuggedFile[FolderPath] = LastFile;
    } else {
        if (lastDebuggedFile[FolderPath] != LastFile) {
            std::cout << "[DEBUG] No new image. Latest: " << LastFile << std::endl;
            lastDebuggedFile[FolderPath] = LastFile;
        }
    }
}

void DeleteAllTextures(std::map<std::string, GLuint>& Textures, std::vector<GLuint>& SortedKeys) {
    for (const auto& pair : Textures) {
        GLuint texID = pair.second;
        if (texID != 0) {
            std::cout << "[INFO] Deleting texture '" << pair.first << "' (ID: " << texID << ")" << std::endl;
            glDeleteTextures(1, &texID);
        }
    }
    Textures.clear();  // Clear the map after deletion
    SortedKeys.clear();
}

void SetAppWorkingDirectory()
{
    // Get current working directory
    fs::path cwd = fs::current_path();
    std::cout << "[INFO] Current working directory: " << cwd << std::endl;

    // Check if we're running from the "build/bin/Release" path
    if (cwd.filename() == "Release" && cwd.parent_path().filename() == "bin") {
        // We're likely running from the .exe
        // Calculate project root: cwd = DECA/build/bin/Release → project = DECA
        fs::path projectRoot = cwd.parent_path().parent_path().parent_path();
        std::cout << "[INFO] Switching working directory to: " << projectRoot << std::endl;

        if (!SetCurrentDirectoryA(projectRoot.string().c_str())) {
            std::cerr << "[ERROR] Failed to set working directory to " << projectRoot << std::endl;
        }
    } else {
        std::cout << "[INFO] Running from VS Code or already in project root. No directory change needed." << std::endl;
    }
}
