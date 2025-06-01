#define NOMINMAX

//UNIVERSAL PACKAGE
#include <iostream>
#include <cstdlib>
#include <string>
#include <thread>
#include <atomic>
using namespace std::chrono_literals;
#include <chrono>
//#include "progress.hpp"
#include <sstream> // for std::stringstream
#include <cmath> // For sin() and M_PI
#include <algorithm>  // for std::clamp
//FILE MANAGEMENT
#include <filesystem>
//WINDOWS PLATFORM DEV
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "Dwmapi.lib")
//OPENGL SERIES
//#include <GL/glew.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "ImGuiFileDialog.h"
//PYTHON INTERPRETER
#include <pybind11/embed.h> // Everything needed for embedding
#include <Python.h>
namespace py = pybind11;
//INCLUDED HEADER
#include "progress_shared.hpp"
#include "Model.hpp"
#include "Shader.hpp"
#include "Camera.hpp"
#include "TextureLoader.hpp"


namespace fs = std::filesystem;

typedef void (__cdecl *update_progress_t)(int, int);
typedef int (__cdecl *get_progress_t)();
typedef int (*GetProgressFunc)();

// Global or class-level flag and thread
std::thread pythonThread;// Face_reconstruction thread
std::atomic<bool> shouldRunPython(false);
std::string pendingPythonPath;
bool pythonThreadRunning = false; 
bool done_reconstruction = true;
std::thread g_TextureThread;//Textures_loading thread
std::atomic<bool> textureUpdateRequested(false);
// Global atomic status variable to track progress inside the thread
std::atomic<int> g_TextureThreadProgress(0);
//std::atomic<bool> g_TextureThreadRunning(false);
// Static or global variable to remember last printed progress
static int lastPrintedProgress = -1;
static int frameCount = 0; // Keeps track of how many frames have passed

// Global variables for progress
static bool hasShownNoFileInfo = false;
int cur = get_current_progress();
int prevCur = -1;
int prevTotal = -1;
int total = get_total_progress();
std::string progressLabel = "Waiting...";

// Global variables for playback control
static bool autoPlay = true;
static bool isDragging = false;

//Global variable for ModelView
Camera camera;
bool mouseDragging = false;
float lastX = 0, lastY = 0;
GLuint fbo, colorTex, depthRb;
static Shader shader;
static Model model;
static Model model2;
std::string currentShaderVertexPath;
std::string currentShaderFragmentPath;
std::string currentModelPath;
std::string reconstructModelPath;
std::string reconstructModelPath2;
static int animationMode = 0; // 0 = Dynamic Capture, 1 = Predefined Expression
static float expressions[50] = {}; // range [-1, 1]

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

//=====================Functions==========================
//Interface Rendering---------------------------------------------------------------------------
//GUI of Facial Tracking-----
void RenderFacialTrackingTab();
void UpdateFacialTracking();
void RenderFacialTrackingUI();
//GUI of Facial Reconstruction
void RenderFacialReconstructionTab();
void UpdateFacialReconstruction(const std::string& modelPath);
void DrawModelView();
void InitFramebuffer(int width, int height);

//IMGUI Texture Rendering------------------------------------------------------------------------
//Default Texture
static GLuint defaultTexture = 0;
//Frames Playback-----
float playbackFPS = 30.0f;            // Target playback rate
static size_t currentFrameIndex = 0;
static double frameDuration = 1.0 / playbackFPS;
bool isFrameDurationInitialized = false;
static double lastTime = 0.0;
static double currentTime = 0.0;
float alpha;
static int startFrame = 0, endFrame = 0;

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
//Export Directory Configuration-----
static std::string exportFilePath;
static std::string exportStatusMsg;
//---------------------

//Configuring Working Directory
void SetAppWorkingDirectory();

//Updating Tracking Images
void UpdateTexture(const std::string& FolderPath, std::map<std::string, GLuint> &Textures,std::vector<GLuint> &SortedKeys,std::string &LastFile);

//Loading Single Images as Texture
//GLuint LoadTextureFromFile(const char* filename);

//Delete Tracking Images
void DeleteAllTextures(std::map<std::string, GLuint>& Textures, std::vector<GLuint>& sortedKeys); 
//----------------------------------------------------------------------------------------------

//Running Face Reconstruction Task in Python
void runPythonConstruct(const std::string& selectedFilePath);

//Playback Control
bool TimelineWidget(const char* label, size_t& currentFrameIndex, size_t totalFrames, bool& autoPlay);


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
        const char* glsl_version = "#version 330";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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
        // Create shared window for the background pipeline
        GLFWwindow* sharedWindow = glfwCreateWindow(1, 1, "", nullptr, window);
        // Keep it hidden, never show it

        HWND hwnd = glfwGetWin32Window(window);
        EnableDarkTitleBar(hwnd);

        // Force Windows to redraw the title bar immediately
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);

        // Now show the window
        glfwShowWindow(window);
        
        glfwMakeContextCurrent(window);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        shader = Shader("shaders/vertex.glsl", "shaders/fragment.glsl");
        //model = Model("Animation1.glb", shader);
        //static Shader shader("shaders/vertex.glsl", "shaders/fragment.glsl");
        //static Model model("Animation.glb");
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        stbi_set_flip_vertically_on_load(true); // Flip if needed

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
                if (!isFrameDurationInitialized && total != 0) {
                    float fps = get_FPS();
                    if (fps <= 0.0f) {
                        std::cerr << "Error: get_FPS() returned non-positive value: " << fps << std::endl;
                    } else {
                        frameDuration = 1.0f / fps;
                        isFrameDurationInitialized = true;
                        std::cout << "frameDuration initialized to: " << frameDuration << " (FPS = " << fps << ")" << std::endl;
                    }
                    isFrameDurationInitialized = true;
                }
            }

            py::gil_scoped_release release; // This releases the GIL for this scope   
            glfwPollEvents();

            // At the top of your loop
            //std::cout << "\n========== Frame: " << frameCount << " ==========" << std::endl;
            
            // Updating Textures
            if (shouldRunPython && cur!= prevCur) {
                hasShownNoFileInfo = false;
            
                //std::cout << "[DEBUG] Frame " << frameCount << " | textureUpdateRequested: " << textureUpdateRequested << "\n";
                //std::cout << "[DEBUG] Frame " << frameCount << " | g_TextureThreadRunning: " << g_TextureThreadRunning << "\n";
                std::cout << "[DEBUG] TextureUpdateFrame "  << frameCount << "\n";
            
                if (!textureUpdateRequested) {
                    textureUpdateRequested = true;
                    //g_TextureThreadRunning = true;
            
                    //std::cout << "[DEBUG] Frame " << frameCount << " | Starting texture thread...\n";
                    static int threadframeCount = 0; // Keeps track of how many frames have passed
            
                    g_TextureThread = std::thread([&]() {
                        // Updating Textures
                        
                        //std::cout << "[DEBUG THREAD] (Frame " << threadframeCount << ") Entered thread\n";
            
                        g_TextureThreadProgress = 0;
            
                        glfwMakeContextCurrent(sharedWindow);
            
                        g_TextureThreadProgress = 1;
                        //std::cout << "[DEBUG THREAD] (Frame " << threadframeCount  << ") Loading preview textures...\n";
                        UpdateTexture(PreviewFolderPath, previewTextures, sortedPreviewKeys, lastPreviewPath);
            
                        g_TextureThreadProgress = 2;
                        //std::cout << "[DEBUG THREAD] (Frame " << threadframeCount  << ") Loading landmark textures...\n";
                        UpdateTexture(LandmarkFolderPath, trackingTextures, sortedTrackingKeys, lastTrackingPath);
            
                        glfwMakeContextCurrent(nullptr);
            
                        g_TextureThreadProgress = 3;
                        //std::cout << "[DEBUG THREAD] (Frame " << threadframeCount  << ") Texture loading finished.\n";
            
                        textureUpdateRequested = false;
                        //g_TextureThreadRunning = false;
            
                        //std::cout << "[DEBUG THREAD] (Frame " << threadframeCount  << ") Thread flags reset\n";
                        threadframeCount++;

                    });
            
                    g_TextureThread.detach();
                }
                


                if(cur == total){
                    shouldRunPython = false;
                    pythonThreadRunning = false;
                    //done_reconstruction = false;
                }


            
                /*if (g_TextureThreadRunning) {
                    int progress = g_TextureThreadProgress.load();
                    if (progress != lastPrintedProgress) {
                        lastPrintedProgress = progress;
                        std::cout << "[FRAME " << frameCount << "] ";
                        switch (progress) {
                            case 0: std::cout << "Texture thread started\n"; break;
                            case 1: std::cout << "Loading preview textures...\n"; break;
                            case 2: std::cout << "Loading landmark textures...\n"; break;
                            case 3: std::cout << "Texture loading finished.\n"; break;
                            default: std::cout << "Unknown progress state.\n"; break;
                        }
                    }
                } else {
                    if (lastPrintedProgress != -2) {
                        std::cout << "[FRAME " << frameCount << "] Texture thread not running\n";
                        lastPrintedProgress = -2;
                    }
                }*/
            
            } else if (!hasShownNoFileInfo) {
                std::cout << "[INFO] No file selected. Using default black texture." << std::endl;
                hasShownNoFileInfo = true;
            }

            if (cur != prevCur || total != prevTotal) {
                std::cout << "C++ Progress: " << cur << " / " << total << std::endl;
                prevCur = cur;
                prevTotal = total;
            }
            
            
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            UpdateFacialTracking();

            UpdateFacialReconstruction(reconstructModelPath);

            
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
                    ImGuiWindowFlags_NoTitleBar);

                //Application Layout Configuration
                ImVec2 windowSize = ImGui::GetContentRegionAvail();
                float tabHeight = windowSize.y*0.8f;         // 70% for tabs
                float timelineHeight = windowSize.y - tabHeight; // remaining for timeline
                
                // Top Region: Tabs
                ImGui::BeginChild("MainTabsRegion", ImVec2(0, tabHeight), true);
                {
                    if (ImGui::BeginTabBar("MainTabs"))
                    {
                        if (ImGui::BeginTabItem("Facial Capture"))
                        {
                            //RenderFacialTrackingTab();
                            RenderFacialTrackingUI();
                            ImGui::EndTabItem();
                        }

                        if (ImGui::BeginTabItem("Facial Reconstruction"))
                        {
                            //ImGui::Text("Facial Capture content goes here...");
                            RenderFacialReconstructionTab();
                            ImGui::EndTabItem();
                        }

                        ImGui::EndTabBar();
                    }
                }
                ImGui::EndChild();

                // Bottom Region: Timeline
                ImGui::BeginChild("TimelineRegion", ImVec2(0, timelineHeight), true);
                {
                    TimelineWidget("MyTimeline", currentFrameIndex, sortedTrackingKeys.size(), autoPlay);
                }
                ImGui::EndChild();

                ImGui::End(); // End of window               

                // After ImGui::Render() or inside your main loop:
                static std::thread pythonThread;
                if (shouldRunPython && !pythonThreadRunning) {
                    pythonThreadRunning = true;
                    std::string capturedPath = pendingPythonPath;
                        
                    
                    //Delete Previous Cache
                    DeleteAllTextures(previewTextures, sortedPreviewKeys);
                    DeleteAllTextures(trackingTextures, sortedTrackingKeys);
                    lastPreviewPath = "";
                    lastTrackingPath = "";
                    reconstructModelPath = "";
                    reconstructModelPath2 = "";
                    startFrame = 0;
                    endFrame = 0;
                    std::memset(expressions, 0, sizeof(expressions));
                    currentFrameIndex = 0;


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
                                //reconstructModelPath = "Animation1.glb";
                                reconstructModelPath = get_model_path();
                                reconstructModelPath2 = get_expression_path();
                                
                                std::cout << "Successfully constructed the face model.\n";
                                std::cout << "All done!" << std::endl;

                                endFrame = sortedTrackingKeys.size();
                            }
                
                        } catch (const std::exception& e) {
                            std::cerr << "C++ exception in thread: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "Unknown exception in Python thread!\n";
                        }
                        
                        //pythonThreadRunning = false;
                        
                        //shouldRunPython = false;
                    });
                
                    pythonThread.detach();
                    
                }
                // ---------- Overlay Progress Bar ----------
                //if (shouldRunPython) {
                if (!done_reconstruction) {
                
                    //std::cout <<"Overlay block running!\n";
                    ImGui::SetNextWindowPos(ImVec2(0, 0));
                    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
                    ImGui::SetNextWindowBgAlpha(0.8f); // semi-transparent background
                    ImGui::SetNextWindowFocus();
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
            ++frameCount; // Advance frame counter at end of the loop
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

void UpdateFacialTracking()
{
    //static GLuint defaultTexture = 0;

    // Lazy load default black texture once
    if (defaultTexture == 0) {
        const int width = 256;
        const int height = 256;
        unsigned char blackPixels[width * height * 3];
        std::fill_n(blackPixels, width * height * 3, 0);

        glGenTextures(1, &defaultTexture);
        glBindTexture(GL_TEXTURE_2D, defaultTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, blackPixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // Handle ImGuiFileDialog state changes here,
    // file selection processing, updating paths, flags
    if (ImGuiFileDialog::Instance()->Display("ChooseCapture")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            selectedFilePath = ImGuiFileDialog::Instance()->GetFilePathName();
            Last_selectedFilePath = selectedFilePath;
            shouldRunPython = true;
            done_reconstruction = false;
            isFrameDurationInitialized = false;
            pendingPythonPath = selectedFilePath;
            update_progress(0,0);
            prevCur = 0;

            std::filesystem::path selectedPath(selectedFilePath);
            std::string filename = selectedPath.stem().string();
            std::string baseOutputPath = "./output/";
            LandmarkFolderPath = baseOutputPath + filename + "/landmarks2d";
            PreviewFolderPath = baseOutputPath + filename + "/inputs";
            targetDir = baseOutputPath + filename;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Any other update logic you want, e.g.:
    // - Update currentFrameIndex if autoplay enabled
    // - Manage texture arrays sortedPreviewKeys, sortedTrackingKeys
}

void RenderFacialTrackingUI()
{
    ImGui::BeginChild("LeftSection", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0), false);
    {
        ImGui::BeginChild("LeftTop", ImVec2(0, ImGui::GetContentRegionAvail().y), true);
        {
            ImGui::BeginChild("LeftTab", ImVec2(ImGui::GetContentRegionAvail().x * 0.5f, 0), true);
            {
                if (ImGui::BeginTabBar("LeftTabBar"))
                {
                    if (ImGui::BeginTabItem("Image Preview"))
                    {
                        ImVec2 imageSize = ImGui::GetContentRegionAvail();
                        float padding = 10.0f;
                        if (imageSize.x > padding * 2 && imageSize.y > padding * 2)
                            imageSize = ImVec2(imageSize.x - padding * 2, imageSize.y - padding * 2);

                        float availWidth = ImGui::GetContentRegionAvail().x;
                        float offsetX = (availWidth - imageSize.x) * 0.5f;
                        if (offsetX > 0)
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

                        if (!selectedFilePath.empty() && !sortedPreviewKeys.empty()) {
                            const GLuint& frameKey = sortedPreviewKeys[currentFrameIndex];
                            ImGui::Image((ImTextureID)(intptr_t)frameKey, imageSize, ImVec2(0, 1), ImVec2(1, 0));
                        }
                        else {
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, imageSize, ImVec2(0, 1), ImVec2(1, 0));
                        }
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
                        ImVec2 imageSize = ImGui::GetContentRegionAvail();
                        float padding = 10.0f;
                        if (imageSize.x > padding * 2 && imageSize.y > padding * 2)
                            imageSize = ImVec2(imageSize.x - padding * 2, imageSize.y - padding * 2);

                        float availWidth = ImGui::GetContentRegionAvail().x;
                        float offsetX = (availWidth - imageSize.x) * 0.5f;
                        if (offsetX > 0)
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

                        if (!selectedFilePath.empty() && !sortedTrackingKeys.empty()) {
                            const GLuint& frameKey = sortedTrackingKeys[currentFrameIndex];
                            ImGui::Image((ImTextureID)(intptr_t)frameKey, imageSize, ImVec2(0, 1), ImVec2(1, 0));
                        }
                        else {
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, imageSize, ImVec2(0, 1), ImVec2(1, 0));
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("RightSection", ImVec2(0, 0), true);
    {
        // Section Header
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f), "Capture Source");
        ImGui::Spacing();
    
        // File Selection
        ImGui::BeginGroup();
        float buttonWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Select File...", ImVec2(buttonWidth, 0))) {
            IGFD::FileDialogConfig config;
            config.path = GetDesktopPath();
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_Modal;
    
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCapture",
                "Select Capture Source",
                "Images and Videos{.png,.jpg,.jpeg,.bmp,.gif,.mp4,.avi,.mov,.mkv}",
                config
            );
        }
    
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        ImGui::TextWrapped("%s", selectedFilePath.empty() ? "< No file selected >" : selectedFilePath.c_str());
        ImGui::PopStyleColor();
        ImGui::EndGroup();
    
        // Divider (Minimal style)
        ImVec2 start = ImGui::GetCursorScreenPos();
        ImVec2 end = ImVec2(start.x + ImGui::GetContentRegionAvail().x, start.y + 1.0f);
        ImGui::GetWindowDrawList()->AddLine(start, end, IM_COL32(80, 80, 80, 100));
        ImGui::Dummy(ImVec2(0.0f, 6.0f));
    
        // Default frame values
        int defaultStartFrame = 0;
        int defaultEndFrame = sortedTrackingKeys.size();

        ImGui::BeginGroup();

        // Section title with subtle uppercase and spacing
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // customize your font if available
        ImGui::TextUnformatted("FRAME RANGE TO PROCESS");
        ImGui::PopFont();

        ImGui::Spacing();

        // Container with light background and rounded corners
        ImGui::BeginChild("FrameRangeContainer", ImVec2(0, 120), true,
                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.13f, 0.15f, 0.18f, 1.0f)); // dark gray background
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.17f, 0.20f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.20f, 0.23f, 0.27f, 1.0f));

        // Use a two-column grid for labels and inputs with spacing
        ImGui::Columns(2, nullptr, false);
        ImGui::SetColumnWidth(0, 130);

        // Clamp values before showing
        startFrame = std::clamp(startFrame, 0, std::max(0, defaultEndFrame - 1));
        endFrame = std::clamp(endFrame, startFrame + 1, std::max(startFrame + 1, defaultEndFrame));

        // Start Frame
        ImGui::Text("Start Frame");
        ImGui::NextColumn();
        ImGui::PushItemWidth(-1);
        if (ImGui::DragInt("##StartFrame", &startFrame, 1.0f)) {
            startFrame = std::clamp(startFrame, 0, std::max(0, defaultEndFrame - 1));

            if (endFrame <= startFrame)
                endFrame = std::min(startFrame + 1, defaultEndFrame);
        }
        ImGui::PopItemWidth();
        ImGui::NextColumn();

        // End Frame
        ImGui::Text("End Frame");
        ImGui::NextColumn();
        ImGui::PushItemWidth(-1);
        if (ImGui::DragInt("##EndFrame", &endFrame, 1.0f)) {
            //endFrame = std::clamp(endFrame, startFrame + 1, defaultEndFrame);
            endFrame = std::clamp(endFrame, 0, defaultEndFrame);
            
            if (endFrame <= startFrame)
                startFrame = std::max(endFrame - 1, defaultStartFrame);
        }
        ImGui::PopItemWidth();


        ImGui::Columns(1);

        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar();

        ImGui::Spacing();

        // Validation / status line with modern subtle text
        ImVec4 textColorGood = ImVec4(0.40f, 0.85f, 0.55f, 1.0f);
        ImVec4 textColorBad = ImVec4(0.95f, 0.40f, 0.40f, 1.0f);

        if (startFrame < 0) {
            ImGui::TextColored(textColorBad, "Start frame cannot be negative.");
        } else if (endFrame <= startFrame) {
            ImGui::TextColored(textColorBad, "End frame must be greater than start frame.");
        } else {
            ImGui::TextColored(textColorGood, "Processing frames %d to %d", startFrame, endFrame);
        }

        ImGui::Spacing();

        // Reset button with flat style and accent color, right-aligned
        float btnWidth = 110.0f;
        ImGui::Dummy(ImVec2(0,0));
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - btnWidth);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.65f, 0.87f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.72f, 0.92f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.58f, 0.78f, 1.0f));
        if (ImGui::Button("Reset Frames", ImVec2(btnWidth, 30))) {
            startFrame = defaultStartFrame;
            endFrame = defaultEndFrame;
        }
        ImGui::PopStyleColor(3);

        ImGui::EndChild();

        ImGui::EndGroup();



    }
    
    ImGui::EndChild();
}

void RenderFacialTrackingTab()
{
     //py::gil_scoped_acquire acquire;

    
     //Loading Screen------------------------------------------------------------------------
    // Constants
    //static double lastTime = ImGui::GetTime();
    /*
    const int numSegments = 20;
    const float barWidth = 300.0f;
    const float barHeight = 20.0f;
    const float spacing = 2.0f;
    const float segmentWidth = (barWidth - spacing * (numSegments - 1)) / numSegments;
    const float speed = 2.0f;  // how fast the animation loops*/

    // Get draw position
    /*
    ImVec2 barSize(barWidth, barHeight);
    float availWidth = ImGui::GetContentRegionAvail().x;
    float offsetX = (availWidth - barSize.x) * 0.5f;
    if (offsetX > 0)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);*/

    // Reserve space for the bar
    /*
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(barSize);
    */
    // Animate
    /*
    float t = ImGui::GetTime();
    float offset = fmod(t * speed, numSegments);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    */
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
    // Split vertically: Left big panel + Right control panel
    ImGui::BeginChild("LeftSection", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0), false);
    {
        // Top Part: Image Preview and Facial Tracking Panels
        ImGui::BeginChild("LeftTop", ImVec2(0, ImGui::GetContentRegionAvail().y), true);
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
                        ImVec2 imageSize = ImGui::GetContentRegionAvail();
                        float padding = 10.0f;
                        if (imageSize.x > padding * 2 && imageSize.y > padding * 2)
                            imageSize = ImVec2(imageSize.x - padding * 2, imageSize.y - padding * 2);
                        float availWidth = ImGui::GetContentRegionAvail().x;
                        float offsetX = (availWidth - imageSize.x) * 0.5f;
                        if (offsetX > 0)
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
                        if (!selectedFilePath.empty() && !sortedPreviewKeys.empty()) {
                                /*if(autoPlay){
                                    double currentTime = ImGui::GetTime();
                                    if (currentTime - lastTime >= frameDuration) {
                                        currentFrameIndex = (currentFrameIndex + 1) % previewTextures.size();
                                        lastTime = currentTime;
                                    }
                                }*/
                                const GLuint& frameKey = sortedPreviewKeys[currentFrameIndex];
                                ImVec2 imagePos = ImGui::GetCursorScreenPos();
                                GLuint texID = frameKey;
                                ImGui::Image((ImTextureID)(intptr_t)texID, imageSize,ImVec2(0, 1), ImVec2(1, 0));
                           
                        } else {
                            ImVec2 imagePos = ImGui::GetCursorScreenPos();
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, imageSize,ImVec2(0, 1), ImVec2(1, 0));
                        }
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
                            /*if(autoPlay){
                                double currentTime = ImGui::GetTime();
                                if (currentTime - lastTime >= frameDuration) {
                                    currentFrameIndex = (currentFrameIndex + 1) % trackingTextures.size();
                                    lastTime = currentTime;
                                }
                            };*/
                            const GLuint& frameKey = sortedTrackingKeys[currentFrameIndex];
                            GLuint texID = frameKey;
                            ImVec2 imagePos = ImGui::GetCursorScreenPos();
                            ImGui::Image((ImTextureID)(intptr_t)texID, imageSize,ImVec2(0, 1), ImVec2(1, 0));
                    
                            // Debug output
                           /* ImGui::Text("sortedTrackingKeys size: %d", (int)sortedTrackingKeys.size());
                            ImGui::Text("Current frame index: %d", currentFrameIndex);
                            ImGui::Text("texID (GLuint): %u", texID);*/
                        } else {
                            ImVec2 imagePos = ImGui::GetCursorScreenPos();
                            ImGui::Image((ImTextureID)(intptr_t)defaultTexture, imageSize,ImVec2(0, 1), ImVec2(1, 0));
                    
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
                update_progress(0,0);
                prevCur=0;
                std::filesystem::path selectedPath(selectedFilePath);
                std::string filename = selectedPath.stem().string();
                //Setting up directory for selected file
                std::string baseOutputPath = "./output/";
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

bool TimelineWidget(const char* id, size_t& currentFrame, size_t totalFrames, bool& autoPlay) {


    // --- Constants ---
    const float controlsH     = 28.0f;  // space for buttons above
    const float headerH       = 18.0f;
    const float trackH        = 60.0f;
    float availableH = ImGui::GetContentRegionAvail().y;
    float widgetH = availableH;
    const ImU32 bgHeaderCol   = IM_COL32(60, 60, 60, 255);
    const ImU32 bgTrackCol    = IM_COL32(32, 32, 32, 255);
    const ImU32 borderCol     = IM_COL32(90, 90, 90, 255);
    const ImU32 tickMajorCol  = IM_COL32(140, 140, 140, 180);
    const ImU32 tickMinorCol  = IM_COL32(90, 90, 90, 140);
    const ImU32 keyframeCol   = IM_COL32(255, 150, 20, 255);
    const ImU32 playheadCol   = IM_COL32(50, 255, 50, 255);
    const float keyDiamondR   = 5.0f;
    const float majorTickH    = 12.0f;
    const float minorTickH    = 6.0f;

    // -- Frame Update --
    currentTime = ImGui::GetTime();
    double delta = currentTime - lastTime;
    
    if (autoPlay && totalFrames > 0) {
        if (delta >= frameDuration) {
            currentFrame = (currentFrame + 1) % totalFrames;
            lastTime = currentTime;
            delta = 0.0; // reset delta so alpha doesn't overshoot
        }
    
        alpha = static_cast<float>(delta / frameDuration);
        alpha = std::clamp(alpha, 0.0f, 1.0f);
    }

    // --- Layout ---
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 startPos = ImGui::GetCursorScreenPos();
    ImVec2 fullSize = ImVec2(ImGui::GetContentRegionAvail().x, widgetH);

    // Play controls above timeline, centered
    ImVec2 controlSize(120, controlsH);
    ImVec2 controlPos = ImVec2(startPos.x + (fullSize.x - controlSize.x) * 0.5f, startPos.y);
    ImGui::SetCursorScreenPos(controlPos);

    ImGui::BeginGroup();
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);  // Rounded button
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 4));
    if (ImGui::Button(autoPlay ? "Pause##tl" : "Play##tl", ImVec2(60, 0)))
        autoPlay = !autoPlay;
    ImGui::SameLine();
    char frameLabel[64];
    snprintf(frameLabel, sizeof(frameLabel), " %zu / %zu ", currentFrame, totalFrames ? totalFrames - 1 : 0);
    
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(40, 40, 40, 255));
    ImGui::PushStyleColor(ImGuiCol_Border,  IM_COL32(80, 80, 80, 255));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,   4.0f);
    ImGui::PushItemWidth(100.0f);
    
    ImGui::BeginDisabled(); // disable interaction
    ImGui::InputText("##frame_display", frameLabel, IM_ARRAYSIZE(frameLabel), ImGuiInputTextFlags_ReadOnly);
    ImGui::EndDisabled();
    
    ImGui::PopItemWidth();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    
    ImGui::PopStyleVar(3);
    ImGui::EndGroup();

    // Leave space before drawing timeline
    ImGui::SetCursorScreenPos(ImVec2(startPos.x, startPos.y + controlsH));
    ImVec2 timelinePos = ImGui::GetCursorScreenPos();

    size_t framesMax = totalFrames ? totalFrames : 1;
    float pxPerFrame = fullSize.x / static_cast<float>(framesMax);

    // --- Background ---
    dl->AddRectFilled(timelinePos, ImVec2(timelinePos.x + fullSize.x, timelinePos.y + headerH), bgHeaderCol, 4.0f, ImDrawFlags_RoundCornersTop);
    dl->AddRectFilled(ImVec2(timelinePos.x, timelinePos.y + headerH),
                      ImVec2(timelinePos.x + fullSize.x, timelinePos.y + fullSize.y - controlsH), bgTrackCol);
    dl->AddRect(timelinePos, ImVec2(timelinePos.x + fullSize.x, timelinePos.y + fullSize.y - controlsH), borderCol, 4.0f, 0, 1.5f);

    // --- Ticks ---
    float baseY = timelinePos.y + headerH;
    for (size_t f = 0; f < totalFrames; ++f) {
        float x = timelinePos.x + f * pxPerFrame;
        bool major = (f % 10 == 0);
        bool minor = (f % 5 == 0);

        if (major || minor) {
            float h = major ? majorTickH : minorTickH;
            dl->AddLine(ImVec2(x, baseY - h), ImVec2(x, baseY), major ? tickMajorCol : tickMinorCol);
        }

        if (major) {
            char buf[8];
            snprintf(buf, sizeof(buf), "%zu", f);
            dl->AddText(ImVec2(x + 2, timelinePos.y + 2), IM_COL32(220, 220, 220, 200), buf);
        }
    }

    // --- Keyframes ---
    for (size_t f = 0; f < totalFrames; f += 15) {
        float cx = timelinePos.x + f * pxPerFrame;
        float cy = baseY + trackH * 0.5f;
        dl->AddQuadFilled(
            ImVec2(cx,             cy - keyDiamondR),
            ImVec2(cx + keyDiamondR, cy),
            ImVec2(cx,             cy + keyDiamondR),
            ImVec2(cx - keyDiamondR, cy),
            keyframeCol);
    }

    // --- Playhead ---
    float headX = timelinePos.x + currentFrame * pxPerFrame;
    dl->AddLine(ImVec2(headX, timelinePos.y), ImVec2(headX, timelinePos.y + widgetH - controlsH), playheadCol, 2.0f);
    dl->AddTriangleFilled(
        ImVec2(headX,             timelinePos.y - 6.0f),
        ImVec2(headX - 5.0f,      timelinePos.y),
        ImVec2(headX + 5.0f,      timelinePos.y),
        playheadCol);

    // --- Interaction ---
    ImGui::InvisibleButton(id, ImVec2(fullSize.x, headerH + trackH));
    if (ImGui::IsItemActive()) {
        float mouseX = ImGui::GetIO().MousePos.x;
        size_t newFrame = static_cast<size_t>((mouseX - timelinePos.x) / pxPerFrame);
        if (newFrame < totalFrames) currentFrame = newFrame;
    }

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Click to scrub timeline");

    return ImGui::IsItemHovered() || ImGui::IsItemActive();
}

void UpdateFacialReconstruction(const std::string& modelPath){
    
    //shader = Shader("shaders/vertex.glsl", "shaders/fragment.glsl");
    if(modelPath != currentModelPath){
        if (modelPath.empty()) {
            currentModelPath = modelPath;
            model = Model();  // Default constructor
            model2 = Model();
            //done_reconstruction = false;  // or whatever logic fits
        } else if (!modelPath.empty()) {
            model = Model(modelPath.c_str(), shader);  // Load the new model
            // Construct manual_animation.glb path
            fs::path originalPath(modelPath);
            fs::path modifiedPath = originalPath.parent_path() / "manual_animation.glb";

            model2 = Model(modifiedPath.string().c_str(), shader);
            currentModelPath = modelPath;
            done_reconstruction = true;
        }
    } 
}

void RenderFacialReconstructionTab() {
    // Get the total width and set the first column to 70% of it
    float totalWidth = ImGui::GetContentRegionAvail().x;
    ImGui::Columns(2, nullptr, true);
    ImGui::SetColumnWidth(0, totalWidth * 0.7f);  // 70% for the first column (Face Preview)

    // --- First Column: Face Preview with tab bar ---
    if (ImGui::BeginChild("Previewer", ImVec2(0, 0), true)) {
        if (ImGui::BeginTabBar("FacePreviewTabs")) {
            if (ImGui::BeginTabItem("Model View")) {
                DrawModelView();  // Display model preview
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Render View")) {
                ImGui::Text("Render view content goes here...");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::EndChild();
    }

    ImGui::NextColumn();

    // --- Second Column: Manager Tabs or tools (30%) ---
    float buttonWidth = ImGui::GetColumnWidth();
    ImVec2 fullSize = ImVec2(buttonWidth - ImGui::GetStyle().ItemSpacing.x, 32);

    ImGui::BeginChild("AnimationModeChild", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysUseWindowPadding);

    ImGui::TextDisabled("Animation Mode");
    ImGui::Spacing();

    // Live Capture button
    if (animationMode == 0)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 1.0f, 1.0f));
    else
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_Button));

    if (ImGui::Button("Real Time Capture", fullSize))
        animationMode = 0;

    ImGui::PopStyleColor();
    ImGui::Spacing();

    // Preset Expression button
    if (animationMode == 1)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 1.0f, 1.0f));
    else
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_Button));

    if (ImGui::Button("Expression Control", fullSize))
        animationMode = 1;

    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Dummy(ImVec2(0.0f, 1.0f)); // thin divider-like spacing
    // --- Export Section Header ---
    ImGui::Spacing();
    ImGui::Separator();
    //ImGui::Dummy(ImVec2(0.0f, 6.0f));
    ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f), "Export Model");
    //ImGui::TextUnformatted("Export Model");
    //ImGui::Dummy(ImVec2(0.0f, 6.0f));
    ImGui::Spacing();

    // --- Choose Export Path Button ---
    ImVec2 buttonSize = ImVec2(fullSize.x, 0); // Fixed height for consistency
    float buttonX = (ImGui::GetContentRegionAvail().x - buttonSize.x) * 0.5f;
    if (buttonX > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + buttonX);

    bool canExport = (startFrame >= 0) && (endFrame > startFrame) && model.loaded;
    // Export Button
    ImGui::BeginDisabled(!canExport);  // Disable if condition not met
    if (ImGui::Button("Choose Export Path (.glb)", buttonSize))
    {
        IGFD::FileDialogConfig config;
        config.path = GetDesktopPath();
        config.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_ConfirmOverwrite;

        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseExport",
            "Select Export Location",
            ".glb",
            config
        );
    }
    ImGui::EndDisabled();  // End disabling block

    // Show export status message
    ImGui::Spacing();
    if (!canExport) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));  // Reddish warning color
        ImGui::TextWrapped("Export unavailable: Make sure the model is loaded and frame range is valid.");
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // Show selected export path or placeholder
    ImGui::PushStyleColor(ImGuiCol_Text, exportFilePath.empty() 
        ? ImVec4(1.0f, 0.6f, 0.4f, 1.0f)   // Orange-ish placeholder color
        : ImVec4(0.6f, 0.6f, 0.6f, 1.0f)); // Neutral gray for selected path
    ImGui::TextWrapped("%s", exportFilePath.empty() ? "< No export path selected >" : exportFilePath.c_str());
    ImGui::PopStyleColor();

    // --- Display Save Dialog ---
    if (ImGuiFileDialog::Instance()->Display("ChooseExport", ImGuiWindowFlags_NoCollapse, ImVec2(700, 400)))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            exportFilePath = ImGuiFileDialog::Instance()->GetFilePathName();
            exportStatusMsg = "";

            std::cout << "Selected Export File: " << exportFilePath << std::endl;
            std::cout << "Reconstruct Model Path: " << reconstructModelPath << std::endl;

            fs::path modelPath = reconstructModelPath;
            fs::path frameDir = modelPath.parent_path().parent_path() / "frames_model";

            std::cout << "Resolved Frame Directory: " << frameDir << std::endl;
            std::cout << "Start Frame: " << startFrame << ", End Frame: " << endFrame << std::endl;
            std::cout << "FPS: " << get_FPS() << std::endl;

            model.ExportModel(frameDir.string(), exportFilePath, get_FPS() ,startFrame, endFrame);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    /*
    // --- Export Now Button ---
    if (ImGui::Button("Export Now", ImVec2(140, 0)))
    {
        if (!exportFilePath.empty())
        {
            bool success = true; // Replace with model2.Export(exportFilePath);
            if (success)
                exportStatusMsg = "Export successful:\n" + exportFilePath;
            else
                exportStatusMsg = "Export failed!";
        }
        else
        {
            exportStatusMsg = "Please select an export path first.";
        }
    }

    // --- Show Export Status ---
    if (!exportStatusMsg.empty())
    {
        ImGui::SameLine();
        ImGui::TextWrapped("%s", exportStatusMsg.c_str());
    }

    // --- Tooltip (Only on Hovering "Export Now") ---
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Click to export the current model as a .glb file.");
    }*/

    // --- Padding & Divider ---
    //ImGui::Dummy(ImVec2(0.0f, 10.0f));
    ImGui::Separator();
    ImGui::Spacing();

    if (animationMode == 1)
    {
        bool expressionChanged = false;     // Track changes
    
        ImGui::TextDisabled("Expression Sliders");
        ImGui::Spacing();

        // Reset Button
        if (ImGui::Button("Reset All")) {
            for (int i = 0; i < 100; ++i)
                expressions[i] = 0.0f;
            expressionChanged = true;
        }
    
        float childHeight = ImGui::GetContentRegionAvail().y;
        ImGui::BeginChild("ExpressionControls", ImVec2(0, childHeight), true, ImGuiWindowFlags_AlwaysUseWindowPadding);

        // Optional: consistent fixed width for label area
        const float labelWidth = 120.0f;
        const float spacing = 12.0f;  // Space between label and slider
    
        for (int i = 0; i < 50; ++i)
        {
            std::string label = "Expression " + std::to_string(i + 1);
            ImGui::PushID(i);
    
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label.c_str());
            ImGui::SameLine(labelWidth + spacing);  // Apply consistent spacing
    
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
            if (ImGui::SliderFloat("##slider", &expressions[i], -1.0f, 1.0f))
                expressionChanged = true;
            ImGui::PopItemWidth();
    
            ImGui::PopID();
        }
    
        ImGui::EndChild();
    
        // Apply changes to model if any slider was touched
        if (expressionChanged)
        {
            model2.ExpressionControl(expressions);
        }
    }
    
    ImGui::EndChild();

    ImGui::Columns(1);  // Reset column layout
}

void DrawModelView() {
    ImGuiIO& io = ImGui::GetIO();

    ImVec2 viewportSize = ImGui::GetContentRegionAvail();
    static int fbWidth = 0, fbHeight = 0;
    if ((int)viewportSize.x != fbWidth || (int)viewportSize.y != fbHeight) {
        fbWidth = (int)viewportSize.x;
        fbHeight = (int)viewportSize.y;
        InitFramebuffer(fbWidth, fbHeight); // Resize FBO
    }

    // -- Render to FBO --
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, fbWidth, fbHeight);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)fbWidth / fbHeight, 0.1f, 100.0f);
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::rotate(modelMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    //static Shader shader("shaders/vertex.glsl", "shaders/fragment.glsl");
    //static Model model("Animation.glb");
    // Update morph animation
    //float time = glfwGetTime(); // or your time source
    //float time = ImGui::GetTime(); // or your time source
    //model.UpdateAnimation(time);
    
    shader.use();
    shader.setMat4("model", modelMatrix);
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);
    shader.setVec3("lightDir", glm::vec3(-1, -1, -1));
    shader.setVec3("lightColor", glm::vec3(1.0f));
    shader.setVec3("objectColor", glm::vec3(1.0f, 0.8f, 0.7f));
    shader.setVec3("viewPos", camera.Position);
    shader.setVec3("lightPos", camera.Position + camera.Front * 5.0f);  // dynamic lig
    shader.setInt("diffuseMap", 0);  // Bind to texture unit 0
    if(animationMode == 0){
        model.UpdateAnimationWithFrame(currentFrameIndex, alpha);
        model.Draw();
    }
    else{
        model2.Draw();
    }


    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // -- Display in ImGui inside column --
    ImGui::Image((ImTextureID)(intptr_t)colorTex, viewportSize, ImVec2(0, 1), ImVec2(1, 0)); // Flip vertically

    // Mouse controls
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 delta = ImGui::GetMouseDragDelta();
            camera.ProcessMouseMovement(delta.x, delta.y);
            ImGui::ResetMouseDragDelta();
        }

        // Only process scroll if hovered
        if (io.MouseWheel != 0.0f) {
            camera.ProcessMouseScroll(io.MouseWheel);
        }
    }    
}

void InitFramebuffer(int width, int height) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Create color texture
    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

    // Create depth renderbuffer
    glGenRenderbuffers(1, &depthRb);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRb);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Framebuffer is not complete!" << std::endl;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}