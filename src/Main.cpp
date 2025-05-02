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

// Global or class-level flag and thread
std::thread pythonThread;
std::atomic<bool> shouldRunPython(false);
std::string pendingPythonPath;
bool pythonThreadRunning = false;



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

void EnableDarkTitleBar(HWND hwnd) {
    BOOL value = TRUE;
    DwmSetWindowAttribute(hwnd, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/, &value, sizeof(value));
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void RenderFacialReconstructionTab();

GLuint LoadTextureFromFile(const char* filename)
{
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 4);
    if (data == NULL)
    {
        std::cout << "Failed to load image: " << filename << std::endl;
        return 0;
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);

    return textureID;
}

void runPythonTest() {
    try {
        py::print("Testing Python from C++");
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
    }
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


int main()
{
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
                        RenderFacialReconstructionTab();
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

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Python initialization error: " << e.what() << std::endl;
        return -1;
    }
}

void RenderFacialReconstructionTab()
{
    //py::gil_scoped_acquire acquire;

    static GLuint imageTexture = 0; // <--- static, so it loads only once

    if (imageTexture == 0)
    {
        imageTexture = LoadTextureFromFile("E:/Project/DECA3/DECA/TestSamples/examples/results/IMG_0392_inputs/IMG_0392_inputs_inputs.jpg"); // <-- change path here
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
                        ImGui::Image((ImTextureID)(intptr_t)imageTexture, availableSize);
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
                        ImGui::Text("Facial Tracking Content");
                        ImGui::Image((ImTextureID)789, ImVec2(-1, -1)); // Placeholder image
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
    
        static std::string selectedFilePath;
    
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




