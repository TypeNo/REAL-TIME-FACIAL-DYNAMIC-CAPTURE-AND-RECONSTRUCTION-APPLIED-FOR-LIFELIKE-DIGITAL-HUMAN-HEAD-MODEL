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
//#include "stb_image.h"

void EnableDarkTitleBar(HWND hwnd) {
    BOOL value = TRUE;
    DwmSetWindowAttribute(hwnd, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/, &value, sizeof(value));
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void RenderFacialReconstructionTab();

int main()
{
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
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
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
}

void RenderFacialReconstructionTab()
{
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
                        ImGui::Image((ImTextureID)123, ImVec2(-1, -1)); // Placeholder image
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
        ImGui::Text("Capture Source");

        static int captureSourceType = 0;
        ImGui::Combo("Capture Source Type", &captureSourceType, "Offline\0Live\0");

        static char captureSourcePath[256] = "C:/Document/Clip1";
        ImGui::InputText("Capture Source", captureSourcePath, IM_ARRAYSIZE(captureSourcePath));

        static int startFrame = 0, endFrame2 = 464;
        ImGui::InputInt("Start Frame to Process", &startFrame);
        ImGui::InputInt("End Frame to Process", &endFrame2);
    }
    ImGui::EndChild();
}




