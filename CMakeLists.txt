cmake_minimum_required(VERSION 3.15)

project(FYP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "E:/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/share" ${CMAKE_PREFIX_PATH})
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------- Python Setup ----------
# Point to your Anaconda environment
set(Python3_ROOT_DIR "E:/anaconda3/envs/pytorch3d")
set(Python3_INCLUDE_DIR "E:/anaconda3/envs/pytorch3d/include")
set(Python3_LIBRARIES "E:/anaconda3/envs/pytorch3d/libs/python311.lib")

include_directories(${Python3_INCLUDE_DIR})
link_directories("E:/anaconda3/envs/pytorch3d/libs")

# Python and pybind11
find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# ---------- External Libraries ----------
# ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vcpkg_installed/vcpkg/blds/imgui/src/v1.90.2-4442117b09.clean)
add_library(imgui
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp 
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/include
    ${IMGUI_DIR}/examples/libs/glfw/include
)

# ImGuiFileDialog
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
add_library(IGFD STATIC
    ${EXTERNAL_DIR}/ImGuiFileDialog.cpp
    ${EXTERNAL_DIR}/ImGuiFileDialog.h
    ${EXTERNAL_DIR}/ImGuiFileDialogConfig.h
)
target_include_directories(IGFD PUBLIC
    ${EXTERNAL_DIR}
    ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/include
)

# ----------------------------------------------------------------------------
# Shared Progress Logic (both app and Python module will link to this)
add_library(progress_shared SHARED 
    src/progress_shared.cpp 
)
target_include_directories(progress_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(progress_shared PRIVATE PROGRESS_SHARED_EXPORTS)

# ---------- Source Files ----------
file(GLOB SRC_FILES
    src/*.cpp
    src/*.hpp
)

add_executable(FYP ${SRC_FILES} ${EXTERNAL_DIR}/ImGuiFileDialog.cpp)


# ---------- pybind11 Module ----------
pybind11_add_module(mybridge src/mybridge.cpp)
target_link_libraries(mybridge PRIVATE progress_shared)

# ---------- Dependencies ----------
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(FYP
    imgui
    glfw
    IGFD
    ${Python3_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    #progress_shared
)

#set_target_properties(mybridge PROPERTIES PREFIX "" SUFFIX ".pyd") # for Windows

# ---------- DLL Copy for Windows ----------
if(WIN32)
    find_file(GLFW_DLL NAMES glfw3.dll PATHS "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin")
    if(GLFW_DLL)
        add_custom_command(TARGET FYP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLFW_DLL}"
            "$<TARGET_FILE_DIR:FYP>"
        )
    endif()
endif()

# ---------- Messages for Debugging ----------
message(STATUS "Python3 Include Dir: ${Python3_INCLUDE_DIR}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")
message(STATUS "pybind11 Dir: ${pybind11_DIR}")
