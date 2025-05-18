cmake_minimum_required(VERSION 3.15)

# Vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "E:/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
project(FYP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------- Python Setup ----------
set(Python3_ROOT_DIR "E:/anaconda3/envs/pytorch3d")
find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Interpreter Development)

include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# Python + pybind11
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
target_include_directories(IGFD PUBLIC ${EXTERNAL_DIR} ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/include)

# ---------- Shared Library ----------
add_library(progress_shared SHARED src/progress_shared.cpp)
target_include_directories(progress_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(progress_shared PRIVATE PROGRESS_SHARED_EXPORTS)

# ---------- Main Executable ----------
file(GLOB SRC_FILES src/*.cpp src/*.hpp)
add_executable(FYP ${SRC_FILES})
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

target_link_libraries(FYP
    imgui
    IGFD
    glfw
    glad::glad
    glm::glm
    assimp::assimp
    ${Python3_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    #progress_shared
)

# Define VCPKG root (adjust if it's outside your project)
set(VCPKG_BIN_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin")
set(VCPKG_BIN_DIR_DEBUG "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/debug/bin")

# Copy all DLLs based on build type
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    file(GLOB DLLS_TO_COPY "${VCPKG_BIN_DIR_DEBUG}/*.dll")
else()
    file(GLOB DLLS_TO_COPY "${VCPKG_BIN_DIR}/*.dll")
endif()

# Copy all DLLs to the target output dir
foreach(DLL ${DLLS_TO_COPY})
    add_custom_command(TARGET FYP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL}"
        $<TARGET_FILE_DIR:FYP>
    )
endforeach()

# ---------- pybind11 Module ----------
pybind11_add_module(mybridge src/mybridge.cpp)
target_include_directories(mybridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mybridge PRIVATE progress_shared pybind11::module)

# ---------- Windows DLL Copy ----------
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

# ---------- Debug Messages ----------
message(STATUS "Python3 Include Dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")
message(STATUS "pybind11 Dir: ${pybind11_DIR}")
