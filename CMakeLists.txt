# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.15)

project(FYP)

# Set the C++ standard version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake setup for local vcpkg installation
set(CMAKE_TOOLCHAIN_FILE "E:/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Enable verbose makefile output for debugging (optional)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set output directory for compiled binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/ImGuiFileDialog-master)

# Gather source files
file(GLOB SRC_FILES
    src/*.cpp
)

# Create the executable
add_executable(FYP ${SRC_FILES} ${EXTERNAL_DIR}/ImGuiFileDialog.cpp)


#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/share")
# Link external libraries installed via vcpkg
# Add libraries you installed with vcpkg (e.g., glfw3, imgui, pybind11, etc.)

# Find vcpkg packages
# set(glfw3_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/share/glfw3" CACHE PATH "Path to glfw3 config")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/share" ${CMAKE_PREFIX_PATH})
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)


# Set the paths to the Conda Python include directory and library
set(Python3_EXECUTABLE "E:/anaconda3/envs/pytorch3d/python")
set(Python3_INCLUDE_DIR "E:/anaconda3/envs/pytorch3d/include")
set(Python3_LIBRARY "E:/anaconda3/envs/pytorch3d/libs/python311.lib")

include_directories(${Python3_INCLUDE_DIR})
link_directories(${Python3_LIBRARY})


# Find Python from the Conda environment
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED)




# Link libraries (vcpkg will handle these automatically)
target_link_libraries(FYP imgui glfw IGFD ${Python3_LIBRARIES})

# OpenGL setup for macOS
if(APPLE)
    target_link_libraries(FYP "-framework OpenGL")
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(FYP ${OPENGL_gl_LIBRARY})
endif()

# Copy GLFW DLL to output directory (for dynamic linking)
if(WIN32)
    find_file(GLFW_DLL NAMES glfw3.dll PATHS "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/bin")
    if(GLFW_DLL)
        add_custom_command(TARGET FYP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLFW_DLL}"
            "$<TARGET_FILE_DIR:FYP>")
    endif()
endif()