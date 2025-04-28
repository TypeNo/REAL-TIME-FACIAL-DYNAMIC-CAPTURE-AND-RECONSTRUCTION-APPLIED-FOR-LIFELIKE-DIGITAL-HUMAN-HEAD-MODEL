# Install script for directory: E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/src/v2.11.1-9a1aec74ca.clean

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/pkgs/pybind11_x64-windows/debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "OFF")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/src/v2.11.1-9a1aec74ca.clean/include/pybind11")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11" TYPE FILE FILES
    "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/x64-windows-dbg/pybind11Config.cmake"
    "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/x64-windows-dbg/pybind11ConfigVersion.cmake"
    "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/src/v2.11.1-9a1aec74ca.clean/tools/FindPythonLibsNew.cmake"
    "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/src/v2.11.1-9a1aec74ca.clean/tools/pybind11Common.cmake"
    "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/src/v2.11.1-9a1aec74ca.clean/tools/pybind11Tools.cmake"
    "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/src/v2.11.1-9a1aec74ca.clean/tools/pybind11NewTools.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/pybind11Targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/pybind11Targets.cmake"
         "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/x64-windows-dbg/CMakeFiles/Export/890f2caa8fc1b4df326ed06c7ff9ddbd/pybind11Targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/pybind11Targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/pybind11Targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11" TYPE FILE FILES "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/x64-windows-dbg/CMakeFiles/Export/890f2caa8fc1b4df326ed06c7ff9ddbd/pybind11Targets.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" TYPE FILE FILES "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/x64-windows-dbg/pybind11.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "E:/Project/DECA3/DECA/vcpkg_installed/vcpkg/blds/pybind11/x64-windows-dbg/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
