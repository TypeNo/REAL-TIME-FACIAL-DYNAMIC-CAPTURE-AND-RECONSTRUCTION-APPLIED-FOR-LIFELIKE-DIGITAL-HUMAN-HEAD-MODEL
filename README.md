# 🧠 REAL TIME FACIAL DYNAMIC CAPTURE AND RECONSTRUCTION FOR LIFELIKE DIGITAL HUMAN HEAD MODEL

## ⚙️ Requirements

- **CUDA Toolkit 12.1.0**
  > 🔸 Ensure your GPU supports this version.  
  > 🔸 [Check compatibility here](https://developer.nvidia.com/cuda-gpus)

---

## 🚀 Getting Started

### 1️⃣ Clone This Repository & Set Up vcpkg

```bash
git clone https://github.com/TypeNo/REAL-TIME-FACIAL-DYNAMIC-CAPTURE-AND-RECONSTRUCTION-APPLIED-FOR-LIFELIKE-DIGITAL-HUMAN-HEAD-MODEL.git
cd REAL-TIME-FACIAL-DYNAMIC-CAPTURE-AND-RECONSTRUCTION-APPLIED-FOR-LIFELIKE-DIGITAL-HUMAN-HEAD-MODEL

# Clone and bootstrap vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh       # Use .\bootstrap-vcpkg.bat on Windows
cd ..
```

---

### 2️⃣ Configure `vcpkg-configuration.json` Baseline

1. Visit 👉 [vcpkg GitHub](https://github.com/microsoft/vcpkg)
2. Copy the latest **commit hash** from the top of the commit history.
3. Open `vcpkg-configuration.json` and update the `baseline` field:

**Example:**

Before:
```json
"baseline": "7f9f0e44db287e8e67c0e888141bfa200ab45121"
```

After:
```json
"baseline": "your_new_commit_hash_here"
```

---

### 3️⃣ Install Dependencies via vcpkg

```bash
# Installs all dependencies listed in vcpkg.json manifest
./vcpkg/vcpkg install --triplet x64-windows   # or x64-linux / x64-osx as needed
```

---

### 4️⃣ Python Environment Setup (Anaconda)

```bash
# Create and activate conda environment from lock file
conda env create -f environment.lock.yml
conda activate pytorch3d
```

📦 Some dependencies not on PyPI must be installed manually:
- [Install PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

---

### 5️⃣ Configure `CMakeLists.txt`

#### 🔹 Set the vcpkg Toolchain File Path

```cmake
# Modify this line to match your local vcpkg location:
set(CMAKE_TOOLCHAIN_FILE "E:/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
```

#### 🔹 Set the Python Environment Root

```cmake
# Set to your local conda environment path:
set(Python3_ROOT_DIR "E:/anaconda3/envs/pytorch3d")
find_package(Python3 3.11 EXACT REQUIRED COMPONENTS Interpreter Development)
```

#### 🔹 ImGui Linking Options

If you're using ImGui from vcpkg:
```cmake
# Recommended: use the installed version from vcpkg
find_package(imgui CONFIG REQUIRED)
```

If manually linking from source:
```cmake
# Warning: This path points to a temporary vcpkg build directory
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vcpkg_installed/vcpkg/blds/imgui/src/v1.90.2-4442117b09.clean)
```

---

### 6️⃣ Configure & Build the Project

```bash
# Configure with CMake (using vcpkg toolchain)
# If you cloned vcpkg inside the project as shown above, DCMAKE_TOOLCHAIN_FILE should point to that local vcpkg
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake

# Build the project
cmake --build build
```

---

✅ You're ready to go! Launch the application from the `build/bin` directory.
