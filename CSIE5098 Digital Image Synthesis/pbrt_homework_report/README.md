# CSIE5098 Digital Image Synthesis - pbrt-v4 Installation

**Student:** James Christian (R13921031)  
**Date:** February 11, 2026  
**Course:** CSIE5098 Digital Image Synthesis

## 📋 Overview

This repository contains the complete documentation of my pbrt-v4 installation and setup process for the CSIE5098 homework assignment.

## 📁 Repository Structure

```
pbrt_homework_report/
├── pbrt_installation_report.html    # Complete installation report
├── images/
│   ├── killeroo-simple.png          # Test render 1
│   └── lte-orb-simple-ball.png      # Test render 2
└── README.md                         # This file
```

## ✅ Objectives Completed

- ✓ Installed pbrt-v4 from source on Ubuntu Linux (no sudo access)
- ✓ Resolved 5 major technical challenges (OpenGL headers, ABI compatibility, POSIX definitions, library linking, Wayland support)
- ✓ Successfully built all executables (pbrt, imgtool, pbrt_test)
- ✓ Downloaded and tested pbrt-v4-scenes (7.05 GB, 11,140 files)
- ✓ Rendered test scenes successfully
- ✓ Configured debugging environment with GDB
- ✓ Created comprehensive HTML documentation

## 🖼️ Rendered Images

### Test Render 1: Killeroo Simple
- **Scene:** killeroo-simple.pbrt
- **Render Time:** 2.9 seconds
- **Output:** 1.6 MB EXR → 597 KB PNG

![Killeroo Simple](images/killeroo-simple.png)

### Test Render 2: LTE Orb Simple Ball
- **Scene:** lte-orb-simple-ball.pbrt
- **Render Time:** 65.6 seconds
- **Output:** 23 MB EXR → 1.6 MB PNG

![LTE Orb Simple Ball](images/lte-orb-simple-ball.png)

## 🖥️ System Configuration

- **GPU:** 8x NVIDIA H200 (143GB VRAM each)
- **CUDA:** 12.4
- **OS:** Ubuntu Linux
- **Python:** 3.12 (Conda environment)
- **Compiler:** GCC (conda-forge gxx_linux-64)

## 🔧 Key Technical Challenges Solved

1. **Missing OpenGL Headers** - Used conda OpenGL libraries with system headers
2. **C++ ABI Compatibility** - Installed complete conda compiler toolchain
3. **GLFW POSIX Definitions** - Added `-D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE`
4. **Missing libGL.so Symlink** - Direct path to libGL.so.1
5. **Wayland Scanner** - Disabled Wayland, used X11

## 📄 Documentation

Open `pbrt_installation_report.html` in any web browser to view the complete installation report with:
- Detailed installation steps with all commands
- Challenge-solution documentation
- Build verification
- Testing results with rendered images
- Debugging setup instructions

## 🚀 Quick Start

To reproduce this installation:

```bash
# 1. Clone pbrt-v4
git clone --recursive https://github.com/mmp/pbrt-v4.git
cd pbrt-v4

# 2. Create conda environment
conda create -n pbrt_project python=3.12
conda activate pbrt_project

# 3. Install dependencies
conda install -c conda-forge cmake mesa-libgl-devel-cos7-x86_64 \
    mesa-dri-drivers-cos7-x86_64 xorg-libx11 xorg-libxext \
    xorg-libxrender mesa-libgl-cos7-x86_64 libglvnd-glx-cos7-x86_64 \
    openexr doxygen xorg-libxrandr xorg-libxinerama \
    xorg-libxcursor xorg-libxi gxx_linux-64

# 4. Configure and build
mkdir build && cd build
CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
CFLAGS="-D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE" \
cmake -D OpenGL_GL_PREFERENCE=LEGACY \
      -D OPENGL_INCLUDE_DIR=/usr/include \
      -D OPENGL_gl_LIBRARY=/usr/lib/x86_64-linux-gnu/libGL.so.1 \
      -D GLFW_BUILD_WAYLAND=OFF ..
make -j8

# 5. Test installation
./pbrt --help
```

## 🔗 References

- [pbrt-v4 GitHub](https://github.com/mmp/pbrt-v4)
- [pbrt-v4-scenes Repository](https://github.com/mmp/pbrt-v4-scenes)
- [Physically Based Rendering Book](https://www.pbrt.org/)

## 📝 License

This documentation is created for academic purposes as part of CSIE5098 coursework.