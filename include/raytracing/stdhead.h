#pragma once

#define _WINDOWS
//#define _LINUX

//#define PREFER_DEDICATED_GPU
//#define _OPENGL_EXT

#ifdef _WINDOWS
#define WIN32_LEAN_AND_MEAN 1
#define _WINSOCKAPI_
#define _CRT_SECURE_NO_WARNINGS
#define _CRTDBG_MAP_ALLOC 
#include <crtdbg.h>
#include <Windows.h>
#endif

#include <gl/GL.h>
#include "GLFW/glfw3.h"

//Use of GL13 and above
#ifdef _OPENGL_EXT
#include <glext.h>
#include <glxext.h>
#include <wglext.h>
#include <glcorearb.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

// documented in NVidia's Optimus specification
// http://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
//#define PREFER_DEDICATED_GPU
#ifdef PREFER_DEDICATED_GPU
_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
#endif



// TODO: reference additional headers your program requires here
