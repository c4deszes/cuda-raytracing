cmake_minimum_required(VERSION 3.20)
project(cuda-raytracing LANGUAGES C CXX CUDA)

add_subdirectory(lib/glfw)
add_subdirectory(lib/glad)
find_package(OpenGL REQUIRED)

add_subdirectory(cpu)

add_executable(raytracing
	src/main.c
	src/cuda/cuda_defs.cu
	src/cuda/cuda_model.cu
	src/cuda/raytracer.cu
)
target_link_libraries(raytracing PRIVATE glfw glad OpenGL::GL)
set_target_properties(raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(raytracing PUBLIC include)
set_property(TARGET raytracing PROPERTY CUDA_ARCHITECTURES "35;50;72")
target_compile_features(raytracing PUBLIC cxx_std_11)
