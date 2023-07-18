add_executable(cpu-raytracing
    src/defs.c
    src/main.c
    src/model.c
)
target_include_directories(cpu-raytracing PRIVATE include)
target_link_libraries(cpu-raytracing PRIVATE glfw glad OpenGL::GL)
