#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

extern cudaError_t render_scene(Camera camera, uint32_t* framebuffer);
extern cudaError_t delete_scene();
extern cudaError_t copy_scene(Camera camera, List_Mesh* ofirst, List_Material* mfirst, List_Light* lfirst);

extern void translate(Point* points, int length, Vector3 v);
extern int bufferData(int bufferid, uint32_t offset, uint32_t length, size_t size, const void* newdata);