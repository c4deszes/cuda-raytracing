#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "raytracing/cuda_defs.cuh"
#include "raytracing/cuda_model.cuh"

extern "C" {
	#include "raytracing/wrapper.h"

	int bufferData(int bufferid, uint32_t offset, uint32_t length, size_t size, const void* newdata);
}
#include <stdio.h>

int genBuffer(uint8_t type, int length, size_t size);

#define _POINT_ARRAY 0
#define _NORMAL_ARRAY 1
#define _INDEX_ARRAY 2
#define _OBJECT 3
#define _MATERIAL 4
#define _LIGHT 5
#define _FRAME_BUFFER 6

#define MAX_BUFFERS 32
long long alloc_table[MAX_BUFFERS][4];


long long* dloc;
int framebuffer_id = -1;
int light_sim = 0;

__global__ void trace_ray(Camera camera, long long* alloc , uint32_t* framebuffer) {
	int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
	int yi = (blockIdx.y * blockDim.y) + threadIdx.y;
	

	Ray ray = {
		camera.origin,

		{ (xi - camera.width / 2.0) / 2.0 * 0.02, (yi - camera.height / 2.0) / 2.0 * 0.02, -10.0},
		//rotate(&camera.direction, (xi - camera.width/2.0f) / 400.0f, (yi - camera.height) / 400.0f)
	};
	Point hit;
	flt t;
	Triangle polygon;
	Vector3 ldir;
	Vector3 ndir;
	Color pixel = {0, 0, 0};
	Color lightness;
	Geometry* currentModel = NULL;

	boolean light_blocked = false;
	Ray light_cast;
	flt light_t;

	flt min = 10000000000000000.0f;
	int material_id = -1;

	for (int i = 0; i < MAX_BUFFERS;i++) {
		if (*(alloc + i * 4) != 0 && *(alloc + i * 4 + 1) == _OBJECT) {
			currentModel = getGeometryBuffer(i, alloc);
			if (mesh_intersection(currentModel, alloc, &ray, &t, &polygon) && t > 0 && t < min) {
				Vector3 normal = triangle_normal(&polygon);
				flt dot = vect_dot(&ray.direction, &normal);
				if (dot > 0) {
					material_id = currentModel->mat_id;
					min = t;
				}
			}
		}
	}

	if (material_id != -1) {
		//point at which the intersection was found
		Vector3 loc = vect_mul(&(ray.direction), min);
		hit = vect_add(&loc, &ray.origin);
		pixel = getMaterialBuffer(material_id, alloc)->diffuse;

		//now for every light source test this point
		for (int q = 0; q < MAX_BUFFERS; q++) {
			Light* currentLight = NULL;
			if (*(alloc + q * 4) != 0 && *(alloc + q * 4 + 1) == _LIGHT) {
				currentLight = getLightBuffer(q, alloc);
				ldir = vect_sub(&hit, &(currentLight->pos));

				light_cast.origin = hit;
				light_cast.direction = vect_mul(&ldir, -1);

				light_blocked = false;

				Geometry* currentModel2;
				Triangle tri;
				for (int m = 0; m < MAX_BUFFERS; m++) {
					if (*(alloc + m * 4) != 0 && *(alloc + m * 4 + 1) == _OBJECT) {
						currentModel2 = getGeometryBuffer(m, alloc);
						if (mesh_intersection(currentModel2, alloc, &light_cast, &light_t, &tri) && light_t < 1.0) {
							light_blocked = true;
							pixel = { 0,0,0 };
						}
					}
				}


				if (!light_blocked) {
					double intensity = currentLight->intensity / vect_len(&ldir);
					if (intensity > 0 && vect_len(&ldir) != 0) {
						ldir = vect_norm(&ldir);
						ndir = triangle_normal(&polygon);
						ndir = vect_norm(&ndir);

						lightness = color_mulc(&(currentLight->color), vect_dot(&ldir, &ndir) * intensity);
						pixel = color_mul(&pixel, &lightness);
					}
				}
			}
		}
	}
	//oversaturation detection
	flt m = fmaxf(fmaxf(pixel.r, pixel.g), pixel.b);
	if (m < 1.0f) {
		m = 1.0f;
	}
	*(framebuffer + yi * camera.width + xi) = ((uint8_t)(pixel.r / m * 255.0f)) << 16 | ((uint8_t)(pixel.g / m * 255.0f)) << 8 | ((uint8_t)(pixel.b / m * 255.0f));
}

dim3 threads(8, 8);

Light up_light = {
	-1,
	{ -100, 200, 400 },
	{ 1, 1, 1 },
	400,
	POINT_LIGHT,
	{ 0,0,0 }
};

cudaError_t render_scene(Camera camera, uint32_t* frame) {
	cudaError_t status;
	dim3 blocks(camera.width / threads.x, camera.height / threads.y);

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	trace_ray <<< blocks, threads >>> (camera, dloc, (uint32_t*)alloc_table[framebuffer_id][2]);
	status = cudaDeviceSynchronize();	

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Render took: %f sec\n", time/1000.0);

	status = cudaMemcpyAsync(frame, (void*)alloc_table[framebuffer_id][2], camera.width * camera.height * sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);
	if (status != cudaSuccess) {
		printf("Error code %d after copying framebuffer!\n", status);
		goto error_handler;
	}

	//bufferData(light_sim, 0, 1, sizeof(Light), &up_light);
	//up_light.pos.x += 10;

	error_handler:

	return status;
}

cudaError_t delete_scene() {
	cudaError_t status;
	for (int i = 0; i < MAX_BUFFERS; i++) {
		if (alloc_table[i][0] != 0) {
			status = cudaFree((void*)alloc_table[i][2]);
			if (status != cudaSuccess) {
				printf("Error code %d after freeing buffer: %lld, type: %lld!\n", status, alloc_table[i][0], alloc_table[i][1]);
				goto error_handler;
			}
		}
	}

	status = cudaFree(dloc);
	if (status != cudaSuccess) {
		printf("Error code %d after freeing allocation table!\n", status);
		goto error_handler;
	}

	printf("Scene deleted.\n");

error_handler:

	return status;
}

cudaError_t copy_scene(Camera camera, List_Mesh* ofirst, List_Material* mfirst, List_Light* lfirst) {

	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	if (!nDevices) {
		printf("No CUDA capable GPU was found.\n");
		return cudaErrorDevicesUnavailable;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device name: %s\n", prop.name);

	//copy objects and light source to GPU buffers
	cudaError_t status;

	status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		printf("CudaSetDevice failed!\n");
		goto error_handler;
	}

	framebuffer_id = genBuffer(_FRAME_BUFFER, camera.width * camera.height, sizeof(uint32_t));
	printf("Generated Framebuffer id: %d start: %lld \n", framebuffer_id, alloc_table[framebuffer_id][2]);

	status = cudaMalloc((void**)&dloc, MAX_BUFFERS * 4 * sizeof(long long));

	while (mfirst != NULL) {
		int mat_id = genBuffer(_MATERIAL, 1, sizeof(Material));
		bufferData(mat_id, 0, 1, sizeof(Material), mfirst->mat);
		mfirst->mat->buffer_id = mat_id;
		mfirst = mfirst->next;
	}

	while (lfirst != NULL) {
		int light_id = genBuffer(_LIGHT, 1, sizeof(Light));
		light_sim = light_id;
		bufferData(light_id, 0, 1, sizeof(Light), lfirst->light);
		lfirst->light->buffer_id = light_id;
		lfirst = lfirst->next;
	}

	while (ofirst != NULL) {
		int normal_id = -1;
		int point_id = genBuffer(_POINT_ARRAY, ofirst->mesh->points_length, sizeof(Point));
		bufferData(point_id, 0, ofirst->mesh->points_length, sizeof(Point), ofirst->mesh->points);
		ofirst->mesh->point_buffer_id = point_id;

		if (ofirst->mesh->normals != NULL) {
			normal_id = genBuffer(_NORMAL_ARRAY, ofirst->mesh->points_length, sizeof(Vector3));
			bufferData(normal_id, 0, ofirst->mesh->points_length, sizeof(Vector3), ofirst->mesh->normals);
			ofirst->mesh->point_buffer_id = point_id;
		}

		int index_id = genBuffer(_INDEX_ARRAY, ofirst->mesh->indices_length, sizeof(int));
		bufferData(index_id, 0, ofirst->mesh->indices_length, sizeof(int), ofirst->mesh->index);
		ofirst->mesh->point_buffer_id = point_id;
		
		int mesh_id = genBuffer(_OBJECT, 6, sizeof(int));
		int geometry[6] = {ofirst->mesh->points_length, point_id, normal_id, ofirst->mesh->indices_length, index_id, ofirst->mesh->mat->buffer_id};
		bufferData(mesh_id, 0, 6, sizeof(int), geometry);
		ofirst->mesh->mesh_id = mesh_id;
		
		//printf("Mesh buffers: %d, %d, %d\n", point_id, index_id, normal_id);

		ofirst = ofirst->next;
	}

	status = cudaMemcpy(dloc, alloc_table, MAX_BUFFERS * 4 * sizeof(long long), cudaMemcpyHostToDevice);

	//printf("Starting render job.\n");

	status = cudaGetLastError();
	if (status != cudaSuccess) {
		printf("Render job failed: %s\n", cudaGetErrorString(status));
		goto error_handler;
	}
	
	printf("Scene copied.\n");
	return cudaSuccess;

error_handler:

	
	return status;
}

int genBuffer(uint8_t type, int length, size_t size) {
	cudaError_t status = cudaSuccess;
	int id = -1;
	for (int i = 0; i < MAX_BUFFERS; i++) {
		if (alloc_table[i][0] == 0) {
			id = i;
			break;
		}
	}
	if (id != -1 && id < MAX_BUFFERS) {
		status = cudaMalloc((void**)&alloc_table[id][2], length * size);
		if (status != cudaSuccess) {
			printf("Out of memory exception!\n");
			return -1;
		}
		alloc_table[id][0] = 0xFF;
		alloc_table[id][1] = type;
		alloc_table[id][3] = alloc_table[id][2] + length*size;
	}
	else {
		printf("Buffer allocation table overflow!\n");
		return -1;
	}
	return id;
}

int deleteBuffer(int bufferid) {
	if (alloc_table[bufferid][0] != 0) {
		cudaError_t status = cudaFree((void*)alloc_table[bufferid][2]);
		alloc_table[bufferid][0] = 0;
		if (status != cudaSuccess) {
			printf("Can't delete buffer: %d", bufferid);
			return -1;
		}
		return 1;
	}
	return 0;
}

int bufferData(int bufferid, uint32_t offset, uint32_t length, size_t size, const void* newdata) {
	if (alloc_table[bufferid][2] + offset + length * size > alloc_table[bufferid][3]) {
		return -1;
	}
	cudaError_t status = cudaMemcpy((void*)(alloc_table[bufferid][2] + offset), newdata, length * size, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		return -1;
	}
	return 1;
}