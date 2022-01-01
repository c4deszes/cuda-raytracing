#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_defs.cuh"

typedef struct Ray {
	Point origin;
	Vector3 direction;
} Ray;

/*
FOV -> field of horizontal view in rads
Aspect Ratio -> determines the field of vertical view
*/

typedef struct Camera {
	Point origin;
	Vector3 direction;
	double fov;
	double aspect_ratio;

	Color background;

	//resolution
	int width;
	int height;
	int color_format;

} Camera;

/* MATERIAL */

typedef struct Material {
	int buffer_id;

	Color diffuse;
	flt diffuse_amount;

	Color reflect;
	flt reflection_amount;

} Material;

typedef struct List_Material {
	Material* mat;
	struct List_Material* next;
} List_Material;

/* LIGHT OBJECT	*/

#define POINT_LIGHT 0
#define SPOT_LIGHT 1

typedef struct Light {
	int buffer_id;

	Point pos;
	Color color;
	flt intensity;
	int type;
	Vector3 dir;
} Light;

typedef struct List_Light {
	Light* light;
	struct List_Light* next;
} List_Light;

void light_add(List_Light* list, Light* light);

/* OBJECTS */

typedef struct Mesh {
	int mesh_id;
	int points_length;

	int point_buffer_id;		//GPU
	Point* points;

	int normal_buffer_id;		//GPU
	Vector3* normals;

	int indices_length;

	int index_buffer_id;		//GPU
	int* index;

	Material* mat;

} Mesh;

typedef struct List_Mesh {
	Mesh* mesh;
	struct List_Mesh* next;
} List_Mesh;

typedef struct Geometry {
	int points;
	int point_id;
	int normal_id;
	int indices;
	int index_id;
	int mat_id;
} Geometry;

/* TRIANGLES */

typedef struct Triangle {
	Point p1, p2, p3;
	Vector3 n1, n2, n3;
} Triangle;

//int mesh_intersection(const Mesh* mesh, const Ray* ray, double* out, Triangle* out2);
//int triangle_intersection(const Triangle* triangle, const Ray* ray, double* out);
//Vector3 triangle_normal(const Triangle* triangle);

__device__ int triangle_intersection(Triangle* triangle, Ray* ray, flt* out);
__device__ int mesh_intersection(Geometry* mesh, long long* alloc, Ray* ray, flt* out, Triangle* out2);
__device__ Vector3 triangle_normal(Triangle* triangle);

__device__ Light* getLightBuffer(int bufferid, long long* alloc);
__device__ Material* getMaterialBuffer(int bufferid, long long* alloc);
__device__ Geometry* getGeometryBuffer(int bufferid, long long* alloc);

__device__ Point* getPointBuffer(int bufferid, long long* alloc);
__device__ Vector3* getNormalBuffer(int bufferid, long long* alloc);
__device__ int* getIndexBuffer(int bufferid, long long* alloc);