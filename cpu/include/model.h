#pragma once
#include "defs.h"

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

	//resolution
	int width;
	int height;
	int color_format;

} Camera;

/* MATERIAL */

typedef struct Material {
	int buffer_id;

	Color diffuse;
	double diffuse_amount;

	Color reflect;
	double reflection_amount;

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
	double intensity;
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
	int points_length;

	int point_buffer_id;		//GPU
	Point* points;

	int normal_buffer_id;		//GPU
	Vector3* normals;

	int indices_length;

	int index_buffer_id;		//GPU
	int* index;

	int mat_buffer_id;
	Material* mat;

} Mesh;

typedef struct List_Mesh {
	Mesh* mesh;
	struct List_Mesh* next;
} List_Mesh;

/* TRIANGLES */

typedef struct Triangle {
	Point p1, p2, p3;
	Vector3 n1, n2, n3;
} Triangle;

int mesh_intersection(const Mesh* mesh, const Ray* ray, double* out, Triangle* out2);
int triangle_intersection(const Triangle* triangle, const Ray* ray, double* out);
Vector3 triangle_normal(const Triangle* triangle);