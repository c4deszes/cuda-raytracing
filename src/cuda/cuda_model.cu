#include "raytracing/cuda_defs.cuh"
#include "raytracing/cuda_model.cuh"

#define EPSILON 0.001

extern "C" void translate(Point* points, int length, Vector3 v) {
	for (int i = 0; i < length; i++) {
		points[i].x += v.x;
		points[i].y += v.y;
		points[i].z += v.z;
	}
}

__device__ int triangle_intersection(Triangle* triangle, Ray* ray, flt* out) {
	Vector3 e1, e2;  //Edge1, Edge2
	Vector3 P, Q, T;
	flt det, inv_det, u, v;
	flt t;

	//Find vectors for two edges sharing V1
	e1 = vect_sub(&(triangle->p2), &(triangle->p1));

	e2 = vect_sub(&(triangle->p3), &(triangle->p1));

	//Begin calculating determinant - also used to calculate u parameter
	//CROSS(P, D, e2);
	P = vect_cross(&(ray->direction), &e2);
	//if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = vect_dot(&e1, &P);
	//NOT CULLING
	if (det > -EPSILON && det < EPSILON) {
		return 0;
	}
	inv_det = 1.0 / det;

	//calculate distance from V1 to ray origin
	//SUB(T, O, V1);
	T.x = ray->origin.x - triangle->p1.x;
	T.y = ray->origin.y - triangle->p1.y;
	T.z = ray->origin.z - triangle->p1.z;

	//Calculate u parameter and test bound
	u = vect_dot(&T, &P) * inv_det;
	//The intersection lies outside of the triangle
	if (u < 0.0 || u > 1.0) {
		return 0;
	}

	//Prepare to test v parameter
	Q = vect_cross(&T, &e1);

	//Calculate V parameter and test bound
	v = vect_dot(&(ray->direction), &Q) * inv_det;
	//The intersection lies outside of the triangle
	if (v < 0.0 || u + v  > 1.0) {
		return 0;
	}

	t = vect_dot(&e2, &Q) * inv_det;

	if (t > EPSILON) { //ray intersection
		*out = t;
		return 1;
	}

	// No hit, no win
	return 0;
}

__device__ int mesh_intersection(Geometry* mesh, long long* loc,  Ray* ray, flt* out, Triangle* out2) {
	flt min = 10000000000000;
	flt max = -10000000000000;
	flt t;
	Triangle temp;
	Triangle first;

	/* GEOMETRY
	0 POINT SIZE
	1 POINT ARRAY ID
	2 NORMAL ARRAY ID
	3 INDEX SIZE
	4 INDEX ARRAY ID
	5 MATERIAL ID
	*/

	int indices = mesh->indices;
	int* index_buffer = getIndexBuffer(mesh->index_id, loc);
	Point* point_buffer = getPointBuffer(mesh->point_id, loc);

	for (int i = 0; i < indices - 2; i += 3) {
		temp.p1 = point_buffer[index_buffer[i]];
		temp.p2 = point_buffer[index_buffer[i + 1]];
		temp.p3 = point_buffer[index_buffer[i + 2]];
		if (triangle_intersection(&temp, ray, &t)) {
			if (t < min) {
				min = t;
				first = temp;
			}
			if (t > max) {
				max = t;
				//first = temp;
			}
		}
	}

	if (min != 10000000000000 || max != -10000000000000) {
		*out = min;
		*out2 = first;
		return 1;
	}
	return 0;
}

__device__ Vector3 triangle_normal(Triangle* triangle) {
	Vector3 e1 = vect_sub(&(triangle->p2), &(triangle->p1));
	Vector3 e2 = vect_sub(&(triangle->p3), &(triangle->p1));
	return vect_cross(&e1, &e2);
}

__device__ Light* getLightBuffer(int bufferid, long long* alloc) {
	return (Light*)(*(alloc + bufferid * 4 + 2));
}

__device__ Material* getMaterialBuffer(int bufferid, long long* alloc) {
	return (Material*)(*(alloc + bufferid * 4 + 2));
}

__device__ Geometry* getGeometryBuffer(int bufferid, long long* alloc) {
	return (Geometry*)(*(alloc + bufferid * 4 + 2));
}

__device__ Point* getPointBuffer(int bufferid, long long* alloc) {
	return (Point*)(*(alloc + bufferid * 4 + 2));
}

__device__ Vector3* getNormalBuffer(int bufferid, long long* alloc) {
	return (Vector3*)(*(alloc + bufferid * 4 + 2));
}
__device__ int* getIndexBuffer(int bufferid, long long* alloc) {
	return (int*)(*(alloc + bufferid * 4 + 2));
}