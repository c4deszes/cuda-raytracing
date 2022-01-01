#include "raytracing/cuda_defs.cuh"

__device__ Vector3 rotate(Vector3* a, flt x, flt y) {
	Vector3 out;
	out.x = a->x * cos(x) - a->z * sin(x);
	out.y = a->z * cos(x) + a->x * sin(x);
	out.z = a->z;
	out.z = out.y * cos(y) - out.z * sin(y);
	out.y = out.z * cos(y) + out.y * sin(y);
	return out;
}

__device__ Vector3 vect_add(Vector3* a, Vector3* b) {
	Vector3 out;
	out.x = a->x + b->x;
	out.y = a->y + b->y;
	out.z = a->z + b->z;
	return out;
}

__device__ Vector3 vect_sub(Vector3* a, Vector3* b) {
	Vector3 out;
	out.x = a->x - b->x;
	out.y = a->y - b->y;
	out.z = a->z - b->z;
	return out;
}

__device__ Vector3 vect_mul(Vector3* a, flt t) {
	Vector3 out;
	out.x = a->x * t;
	out.y = a->y * t;
	out.z = a->z * t;
	return out;
}

__device__ Vector3 vect_cross(Vector3* a, Vector3* b) {
	Vector3 out;
	out.x = a->y * b->z - a->z * b->y;
	out.y = a->z * b->x - a->x * b->z;
	out.z = a->x * b->y - a->y * b->x;
	return out;
}

__device__ Vector3 vect_norm(Vector3* a) {
	flt n = vect_len(a);
	Vector3 out;
	out.x = a->x / n;
	out.y = a->y / n;
	out.z = a->z / n;
	return out;
}

__device__ flt vect_len(Vector3* a) {
	return sqrt(a->x * a->x + a->y * a->y + a->z * a->z);
}

__device__ flt vect_dot(Vector3* a, Vector3* b) {
	return (a->x * b->x) + (a->y * b->y) + (a->z * b->z);
}

__device__ Color color_mulc(Color* a, flt t) {
	Color out;
	out.r = a->r * t;
	out.g = a->g * t;
	out.b = a->b * t;
	return out;
}

__device__ Color color_mul(Color* a, Color *b) {
	Color out;
	out.r = a->r * b->r;
	out.g = a->g * b->g;
	out.b = a->b * b->b;
	return out;
}
__device__ Color color_screen(Color* a, Color *b) {
	Color out;
	out.r = 1 - (1 - a->r) * (1 - b->r);
	out.g = 1 - (1 - a->g) * (1 - b->g);
	out.b = 1 - (1 - a->b) * (1 - b->b);
	return out;
}
__device__ Color color_overlay(Color* a, Color *b) {
	Color out;
	if (a->r >= 0.5) {
		out.r = 1 - 2 * (1 - a->r) * (1 - b->r);
	}
	else {
		out.r = 2 * a->r * b->r;
	}
	if (a->g >= 0.5) {
		out.g = 1 - 2 * (1 - a->g) * (1 - b->g);
	}
	else {
		out.g = 2 * a->g * b->g;
	}
	if (a->b >= 0.5) {
		out.b = 1 - 2 * (1 - a->b) * (1 - b->b);
	}
	else {
		out.b = 2 * a->b * b->b;
	}
	return out;
}