#pragma once
#include "model.h"
#include <stddef.h>


void light_add(List_Light* list, Light* light) {
	while (list->next != NULL) {
		list = list->next;
	}
	list->next = malloc(sizeof(List_Light));
	list->next->light = light;
	list->next->next = NULL;
}

#define EPSILON 0.001

int triangle_intersection(const Triangle* triangle, const Ray* ray, double* out) {
	Vector3 e1, e2;  //Edge1, Edge2
	Vector3 P, Q, T;
	double det, inv_det, u, v;
	double t;

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

int mesh_intersection(const Mesh* mesh, const Ray* ray, double* out, Triangle* out2) {
	double min = 10000000000000;
	double max = -10000000000000;
	double t;
	Triangle temp;
	Triangle first;

	for (int i = 0; i < mesh->indices_length - 2; i += 3) {
		temp.p1 = mesh->points[mesh->index[i]];
		temp.p2 = mesh->points[mesh->index[i + 1]];
		temp.p3 = mesh->points[mesh->index[i + 2]];
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

Vector3 triangle_normal(const Triangle* triangle) {
	Vector3 e1 = vect_sub(&(triangle->p2), &(triangle->p1));
	Vector3 e2 = vect_sub(&(triangle->p3), &(triangle->p1));
	return vect_cross(&e1, &e2);
}