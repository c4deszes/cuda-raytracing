#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

/* TYPE REDEFS*/
typedef bool boolean;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;

typedef float flt;

/* VECTOR TYPE */
typedef struct Vector3 {
	flt x, y, z;
} Vector3;

//for increasing readability
typedef Vector3 Point;

//Gimbal lock!!!
__device__ Vector3 rotate(Vector3* a, flt x, flt y);
__device__ Vector3 vect_add(Vector3* a, Vector3* b);
__device__ Vector3 vect_sub(Vector3* a, Vector3* b);
__device__ Vector3 vect_mul(Vector3* a, flt t);
__device__ Vector3 vect_cross(Vector3* a, Vector3* b);
__device__ Vector3 vect_norm(Vector3* a);
__device__ flt vect_len(Vector3* a);
__device__ flt vect_dot(Vector3* a, Vector3* b);

/* COLOR TYPE */
#define COLOR_RGB 0
#define COLOR_GRAYSCALE 1

typedef struct Color {
	flt r, g, b;
} Color;

__device__ Color color_mulc(Color* a, flt t);
__device__ Color color_mul(Color* a, Color *b);
__device__ Color color_screen(Color* a, Color *b);
__device__ Color color_overlay(Color* a, Color *b);