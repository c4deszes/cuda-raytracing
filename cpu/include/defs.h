#pragma once

/* TYPE REDEFS*/
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;

/* VECTOR TYPE */
typedef struct Vector3 {
	double x, y, z;
} Vector3;

//for increasing readability
typedef Vector3 Point;

Vector3 vect_add(Vector3* a, Vector3* b);
Vector3 vect_sub(Vector3* a, Vector3* b);
Vector3 vect_mul(Vector3* a, double t);
Vector3 vect_cross(Vector3* a, Vector3* b);
Vector3 vect_norm(Vector3* a);
double vect_len(Vector3* a);
double vect_dot(Vector3* a, Vector3* b);

/* COLOR TYPE */
#define COLOR_RGB 0
#define COLOR_GRAYSCALE 1

typedef struct Color {
	double r, g, b;
} Color;

Color color_mulc(Color* a, double t);
Color color_mul(Color* a, Color *b);
Color color_screen(Color* a, Color *b);
Color color_overlay(Color* a, Color *b);
//Color color_hardlight(Color* a, Color *b);
//Color color_softlight(Color* a, Color *b);
