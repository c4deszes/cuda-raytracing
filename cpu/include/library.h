#pragma once
//Java style
typedef bool boolean;
//C++ style
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;

//2 Dimensional vector structure
typedef struct Vector2 {
	double x, y;
} Vector2;

//RGBA format
typedef struct Color {
	uint8_t r, g, b, a;
} Color;

//Builtin constants
#define PI 3.141592

//Default colors
#define RED 0xFF0000FF
#define GREEN 0x00FF00FF
#define BLUE 0x0000FFFF

#define YELLOW 0xFFFF00FF
#define MAGENTA 0xFF00FFFF
#define CYAN 0x00FFFFFF

#define ORANGE 0xFF8000FF
#define LIME 0x80FF00FF

#define AZURE 0x0080FFFF
#define	SPRING_GREEN 0x00FF80FF

#define PINK 0xFF0080FF
#define PURPLE 0x8000FFFF

#define CORAL 0xFF7F50FF
#define SKIN 0xFFDEADFF

#define BLACK 0x000000FF
#define DARK_GREY 0x404040FF
#define GREY 0x808080FF
#define LIGHT_GREY
#define WHITE 0xFFFFFFFF

//returns color compressed in a single integer
int color_rgba(Color *c);
//returns a random color in integer format
int color_rand();
//returns the inverse of a color
int color_inv(int color);
//equivalent to the glColor function call
void select_color(int color);

//returns a random double from 0 to max
double rand_max(double max);
//returns a random integer from 0 to max
int randint_max(int max);