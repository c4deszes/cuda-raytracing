#define WIN32_LEAN_AND_MEAN 1
#define _WINSOCKAPI_
#define _CRT_SECURE_NO_WARNINGS
#define _CRTDBG_MAP_ALLOC 
#include <crtdbg.h>
#include <Windows.h>

#include <gl/GL.h>
#include "GLFW\glfw3.h"

//Use of GL13 and above
#ifdef _OPENGL_EXT
#include <glext.h>
#include <glxext.h>
#include <wglext.h>
#include <glcorearb.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include "defs.h"
#include "model.h"


#include <stdio.h>
#include "defs.h"
#include "model.h"

void init_graphics();
void update(double delta);
void render(double delta);
void dispose();

GLFWwindow* main_window = NULL;

#define THREAD_NUMBER 32
bool thread_running = true;
HANDLE threads[THREAD_NUMBER];

#define window_width 1600
#define window_height 800

Color framebuffer[window_height][window_width];
bool vsync = false;

Material diffuse_red = {
	-1,

	{ 0,1,0 },
	1.0,

	{ 0,0,0 },
	0.0
	
};

Material diffuse_blue = {
	-1,
	{ 1,0,0 },

	1.0,
	{ 0,0,0 },
	0.0
};

Point cube_points[] = {
	{ -50, 50, 50 },
	{ -50, 50, -50 },
	{ -50, -50, 50 },
	{ -50, -50, -50 },

	{ 50, 50, 50 },
	{ 50, 50, -50 },
	{ 50, -50, 50 },
	{ 50, -50, -50 },
};

int cube_indices[36] = {
	7, 1, 3,
	1, 7, 5,

	7, 4, 5,
	6, 4, 7,

	2, 1, 0,
	1, 2, 3,

	0, 1, 4,
	5, 4, 1,

	7, 3, 2,
	2, 6, 7,

	0, 4, 2,
	4, 6, 2
};

Mesh cube = {
	8,

	-1,
	cube_points,

	-1,
	NULL,

	36,
	-1,
	cube_indices,

	-1
};


#define FLOOR_SIZE 1000
#define FLOOR_HEIGHT -50
Point floor_points[] = {
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		FLOOR_SIZE },
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		FLOOR_SIZE },
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
};

int floor_indices[6] = {
	2, 1, 0,
	1, 2, 3
};

Mesh floor_plane = {
	4,

	-1,
	floor_points, 

	-1,
	NULL,

	6,
	-1,
	floor_indices, 
	
	-1
};

#define AMOUNT_OF_OBJECTS 2
Mesh* objects[] = {
	&cube, &floor_plane
};

Light light = {
	-1,

	{-200, 300, 200},
	{1,1,1},
	200,
	POINT_LIGHT
};

#define CAM_X -300
#define CAM_Y 100
#define CAM_Z 1500

int x_pos = 0;
int y_pos = 0;
bool locked = false;

#define SECTOR_SIZE 40
bool sectors[window_height / SECTOR_SIZE][window_width / SECTOR_SIZE];

void getSector(int *x1, int *y1) {
	while (locked);
	locked = true;
	for (int a = 0; a < window_height / SECTOR_SIZE; a++) {
		for (int b = 0; b < window_width / SECTOR_SIZE; b++) {
			if (!sectors[a][b]) {
				*x1 = b;
				*y1 = a;
				sectors[a][b] = true;
				locked = false;
				return;
			}
		}
	}
	vsync = true;
	locked = false;
}

DWORD WINAPI ThreadFunc(void* data) {
	int x = 0;
	int y = 0;

	Ray ray = {
		{ CAM_X, CAM_Y, CAM_Z },
		{ 0,0,-10.0 }
	};

	Point hit;
	Vector3 ldir, ndir;
	Color pixel;
	Color lightness;

	Ray temp;
	double idk;
	Triangle idk2;
	bool cansee;

	double t;
	Triangle polygon;

	while (thread_running) {
		getSector(&x, &y);

		for (int i = y * SECTOR_SIZE; i < (y + 1) * SECTOR_SIZE && thread_running; i++) {
			for (int j = x * SECTOR_SIZE; j < (x + 1) * SECTOR_SIZE && thread_running; j++) {
				ray.direction.x = (j - window_width / 2.0) / 2.0 * 0.02;
				ray.direction.y = (i - window_height / 2.0) / 2.0 * 0.02;

				double min = 10000000000000000.0;

				for (int n = 0; n < AMOUNT_OF_OBJECTS; n++) {
					if (mesh_intersection(objects[n], &ray, &t, &polygon)) {
						if (t < min) {
							Vector3 loc = vect_mul(&(ray.direction), t);
							hit = vect_add(&loc, &ray.origin);
							ldir = vect_sub(&hit, &(light.pos));

							temp.origin = hit;
							temp.direction = vect_mul(&ldir, -1);

							cansee = true;
							for (int m = 0; m < AMOUNT_OF_OBJECTS; m++) {
								//idk < 1 means that the vector from ray origin to object is shorter
								//then the vector pointing from ray origin to light source
								if (mesh_intersection(objects[m], &temp, &idk, &idk2) && idk < 1.0) {
									cansee = false;
								}
							}
							if (cansee) {
								double intensity = light.intensity / vect_len(&ldir);
								ldir = vect_norm(&ldir);
								ndir = triangle_normal(&polygon);
								ndir = vect_norm(&ndir);
								lightness = color_mulc(&(light.color), vect_dot(&ldir, &ndir) * intensity);
								pixel = color_mul(&(objects[n]->mat->diffuse), &lightness);

								framebuffer[i][j] = pixel;
							}
							min = t;
						}
					}
				}
			}
		}
	}
	

	return 0;
}

double tstart, tend;

int main()
{
	if (!glfwInit()) {
		return -1;
	}

	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	main_window = glfwCreateWindow(window_width, window_height, "Raytracer 1.0", NULL, NULL);

	if (main_window == NULL) {
		return -1;
	}

	//Create OpenGL Context
	glfwMakeContextCurrent(main_window);
	glfwSwapInterval(1);

	//Set object materials
	objects[0]->mat = &diffuse_red;
	objects[1]->mat = &diffuse_blue;

	init_graphics();

	tstart = glfwGetTime();
	for (int i = 0; i < THREAD_NUMBER; i++) {
		threads[i] = CreateThread(NULL, 4096, ThreadFunc, NULL, 0, NULL);
	}

	double now, last, delta;
	last = 0;

	while (glfwWindowShouldClose(main_window) != GLFW_TRUE) {
		now = glfwGetTime();
		delta = now - last;
		last = now;
		update(delta);
		render(delta);
		glfwPollEvents();
		glfwSwapBuffers(main_window);
	}
	dispose();
	glfwDestroyWindow(main_window);
	glfwTerminate();

	thread_running = false;

    return 0;
}

void init_graphics() {
	//Graphics setup
	glViewport(0, 0, window_width, window_height);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glOrtho(0, window_width, 0, window_height, -1, 1);
}

void update(double delta) {
	if (vsync && thread_running) {
		thread_running = false;
		tend = glfwGetTime();

		printf("Render took: %f sec", tend - tstart);
	}
}

void render(double delta) {
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POINTS);
	for (int y = 0; y < window_height; y++) {
		for (int x = 0; x < window_width; x++) {
			glColor3d(framebuffer[y][x].r, framebuffer[y][x].g, framebuffer[y][x].b);
			glVertex2i(x, y);
		}
	}
	glEnd();
}

void dispose() {

}