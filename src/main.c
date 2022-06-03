#include "raytracing/stdhead.h"
#include <stdio.h>
#include "raytracing/cuda_defs.cuh"
#include "raytracing/cuda_model.cuh"

#include "raytracing/wrapper.h"


void init_graphics();
void update(double delta);
void render(double delta);
void dispose();
#define RESX 800
#define RESY 800

#define window_width RESX
#define window_height RESY
GLFWwindow* main_window = NULL;

/*
camera -> ray


ray.x
*/

Camera main_camera = {
	{ 0, 0 ,1500 },
	{ 0, 0, -10 },
	2.0944,			//fov = 120�
	1.77778,		//16:9

	{0,0,0},

	RESX,
	RESY,
	COLOR_RGB
};

Light main_light = {
	-1,
	{ 300, 100, 200 },
	{ 1, 1, 1 },
	400,
	POINT_LIGHT,
	{ 0,0,0 }
};

List_Light lights = {
	&main_light,
	NULL
};

Material diffuse_red = {
	-1,
	{ 1,0,0 },
	1.0,
	{ 0,0,0 },
	0.0
};
Material diffuse_green = {
	-1,
	{ 0,1,0 },
	1.0,
	{ 0,0,0 },
	0.0
};
Material diffuse_white = {
	-1,
	{ 1,1,1 },
	1.0,
	{ 0,0,0 },
	0.0
};

List_Material mat3 = {
	&diffuse_white,
	NULL
};

List_Material mat2 = {
	&diffuse_green,
	&mat3
};

List_Material materials = {
	&diffuse_red,
	&mat2
};

#define FLOOR_SIZE 400
#define FLOOR_HEIGHT 400
Point floor_points[] = {
	{ FLOOR_SIZE,		-FLOOR_HEIGHT,		FLOOR_SIZE },
	{ FLOOR_SIZE,		-FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ -FLOOR_SIZE,		-FLOOR_HEIGHT,		FLOOR_SIZE },
	{ -FLOOR_SIZE,		-FLOOR_HEIGHT,		-FLOOR_SIZE },
};

Point ceiling_points[] = {
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		FLOOR_SIZE },
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		FLOOR_SIZE },
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
};

Point rightside_points[] = {
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		FLOOR_SIZE },
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ FLOOR_SIZE,		-FLOOR_HEIGHT,		FLOOR_SIZE },
	{ FLOOR_SIZE,		-FLOOR_HEIGHT,		-FLOOR_SIZE },
};

Point leftside_points[] = {
	{ -FLOOR_SIZE,		-FLOOR_HEIGHT,		FLOOR_SIZE },
	{ -FLOOR_SIZE,		-FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		FLOOR_SIZE },
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
};

Point backside_points[] = {
	{ -FLOOR_SIZE,		-FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ FLOOR_SIZE,		-FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ -FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
	{ FLOOR_SIZE,		FLOOR_HEIGHT,		-FLOOR_SIZE },
};

int floor_indices[6] = {
	2, 1, 0,
	1, 2, 3
};

Mesh obj_floor = {
	-1,

	4,

	-1,
	floor_points,

	-1,
	NULL,

	6,

	-1,
	floor_indices,

	&diffuse_white
};

Mesh obj_ceiling = {
	-1,

	4,

	-1,
	ceiling_points,

	-1,
	NULL,

	6,

	-1,
	floor_indices,

	&diffuse_white
};

Mesh obj_rightside = {
	-1,

	4,

	-1,
	rightside_points,

	-1,
	NULL,

	6,

	-1,
	floor_indices,

	&diffuse_green
};

Mesh obj_leftside = {
	-1,

	4,

	-1,
	leftside_points,

	-1,
	NULL,

	6,

	-1,
	floor_indices,

	&diffuse_red
};

Mesh obj_backside = {
	-1,

	4,

	-1,
	backside_points,

	-1,
	NULL,

	6,

	-1,
	floor_indices,

	&diffuse_white
};

#define CUBE_SIZE 100

Point cube_points[8] = {
	{ -CUBE_SIZE, CUBE_SIZE, CUBE_SIZE },
	{ -CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE },
	{ -CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE },
	{ -CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE },

	{ CUBE_SIZE, CUBE_SIZE, CUBE_SIZE },
	{ CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE },
	{ CUBE_SIZE, -CUBE_SIZE, CUBE_SIZE },
	{ CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE },
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

Mesh obj_cube = {
	-1,

	8,

	-1,
	cube_points,

	-1,
	NULL,

	36,

	-1,
	cube_indices,

	&diffuse_red

};

List_Mesh obj6 = {
	&obj_cube,
	NULL
};

List_Mesh obj5 = {
	&obj_backside,
	&obj6
};

List_Mesh obj4 = {
	&obj_leftside,
	&obj5
};

List_Mesh obj3 = {
	&obj_rightside,
	&obj4
};

List_Mesh obj2 = {
	&obj_ceiling,
	&obj3
};

List_Mesh objects = {
	&obj_floor,
	&obj2
};

Vector3 v = { -100, -300, 0 };

uint32_t* framebuffer;



int main()
{
	if (!glfwInit()) {
		return -1;
	}

	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	main_window = glfwCreateWindow(window_width, window_height, "Raytracer 2.0 - Now with 100% more CUDA", NULL, NULL);

	if (main_window == NULL) {
		return -1;
	}

	//Create OpenGL Context
	glfwMakeContextCurrent(main_window);
	glfwSwapInterval(1);

	init_graphics();

	framebuffer = (uint32_t*) malloc(main_camera.width * main_camera.height * sizeof(uint32_t));
	copy_scene(main_camera, &objects, &materials, &lights);

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

	translate(obj_cube.points, obj_cube.points_length, v);
}

void update(double delta) {
	double rx = 1;
	double ry = 0;
	double x, y, z;
	x = main_camera.direction.x * cos(ry) - main_camera.direction.z * sin(ry);
	z = main_camera.direction.z * cos(ry) + main_camera.direction.x * sin(ry);
	//z = main_camera.direction.z;
	z = main_camera.direction.z * cos(ry) - main_camera.direction.y * sin(ry);
	y = main_camera.direction.y * cos(ry) + main_camera.direction.z * sin(ry);

	main_camera.direction.x = x;
	main_camera.direction.y = y;
	main_camera.direction.z = z;
	//translate(obj_cube.points, obj_cube.points_length, v);
	//bufferData(obj_cube.point_buffer_id, 0, obj_cube.points_length, sizeof(Point), obj_cube.points);
}

void render(double delta) {
	render_scene(main_camera, framebuffer);

	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POINTS);
	for (int i = 0; i < main_camera.height; i++) {
		for (int j = 0; j < main_camera.width; j++) {
			uint32_t color = *(framebuffer + i * main_camera.width + j);
			glColor3ub((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
			glVertex2i(j, i);
		}
	}
	glEnd();

}

void dispose() {
	delete_scene();
	free(framebuffer);
}
