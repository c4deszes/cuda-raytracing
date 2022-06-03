# CUDA raytracer

> This is a bare minimum raytracing renderer written in C and CUDA as side project in first semester
> uni. You can provide triangle based meshes, basic materials and lights.

## Building and running

### Tools

+ NVidia CUDA Compiler 11.1
+ Visual Studio 2019 (Build tools should be enough)
+ CMake 3.20 or above

### CMake environment

Create your own `CMakeUserPresets.json` file that extends the included presets, for example on
Windows with Visual Studio:

```json
{
    "version": 2,
    "configurePresets": [
        {
            "name": "user-default",
            "inherits": "vstudio-default",
            "environment": {
                "PATH": "$penv{PATH};<your tool paths>"
            }
        }
    ]
}
```
