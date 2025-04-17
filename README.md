# ODIN + WGPU test project

This project was created as a challenge to learn how to make the functionalities of
a game engine (starting with a rendering engine) but also to further experiment with
ODIN and to learn how to work with WGPU for cross-platform game development from
scratch.

The project may progress with some poorly positioned code as I learn and improve.

Below are some features I hope to impleent over time if I continue with the project
and don't find something else to fill my spare time. I will continnue to add to the
list as I think of new things that would be interesting to work on.

Web build: (I currently use python's test web server)
* `build_web.bat` on windows. I will add a linux shell script at some point also.
* `python -m http.server -d web`
* open browser to localhost:8000

# Building for windows
Windows build needs the SDL2.dll file in the build output directory. I have created a script prepare_build.bat which can create the
output directory and copy the required files over. VSCode is configured to run this as a pre-build step but it can be run manually using
`prepare_build.bat debug` or `prepare_build.bat release`

# Building for linux
Odin does not have some of the default libraries required to run this sample installed by default. Here are the steps I used to get the test running on Linux Mint.

Download https://github.com/gfx-rs/wgpu-native/releases/tag/v24.0.0.2
These are expected in Odin's `lib` folder under the same name as they are released (just unzipped).

```
make -C "~/Tools/Odin/vendor/stb/src"
make -C "~/Tools/Odin/vendor/cgltf/src"

sudo apt-get install libsdl2-dev
odin run .
```

The ability to use VSCode for build/debug has also been included by selecting the "(Linux Debug) Launch" launch configuration

### FEATURES TODO:

- [x] Simple WGPU sample (drawing a triangle)
- [x] Change shader to support passing a vertex buffer
- [x] Support vertex colours
- [x] Separate wgsl into its own file
- [x] Look at how to support more shaders, materials, meshes
- [x] Support model, view, projection matrices for 3D rendering
- [x] Rendering multiple objects
- [x] Depth buffer
- [x] Texturing
- [x] Render a cube
- [x] Render using index buffer
- [x] Simple glTF Model loading (specific structure only)
- [ ] Improve glTF Model loading
- [x] Point light
- [ ] Directional light
- [ ] Spot light
- [x] Flycam
- [ ] Normal mapping
- [ ] Frustum culling
- [ ] Shadow mapping
- [ ] Multiple lights
- [ ] Coloured lights
- [ ] Render Textures
- [ ] Cube Map
- [ ] Skybox
- [ ] Reflections
- [ ] Post Processing (Bloom)
- [ ] Skeletal Animation (Foward Kinematics)
- [ ] Inverse Kinematics
- [ ] PBR (Physically Based Rendering)
- [ ] Collision detection
- [ ] Ray picking
- [ ] Particles
- [ ] Add physics
- [ ] Global Illumination
- [x] MicroUI
- [x] Mouse/Keyboard support for MicroUI
- [ ] Dev Console
- [ ] Terrain
- [ ] SSR (Screen-space reflections)
- [ ] IBL (Image Based Lighting)
- [ ] ECS (Entity Component System)
- [ ] Text Rendering
- [x] Web Export
- [x] Linux build
- [ ] MacOS build
- [ ] Android build
- [ ] iOS build
- [ ] glTF support for web export
- [ ] Other asset support for web export
- [ ] Compute Shaders
