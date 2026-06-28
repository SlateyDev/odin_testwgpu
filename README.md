# ODIN + WGPU test project

A WGPU-based game engine written in Odin.

## Purpose

This project was created as a challenge to learn how to make the functionalities of
a game engine (starting with a rendering engine) but also to further experiment with
ODIN and to learn how to work with WGPU for cross-platform game development from
scratch.

The project may progress with some poorly positioned code as I learn and improve.

## Features

- Cross-platform rendering using WGPU
- Asset loading and management
- Input handling for SDL2 and web platforms
- UI rendering with microui
- GLTF model loading
- Cascade shadow mapping

## Building

### WebAssembly

* `build_web.bat` on windows. I will add a linux shell script at some point also.
* `python -m http.server -d web`
* open browser to localhost:8000

### Windows

Windows build needs the SDL2.dll file in the build output directory. I have created a script prepare_build.bat which can create the
output directory and copy the required files over. VSCode is configured to run this as a pre-build step but it can be run manually using
`prepare_build.bat debug` or `prepare_build.bat release`

# Linux
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

Below are some features I hope to impleent over time if I continue with the project
and don't find something else to fill my spare time. I will continnue to add to the
list as I think of new things that would be interesting to work on.

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
- [ ] glTF Model loading
  - [x] Retrieve vertex positions
  - [x] Retrieve vertex tex-coords
  - [x] Retrieve vertex normals
  - [x] Retrieve indexes
  - [x] Support base colour texture
  - [s] Support for primitives with no texture
  - [ ] Support child node transforms
  - [ ] Support materials without textures (shader support?)
  - [ ] Support normal texture (needs shader support)
  - [ ] Support metallic/roughness texture (needs shader support)
  - [ ] Support ambient/occlusion texture (needs shader support)
  - [ ] Support animation (needs engine support)
- [ ] Make glTF tester work out bounds of model and resize accordingly
- [x] Directional light (Sun)
- [ ] Point light
- [ ] Spot light
- [x] Flycam
- [ ] Normal mapping
- [ ] Frustum culling
- [x] Cascaded shadow mapping
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
- [x] glTF support for web export
- [x] Other asset support for web export
- [ ] Compute Shaders

## Project Structure

- `main.odin` - Main entry point and game loop
- `game_state.odin` - Game state management
- `os_sdl2.odin` - SDL2 platform specific code
- `os_js.odin` - Web platform specific code
- `wgpu_ext.odin` - WGPU extension functions
- `cube.odin`, `plane.odin`, `triangle.odin` - Basic geometry primitives
- `texture.odin` - Texture loading and management
- `microui_renderer.odin` - UI rendering with microui
- `assets/` - Asset files (models, textures)
- `shaders/` - Shader source files

## Dependencies

- Odin compiler
- SDL2 (for desktop builds)
- WebAssembly support for web builds

## Shaders

The project uses the following shaders:

- `shader.wgsl`: Main vertex and fragment shader with PBR lighting, shadows, and cascade shadow mapping
- `shadow_caster.wgsl`: Basic shader for rendering to depth buffer which is likely overkill
- `mu_shader.wgsl`: Microui UI shader

## Technical Details

### Cascade Shadow Mapping
This implementation uses a cascade shadow mapping system with 4 cascades to provide good shadow quality across different distances. The shader calculates which cascade level to use based on the depth of each fragment.

### Lighting Model
The lighting model implements:
- Directional light source
- Diffuse and specular lighting calculations
- Ambient lighting

### Camera Controls
The camera supports:
- Mouse look for rotation (hold right mouse button to look around)
- WASD movement for translation
- ESC to exit the application

## License

This project is licensed under the MIT license.