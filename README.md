# Vulkan Hardware Ray Tracing

A real-time, hardware-accelerated ray tracer built on the Vulkan ray tracing
pipeline (`VK_KHR_ray_tracing_pipeline`). It runs on the GPU's dedicated RT cores
using acceleration structures, a ray-gen / closest-hit / miss shader pipeline and
a shader binding table. This is genuine hardware ray tracing, not a compute-shader
software tracer.

Zero external dependencies beyond the Vulkan SDK itself: the window is created
with raw Win32 (no GLFW) and all math is inline (no GLM).

## Screenshot

![preview](preview.png)

## Features

- **Hardware ray tracing** — BLAS + TLAS, a full RT pipeline (ray generation,
  closest-hit and two miss shaders) and a correctly aligned shader binding table.
- **Animated geometry** — procedural super-torus, box, pyramid, tetrahedron and
  spheres, transformed on the CPU and rebuilt into the acceleration structure each
  frame, so objects orbit and spin and the light flies around, with no input needed.
- **Procedural textures + bump mapping** — checker, marble, wood and granite are
  generated from value-noise in object space (so they stick to moving objects),
  evaluated in the closest-hit shader. Granite and marble also perturb the shading
  normal from the noise gradient for a bumpy surface.
- **Cinematic camera** — thin-lens depth of field (the scene centre stays sharp,
  everything else melts into bokeh, accumulated across samples and TAA) and an
  ACES filmic tone-map over a linear-HDR pipeline, so emissive surfaces and the
  light keep a hot coloured core instead of clipping to flat white.
- **Spatial denoiser** — an edge-avoiding a-trous wavelet filter (SVGF-style) runs
  as a set of compute passes after the temporal pass, guided by a normal/depth
  buffer so it smooths noise without bleeding across geometry or shadow edges.
  Combined with the temporal accumulation this keeps the always-animating scene
  clean at a handful of samples per pixel.
- **Reflections** — traced iteratively from the ray-gen shader, so
  `maxPipelineRayRecursionDepth = 1`, which every RT-capable GPU supports.
- **Temporal reprojection (TAA) + anti-aliasing** — each frame casts several
  jittered samples per pixel and reprojects the previous frame's result through
  the previous camera matrix, blending the two in an exponential history buffer.
  The image stays clean while the camera moves, not just when it is still.
  History is rejected per pixel by a world-space distance test, so disocclusions
  fall back to the current frame instead of smearing.
- **Soft shadows from an area light** — the scene is lit by a spherical area
  light (a visible emissive sphere). Each shading point samples the cone the light
  subtends, so the penumbra widens with the occluder's distance, like real soft
  shadows, and sharpens as the accumulation converges.
- **Glass with chromatic dispersion** — the central sphere is dielectric glass
  with Fresnel-weighted reflection and refraction (Schlick) and total internal
  reflection. Refraction uses a different IOR per colour channel via spectral
  (hero-wavelength) sampling, so the glass splits light into faint rainbow edges.

The scene is an animated set of procedural shapes: a tumbling super-torus at the
centre, with a cube, a pyramid, a tetrahedron and spheres (one of them glass)
orbiting and spinning around it, lit by an area light that flies overhead.
Everything moves on its own; the camera orbit is layered on top.

## Controls

| Input            | Action                          |
|------------------|---------------------------------|
| Left mouse drag  | Orbit the camera                |
| Mouse wheel      | Zoom in / out                   |
| Esc              | Quit                            |

Drag to orbit; the temporal reprojection keeps the image clean during the motion
and it refines further the instant you stop.

## Requirements

- Windows 10/11, x64.
- Visual Studio 2022 (toolset v143). The free Community edition is fine.
- [Vulkan SDK](https://vulkan.lunarg.com/) installed (sets the `VULKAN_SDK`
  environment variable, which the project uses for include/lib paths and the
  shader compiler).
- A ray-tracing capable GPU with current drivers:
  NVIDIA RTX 20-series or newer, AMD RX 6000-series or newer, or Intel Arc.

## Build & run

1. Open `VulkanRayTracing.sln` in Visual Studio 2022.
2. Select the `x64` platform (`Debug` or `Release`).
3. Build and run (F5).

The shaders are compiled to SPIR-V automatically by a pre-build step
(`compile_shaders.bat`) and copied next to the executable. You can also run that
batch file by hand at any time.

A console window shows the selected GPU and, in Debug builds, Vulkan validation
output.

## Project layout

```
VulkanRayTracing.sln
VulkanRayTracing.vcxproj
compile_shaders.bat          shader -> SPIR-V build step
src/main.cpp                 all host code (window, Vulkan, RT setup, render loop)
shaders/raygen.rgen          camera rays, accumulation, AA, reflections,
                             soft shadows, glass, gamma
shaders/closesthit.rchit     vertex fetch + barycentric interpolation -> surface payload
shaders/miss.rmiss           primary miss (escaped ray -> sky)
shaders/shadow.rmiss         shadow miss (point is lit)
```

## Notes

- The window is a fixed 1280x720 and non-resizable, which keeps the swapchain
  logic minimal and the code easy to read.
- Vulkan clip space has Y pointing down, handled by a negative `m[5]` in
  `perspectiveVk`. If the image ever shows up vertically flipped on your driver,
  flip the sign of that one term and rebuild.
- The displayed image is an `R8G8B8A8_UNORM` storage image (matching the `rgba8`
  shader qualifier) copied to the swapchain with `vkCmdBlitImage`, which maps
  colour components by name, so colours stay correct whether the swapchain is
  RGBA or BGRA.
- Quality / look knobs: `shaders/raygen.rgen` holds the bounce count, the
  per-channel `IOR_R/G/B` (widen the spread for stronger dispersion), and the
  emissive light brightness. `updateUniforms` on the host sets the light position,
  the light radius (`lightPos[3]` — bigger = softer shadows), the intensity, and
  `params[3]`, the number of samples cast per frame (raise it for smoother motion,
  lower it for more speed on weaker GPUs).
