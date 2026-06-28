// -----------------------------------------------------------------------------
//  Hardware-accelerated Vulkan ray tracing  (VK_KHR_ray_tracing_pipeline)
//
//  - Pure Win32 window, no GLFW / GLM (only the Vulkan SDK is required).
//  - Builds a BLAS + TLAS, a ray-tracing pipeline and a shader binding table.
//  - Real RT cores: ray-gen / closest-hit / miss shaders, reflections + shadows.
//
//  The scene is a reflective checkerboard floor with a ring of coloured spheres
//  and a large mirror sphere in the middle. The camera orbits automatically.
// -----------------------------------------------------------------------------

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <vector>
#include <string>
#include <array>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <fstream>

// -----------------------------------------------------------------------------
//  Small helpers
// -----------------------------------------------------------------------------
static const uint32_t WIDTH  = 1280;
static const uint32_t HEIGHT = 720;

#ifdef NDEBUG
static const bool kEnableValidation = false;
#else
static const bool kEnableValidation = true;
#endif

static void vkCheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "Vulkan call failed (%d): %s", (int)r, what);
        throw std::runtime_error(buf);
    }
}

static uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// -----------------------------------------------------------------------------
//  Tiny column-major 4x4 math (just enough for the camera)
// -----------------------------------------------------------------------------
struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};
static Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static Vec3 operator*(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 cross(Vec3 a, Vec3 b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static Vec3 normalize(Vec3 v) {
    float l = std::sqrt(dot(v, v));
    return l > 0 ? Vec3{ v.x / l, v.y / l, v.z / l } : v;
}

// Small 3x3 (columns) for rotating object vertices/normals each frame.
struct Mat3 { Vec3 c0{ 1,0,0 }, c1{ 0,1,0 }, c2{ 0,0,1 }; };
static Vec3 mul3(const Mat3& m, Vec3 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }
static Mat3 rotAxis(Vec3 axis, float ang) {
    axis = normalize(axis);
    float c = std::cos(ang), s = std::sin(ang), t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;
    Mat3 m;
    m.c0 = { c + x*x*t,     x*y*t + z*s,  x*z*t - y*s };
    m.c1 = { x*y*t - z*s,   c + y*y*t,    y*z*t + x*s };
    m.c2 = { x*z*t + y*s,   y*z*t - x*s,  c + z*z*t   };
    return m;
}

struct Mat4 { float m[16] = {}; }; // column-major: m[col*4 + row]

static Mat4 mat4Identity() {
    Mat4 r; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f; return r;
}

static Mat4 lookAtRH(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = mat4Identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] =  dot(f, eye);
    return r;
}

// Right-handed, zero-to-one depth (Vulkan), with the Y flip baked in.
static Mat4 perspectiveVk(float fovY, float aspect, float zNear, float zFar) {
    float t = std::tan(fovY * 0.5f);
    Mat4 r; // all zero
    r.m[0]  = 1.0f / (aspect * t);
    r.m[5]  = -1.0f / t;                  // negative => Vulkan Y-down clip space
    r.m[10] = zFar / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = -(zFar * zNear) / (zFar - zNear);
    return r;
}

static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 c;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[col * 4 + k];
            c.m[col * 4 + row] = s;
        }
    return c;
}

// General 4x4 inverse (adjugate / determinant).
static Mat4 inverse(const Mat4& in) {
    const float* m = in.m;
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    Mat4 out;
    if (det == 0.0f) return mat4Identity();
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) out.m[i] = inv[i] * det;
    return out;
}

// -----------------------------------------------------------------------------
//  Vertex / scene data (matches the GLSL std430 layout)
// -----------------------------------------------------------------------------
struct Vertex {
    float pos[3];  float reflectivity;  // vec4: xyz world position, w reflectivity
    float nrm[3];  float matId;         // vec4: xyz world normal,   w material id
    float col[3];  float texId;         // vec4: xyz colour,         w texture id
    float loc[3];  float bump;          // vec4: xyz object-space pos, w bump strength
    float prv[3];  float emissive;      // vec4: xyz previous world pos, w emissive strength
};

struct UBO {
    Mat4     viewInverse;
    Mat4     projInverse;
    Mat4     prevViewProj;
    float    lightPos[4];   // xyz = position, w = radius
    float    prevCamPos[4]; // xyz = previous camera position, w = TAA blend alpha
    float    params[4];     // x = time, y = maxBounces, z = intensity, w = samples/frame
    uint32_t frame[4];      // y = free-running frame, z = historyValid, w = parity
    float    lens[4];       // x = aperture radius, y = focus distance, z = exposure, w = unused
};

// -----------------------------------------------------------------------------
//  Ray tracing entry points (loaded at runtime; not exported by the loader lib)
// -----------------------------------------------------------------------------
static PFN_vkGetBufferDeviceAddress                 pvkGetBufferDeviceAddress              = nullptr;
static PFN_vkCreateAccelerationStructureKHR         pvkCreateAccelerationStructure         = nullptr;
static PFN_vkDestroyAccelerationStructureKHR        pvkDestroyAccelerationStructure        = nullptr;
static PFN_vkGetAccelerationStructureBuildSizesKHR  pvkGetAccelerationStructureBuildSizes  = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR      pvkCmdBuildAccelerationStructures      = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pvkGetAccelerationStructureDeviceAddress = nullptr;
static PFN_vkCreateRayTracingPipelinesKHR           pvkCreateRayTracingPipelines           = nullptr;
static PFN_vkGetRayTracingShaderGroupHandlesKHR     pvkGetRayTracingShaderGroupHandles     = nullptr;
static PFN_vkCmdTraceRaysKHR                        pvkCmdTraceRays                        = nullptr;

#define LOAD_DEV(name, var) \
    var = (PFN_##name)vkGetDeviceProcAddr(dev, #name); \
    if (!var) throw std::runtime_error("Failed to load " #name);

// =============================================================================
//  Renderer
// =============================================================================
class RayTracer {
public:
    void run(HINSTANCE hInst) {
        createWindow(hInst);
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // ---- Win32 ----
    HWND hwnd = nullptr;
    bool quit = false;

    // ---- Interactive orbit camera ----
    float camYaw      = 2.2f;
    float camPitch    = 0.42f;
    float camDist     = 9.0f;
    bool  dragging    = false;
    POINT lastMouse{};
    bool  cameraDirty = true;   // forces an accumulation reset
    uint32_t accumFrame = 0;    // resets on camera move (drives the running average)
    uint32_t totalFrame = 0;    // never resets (drives RNG decorrelation)

    // ---- Core Vulkan ----
    VkInstance       instance = VK_NULL_HANDLE;
    VkSurfaceKHR     surface  = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         dev      = VK_NULL_HANDLE;
    uint32_t         queueFamily = 0;
    VkQueue          queue    = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    // ---- Swapchain ----
    VkSwapchainKHR        swapchain = VK_NULL_HANDLE;
    VkFormat              swapFormat;
    VkExtent2D            swapExtent;
    std::vector<VkImage>  swapImages;

    // ---- Command / sync ----
    VkCommandPool   cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd     = VK_NULL_HANDLE;
    VkSemaphore     semImageAvailable = VK_NULL_HANDLE;
    VkSemaphore     semRenderFinished = VK_NULL_HANDLE;
    VkFence         inFlight = VK_NULL_HANDLE;

    // ---- RT properties ----
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};

    // ---- Geometry ----
    struct Buffer { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceSize size = 0; };
    Buffer vertexBuffer, indexBuffer, ubo;
    uint32_t indexCount = 0;

    // ---- Animated scene ----
    struct SceneObject {
        std::vector<Vertex> base;   // local-space vertices (pos/normal in local frame)
        uint32_t vertexOffset = 0;  // where this object's verts live in the merged buffer
        Vec3  orbitCenter{};
        float orbitRadius = 0, orbitSpeed = 0, orbitPhase = 0;
        Vec3  spinAxis{ 0, 1, 0 };
        float spinSpeed = 0;
        bool  isLight = false;      // follows the moving light position
        Mat3  prevRot{};            // transform from the previous frame (motion vectors)
        Vec3  prevPos{};
        bool  prevInit = false;
    };
    std::vector<SceneObject> objects;
    std::vector<Vertex>   sceneVerts;     // merged, rewritten each frame
    std::vector<uint32_t> sceneIndices;   // merged, static topology
    uint32_t totalVerts = 0;
    Vec3 curLightPos{ 4, 7, -2 };

    // ---- Acceleration structures (rebuilt each frame) ----
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE; Buffer blasBuffer;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE; Buffer tlasBuffer;
    Buffer asScratch, asInstance;
    VkAccelerationStructureGeometryKHR blasGeom{};
    VkAccelerationStructureGeometryKHR tlasGeom{};
    uint32_t blasTriCount = 0;

    // ---- Output storage image (rgba8, blitted to the swapchain) ----
    VkImage        storageImage = VK_NULL_HANDLE;
    VkDeviceMemory storageMem   = VK_NULL_HANDLE;
    VkImageView    storageView  = VK_NULL_HANDLE;

    // ---- TAA history (rgba32f, ping-pong; rgb = colour, a = distance key) ----
    VkImage        histImg[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory histMem[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView    histView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    // ---- Denoiser: ping-pong colour (rgba32f) + guide (xyz normal, w depth) ----
    VkImage        denImg[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory denMem[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView    denView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImage        guideImg = VK_NULL_HANDLE;
    VkDeviceMemory guideMem = VK_NULL_HANDLE;
    VkImageView    guideView = VK_NULL_HANDLE;
    int            atrousPasses = 3;   // 0 = denoiser off (tone-map only), 3 = full

    VkDescriptorSetLayout compDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet       compSet[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout      compPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            compPipeline = VK_NULL_HANDLE;

    // Previous-frame camera state for reprojection.
    Mat4 prevViewProj{};
    Vec3 prevCamPos{};
    bool haveHistory = false;

    // ---- Pipeline ----
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descPool   = VK_NULL_HANDLE;
    VkDescriptorSet       descSet    = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            rtPipeline     = VK_NULL_HANDLE;

    // ---- Shader binding table ----
    Buffer sbtBuffer;
    VkStridedDeviceAddressRegionKHR rgenRegion{}, missRegion{}, hitRegion{}, callRegion{};

    std::chrono::high_resolution_clock::time_point startTime;

    // -------------------------------------------------------------------------
    //  Win32 window
    // -------------------------------------------------------------------------
    static LRESULT CALLBACK wndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
        RayTracer* self = reinterpret_cast<RayTracer*>(GetWindowLongPtr(h, GWLP_USERDATA));
        switch (msg) {
        case WM_CREATE: {
            auto* cs = reinterpret_cast<CREATESTRUCT*>(lp);
            SetWindowLongPtr(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
            return 0;
        }
        case WM_KEYDOWN:
            if (wp == VK_ESCAPE && self) self->quit = true;
            return 0;
        case WM_LBUTTONDOWN:
            if (self) {
                self->dragging = true;
                self->lastMouse.x = (short)LOWORD(lp);
                self->lastMouse.y = (short)HIWORD(lp);
                SetCapture(h);
            }
            return 0;
        case WM_LBUTTONUP:
            if (self) { self->dragging = false; ReleaseCapture(); }
            return 0;
        case WM_MOUSEMOVE:
            if (self && self->dragging) {
                int x = (short)LOWORD(lp), y = (short)HIWORD(lp);
                int dx = x - self->lastMouse.x;
                int dy = y - self->lastMouse.y;
                self->lastMouse.x = x; self->lastMouse.y = y;
                self->camYaw   += dx * 0.005f;
                self->camPitch += dy * 0.005f;
                const float lim = 1.5f; // ~85 degrees
                if (self->camPitch >  lim) self->camPitch =  lim;
                if (self->camPitch < -lim) self->camPitch = -lim;
                self->cameraDirty = true;
            }
            return 0;
        case WM_MOUSEWHEEL:
            if (self) {
                float delta = (float)GET_WHEEL_DELTA_WPARAM(wp) / 120.0f;
                self->camDist *= std::pow(0.9f, delta);
                if (self->camDist < 3.0f)  self->camDist = 3.0f;
                if (self->camDist > 30.0f) self->camDist = 30.0f;
                self->cameraDirty = true;
            }
            return 0;
        case WM_CLOSE:
            if (self) self->quit = true;
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProc(h, msg, wp, lp);
    }

    void createWindow(HINSTANCE hInst) {
        WNDCLASSEX wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = wndProc;
        wc.hInstance = hInst;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.lpszClassName = "VkRayTracerWindow";
        RegisterClassEx(&wc);

        // Fixed-size window (no resize) keeps the swapchain logic simple.
        DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        RECT rect = { 0, 0, (LONG)WIDTH, (LONG)HEIGHT };
        AdjustWindowRect(&rect, style, FALSE);

        hwnd = CreateWindowEx(0, wc.lpszClassName,
            "Vulkan Hardware Ray Tracing", style,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left, rect.bottom - rect.top,
            nullptr, nullptr, hInst, this);

        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
    }

    // -------------------------------------------------------------------------
    //  Vulkan setup
    // -------------------------------------------------------------------------
    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createDevice();
        loadRayTracingFunctions();
        createSwapchain();
        createCommandResources();
        createScene();
        createAccelStructures();
        createStorageImage();
        createUniformBuffer();
        createDescriptors();
        createRayTracingPipeline();
        createShaderBindingTable();
        createComputePipeline();
        startTime = std::chrono::high_resolution_clock::now();
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCB(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            std::fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
        return VK_FALSE;
    }

    void createInstance() {
        VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "VkRayTracer";
        app.apiVersion = VK_API_VERSION_1_2;

        std::vector<const char*> exts = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        };
        std::vector<const char*> layers;
        if (kEnableValidation) {
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }

        VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.ppEnabledExtensionNames = exts.data();
        ci.enabledLayerCount = (uint32_t)layers.size();
        ci.ppEnabledLayerNames = layers.data();
        vkCheck(vkCreateInstance(&ci, nullptr, &instance), "vkCreateInstance");

        if (kEnableValidation) {
            auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                instance, "vkCreateDebugUtilsMessengerEXT");
            if (fn) {
                VkDebugUtilsMessengerCreateInfoEXT d{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
                d.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                d.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                d.pfnUserCallback = debugCB;
                fn(instance, &d, nullptr, &debugMessenger);
            }
        }
    }

    void createSurface() {
        VkWin32SurfaceCreateInfoKHR ci{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
        ci.hinstance = GetModuleHandle(nullptr);
        ci.hwnd = hwnd;
        vkCheck(vkCreateWin32SurfaceKHR(instance, &ci, nullptr, &surface), "vkCreateWin32SurfaceKHR");
    }

    bool deviceSupportsRt(VkPhysicalDevice d) {
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> props(n);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &n, props.data());
        auto has = [&](const char* name) {
            for (auto& p : props) if (std::strcmp(p.extensionName, name) == 0) return true;
            return false;
        };
        return has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
               has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
               has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) &&
               has(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    void pickPhysicalDevice() {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        if (n == 0) throw std::runtime_error("No Vulkan devices found");
        std::vector<VkPhysicalDevice> devices(n);
        vkEnumeratePhysicalDevices(instance, &n, devices.data());

        for (auto d : devices) {
            if (!deviceSupportsRt(d)) continue;
            // Needs a queue with graphics + present.
            uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qprops(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qprops.data());
            for (uint32_t i = 0; i < qn; ++i) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
                if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                    phys = d;
                    queueFamily = i;
                    break;
                }
            }
            if (phys != VK_NULL_HANDLE) break;
        }
        if (phys == VK_NULL_HANDLE)
            throw std::runtime_error("No GPU with VK_KHR_ray_tracing_pipeline support found.\n"
                                     "A ray-tracing capable GPU and recent drivers are required.");

        rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(phys, &props2);

        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(phys, &p);
        std::printf("GPU: %s\n", p.deviceName);
    }

    void createDevice() {
        float priority = 1.0f;
        VkDeviceQueueCreateInfo q{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        q.queueFamilyIndex = queueFamily;
        q.queueCount = 1;
        q.pQueuePriorities = &priority;

        const char* deviceExts[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        };

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeat{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
        rtFeat.rayTracingPipeline = VK_TRUE;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
        asFeat.accelerationStructure = VK_TRUE;
        asFeat.pNext = &rtFeat;

        VkPhysicalDeviceVulkan12Features v12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        v12.bufferDeviceAddress = VK_TRUE;
        v12.pNext = &asFeat;

        VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        f2.pNext = &v12;

        VkDeviceCreateInfo ci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        ci.pNext = &f2;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &q;
        ci.enabledExtensionCount = (uint32_t)(sizeof(deviceExts) / sizeof(deviceExts[0]));
        ci.ppEnabledExtensionNames = deviceExts;
        vkCheck(vkCreateDevice(phys, &ci, nullptr, &dev), "vkCreateDevice");

        vkGetDeviceQueue(dev, queueFamily, 0, &queue);
    }

    void loadRayTracingFunctions() {
        LOAD_DEV(vkGetBufferDeviceAddress,                   pvkGetBufferDeviceAddress);
        LOAD_DEV(vkCreateAccelerationStructureKHR,            pvkCreateAccelerationStructure);
        LOAD_DEV(vkDestroyAccelerationStructureKHR,           pvkDestroyAccelerationStructure);
        LOAD_DEV(vkGetAccelerationStructureBuildSizesKHR,     pvkGetAccelerationStructureBuildSizes);
        LOAD_DEV(vkCmdBuildAccelerationStructuresKHR,         pvkCmdBuildAccelerationStructures);
        LOAD_DEV(vkGetAccelerationStructureDeviceAddressKHR,  pvkGetAccelerationStructureDeviceAddress);
        LOAD_DEV(vkCreateRayTracingPipelinesKHR,              pvkCreateRayTracingPipelines);
        LOAD_DEV(vkGetRayTracingShaderGroupHandlesKHR,        pvkGetRayTracingShaderGroupHandles);
        LOAD_DEV(vkCmdTraceRaysKHR,                           pvkCmdTraceRays);
    }

    void createSwapchain() {
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps);

        uint32_t fn = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &fn, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(fn);
        vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &fn, formats.data());

        VkSurfaceFormatKHR chosen = formats[0];
        for (auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { chosen = f; break; }
        }
        swapFormat = chosen.format;

        swapExtent = caps.currentExtent;
        if (swapExtent.width == UINT32_MAX) { swapExtent = { WIDTH, HEIGHT }; }

        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
            imageCount = caps.maxImageCount;

        VkSwapchainCreateInfoKHR ci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        ci.surface = surface;
        ci.minImageCount = imageCount;
        ci.imageFormat = chosen.format;
        ci.imageColorSpace = chosen.colorSpace;
        ci.imageExtent = swapExtent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;
        vkCheck(vkCreateSwapchainKHR(dev, &ci, nullptr, &swapchain), "vkCreateSwapchainKHR");

        uint32_t n = 0;
        vkGetSwapchainImagesKHR(dev, swapchain, &n, nullptr);
        swapImages.resize(n);
        vkGetSwapchainImagesKHR(dev, swapchain, &n, swapImages.data());
    }

    void createCommandResources() {
        VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = queueFamily;
        vkCheck(vkCreateCommandPool(dev, &pci, nullptr, &cmdPool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkCheck(vkAllocateCommandBuffers(dev, &ai, &cmd), "vkAllocateCommandBuffers");

        VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        vkCheck(vkCreateSemaphore(dev, &sci, nullptr, &semImageAvailable), "sem");
        vkCheck(vkCreateSemaphore(dev, &sci, nullptr, &semRenderFinished), "sem");

        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCheck(vkCreateFence(dev, &fci, nullptr, &inFlight), "fence");
    }

    // ---- memory helpers ----
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("No suitable memory type");
    }

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps) {
        Buffer b; b.size = size;
        VkBufferCreateInfo ci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        ci.size = size;
        ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCheck(vkCreateBuffer(dev, &ci, nullptr, &b.buf), "vkCreateBuffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, b.buf, &req);

        VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.pNext = &flags;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memProps);
        vkCheck(vkAllocateMemory(dev, &ai, nullptr, &b.mem), "vkAllocateMemory");
        vkCheck(vkBindBufferMemory(dev, b.buf, b.mem, 0), "vkBindBufferMemory");
        return b;
    }

    void uploadToBuffer(Buffer& b, const void* data, VkDeviceSize size) {
        void* mapped = nullptr;
        vkCheck(vkMapMemory(dev, b.mem, 0, size, 0, &mapped), "vkMapMemory");
        std::memcpy(mapped, data, (size_t)size);
        vkUnmapMemory(dev, b.mem);
    }

    VkDeviceAddress bufferAddress(VkBuffer buf) {
        VkBufferDeviceAddressInfo ai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        ai.buffer = buf;
        return pvkGetBufferDeviceAddress(dev, &ai);
    }

    VkCommandBuffer beginOneShot() {
        VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer c;
        vkAllocateCommandBuffers(dev, &ai, &c);
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(c, &bi);
        return c;
    }

    void endOneShot(VkCommandBuffer c) {
        vkEndCommandBuffer(c);
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(dev, cmdPool, 1, &c);
    }

    // -------------------------------------------------------------------------
    //  Procedural meshes (local space, centred at the origin)
    // -------------------------------------------------------------------------
    static void pushV(std::vector<Vertex>& V, Vec3 p, Vec3 n, Vec3 c, float refl, float mat) {
        Vertex v{};
        v.pos[0] = p.x; v.pos[1] = p.y; v.pos[2] = p.z; v.reflectivity = refl;
        v.nrm[0] = n.x; v.nrm[1] = n.y; v.nrm[2] = n.z; v.matId = mat;
        v.col[0] = c.x; v.col[1] = c.y; v.col[2] = c.z; v.texId = 0.0f;
        v.loc[0] = p.x; v.loc[1] = p.y; v.loc[2] = p.z; v.bump = 0.0f; // object-space copy
        v.prv[0] = p.x; v.prv[1] = p.y; v.prv[2] = p.z; v.emissive = 0.0f;
        V.push_back(v);
    }

    static void setTexture(std::vector<Vertex>& V, float texId, float bump) {
        for (auto& v : V) { v.texId = texId; v.bump = bump; }
    }
    static void setEmissive(std::vector<Vertex>& V, float strength) {
        for (auto& v : V) v.emissive = strength;
    }

    static void genSphere(std::vector<Vertex>& V, std::vector<uint32_t>& I, float radius,
                          Vec3 col, float refl, float mat, int stacks = 24, int slices = 48) {
        const float PI = 3.14159265358979f;
        uint32_t base = (uint32_t)V.size();
        for (int i = 0; i <= stacks; ++i) {
            float phi = (float)i / stacks * PI;
            for (int j = 0; j <= slices; ++j) {
                float th = (float)j / slices * 2.0f * PI;
                Vec3 n{ std::sin(phi) * std::cos(th), std::cos(phi), std::sin(phi) * std::sin(th) };
                pushV(V, n * radius, n, col, refl, mat);
            }
        }
        int ring = slices + 1;
        for (int i = 0; i < stacks; ++i)
            for (int j = 0; j < slices; ++j) {
                uint32_t a = base + i * ring + j, b = base + (i + 1) * ring + j;
                I.push_back(a); I.push_back(b); I.push_back(a + 1);
                I.push_back(a + 1); I.push_back(b); I.push_back(b + 1);
            }
    }

    static void genBox(std::vector<Vertex>& V, std::vector<uint32_t>& I, float s,
                       Vec3 col, float refl, float mat) {
        Vec3 nrm[6] = { {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };
        float h = s * 0.5f;
        for (int f = 0; f < 6; ++f) {
            Vec3 N = nrm[f], u, w;
            if (std::fabs(N.y) > 0.5f) { u = { 1,0,0 }; w = { 0,0,1 }; }
            else { u = { 0,1,0 }; w = cross(N, u); }
            Vec3 c = N * h;
            Vec3 p0 = c - u * h - w * h, p1 = c + u * h - w * h, p2 = c + u * h + w * h, p3 = c - u * h + w * h;
            uint32_t b = (uint32_t)V.size();
            pushV(V, p0, N, col, refl, mat); pushV(V, p1, N, col, refl, mat);
            pushV(V, p2, N, col, refl, mat); pushV(V, p3, N, col, refl, mat);
            I.push_back(b); I.push_back(b + 1); I.push_back(b + 2);
            I.push_back(b); I.push_back(b + 2); I.push_back(b + 3);
        }
    }

    static void genPyramid(std::vector<Vertex>& V, std::vector<uint32_t>& I, float base, float height,
                           Vec3 col, float refl, float mat) {
        float h = base * 0.5f;
        Vec3 c0{ -h,0,-h }, c1{ h,0,-h }, c2{ h,0,h }, c3{ -h,0,h }, ap{ 0,height,0 };
        uint32_t b = (uint32_t)V.size(); Vec3 nd{ 0,-1,0 };
        pushV(V, c0, nd, col, refl, mat); pushV(V, c1, nd, col, refl, mat);
        pushV(V, c2, nd, col, refl, mat); pushV(V, c3, nd, col, refl, mat);
        I.push_back(b); I.push_back(b + 2); I.push_back(b + 1);
        I.push_back(b); I.push_back(b + 3); I.push_back(b + 2);
        Vec3 corners[4] = { c0,c1,c2,c3 };
        for (int k = 0; k < 4; ++k) {
            Vec3 a = corners[k], d = corners[(k + 1) % 4];
            Vec3 N = normalize(cross(d - a, ap - a));
            uint32_t t = (uint32_t)V.size();
            pushV(V, a, N, col, refl, mat); pushV(V, d, N, col, refl, mat); pushV(V, ap, N, col, refl, mat);
            I.push_back(t); I.push_back(t + 1); I.push_back(t + 2);
        }
    }

    static void genTetra(std::vector<Vertex>& V, std::vector<uint32_t>& I, float s,
                         Vec3 col, float refl, float mat) {
        Vec3 v[4] = { {1,1,1},{-1,-1,1},{-1,1,-1},{1,-1,-1} };
        for (int i = 0; i < 4; ++i) v[i] = v[i] * (s * 0.5f);
        int faces[4][3] = { {0,1,2},{0,3,1},{0,2,3},{1,3,2} };
        for (int f = 0; f < 4; ++f) {
            Vec3 a = v[faces[f][0]], b = v[faces[f][1]], c = v[faces[f][2]];
            Vec3 N = normalize(cross(b - a, c - a));
            if (dot(N, a) < 0) N = N * -1.0f;
            uint32_t t = (uint32_t)V.size();
            pushV(V, a, N, col, refl, mat); pushV(V, b, N, col, refl, mat); pushV(V, c, N, col, refl, mat);
            I.push_back(t); I.push_back(t + 1); I.push_back(t + 2);
        }
    }

    static void genSuperTorus(std::vector<Vertex>& V, std::vector<uint32_t>& I,
                              float R, float r, float e, float n, int ures, int vres,
                              Vec3 col, float refl, float mat) {
        const float PI = 3.14159265358979f;
        auto sp = [](float c, float p) { float s = c < 0 ? -1.f : 1.f; return s * std::pow(std::fabs(c), p); };
        auto P = [&](int i, int j) {
            float u = -PI + 2 * PI * (float)((i % ures + ures) % ures) / ures;
            float v = -PI + 2 * PI * (float)((j % vres + vres) % vres) / vres;
            float cu = std::cos(u), su = std::sin(u), cv = std::cos(v), sv = std::sin(v);
            float a = R + r * sp(cu, e);
            return Vec3{ a * sp(cv, n), r * sp(su, e), a * sp(sv, n) };
        };
        uint32_t base = (uint32_t)V.size();
        for (int i = 0; i <= ures; ++i)
            for (int j = 0; j <= vres; ++j) {
                Vec3 p = P(i, j);
                Vec3 du = P(i + 1, j) - P(i - 1, j);
                Vec3 dv = P(i, j + 1) - P(i, j - 1);
                Vec3 nn = normalize(cross(du, dv));
                pushV(V, p, nn, col, refl, mat);
            }
        int stride = vres + 1;
        for (int i = 0; i < ures; ++i)
            for (int j = 0; j < vres; ++j) {
                uint32_t a = base + i * stride + j, b = base + (i + 1) * stride + j;
                I.push_back(a); I.push_back(b); I.push_back(a + 1);
                I.push_back(a + 1); I.push_back(b); I.push_back(b + 1);
            }
    }

    static void genFloor(std::vector<Vertex>& V, std::vector<uint32_t>& I, float half, float refl) {
        Vec3 n{ 0,1,0 }; Vec3 col{ 0.8f,0.8f,0.8f };
        Vec3 c[4] = { {-half,0,-half},{half,0,-half},{half,0,half},{-half,0,half} };
        uint32_t b = (uint32_t)V.size();
        for (auto& cc : c) pushV(V, cc, n, col, refl, 0.0f);
        I.push_back(b); I.push_back(b + 1); I.push_back(b + 2);
        I.push_back(b); I.push_back(b + 2); I.push_back(b + 3);
    }

    // -------------------------------------------------------------------------
    //  Scene assembly (object list + merged buffers)
    // -------------------------------------------------------------------------
    void addObject(std::vector<Vertex>& base, std::vector<uint32_t>& localIdx, SceneObject proto) {
        proto.vertexOffset = totalVerts;
        for (uint32_t id : localIdx) sceneIndices.push_back(proto.vertexOffset + id);
        totalVerts += (uint32_t)base.size();
        proto.base = std::move(base);
        objects.push_back(std::move(proto));
    }

    void createScene() {
        const float PI = 3.14159265358979f;

        // Floor (static, checker texture).
        { std::vector<Vertex> v; std::vector<uint32_t> idx; genFloor(v, idx, 22.0f, 0.35f);
          setTexture(v, 1.0f, 0.0f);
          SceneObject o; addObject(v, idx, o); }

        // Central super-torus: iridescent metal (rainbow sheen that shifts with angle).
        { std::vector<Vertex> v; std::vector<uint32_t> idx;
          genSuperTorus(v, idx, 2.0f, 0.62f, 0.6f, 0.6f, 72, 36, { 0.85f,0.78f,0.70f }, 0.40f, 4.0f);
          SceneObject o; o.orbitCenter = { 0,1.9f,0 }; o.spinAxis = { 0.25f,1.0f,0.0f }; o.spinSpeed = 0.5f;
          addObject(v, idx, o); }

        // Ring of mixed shapes, each spinning, the whole ring orbiting slowly.
        const int N = 6;
        for (int i = 0; i < N; ++i) {
            float phase = (float)i / N * 2.0f * PI;
            std::vector<Vertex> v; std::vector<uint32_t> idx;
            SceneObject o;
            o.orbitCenter = { 0,1.35f,0 }; o.orbitRadius = 4.8f; o.orbitSpeed = 0.13f; o.orbitPhase = phase;
            switch (i) {
            case 0: genBox(v, idx, 1.3f, { 0.70f,0.45f,0.25f }, 0.08f, 1.0f);       // wood
                    setTexture(v, 3.0f, 0.0f);
                    o.spinAxis = { 0.3f,1,0.2f }; o.spinSpeed = 0.9f; break;
            case 1: genPyramid(v, idx, 1.5f, 1.7f, { 0.55f,0.55f,0.58f }, 0.08f, 1.0f); // granite
                    setTexture(v, 4.0f, 0.30f);
                    o.spinAxis = { 0,1,0 }; o.spinSpeed = 0.8f; break;
            case 2: genTetra(v, idx, 1.7f, { 0.85f,0.80f,0.75f }, 0.12f, 1.0f);     // marble
                    setTexture(v, 2.0f, 0.12f);
                    o.spinAxis = { 0.4f,1,0.1f }; o.spinSpeed = 1.1f; break;
            case 3: genSphere(v, idx, 0.85f, { 1.0f,1.0f,1.0f }, 0.0f, 2.0f);       // glass (dispersion)
                    o.spinSpeed = 0.0f; break;
            case 4: genSphere(v, idx, 0.85f, { 0.20f,0.80f,1.00f }, 0.0f, 1.0f);   // emissive plasma
                    setEmissive(v, 2.5f);
                    o.spinAxis = { 0,1,0 }; o.spinSpeed = 0.25f; break;
            default: genSphere(v, idx, 0.85f, { 0.65f,0.42f,0.20f }, 0.08f, 1.0f);  // wood
                    setTexture(v, 3.0f, 0.0f);
                    o.spinAxis = { 0,1,0 }; o.spinSpeed = 0.25f; break;
            }
            addObject(v, idx, o);
        }

        // Moving area light (emissive). Follows curLightPos.
        { std::vector<Vertex> v; std::vector<uint32_t> idx;
          genSphere(v, idx, 1.2f, { 1.0f,0.95f,0.85f }, 0.0f, 3.0f, 16, 32);
          SceneObject o; o.isLight = true; addObject(v, idx, o); }

        indexCount = (uint32_t)sceneIndices.size();
        sceneVerts.resize(totalVerts);

        VkBufferUsageFlags geomUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        vertexBuffer = createBuffer(totalVerts * sizeof(Vertex), geomUsage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        indexBuffer = createBuffer(sceneIndices.size() * sizeof(uint32_t), geomUsage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        uploadToBuffer(indexBuffer, sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t));

        updateScene(0.0f); // fill + upload initial vertex positions

        std::printf("Scene: %u objects, %u vertices, %u triangles\n",
            (uint32_t)objects.size(), totalVerts, indexCount / 3);
    }

    // Recompute world-space vertices for the current time and upload them.
    void updateScene(float t) {
        curLightPos = { std::cos(t * 0.35f) * 6.0f, 7.0f, std::sin(t * 0.35f) * 6.0f };

        for (auto& o : objects) {
            Mat3 R; // identity by default
            if (o.spinSpeed != 0.0f) R = rotAxis(o.spinAxis, o.spinSpeed * t);
            Vec3 pos;
            if (o.isLight) pos = curLightPos;
            else pos = o.orbitCenter + Vec3{ std::cos(o.orbitSpeed * t + o.orbitPhase) * o.orbitRadius,
                                             0.0f,
                                             std::sin(o.orbitSpeed * t + o.orbitPhase) * o.orbitRadius };

            // Previous-frame transform (for motion vectors). Identity on first frame.
            Mat3 Rp = o.prevInit ? o.prevRot : R;
            Vec3 pp = o.prevInit ? o.prevPos : pos;

            for (size_t k = 0; k < o.base.size(); ++k) {
                const Vertex& bv = o.base[k];
                Vec3 lp{ bv.loc[0], bv.loc[1], bv.loc[2] };
                Vec3 ln{ bv.nrm[0], bv.nrm[1], bv.nrm[2] };
                Vec3 wp = mul3(R, lp) + pos;
                Vec3 wn = mul3(R, ln);
                Vec3 pw = mul3(Rp, lp) + pp;          // previous world position
                Vertex& dv = sceneVerts[o.vertexOffset + k];
                dv = bv;
                dv.pos[0] = wp.x; dv.pos[1] = wp.y; dv.pos[2] = wp.z;
                dv.nrm[0] = wn.x; dv.nrm[1] = wn.y; dv.nrm[2] = wn.z;
                dv.prv[0] = pw.x; dv.prv[1] = pw.y; dv.prv[2] = pw.z;
            }
            o.prevRot = R; o.prevPos = pos; o.prevInit = true;
        }
        uploadToBuffer(vertexBuffer, sceneVerts.data(), totalVerts * sizeof(Vertex));
    }

    // -------------------------------------------------------------------------
    //  Acceleration structures (created once, rebuilt every frame)
    // -------------------------------------------------------------------------
    void createAccelStructures() {
        blasTriCount = indexCount / 3;

        blasGeom = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        blasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        blasGeom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        blasGeom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        blasGeom.geometry.triangles.vertexData.deviceAddress = bufferAddress(vertexBuffer.buf);
        blasGeom.geometry.triangles.vertexStride = sizeof(Vertex);
        blasGeom.geometry.triangles.maxVertex = totalVerts - 1;
        blasGeom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        blasGeom.geometry.triangles.indexData.deviceAddress = bufferAddress(indexBuffer.buf);

        VkAccelerationStructureBuildGeometryInfoKHR bi{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1; bi.pGeometries = &blasGeom;

        VkAccelerationStructureBuildSizesInfoKHR bs{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        pvkGetAccelerationStructureBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &bi, &blasTriCount, &bs);

        blasBuffer = createBuffer(bs.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR bci{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        bci.buffer = blasBuffer.buf; bci.size = bs.accelerationStructureSize;
        bci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vkCheck(pvkCreateAccelerationStructure(dev, &bci, nullptr, &blas), "create BLAS");

        VkDeviceSize scratchSize = bs.buildScratchSize;

        // Instance referencing the BLAS (identity transform).
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
        addrInfo.accelerationStructure = blas;
        VkDeviceAddress blasAddr = pvkGetAccelerationStructureDeviceAddress(dev, &addrInfo);

        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1.0f; inst.transform.matrix[1][1] = 1.0f; inst.transform.matrix[2][2] = 1.0f;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blasAddr;
        asInstance = createBuffer(sizeof(inst),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        uploadToBuffer(asInstance, &inst, sizeof(inst));

        tlasGeom = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
        tlasGeom.geometry.instances.data.deviceAddress = bufferAddress(asInstance.buf);

        VkAccelerationStructureBuildGeometryInfoKHR ti{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        ti.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        ti.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        ti.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        ti.geometryCount = 1; ti.pGeometries = &tlasGeom;

        uint32_t one = 1;
        VkAccelerationStructureBuildSizesInfoKHR ts{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        pvkGetAccelerationStructureBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &ti, &one, &ts);

        tlasBuffer = createBuffer(ts.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR tci{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        tci.buffer = tlasBuffer.buf; tci.size = ts.accelerationStructureSize;
        tci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vkCheck(pvkCreateAccelerationStructure(dev, &tci, nullptr, &tlas), "create TLAS");

        if (ts.buildScratchSize > scratchSize) scratchSize = ts.buildScratchSize;
        asScratch = createBuffer(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        rebuildAccel(); // initial build so the TLAS is valid before the first trace
    }

    void rebuildAccel() {
        VkDeviceAddress scratch = bufferAddress(asScratch.buf);

        VkAccelerationStructureBuildGeometryInfoKHR bi{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.dstAccelerationStructure = blas;
        bi.geometryCount = 1; bi.pGeometries = &blasGeom;
        bi.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR br{}; br.primitiveCount = blasTriCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pbr = &br;

        VkAccelerationStructureBuildGeometryInfoKHR ti{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        ti.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        ti.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        ti.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        ti.dstAccelerationStructure = tlas;
        ti.geometryCount = 1; ti.pGeometries = &tlasGeom;
        ti.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR tr{}; tr.primitiveCount = 1;
        const VkAccelerationStructureBuildRangeInfoKHR* ptr = &tr;

        VkCommandBuffer c = beginOneShot();
        pvkCmdBuildAccelerationStructures(c, 1, &bi, &pbr);
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        vkCmdPipelineBarrier(c,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0, 1, &mb, 0, nullptr, 0, nullptr);
        pvkCmdBuildAccelerationStructures(c, 1, &ti, &ptr);
        endOneShot(c);
    }

    // -------------------------------------------------------------------------
    //  Storage image the ray tracer writes into
    // -------------------------------------------------------------------------
    void createStorageImage() {
        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;       // matches rgba8 in the shader
        ici.extent = { swapExtent.width, swapExtent.height, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCheck(vkCreateImage(dev, &ici, nullptr, &storageImage), "create storage image");

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, storageImage, &req);
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkCheck(vkAllocateMemory(dev, &ai, nullptr, &storageMem), "alloc storage image");
        vkBindImageMemory(dev, storageImage, storageMem, 0);

        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = storageImage;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ici.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCheck(vkCreateImageView(dev, &vci, nullptr, &storageView), "storage view");

        // ---- TAA history images (float, ping-pong, kept in GENERAL) ----
        for (int i = 0; i < 2; ++i) {
            VkImageCreateInfo aci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            aci.imageType = VK_IMAGE_TYPE_2D;
            aci.format = VK_FORMAT_R32G32B32A32_SFLOAT;  // matches rgba32f in the shader
            aci.extent = { swapExtent.width, swapExtent.height, 1 };
            aci.mipLevels = 1;
            aci.arrayLayers = 1;
            aci.samples = VK_SAMPLE_COUNT_1_BIT;
            aci.tiling = VK_IMAGE_TILING_OPTIMAL;
            aci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            aci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            vkCheck(vkCreateImage(dev, &aci, nullptr, &histImg[i]), "create hist image");

            VkMemoryRequirements areq;
            vkGetImageMemoryRequirements(dev, histImg[i], &areq);
            VkMemoryAllocateInfo aai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            aai.allocationSize = areq.size;
            aai.memoryTypeIndex = findMemoryType(areq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vkCheck(vkAllocateMemory(dev, &aai, nullptr, &histMem[i]), "alloc hist image");
            vkBindImageMemory(dev, histImg[i], histMem[i], 0);

            VkImageViewCreateInfo avci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            avci.image = histImg[i];
            avci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            avci.format = aci.format;
            avci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCheck(vkCreateImageView(dev, &avci, nullptr, &histView[i]), "hist view");

            VkCommandBuffer c = beginOneShot();
            imageBarrier(c, histImg[i],
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
            endOneShot(c);
        }

        // ---- Denoiser images (float, GENERAL): two colour buffers + one guide ----
        VkImage* dimgs[3] = { &denImg[0], &denImg[1], &guideImg };
        VkDeviceMemory* dmems[3] = { &denMem[0], &denMem[1], &guideMem };
        VkImageView* dviews[3] = { &denView[0], &denView[1], &guideView };
        for (int i = 0; i < 3; ++i) {
            VkImageCreateInfo dci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            dci.imageType = VK_IMAGE_TYPE_2D;
            dci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            dci.extent = { swapExtent.width, swapExtent.height, 1 };
            dci.mipLevels = 1;
            dci.arrayLayers = 1;
            dci.samples = VK_SAMPLE_COUNT_1_BIT;
            dci.tiling = VK_IMAGE_TILING_OPTIMAL;
            dci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            dci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            vkCheck(vkCreateImage(dev, &dci, nullptr, dimgs[i]), "create denoise image");

            VkMemoryRequirements dreq;
            vkGetImageMemoryRequirements(dev, *dimgs[i], &dreq);
            VkMemoryAllocateInfo dai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            dai.allocationSize = dreq.size;
            dai.memoryTypeIndex = findMemoryType(dreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vkCheck(vkAllocateMemory(dev, &dai, nullptr, dmems[i]), "alloc denoise image");
            vkBindImageMemory(dev, *dimgs[i], *dmems[i], 0);

            VkImageViewCreateInfo dvci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            dvci.image = *dimgs[i];
            dvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            dvci.format = dci.format;
            dvci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCheck(vkCreateImageView(dev, &dvci, nullptr, dviews[i]), "denoise view");

            VkCommandBuffer c = beginOneShot();
            imageBarrier(c, *dimgs[i],
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
            endOneShot(c);
        }
    }

    void createUniformBuffer() {
        ubo = createBuffer(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // -------------------------------------------------------------------------
    //  Descriptors
    // -------------------------------------------------------------------------
    void createDescriptors() {
        std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };

        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = (uint32_t)bindings.size();
        lci.pBindings = bindings.data();
        vkCheck(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &descLayout), "descriptor layout");

        // Compute (denoiser) layout: colorIn, colorOut, guide, outLDR + push constant.
        std::array<VkDescriptorSetLayoutBinding, 4> cb{};
        cb[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        cb[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        cb[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        cb[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo clci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        clci.bindingCount = (uint32_t)cb.size();
        clci.pBindings = cb.data();
        vkCheck(vkCreateDescriptorSetLayout(dev, &clci, nullptr, &compDescLayout), "compute descriptor layout");

        std::array<VkDescriptorPoolSize, 4> sizes{ {
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12 },   // RT(4) + 2 compute sets(4 each)
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        } };
        VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pci.maxSets = 3;   // 1 RT set + 2 compute sets
        pci.poolSizeCount = (uint32_t)sizes.size();
        pci.pPoolSizes = sizes.data();
        vkCheck(vkCreateDescriptorPool(dev, &pci, nullptr, &descPool), "descriptor pool");

        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = descPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &descLayout;
        vkCheck(vkAllocateDescriptorSets(dev, &ai, &descSet), "alloc descriptor set");

        // TLAS
        VkWriteDescriptorSetAccelerationStructureKHR asWrite{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &tlas;

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView = denView[0];   // ray-gen writes linear HDR here for the denoiser
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo guideInfo{};
        guideInfo.imageView = guideView;
        guideInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo histInfo0{};
        histInfo0.imageView = histView[0];
        histInfo0.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo histInfo1{};
        histInfo1.imageView = histView[1];
        histInfo1.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo uboInfo{ ubo.buf, 0, sizeof(UBO) };
        VkDescriptorBufferInfo vInfo{ vertexBuffer.buf, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo iInfo{ indexBuffer.buf, 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 8> writes{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].pNext = &asWrite;
        writes[0].dstSet = descSet; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = descSet; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &imgInfo;

        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = descSet; writes[2].dstBinding = 2; writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[2].pBufferInfo = &uboInfo;

        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = descSet; writes[3].dstBinding = 3; writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].pBufferInfo = &vInfo;

        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[4].dstSet = descSet; writes[4].dstBinding = 4; writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[4].pBufferInfo = &iInfo;

        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[5].dstSet = descSet; writes[5].dstBinding = 5; writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[5].pImageInfo = &histInfo0;

        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[6].dstSet = descSet; writes[6].dstBinding = 6; writes[6].descriptorCount = 1;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[6].pImageInfo = &histInfo1;

        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[7].dstSet = descSet; writes[7].dstBinding = 7; writes[7].descriptorCount = 1;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[7].pImageInfo = &guideInfo;

        vkUpdateDescriptorSets(dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);

        // ---- Compute (denoiser) descriptor sets: two for ping-pong ----
        VkDescriptorSetLayout cl[2] = { compDescLayout, compDescLayout };
        VkDescriptorSetAllocateInfo cai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        cai.descriptorPool = descPool;
        cai.descriptorSetCount = 2;
        cai.pSetLayouts = cl;
        vkCheck(vkAllocateDescriptorSets(dev, &cai, compSet), "alloc compute sets");

        VkDescriptorImageInfo den0{ VK_NULL_HANDLE, denView[0], VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo den1{ VK_NULL_HANDLE, denView[1], VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo gInf{ VK_NULL_HANDLE, guideView,  VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo ldr { VK_NULL_HANDLE, storageView, VK_IMAGE_LAYOUT_GENERAL };
        // set 0: in = den0, out = den1 ; set 1: in = den1, out = den0
        VkDescriptorImageInfo* inInfo[2]  = { &den0, &den1 };
        VkDescriptorImageInfo* outInfo[2] = { &den1, &den0 };

        std::array<VkWriteDescriptorSet, 8> cw{};
        for (int s = 0; s < 2; ++s) {
            int b = s * 4;
            cw[b + 0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            cw[b + 0].dstSet = compSet[s]; cw[b + 0].dstBinding = 0; cw[b + 0].descriptorCount = 1;
            cw[b + 0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; cw[b + 0].pImageInfo = inInfo[s];
            cw[b + 1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            cw[b + 1].dstSet = compSet[s]; cw[b + 1].dstBinding = 1; cw[b + 1].descriptorCount = 1;
            cw[b + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; cw[b + 1].pImageInfo = outInfo[s];
            cw[b + 2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            cw[b + 2].dstSet = compSet[s]; cw[b + 2].dstBinding = 2; cw[b + 2].descriptorCount = 1;
            cw[b + 2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; cw[b + 2].pImageInfo = &gInf;
            cw[b + 3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            cw[b + 3].dstSet = compSet[s]; cw[b + 3].dstBinding = 3; cw[b + 3].descriptorCount = 1;
            cw[b + 3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; cw[b + 3].pImageInfo = &ldr;
        }
        vkUpdateDescriptorSets(dev, (uint32_t)cw.size(), cw.data(), 0, nullptr);
    }

    // -------------------------------------------------------------------------
    //  Ray tracing pipeline
    // -------------------------------------------------------------------------
    VkShaderModule loadShader(const char* path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error(std::string("Cannot open shader: ") + path);
        size_t size = (size_t)f.tellg();
        std::vector<char> code(size);
        f.seekg(0);
        f.read(code.data(), size);

        VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        ci.codeSize = size;
        ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule mod;
        vkCheck(vkCreateShaderModule(dev, &ci, nullptr, &mod), "create shader module");
        return mod;
    }

    void createRayTracingPipeline() {
        VkPipelineLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &descLayout;
        vkCheck(vkCreatePipelineLayout(dev, &lci, nullptr, &pipelineLayout), "pipeline layout");

        VkShaderModule rgen = loadShader("shaders/raygen.rgen.spv");
        VkShaderModule miss = loadShader("shaders/miss.rmiss.spv");
        VkShaderModule smiss = loadShader("shaders/shadow.rmiss.spv");
        VkShaderModule chit = loadShader("shaders/closesthit.rchit.spv");

        std::array<VkPipelineShaderStageCreateInfo, 4> stages{};
        auto stage = [](VkShaderStageFlagBits s, VkShaderModule m) {
            VkPipelineShaderStageCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
            ci.stage = s; ci.module = m; ci.pName = "main"; return ci;
        };
        stages[0] = stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen);
        stages[1] = stage(VK_SHADER_STAGE_MISS_BIT_KHR, miss);
        stages[2] = stage(VK_SHADER_STAGE_MISS_BIT_KHR, smiss);
        stages[3] = stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chit);

        std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{};
        auto general = [](uint32_t idx) {
            VkRayTracingShaderGroupCreateInfoKHR g{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
            g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            g.generalShader = idx;
            g.closestHitShader = VK_SHADER_UNUSED_KHR;
            g.anyHitShader = VK_SHADER_UNUSED_KHR;
            g.intersectionShader = VK_SHADER_UNUSED_KHR;
            return g;
        };
        groups[0] = general(0); // raygen
        groups[1] = general(1); // miss
        groups[2] = general(2); // shadow miss
        groups[3] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[3].generalShader = VK_SHADER_UNUSED_KHR;
        groups[3].closestHitShader = 3;
        groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
        groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

        VkRayTracingPipelineCreateInfoKHR pci{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
        pci.stageCount = (uint32_t)stages.size();
        pci.pStages = stages.data();
        pci.groupCount = (uint32_t)groups.size();
        pci.pGroups = groups.data();
        pci.maxPipelineRayRecursionDepth = 1; // all traceRay calls happen in ray-gen
        pci.layout = pipelineLayout;
        vkCheck(pvkCreateRayTracingPipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE,
            1, &pci, nullptr, &rtPipeline), "create RT pipeline");

        vkDestroyShaderModule(dev, rgen, nullptr);
        vkDestroyShaderModule(dev, miss, nullptr);
        vkDestroyShaderModule(dev, smiss, nullptr);
        vkDestroyShaderModule(dev, chit, nullptr);
    }

    // -------------------------------------------------------------------------
    //  Denoiser compute pipeline (a-trous wavelet filter + tone-map)
    // -------------------------------------------------------------------------
    struct AtrousPC { int32_t sizeX, sizeY, step, finalPass; float exposure; };

    void createComputePipeline() {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(AtrousPC);

        VkPipelineLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &compDescLayout;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pcr;
        vkCheck(vkCreatePipelineLayout(dev, &lci, nullptr, &compPipelineLayout), "compute pipeline layout");

        VkShaderModule cs = loadShader("shaders/atrous.comp.spv");
        VkPipelineShaderStageCreateInfo st{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        st.module = cs;
        st.pName = "main";

        VkComputePipelineCreateInfo cci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cci.stage = st;
        cci.layout = compPipelineLayout;
        vkCheck(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cci, nullptr, &compPipeline),
            "create compute pipeline");

        vkDestroyShaderModule(dev, cs, nullptr);
    }

    // -------------------------------------------------------------------------
    //  Shader binding table
    // -------------------------------------------------------------------------
    void createShaderBindingTable() {
        const uint32_t handleSize = rtProps.shaderGroupHandleSize;
        const uint32_t handleAligned = alignUp(handleSize, rtProps.shaderGroupHandleAlignment);
        const uint32_t baseAlign = rtProps.shaderGroupBaseAlignment;

        const uint32_t rgenCount = 1, missCount = 2, hitCount = 1;
        const uint32_t totalGroups = rgenCount + missCount + hitCount;

        // Region strides/sizes (each region base must be baseAlignment-aligned;
        // for ray-gen, size must equal stride).
        rgenRegion.stride = alignUp(handleAligned, baseAlign);
        rgenRegion.size   = rgenRegion.stride;
        missRegion.stride = handleAligned;
        missRegion.size   = alignUp(missCount * handleAligned, baseAlign);
        hitRegion.stride  = handleAligned;
        hitRegion.size    = alignUp(hitCount * handleAligned, baseAlign);

        VkDeviceSize sbtSize = rgenRegion.size + missRegion.size + hitRegion.size;

        sbtBuffer = createBuffer(sbtSize,
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        std::vector<uint8_t> handles(totalGroups * handleSize);
        vkCheck(pvkGetRayTracingShaderGroupHandles(dev, rtPipeline, 0, totalGroups,
            handles.size(), handles.data()), "get shader group handles");

        VkDeviceAddress base = bufferAddress(sbtBuffer.buf);
        rgenRegion.deviceAddress = base;
        missRegion.deviceAddress = base + rgenRegion.size;
        hitRegion.deviceAddress  = base + rgenRegion.size + missRegion.size;

        uint8_t* mapped = nullptr;
        vkMapMemory(dev, sbtBuffer.mem, 0, sbtSize, 0, (void**)&mapped);
        auto getHandle = [&](uint32_t i) { return handles.data() + i * handleSize; };

        // ray-gen (group 0)
        std::memcpy(mapped, getHandle(0), handleSize);
        // miss (groups 1, 2)
        uint8_t* missDst = mapped + rgenRegion.size;
        std::memcpy(missDst, getHandle(1), handleSize);
        std::memcpy(missDst + missRegion.stride, getHandle(2), handleSize);
        // hit (group 3)
        uint8_t* hitDst = mapped + rgenRegion.size + missRegion.size;
        std::memcpy(hitDst, getHandle(3), handleSize);

        vkUnmapMemory(dev, sbtBuffer.mem);
    }

    // -------------------------------------------------------------------------
    //  Per-frame
    // -------------------------------------------------------------------------
    void updateUniforms(float t) {
        totalFrame++;   // free-running; drives the RNG and the ping-pong parity

        Vec3 center{ 0.0f, 1.0f, 0.0f };
        Vec3 up{ 0.0f, 1.0f, 0.0f };
        float cp = std::cos(camPitch), sp = std::sin(camPitch);
        float cy = std::cos(camYaw),   sy = std::sin(camYaw);
        Vec3 eye{ center.x + camDist * cp * cy,
                  center.y + camDist * sp,
                  center.z + camDist * cp * sy };

        Mat4 view = lookAtRH(eye, center, up);
        Mat4 proj = perspectiveVk(60.0f * 3.14159265f / 180.0f,
                                   (float)swapExtent.width / swapExtent.height, 0.1f, 100.0f);
        Mat4 viewProj = mul(proj, view);

        UBO data{};
        data.viewInverse  = inverse(view);
        data.projInverse  = inverse(proj);
        data.prevViewProj = haveHistory ? prevViewProj : viewProj;
        data.lightPos[0] = curLightPos.x; data.lightPos[1] = curLightPos.y; data.lightPos[2] = curLightPos.z;
        data.lightPos[3] = 1.2f;
        Vec3 pc = haveHistory ? prevCamPos : eye;
        data.prevCamPos[0] = pc.x; data.prevCamPos[1] = pc.y; data.prevCamPos[2] = pc.z;
        data.prevCamPos[3] = 0.085f;  // TAA blend alpha (lower = smoother, more lag)
        data.params[0] = t;
        data.params[1] = 8.0f;        // max bounces (glass needs a few)
        data.params[2] = 1.4f;        // light intensity
        data.params[3] = 9.0f;        // samples/frame (multiple of 3 for even dispersion)
        data.frame[1]  = totalFrame;
        data.frame[2]  = haveHistory ? 1u : 0u;   // historyValid
        data.frame[3]  = totalFrame & 1u;         // ping-pong parity
        data.lens[0]   = 0.0f;                     // aperture radius (0 = pinhole/sharp; >0 = DoF, noisy without a denoiser)
        data.lens[1]   = std::sqrt(dot(eye - center, eye - center)); // focus distance (centre sharp)
        data.lens[2]   = 1.1f;                      // exposure (pre tone-map)
        uploadToBuffer(ubo, &data, sizeof(data));

        // Remember this frame's camera for the next reprojection.
        prevViewProj = viewProj;
        prevCamPos   = eye;
        haveHistory  = true;
    }

    void imageBarrier(VkCommandBuffer c, VkImage img,
        VkImageLayout oldL, VkImageLayout newL,
        VkAccessFlags srcA, VkAccessFlags dstA,
        VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcAccessMask = srcA; b.dstAccessMask = dstA;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(c, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void drawFrame() {
        vkWaitForFences(dev, 1, &inFlight, VK_TRUE, UINT64_MAX);
        vkResetFences(dev, 1, &inFlight);

        uint32_t imageIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX,
            semImageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return;

        float t = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - startTime).count();

        updateScene(t);     // animate objects + light (CPU transform, upload vertices)
        rebuildAccel();     // rebuild BLAS/TLAS for the new positions
        updateUniforms(t);

        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &bi);

        // Output (rgba8) -> GENERAL; the denoiser's final compute pass writes it.
        imageBarrier(cmd, storageImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Make the previous frame's history writes visible to this frame's reads.
        for (int i = 0; i < 2; ++i)
            imageBarrier(cmd, histImg[i],
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            pipelineLayout, 0, 1, &descSet, 0, nullptr);

        pvkCmdTraceRays(cmd, &rgenRegion, &missRegion, &hitRegion, &callRegion,
            swapExtent.width, swapExtent.height, 1);

        // ---- Denoiser: a-trous compute passes on the ray-traced result ----
        // RT outputs (colour den[0], guide) must be visible to compute reads.
        imageBarrier(cmd, denImg[0],
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        imageBarrier(cmd, guideImg,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
        uint32_t gx = (swapExtent.width + 7) / 8, gy = (swapExtent.height + 7) / 8;
        float exposure = 1.1f;
        int passes = atrousPasses;

        if (passes <= 0) {
            // Denoiser off: a single pass with step 0 = pure tone-map copy.
            AtrousPC pc{ (int)swapExtent.width, (int)swapExtent.height, 0, 1, exposure };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                compPipelineLayout, 0, 1, &compSet[0], 0, nullptr);
            vkCmdPushConstants(cmd, compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, gx, gy, 1);
        } else {
            for (int i = 0; i < passes; ++i) {
                int setIdx = i & 1;                 // set0 reads den0, set1 reads den1
                int outIdx = 1 - setIdx;            // image this pass writes
                AtrousPC pc{ (int)swapExtent.width, (int)swapExtent.height,
                             1 << i, (i == passes - 1) ? 1 : 0, exposure };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    compPipelineLayout, 0, 1, &compSet[setIdx], 0, nullptr);
                vkCmdPushConstants(cmd, compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, gx, gy, 1);
                if (i < passes - 1)                 // make this output readable next pass
                    imageBarrier(cmd, denImg[outIdx],
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }
        }

        // Final compute pass wrote storageImage -> TRANSFER_SRC; swapchain -> TRANSFER_DST.
        imageBarrier(cmd, storageImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        imageBarrier(cmd, swapImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Blit (component-aware: keeps colours correct across RGBA/BGRA).
        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.srcOffsets[1] = { (int32_t)swapExtent.width, (int32_t)swapExtent.height, 1 };
        blit.dstOffsets[1] = { (int32_t)swapExtent.width, (int32_t)swapExtent.height, 1 };
        vkCmdBlitImage(cmd,
            storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_NEAREST);

        // Swapchain image -> PRESENT.
        imageBarrier(cmd, swapImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        vkEndCommandBuffer(cmd);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &semImageAvailable;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &semRenderFinished;
        vkCheck(vkQueueSubmit(queue, 1, &si, inFlight), "queue submit");

        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &semRenderFinished;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &imageIndex;
        vkQueuePresentKHR(queue, &pi);
    }

    void mainLoop() {
        MSG msg{};
        while (!quit) {
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) quit = true;
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if (quit) break;
            drawFrame();
        }
        vkDeviceWaitIdle(dev);
    }

    // -------------------------------------------------------------------------
    //  Teardown
    // -------------------------------------------------------------------------
    void destroyBuffer(Buffer& b) {
        if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
        b.buf = VK_NULL_HANDLE; b.mem = VK_NULL_HANDLE;
    }

    void cleanup() {
        destroyBuffer(sbtBuffer);
        if (compPipeline)       vkDestroyPipeline(dev, compPipeline, nullptr);
        if (compPipelineLayout) vkDestroyPipelineLayout(dev, compPipelineLayout, nullptr);
        if (compDescLayout)     vkDestroyDescriptorSetLayout(dev, compDescLayout, nullptr);
        if (rtPipeline)     vkDestroyPipeline(dev, rtPipeline, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
        if (descPool)       vkDestroyDescriptorPool(dev, descPool, nullptr);
        if (descLayout)     vkDestroyDescriptorSetLayout(dev, descLayout, nullptr);

        if (storageView) vkDestroyImageView(dev, storageView, nullptr);
        if (storageImage) vkDestroyImage(dev, storageImage, nullptr);
        if (storageMem) vkFreeMemory(dev, storageMem, nullptr);

        for (int i = 0; i < 2; ++i) {
            if (histView[i]) vkDestroyImageView(dev, histView[i], nullptr);
            if (histImg[i])  vkDestroyImage(dev, histImg[i], nullptr);
            if (histMem[i])  vkFreeMemory(dev, histMem[i], nullptr);
        }
        for (int i = 0; i < 2; ++i) {
            if (denView[i]) vkDestroyImageView(dev, denView[i], nullptr);
            if (denImg[i])  vkDestroyImage(dev, denImg[i], nullptr);
            if (denMem[i])  vkFreeMemory(dev, denMem[i], nullptr);
        }
        if (guideView) vkDestroyImageView(dev, guideView, nullptr);
        if (guideImg)  vkDestroyImage(dev, guideImg, nullptr);
        if (guideMem)  vkFreeMemory(dev, guideMem, nullptr);

        if (tlas) pvkDestroyAccelerationStructure(dev, tlas, nullptr);
        if (blas) pvkDestroyAccelerationStructure(dev, blas, nullptr);
        destroyBuffer(tlasBuffer);
        destroyBuffer(blasBuffer);
        destroyBuffer(asScratch);
        destroyBuffer(asInstance);
        destroyBuffer(ubo);
        destroyBuffer(vertexBuffer);
        destroyBuffer(indexBuffer);

        if (inFlight) vkDestroyFence(dev, inFlight, nullptr);
        if (semImageAvailable) vkDestroySemaphore(dev, semImageAvailable, nullptr);
        if (semRenderFinished) vkDestroySemaphore(dev, semRenderFinished, nullptr);
        if (cmdPool) vkDestroyCommandPool(dev, cmdPool, nullptr);

        if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
        if (dev) vkDestroyDevice(dev, nullptr);

        if (debugMessenger) {
            auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                instance, "vkDestroyDebugUtilsMessengerEXT");
            if (fn) fn(instance, debugMessenger, nullptr);
        }
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

// =============================================================================
//  Entry point
// =============================================================================
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    // Allocate a console so printf / validation output is visible.
    if (AllocConsole()) {
        FILE* f;
        freopen_s(&f, "CONOUT$", "w", stdout);
        freopen_s(&f, "CONOUT$", "w", stderr);
    }
    try {
        RayTracer app;
        app.run(hInst);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        MessageBoxA(nullptr, e.what(), "Vulkan Ray Tracer - Error", MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}
