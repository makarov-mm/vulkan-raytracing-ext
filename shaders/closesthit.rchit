#version 460
#extension GL_EXT_ray_tracing : require

struct HitPayload {
    vec3  hitPos;
    vec3  normal;
    vec3  albedo;
    vec3  prevHitPos;   // previous-frame world position (motion vectors)
    vec3  emission;
    float reflectivity;
    float matId;
    float t;
    float frontFace;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 attribs;

// Vertex: posRefl(xyz world pos, w refl), normal(xyz world, w matId),
//         color(xyz albedo, w texId), local(xyz object-space pos, w bump).
struct Vertex { vec4 posRefl; vec4 normal; vec4 color; vec4 local; vec4 prev; };
layout(std430, set = 0, binding = 3) readonly buffer Vertices { Vertex v[]; } vertices;
layout(std430, set = 0, binding = 4) readonly buffer Indices  { uint   i[]; } indices;

// ---- value noise (procedural textures are evaluated in object space) ----
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i + vec3(0,0,0)), n100 = hash13(i + vec3(1,0,0));
    float n010 = hash13(i + vec3(0,1,0)), n110 = hash13(i + vec3(1,1,0));
    float n001 = hash13(i + vec3(0,0,1)), n101 = hash13(i + vec3(1,0,1));
    float n011 = hash13(i + vec3(0,1,1)), n111 = hash13(i + vec3(1,1,1));
    return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y),
               mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int k = 0; k < 5; ++k) { s += a * vnoise(p); p *= 2.02; a *= 0.5; }
    return s;
}

vec3 texColor(int id, vec3 p, vec3 base) {
    if (id == 1) {                                  // checker
        float s = floor(p.x) + floor(p.z);
        return mix(vec3(0.85), vec3(0.10), mod(s, 2.0));
    } else if (id == 2) {                           // marble
        float t = fbm(p * 1.1);
        float veins = abs(sin((p.x + p.y * 0.6) * 1.6 + t * 5.0));
        veins = pow(veins, 0.35);
        return mix(base * 0.25, base, veins);
    } else if (id == 3) {                           // wood
        float r = length(p.xz) * 1.4 + fbm(p * 0.7) * 1.5;
        float rings = 0.5 + 0.5 * sin(r * 6.2831853);
        rings = pow(rings, 0.6);
        return mix(vec3(0.32,0.16,0.06), vec3(0.62,0.36,0.16), rings);
    } else if (id == 4) {                           // granite / speckle
        float n = fbm(p * 2.5);
        vec3 g = mix(vec3(0.18), vec3(0.55), n);
        float sp = step(0.78, hash13(floor(p * 10.0)));
        return mix(g, base, sp * 0.7);
    }
    return base;
}

vec3 applyBump(vec3 N, vec3 p, float strength) {
    float e = 0.04;
    float n0 = fbm(p);
    vec3 grad = vec3(fbm(p + vec3(e,0,0)) - n0,
                     fbm(p + vec3(0,e,0)) - n0,
                     fbm(p + vec3(0,0,e)) - n0) / e;
    vec3 tang = grad - dot(grad, N) * N;   // keep only the tangential part
    return normalize(N - strength * tang);
}

void main() {
    uint i0 = indices.i[3 * gl_PrimitiveID + 0];
    uint i1 = indices.i[3 * gl_PrimitiveID + 1];
    uint i2 = indices.i[3 * gl_PrimitiveID + 2];
    Vertex a = vertices.v[i0], b = vertices.v[i1], c = vertices.v[i2];
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 N    = normalize(a.normal.xyz * bary.x + b.normal.xyz * bary.y + c.normal.xyz * bary.z);
    vec3 base = a.color.xyz * bary.x + b.color.xyz * bary.y + c.color.xyz * bary.z;
    vec3 op   = a.local.xyz * bary.x + b.local.xyz * bary.y + c.local.xyz * bary.z; // object space
    vec3 pw   = a.prev.xyz  * bary.x + b.prev.xyz  * bary.y + c.prev.xyz  * bary.z; // prev world
    int   texId = int(a.color.w + 0.5);
    float bump  = a.local.w;
    float em    = a.prev.w;

    bool front = dot(N, gl_WorldRayDirectionEXT) < 0.0;
    if (!front) N = -N;

    vec3 albedo = texColor(texId, op, base);
    if (bump > 0.0001) N = applyBump(N, op, bump);

    vec3 emission = vec3(0.0);
    if (em > 0.0) {                       // glowing patches in the object's colour
        float g = smoothstep(0.35, 0.75, fbm(op * 2.6));
        emission = base * (0.3 + g) * em;
    }

    payload.hitPos       = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.normal       = N;
    payload.albedo       = albedo;
    payload.prevHitPos   = pw;
    payload.emission     = emission;
    payload.reflectivity = a.posRefl.w;
    payload.matId        = a.normal.w;
    payload.t            = gl_HitTEXT;
    payload.frontFace    = front ? 1.0 : 0.0;
}
