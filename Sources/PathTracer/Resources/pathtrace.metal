//===----------------------------------------------------------------------===//
// pathtrace.metal — Realtime PBR path tracer on Metal (hardware ray tracing)
//
// Structure mirrors PBRT / OpenMoonRay-style unidirectional path tracing:
//   * GGX (Trowbridge-Reitz) microfacet specular, metallic-roughness workflow
//   * Lambert diffuse, Fresnel-Schlick, height-correlated Smith masking-shadowing
//   * Visible-Normal-Distribution-Function (VNDF) importance sampling (Heitz 2018)
//   * Next-Event Estimation (direct light sampling of emissive triangles)
//   * Multiple Importance Sampling (power heuristic) between BSDF & light sampling
//   * Russian-roulette path termination, progressive accumulation
//
// Compiled at runtime via MTLDevice.makeLibrary(source:) so it needs no offline
// `metal` compiler (works with Command Line Tools only).
//===----------------------------------------------------------------------===//
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

constant float PI      = 3.14159265358979323846;
constant float INV_PI  = 0.31830988618379067154;
constant float EPS     = 1e-3;

//===----------------------------------------------------------------------===//
// Shared host/device data layouts. Must match the Swift mirrors exactly.
//===----------------------------------------------------------------------===//
struct Camera {
    float3 position;
    float3 right;
    float3 up;
    float3 forward;
    float  tanHalfFovY;
    float  aspect;
    float  aperture;     // lens radius for depth of field (0 = pinhole)
    float  focusDist;
};

struct Uniforms {
    Camera camera;
    uint2  imageSize;
    uint   frameIndex;       // 0-based; used for RNG decorrelation + accumulation
    uint   samplesPerFrame;
    uint   maxBounces;
    uint   numEmissive;
    uint   numTriangles;
    uint   flags;            // bit0: reset accumulation
    float3 background;       // constant environment radiance
    float  radianceClamp;    // per-sample luminance clamp (0 = disabled, unbiased)
};

struct Material {
    float3 baseColor;
    float3 emission;
    float  metallic;
    float  roughness;
    float  ior;
    float  texId;        // index into the albedo texture array, or -1 for none
};

// Per-pixel feature record exported for the neural-amplifier research (P5).
struct Feature {
    float3 hitPos;       // first-hit world position
    float3 normal;       // first-hit shading normal
    float3 wo;           // view direction (toward camera)
    float3 baseColor;
    float  metallic;
    float  roughness;
    float  hit;          // 1 if primary ray hit geometry, else 0
    float  _pad;
    float3 oneSample;    // single-path radiance estimate (1 spp) for this pixel
    float3 firstDir;     // first sampled bounce direction (the one cast ray)
};

//===----------------------------------------------------------------------===//
// RNG — PCG hash + LCG stream. Cheap, decorrelated per pixel/frame/dimension.
//===----------------------------------------------------------------------===//
struct RNG { uint state; };

inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float randf(thread RNG& r) {
    r.state = r.state * 747796405u + 2891336453u;
    uint word = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    uint res  = (word >> 22u) ^ word;
    return float(res) * (1.0 / 4294967296.0);
}

//===----------------------------------------------------------------------===//
// Orthonormal basis (Frisvad / Duff 2017) and sampling helpers.
//===----------------------------------------------------------------------===//
inline void onb(float3 n, thread float3& t, thread float3& b) {
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float bx = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * bx, -s * n.x);
    b = float3(bx, s + n.y * n.y * a, -n.y);
}
inline float3 toWorld(float3 v, float3 t, float3 b, float3 n) { return v.x * t + v.y * b + v.z * n; }
inline float3 toLocal(float3 v, float3 t, float3 b, float3 n) { return float3(dot(v, t), dot(v, b), dot(v, n)); }

inline float2 concentricDisk(float2 u) {
    float2 o = 2.0 * u - 1.0;
    if (o.x == 0.0 && o.y == 0.0) return float2(0.0);
    float r, theta;
    if (abs(o.x) > abs(o.y)) { r = o.x; theta = (PI / 4.0) * (o.y / o.x); }
    else                     { r = o.y; theta = (PI / 2.0) - (PI / 4.0) * (o.x / o.y); }
    return r * float2(cos(theta), sin(theta));
}
inline float3 cosineHemisphere(float2 u) {
    float2 d = concentricDisk(u);
    float z = sqrt(max(0.0, 1.0 - dot(d, d)));
    return float3(d.x, d.y, z);
}
inline float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

//===----------------------------------------------------------------------===//
// GGX microfacet (isotropic) — distribution, Smith masking, VNDF sampling.
//===----------------------------------------------------------------------===//
inline float ggxD(float3 h, float a) {
    float a2 = a * a;
    float nh = h.z;
    float d  = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-9);
}
inline float smithG1(float3 v, float a) {
    float c = abs(v.z);
    float t2 = max(0.0, 1.0 - c * c) / max(1e-8, c * c);
    float lambda = (-1.0 + sqrt(1.0 + a * a * t2)) * 0.5;
    return 1.0 / (1.0 + lambda);
}
inline float3 fresnelSchlick(float cosT, float3 F0) {
    float m = clamp(1.0 - cosT, 0.0, 1.0);
    float m2 = m * m;
    return F0 + (1.0 - F0) * (m2 * m2 * m);
}
// Sample a microfacet normal from the GGX VNDF (Heitz 2018). Ve in local frame.
inline float3 sampleGGXVNDF(float3 Ve, float a, float u1, float u2) {
    float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(float3(a * Nh.x, a * Nh.y, max(1e-6, Nh.z)));
}

//===----------------------------------------------------------------------===//
// BSDF — metallic-roughness principled-style (diffuse + GGX specular lobes).
// All vectors in the local shading frame (z = normal). wo, wi point away from
// the surface. Returns combined value `f` and combined pdf for MIS.
//===----------------------------------------------------------------------===//
struct BsdfSample { float3 wi; float3 f; float pdf; };

inline void bsdfWeights(Material m, thread float3& albedo, thread float3& F0, thread float& pSpec) {
    albedo = m.baseColor * (1.0 - m.metallic);
    F0 = mix(float3(0.04), m.baseColor, m.metallic);
    float ld = luminance(albedo);
    float ls = luminance(F0);
    pSpec = clamp(ls / (ld + ls + 1e-4), 0.1, 0.9);
}

inline void bsdfEval(Material m, float3 wo, float3 wi, thread float3& f, thread float& pdf) {
    f = float3(0.0); pdf = 0.0;
    if (wo.z <= 0.0 || wi.z <= 0.0) return;
    float3 albedo, F0; float pSpec;
    bsdfWeights(m, albedo, F0, pSpec);
    float a = max(1e-3, m.roughness * m.roughness);

    // Diffuse lobe.
    float3 fd = albedo * INV_PI;
    float pdfD = wi.z * INV_PI;

    // Specular lobe.
    float3 h = normalize(wo + wi);
    float D  = ggxD(h, a);
    float G  = smithG1(wo, a) * smithG1(wi, a);
    float3 F = fresnelSchlick(max(0.0, dot(wo, h)), F0);
    float3 fs = (D * G) * F / (4.0 * wo.z * wi.z + 1e-9);
    float pdfSpec = smithG1(wo, a) * max(0.0, dot(wo, h)) * D / (wo.z + 1e-9);
    pdfSpec /= (4.0 * max(0.0, dot(wo, h)) + 1e-9);

    f   = fd + fs;
    pdf = (1.0 - pSpec) * pdfD + pSpec * pdfSpec;
}

inline BsdfSample bsdfSample(Material m, float3 wo, thread RNG& rng) {
    BsdfSample s; s.wi = float3(0,0,1); s.f = float3(0.0); s.pdf = 0.0;
    if (wo.z <= 0.0) return s;
    float3 albedo, F0; float pSpec;
    bsdfWeights(m, albedo, F0, pSpec);
    float a = max(1e-3, m.roughness * m.roughness);

    float3 wi;
    if (randf(rng) < pSpec) {
        float3 h = sampleGGXVNDF(wo, a, randf(rng), randf(rng));
        wi = reflect(-wo, h);
        if (wi.z <= 0.0) return s;
    } else {
        wi = cosineHemisphere(float2(randf(rng), randf(rng)));
    }
    bsdfEval(m, wo, wi, s.f, s.pdf);
    s.wi = wi;
    return s;
}

inline float powerHeuristic(float a, float b) {
    float a2 = a * a;
    return a2 / (a2 + b * b + 1e-9);
}

//===----------------------------------------------------------------------===//
// Geometry access.
//===----------------------------------------------------------------------===//
struct Hit { bool valid; float3 pos; float3 ns; float3 ng; uint matId; float dist; uint prim; float2 uv; };

inline float3 vtx(const device float3* P, const device uint* I, uint tri, uint k) {
    return float3(P[I[tri * 3 + k]]);
}
inline float3 nrm(const device float3* N, const device uint* I, uint tri, uint k) {
    return float3(N[I[tri * 3 + k]]);
}
inline float triArea(const device float3* P, const device uint* I, uint tri) {
    float3 p0 = vtx(P, I, tri, 0), p1 = vtx(P, I, tri, 1), p2 = vtx(P, I, tri, 2);
    return 0.5 * length(cross(p1 - p0, p2 - p0));
}

//===----------------------------------------------------------------------===//
// Camera ray generation (pinhole + optional thin-lens depth of field).
//===----------------------------------------------------------------------===//
inline ray makeCameraRayCam(Camera cam, uint2 imageSize, float2 pix, thread RNG& rng) {
    float2 jitter = float2(randf(rng), randf(rng));
    float2 ndc = (pix + jitter) / float2(imageSize);
    float sx = (2.0 * ndc.x - 1.0) * cam.tanHalfFovY * cam.aspect;
    float sy = (1.0 - 2.0 * ndc.y) * cam.tanHalfFovY;     // flip Y
    float3 dir = normalize(cam.forward + sx * cam.right + sy * cam.up);

    float3 origin = cam.position;
    if (cam.aperture > 0.0) {
        float3 focal = origin + dir * (cam.focusDist / dot(dir, cam.forward));
        float2 lens = cam.aperture * concentricDisk(float2(randf(rng), randf(rng)));
        origin += lens.x * cam.right + lens.y * cam.up;
        dir = normalize(focal - origin);
    }
    ray r; r.origin = origin; r.direction = dir; r.min_distance = 0.0; r.max_distance = INFINITY;
    return r;
}
inline ray makeCameraRay(constant Uniforms& u, float2 pix, thread RNG& rng) {
    return makeCameraRayCam(u.camera, u.imageSize, pix, rng);
}

//===----------------------------------------------------------------------===//
// Path tracing core, shared by the realtime kernel and the data-export kernel.
//===----------------------------------------------------------------------===//
struct SceneRefs {
    const device float3* positions;
    const device float3* normals;
    const device uint*          indices;
    const device uint*          triMaterial;
    const device Material*      materials;
    const device uint*          emissive;
};

inline Hit traceClosest(ray r, primitive_acceleration_structure accel, SceneRefs s) {
    intersector<triangle_data> it;
    it.assume_geometry_type(geometry_type::triangle);
    it.force_opacity(forced_opacity::opaque);
    auto res = it.intersect(r, accel);
    Hit h; h.valid = false;
    if (res.type == intersection_type::none) return h;
    uint tri = res.primitive_id;
    float2 bc = res.triangle_barycentric_coord;
    float w = 1.0 - bc.x - bc.y;
    h.valid = true;
    h.prim  = tri;
    h.dist  = res.distance;
    h.pos   = r.origin + r.direction * res.distance;
    float3 n0 = nrm(s.normals, s.indices, tri, 0);
    float3 n1 = nrm(s.normals, s.indices, tri, 1);
    float3 n2 = nrm(s.normals, s.indices, tri, 2);
    h.ns = normalize(w * n0 + bc.x * n1 + bc.y * n2);
    float3 p0 = vtx(s.positions, s.indices, tri, 0);
    float3 p1 = vtx(s.positions, s.indices, tri, 1);
    float3 p2 = vtx(s.positions, s.indices, tri, 2);
    h.ng = normalize(cross(p1 - p0, p2 - p0));
    h.matId = s.triMaterial[tri];
    return h;
}

inline bool traceOccluded(float3 p, float3 q, primitive_acceleration_structure accel) {
    float3 d = q - p;
    float dist = length(d);
    ray r; r.origin = p; r.direction = d / dist;
    r.min_distance = EPS; r.max_distance = dist - 2.0 * EPS;
    intersector<triangle_data> it;
    it.assume_geometry_type(geometry_type::triangle);
    it.force_opacity(forced_opacity::opaque);
    it.accept_any_intersection(true);
    auto res = it.intersect(r, accel);
    return res.type != intersection_type::none;
}

// Shadow test between two surface points, offsetting along their normals.
inline bool occludedSeg(float3 a, float3 aNg, float3 b, float3 bNg,
                        primitive_acceleration_structure accel) {
    float3 w = normalize(b - a);
    float3 oa = a + (dot(aNg, w) > 0.0 ? aNg : -aNg) * EPS;
    float3 ob = b + (dot(bNg, -w) > 0.0 ? bNg : -bNg) * EPS;
    return traceOccluded(oa, ob, accel);
}

// Direct lighting via NEE (samples one emissive triangle). Returns MIS-weighted
// contribution already multiplied by throughput-independent factors.
inline float3 sampleLights(float3 x, float3 ns, float3 ng, Material m, float3 woWorld,
                           primitive_acceleration_structure accel, SceneRefs s,
                           uint numEmissive, thread RNG& rng, bool noMis) {
    if (numEmissive == 0) return float3(0.0);
    uint pick = min(uint(randf(rng) * float(numEmissive)), numEmissive - 1);
    uint tri = s.emissive[pick];

    float3 p0 = vtx(s.positions, s.indices, tri, 0);
    float3 p1 = vtx(s.positions, s.indices, tri, 1);
    float3 p2 = vtx(s.positions, s.indices, tri, 2);
    float u = randf(rng), v = randf(rng);
    float su = sqrt(u);
    float b0 = 1.0 - su, b1 = su * (1.0 - v), b2 = su * v;
    float3 lp = b0 * p0 + b1 * p1 + b2 * p2;
    float3 lng = normalize(cross(p1 - p0, p2 - p0));
    float area = 0.5 * length(cross(p1 - p0, p2 - p0));

    float3 toL = lp - x;
    float dist2 = dot(toL, toL);
    float dist = sqrt(dist2);
    float3 wiW = toL / dist;
    float cosL = dot(lng, -wiW);
    if (cosL <= 0.0) return float3(0.0);                 // one-sided emitter

    float pdfA = 1.0 / (float(numEmissive) * area);
    float pdfLight = pdfA * dist2 / cosL;                 // area -> solid angle

    // Local-frame BSDF evaluation.
    float3 t, b; onb(ns, t, b);
    float3 wo = toLocal(woWorld, t, b, ns);
    float3 wi = toLocal(wiW, t, b, ns);
    float3 f; float pdfB; bsdfEval(m, wo, wi, f, pdfB);
    if (pdfB <= 0.0 || all(f == float3(0.0))) return float3(0.0);

    if (occludedSeg(x, ng, lp, lng, accel)) return float3(0.0);

    float3 Le = s.materials[s.triMaterial[tri]].emission;
    float mis = noMis ? 1.0 : powerHeuristic(pdfLight, pdfB);
    return f * Le * abs(wi.z) * mis / pdfLight;
}

// Full path integrator for one camera ray. Optionally records first-hit features.
inline float3 integrate(ray r, primitive_acceleration_structure accel, SceneRefs s,
                        constant Uniforms& u, thread RNG& rng,
                        thread Feature& feat, bool wantFeature) {
    float3 L = float3(0.0);
    float3 beta = float3(1.0);
    float  bsdfPdfPrev = 0.0;
    bool   prevDelta = true;          // camera as a delta "lens": first emission full
    float3 prevPos = r.origin;
    bool dbgNoEmis = (u.flags >> 16) & 1u;
    bool dbgNoNEE  = (u.flags >> 17) & 1u;
    bool dbgMis1   = (u.flags >> 18) & 1u;

    for (uint bounce = 0; bounce < u.maxBounces; ++bounce) {
        Hit h = traceClosest(r, accel, s);
        if (!h.valid) {
            L += beta * u.background;
            if (wantFeature && bounce == 0) feat.hit = 0.0;
            break;
        }
        Material m = s.materials[h.matId];
        float3 woW = -r.direction;
        // Face-forward geometric normal for consistent offsetting.
        float3 ngf = (dot(h.ng, woW) < 0.0) ? -h.ng : h.ng;
        float3 nsf = (dot(h.ns, woW) < 0.0) ? -h.ns : h.ns;

        // Emission (with MIS against the NEE that could have sampled this light).
        if (!dbgNoEmis && any(m.emission > float3(0.0))) {
            float w = 1.0;
            if (!prevDelta) {
                float cosL = dot(ngf, woW);
                if (cosL > 0.0) {
                    float area = triArea(s.positions, s.indices, h.prim);
                    float dist2 = dot(h.pos - prevPos, h.pos - prevPos);
                    float pdfLight = dist2 / (float(u.numEmissive) * area * cosL);
                    w = dbgMis1 ? 1.0 : powerHeuristic(bsdfPdfPrev, pdfLight);
                } else { w = 0.0; }
            }
            L += beta * m.emission * w;
        }

        if (wantFeature && bounce == 0) {
            feat.hitPos = h.pos; feat.normal = nsf; feat.wo = woW;
            feat.baseColor = m.baseColor; feat.metallic = m.metallic;
            feat.roughness = m.roughness; feat.hit = 1.0;
        }

        // Next-event estimation (direct lighting).
        if (!dbgNoNEE)
            L += beta * sampleLights(h.pos, nsf, ngf, m, woW, accel, s, u.numEmissive, rng, dbgMis1);

        // BSDF sampling for the indirect bounce.
        float3 t, bb; onb(nsf, t, bb);
        float3 woL = toLocal(woW, t, bb, nsf);
        BsdfSample bs = bsdfSample(m, woL, rng);
        if (bs.pdf <= 0.0 || all(bs.f == float3(0.0))) break;
        float3 wiW = toWorld(bs.wi, t, bb, nsf);

        if (wantFeature && bounce == 0) feat.firstDir = wiW;

        beta *= bs.f * abs(bs.wi.z) / bs.pdf;
        bsdfPdfPrev = bs.pdf;
        prevDelta = false;
        prevPos = h.pos;

        // Russian roulette.
        if (bounce > 3) {
            float q = clamp(max(beta.x, max(beta.y, beta.z)), 0.05, 1.0);
            if (randf(rng) > q) break;
            beta /= q;
        }
        r.origin = h.pos + ngf * EPS;
        r.direction = wiW;
        r.min_distance = 0.0;
        r.max_distance = INFINITY;
    }
    return L;
}

//===----------------------------------------------------------------------===//
// Bidirectional Path Tracing (Veach & Guibas 1995), GPU implementation.
//
// Builds a camera subpath and a light subpath, then connects every camera vertex
// to every light vertex (plus the s=0 "hit a light" and s=1 NEE strategies),
// combining all strategies with multiple importance sampling using per-vertex
// forward/reverse area densities (PBRT Ch.16 formulation). The t=1 "light image"
// strategy is omitted: for a pinhole camera and non-specular BSDFs every path is
// still covered by the remaining strategies, so the estimator stays unbiased and
// converges to the same image as the unidirectional integrator (validated).
//===----------------------------------------------------------------------===//
constant int BDPT_MAXV = 8;          // endpoint + up to 7 surface vertices

struct Vertex {
    float3 p;
    float3 ns;
    float3 ng;
    float3 beta;     // throughput arriving at this vertex (excludes its own BSDF)
    uint   matId;
    uint   prim;
    int    type;     // 0 = camera, 1 = light, 2 = surface
    float  pdfFwd;   // forward area density
    float  pdfRev;   // reverse area density
};

// Convert a solid-angle density at `fromP` to an area density at `toP`.
inline float toAreaPdf(float pdfW, float3 fromP, float3 toP, float3 toNs) {
    float3 d = toP - fromP;
    float inv = 1.0 / max(dot(d, d), 1e-12);
    float cosT = abs(dot(toNs, d * sqrt(inv)));
    return pdfW * cosT * inv;
}

// Random walk shared by both subpaths. path[0] is the (already-filled) endpoint.
// Fills path[1..n] with surface vertices and returns n.
inline int randomWalk(ray r, float3 beta, float pdfDir,
                      primitive_acceleration_structure accel, SceneRefs sc,
                      constant Uniforms& u, thread RNG& rng,
                      thread Vertex* path, int maxDepth,
                      thread float3& envOut, bool addEnv) {
    if (maxDepth == 0) return 0;
    float pdfFwd = pdfDir;
    int n = 0;
    float3 prevP = path[0].p;
    float3 prevNs = path[0].ns;

    for (int bounce = 0; ; ++bounce) {
        Hit h = traceClosest(r, accel, sc);
        if (!h.valid) {
            if (addEnv) envOut += beta * u.background;     // escaped ray sees the environment
            break;
        }
        Material m = sc.materials[h.matId];
        float3 woW = -r.direction;
        int idx = n + 1;

        Vertex v;
        v.p = h.pos; v.ns = h.ns; v.ng = h.ng; v.matId = h.matId; v.prim = h.prim;
        v.type = 2; v.beta = beta; v.pdfRev = 0.0;
        v.pdfFwd = toAreaPdf(pdfFwd, prevP, h.pos, h.ns);
        path[idx] = v;
        n++;
        if (n >= maxDepth) break;

        float3 t, b; onb(h.ns, t, b);
        float3 woL = toLocal(woW, t, b, h.ns);
        BsdfSample bs = bsdfSample(m, woL, rng);
        if (bs.pdf <= 0.0 || all(bs.f == float3(0.0))) break;
        float3 wiW = toWorld(bs.wi, t, b, h.ns);

        float3 fRev; float pdfRevDir;
        bsdfEval(m, bs.wi, woL, fRev, pdfRevDir);
        path[idx - 1].pdfRev = toAreaPdf(pdfRevDir, h.pos, prevP, prevNs);

        beta *= bs.f * abs(bs.wi.z) / bs.pdf;
        pdfFwd = bs.pdf;
        float3 ngf = dot(h.ng, wiW) < 0.0 ? -h.ng : h.ng;
        r.origin = h.pos + ngf * EPS; r.direction = wiW;
        r.min_distance = 0.0; r.max_distance = INFINITY;
        prevP = h.pos; prevNs = h.ns;

        if (bounce > 3) {
            float q = clamp(max(beta.x, max(beta.y, beta.z)), 0.05, 1.0);
            if (randf(rng) > q) break;
            beta /= q;
        }
    }
    return n;
}

inline int generateLightSubpath(thread Vertex* path,
                                primitive_acceleration_structure accel, SceneRefs sc,
                                constant Uniforms& u, thread RNG& rng, int maxDepth) {
    if (u.numEmissive == 0) return 0;
    uint pick = min(uint(randf(rng) * float(u.numEmissive)), u.numEmissive - 1);
    uint tri = sc.emissive[pick];
    float3 p0 = vtx(sc.positions, sc.indices, tri, 0);
    float3 p1 = vtx(sc.positions, sc.indices, tri, 1);
    float3 p2 = vtx(sc.positions, sc.indices, tri, 2);
    float u1 = randf(rng), v1 = randf(rng), su = sqrt(u1);
    float3 lp = (1.0 - su) * p0 + (su * (1.0 - v1)) * p1 + (su * v1) * p2;
    float3 lng = normalize(cross(p1 - p0, p2 - p0));
    float area = 0.5 * length(cross(p1 - p0, p2 - p0));
    float pdfPos = 1.0 / area;
    float pdfChoose = 1.0 / float(u.numEmissive);

    float3 t, b; onb(lng, t, b);
    float3 dl = cosineHemisphere(float2(randf(rng), randf(rng)));
    float3 dirW = toWorld(dl, t, b, lng);
    float pdfDir = dl.z * INV_PI;
    float3 Le = sc.materials[sc.triMaterial[tri]].emission;

    Vertex l0;
    l0.p = lp; l0.ns = lng; l0.ng = lng; l0.beta = Le;
    l0.matId = sc.triMaterial[tri]; l0.prim = tri; l0.type = 1;
    l0.pdfFwd = pdfPos * pdfChoose; l0.pdfRev = 0.0;
    path[0] = l0;
    if (pdfDir <= 0.0) return 0;

    float3 beta = Le * dl.z / (pdfChoose * pdfPos * pdfDir);
    ray r; r.origin = lp + lng * EPS; r.direction = dirW;
    r.min_distance = 0.0; r.max_distance = INFINITY;
    float3 envDummy = float3(0.0);
    return randomWalk(r, beta, pdfDir, accel, sc, u, rng, path, maxDepth, envDummy, false);
}

// Area density of generating vertex `to` from vertex `v` (whose previous vertex
// is at prevP). Handles surface (BSDF) and light (cosine emission) vertices.
inline float pdfFromVertex(Vertex v, float3 prevP, bool hasPrev, Vertex to, SceneRefs sc) {
    if (v.type == 1) {
        float3 w = to.p - v.p;
        float dist = length(w); w /= dist;
        float cosE = dot(v.ng, w);
        if (cosE <= 0.0) return 0.0;
        return toAreaPdf(cosE * INV_PI, v.p, to.p, to.ns);
    }
    float3 wiW = normalize(to.p - v.p);
    float3 woW = hasPrev ? normalize(prevP - v.p) : wiW;
    float3 t, b; onb(v.ns, t, b);
    float3 wo = toLocal(woW, t, b, v.ns);
    float3 wi = toLocal(wiW, t, b, v.ns);
    float3 f; float pdfW; bsdfEval(sc.materials[v.matId], wo, wi, f, pdfW);
    return toAreaPdf(pdfW, v.p, to.p, to.ns);
}

inline float pdfLightOrigin(Vertex v, SceneRefs sc, uint numEmissive) {
    float area = triArea(sc.positions, sc.indices, v.prim);
    return 1.0 / (max(area, 1e-9) * float(numEmissive));
}

// MIS weight for connecting camera prefix of length t with light prefix of
// length s, using the balance-style power heuristic over area densities.
inline float misWeightBDPT(thread Vertex* cam, int t, thread Vertex* lit, int s,
                           SceneRefs sc, uint numEm, int sMax) {
    if (s + t == 2) return 1.0;
    float ptRev = 0, ptMinusRev = 0, qsRev = 0, qsMinusRev = 0;
    bool hasPtM = (t > 1), hasQs = (s > 0), hasQsM = (s > 1);
    Vertex pt = cam[t - 1];

    if (hasQs) {
        Vertex qs = lit[s - 1];
        ptRev = pdfFromVertex(qs, hasQsM ? lit[s - 2].p : float3(0.0), hasQsM, pt, sc);
        qsRev = pdfFromVertex(pt, hasPtM ? cam[t - 2].p : float3(0.0), hasPtM, qs, sc);
        if (hasPtM) ptMinusRev = pdfFromVertex(pt, qs.p, true, cam[t - 2], sc);
        if (hasQsM) qsMinusRev = pdfFromVertex(qs, pt.p, true, lit[s - 2], sc);
    } else {
        ptRev = pdfLightOrigin(pt, sc, numEm);                 // PdfLightOrigin
        if (hasPtM) {
            float3 w = cam[t - 2].p - pt.p;
            float dist = length(w); w /= dist;
            float cosE = dot(pt.ng, w);
            ptMinusRev = cosE > 0.0 ? toAreaPdf(cosE * INV_PI, pt.p, cam[t - 2].p, cam[t - 2].ns) : 0.0;
        }
    }

    float sumRi = 0.0, ri = 1.0;
    int lo = max(2, (s + t) - sMax);             // only count strategies with s' <= sMax
    for (int i = t - 1; i >= lo; --i) {           // camera side (omit t=1)
        float rev = (i == t - 1) ? ptRev : (i == t - 2 ? ptMinusRev : cam[i].pdfRev);
        float fwd = cam[i].pdfFwd;
        ri *= (rev != 0.0 ? rev : 1.0) / (fwd != 0.0 ? fwd : 1.0);
        sumRi += ri;
    }
    ri = 1.0;
    for (int i = s - 1; i >= 0; --i) {            // light side (down to s=0)
        float rev = (i == s - 1) ? qsRev : (i == s - 2 ? qsMinusRev : lit[i].pdfRev);
        float fwd = lit[i].pdfFwd;
        ri *= (rev != 0.0 ? rev : 1.0) / (fwd != 0.0 ? fwd : 1.0);
        sumRi += ri;
    }
    return 1.0 / (1.0 + sumRi);
}

// Connect the camera and light subpaths, summing all MIS-weighted strategies.
inline float3 connectBDPT(thread Vertex* cam, int nCam, thread Vertex* lit, int nLight,
                          primitive_acceleration_structure accel, SceneRefs sc,
                          constant Uniforms& u, thread RNG& rng) {
    float3 L = float3(0.0);
    int totalCam = nCam + 1;        // including camera endpoint
    int totalLight = nLight + 1;    // including light endpoint
    int sMax = int((u.flags >> 8) & 0xFFu);   // debug cap on light-subpath length
    if (sMax == 0) sMax = BDPT_MAXV;
    bool dNoS0 = (u.flags >> 16) & 1u;
    bool dNoS1 = (u.flags >> 17) & 1u;
    bool dMis1 = (u.flags >> 18) & 1u;

    for (int t = 2; t <= totalCam; ++t) {
        int ct = t - 1;
        Vertex pt = cam[ct];
        Material mPt = sc.materials[pt.matId];
        float3 toPrev = normalize(cam[ct - 1].p - pt.p);

        // s = 0: the camera vertex is itself on an emitter.
        if (!dNoS0 && any(mPt.emission > float3(0.0))) {
            float cosL = dot(pt.ng, toPrev);
            if (cosL > 0.0) {
                float w = dMis1 ? 1.0 : misWeightBDPT(cam, t, lit, 0, sc, u.numEmissive, sMax);
                L += pt.beta * mPt.emission * w;
            }
        }

        // s = 1: next-event estimation to a freshly sampled light point.
        if (!dNoS1 && u.numEmissive > 0) {
            uint pick = min(uint(randf(rng) * float(u.numEmissive)), u.numEmissive - 1);
            uint tri = sc.emissive[pick];
            float3 p0 = vtx(sc.positions, sc.indices, tri, 0);
            float3 p1 = vtx(sc.positions, sc.indices, tri, 1);
            float3 p2 = vtx(sc.positions, sc.indices, tri, 2);
            float su = sqrt(randf(rng)); float vv = randf(rng);
            float3 lp = (1.0 - su) * p0 + (su * (1.0 - vv)) * p1 + (su * vv) * p2;
            float3 lng = normalize(cross(p1 - p0, p2 - p0));
            float area = 0.5 * length(cross(p1 - p0, p2 - p0));
            float3 d = lp - pt.p; float dist2 = dot(d, d); float dist = sqrt(dist2);
            float3 wiW = d / dist;
            float cosLight = dot(lng, -wiW);
            if (cosLight > 0.0) {
                float3 t_, b_; onb(pt.ns, t_, b_);
                float3 woL = toLocal(toPrev, t_, b_, pt.ns);
                float3 wiL = toLocal(wiW, t_, b_, pt.ns);
                float3 f; float pdfB; bsdfEval(mPt, woL, wiL, f, pdfB);
                if (pdfB > 0.0 && !all(f == float3(0.0))) {
                    float pdfA = 1.0 / (float(u.numEmissive) * area);
                    float pdfW = pdfA * dist2 / cosLight;
                    if (!occludedSeg(pt.p, pt.ng, lp, lng, accel)) {
                        float3 Le = sc.materials[sc.triMaterial[tri]].emission;
                        Vertex sv;
                        sv.p = lp; sv.ns = lng; sv.ng = lng; sv.type = 1;
                        sv.matId = sc.triMaterial[tri]; sv.prim = tri;
                        sv.pdfFwd = pdfA; sv.pdfRev = 0.0; sv.beta = Le;
                        thread Vertex svArr[1]; svArr[0] = sv;
                        float w = dMis1 ? 1.0 : misWeightBDPT(cam, t, svArr, 1, sc, u.numEmissive, sMax);
                        L += pt.beta * f * abs(wiL.z) * Le * w / pdfW;
                    }
                }
            }
        }

        // s >= 2: connect to interior light-subpath vertices.
        for (int s = 2; s <= min(totalLight, sMax); ++s) {
            int cs = s - 1;
            Vertex qs = lit[cs];
            float3 d = qs.p - pt.p; float dist2 = dot(d, d); float dist = sqrt(dist2);
            float3 w = d / dist;

            float3 tP, bP; onb(pt.ns, tP, bP);
            float3 woPt = toLocal(toPrev, tP, bP, pt.ns);
            float3 wiPt = toLocal(w, tP, bP, pt.ns);
            float3 fPt; float pdfPt; bsdfEval(mPt, woPt, wiPt, fPt, pdfPt);
            if (all(fPt == float3(0.0))) continue;

            Material mQs = sc.materials[qs.matId];
            float3 tQ, bQ; onb(qs.ns, tQ, bQ);
            float3 woQs = toLocal(normalize(lit[cs - 1].p - qs.p), tQ, bQ, qs.ns);
            float3 wiQs = toLocal(-w, tQ, bQ, qs.ns);
            float3 fQs; float pdfQs; bsdfEval(mQs, woQs, wiQs, fQs, pdfQs);
            if (all(fQs == float3(0.0))) continue;

            float G = abs(dot(pt.ns, w)) * abs(dot(qs.ns, w)) / dist2;
            if (G <= 0.0) continue;
            if (occludedSeg(pt.p, pt.ng, qs.p, qs.ng, accel)) continue;

            float3 contrib = pt.beta * fPt * G * fQs * qs.beta;
            if (all(contrib == float3(0.0))) continue;
            float wMis = dMis1 ? 1.0 : misWeightBDPT(cam, t, lit, s, sc, u.numEmissive, sMax);
            L += contrib * wMis;
        }
    }
    return L;
}

//===----------------------------------------------------------------------===//
// Kernel: bidirectional path tracing into the accumulation buffer.
//===----------------------------------------------------------------------===//
kernel void bdpt(device float4*                       accum       [[buffer(0)]],
                 constant Uniforms&                   u           [[buffer(1)]],
                 const device float3*                 positions   [[buffer(2)]],
                 const device float3*                 normals     [[buffer(3)]],
                 const device uint*                   indices     [[buffer(4)]],
                 const device uint*                   triMaterial [[buffer(5)]],
                 const device Material*               materials   [[buffer(6)]],
                 const device uint*                   emissive    [[buffer(7)]],
                 primitive_acceleration_structure     accel       [[buffer(8)]],
                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.imageSize.x || gid.y >= u.imageSize.y) return;
    uint idx = gid.y * u.imageSize.x + gid.x;
    SceneRefs s{positions, normals, indices, triMaterial, materials, emissive};
    RNG rng; rng.state = pcgHash(idx + 0x9e3779b9u * (u.frameIndex + 1u));
    int maxDepth = int(min(u.maxBounces, uint(BDPT_MAXV - 1)));

    float3 sum = float3(0.0);
    for (uint sp = 0; sp < u.samplesPerFrame; ++sp) {
        thread Vertex camPath[BDPT_MAXV];
        thread Vertex lightPath[BDPT_MAXV];

        Vertex c0;
        c0.p = u.camera.position; c0.ns = u.camera.forward; c0.ng = u.camera.forward;
        c0.beta = float3(1.0); c0.type = 0; c0.matId = 0; c0.prim = 0;
        c0.pdfFwd = 1.0; c0.pdfRev = 0.0;
        camPath[0] = c0;
        ray r = makeCameraRay(u, float2(gid), rng);
        float3 env = float3(0.0);
        int nCam = randomWalk(r, float3(1.0), 1.0, accel, s, u, rng, camPath, maxDepth, env, true);

        int nLight = generateLightSubpath(lightPath, accel, s, u, rng, maxDepth);

        float3 c = connectBDPT(camPath, nCam, lightPath, nLight, accel, s, u, rng) + env;
        if (u.radianceClamp > 0.0) { float l = luminance(c); if (l > u.radianceClamp) c *= u.radianceClamp / l; }
        sum += c;
    }
    sum /= float(u.samplesPerFrame);

    float4 prev = (u.flags & 1u) ? float4(0.0) : accum[idx];
    accum[idx] = prev + float4(sum, 1.0);
}

//===----------------------------------------------------------------------===//
// Kernel: progressive path tracing into an accumulation buffer.
//===----------------------------------------------------------------------===//
kernel void pathtrace(device float4*                       accum       [[buffer(0)]],
                      constant Uniforms&                   u           [[buffer(1)]],
                      const device float3*          positions   [[buffer(2)]],
                      const device float3*          normals     [[buffer(3)]],
                      const device uint*                   indices     [[buffer(4)]],
                      const device uint*                   triMaterial [[buffer(5)]],
                      const device Material*               materials   [[buffer(6)]],
                      const device uint*                   emissive    [[buffer(7)]],
                      primitive_acceleration_structure     accel       [[buffer(8)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.imageSize.x || gid.y >= u.imageSize.y) return;
    uint idx = gid.y * u.imageSize.x + gid.x;

    SceneRefs s{positions, normals, indices, triMaterial, materials, emissive};
    RNG rng; rng.state = pcgHash(idx + 0x9e3779b9u * (u.frameIndex + 1u));

    float3 sum = float3(0.0);
    Feature dummy;
    for (uint sp = 0; sp < u.samplesPerFrame; ++sp) {
        ray r = makeCameraRay(u, float2(gid), rng);
        float3 c = integrate(r, accel, s, u, rng, dummy, false);
        if (u.radianceClamp > 0.0) { float l = luminance(c); if (l > u.radianceClamp) c *= u.radianceClamp / l; }
        sum += c;
    }
    sum /= float(u.samplesPerFrame);

    float4 prev = (u.flags & 1u) ? float4(0.0) : accum[idx];
    accum[idx] = prev + float4(sum, 1.0);
}

//===----------------------------------------------------------------------===//
// Kernel: render geometric/shading normals (P1 de-risk + debug visualization).
//===----------------------------------------------------------------------===//
kernel void renderNormals(device float4*                   accum       [[buffer(0)]],
                          constant Uniforms&               u           [[buffer(1)]],
                          const device float3*      positions   [[buffer(2)]],
                          const device float3*      normals     [[buffer(3)]],
                          const device uint*               indices     [[buffer(4)]],
                          const device uint*               triMaterial [[buffer(5)]],
                          const device Material*           materials   [[buffer(6)]],
                          const device uint*               emissive    [[buffer(7)]],
                          primitive_acceleration_structure accel       [[buffer(8)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.imageSize.x || gid.y >= u.imageSize.y) return;
    uint idx = gid.y * u.imageSize.x + gid.x;
    SceneRefs s{positions, normals, indices, triMaterial, materials, emissive};
    RNG rng; rng.state = pcgHash(idx + 1u);
    ray r = makeCameraRay(u, float2(gid), rng);
    Hit h = traceClosest(r, accel, s);
    float3 c = h.valid ? (0.5 * (h.ns + 1.0)) : float3(0.0);
    accum[idx] = float4(c, 1.0);
}

//===----------------------------------------------------------------------===//
// Kernel: tonemap+sRGB resolve of the accumulation buffer into a drawable.
//===----------------------------------------------------------------------===//
kernel void resolve(texture2d<float, access::write> outTex [[texture(0)]],
                    const device float4*             accum  [[buffer(0)]],
                    constant uint2&                  size   [[buffer(1)]],
                    constant float&                  exposure [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    uint idx = gid.y * size.x + gid.x;
    float4 a = accum[idx];
    float n = max(a.w, 1.0);
    float3 c = a.rgb / n * exposure;
    float3 num = c * (2.51 * c + 0.03);
    float3 den = c * (2.43 * c + 0.59) + 0.14;
    c = clamp(num / den, 0.0, 1.0);
    c = select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
    outTex.write(float4(c, 1.0), gid);
}

//===----------------------------------------------------------------------===//
// Kernel: ML-pipeline feature diagnostics.
//
// Visualize ONE channel of the 21-dim per-pixel feature vector that the neural
// ray-amplifier consumes (see `exportFeatures` + research/model.py), so the
// interactive viewer can show exactly what the network ingests for the current
// camera. `feats` is the buffer filled by `exportFeatures`; `mode` selects the
// channel and matches Renderer.featureViewModeNames (index 0 = Beauty, handled
// by the normal resolve path, so this kernel only sees mode >= 1).
//===----------------------------------------------------------------------===//

// Project a world point to pixel coordinates under `cam` (inverse of
// makeCameraRayCam). Returns (pixel.xy, depth-along-forward); depth <= 0 means
// the point is behind the camera. Used to build screen-space motion vectors.
inline float3 projectToPixel(Camera cam, uint2 size, float3 P) {
    float3 d = P - cam.position;
    float fz = dot(d, cam.forward);
    float sx = dot(d, cam.right) / fz;
    float sy = dot(d, cam.up) / fz;
    float ndcx = (sx / (cam.tanHalfFovY * cam.aspect) + 1.0) * 0.5;
    float ndcy = (1.0 - sy / cam.tanHalfFovY) * 0.5;
    return float3(ndcx * float(size.x), ndcy * float(size.y), fz);
}

kernel void resolveFeatureViz(texture2d<float, access::write> outTex     [[texture(0)]],
                              const device Feature*            feats      [[buffer(0)]],
                              constant uint2&                  size       [[buffer(1)]],
                              constant uint&                   mode       [[buffer(2)]],
                              constant float&                  exposure   [[buffer(3)]],
                              constant float3&                 normCenter [[buffer(4)]],
                              constant float&                  normInvExtent [[buffer(5)]],
                              constant Camera&                 prevCam    [[buffer(6)]],
                              constant float&                  motionScale [[buffer(7)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    uint idx = gid.y * size.x + gid.x;
    Feature f = feats[idx];
    bool hit = f.hit > 0.5;
    float3 c = float3(0.0);
    switch (mode) {
        case 1: {                                       // 1-spp radiance: the cheap, noisy
            float3 r = f.oneSample * exposure;           // signal the amplifier must denoise
            float3 num = r * (2.51 * r + 0.03);          // (ACES+sRGB, matches `resolve`)
            float3 den = r * (2.43 * r + 0.59) + 0.14;
            r = clamp(num / den, 0.0, 1.0);
            c = select(1.055 * pow(r, 1.0 / 2.4) - 0.055, r * 12.92, r <= 0.0031308);
            break;                                       // shown everywhere, incl. background
        }
        case 2: c = hit ? 0.5 * (f.normal + 1.0) : float3(0.0); break;     // shading normal
        case 3: c = hit ? 0.5 * (f.wo + 1.0) : float3(0.0); break;         // view dir (wo)
        case 4: c = hit ? f.baseColor : float3(0.0); break;                // base color
        case 5: c = hit ? float3(f.metallic) : float3(0.0); break;         // metallic
        case 6: c = hit ? float3(f.roughness) : float3(0.0); break;        // roughness
        case 7: {                                                          // scene-normalized
            float3 p = (f.hitPos - normCenter) * normInvExtent;            // position (model input)
            c = hit ? clamp(0.5 * (p + 1.0), 0.0, 1.0) : float3(0.0);
            break;
        }
        case 8: c = hit ? 0.5 * (f.firstDir + 1.0) : float3(0.0); break;   // first bounce dir
        case 9: c = float3(f.hit); break;                                  // hit / valid mask
        case 10: {                                                         // screen-space motion
            if (!hit) break;                                               // vectors (temporal cue)
            float3 prev = projectToPixel(prevCam, size, f.hitPos);
            float2 mv = (float2(gid) + 0.5 - prev.xy) / max(motionScale, 1e-4);
            c = float3(clamp(0.5 + 0.5 * mv.x, 0.0, 1.0),
                       clamp(0.5 - 0.5 * mv.y, 0.0, 1.0),    // flip Y for image-space view
                       0.5);
            break;
        }
        default: break;
    }
    outTex.write(float4(c, 1.0), gid);
}

//===----------------------------------------------------------------------===//
// Kernel: data export for the neural-amplifier research (P5).
//   accum  : converged high-spp target (mean radiance) for this frame batch
//   feats  : per-pixel first-hit features + a single-sample (1 spp) estimate
//===----------------------------------------------------------------------===//
kernel void exportFeatures(device float4*                   accum       [[buffer(0)]],
                           constant Uniforms&               u           [[buffer(1)]],
                           const device float3*      positions   [[buffer(2)]],
                           const device float3*      normals     [[buffer(3)]],
                           const device uint*               indices     [[buffer(4)]],
                           const device uint*               triMaterial [[buffer(5)]],
                           const device Material*           materials   [[buffer(6)]],
                           const device uint*               emissive    [[buffer(7)]],
                           primitive_acceleration_structure accel       [[buffer(8)]],
                           device Feature*                  feats       [[buffer(9)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.imageSize.x || gid.y >= u.imageSize.y) return;
    uint idx = gid.y * u.imageSize.x + gid.x;
    SceneRefs s{positions, normals, indices, triMaterial, materials, emissive};

    // High-spp converged target (accumulated across export frames).
    RNG rng; rng.state = pcgHash(idx + 0x9e3779b9u * (u.frameIndex + 1u));
    float3 sum = float3(0.0);
    Feature dummy;
    for (uint sp = 0; sp < u.samplesPerFrame; ++sp) {
        ray r = makeCameraRay(u, float2(gid), rng);
        float3 c = integrate(r, accel, s, u, rng, dummy, false);
        if (u.radianceClamp > 0.0) { float l = luminance(c); if (l > u.radianceClamp) c *= u.radianceClamp / l; }
        sum += c;
    }
    sum /= float(u.samplesPerFrame);
    float4 prev = (u.flags & 1u) ? float4(0.0) : accum[idx];
    accum[idx] = prev + float4(sum, 1.0);

    // One physically cast ray + its features (only on the reset/first frame).
    if (u.flags & 1u) {
        RNG r1; r1.state = pcgHash(idx * 2654435761u + 12345u);
        Feature f;
        f.hitPos = f.normal = f.wo = f.baseColor = float3(0.0);
        f.metallic = f.roughness = f.hit = f._pad = 0.0;
        f.oneSample = f.firstDir = float3(0.0);
        ray r = makeCameraRay(u, float2(gid), r1);
        float3 one = integrate(r, accel, s, u, r1, f, true);
        if (u.radianceClamp > 0.0) { float l = luminance(one); if (l > u.radianceClamp) one *= u.radianceClamp / l; }
        f.oneSample = one;
        feats[idx] = f;
    }
}

//===----------------------------------------------------------------------===//
// INSTANCED RENDERING (BLAS/TLAS) — the engine for Moana-scale scenes.
//
// A two-level acceleration structure (per-mesh BLAS + a top-level instance
// acceleration structure) lets a handful of unique meshes be replicated into
// millions of instances (billions of effective triangles) with a tiny memory
// footprint — exactly the technique the Disney Moana Island Scene requires.
//===----------------------------------------------------------------------===//
using namespace raytracing;

// Per-instance shading data, indexed by the intersector's instance_id.
struct InstanceData {
    float4 nrm0;        // columns of the 3x3 object->world normal matrix (xyz)
    float4 nrm1;
    float4 nrm2;
    uint   indexOffset; // first index of this instance's mesh in the global index buffer
    uint   materialId;
    uint   _pad0;
    uint   _pad1;
};

// Sun + sky + camera uniforms for outdoor (island) rendering.
struct IslandUniforms {
    Camera camera;
    uint2  imageSize;
    uint   frameIndex;
    uint   samplesPerFrame;
    uint   maxBounces;
    uint   flags;
    float3 sunDir;        // direction TOWARD the sun (normalized)
    float3 sunColor;      // sun radiance
    float3 skyZenith;
    float3 skyHorizon;
    float  radianceClamp;
    float  exposure;
};

struct InstRefs {
    const device float3*        positions;
    const device float3*        normals;
    const device float2*        uvs;
    const device uint*          indices;
    const device InstanceData*  instances;
    const device Material*      materials;
};

inline float3 skyColor(float3 d, constant IslandUniforms& u) {
    float t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    float3 sky = mix(u.skyHorizon, u.skyZenith, t * t);
    // Sun disk + soft atmospheric glow (also seen in ocean reflections).
    float c = clamp(dot(normalize(d), u.sunDir), 0.0, 1.0);
    float disk = smoothstep(0.9985, 0.9994, c);
    float glow = pow(c, 220.0);
    sky += u.sunColor * (disk * 5.0 + glow * 0.6);
    return sky;
}

inline Hit traceClosestInst(ray r, instance_acceleration_structure accel, InstRefs s) {
    intersector<triangle_data, instancing> it;
    it.assume_geometry_type(geometry_type::triangle);
    it.force_opacity(forced_opacity::opaque);
    auto res = it.intersect(r, accel);
    Hit h; h.valid = false;
    if (res.type == intersection_type::none) return h;
    InstanceData id = s.instances[res.instance_id];
    uint base = id.indexOffset + res.primitive_id * 3u;
    uint i0 = s.indices[base], i1 = s.indices[base + 1], i2 = s.indices[base + 2];
    float2 bc = res.triangle_barycentric_coord;
    float wgt = 1.0 - bc.x - bc.y;
    float3x3 nm = float3x3(id.nrm0.xyz, id.nrm1.xyz, id.nrm2.xyz);
    float3 nObj = wgt * s.normals[i0] + bc.x * s.normals[i1] + bc.y * s.normals[i2];
    h.ns = normalize(nm * nObj);
    h.uv = wgt * s.uvs[i0] + bc.x * s.uvs[i1] + bc.y * s.uvs[i2];
    float3 p0 = s.positions[i0], p1 = s.positions[i1], p2 = s.positions[i2];
    h.ng = normalize(nm * cross(p1 - p0, p2 - p0));
    h.pos = r.origin + r.direction * res.distance;
    h.dist = res.distance;
    h.matId = id.materialId;
    h.valid = true;
    return h;
}

inline bool occludedInstDir(float3 o, float3 dir, float maxDist,
                            instance_acceleration_structure accel) {
    ray r; r.origin = o; r.direction = dir; r.min_distance = EPS; r.max_distance = maxDist;
    intersector<triangle_data, instancing> it;
    it.assume_geometry_type(geometry_type::triangle);
    it.force_opacity(forced_opacity::opaque);
    it.accept_any_intersection(true);
    auto res = it.intersect(r, accel);
    return res.type != intersection_type::none;
}

// --- Alpha-tested traversal --------------------------------------------------
// Sponza foliage / chains / thorns carry their cut-out shape in the albedo
// texture's own alpha channel. Treat any textured hit whose sampled alpha < 0.5
// as empty space and march the ray onward, so leaves/chains/plants get proper
// silhouettes (and cast correctly shaped shadows). Opaque (3-channel) textures
// load with alpha == 1 everywhere, so they are never skipped.
inline Hit traceClosestInstAT(ray r, instance_acceleration_structure accel, InstRefs s,
                              array<texture2d<float>, 32> tex, sampler smp) {
    float tmin = max(r.min_distance, 0.0);
    for (int iter = 0; iter < 16; ++iter) {
        intersector<triangle_data, instancing> it;
        it.assume_geometry_type(geometry_type::triangle);
        it.force_opacity(forced_opacity::opaque);
        ray rr = r; rr.min_distance = tmin;
        auto res = it.intersect(rr, accel);
        Hit h; h.valid = false;
        if (res.type == intersection_type::none) return h;
        InstanceData id = s.instances[res.instance_id];
        uint base = id.indexOffset + res.primitive_id * 3u;
        uint i0 = s.indices[base], i1 = s.indices[base + 1], i2 = s.indices[base + 2];
        float2 bc = res.triangle_barycentric_coord;
        float wgt = 1.0 - bc.x - bc.y;
        float2 uv = wgt * s.uvs[i0] + bc.x * s.uvs[i1] + bc.y * s.uvs[i2];
        Material m = s.materials[id.materialId];
        if (m.texId >= 0.0 &&
            tex[(uint)m.texId].sample(smp, float2(uv.x, 1.0 - uv.y)).a < 0.5) {
            tmin = res.distance + max(EPS, res.distance * 1e-4);    // see through, march on
            continue;
        }
        float3x3 nm = float3x3(id.nrm0.xyz, id.nrm1.xyz, id.nrm2.xyz);
        float3 nObj = wgt * s.normals[i0] + bc.x * s.normals[i1] + bc.y * s.normals[i2];
        h.ns = normalize(nm * nObj);
        h.uv = uv;
        float3 p0 = s.positions[i0], p1 = s.positions[i1], p2 = s.positions[i2];
        h.ng = normalize(nm * cross(p1 - p0, p2 - p0));
        h.pos = rr.origin + rr.direction * res.distance;
        h.dist = res.distance;
        h.matId = id.materialId;
        h.valid = true;
        return h;
    }
    Hit h; h.valid = false; return h;
}

inline bool occludedInstAT(float3 o, float3 dir, float maxDist,
                           instance_acceleration_structure accel, InstRefs s,
                           array<texture2d<float>, 32> tex, sampler smp) {
    float tmin = EPS;
    for (int iter = 0; iter < 16; ++iter) {
        intersector<triangle_data, instancing> it;
        it.assume_geometry_type(geometry_type::triangle);
        it.force_opacity(forced_opacity::opaque);
        ray r; r.origin = o; r.direction = dir; r.min_distance = tmin; r.max_distance = maxDist;
        auto res = it.intersect(r, accel);
        if (res.type == intersection_type::none) return false;
        InstanceData id = s.instances[res.instance_id];
        Material m = s.materials[id.materialId];
        if (m.texId >= 0.0) {
            uint base = id.indexOffset + res.primitive_id * 3u;
            uint i0 = s.indices[base], i1 = s.indices[base + 1], i2 = s.indices[base + 2];
            float2 bc = res.triangle_barycentric_coord;
            float wgt = 1.0 - bc.x - bc.y;
            float2 uv = wgt * s.uvs[i0] + bc.x * s.uvs[i1] + bc.y * s.uvs[i2];
            if (tex[(uint)m.texId].sample(smp, float2(uv.x, 1.0 - uv.y)).a < 0.5) {
                tmin = res.distance + max(EPS, res.distance * 1e-4);
                continue;
            }
        }
        return true;
    }
    return true;
}

inline float3 hashColor(uint x) {
    x = pcgHash(x + 0x9e3779b9u);
    return float3(float(x & 255u), float((x >> 8) & 255u), float((x >> 16) & 255u)) / 255.0;
}

// Unidirectional path tracer for outdoor scenes: analytic directional sun
// (hard shadows) + smooth sky environment, with GGX/metallic-roughness PBR and
// optional per-material albedo textures (sRGB, sampled to linear).
inline float3 integrateIsland(ray r, instance_acceleration_structure accel, InstRefs s,
                              constant IslandUniforms& u, thread RNG& rng,
                              array<texture2d<float>, 32> tex, sampler smp, bool texturesOff) {
    float3 L = float3(0.0);
    float3 beta = float3(1.0);
    for (uint bounce = 0; bounce < u.maxBounces; ++bounce) {
        Hit h = traceClosestInstAT(r, accel, s, tex, smp);
        if (!h.valid) { L += beta * skyColor(r.direction, u); break; }
        Material m = s.materials[h.matId];
        if (m.texId >= 0.0 && !texturesOff) {
            float3 c = tex[(uint)m.texId].sample(smp, float2(h.uv.x, 1.0 - h.uv.y)).rgb;
            m.baseColor = c;
        }
        float3 woW = -r.direction;
        float3 ngf = (dot(h.ng, woW) < 0.0) ? -h.ng : h.ng;
        float3 nsf = (dot(h.ns, woW) < 0.0) ? -h.ns : h.ns;

        float3 t, b; onb(nsf, t, b);
        float3 woL = toLocal(woW, t, b, nsf);

        // Sun (analytic directional light, hard shadows).
        float cosSun = dot(nsf, u.sunDir);
        if (cosSun > 0.0) {
            float3 wiL = toLocal(u.sunDir, t, b, nsf);
            float3 f; float pdfB; bsdfEval(m, woL, wiL, f, pdfB);
            if (!all(f == float3(0.0)) && !occludedInstAT(h.pos + ngf * EPS, u.sunDir, INFINITY, accel, s, tex, smp))
                L += beta * f * u.sunColor * cosSun;
        }

        BsdfSample bs = bsdfSample(m, woL, rng);
        if (bs.pdf <= 0.0 || all(bs.f == float3(0.0))) break;
        float3 wiW = toWorld(bs.wi, t, b, nsf);
        beta *= bs.f * abs(bs.wi.z) / bs.pdf;

        if (bounce > 3) {
            float q = clamp(max(beta.x, max(beta.y, beta.z)), 0.05, 1.0);
            if (randf(rng) > q) break;
            beta /= q;
        }
        r.origin = h.pos + ngf * EPS; r.direction = wiW;
        r.min_distance = 0.0; r.max_distance = INFINITY;
    }
    return L;
}

// Feature visualisation (AOV) shading for the interactive viewer's toggles.
// mode: 1 Albedo · 2 Shading normals · 3 Geometric normals · 4 UVs ·
//       5 Material ID · 6 Ambient occlusion (sky visibility) · 7 Direct sun only.
inline float3 shadeAOV(ray r, instance_acceleration_structure accel, InstRefs s,
                       constant IslandUniforms& u, thread RNG& rng,
                       array<texture2d<float>, 32> tex, sampler smp,
                       uint mode, bool texturesOff) {
    Hit h = traceClosestInstAT(r, accel, s, tex, smp);
    if (!h.valid) return skyColor(r.direction, u);
    float3 woW = -r.direction;
    float3 nsf = (dot(h.ns, woW) < 0.0) ? -h.ns : h.ns;
    float3 ngf = (dot(h.ng, woW) < 0.0) ? -h.ng : h.ng;
    Material m = s.materials[h.matId];
    float3 albedo = m.baseColor;
    if (m.texId >= 0.0 && !texturesOff)
        albedo = tex[(uint)m.texId].sample(smp, float2(h.uv.x, 1.0 - h.uv.y)).rgb;
    m.baseColor = albedo;
    switch (mode) {
        case 1: return albedo;                              // Albedo / textures
        case 2: return 0.5 * (nsf + 1.0);                   // Shading normals
        case 3: return 0.5 * (ngf + 1.0);                   // Geometric normals
        case 4: return float3(fract(h.uv), 0.0);            // UV coordinates
        case 5: return hashColor(h.matId);                  // Material ID
        case 6: {                                           // Ambient occlusion (sky visibility)
            float3 t, b; onb(nsf, t, b);
            float3 wd = toWorld(cosineHemisphere(float2(randf(rng), randf(rng))), t, b, nsf);
            return occludedInstAT(h.pos + ngf * EPS, wd, INFINITY, accel, s, tex, smp)
                 ? float3(0.03) : float3(1.0);
        }
        case 7: {                                           // Direct sunlight only (no GI)
            float3 t, b; onb(nsf, t, b);
            float3 woL = toLocal(woW, t, b, nsf);
            float cosSun = dot(nsf, u.sunDir);
            float3 Ld = float3(0.0);
            if (cosSun > 0.0 && !occludedInstAT(h.pos + ngf * EPS, u.sunDir, INFINITY, accel, s, tex, smp)) {
                float3 wiL = toLocal(u.sunDir, t, b, nsf);
                float3 f; float pdfB; bsdfEval(m, woL, wiL, f, pdfB);
                Ld = f * u.sunColor * cosSun;
            }
            return Ld;
        }
        default: return albedo;
    }
}

kernel void instPathtrace(device float4*                    accum     [[buffer(0)]],
                          constant IslandUniforms&          u         [[buffer(1)]],
                          const device float3*              positions [[buffer(2)]],
                          const device float3*              normals   [[buffer(3)]],
                          const device uint*                indices   [[buffer(4)]],
                          const device InstanceData*        instances [[buffer(5)]],
                          const device Material*            materials [[buffer(6)]],
                          instance_acceleration_structure   accel     [[buffer(7)]],
                          const device float2*              uvs       [[buffer(8)]],
                          array<texture2d<float>, 32>       albedoTex [[texture(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.imageSize.x || gid.y >= u.imageSize.y) return;
    uint idx = gid.y * u.imageSize.x + gid.x;
    InstRefs s{positions, normals, uvs, indices, instances, materials};
    constexpr sampler smp(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(8));
    RNG rng; rng.state = pcgHash(idx + 0x9e3779b9u * (u.frameIndex + 1u));

    uint viewMode = (u.flags >> 4) & 0xFu;
    bool texturesOff = (u.flags >> 3) & 1u;
    float3 sum = float3(0.0);
    for (uint sp = 0; sp < u.samplesPerFrame; ++sp) {
        ray r = makeCameraRayCam(u.camera, u.imageSize, float2(gid), rng);
        float3 c = (viewMode == 0u)
            ? integrateIsland(r, accel, s, u, rng, albedoTex, smp, texturesOff)
            : shadeAOV(r, accel, s, u, rng, albedoTex, smp, viewMode, texturesOff);
        if (u.radianceClamp > 0.0) { float l = luminance(c); if (l > u.radianceClamp) c *= u.radianceClamp / l; }
        sum += c;
    }
    sum /= float(u.samplesPerFrame);
    float4 prev = (u.flags & 1u) ? float4(0.0) : accum[idx];
    accum[idx] = prev + float4(sum, 1.0);
}

kernel void instNormals(device float4*                    accum     [[buffer(0)]],
                        constant IslandUniforms&          u         [[buffer(1)]],
                        const device float3*              positions [[buffer(2)]],
                        const device float3*              normals   [[buffer(3)]],
                        const device uint*                indices   [[buffer(4)]],
                        const device InstanceData*        instances [[buffer(5)]],
                        const device Material*            materials [[buffer(6)]],
                        instance_acceleration_structure   accel     [[buffer(7)]],
                        const device float2*              uvs       [[buffer(8)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.imageSize.x || gid.y >= u.imageSize.y) return;
    uint idx = gid.y * u.imageSize.x + gid.x;
    InstRefs s{positions, normals, uvs, indices, instances, materials};
    RNG rng; rng.state = pcgHash(idx + 1u);
    ray r = makeCameraRayCam(u.camera, u.imageSize, float2(gid), rng);
    Hit h = traceClosestInst(r, accel, s);
    float3 c = h.valid ? (0.5 * (h.ns + 1.0)) : skyColor(r.direction, u);
    accum[idx] = float4(c, 1.0);
}
