#pragma once

#include "tinypt/cuda/geometry.h"

namespace tinypt {
namespace cuda {

struct Material;

struct Ray {
    Vec3f org; // origin
    Vec3f dir; // direction

    Ray() = default;
    __device__ Ray(const Vec3f &org_, const Vec3f &dir_) : org(org_), dir(dir_) {}

    __device__ Vec3f point_at(float t) const { return org + dir * t; }

    __device__ static Vec3f reflect(const Vec3f &dir, const Vec3f &normal) {
        Vec3f refl_dir = dir - 2 * normal.dot(dir) * normal;
        return refl_dir;
    }

    __device__ static Vec3f refract(const Vec3f &dir, const Vec3f &normal, float etai_over_etat, float cos_i,
                                    float cos_t) {
        Vec3f refr_dir = (etai_over_etat * dir + (etai_over_etat * cos_i - cos_t) * normal).normalized();
        return refr_dir;
    }
};

struct Hit {
    float t;                  // distant from ray origin to intersection point
    Vec3f normal;             // surface normal
    Vec3f shade_normal;       // shading normal
    bool into;                // is ray shooting into the object?
    Vec2f uv;                 // uv position on surface coordinates
    const Material *material; // object material at intersection

    __device__ Hit() : Hit(INF, Vec3f::Zero(), Vec3f::Zero(), true, Vec2f::Zero(), nullptr) {}
    __device__ Hit(float t_, const Vec3f &normal_, const Vec3f &shade_normal_, bool into_, const Vec2f &uv_,
                   const Material *material_)
        : t(t_), normal(normal_), shade_normal(shade_normal_), into(into_), uv(uv_), material(material_) {}

    __device__ bool is_hit() const { return t < INF; }
};

} // namespace cuda
} // namespace tinypt