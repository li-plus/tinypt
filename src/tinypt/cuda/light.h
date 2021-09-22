#pragma once

#include "tinypt/cuda/object.h"

#ifndef __NVCC__
#include "tinypt/cpu/light.h"
#endif

namespace tinypt {
namespace cuda {

struct Light {
    enum Type { TYPE_DISTANT_LIGHT };
    Type _type;

    Light(Type type) : _type(type) {}
    __device__ Vec3f sample(const Vec3f &pos, RandEngine &rng) const;
    __device__ Vec3f emission() const;
};

struct DistantLight : public Light {
    DistantLight() : Light(TYPE_DISTANT_LIGHT) {}
    DistantLight(const Rotation &rot, const Vec3f &color, float power, float angular_radius)
        : Light(TYPE_DISTANT_LIGHT), _rot(rot), _color(color), _power(power), _angular_radius(angular_radius) {}

#ifndef __NVCC__
    DistantLight(const cpu::DistantLight &cpu_light)
        : DistantLight(cpu_light._rot, cpu_light._color, cpu_light._power, cpu_light._angular_radius) {}
#endif

    __device__ Vec3f sample(const Vec3f &pos, RandEngine &rng) const {
        if (_angular_radius < EPS) {
            return _rot.w();
        }
        float diff_u = rng.random(-_angular_radius, _angular_radius);
        float diff_v = rng.random(-_angular_radius, _angular_radius);
        Vec3f out_dir = (_rot * Vec3f(diff_u, diff_v, 1)).normalized();
        return out_dir;
    }
    __device__ Vec3f emission() const { return _color * _power; }

    Rotation _rot;
    Vec3f _color;
    float _power;
    float _angular_radius;
};

__device__ inline Vec3f Light::sample(const Vec3f &pos, RandEngine &rng) const {
    switch (_type) {
    case TYPE_DISTANT_LIGHT:
        return ((DistantLight *)this)->sample(pos, rng);
    default:
        printf("unreachable!\n");
        return Vec3f::Zero();
    }
}

__device__ inline Vec3f Light::emission() const {
    switch (_type) {
    case TYPE_DISTANT_LIGHT:
        return ((DistantLight *)this)->emission();
    default:
        printf("unreachable!\n");
        return Vec3f::Zero();
    }
}

struct DeltaLightGroup {
    DeltaLightGroup() = default;
    DeltaLightGroup(const Array<DistantLight> &distant_lights) : _distant_lights(distant_lights) {}
#ifndef __NVCC__
    static DeltaLightGroup create(const cpu::LightGroup &cpu_light_group) {
        std::vector<DistantLight> host_distant_lights;
        for (const auto &cpu_light : cpu_light_group.lights()) {
            auto cpu_distant_light = dynamic_cast<cpu::DistantLight *>(cpu_light.get());
            TINYPT_CHECK(cpu_distant_light);
            host_distant_lights.emplace_back(*cpu_distant_light);
        }
        auto distant_lights = Array<DistantLight>::create(host_distant_lights);
        return DeltaLightGroup(distant_lights);
    }
    static void destroy(DeltaLightGroup &light_group) { Array<DistantLight>::destroy(light_group._distant_lights); }
#endif

    __device__ bool empty() const { return _distant_lights.empty(); }
    __device__ const Light *sample_light(RandEngine &rng) const {
        return &_distant_lights[rng.rand_uint(_distant_lights.size())];
    }
    __device__ float pdf_light() const { return 1.f / (float)_distant_lights.size(); }

    Array<DistantLight> _distant_lights;
};

struct ObjectLightGroup {
    ObjectLightGroup() = default;
    ObjectLightGroup(const Array<Object *> &lights) : _lights(lights) {}

#ifndef __NVCC__
    static ObjectLightGroup create(const std::vector<Object *> &lights) {
        return ObjectLightGroup(Array<Object *>::create(lights));
    }
    static void destroy(ObjectLightGroup &light_group) { Array<Object *>::destroy(light_group._lights); }
#endif

    __device__ bool empty() const { return _lights.empty(); }

    __device__ Vec3f sample(const Vec3f &pos, RandEngine &rng) const {
        const auto *light = _lights[rng.rand_uint(_lights.size())];
        return light->sample(pos, rng);
    }

    __device__ float pdf(const Vec3f &pos, const Vec3f &out_dir) const {
        float val = 0;
        for (uint32_t i = 0; i < _lights.size(); i++) {
            const auto *light = _lights[i];
            val += light->pdf(Ray(pos, out_dir));
        }
        return val / (float)_lights.size();
    }

    Array<Object *> _lights;
};

} // namespace cuda
} // namespace tinypt