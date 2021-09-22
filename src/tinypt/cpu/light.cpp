#include "tinypt/cpu/light.h"

namespace tinypt {
namespace cpu {

DistantLight::DistantLight(const Mat3f &rotation, const Vec3f &color, float power, float angle)
    : Light(FLAG_DELTA), _rot(rotation), _color(color), _power(power), _angular_radius(angle / 2) {
    check_angle(angle);
}

Vec3f DistantLight::sample(const Vec3f &pos, RandEngine &rng) const {
    if (_angular_radius < EPS) {
        return _rot.w();
    }
    float diff_u = rng.random(-_angular_radius, _angular_radius);
    float diff_v = rng.random(-_angular_radius, _angular_radius);
    Vec3f out_dir = (_rot * Vec3f(diff_u, diff_v, 1)).normalized();
    return out_dir;
}

Vec3f PointLight::sample(const Vec3f &pos, RandEngine &rng) const {
    Vec3f center = _center;
    if (_radius >= EPS) {
        center += rng.rand_on_sphere(_radius);
    }
    return (center - pos).normalized();
}

Vec3f EnvLight::sample(const Vec3f &pos, RandEngine &rng) const { return _env->sample(pos, rng); }

float EnvLight::pdf(const Vec3f &pos, const Vec3f &out_dir) const { return _env->pdf(pos, out_dir); }

Vec3f EnvLight::emission() const { TINYPT_THROW_EX(std::logic_error) << "not implemented"; }

Vec3f LightGroup::sample(const Vec3f &pos, RandEngine &rng) const {
    DCHECK(!_lights.empty());
    const auto &light = _lights[rng.rand_int(_lights.size())];
    return light->sample(pos, rng);
}

float LightGroup::pdf(const Vec3f &pos, const Vec3f &out_dir) const {
    DCHECK(!_lights.empty());
    float val = 0;
    for (const auto &light : _lights) {
        val += light->pdf(pos, out_dir);
    }
    return val / (float)_lights.size();
}

const Light *LightGroup::sample_light(RandEngine &rng) const {
    DCHECK(!_lights.empty());
    return _lights[rng.rand_int(_lights.size())].get();
}

float LightGroup::pdf_light() const {
    DCHECK(!_lights.empty());
    return 1.f / (float)_lights.size();
}

} // namespace cpu
} // namespace tinypt