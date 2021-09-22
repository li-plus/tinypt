#include "tinypt/cpu/camera.h"

namespace tinypt {
namespace cpu {

Camera::Camera(const Vec3f &center, const Mat3f &rotation, int width, int height, float fov, float aperture,
               float focus_dist)
    : _center(center), _rot(rotation), _width(width), _height(height), _fov(fov), _aperture(aperture),
      _focus_dist(focus_dist) {
    float view_height, view_width;
    if (width > height) {
        view_width = 2.f * std::tan(fov * .5f);
        view_height = view_width * (float)height / (float)width;
    } else {
        view_height = 2.f * std::tan(fov * .5f);
        view_width = view_height * (float)width / (float)height;
    }
    _direct = _focus_dist * -_rot.w();
    _right = _focus_dist * view_width * _rot.u();
    _up = _focus_dist * view_height * _rot.v();
}

Ray Camera::shoot_ray(float x, float y, RandEngine &rng) const {
    Vec3f offset;
    if (_aperture >= EPS) {
        Vec2f uv_offset = rng.rand_in_disk(_aperture);
        offset = uv_offset.x() * _rot.u() + uv_offset.y() * _rot.v();
    } else {
        offset = Vec3f::Zero();
    }
    Vec3f dir = (x / (float)_width - .5f) * _right + (y / (float)_height - .5f) * _up + _direct - offset;
    Ray ray(_center + offset, dir.normalized());
    return ray;
}

} // namespace cpu
} // namespace tinypt
