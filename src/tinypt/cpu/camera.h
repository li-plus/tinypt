#pragma once

#include "tinypt/cpu/mapping.h"
#include "tinypt/cpu/rand.h"
#include "tinypt/cpu/ray.h"

namespace tinypt {

namespace cuda {
struct Camera;
} // namespace cuda

namespace cpu {

class Camera {
    friend tinypt::cuda::Camera;

  public:
    Camera() : Camera(Vec3f::Zero(), Mat3f::Identity(), 1920, 1080, 45, 0, 1) {}

    int width() const { return _width; }
    void set_width(int width) {
        *this = Camera(location(), rotation(), width, height(), fov(), aperture(), focus_dist());
    }
    int height() const { return _height; }
    void set_height(int height) {
        *this = Camera(location(), rotation(), width(), height, fov(), aperture(), focus_dist());
    }
    void set_resolution(int width, int height) {
        *this = Camera(location(), rotation(), width, height, fov(), aperture(), focus_dist());
    }

    Vec3f location() const { return _center; }
    void set_location(const Vec3f &location) { _center = location; }

    Mat3f rotation() const { return _rot.matrix(); }
    void set_rotation(const Mat3f &rotation) {
        *this = Camera(location(), rotation, width(), height(), fov(), aperture(), focus_dist());
    }

    float fov() const { return _fov; }
    void set_fov(float fov) {
        *this = Camera(location(), rotation(), width(), height(), fov, aperture(), focus_dist());
    }

    float aperture() const { return _aperture; }
    float focus_dist() const { return _focus_dist; }
    void set_lens(float aperture, float focus_dist) {
        *this = Camera(location(), rotation(), width(), height(), fov(), aperture, focus_dist);
    }

    Ray shoot_ray(float x, float y, RandEngine &rng) const;

  private:
    Camera(const Vec3f &center, const Mat3f &rotation, int width, int height, float fov, float aperture,
           float focus_dist);

  private:
    // location
    Vec3f _center;
    // rotation
    Rotation _rot;
    // image size
    int _width;
    int _height;
    // field of view
    float _fov;
    // lens params for Depth of Field (DOF)
    float _aperture;
    float _focus_dist;
    // view vector
    Vec3f _direct;
    Vec3f _up;
    Vec3f _right;
};

} // namespace cpu
} // namespace tinypt
