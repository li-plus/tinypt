#pragma once

#include "tinypt/cpu/defs.h"

namespace tinypt {
namespace cpu {

class Rotation {
  public:
    Rotation() : Rotation(Mat3f::Identity()) {}
    Rotation(const Mat3f &rotation) : Rotation(rotation.col(0), rotation.col(1), rotation.col(2)) {}
    Rotation(const Vec3f &u, const Vec3f &v, const Vec3f &w);

    const Vec3f &u() const { return _u; }
    const Vec3f &v() const { return _v; }
    const Vec3f &w() const { return _w; }
    Mat3f matrix() const;

    bool isApprox(const Rotation &other) const {
        return _u.isApprox(other._u) && _v.isApprox(other._v) && _w.isApprox(other._w);
    }

    static Rotation identity() { return Rotation(Mat3f::Identity()); }

    static Rotation from_direction(const Vec3f &w);
    static Rotation from_euler(const Vec3i &axes, const Vec3f &angles, bool is_degrees = false);

    Vec3f operator*(const Vec3f &vec) const { return _u * vec.x() + _v * vec.y() + _w * vec.z(); }

    friend std::ostream &operator<<(std::ostream &os, const Rotation &self) {
        return os << "Rotation(" << self.matrix() << ")";
    }

  private:
    Vec3f _u, _v, _w;
};

class AABB;
struct Ray;

class Mapping {
  public:
    Mapping() : Mapping(Vec3f::Zero(), Mat3f::Identity()) {}
    Mapping(const Vec3f &location, const Mat3f &rotation)
        : _location(location), _rotation(rotation), _inv_rotation(rotation.inverse()) {}

    const Vec3f &location() const { return _location; }
    void set_location(const Vec3f &location) { _location = location; }
    const Mat3f &rotation() const { return _rotation; }
    void set_rotation(const Mat3f &rotation) {
        _rotation = rotation;
        _inv_rotation = rotation.inverse();
    }

    AABB map_bounding_box(const AABB &box) const;

    Vec3f map_point(const Vec3f &point) const { return _rotation * point + _location; }
    Vec3f map_point_inverse(const Vec3f &point) const { return _inv_rotation * (point - _location); }
    Vec3f map_direction(const Vec3f &direction) const { return _rotation * direction; }
    Vec3f map_direction_inverse(const Vec3f &direction) const { return _inv_rotation * direction; }
    Ray map_ray(const Ray &ray) const;
    Ray map_ray_inverse(const Ray &ray) const;

  private:
    Vec3f _location;
    Mat3f _rotation;
    Mat3f _inv_rotation;
};

} // namespace cpu
} // namespace tinypt
