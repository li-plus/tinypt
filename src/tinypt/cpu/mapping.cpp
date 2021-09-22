#include "tinypt/cpu/mapping.h"
#include "tinypt/cpu/bvh.h"

#include <glog/logging.h>

namespace tinypt {
namespace cpu {

Rotation::Rotation(const Vec3f &u, const Vec3f &v, const Vec3f &w) : _u(u), _v(v), _w(w) {
    DCHECK(is_normalized(_u) && is_normalized(_v) && is_normalized(_w));
    DCHECK(_w.isApprox(_u.cross(_v)) && _v.isApprox(_w.cross(_u)));
}

Mat3f Rotation::matrix() const {
    Mat3f mat;
    mat << _u, _v, _w;
    return mat;
}

Rotation Rotation::from_direction(const Vec3f &w) {
    DCHECK(is_normalized(w));
    Vec3f u;
    if (std::abs(w.y()) < .9f) {
        u = Vec3f(w.z(), 0, -w.x()).normalized(); // (0,1,0) x (w)
    } else {
        u = Vec3f(0, -w.z(), w.y()).normalized(); // (1,0,0) x (w)
    }
    Vec3f v = w.cross(u);
    return Rotation(u, v, w);
}

Rotation Rotation::from_euler(const Vec3i &axes, const Vec3f &angles, bool is_degrees) {
    for (int i = 0; i < 3; i++) {
        TINYPT_CHECK_EX(0 <= axes[i] && axes[i] <= 2, std::invalid_argument)
            << "expect all axes to be between 0 and 2, got axes[" << i << "] = " << axes[i];
    }
    Vec3f rad_angles;
    if (is_degrees) {
        rad_angles = {radians(angles.x()), radians(angles.y()), radians(angles.z())};
    } else {
        rad_angles = angles;
    }
    auto rot = AngleAxisf(rad_angles.z(), Vec3f::Unit(axes.z())) * AngleAxisf(rad_angles.y(), Vec3f::Unit(axes.y())) *
               AngleAxisf(rad_angles.x(), Vec3f::Unit(axes.x()));
    return rot.matrix();
}

Ray Mapping::map_ray(const Ray &ray) const { return Ray(map_point(ray.org), map_direction(ray.dir)); }

Ray Mapping::map_ray_inverse(const Ray &ray) const {
    return Ray(map_point_inverse(ray.org), map_direction_inverse(ray.dir));
}

AABB Mapping::map_bounding_box(const AABB &box) const {
    if (box.empty()) {
        return box;
    }
    std::vector<Vec3f> vertices;
    for (int i = 0; i < 8; i++) {
        float x = (i & 1) ? box.min().x() : box.max().x();
        float y = (i & 2) ? box.min().y() : box.max().y();
        float z = (i & 4) ? box.min().z() : box.max().z();
        vertices.emplace_back(Vec3f(x, y, z));
    }
    for (auto &v : vertices) {
        v = _rotation * v + _location;
    }
    Vec3f minv(INF, INF, INF);
    Vec3f maxv(-INF, -INF, -INF);
    for (const auto &v : vertices) {
        minv = minv.cwiseMin(v);
        maxv = maxv.cwiseMax(v);
    }
    return AABB(minv, maxv);
}

} // namespace cpu
} // namespace tinypt