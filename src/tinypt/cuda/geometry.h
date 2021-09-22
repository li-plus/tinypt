#pragma once

#include "tinypt/cuda/defs.h"

#ifndef __NVCC__
#include "tinypt/cpu/mapping.h"
#endif

namespace tinypt {
namespace cuda {

struct Vec3f {
    Vec3f() = default;
    __device__ Vec3f(float x, float y, float z) : _data{x, y, z} {}

#ifndef __NVCC__
    Vec3f(const Eigen::Vector3f &cpu_vec) : Vec3f(cpu_vec.x(), cpu_vec.y(), cpu_vec.z()) {}
#endif

    __device__ bool isZero() const { return is_close(x(), 0) && is_close(y(), 0) && is_close(z(), 0); }
    __host__ __device__ static Vec3f Zero() { return {0, 0, 0}; }
    __host__ __device__ static Vec3f Ones() { return {1, 1, 1}; }

    __host__ __device__ static Vec3f UnitX() { return {1, 0, 0}; }
    __host__ __device__ static Vec3f UnitY() { return {0, 1, 0}; }
    __host__ __device__ static Vec3f UnitZ() { return {0, 0, 1}; }

    __device__ float x() const { return _data[0]; }
    __device__ float y() const { return _data[1]; }
    __device__ float z() const { return _data[2]; }

    __device__ bool operator==(const Vec3f &o) const { return x() == o.x() && y() == o.y() && z() == o.z(); }
    __device__ bool operator!=(const Vec3f &o) const { return !(*this == o); }

    __device__ Vec3f operator+(const Vec3f &o) const { return {x() + o.x(), y() + o.y(), z() + o.z()}; }
    __device__ Vec3f &operator+=(const Vec3f &o) { return *this = *this + o; }

    __device__ Vec3f operator-(const Vec3f &o) const { return {x() - o.x(), y() - o.y(), z() - o.z()}; }
    __device__ Vec3f &operator-=(const Vec3f &o) { return *this = *this - o; }
    __device__ Vec3f operator-() const { return {-x(), -y(), -z()}; }

    __device__ friend Vec3f operator*(float s, const Vec3f &v) { return v * s; }
    __device__ Vec3f operator*(float s) const { return {x() * s, y() * s, z() * s}; }
    __device__ Vec3f &operator*=(float s) { return *this = *this * s; }

    __device__ friend Vec3f operator/(float s, const Vec3f &v) { return {s / v.x(), s / v.y(), s / v.z()}; }
    __device__ Vec3f operator/(float s) const { return {x() / s, y() / s, z() / s}; }
    __device__ Vec3f &operator/=(float s) { return *this = *this / s; }

    __device__ Vec3f cwiseProduct(const Vec3f &o) const { return {x() * o.x(), y() * o.y(), z() * o.z()}; }

    __device__ float dot(const Vec3f &o) const { return x() * o.x() + y() * o.y() + z() * o.z(); }

    __device__ Vec3f cross(const Vec3f &o) const {
        return {y() * o.z() - z() * o.y(), z() * o.x() - x() * o.z(), x() * o.y() - y() * o.x()};
    }

    __device__ float squaredNorm() const { return x() * x() + y() * y() + z() * z(); }

    __device__ float norm() const { return std::sqrt(squaredNorm()); }

    __device__ void normalize() {
        float n = norm();
        if (n != 0) {
            *this /= n;
        }
    }

    __device__ Vec3f normalized() const {
        Vec3f out = *this;
        out.normalize();
        return out;
    }

    __device__ Vec3f clip(float minv, float maxv) const { return cwiseMax(minv).cwiseMin(maxv); }

    __device__ Vec3f cwiseMin(float v) const { return {fminf(x(), v), fminf(y(), v), fminf(z(), v)}; }
    __device__ Vec3f cwiseMin(const Vec3f &o) const {
        return {fminf(x(), o.x()), fminf(y(), o.y()), fminf(z(), o.z())};
    }

    __device__ Vec3f cwiseMax(float v) const { return {fmaxf(x(), v), fmaxf(y(), v), fmaxf(z(), v)}; }
    __device__ Vec3f cwiseMax(const Vec3f &o) const {
        return {fmaxf(x(), o.x()), fmaxf(y(), o.y()), fmaxf(z(), o.z())};
    }

    __device__ bool allFinite() const { return std::isfinite(x()) && std::isfinite(y()) && std::isfinite(z()); }

    __device__ float operator[](size_t i) const { return _data[i]; }
    __device__ float &operator[](size_t i) { return _data[i]; }

  private:
    float _data[3];
};

__device__ static inline bool is_same_side(const Vec3f &a, const Vec3f &b) { return a.dot(b) > 0; }

struct Vec2f {
    Vec2f() = default;
    __device__ Vec2f(float x, float y) : _data{x, y} {}
#ifndef __NVCC__
    Vec2f(const Eigen::Vector2f &cpu_vec) : Vec2f(cpu_vec.x(), cpu_vec.y()) {}
#endif

    __device__ static Vec2f Zero() { return {0, 0}; }
    __device__ static Vec2f Ones() { return {1, 1}; }

    __device__ float x() const { return _data[0]; }
    __device__ float y() const { return _data[1]; }

    __device__ Vec2f operator+(const Vec2f &o) const { return {x() + o.x(), y() + o.y()}; }
    __device__ Vec2f &operator+=(const Vec2f &o) { return *this = *this + o; }

    __device__ Vec2f operator-(const Vec2f &o) const { return {x() - o.x(), y() - o.y()}; }
    __device__ Vec2f &operator-=(const Vec2f &o) { return *this = *this - o; }
    __device__ Vec2f operator-() const { return {-x(), -y()}; }

    __device__ friend Vec2f operator*(float s, const Vec2f &v) { return v * s; }
    __device__ Vec2f operator*(float s) const { return {x() * s, y() * s}; }
    __device__ Vec2f &operator*=(float s) { return *this = *this * s; }

  private:
    float _data[2];
};

struct Vec3i {
    Vec3i() = default;
    __device__ Vec3i(int x, int y, int z) : _data{x, y, z} {}
#ifndef __NVCC__
    Vec3i(const Eigen::Vector3i &cpu_vec) : Vec3i(cpu_vec.x(), cpu_vec.y(), cpu_vec.z()) {}
#endif

    __device__ int x() const { return _data[0]; }
    __device__ int y() const { return _data[1]; }
    __device__ int z() const { return _data[2]; }

    __device__ int operator[](size_t i) const { return _data[i]; }
    __device__ int &operator[](size_t i) { return _data[i]; }

  private:
    int _data[3];
};

struct Rotation {
    __device__ Rotation() : _u(Vec3f::UnitX()), _v(Vec3f::UnitY()), _w(Vec3f::UnitZ()) {}
    __device__ Rotation(const Vec3f &u, const Vec3f &v, const Vec3f &w) : _u(u), _v(v), _w(w) {}
#ifndef __NVCC__
    Rotation(const cpu::Rotation &cpu_rotation) : Rotation(cpu_rotation.u(), cpu_rotation.v(), cpu_rotation.w()) {}
#endif

    __device__ const Vec3f &u() const { return _u; }
    __device__ const Vec3f &v() const { return _v; }
    __device__ const Vec3f &w() const { return _w; }

    __device__ static Rotation from_direction(const Vec3f &w) {
        Vec3f u;
        if (std::abs(w.y()) < .9f) {
            u = Vec3f(w.z(), 0, -w.x()).normalized(); // (0,1,0) x (w)
        } else {
            u = Vec3f(0, -w.z(), w.y()).normalized(); // (1,0,0) x (w)
        }
        Vec3f v = w.cross(u);
        return Rotation(u, v, w);
    }

    __device__ Vec3f operator*(const Vec3f &vec) const { return _u * vec.x() + _v * vec.y() + _w * vec.z(); }

  private:
    Vec3f _u, _v, _w;
};

struct RandEngine {
    __device__ RandEngine(uint64_t seed) { curand_init(seed, 0, 0, &state); }

    __device__ float random() { return curand_uniform(&state); }
    __device__ float random(float hi) { return hi * random(); }
    __device__ float random(float lo, float hi) { return (hi - lo) * random() + lo; }

    __device__ uint32_t rand_uint() { return curand(&state); }
    __device__ uint32_t rand_uint(uint32_t n) { return rand_uint() % n; }
    __device__ uint32_t rand_uint(uint32_t lo, uint32_t hi) { return rand_uint() % (hi - lo) + lo; }

    __device__ Vec2f rand_on_disk() {
        float theta = random(2.f * M_PIf32);
        return {std::cos(theta), std::sin(theta)};
    }
    __device__ Vec2f rand_on_disk(float radius) { return rand_on_disk() * radius; }
    __device__ Vec2f rand_in_disk() { return rand_on_disk() * std::sqrt(random()); }
    __device__ Vec2f rand_in_disk(float radius) { return rand_in_disk() * radius; }

  private:
    curandState state;
};

} // namespace cuda
} // namespace tinypt