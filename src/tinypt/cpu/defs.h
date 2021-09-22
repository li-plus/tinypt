#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace tinypt {
namespace cpu {

typedef Eigen::Vector4f Vec4f;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Matrix4f Mat4f;
typedef Eigen::Matrix3f Mat3f;
typedef Eigen::Matrix2f Mat2f;
using Eigen::Affine2f;
using Eigen::Affine3f;
using Eigen::AlignedBox3f;
using Eigen::AngleAxisf;

static constexpr float INF = 1e20;
static constexpr float EPS = 1e-3;
static constexpr float SRGB_GAMMA = 2.2;

static inline float square(float x) { return x * x; }

static inline bool is_close(float x, float y) { return std::abs(x - y) < EPS; }

static inline bool is_normalized(const Vec3f &v) { return is_close(v.norm(), 1.f); }

static inline bool is_same_side(const Vec3f &a, const Vec3f &b) { return a.dot(b) > 0; }

static inline float radians(float degrees) { return degrees * (M_PIf32 / 180.f); }

static inline float degrees(float radians) { return radians * (180.f / M_PIf32); }

static inline Vec3f rgb_color(uint8_t r, uint8_t g, uint8_t b) {
    return (Vec3f(r, g, b) / 255.f).array().pow(SRGB_GAMMA);
}

template <typename except_t>
class LogMessageFatal {
    static_assert(std::is_base_of_v<std::exception, except_t>, "Expect except_t to be base of std::exception");

  public:
    LogMessageFatal(const char *file, int line) { _oss << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw except_t(_oss.str()); }
    std::ostringstream &stream() { return _oss; }

  private:
    std::ostringstream _oss;
};

#define TINYPT_THROW_EX(except_t) ::tinypt::cpu::LogMessageFatal<except_t>(__FILE__, __LINE__).stream()
#define TINYPT_THROW TINYPT_THROW_EX(std::runtime_error)

#define TINYPT_CHECK_EX(cond, except_t)                                                                                \
    if (!(cond))                                                                                                       \
    TINYPT_THROW_EX(except_t) << "Check failed: " #cond " "
#define TINYPT_CHECK(cond) TINYPT_CHECK_EX(cond, std::runtime_error)

} // namespace cpu
} // namespace tinypt
