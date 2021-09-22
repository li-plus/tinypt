#pragma once

#include "tinypt/cpu/defs.h"

#include <glog/logging.h>

namespace tinypt {
namespace cpu {

class RandEngine {
  public:
    RandEngine() : RandEngine(0) {}
    explicit RandEngine(uint64_t seed_) {
        seed[0] = seed_ & 0xffff;
        seed[1] = (seed_ >> 16) & 0xffff;
        seed[2] = ((seed_ >> 32) & 0xffff) + ((seed_ >> 48) & 0xffff);
    }

    float random() { return (float)erand48(seed); }
    float random(float hi) { return hi * random(); }
    float random(float lo, float hi) { return (hi - lo) * random() + lo; }

    int rand_int() { return rand_r((uint32_t *)seed); }
    int rand_int(int n) { return rand_int() % n; }
    int rand_int(int lo, int hi) { return rand_int() % (hi - lo) + lo; }

    Vec2f rand_on_disk() {
        float theta = random(2.f * M_PIf32);
        return {std::cos(theta), std::sin(theta)};
    }
    Vec2f rand_on_disk(float radius) { return rand_on_disk() * radius; }

    Vec2f rand_in_disk() { return rand_on_disk() * std::sqrt(random()); }
    Vec2f rand_in_disk(float radius) { return rand_in_disk() * radius; }

    Vec3f rand_on_sphere() {
        float phi = random(2.f * M_PIf32);
        float cos_theta = random(-1, 1);
        float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
        DCHECK(std::isfinite(sin_theta));
        return {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
    }
    Vec3f rand_on_sphere(float radius) { return rand_on_sphere() * radius; }

    Vec3f rand_in_sphere() { return rand_on_sphere() * std::cbrt(random()); }
    Vec3f rand_in_sphere(float radius) { return rand_in_sphere() * radius; }

  private:
    uint16_t seed[3];
};

} // namespace cpu
} // namespace tinypt