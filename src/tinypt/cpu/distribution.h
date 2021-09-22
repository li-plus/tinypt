#pragma once

#include "tinypt/cpu/rand.h"

#include <iostream>
#include <numeric>

namespace tinypt {
namespace cpu {

struct Distribution1D {
    std::vector<float> _cdf;

    Distribution1D() : Distribution1D(std::vector<float>{1}) {}
    Distribution1D(const std::vector<float> &weights);

    int count() const { return (int)_cdf.size() - 1; }

    float sample(RandEngine &rng) const { return sample(rng.random()); }
    float sample(float u) const;
    float pdf(float x) const;

    int to_anchor(float x) const { return (int)(x * ((float)count() - EPS)); }

    friend std::ostream &operator<<(std::ostream &os, const Distribution1D &self);
};

struct Distribution2D {
    std::vector<Distribution1D> _conditional;
    Distribution1D _marginal;

    Distribution2D() : Distribution2D(std::vector<std::vector<float>>{{1}}) {}
    Distribution2D(const std::vector<std::vector<float>> &weights);

    Vec2f sample(RandEngine &rng) const { return sample({rng.random(), rng.random()}); }
    Vec2f sample(const Vec2f &uv) const;
    float pdf(const Vec2f &xy) const;
};

} // namespace cpu
} // namespace tinypt
