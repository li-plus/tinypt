#include "tinypt/cpu/distribution.h"

namespace tinypt {
namespace cpu {

Distribution1D::Distribution1D(const std::vector<float> &weights) {
    if (std::any_of(weights.begin(), weights.end(), [](float x) { return x < 0; })) {
        TINYPT_THROW_EX(std::invalid_argument) << "negative weight";
    }
    if (weights.empty()) {
        _cdf = {0, 1};
        return;
    }
    // cumulative sum (std::partial_sum)
    _cdf.resize(weights.size() + 1);
    _cdf[0] = 0;
    for (size_t i = 0; i < weights.size(); i++) {
        _cdf[i + 1] = _cdf[i] + weights[i];
    }
    float sum = _cdf.back();
    if (sum > 0) {
        for (auto &val : _cdf) {
            val /= sum;
        }
    } else {
        // zero weights: falling back to uniform distribution
        for (size_t i = 0; i < _cdf.size(); i++) {
            _cdf[i] = (float)i / (float)count();
        }
    }
}

float Distribution1D::sample(float u) const {
    int i = std::upper_bound(_cdf.begin(), _cdf.end(), u) - _cdf.begin();
    i = std::min(i, count());
    // TODO: cdf[i] == cdf[i-1]
    DCHECK_NE(_cdf[i], _cdf[i - 1]);
    float x = ((float)i - 1 + (u - _cdf[i - 1]) / (_cdf[i] - _cdf[i - 1])) / (float)count();
    return x;
}

float Distribution1D::pdf(float x) const {
    int i = to_anchor(x);
    return (_cdf[i + 1] - _cdf[i]) * (float)count();
}

std::ostream &operator<<(std::ostream &os, const Distribution1D &self) {
    os << "Distribution1D(cdf={";
    for (const float val : self._cdf) {
        os << val << ',';
    }
    os << "})";
    return os;
}

Distribution2D::Distribution2D(const std::vector<std::vector<float>> &weights) {
    // TODO fall back
    TINYPT_CHECK_EX(!weights.empty() && !weights.front().empty(), std::invalid_argument) << "empty weights";

    size_t x_size = weights.front().size();

    _conditional.reserve(weights.size());
    std::vector<float> y_weights;
    y_weights.reserve(weights.size());
    for (const auto &weight : weights) {
        TINYPT_CHECK_EX(weight.size() == x_size, std::invalid_argument) << "irregular weights";
        y_weights.emplace_back(std::accumulate(weight.begin(), weight.end(), 0.f));
        _conditional.emplace_back(weight);
    }
    _marginal = Distribution1D(y_weights);
}

Vec2f Distribution2D::sample(const Vec2f &uv) const {
    float u = uv.x(), v = uv.y();
    float y = _marginal.sample(v);
    int yi = _marginal.to_anchor(y);
    const auto &dist_x = _conditional[yi];
    float x = dist_x.sample(u);
    return {x, y};
}

float Distribution2D::pdf(const Vec2f &xy) const {
    float x = xy.x(), y = xy.y();
    float pdf_y = _marginal.pdf(y);
    int yi = _marginal.to_anchor(y);
    const auto &dist_x = _conditional[yi];
    float pdf_x = dist_x.pdf(x);
    return pdf_y * pdf_x;
}

} // namespace cpu
} // namespace tinypt
