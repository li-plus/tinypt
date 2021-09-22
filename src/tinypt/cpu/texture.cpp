#include "tinypt/cpu/texture.h"

namespace tinypt {
namespace cpu {

static inline Vec2i denormalize_coords(const Vec2f &uv, int width, int height) {
    float iu = uv.x() - std::floor(uv.x());
    float iv = uv.y() - std::floor(uv.y());
    int x = (int)(iu * ((float)width - EPS));
    int y = (int)(iv * ((float)height - EPS));
    DCHECK_LT(x, width);
    DCHECK_LT(y, height);
    return {x, y};
}

const Vec3f &RGBTexture::color_at(const Vec2f &uv) const {
    if (_map.empty()) {
        return _value;
    } else {
        Vec2i xy = denormalize_coords(uv, _map.width(), _map.height());
        return _map.at<Vec3f>(xy.x(), xy.y());
    }
}

float AlphaTexture::alpha_at(const Vec2f &uv) const {
    if (_map.empty()) {
        return _value;
    } else {
        Vec2i xy = denormalize_coords(uv, _map.width(), _map.height());
        return _map.at<float>(xy.x(), xy.y());
    }
}

BumpTexture::BumpTexture(const Image &height_map, float scale) {
    if (!height_map.empty()) {
        TINYPT_CHECK(height_map.channels() == 1)
            << "Expect height map to be grayscale, got " << height_map.channels() << " channels";
        scale /= 8;
        cv::Mat grad[2];
        cv::Sobel(height_map.mat(), grad[0], CV_32F, 1, 0, 3, scale, 0, cv::BORDER_REPLICATE);
        cv::Sobel(height_map.mat(), grad[1], CV_32F, 0, 1, 3, scale, 0, cv::BORDER_REPLICATE);
        cv::merge(grad, 2, _grad.mat());
    }
}

Vec2f BumpTexture::bump_at(const Vec2f &uv) const {
    DCHECK(!empty());
    Vec2i xy = denormalize_coords(uv, width(), height());
    return _grad.at<Vec2f>(xy.x(), xy.y());
}

Vec3f BumpTexture::bump_normal(const Vec2f &uv, const Vec3f &normal, const Vec3f &dpdu, const Vec3f &dpdv) const {
    DCHECK(is_normalized(normal));
    Vec2f dbduv = bump_at(uv);
    float dbdu = dbduv.x(), dbdv = dbduv.y();
    Vec3f delta_normal = dpdv * dbdv + dpdu * dbdu;
    DCHECK(delta_normal.isOrthogonal(normal));
    delta_normal /= (std::sqrt(std::sqrt(dpdu.squaredNorm() * dpdv.squaredNorm())) + EPS);
    Vec3f new_normal = (normal - delta_normal).normalized();
    DCHECK(new_normal.allFinite());
    DCHECK(is_same_side(normal, new_normal));
    return new_normal;
}

EnvTexture::EnvTexture(RGBTexture env, const Mat3f &rotation)
    : _env(std::move(env)), _mapping(Vec3f::Zero(), rotation) {
    if (env.is_map()) {
        const auto &env_map = env.map();
        std::vector<std::vector<float>> weights(env_map.height(), std::vector<float>(env_map.width()));
        for (int y = 0; y < env_map.height(); y++) {
            for (int x = 0; x < env_map.width(); x++) {
                weights[y][x] = env_map.at<Vec3f>(x, y).maxCoeff();
            }
        }
        _dist = Distribution2D(weights);
    }
}

Vec3f EnvTexture::sample(const Vec3f &pos, RandEngine &rng) const {
    DCHECK(_env.is_map());
    Vec2f uv = _dist.sample(rng);
    Vec3f dir = uv2dir(uv);
    return _mapping.map_direction_inverse(dir);
}

float EnvTexture::pdf(const Vec3f &pos, const Vec3f &out_dir) const {
    DCHECK(_env.is_map());
    Vec2f uv = dir2uv(_mapping.map_direction(out_dir));
    return _dist.pdf(uv) * (0.25f * M_1_PIf32);
}

Vec3f EnvTexture::color_at(const Vec3f &dir) const {
    Vec2f uv;
    if (_env.is_map()) {
        uv = dir2uv(_mapping.map_direction(dir));
    } else {
        uv = Vec2f::Zero();
    }
    return _env.color_at(uv);
}

Vec3f EnvTexture::uv2dir(const Vec2f &uv) {
    float u = uv.x(), v = uv.y();
    float phi = 2 * M_PIf32 * (1 - u) - M_PIf32;
    float theta = M_PIf32 * (1 - v);
    float cos_theta = std::cos(theta);
    float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    DCHECK(std::isfinite(sin_theta));
    Vec3f out_dir(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
    return out_dir;
}

Vec2f EnvTexture::dir2uv(const Vec3f &out_dir) {
    DCHECK(is_normalized(out_dir));
    float u = 1 - (std::atan2(out_dir.y(), out_dir.x()) + M_PIf32) * (.5f * M_1_PIf32);
    float v = 1 - std::acos(std::clamp(out_dir.z(), -1.f, 1.f)) * M_1_PIf32;
    DCHECK(0 <= u && u <= 1) << "u = " << u;
    DCHECK(0 <= v && v <= 1) << "v = " << v;
    return {u, v};
}

} // namespace cpu
} // namespace tinypt
