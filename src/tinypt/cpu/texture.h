#pragma once

#include "tinypt/cpu/distribution.h"
#include "tinypt/cpu/image.h"
#include "tinypt/cpu/mapping.h"

#include <glog/logging.h>

namespace tinypt {
namespace cpu {

class RGBTexture {
  public:
    RGBTexture() : _value(Vec3f::Zero()) {}
    RGBTexture(const Vec3f &value) : _value(value) {}
    RGBTexture(Image map) : _value(Vec3f::Zero()), _map(std::move(map)) {
        TINYPT_CHECK(_map.channels() == 3) << "invalid rgb texture map";
    }

    bool empty() const { return _map.empty() && _value.isZero(); }

    bool is_value() const { return _map.empty(); }
    const Vec3f &value() const { return _value; }
    void set_value(const Vec3f &value) { _value = value; }

    bool is_map() const { return !_map.empty(); }
    const Image &map() const { return _map; }
    void set_map(Image map) { _map = std::move(map); }

    const Vec3f &color_at(const Vec2f &uv) const;

  private:
    Vec3f _value;
    Image _map;
};

class AlphaTexture {
  public:
    AlphaTexture() : _value(1) {}
    AlphaTexture(float value) : _value(value) {}
    AlphaTexture(Image map) : _value(1), _map(std::move(map)) {
        if (!_map.empty()) {
            TINYPT_CHECK_EX(_map.channels() == 1, std::invalid_argument)
                << "expect alpha map to be grayscale, got " << _map.channels() << " channels";
        }
    }

    bool empty() const { return _map.empty() && _value == 1; }

    bool is_value() const { return _map.empty(); }
    float value() const { return _value; }
    void set_value(float value) { _value = value; }

    bool is_map() const { return !_map.empty(); }
    const Image &map() const { return _map; }
    void set_map(Image map) { _map = std::move(map); }

    float alpha_at(const Vec2f &uv) const;

  private:
    float _value;
    Image _map;
};

class BumpTexture {
  public:
    BumpTexture() = default;
    BumpTexture(const Image &height_map, float scale = 64);

    const Image &map() const { return _grad; }
    bool empty() const { return _grad.empty(); }
    int height() const { return _grad.height(); }
    int width() const { return _grad.width(); }

    Vec2f bump_at(const Vec2f &uv) const;

    Vec3f bump_normal(const Vec2f &uv, const Vec3f &normal, const Vec3f &dpdu, const Vec3f &dpdv) const;

  private:
    Image _grad;
};

class EnvTexture {
    friend class EnvTextureTest;

  public:
    EnvTexture() : EnvTexture({}, Mat3f::Identity()) {}
    EnvTexture(RGBTexture env, const Mat3f &rotation = Mat3f::Identity());

    const Mat3f &rotation() const { return _mapping.rotation(); }
    void set_rotation(const Mat3f &rotation) { _mapping.set_rotation(rotation); }

    const RGBTexture &rgb_texture() const { return _env; }
    void set_rgb_texture(RGBTexture rgb_texture) { _env = std::move(rgb_texture); }

    Vec3f sample(const Vec3f &pos, RandEngine &rng) const;
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const;

    Vec3f color_at(const Vec3f &dir) const;

  private:
    static Vec3f uv2dir(const Vec2f &uv);
    static Vec2f dir2uv(const Vec3f &out_dir);

  private:
    RGBTexture _env;
    Distribution2D _dist;
    Mapping _mapping;
};

} // namespace cpu
} // namespace tinypt