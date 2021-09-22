#pragma once

#include "tinypt/cpu/distribution.h"
#include "tinypt/cpu/object.h"

namespace tinypt {
namespace cuda {
struct DistantLight;
}
namespace cpu {

class Light {
  public:
    enum Flag { FLAG_NONE = 0x0, FLAG_DELTA = 0x1 };

    Light() : _flag(FLAG_NONE) {}
    Light(uint32_t flag) : _flag(flag) {}
    virtual ~Light() = default;

    bool is_delta() const { return _flag & FLAG_DELTA; }

    virtual Vec3f sample(const Vec3f &pos, RandEngine &rng) const = 0;
    virtual float pdf(const Vec3f &pos, const Vec3f &out_dir) const = 0;
    virtual Vec3f emission() const = 0;

  private:
    uint32_t _flag;
};

class ObjectLight : public Light {
  public:
    ObjectLight() : _obj(nullptr) {}
    explicit ObjectLight(const Object *obj) : _obj(obj) {}

    const Object *object() const { return _obj; }

    Vec3f sample(const Vec3f &pos, RandEngine &rng) const override { return _obj->sample(pos, rng); }
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const override { return _obj->pdf({pos, out_dir}); }
    Vec3f emission() const override { TINYPT_THROW_EX(std::logic_error) << "not implemented"; }

  private:
    const Object *_obj;
};

class DistantLight : public Light {
    friend tinypt::cuda::DistantLight;

  public:
    DistantLight() : DistantLight(Mat3f::Identity(), Vec3f::Ones(), 1, 0) {}

    Mat3f rotation() const { return _rot.matrix(); }
    void set_rotation(const Mat3f &rotation) { _rot = rotation; }
    const Vec3f &color() const { return _color; }
    void set_color(const Vec3f &color) { _color = color; }
    float power() const { return _power; }
    void set_power(float power) { _power = power; }
    float angle() const { return _angular_radius * 2; }
    void set_angle(float angle) {
        check_angle(angle);
        _angular_radius = angle / 2;
    }

    Vec3f sample(const Vec3f &pos, RandEngine &rng) const override;
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const override {
        TINYPT_THROW_EX(std::logic_error) << "not implemented";
    }
    Vec3f emission() const override { return _color * _power; }

  private:
    DistantLight(const Mat3f &rotation, const Vec3f &color, float power, float angle);

    static void check_angle(float angle) {
        TINYPT_CHECK_EX(0 <= angle && angle <= M_PIf32, std::invalid_argument) << "angle out of range: " << angle;
    }

  private:
    Rotation _rot;
    Vec3f _color;
    float _power;
    float _angular_radius;
};

class PointLight : public Light {
  public:
    PointLight() : PointLight(Vec3f::Zero(), Vec3f::Ones(), 1, 0) {}

    const Vec3f &location() const { return _center; }
    void set_location(const Vec3f &location) { _center = location; }
    const Vec3f &color() const { return _color; }
    void set_color(const Vec3f &color) { _color = color; }
    float power() const { return _power; }
    void set_power(float power) { _power = power; }
    float radius() const { return _radius; }
    void set_radius(float radius) { _radius = radius; }

    Vec3f sample(const Vec3f &pos, RandEngine &rng) const override;
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const override {
        TINYPT_THROW_EX(std::logic_error) << "not implemented";
    }
    Vec3f emission() const override { return _color * _power; }

  private:
    PointLight(const Vec3f &center, const Vec3f &color, float power, float radius)
        : Light(FLAG_DELTA), _center(center), _color(color), _power(power), _radius(radius) {}

  private:
    Vec3f _center;
    Vec3f _color;
    float _power;
    float _radius;
};

class EnvLight : public Light {
  public:
    EnvLight() = default;
    EnvLight(const EnvTexture *env) : _env(env) {}

    Vec3f sample(const Vec3f &pos, RandEngine &rng) const override;
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const override;
    Vec3f emission() const override;

  private:
    const EnvTexture *_env;
};

class LightGroup {
  public:
    LightGroup() = default;
    LightGroup(std::vector<std::shared_ptr<Light>> lights) : _lights(std::move(lights)) {}

    bool empty() const { return _lights.empty(); }
    const std::vector<std::shared_ptr<Light>> &lights() const { return _lights; }
    std::vector<std::shared_ptr<Light>> &mutable_lights() { return _lights; }

    Vec3f sample(const Vec3f &pos, RandEngine &rng) const;
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const;

    const Light *sample_light(RandEngine &rng) const;
    float pdf_light() const;

  private:
    std::vector<std::shared_ptr<Light>> _lights;
};

} // namespace cpu
} // namespace tinypt
