#pragma once

#include "tinypt/cpu/camera.h"
#include "tinypt/cpu/light.h"
#include "tinypt/cpu/object.h"

namespace tinypt {
namespace cpu {

class Scene {
  public:
    Scene() : _background(default_background()) {}
    Scene(const Camera &camera, std::vector<std::shared_ptr<Object>> objects,
          std::vector<std::shared_ptr<Light>> delta_lights, std::shared_ptr<EnvTexture> background);

    const EnvTexture &background() const { return *_background; }
    void set_background(std::shared_ptr<EnvTexture> background) { _background = std::move(background); }

    const Camera &camera() const { return _camera; }
    Camera &mutable_camera() { return _camera; }
    void set_camera(const Camera &camera) { _camera = camera; }

    const ObjectGroup &object_group() const { return _obj_group; }
    const LightGroup &object_lights() const { return _object_lights; }
    const LightGroup &delta_lights() const { return _delta_lights; }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { _obj_group.intersect(ray, hit, rng); }
    void intersect_t(const Ray &ray, Hit &hit) const { _obj_group.intersect_t(ray, hit); }

  private:
    static std::shared_ptr<EnvTexture> default_background() {
        return std::make_shared<EnvTexture>((Vec3f)Vec3f::Zero());
    }

    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

  private:
    // attributes
    Camera _camera;
    ObjectGroup _obj_group;
    std::vector<std::shared_ptr<Light>> _lights;
    std::shared_ptr<EnvTexture> _background;
    // derived
    LightGroup _object_lights;
    LightGroup _delta_lights;
};

} // namespace cpu
} // namespace tinypt