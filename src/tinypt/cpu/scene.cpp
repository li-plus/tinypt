#include "tinypt/cpu/scene.h"

namespace tinypt {
namespace cpu {

Scene::Scene(const Camera &camera, std::vector<std::shared_ptr<Object>> objects,
             std::vector<std::shared_ptr<Light>> lights, std::shared_ptr<EnvTexture> background)
    : _camera(camera), _obj_group(std::move(objects)), _lights(std::move(lights)), _background(std::move(background)) {
    std::vector<std::shared_ptr<Light>> object_lights;
    std::vector<Object *> obj_stk = {&_obj_group};
    while (!obj_stk.empty()) {
        auto top = obj_stk.back();
        obj_stk.pop_back();
        auto children = top->children();
        obj_stk.insert(obj_stk.end(), children.begin(), children.end());
        if (top->has_material() && !top->material().emission_texture().empty()) {
            object_lights.emplace_back(std::make_shared<ObjectLight>(top));
        }
    }

    for (const auto &light : _lights) {
        if (light->is_delta()) {
            _delta_lights.mutable_lights().emplace_back(light);
        } else {
            object_lights.emplace_back(light);
        }
    }

    if (_background == nullptr) {
        _background = default_background();
    }
    if (_background->rgb_texture().is_map()) {
        object_lights.emplace_back(std::make_shared<EnvLight>(_background.get()));
    }
    _object_lights = LightGroup(std::move(object_lights));
}

} // namespace cpu
} // namespace tinypt
