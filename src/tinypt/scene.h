#pragma once

#include "tinypt/cpu/scene.h"
#ifdef TINYPT_ENABLE_CUDA
#include "tinypt/cuda/scene.h"
#endif

#include "tinypt/defs.h"

namespace tinypt {

class Scene {
    friend class PathTracer;

  public:
    Scene() : Scene(std::make_shared<cpu::Scene>()) {}
    explicit Scene(const Camera &camera, std::vector<std::shared_ptr<Object>> objects,
                   std::vector<std::shared_ptr<Light>> lights = {}, std::shared_ptr<EnvTexture> background = nullptr)
        : Scene(std::make_shared<cpu::Scene>(camera, std::move(objects), std::move(lights), std::move(background))) {}

    bool is_cuda() const { return _device.is_cuda(); }
    bool is_cpu() const { return _device.is_cpu(); }

    Scene to(Device device) const {
        if (_device == device) {
            return *this;
        }
        if (device.is_cuda()) {
#ifdef TINYPT_ENABLE_CUDA
            auto cuda_scene = std::make_shared<cuda::Scene>(*_cpu_scene);
            return Scene(std::move(cuda_scene));
#else
            TINYPT_THROW << "CUDA is not enabled at compile time";
#endif
        } else {
            TINYPT_THROW << "Not implemented";
        }
    }

  private:
    Scene(std::shared_ptr<cpu::Scene> cpu_scene) : _device(Device::DEVICE_CPU), _cpu_scene(std::move(cpu_scene)) {}
#ifdef TINYPT_ENABLE_CUDA
    Scene(std::shared_ptr<cuda::Scene> cuda_scene) : _device(Device::DEVICE_CUDA), _cuda_scene(std::move(cuda_scene)) {}
#endif

  private:
    Device _device;
    std::shared_ptr<cpu::Scene> _cpu_scene;
#ifdef TINYPT_ENABLE_CUDA
    std::shared_ptr<cuda::Scene> _cuda_scene;
#endif
};

} // namespace tinypt