#pragma once

#include "tinypt/cpu/path_tracer.h"
#ifdef TINYPT_ENABLE_CUDA
#include "tinypt/cuda/path_tracer.h"
#endif

#include "tinypt/scene.h"

namespace tinypt {

class PathTracer {
  public:
    PathTracer(Device device_ = {}) : _device(device_) {
        if (_device.is_cpu()) {
            _cpu_pt = std::make_unique<cpu::PathTracer>();
        } else {
#ifdef TINYPT_ENABLE_CUDA
            _cuda_pt = std::make_unique<cuda::PathTracer>();
#else
            TINYPT_THROW << "CUDA is not enabled at compile time";
#endif
        }
    }

    Image render(const Scene &scene, int num_samples) const {
        if (_device.is_cpu()) {
            TINYPT_CHECK(scene.is_cpu()) << "Expect scene on cpu, got cuda scene";
            return _cpu_pt->render(*scene._cpu_scene, num_samples);
        } else {
#ifdef TINYPT_ENABLE_CUDA
            TINYPT_CHECK(scene.is_cuda()) << "Expect scene on cuda, got cpu scene";
            const auto &camera = scene._cuda_scene->_camera;
            auto cuda_img = cuda::Image::create(camera._width, camera._height);
            _cuda_pt->render(*scene._cuda_scene, num_samples, cuda_img);
            auto cpu_img = cuda_img.to_cpu();
            cuda::Image::destroy(cuda_img);
            return cpu_img;
#else
            TINYPT_THROW << "CUDA is not enabled at compile time";
#endif
        }
    }

  private:
    Device _device;
    std::unique_ptr<cpu::PathTracer> _cpu_pt;
#ifdef TINYPT_ENABLE_CUDA
    std::unique_ptr<cuda::PathTracer> _cuda_pt;
#endif
};

} // namespace tinypt