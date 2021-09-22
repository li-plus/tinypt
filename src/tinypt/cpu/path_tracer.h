#pragma once

#include "tinypt/cpu/image.h"
#include "tinypt/cpu/scene.h"

namespace tinypt {
namespace cpu {

class PathTracer {
  public:
    Image render(const Scene &scene, int num_samples) const;

  private:
    static Vec3f radiance(const Scene &scene, Ray ray, RandEngine &rng);
};

} // namespace cpu
} // namespace tinypt
