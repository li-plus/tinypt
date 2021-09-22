#pragma once

#include "tinypt/cuda/scene.h"

namespace tinypt {
namespace cuda {

struct PathTracer {
    void render(const Scene &scene, int num_samples, Image &image) const;
};

} // namespace cuda
} // namespace tinypt