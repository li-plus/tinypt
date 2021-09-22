#pragma once

#include "tinypt/cuda/scene.h"

namespace tinypt {
namespace cuda {

void path_tracer_kernel_launch(const Scene &scene, int num_samples, Image &kernel_image);

} // namespace cuda
} // namespace tinypt