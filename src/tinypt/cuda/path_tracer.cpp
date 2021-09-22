#include "tinypt/cuda/path_tracer.h"

#include "tinypt/cuda/path_tracer_kernel.cuh"

namespace tinypt {
namespace cuda {

void PathTracer::render(const Scene &scene, int num_samples, Image &image) const {
    path_tracer_kernel_launch(scene, num_samples, image);
    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace cuda
} // namespace tinypt