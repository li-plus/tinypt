#include "tinypt/cuda/path_tracer_kernel.cuh"

#include "tinypt/cuda/scene.h"

#include <algorithm>
#include <cstdio>

namespace tinypt {
namespace cuda {

__device__ static Vec3f radiance(Ray ray, const Scene &scene, RandEngine &rng) {
    constexpr int MAX_DEPTH = 5;
    constexpr float LIGHT_PROB = 1.f / MAX_DEPTH;
    int depth = 0;
    Vec3f decay = Vec3f::Ones();
    Vec3f color = Vec3f::Zero();

    while (true) {
        Hit hit;
        scene.intersect(ray, hit, rng);
        if (!hit.is_hit()) {
            color += decay.cwiseProduct(scene.background().color_at(ray.dir));
            break;
        }
        color += decay.cwiseProduct(hit.material->emission_texture().color_at(hit.uv));
        if (depth >= MAX_DEPTH) {
            break;
        }
        Vec3f hit_pos = ray.point_at(hit.t);
        Vec3f curr_albedo;
        Vec3f out_dir;
        const BXDF *bxdf = hit.material->surface()->sample_bxdf(rng);
        float pdf_bxdf = hit.material->surface()->pdf_bxdf(bxdf);
        if (!bxdf->is_specular()) {
            // accumulate delta light contribution
            if (!scene.delta_lights().empty()) {
                auto delta_light = scene.delta_lights().sample_light(rng);
                Vec3f to_light = delta_light->sample(hit_pos, rng);
                Ray shadow_ray(hit_pos, to_light);
                Hit shadow_hit;
                scene.intersect_t(shadow_ray, shadow_hit);
                if (!shadow_hit.is_hit()) {
                    // TODO check t < t_to_light
                    Vec3f light_albedo = bxdf->f(ray, hit, to_light);
                    color += decay.cwiseProduct(light_albedo).cwiseProduct(delta_light->emission()) /
                             scene.delta_lights().pdf_light();
                }
            }
            // non-specular scatter
            float pdf_val;
            if (!scene.object_lights().empty()) {
                if (rng.random() < LIGHT_PROB) {
                    out_dir = scene.object_lights().sample(hit_pos, rng);
                } else {
                    out_dir = bxdf->sample(ray, hit, rng);
                }
                pdf_val = (1 - LIGHT_PROB) * bxdf->pdf(ray, hit, out_dir) +
                          LIGHT_PROB * scene.object_lights().pdf(hit_pos, out_dir);
            } else {
                out_dir = bxdf->sample(ray, hit, rng);
                pdf_val = bxdf->pdf(ray, hit, out_dir);
            }
            curr_albedo = bxdf->f(ray, hit, out_dir) * (1.f / ((pdf_val + EPS) * pdf_bxdf));
        } else {
            out_dir = bxdf->sample(ray, hit, rng);
            curr_albedo = bxdf->f(ray, hit, out_dir) * (1.f / pdf_bxdf);
        }
        decay = decay.cwiseProduct(curr_albedo);
        if (decay.isZero()) {
            break;
        }
        ray = Ray(hit_pos, out_dir);
        depth++;
    }
    return color;
}

static constexpr int BLOCK_SIZE = 32;

__global__ static void path_tracer_kernel(Scene scene, int num_samples, Image image) {
    uint32_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32_t y = blockIdx.y;
    if (x >= scene._camera._width) {
        return;
    }
    uint32_t idx = y * scene._camera._width + x;

    num_samples /= 4;

    Vec3f color = Vec3f::Zero();

    RandEngine rng(idx);

    for (int sy = 0; sy < 2; sy++) {     // 2x2 subpixel rows
        for (int sx = 0; sx < 2; sx++) { // 2x2 subpixel cols
            Vec3f r = Vec3f::Zero();
            for (int s = 0; s < num_samples; s++) {
                float r1 = rng.random(2.f);
                float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                float r2 = rng.random(2.f);
                float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                Ray ray = scene._camera.shoot_ray(((float)sx - .5f + dx) / 2 + (float)x,
                                                  ((float)sy - .5f + dy) / 2 + (float)y, rng);
                r += radiance(ray, scene, rng);
            }
            color += (r / (float)num_samples).clip(0, 1);
        }
    }
    color /= 4;
    color = color.clip(0, 1);

    image._data[idx] = color;
}

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

void path_tracer_kernel_launch(const Scene &scene, int num_samples, Image &kernel_image) {
    dim3 grid_size(ceil_div(scene._camera._width, BLOCK_SIZE), scene._camera._height);
    path_tracer_kernel<<<grid_size, BLOCK_SIZE>>>(scene, num_samples, kernel_image);
}

} // namespace cuda
} // namespace tinypt