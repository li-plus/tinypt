#include "tinypt/cpu/path_tracer.h"

#include <chrono>

namespace tinypt {
namespace cpu {

class PixelSampler {
  public:
    PixelSampler(int x, int y, int num_samples) {
        constexpr int min_samples_per_subpixel = 16;
        if (num_samples < 2 * 2 * min_samples_per_subpixel) {
            _block_size = 1;
        } else if (num_samples < 3 * 3 * min_samples_per_subpixel) {
            _block_size = 2;
        } else if (num_samples < 4 * 4 * min_samples_per_subpixel) {
            _block_size = 3;
        } else {
            _block_size = 4;
        }
        _num_subsamples = num_samples / (_block_size * _block_size);
        _subpixel_width = 1.f / (float)_block_size;

        _xy = Vec2f(x, y);
        _subpixel_idx = Vec2i::Zero();
        _subpixel = _xy - Vec2f(.5f, .5f);
    }

    Vec2f sample(RandEngine &rng) {
        Vec2f pos(_subpixel.x() + rng.random(_subpixel_width), _subpixel.y() + rng.random(_subpixel_width));
        return pos;
    }

    void next() {
        _subpixel_idx.x()++;
        _subpixel.x() += _subpixel_width;
        if (_subpixel_idx.x() == _block_size) {
            _subpixel_idx.x() = 0;
            _subpixel.x() = _xy.x() - .5f;
            _subpixel_idx.y()++;
            _subpixel.y() += _subpixel_width;
        }
    }

    bool is_end() const { return _subpixel_idx.y() >= _block_size; }

    int num_subsamples() const { return _num_subsamples; }
    int num_subpixels() const { return _block_size * _block_size; }

  private:
    int _block_size;
    int _num_subsamples;
    float _subpixel_width;
    // sample state
    Vec2f _xy;
    Vec2i _subpixel_idx;
    Vec2f _subpixel;
};

static inline Vec3f clip(const Vec3f &v, float minv = 0, float maxv = 1) { return v.cwiseMax(minv).cwiseMin(maxv); }

Image PathTracer::render(const Scene &scene, int num_samples) const {
    Image img(scene.camera().width(), scene.camera().height(), 3);

    auto start = std::chrono::system_clock::now();
    uint32_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count();

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < scene.camera().height(); y++) {
        // TODO: arguments
        auto now = std::chrono::system_clock::now();
        float duration = std::chrono::duration<float>(now - start).count();
        fprintf(stderr, "\rRendering (%d spp) %.2f%% %.3fs", num_samples, 100. * (y + 1) / scene.camera().height(),
                duration);

        RandEngine rng(((uint64_t)y << 32) | timestamp);
        for (int x = 0; x < scene.camera().width(); x++) {
            Vec3f color = Vec3f::Zero();
            PixelSampler sampler(x, y, num_samples);
            while (!sampler.is_end()) {
                Vec3f subcolor = Vec3f::Zero();
                for (int s = 0; s < sampler.num_subsamples(); s++) {
                    Vec2f pos = sampler.sample(rng);
                    DCHECK_GE(pos.x(), x - .5f - EPS);
                    DCHECK_LE(pos.x(), x + .5f + EPS);
                    DCHECK_GE(pos.y(), y - .5f - EPS);
                    DCHECK_LE(pos.y(), y + .5f + EPS);
                    Ray ray = scene.camera().shoot_ray(pos.x(), pos.y(), rng);
                    subcolor += radiance(scene, ray, rng);
                }
                color += clip(subcolor / sampler.num_subsamples());
                sampler.next();
            }
            color = clip(color / sampler.num_subpixels());
            img.at<Vec3f>(x, y) = color;
        }
    }
    fprintf(stderr, "\n");
    return img;
}

Vec3f PathTracer::radiance(const Scene &scene, Ray ray, RandEngine &rng) {
    constexpr int MAX_DEPTH = 5;
    constexpr float LIGHT_PROB = 1.f / MAX_DEPTH;
    int depth = 0;
    Vec3f decay = Vec3f::Ones();
    Vec3f color = Vec3f::Zero();

    while (true) {
        DCHECK(is_normalized(ray.dir));
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
    DCHECK(color.allFinite());
    return color;
}

} // namespace cpu
} // namespace tinypt