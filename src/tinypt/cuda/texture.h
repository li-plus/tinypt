#pragma once

#include "tinypt/cuda/geometry.h"

#ifndef __NVCC__
#include "tinypt/cpu/texture.h"
#endif

#include <cassert>

#ifndef __NVCC__
template <typename T>
T tex2D(cudaTextureObject_t tex, float x, float y) {
    TINYPT_THROW << "dummy function";
}
#endif

namespace tinypt {
namespace cuda {

struct TextureImage {
#ifndef __NVCC__
    TextureImage() : _res{}, _obj{}, _width(-1), _height(-1) {}
    TextureImage(const cudaResourceDesc &res, cudaTextureObject_t obj, int width, int height)
        : _res(res), _obj(obj), _width(width), _height(height) {}

    static TextureImage create(const cpu::Image &cpu_image) {
        if (cpu_image.channels() == 3) {
            std::vector<uchar4> cpu_buf;
            cpu_buf.reserve(cpu_image.width() * cpu_image.height());
            for (int y = 0; y < cpu_image.height(); y++) {
                for (int x = 0; x < cpu_image.width(); x++) {
                    Eigen::Vector3f cpu_color = cpu_image.at<Eigen::Vector3f>(x, y);
                    cpu_buf.push_back({(uint8_t)(cpu_color.x() * 255), (uint8_t)(cpu_color.y() * 255),
                                       (uint8_t)(cpu_color.z() * 255), 0});
                }
            }
            return create(cpu_buf, cpu_image.width(), cpu_image.height());
        } else if (cpu_image.channels() == 1) {
            std::vector<uchar> cpu_buf;
            cpu_buf.reserve(cpu_image.width() * cpu_image.height());
            for (int y = 0; y < cpu_image.height(); y++) {
                for (int x = 0; x < cpu_image.width(); x++) {
                    float value = cpu_image.at<float>(x, y);
                    cpu_buf.push_back((uint8_t)(value * 255));
                }
            }
            return create(cpu_buf, cpu_image.width(), cpu_image.height());
        } else {
            TINYPT_THROW << "Unexpected image channels [" << cpu_image.channels() << "]";
        }
    }

    template <typename DT>
    static TextureImage create(const std::vector<DT> &buf, int width, int height) {
        if (buf.empty()) {
            return {};
        }
        cudaArray_t cuda_buf;
        auto channel_desc = cudaCreateChannelDesc<DT>();
        CHECK_CUDA(cudaMallocArray(&cuda_buf, &channel_desc, width, height));
        CHECK_CUDA(cudaMemcpy2DToArray(cuda_buf, 0, 0, buf.data(), width * sizeof(DT), width * sizeof(DT), height,
                                       cudaMemcpyHostToDevice));

        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = cuda_buf;

        cudaTextureDesc tex_desc{};
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = true;

        cudaTextureObject_t obj{};
        CHECK_CUDA(cudaCreateTextureObject(&obj, &res, &tex_desc, nullptr));

        return TextureImage(res, obj, width, height);
    }

    static void destroy(TextureImage &img) {
        if (!img.empty()) {
            CHECK_CUDA(cudaDestroyTextureObject(img._obj));
            CHECK_CUDA(cudaFreeArray(img._res.res.array.array));
            img._width = img._height = -1;
        }
    }
#endif

    template <typename DT>
    __device__ DT at(const Vec2f &uv) const;

    __device__ bool empty() const { return _width <= 0; }

    cudaResourceDesc _res;
    cudaTextureObject_t _obj;
    int _width;
    int _height;
};

template <>
__device__ inline Vec3f TextureImage::at<Vec3f>(const Vec2f &uv) const {
    auto texel = tex2D<float4>(_obj, uv.x(), uv.y());
    return {texel.x, texel.y, texel.z};
}
template <>
__device__ inline float TextureImage::at<float>(const Vec2f &uv) const {
    return tex2D<float>(_obj, uv.x(), uv.y());
}

struct RGBTexture {
    RGBTexture() : _value(Vec3f::Zero()) {}
    RGBTexture(const Vec3f &value, const TextureImage &map) : _value(value), _map(map) {}

#ifndef __NVCC__
    static RGBTexture create(const cpu::RGBTexture &cpu_tex) {
        return RGBTexture(cpu_tex.value(), TextureImage::create(cpu_tex.map()));
    }
    static void destroy(RGBTexture &obj) { TextureImage::destroy(obj._map); }
#endif
    __device__ bool is_map() const { return !_map.empty(); }

    __device__ Vec3f color_at(const Vec2f &uv) const {
        if (_map.empty()) {
            return _value;
        } else {
            return _map.at<Vec3f>(uv);
        }
    }

    Vec3f _value;
    TextureImage _map;
};

struct AlphaTexture {
    AlphaTexture() : _value(1) {}
    AlphaTexture(float value, const TextureImage &map) : _value(value), _map(map) {}

#ifndef __NVCC__
    static AlphaTexture create(const cpu::AlphaTexture &cpu_alpha_texture) {
        return AlphaTexture(cpu_alpha_texture.value(), TextureImage::create(cpu_alpha_texture.map()));
    }
    static void destroy(AlphaTexture &alpha_texture) { TextureImage::destroy(alpha_texture._map); }
#endif
    __device__ float alpha_at(const Vec2f &uv) const {
        if (_map.empty()) {
            return _value;
        } else {
            return _map.at<float>(uv);
        }
    }

    float _value;
    TextureImage _map;
};

struct EnvTexture {
    EnvTexture() = default;
    EnvTexture(const RGBTexture &env, const Rotation &rotation) : _env(env), _rotation(rotation) {}
#ifndef __NVCC__
    static EnvTexture create(const cpu::EnvTexture &cpu_env_tex) {
        return EnvTexture(RGBTexture::create(cpu_env_tex.rgb_texture()), cpu::Rotation(cpu_env_tex.rotation()));
    }
    static void destroy(EnvTexture &env_tex) { RGBTexture::destroy(env_tex._env); }
#endif

    __device__ Vec3f color_at(const Vec3f &dir) const {
        Vec2f uv;
        if (_env.is_map()) {
            uv = dir2uv(_rotation * dir);
        } else {
            uv = Vec2f::Zero();
        }
        return _env.color_at(uv);
    }

    __device__ static Vec3f uv2dir(const Vec2f &uv) {
        float u = uv.x(), v = uv.y();
        float phi = 2 * M_PIf32 * (1 - u) - M_PIf32;
        float theta = M_PIf32 * (1 - v);
        float cos_theta = std::cos(theta);
        float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
        Vec3f out_dir(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
        return out_dir;
    }
    __device__ static Vec2f dir2uv(const Vec3f &out_dir) {
        float cos_theta = fmaxf(-1.f, fminf(out_dir.z(), 1.f));
        float u = 1 - (std::atan2(out_dir.y(), out_dir.x()) + M_PIf32) * (.5f * M_1_PIf32);
        float v = 1 - std::acos(cos_theta) * M_1_PIf32;
        return {u, v};
    }

    RGBTexture _env;
    Rotation _rotation;
};

} // namespace cuda
} // namespace tinypt