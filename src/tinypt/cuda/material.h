#pragma once

#include "tinypt/cuda/ray.h"
#include "tinypt/cuda/texture.h"

#ifndef __NVCC__
#include "tinypt/cpu/material.h"
#endif

namespace tinypt {
namespace cuda {

struct BXDF {
    enum Type {
        TYPE_LAMBERTIAN,
        TYPE_METAL,
        TYPE_DIELECTRIC,
        TYPE_GLOSSY,
        TYPE_LAMBERTIAN_GLOSSY,
        TYPE_LAMBERTIAN_METAL,
    };
    Type _type;
    bool _is_specular;

    BXDF(Type type = TYPE_LAMBERTIAN, bool is_specular = false) : _type(type), _is_specular(is_specular) {}

    __device__ Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const;
    __device__ Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const;
    __device__ float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const;

    __device__ bool is_specular() const { return _is_specular; }

    __device__ const BXDF *sample_bxdf(RandEngine &rng) const;
    __device__ float pdf_bxdf(const BXDF *bxdf) const;
};

struct Lambertian : public BXDF {
#ifndef __NVCC__
    Lambertian() : Lambertian(RGBTexture()) {}
    Lambertian(const RGBTexture &diffuse_texture) : BXDF(TYPE_LAMBERTIAN, false), _diffuse_texture(diffuse_texture) {}

    static Lambertian create(const cpu::Lambertian &cpu_lambertian) {
        return Lambertian(RGBTexture::create(cpu_lambertian._diffuse_texture));
    }
    static void destroy(Lambertian &lambertian) { RGBTexture::destroy(lambertian._diffuse_texture); }
#endif
    __device__ Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        float cos_theta = hit.shade_normal.dot(out_dir);
        if (cos_theta > 0) {
            return cos_theta * M_1_PIf32 * _diffuse_texture.color_at(hit.uv);
        } else {
            return Vec3f::Zero();
        }
    }

    __device__ Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
        return sample_impl(ray, hit, rng);
    }

    __device__ float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        float cos_theta = out_dir.dot(hit.normal);
        return (cos_theta > 0) ? cos_theta * M_1_PIf32 : 0;
    }

    __device__ static Vec3f sample_impl(const Ray &ray, const Hit &hit, RandEngine &rng) {
        float phi = rng.random(2.f * M_PIf32);
        float sin2_theta = rng.random();
        float sin_theta = std::sqrt(sin2_theta);
        float cos_theta = std::sqrt(1 - sin2_theta);

        Rotation rot = Rotation::from_direction(hit.normal);
        Vec3f diff_dir = rot * Vec3f(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
        return diff_dir;
    }

    RGBTexture _diffuse_texture;
};

struct Metal : public BXDF {
#ifndef __NVCC__
    Metal() : Metal(RGBTexture{}) {}
    Metal(const RGBTexture &diffuse_texture) : BXDF(TYPE_METAL, true), _diffuse_texture(diffuse_texture) {}

    static Metal create(const cpu::Metal &cpu_metal) { return Metal(RGBTexture::create(cpu_metal._diffuse_texture)); }
    static void destroy(Metal &metal) { RGBTexture::destroy(metal._diffuse_texture); }
#endif
    __device__ Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        return _diffuse_texture.color_at(hit.uv);
    }
    __device__ Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
        Vec3f refl_dir = Ray::reflect(ray.dir, hit.normal);
        return refl_dir;
    }
    __device__ float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        printf("unreachable!\n");
        return -1;
    }

    RGBTexture _diffuse_texture;
};

struct Dielectric : public BXDF {
#ifndef __NVCC__
    Dielectric() : Dielectric({}, 1) {}
    Dielectric(const RGBTexture &diffuse_texture, float ior)
        : BXDF(TYPE_DIELECTRIC, true), _diffuse_texture(diffuse_texture), _ior(ior) {}

    static Dielectric create(const cpu::Dielectric &cpu_dielectric) {
        return Dielectric(RGBTexture::create(cpu_dielectric._diffuse_texture), cpu_dielectric._ior);
    }

    static void destroy(Dielectric &dielectric) { RGBTexture::destroy(dielectric._diffuse_texture); }
#endif
    __device__ Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        return _diffuse_texture.color_at(hit.uv);
    }
    __device__ Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
        Vec3f refl_dir = Ray::reflect(ray.dir, hit.normal);
        constexpr float nc = 1;
        const float ng = _ior;
        float eta = hit.into ? (nc / ng) : (ng / nc); // eta = n_incidence / n_transmission
        float cosi = -ray.dir.dot(hit.normal);
        float cos2t = 1 - eta * eta * (1 - cosi * cosi);
        Vec3f out_dir;
        if (cos2t < 0) {
            // total internal reflection
            out_dir = refl_dir;
        } else {
            float cost = std::sqrt(cos2t);
            Vec3f refr_dir = Ray::refract(ray.dir, hit.normal, eta, cosi, cost);
            // Approximation of Fresnel effect
            // paper: An inexpensive BRDF model for physically‐based rendering
            float R0 = square(ng - nc) / square(ng + nc);
            float c = 1 - (hit.into ? cosi : cost);
            float Re = R0 + (1 - R0) * c * c * c * c * c;

            if (rng.random() < Re) {
                out_dir = refl_dir;
            } else {
                out_dir = refr_dir;
            }
        }
        return out_dir;
    }
    __device__ float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        printf("unreachable!\n");
        return -1;
    }

    RGBTexture _diffuse_texture;
    float _ior; // index of refraction
};

struct BlinnPhongModel {
    BlinnPhongModel() : _shininess(0) {}
    BlinnPhongModel(float shininess) : _shininess(shininess) {}
#ifndef __NVCC__
    BlinnPhongModel(const cpu::BlinnPhongModel &cpu_blinn_phong) : BlinnPhongModel(cpu_blinn_phong._shininess) {}
#endif
    __device__ float D(const Vec3f &l, const Vec3f &n, const Vec3f &h) const { return D_impl(l, n, h); }
    __device__ float G(const Vec3f &l, const Vec3f &n, const Vec3f &h, const Vec3f &v) const {
        auto G1 = [this, &n](const Vec3f &w) {
            float cos_theta = w.dot(n);
            cos_theta = fmaxf(fminf(cos_theta, 1.f), 0.f);
            float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
            float a = std::sqrt(.5f * _shininess + 1) * cos_theta / sin_theta;
            if (a < 1.6) {
                float a2 = a * a;
                return (3.535f * a + 2.181f * a2) / (1 + 2.276f * a + 2.577f * a2);
            } else {
                return 1.f;
            }
        };
        return G1(l) * G1(v);
    }

    __device__ Vec3f sample_h(const Vec3f &l, const Vec3f &n, RandEngine &rng) const {
        float cos_beta = std::pow(rng.random(), 1 / (_shininess + 2));
        float sin_beta = std::sqrt(1 - cos_beta * cos_beta);
        float phi = rng.random(2.f * M_PIf32);
        Rotation rot = Rotation::from_direction(n);
        Vec3f h = rot * Vec3f(std::cos(phi) * sin_beta, std::sin(phi) * sin_beta, cos_beta);
        return h;
    }
    __device__ float pdf_h(const Vec3f &l, const Vec3f &n, const Vec3f &h) const { return D_impl(l, n, h); }

    __device__ float D_impl(const Vec3f &l, const Vec3f &n, const Vec3f &h) const {
        float cos_beta = h.dot(n);
        return (cos_beta <= 0) ? 0 : (_shininess + 2) * (.5f * M_1_PIf32) * std::pow(cos_beta, _shininess + 1);
    }

    float _shininess;
};

struct Glossy : public BXDF {
    Glossy() : Glossy({}, {}) {}
    Glossy(const RGBTexture &diffuse_texture, const BlinnPhongModel &micro_model)
        : BXDF(TYPE_GLOSSY, false), _diffuse_texture(diffuse_texture), _micro_model(micro_model) {}
#ifndef __NVCC__
    static Glossy create(const cpu::Glossy &cpu_glossy) {
        auto cpu_micro_model = dynamic_cast<cpu::BlinnPhongModel *>(cpu_glossy._micro_model.get());
        TINYPT_CHECK(cpu_micro_model) << "Micro model not supported on CUDA";
        return Glossy(RGBTexture::create(cpu_glossy._diffuse_texture), *cpu_micro_model);
    }
    static void destroy(Glossy &glossy) { RGBTexture::destroy(glossy._diffuse_texture); }
#endif
    __device__ Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        Vec3f l = -ray.dir;
        const Vec3f &v = out_dir;
        const Vec3f &n = hit.shade_normal;
        if (!is_same_side(v, n)) {
            return Vec3f::Zero();
        }
        Vec3f h = (v + l).normalized();
        Vec3f color = _diffuse_texture.color_at(hit.uv);
        Vec3f decay =
            F(color, v, h) * (_micro_model.D(l, n, h) * _micro_model.G(l, n, h, v) / (4 * n.dot(l) * n.dot(v)));
        return decay.allFinite() ? decay : Vec3f::Zero();
    }
    __device__ Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
        Vec3f micro_normal = _micro_model.sample_h(-ray.dir, hit.normal, rng);
        Vec3f out_dir = Ray::reflect(ray.dir, micro_normal);
        if (!is_same_side(out_dir, hit.normal)) {
            // falling back to lambertian
            return Lambertian::sample_impl(ray, hit, rng);
        }
        return out_dir;
    }
    __device__ float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        Vec3f l = -ray.dir;
        const Vec3f &v = out_dir;
        const Vec3f &n = hit.normal;
        if (!is_same_side(v, n)) {
            return 0;
        }
        Vec3f h = (v + l).normalized();
        return _micro_model.pdf_h(l, n, h) / (4 * n.dot(l) * n.dot(v));
    }

    // Fresnel effect
    __device__ static Vec3f F(const Vec3f &f0, const Vec3f &v, const Vec3f &h) {
        auto pow5 = [](float x) { return (x * x) * (x * x) * x; };
        return f0 + (Vec3f::Ones() - f0) * pow5(1.f - v.dot(h));
    }

    RGBTexture _diffuse_texture;
    BlinnPhongModel _micro_model;
};

struct LambertianGlossy : public BXDF {
    LambertianGlossy() : LambertianGlossy({}, {}) {}
    LambertianGlossy(const Lambertian &lambertian, const Glossy &glossy)
        : BXDF(TYPE_LAMBERTIAN_GLOSSY, false), _lambertian(lambertian), _glossy(glossy) {}
#ifndef __NVCC__
    static LambertianGlossy create(const cpu::AddBXDF &cpu_add_bxdf) {
        auto cpu_lambertian = dynamic_cast<const cpu::Lambertian *>(cpu_add_bxdf.bxdf1());
        TINYPT_CHECK(cpu_lambertian) << "Not supported material on CUDA";
        auto lambertian = Lambertian::create(*cpu_lambertian);
        auto cpu_glossy = dynamic_cast<const cpu::Glossy *>(cpu_add_bxdf.bxdf2());
        TINYPT_CHECK(cpu_glossy) << "Not supported material on CUDA";
        auto glossy = Glossy::create(*cpu_glossy);
        return LambertianGlossy(lambertian, glossy);
    }
    static void destroy(LambertianGlossy &lambertian_glossy) {
        Lambertian::destroy(lambertian_glossy._lambertian);
        Glossy::destroy(lambertian_glossy._glossy);
    }
#endif
    __device__ Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        return _lambertian.f(ray, hit, out_dir) + _glossy.f(ray, hit, out_dir);
    }
    __device__ Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
        return (rng.random() < .5f) ? _lambertian.sample(ray, hit, rng) : _glossy.sample(ray, hit, rng);
    }
    __device__ float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
        return .5f * (_lambertian.pdf(ray, hit, out_dir) + _glossy.pdf(ray, hit, out_dir));
    }

    Lambertian _lambertian;
    Glossy _glossy;
};

struct LambertianMetal : public BXDF {
    LambertianMetal() : LambertianMetal({}, {}) {}
    LambertianMetal(const Lambertian &lambertian, const Metal &metal)
        : BXDF(TYPE_LAMBERTIAN_METAL, true), _lambertian(lambertian), _metal(metal) {}
#ifndef __NVCC__
    static LambertianMetal create(const cpu::AddBXDF &cpu_add_bxdf) {
        auto cpu_lambertian = dynamic_cast<const cpu::Lambertian *>(cpu_add_bxdf.bxdf1());
        TINYPT_CHECK(cpu_lambertian) << "Not supported material on CUDA";
        auto lambertian = Lambertian::create(*cpu_lambertian);
        auto cpu_metal = dynamic_cast<const cpu::Metal *>(cpu_add_bxdf.bxdf2());
        TINYPT_CHECK(cpu_metal) << "Not supported material on CUDA";
        auto metal = Metal::create(*cpu_metal);
        return LambertianMetal(lambertian, metal);
    }
    static void destroy(LambertianMetal &lambertian_metal) {
        Lambertian::destroy(lambertian_metal._lambertian);
        Metal::destroy(lambertian_metal._metal);
    }
#endif

    Lambertian _lambertian;
    Metal _metal;
};

__device__ inline Vec3f BXDF::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    switch (_type) {
    case TYPE_LAMBERTIAN:
        return ((Lambertian *)this)->f(ray, hit, out_dir);
    case TYPE_METAL:
        return ((Metal *)this)->f(ray, hit, out_dir);
    case TYPE_DIELECTRIC:
        return ((Dielectric *)this)->f(ray, hit, out_dir);
    case TYPE_GLOSSY:
        return ((Glossy *)this)->f(ray, hit, out_dir);
    case TYPE_LAMBERTIAN_GLOSSY:
        return ((LambertianGlossy *)this)->f(ray, hit, out_dir);
    case TYPE_LAMBERTIAN_METAL:
    default:
        printf("unreachable!\n");
        return Vec3f::Zero();
    }
}
__device__ inline Vec3f BXDF::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
    switch (_type) {
    case TYPE_LAMBERTIAN:
        return ((Lambertian *)this)->sample(ray, hit, rng);
    case TYPE_METAL:
        return ((Metal *)this)->sample(ray, hit, rng);
    case TYPE_DIELECTRIC:
        return ((Dielectric *)this)->sample(ray, hit, rng);
    case TYPE_GLOSSY:
        return ((Glossy *)this)->sample(ray, hit, rng);
    case TYPE_LAMBERTIAN_GLOSSY:
        return ((LambertianGlossy *)this)->sample(ray, hit, rng);
    case TYPE_LAMBERTIAN_METAL:
    default:
        printf("unreachable!\n");
        return Vec3f::Zero();
    }
}
__device__ inline float BXDF::pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    switch (_type) {
    case TYPE_LAMBERTIAN:
        return ((Lambertian *)this)->pdf(ray, hit, out_dir);
    case TYPE_METAL:
        return ((Metal *)this)->pdf(ray, hit, out_dir);
    case TYPE_DIELECTRIC:
        return ((Dielectric *)this)->pdf(ray, hit, out_dir);
    case TYPE_GLOSSY:
        return ((Glossy *)this)->pdf(ray, hit, out_dir);
    case TYPE_LAMBERTIAN_GLOSSY:
        return ((LambertianGlossy *)this)->pdf(ray, hit, out_dir);
    case TYPE_LAMBERTIAN_METAL:
    default:
        printf("unreachable!\n");
        return -1;
    }
}
__device__ inline const BXDF *BXDF::sample_bxdf(RandEngine &rng) const {
    switch (_type) {
    case TYPE_LAMBERTIAN:
    case TYPE_METAL:
    case TYPE_DIELECTRIC:
    case TYPE_GLOSSY:
    case TYPE_LAMBERTIAN_GLOSSY:
        return this;
    case TYPE_LAMBERTIAN_METAL: {
        auto self = (LambertianMetal *)this;
        return (rng.random() < .5f) ? (const BXDF *)&self->_lambertian : (const BXDF *)&self->_metal;
    }
    default:
        printf("unreachable!\n");
        return nullptr;
    }
}
__device__ inline float BXDF::pdf_bxdf(const BXDF *bxdf) const {
    switch (_type) {
    case TYPE_LAMBERTIAN:
    case TYPE_METAL:
    case TYPE_DIELECTRIC:
    case TYPE_GLOSSY:
    case TYPE_LAMBERTIAN_GLOSSY:
        return 1;
    case TYPE_LAMBERTIAN_METAL:
        return 0.5;
    default:
        printf("unreachable!\n");
        return -1;
    }
}

struct Material {
    BXDF *_surface;
    RGBTexture _emission_texture;
    AlphaTexture _alpha_texture;
    //    BumpTexture _bump_texture;

    Material() : _surface(nullptr) {}
    Material(BXDF *surface, const RGBTexture &emission_texture, const AlphaTexture &alpha_texture)
        : _surface(surface), _emission_texture(emission_texture), _alpha_texture(alpha_texture) {}

#ifndef __NVCC__
    static Material create(const cpu::Material &cpu_material) {
        BXDF *surface;
        if (auto cpu_lambertian = dynamic_cast<const cpu::Lambertian *>(cpu_material.surface())) {
            auto lambertian = Lambertian::create(*cpu_lambertian);
            CHECK_CUDA(cudaMalloc(&surface, sizeof(Lambertian)));
            CHECK_CUDA(cudaMemcpy(surface, &lambertian, sizeof(Lambertian), cudaMemcpyHostToDevice));
        } else if (auto cpu_metal = dynamic_cast<const cpu::Metal *>(cpu_material.surface())) {
            auto metal = Metal::create(*cpu_metal);
            CHECK_CUDA(cudaMalloc(&surface, sizeof(Metal)));
            CHECK_CUDA(cudaMemcpy(surface, &metal, sizeof(Metal), cudaMemcpyHostToDevice));
        } else if (auto cpu_dielectric = dynamic_cast<const cpu::Dielectric *>(cpu_material.surface())) {
            auto dielectric = Dielectric::create(*cpu_dielectric);
            CHECK_CUDA(cudaMalloc(&surface, sizeof(Dielectric)));
            CHECK_CUDA(cudaMemcpy(surface, &dielectric, sizeof(Dielectric), cudaMemcpyHostToDevice));
        } else if (auto cpu_add_bxdf = dynamic_cast<const cpu::AddBXDF *>(cpu_material.surface())) {
            if (dynamic_cast<const cpu::Glossy *>(cpu_add_bxdf->bxdf2())) {
                auto lambertian_glossy = LambertianGlossy::create(*cpu_add_bxdf);
                CHECK_CUDA(cudaMalloc(&surface, sizeof(LambertianGlossy)));
                CHECK_CUDA(cudaMemcpy(surface, &lambertian_glossy, sizeof(LambertianGlossy), cudaMemcpyHostToDevice));
            } else if (dynamic_cast<const cpu::Metal *>(cpu_add_bxdf->bxdf2())) {
                auto lambertian_metal = LambertianMetal::create(*cpu_add_bxdf);
                CHECK_CUDA(cudaMalloc(&surface, sizeof(LambertianMetal)));
                CHECK_CUDA(cudaMemcpy(surface, &lambertian_metal, sizeof(LambertianMetal), cudaMemcpyHostToDevice));
            } else {
                TINYPT_THROW << "Material not supported on CUDA";
            }
        } else {
            TINYPT_THROW << "material not implemented on CUDA";
        }
        return Material(surface, RGBTexture::create(cpu_material.emission_texture()),
                        AlphaTexture::create(cpu_material.alpha_texture()));
    }

    static void destroy(Material &material) {
        BXDF host_surface;
        CHECK_CUDA(cudaMemcpy(&host_surface, material._surface, sizeof(BXDF), cudaMemcpyDeviceToHost));
        switch (host_surface._type) {
        case BXDF::TYPE_LAMBERTIAN: {
            Lambertian lambertian;
            CHECK_CUDA(cudaMemcpy(&lambertian, material._surface, sizeof(Lambertian), cudaMemcpyDeviceToHost));
            Lambertian::destroy(lambertian);
        } break;
        case BXDF::TYPE_METAL: {
            Metal metal;
            CHECK_CUDA(cudaMemcpy(&metal, material._surface, sizeof(Metal), cudaMemcpyDeviceToHost));
            Metal::destroy(metal);
        } break;
        case BXDF::TYPE_DIELECTRIC: {
            Dielectric dielectric;
            CHECK_CUDA(cudaMemcpy(&dielectric, material._surface, sizeof(Dielectric), cudaMemcpyDeviceToHost));
            Dielectric::destroy(dielectric);
        } break;
        case BXDF::TYPE_GLOSSY: {
            Glossy glossy;
            CHECK_CUDA(cudaMemcpy(&glossy, material._surface, sizeof(Glossy), cudaMemcpyDeviceToHost));
            Glossy::destroy(glossy);
        } break;
        case BXDF::TYPE_LAMBERTIAN_GLOSSY: {
            LambertianGlossy lambertian_glossy;
            CHECK_CUDA(
                cudaMemcpy(&lambertian_glossy, material._surface, sizeof(LambertianGlossy), cudaMemcpyDeviceToHost));
            LambertianGlossy::destroy(lambertian_glossy);
        } break;
        case BXDF::TYPE_LAMBERTIAN_METAL: {
            LambertianMetal lambertian_metal;
            CHECK_CUDA(
                cudaMemcpy(&lambertian_metal, material._surface, sizeof(LambertianMetal), cudaMemcpyDeviceToHost));
            LambertianMetal::destroy(lambertian_metal);
        } break;
        default:
            TINYPT_THROW << "unreachable!";
        }
        CHECK_CUDA(cudaFree(material._surface));
        RGBTexture::destroy(material._emission_texture);
        AlphaTexture::destroy(material._alpha_texture);
    }
#endif
    __device__ const RGBTexture &emission_texture() const { return _emission_texture; }
    __device__ const AlphaTexture &alpha_texture() const { return _alpha_texture; }
    __device__ const BXDF *surface() const { return _surface; }
};

} // namespace cuda
} // namespace tinypt