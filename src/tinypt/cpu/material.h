#pragma once

#include "tinypt/cpu/mapping.h"
#include "tinypt/cpu/rand.h"
#include "tinypt/cpu/ray.h"
#include "tinypt/cpu/texture.h"

#include <glog/logging.h>

namespace tinypt {
namespace cuda {
struct Lambertian;
struct Metal;
struct Dielectric;
struct Glossy;
struct BlinnPhongModel;
} // namespace cuda

namespace cpu {

class BXDF {
  public:
    BXDF(bool is_specular = false) : _is_specular(is_specular) {}
    virtual ~BXDF() = default;

    bool is_specular() const { return _is_specular; }
    void set_specular(bool is_specular) { _is_specular = is_specular; }

    virtual const BXDF *sample_bxdf(RandEngine &rng) const { return this; }
    virtual float pdf_bxdf(const BXDF *bxdf) const { return 1; }
    virtual Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const = 0;
    virtual Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const = 0;
    virtual float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const = 0;

  private:
    bool _is_specular;
};

class Lambertian : public BXDF {
    friend tinypt::cuda::Lambertian;

  public:
    Lambertian() = default;
    explicit Lambertian(RGBTexture diffuse_texture) : _diffuse_texture(std::move(diffuse_texture)) {}

    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override;
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;

    static Vec3f sample_impl(const Ray &ray, const Hit &hit, RandEngine &rng);

  private:
    RGBTexture _diffuse_texture;
};

class MicrofacetModel {
  public:
    virtual ~MicrofacetModel() = default;
    // Here we make D = BSDF * n.dot(h)
    virtual float D(const Vec3f &l, const Vec3f &n, const Vec3f &h) const = 0;
    virtual float G(const Vec3f &l, const Vec3f &n, const Vec3f &h, const Vec3f &v) const = 0;
    // sample a microfacet normal vector
    virtual Vec3f sample_h(const Vec3f &l, const Vec3f &n, RandEngine &rng) const = 0;
    // Normal Distribution Function (NDF)
    virtual float pdf_h(const Vec3f &l, const Vec3f &n, const Vec3f &h) const = 0;
};

class BlinnPhongModel : public MicrofacetModel {
    friend tinypt::cuda::BlinnPhongModel;

  public:
    explicit BlinnPhongModel(float shininess) : _shininess(shininess) {}

    float D(const Vec3f &l, const Vec3f &n, const Vec3f &h) const override;
    float G(const Vec3f &l, const Vec3f &n, const Vec3f &h, const Vec3f &v) const override;

    Vec3f sample_h(const Vec3f &l, const Vec3f &n, RandEngine &rng) const override;
    float pdf_h(const Vec3f &l, const Vec3f &n, const Vec3f &h) const override;

    float D_impl(const Vec3f &l, const Vec3f &n, const Vec3f &h) const;

  private:
    float _shininess;
};

class TrowbridgeReitzModel : public MicrofacetModel {
  public:
    explicit TrowbridgeReitzModel(float alpha2) : _alpha2(alpha2) {}

    float D(const Vec3f &l, const Vec3f &n, const Vec3f &h) const override;
    float G(const Vec3f &l, const Vec3f &n, const Vec3f &h, const Vec3f &v) const override;

    Vec3f sample_h(const Vec3f &l, const Vec3f &n, RandEngine &rng) const override;
    float pdf_h(const Vec3f &l, const Vec3f &n, const Vec3f &h) const override;

    float D_impl(const Vec3f &n, const Vec3f &h) const;

  private:
    float _alpha2;
};

class Glossy : public BXDF {
    friend tinypt::cuda::Glossy;

  public:
    enum MicrofacetModelType { MICRO_BLINN_PHONG, MICRO_TROWBRIDGE_REITZ };

    Glossy(float shininess, RGBTexture diffuse_texture, MicrofacetModelType micro_type = MICRO_BLINN_PHONG);

    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override;
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;

    // Fresnel effect
    static Vec3f F(const Vec3f &f0, const Vec3f &v, const Vec3f &h);

  private:
    RGBTexture _diffuse_texture;
    std::unique_ptr<MicrofacetModel> _micro_model;
};

class Metal : public BXDF {
    friend tinypt::cuda::Metal;

  public:
    explicit Metal(RGBTexture diffuse_texture) : BXDF(true), _diffuse_texture(std::move(diffuse_texture)) {}

    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override;
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;

  private:
    RGBTexture _diffuse_texture;
};

class Dielectric : public BXDF {
    friend tinypt::cuda::Dielectric;

  public:
    Dielectric(RGBTexture diffuse_texture, float ior)
        : BXDF(true), _diffuse_texture(std::move(diffuse_texture)), _ior(ior) {}

    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override;
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override {
        TINYPT_THROW_EX(std::logic_error) << "not implemented";
    }

  private:
    RGBTexture _diffuse_texture;
    float _ior; // index of refraction
};

class AddBXDF : public BXDF {
  public:
    AddBXDF() = default;
    AddBXDF(std::shared_ptr<BXDF> bxdf1, std::shared_ptr<BXDF> bxdf2);

    const BXDF *bxdf1() const { return _bxdf1.get(); }
    const BXDF *bxdf2() const { return _bxdf2.get(); }

    const BXDF *sample_bxdf(RandEngine &rng) const override;
    float pdf_bxdf(const BXDF *bxdf) const override;

    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override;
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;

  private:
    std::shared_ptr<BXDF> _bxdf1;
    std::shared_ptr<BXDF> _bxdf2;
};

class MixBXDF : public BXDF {
  public:
    MixBXDF() : _weight(.5) {}
    MixBXDF(std::shared_ptr<BXDF> bxdf1, std::shared_ptr<BXDF> bxdf2, float weight = 0.5);

    float weight() const { return _weight; }
    void set_weight(float weight) { _weight = weight; }

    const BXDF *sample_bxdf(RandEngine &rng) const override;
    float pdf_bxdf(const BXDF *bxdf) const override;
    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override;
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override;

  private:
    std::shared_ptr<BXDF> _bxdf1;
    std::shared_ptr<BXDF> _bxdf2;
    float _weight;
};

class Material {
  public:
    Material() : _surface(std::make_shared<Lambertian>()) {}
    explicit Material(std::shared_ptr<BXDF> surface, RGBTexture emission_texture = {}, AlphaTexture alpha_texture = {},
                      BumpTexture bump_texture = {})
        : _surface(std::move(surface)), _emission_texture(std::move(emission_texture)),
          _alpha_texture(std::move(alpha_texture)), _bump_texture(std::move(bump_texture)) {}

    const BXDF *surface() const { return _surface.get(); }
    void set_surface(std::shared_ptr<BXDF> surface) { _surface = std::move(surface); }
    const RGBTexture &emission_texture() const { return _emission_texture; }
    void set_emission_texture(RGBTexture emission_texture) { _emission_texture = std::move(emission_texture); }
    const AlphaTexture &alpha_texture() const { return _alpha_texture; }
    void set_alpha_texture(AlphaTexture alpha_texture) { _alpha_texture = std::move(alpha_texture); }
    const BumpTexture &bump_texture() const { return _bump_texture; }
    void set_bump_texture(BumpTexture bump_texture) { _bump_texture = std::move(bump_texture); }

  private:
    std::shared_ptr<BXDF> _surface;
    RGBTexture _emission_texture;
    AlphaTexture _alpha_texture;
    BumpTexture _bump_texture;
};

} // namespace cpu
} // namespace tinypt
