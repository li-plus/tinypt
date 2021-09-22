#include "tinypt/cpu/material.h"

namespace tinypt {
namespace cpu {

Vec3f Lambertian::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    float cos_theta = hit.shade_normal.dot(out_dir);
    if (cos_theta > 0) {
        return cos_theta * M_1_PIf32 * _diffuse_texture.color_at(hit.uv);
    } else {
        return Vec3f::Zero();
    }
}

Vec3f Lambertian::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const { return sample_impl(ray, hit, rng); }

float Lambertian::pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    DCHECK(is_normalized(out_dir));
    float cos_theta = out_dir.dot(hit.normal);
    return (cos_theta > 0) ? cos_theta * M_1_PIf32 : 0;
}

Vec3f Lambertian::sample_impl(const Ray &ray, const Hit &hit, RandEngine &rng) {
    float phi = rng.random(2.f * M_PIf32);
    float sin2_theta = rng.random();
    float sin_theta = std::sqrt(sin2_theta);
    float cos_theta = std::sqrt(1 - sin2_theta);

    Rotation rot = Rotation::from_direction(hit.normal);
    Vec3f diff_dir = rot * Vec3f(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
    DCHECK(is_normalized(diff_dir));
    return diff_dir;
}

float BlinnPhongModel::D(const Vec3f &l, const Vec3f &n, const Vec3f &h) const { return D_impl(l, n, h); }

float BlinnPhongModel::G(const Vec3f &l, const Vec3f &n, const Vec3f &h, const Vec3f &v) const {
    // Smith G1 have no closed form solution for the Phong distribution,
    // but Phong is highly similar to Beckmann distribution,
    // so we apply the G1 function of Beckmann for Phong model.
    auto G1 = [this, &n](const Vec3f &w) {
        float cos_theta = w.dot(n);
        cos_theta = std::max(std::min(cos_theta, 1.f), 0.f);
        float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
        DCHECK(std::isfinite(sin_theta));
        float a = std::sqrt(.5f * _shininess + 1) * cos_theta / sin_theta;
        DCHECK_GE(a, 0);
        if (a < 1.6) {
            float a2 = a * a;
            return (3.535f * a + 2.181f * a2) / (1 + 2.276f * a + 2.577f * a2);
        } else {
            return 1.f;
        }
    };
    return G1(l) * G1(v);
}

Vec3f BlinnPhongModel::sample_h(const Vec3f &l, const Vec3f &n, RandEngine &rng) const {
    float cos_beta = std::pow(rng.random(), 1 / (_shininess + 2));
    float sin_beta = std::sqrt(1 - cos_beta * cos_beta);
    float phi = rng.random(2.f * M_PIf32);
    Rotation rot = Rotation::from_direction(n);
    Vec3f h = rot * Vec3f(std::cos(phi) * sin_beta, std::sin(phi) * sin_beta, cos_beta);
    DCHECK_GT(h.dot(n), -EPS);
    return h;
}

float BlinnPhongModel::pdf_h(const Vec3f &l, const Vec3f &n, const Vec3f &h) const { return D_impl(l, n, h); }

float BlinnPhongModel::D_impl(const Vec3f &l, const Vec3f &n, const Vec3f &h) const {
    float cos_beta = h.dot(n);
    return (cos_beta <= 0) ? 0 : (_shininess + 2) * (.5f * M_1_PIf32) * std::pow(cos_beta, _shininess + 1);
}

float TrowbridgeReitzModel::D(const Vec3f &l, const Vec3f &n, const Vec3f &h) const { return D_impl(n, h); }

float TrowbridgeReitzModel::G(const Vec3f &l, const Vec3f &n, const Vec3f &h, const Vec3f &v) const {
    auto G1 = [this, &n](const Vec3f &w) {
        float cos_theta = w.dot(n);
        float tan2_theta = 1 / square(cos_theta) - 1;
        float g = 2 / (1 + std::sqrt(1 + _alpha2 * tan2_theta));
        DCHECK(std::isfinite(g));
        return g;
    };
    return G1(l) * G1(v);
}

Vec3f TrowbridgeReitzModel::sample_h(const Vec3f &l, const Vec3f &n, RandEngine &rng) const {
    float u = rng.random();
    float cos2_beta = (1 - u) / (1 - (1 - _alpha2) * u);
    float cos_beta = std::sqrt(cos2_beta);
    float sin_beta = std::sqrt(1 - cos2_beta);
    DCHECK(std::isfinite(sin_beta));
    float phi = rng.random(2 * M_PIf32);
    Rotation rot = Rotation::from_direction(n);
    Vec3f h = rot * Vec3f(std::cos(phi) * sin_beta, std::sin(phi) * sin_beta, cos_beta);
    DCHECK_GT(h.dot(n), -EPS);
    return h;
}

float TrowbridgeReitzModel::pdf_h(const Vec3f &l, const Vec3f &n, const Vec3f &h) const { return D_impl(n, h); }

float TrowbridgeReitzModel::D_impl(const Vec3f &n, const Vec3f &h) const {
    float cos_beta = n.dot(h);
    if (cos_beta <= 0) {
        return 0;
    }
    float cos2_beta = cos_beta * cos_beta;
    float sin2_beta = 1 - cos2_beta;
    return _alpha2 * M_1_PIf32 * cos_beta / square(_alpha2 * cos2_beta + sin2_beta);
}

Glossy::Glossy(float shininess, RGBTexture diffuse_texture, MicrofacetModelType micro_type)
    : _diffuse_texture(std::move(diffuse_texture)) {
    if (micro_type == MICRO_BLINN_PHONG) {
        _micro_model = std::make_unique<BlinnPhongModel>(shininess);
    } else {
        float alpha2 = 2 / (shininess + 2);
        _micro_model = std::make_unique<TrowbridgeReitzModel>(alpha2);
    }
}

Vec3f Glossy::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    Vec3f l = -ray.dir;
    const Vec3f &v = out_dir;
    const Vec3f &n = hit.shade_normal;
    if (!is_same_side(v, n)) {
        return Vec3f::Zero();
    }
    Vec3f h = (v + l).normalized();
    Vec3f color = _diffuse_texture.color_at(hit.uv);
    Vec3f decay = F(color, v, h) * (_micro_model->D(l, n, h) * _micro_model->G(l, n, h, v) / (4 * n.dot(l) * n.dot(v)));
    return decay.allFinite() ? decay : Vec3f::Zero();
}

Vec3f Glossy::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
    Vec3f micro_normal = _micro_model->sample_h(-ray.dir, hit.normal, rng);
    Vec3f out_dir = Ray::reflect(ray.dir, micro_normal);
    if (!is_same_side(out_dir, hit.normal)) {
        // falling back to lambertian
        return Lambertian::sample_impl(ray, hit, rng);
    }
    return out_dir;
}

float Glossy::pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    Vec3f l = -ray.dir;
    const Vec3f &v = out_dir;
    const Vec3f &n = hit.normal;
    if (!is_same_side(v, n)) {
        return 0;
    }
    Vec3f h = (v + l).normalized();
    return _micro_model->pdf_h(l, n, h) / (4 * n.dot(l) * n.dot(v));
}

// Fresnel effect
Vec3f Glossy::F(const Vec3f &f0, const Vec3f &v, const Vec3f &h) {
    auto pow5 = [](float x) { return (x * x) * (x * x) * x; };
    return f0 + (Vec3f::Ones() - f0) * pow5(1.f - v.dot(h));
}

Vec3f Metal::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const { return _diffuse_texture.color_at(hit.uv); }

Vec3f Metal::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
    Vec3f refl_dir = Ray::reflect(ray.dir, hit.normal);
    DCHECK(is_normalized(refl_dir));
    return refl_dir;
}

float Metal::pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    TINYPT_THROW_EX(std::logic_error) << "unreachable!";
}

Vec3f Dielectric::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    return _diffuse_texture.color_at(hit.uv);
}

Vec3f Dielectric::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
    Vec3f refl_dir = Ray::reflect(ray.dir, hit.normal);
    DCHECK_LT(ray.dir.dot(hit.normal), EPS);
    DCHECK(is_normalized(refl_dir));
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

AddBXDF::AddBXDF(std::shared_ptr<BXDF> bxdf1, std::shared_ptr<BXDF> bxdf2)
    : BXDF(bxdf1->is_specular() || bxdf2->is_specular()), _bxdf1(std::move(bxdf1)), _bxdf2(std::move(bxdf2)) {}

const BXDF *AddBXDF::sample_bxdf(RandEngine &rng) const {
    if (is_specular()) {
        return (rng.random() < .5f) ? _bxdf1.get() : _bxdf2.get();
    } else {
        return this;
    }
}

float AddBXDF::pdf_bxdf(const BXDF *bxdf) const { return is_specular() ? 0.5 : 1; }

Vec3f AddBXDF::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    DCHECK(!is_specular());
    return _bxdf1->f(ray, hit, out_dir) + _bxdf2->f(ray, hit, out_dir);
}

Vec3f AddBXDF::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
    DCHECK(!is_specular());
    return (rng.random() < .5f) ? _bxdf1->sample(ray, hit, rng) : _bxdf2->sample(ray, hit, rng);
}

float AddBXDF::pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    DCHECK(!is_specular());
    return .5f * (_bxdf1->pdf(ray, hit, out_dir) + _bxdf2->pdf(ray, hit, out_dir));
}

MixBXDF::MixBXDF(std::shared_ptr<BXDF> bxdf1, std::shared_ptr<BXDF> bxdf2, float weight)
    : BXDF(bxdf1->is_specular() || bxdf2->is_specular()), _bxdf1(std::move(bxdf1)), _bxdf2(std::move(bxdf2)),
      _weight(weight) {
    TINYPT_CHECK_EX(0 <= weight && weight <= 1, std::invalid_argument) << "[MixBXDF] invalid weight " << weight;
}

const BXDF *MixBXDF::sample_bxdf(RandEngine &rng) const {
    if (is_specular()) {
        return (rng.random() < _weight) ? _bxdf1.get() : _bxdf2.get();
    } else {
        return this;
    }
}

float MixBXDF::pdf_bxdf(const BXDF *bxdf) const {
    if (is_specular()) {
        if (bxdf == _bxdf1.get()) {
            return _weight;
        } else {
            DCHECK(bxdf == _bxdf2.get());
            return 1 - _weight;
        }
    } else {
        return 1;
    }
}

Vec3f MixBXDF::f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    DCHECK(!is_specular());
    return _weight * _bxdf1->f(ray, hit, out_dir) + (1 - _weight) * _bxdf2->f(ray, hit, out_dir);
}

Vec3f MixBXDF::sample(const Ray &ray, const Hit &hit, RandEngine &rng) const {
    DCHECK(!is_specular());
    return (rng.random() < _weight) ? _bxdf1->sample(ray, hit, rng) : _bxdf2->sample(ray, hit, rng);
}

float MixBXDF::pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const {
    DCHECK(!is_specular());
    return _weight * _bxdf1->pdf(ray, hit, out_dir) + (1 - _weight) * _bxdf2->pdf(ray, hit, out_dir);
}

} // namespace cpu
} // namespace tinypt