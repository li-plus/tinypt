#include "tinypt/tinypt.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace tinypt {

namespace py = pybind11;

class PyObject3d : public Object {
  public:
    using Object::Object;

    Vec3f location() const override { PYBIND11_OVERRIDE_PURE(Vec3f, Object, location); }
    void set_location(const Vec3f &location) override { PYBIND11_OVERRIDE_PURE(void, Object, set_location, location); }
    Mat3f rotation() const override { PYBIND11_OVERRIDE_PURE(Mat3f, Object, rotation); }
    void set_rotation(const Mat3f &rotation) override { PYBIND11_OVERRIDE_PURE(void, Object, set_rotation, rotation); }
    bool has_material() const override { PYBIND11_OVERRIDE_PURE(bool, Object, has_material); }
    const Material &material() const override { PYBIND11_OVERRIDE_PURE(const Material &, Object, material); }
    void set_material(Material material) override { PYBIND11_OVERRIDE_PURE(void, Object, set_material, material); }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override {
        PYBIND11_OVERRIDE_PURE(void, Object, intersect, ray, hit);
    }
    void intersect_t(const Ray &ray, Hit &hit) const override {
        PYBIND11_OVERRIDE_PURE(void, Object, intersect_t, ray, hit);
    }
    Vec3f sample(const Vec3f &org, RandEngine &rng) const override {
        PYBIND11_OVERRIDE_PURE(Vec3f, Object, sample, org, rng);
    }
    float pdf(const Ray &ray) const override { PYBIND11_OVERRIDE_PURE(float, Object, pdf, ray); }
    AABB bounding_box() const override { PYBIND11_OVERRIDE_PURE(AABB, Object, bounding_box); }
    std::vector<Object *> children() const override { PYBIND11_OVERRIDE_PURE(std::vector<Object *>, Object, children); }
};

class PyBXDF : public BXDF {
  public:
    using BXDF::BXDF;

    const BXDF *sample_bxdf(RandEngine &rng) const override { PYBIND11_OVERRIDE(const BXDF *, BXDF, sample_bxdf, rng); }
    float pdf_bxdf(const BXDF *bxdf) const override { PYBIND11_OVERRIDE(float, BXDF, pdf_bxdf, bxdf); }
    Vec3f f(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override {
        PYBIND11_OVERRIDE_PURE(Vec3f, BXDF, f, ray, hit, out_dir);
    }
    Vec3f sample(const Ray &ray, const Hit &hit, RandEngine &rng) const override {
        PYBIND11_OVERRIDE_PURE(Vec3f, BXDF, sample, ray, hit, rng);
    }
    float pdf(const Ray &ray, const Hit &hit, const Vec3f &out_dir) const override {
        PYBIND11_OVERRIDE_PURE(float, BXDF, pdf, ray, hit, out_dir);
    }
};

class PyLight : public Light {
  public:
    using Light::Light;
    Vec3f sample(const Vec3f &pos, RandEngine &rng) const override {
        PYBIND11_OVERRIDE_PURE(Vec3f, Light, sample, pos, rng);
    }
    float pdf(const Vec3f &pos, const Vec3f &out_dir) const override {
        PYBIND11_OVERRIDE_PURE(float, Light, pdf, pos, out_dir);
    }
    Vec3f emission() const override { PYBIND11_OVERRIDE_PURE(Vec3f, Light, emission); }
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "tinypt python binding";

    // integrator
    py::class_<PathTracer>(m, "PathTracer")
        .def(py::init([](const std::string &device_str) { return std::make_unique<PathTracer>(device_str); }))
        .def("render", &PathTracer::render);

    // camera & scene
    py::class_<Image>(m, "Image", py::buffer_protocol())
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> array) {
            if (array.size() == 0) {
                return std::make_unique<Image>(cv::Mat());
            }
            TINYPT_CHECK_EX(array.ndim() == 3 && array.shape(2) == 3, std::invalid_argument)
                << "Invalid image, expect an RGB image";
            ssize_t height = array.shape(0);
            ssize_t width = array.shape(1);
            cv::Mat mat(height, width, CV_32FC3);
            std::copy(array.data(), array.data() + array.size(), (float *)mat.data);
            cv::pow(mat, SRGB_GAMMA, mat); // gamma
            cv::flip(mat, mat, 0);         // flip vertically
            return std::make_unique<Image>(std::move(mat));
        }))
        .def("numpy", [](const Image &self) {
            cv::Mat mat = self.mat().clone();
            cv::pow(mat, 1 / SRGB_GAMMA, mat); // gamma
            cv::flip(mat, mat, 0);             // flip vertically

            std::vector<ssize_t> shapes{mat.rows, mat.cols, 3};
            std::vector<ssize_t> strides{(ssize_t)(sizeof(Vec3f) * mat.cols), sizeof(Vec3f), sizeof(float)};
            return py::array_t<float>(
                py::buffer_info(mat.data, sizeof(float), py::format_descriptor<float>::format(), 3, shapes, strides));
        });
    py::class_<Camera>(m, "Camera")
        .def(py::init())
        .def_property("location", &Camera::location, &Camera::set_location)
        .def_property("rotation", &Camera::rotation, &Camera::set_rotation)
        .def_property("width", &Camera::width, &Camera::set_width)
        .def_property("height", &Camera::height, &Camera::set_height)
        .def_property("fov", &Camera::fov, &Camera::set_fov)
        .def("set_lens", &Camera::set_lens);
    py::class_<Scene>(m, "Scene")
        .def(py::init<Camera, std::vector<std::shared_ptr<Object>>, std::vector<std::shared_ptr<Light>>,
                      std::shared_ptr<EnvTexture>>())
        .def("to", [](const Scene &self, const std::string &device_str) { return self.to(device_str); });

    // object
    py::class_<Object, PyObject3d, std::shared_ptr<Object>>(m, "Object3d").def(py::init());
    py::class_<Sphere, Object, std::shared_ptr<Sphere>>(m, "Sphere")
        .def(py::init())
        .def_property("location", &Sphere::location, &Sphere::set_location)
        .def_property("radius", &Sphere::radius, &Sphere::set_radius)
        .def_property("material", &Sphere::material, &Sphere::set_material);
    py::class_<Rectangle, Object, std::shared_ptr<Rectangle>>(m, "Rectangle")
        .def(py::init())
        .def_property("location", &Rectangle::location, &Rectangle::set_location)
        .def_property("rotation", &Rectangle::rotation, &Rectangle::set_rotation)
        .def_property("dimension", &Rectangle::dimension, &Rectangle::set_dimension)
        .def_property("material", &Rectangle::material, &Rectangle::set_material);
    py::class_<Circle, Object, std::shared_ptr<Circle>>(m, "Circle")
        .def(py::init())
        .def_property("location", &Circle::location, &Circle::set_location)
        .def_property("rotation", &Circle::rotation, &Circle::set_rotation)
        .def_property("radius", &Circle::radius, &Circle::set_radius)
        .def_property("material", &Circle::material, &Circle::set_material);
    py::class_<TriangleMesh, Object, std::shared_ptr<TriangleMesh>>(m, "TriangleMesh")
        .def(py::init())
        .def("from_obj", &TriangleMesh::from_obj);

    // light
    py::class_<Light, PyLight, std::shared_ptr<Light>>(m, "Light").def(py::init());
    py::class_<ObjectLight, Light, std::shared_ptr<ObjectLight>>(m, "ObjectLight").def(py::init<const Object *>());
    py::class_<DistantLight, Light, std::shared_ptr<DistantLight>>(m, "DistantLight")
        .def(py::init())
        .def_property("rotation", &DistantLight::rotation, &DistantLight::set_rotation)
        .def_property("color", &DistantLight::color, &DistantLight::set_color)
        .def_property("power", &DistantLight::power, &DistantLight::set_power)
        .def_property("angle", &DistantLight::angle, &DistantLight::set_angle);
    py::class_<PointLight, Light, std::shared_ptr<PointLight>>(m, "PointLight")
        .def(py::init())
        .def_property("location", &PointLight::location, &PointLight::set_location)
        .def_property("color", &PointLight::color, &PointLight::set_color)
        .def_property("power", &PointLight::power, &PointLight::set_power)
        .def_property("radius", &PointLight::radius, &PointLight::set_radius);
    py::class_<EnvLight, Light, std::shared_ptr<EnvLight>>(m, "EnvLight").def(py::init<const EnvTexture *>());

    // bxdf
    py::class_<BXDF, PyBXDF, std::shared_ptr<BXDF>>(m, "BXDF").def(py::init());
    py::class_<Lambertian, BXDF, std::shared_ptr<Lambertian>>(m, "Lambertian").def(py::init<RGBTexture>());
    py::class_<Metal, BXDF, std::shared_ptr<Metal>>(m, "Metal").def(py::init<RGBTexture>());
    py::class_<Dielectric, BXDF, std::shared_ptr<Dielectric>>(m, "Dielectric").def(py::init<RGBTexture, float>());

    // material
    py::class_<Material, std::shared_ptr<Material>>(m, "Material")
        .def(py::init<std::shared_ptr<BXDF>>())
        // TODO surface property
        .def_property("emission_texture", &Material::emission_texture, &Material::set_emission_texture)
        .def_property("alpha_texture", &Material::alpha_texture, &Material::set_alpha_texture)
        .def_property("bump_texture", &Material::bump_texture, &Material::set_bump_texture);

    // texture
    py::class_<RGBTexture, std::shared_ptr<RGBTexture>>(m, "RGBTexture")
        .def(py::init())
        .def_property("value", &RGBTexture::value, &RGBTexture::set_value)
        .def_property("map", &RGBTexture::map, &RGBTexture::set_map);
    py::class_<AlphaTexture, std::shared_ptr<AlphaTexture>>(m, "AlphaTexture")
        .def(py::init())
        .def_property("value", &AlphaTexture::value, &AlphaTexture::set_value)
        .def_property("map", &AlphaTexture::map, &AlphaTexture::set_map);
    py::class_<BumpTexture, std::shared_ptr<BumpTexture>>(m, "BumpTexture").def(py::init());
    py::class_<EnvTexture, std::shared_ptr<EnvTexture>>(m, "EnvTexture")
        .def(py::init<RGBTexture>())
        .def_property("rotation", &EnvTexture::rotation, &EnvTexture::set_rotation);
}

} // namespace tinypt
