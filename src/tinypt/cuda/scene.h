#pragma once

#include "tinypt/cuda/light.h"
#include "tinypt/cuda/object.h"

#ifndef __NVCC__
#include "tinypt/cpu/scene.h"
#endif

namespace tinypt {
namespace cuda {

struct Image {
    Vec3f *_data;
    int _width;
    int _height;

    Image() : _data(nullptr), _width(0), _height(0) {}
    Image(Vec3f *data, int width, int height) : _data(data), _width(width), _height(height) {}

#ifndef __NVCC__
    static Image create(int width, int height) {
        Vec3f *data;
        CHECK_CUDA(cudaMalloc(&data, sizeof(Vec3f) * width * height));
        return Image(data, width, height);
    }

    static void destroy(Image &image) {
        CHECK_CUDA(cudaFree(image._data));
        image._data = nullptr;
        image._width = 0;
        image._height = 0;
    }

    cpu::Image to_cpu() const {
        cpu::Image cpu_image(_width, _height, 3);
        CHECK_CUDA(cudaMemcpy(cpu_image.data(), _data, sizeof(Vec3f) * _width * _height, cudaMemcpyDeviceToHost));
        return cpu_image;
    }
#endif
};

struct Camera {
    Camera() = default;

    Camera(const Vec3f &center, const Rotation &rot, int width, int height, float fov, float aperture, float focus_dist,
           const Vec3f &direct, const Vec3f &up, const Vec3f &right)
        : _center(center), _rot(rot), _width(width), _height(height), _fov(fov), _aperture(aperture),
          _focus_dist(focus_dist), _direct(direct), _up(up), _right(right) {}

#ifndef __NVCC__
    Camera(const cpu::Camera &cpu_camera)
        : Camera(cpu_camera._center, cpu_camera._rot, cpu_camera._width, cpu_camera._height, cpu_camera._fov,
                 cpu_camera._aperture, cpu_camera._focus_dist, cpu_camera._direct, cpu_camera._up, cpu_camera._right) {}
#endif
    __device__ Ray shoot_ray(float x, float y, RandEngine &rng) const {
        Vec3f offset;
        if (_aperture >= EPS) {
            Vec2f uv_offset = rng.rand_in_disk(_aperture);
            offset = uv_offset.x() * _rot.u() + uv_offset.y() * _rot.v();
        } else {
            offset = Vec3f::Zero();
        }
        Vec3f dir = (x / (float)_width - .5f) * _right + (y / (float)_height - .5f) * _up + _direct - offset;
        Ray ray(_center + offset, dir.normalized());
        return ray;
    }

    // location
    Vec3f _center;
    // rotation
    Rotation _rot;
    // image size
    int _width;
    int _height;
    // field of view
    float _fov;
    // lens params for Depth of Field (DOF)
    float _aperture;
    float _focus_dist;
    // view vector
    Vec3f _direct;
    Vec3f _up;
    Vec3f _right;
};

struct Scene {
    Scene(const Camera &camera, const Array<Sphere> &spheres, const Array<Circle> &circles,
          const Array<Rectangle> &rects, const Array<TriangleMesh> &meshes, const Array<Object *> object_lights,
          const EnvTexture &background)
        : _camera(camera), _spheres(spheres), _circles(circles), _rects(rects), _meshes(meshes),
          _background(background), _object_lights(object_lights) {}

#ifndef __NVCC__
    Scene(const cpu::Scene &cpu_scene) {
        size_t num_spheres = 0, sphere_idx = 0;
        size_t num_circles = 0, circle_idx = 0;
        size_t num_rects = 0, rect_idx = 0;
        size_t num_meshes = 0, mesh_idx = 0;
        for (const auto &obj : cpu_scene.object_group().objects()) {
            if (dynamic_cast<cpu::Sphere *>(obj.get())) {
                num_spheres++;
            } else if (dynamic_cast<cpu::Circle *>(obj.get())) {
                num_circles++;
            } else if (dynamic_cast<cpu::Rectangle *>(obj.get())) {
                num_rects++;
            } else if (dynamic_cast<cpu::TriangleMesh *>(obj.get())) {
                num_meshes++;
            } else {
                TINYPT_THROW << "Unknown object";
            }
        }
        auto spheres = Array<Sphere>::create(num_spheres);
        auto circles = Array<Circle>::create(num_circles);
        auto rects = Array<Rectangle>::create(num_rects);
        auto meshes = Array<TriangleMesh>::create(num_meshes);

        std::vector<Sphere> host_spheres;
        std::vector<Circle> host_circles;
        std::vector<Rectangle> host_rects;
        std::vector<TriangleMesh> host_meshes;

        std::vector<Object *> host_object_lights;
        host_object_lights.reserve(cpu_scene.object_lights().lights().size());

        for (const auto &obj : cpu_scene.object_group().objects()) {
            if (const auto cpu_sphere = dynamic_cast<cpu::Sphere *>(obj.get())) {
                host_spheres.emplace_back(Sphere::create(*cpu_sphere));
                if (!cpu_sphere->material().emission_texture().empty()) {
                    host_object_lights.emplace_back(&spheres[sphere_idx]);
                }
                sphere_idx++;
            } else if (const auto cpu_circle = dynamic_cast<cpu::Circle *>(obj.get())) {
                host_circles.emplace_back(Circle::create(*cpu_circle));
                if (!cpu_circle->material().emission_texture().empty()) {
                    host_object_lights.emplace_back(&circles[circle_idx]);
                }
                circle_idx++;
            } else if (const auto cpu_rect = dynamic_cast<cpu::Rectangle *>(obj.get())) {
                host_rects.emplace_back(Rectangle::create(*cpu_rect));
                if (!cpu_rect->material().emission_texture().empty()) {
                    host_object_lights.emplace_back(&rects[rect_idx]);
                }
                rect_idx++;
            } else if (const auto cpu_mesh = dynamic_cast<cpu::TriangleMesh *>(obj.get())) {
                auto mesh = TriangleMesh::create(*cpu_mesh, &meshes[mesh_idx]);
                host_meshes.emplace_back(mesh);
                for (uint32_t face_id = 0; face_id < cpu_mesh->_material_ids.size(); face_id++) {
                    uint32_t material_id = cpu_mesh->_material_ids[face_id];
                    if (!cpu_mesh->_materials[material_id].emission_texture().empty()) {
                        host_object_lights.emplace_back(&mesh._objects[face_id]);
                    }
                }
                mesh_idx++;
            } else {
                TINYPT_THROW << "Unknown object";
            }
        }
        spheres.from_cpu(host_spheres);
        circles.from_cpu(host_circles);
        rects.from_cpu(host_rects);
        meshes.from_cpu(host_meshes);
        auto object_lights = ObjectLightGroup::create(host_object_lights);
        auto delta_lights = DeltaLightGroup::create(cpu_scene.delta_lights());

        _camera = cpu_scene.camera();
        _spheres = spheres;
        _circles = circles;
        _rects = rects;
        _meshes = meshes;
        _object_lights = object_lights;
        _delta_lights = delta_lights;
        _background = EnvTexture::create(cpu_scene.background());
    }

    ~Scene() {
        for (auto &host_sphere : _spheres.to_cpu()) {
            Sphere::destroy(host_sphere);
        }
        Array<Sphere>::destroy(_spheres);
        for (auto &host_circle : _circles.to_cpu()) {
            Circle::destroy(host_circle);
        }
        Array<Circle>::destroy(_circles);
        for (auto &host_rect : _rects.to_cpu()) {
            Rectangle::destroy(host_rect);
        }
        Array<Rectangle>::destroy(_rects);
        for (auto &host_mesh : _meshes.to_cpu()) {
            TriangleMesh::destroy(host_mesh);
        }
        Array<TriangleMesh>::destroy(_meshes);

        ObjectLightGroup::destroy(_object_lights);
        DeltaLightGroup::destroy(_delta_lights);
        EnvTexture::destroy(_background);
    }
#endif

    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }
    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
        for (uint32_t i = 0; i < _spheres.size(); i++) {
            _spheres[i].intersect_impl<ENABLE_SURFACE>(ray, hit, rng);
        }
        for (uint32_t i = 0; i < _circles.size(); i++) {
            _circles[i].intersect_impl<ENABLE_SURFACE>(ray, hit, rng);
        }
        for (uint32_t i = 0; i < _rects.size(); i++) {
            _rects[i].intersect_impl<ENABLE_SURFACE>(ray, hit, rng);
        }
        for (uint32_t i = 0; i < _meshes.size(); i++) {
            _meshes[i].intersect_impl<ENABLE_SURFACE>(ray, hit, rng);
        }
    }
    __device__ const ObjectLightGroup &object_lights() const { return _object_lights; }
    __device__ const DeltaLightGroup &delta_lights() const { return _delta_lights; }
    __device__ const EnvTexture &background() const { return _background; }

    Camera _camera;

    Array<Sphere> _spheres;
    Array<Circle> _circles;
    Array<Rectangle> _rects;
    Array<TriangleMesh> _meshes;

    EnvTexture _background;

    // derived
    ObjectLightGroup _object_lights;
    DeltaLightGroup _delta_lights;
};

} // namespace cuda
} // namespace tinypt