#pragma once

#include "tinypt/cuda/material.h"

#ifndef __NVCC__
#include "tinypt/cpu/object.h"
#endif

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <queue>

namespace tinypt {
namespace cuda {

__device__ static inline float plane_pdf_value(float dist, float area, const Vec3f &normal, const Vec3f &ray_dir) {
    float dist_squared = square(dist);
    float cos_alpha = std::abs(normal.dot(ray_dir));
    float pdf_val = dist_squared / (cos_alpha * area); // inf!
    return (pdf_val < INF) ? pdf_val : INF;
}

struct Object {
    enum Type { TYPE_SPHERE, TYPE_RECTANGLE, TYPE_CIRCLE, TYPE_TRIANGLE_MESH, TYPE_TRIANGLE_MESH_FACE };
    Type _type;

    Object(Type type) : _type(type) {}
    __device__ Vec3f sample(const Vec3f &org, RandEngine &rng) const;
    __device__ float pdf(const Ray &ray) const;
};

struct Sphere : public Object {
#ifndef __NVCC__
    Sphere() : Object(TYPE_SPHERE) {}
    Sphere(const Vec3f &center, float radius, Material material)
        : Object(TYPE_SPHERE), _center(center), _radius(radius), _material(material) {}

    static Sphere create(const cpu::Sphere &cpu_sphere) {
        return Sphere(cpu_sphere._center, cpu_sphere._radius, Material::create(cpu_sphere._material));
    }
    static void destroy(Sphere &sphere) { Material::destroy(sphere._material); }
#endif

    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
        Vec3f op = _center - ray.org;
        float b = op.dot(ray.dir);
        float det = square(b) - op.dot(op) + square(_radius);
        if (det < 0) {
            return;
        }
        det = std::sqrt(det);
        float t;
        bool into;
        if (b - det >= EPS) {
            t = b - det;
            into = true;
        } else if (b + det >= EPS) {
            t = b + det;
            into = false;
        } else {
            t = INF;
            into = true;
        }
        if (t < hit.t) {
            if (ENABLE_SURFACE) {
                Vec3f pos = ray.point_at(t);
                Vec3f normal = (pos - _center).normalized();
                // TODO: merge codes
                float u = (std::atan2(normal.y(), normal.x()) + M_PIf32) * (.5f * M_1_PIf32);
                float v = std::acos(normal.z()) * M_1_PIf32;
                Vec2f uv(u, v);
                if (_material.alpha_texture().alpha_at(uv) < rng->random()) {
                    return;
                }
                if (!into) {
                    normal = -normal;
                }
                hit = Hit(t, normal, normal, into, uv, &_material);
            } else {
                hit.t = t;
            }
        }
    }
    __device__ Vec3f sample(const Vec3f &org, RandEngine &rng) const {
        float cos_theta_max = get_cos_theta_max(org);
        float cos_theta = rng.random(cos_theta_max, 1);
        float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
        float phi = rng.random(2.f * M_PIf32);
        Vec3f local_dir(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
        Vec3f out_dir = Rotation::from_direction((_center - org).normalized()) * local_dir;
        return out_dir;
    }
    __device__ float pdf(const Ray &ray) const {
        Hit hit;
        intersect_impl<false>(ray, hit, nullptr);
        if (!hit.is_hit()) {
            return 0;
        }
        float cos_theta_max = get_cos_theta_max(ray.org);
        float solid_angle = 2 * M_PIf32 * (1 - cos_theta_max);
        float pdf_val = 1 / solid_angle;
        return pdf_val;
    }
    __device__ float get_cos_theta_max(const Vec3f &org) const {
        float dist2 = (_center - org).squaredNorm();
        float radius2 = square(_radius);
        if (dist2 <= radius2) {
            // ray origin inside the sphere, sample the entire surface
            return -1;
        }
        float cos_theta_max = std::sqrt(1 - radius2 / dist2);
        return cos_theta_max;
    }

    Vec3f _center;
    float _radius;
    Material _material;
};

struct Rectangle : public Object {
#ifndef __NVCC__
    Rectangle() : Object(TYPE_RECTANGLE) {}
    Rectangle(const Vec3f &center, const Rotation &rotation, const Vec2f &dimension, const Material &material,
              float dist)
        : Object(TYPE_RECTANGLE), _center(center), _rotation(rotation), _dimension(dimension), _material(material),
          _dist(dist) {}

    static Rectangle create(const cpu::Rectangle &cpu_rect) {
        return Rectangle(cpu_rect._center, cpu_rect._rotation, cpu_rect._dimension,
                         Material::create(cpu_rect._material), cpu_rect._dist);
    }
    static void destroy(Rectangle &rect) { Material::destroy(rect._material); }
#endif
    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }

    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl(ray, hit, nullptr); }

    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
        float t = (_dist - ray.org.dot(normal())) / ray.dir.dot(normal()); // !inf
        if (EPS <= t && t < hit.t) {
            Vec3f cp = ray.point_at(t) - _center; // center -> hit_pos
            float u = cp.dot(_rotation.u()) / _dimension.x() + 0.5f;
            float v = cp.dot(_rotation.v()) / _dimension.y() + 0.5f;
            if (0 <= u && u <= 1 && 0 <= v && v <= 1) {
                if (ENABLE_SURFACE) {
                    Vec3f hit_normal = is_same_side(normal(), ray.dir) ? -normal() : normal();
                    hit = Hit(t, hit_normal, hit_normal, true, {u, v}, &_material);
                } else {
                    hit.t = t;
                }
            }
        }
    }

    __device__ Vec3f sample(const Vec3f &org, RandEngine &rng) const {
        Vec3f u_vec = (rng.random() - .5f) * _dimension.x() * _rotation.u();
        Vec3f v_vec = (rng.random() - .5f) * _dimension.y() * _rotation.v();
        Vec3f target = _center + u_vec + v_vec;
        return (target - org).normalized();
    }

    __device__ float pdf(const Ray &ray) const {
        Hit hit;
        intersect_t(ray, hit);
        if (!hit.is_hit()) {
            return 0;
        }
        float area = _dimension.x() * _dimension.y();
        return plane_pdf_value(hit.t, area, normal(), ray.dir);
    }

  private:
    __device__ const Vec3f &normal() const { return _rotation.w(); }

  private:
    Vec3f _center;
    Rotation _rotation;
    Vec2f _dimension;
    Material _material;
    // derived
    float _dist;
};

struct Circle : public Object {
    Circle() : Object(TYPE_CIRCLE) {}
    Circle(float radius, const Vec3f &center, const Rotation &rotation, const Material &material, float dist)
        : Object(TYPE_CIRCLE), _radius(radius), _center(center), _rot(rotation), _material(material), _dist(dist) {}

#ifndef __NVCC__
    static Circle create(const cpu::Circle &cpu_circle) {
        return Circle(cpu_circle._radius, cpu_circle._center, cpu_circle._rot, Material::create(cpu_circle._material),
                      cpu_circle._dist);
    }
    static void destroy(Circle &circle) { Material::destroy(circle._material); }
#endif

    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
        float t = (_dist - ray.org.dot(normal())) / ray.dir.dot(normal()); // !inf
        if (EPS <= t && t < hit.t) {
            Vec3f cp = ray.point_at(t) - _center; // center -> hit_pos
            if (cp.squaredNorm() <= square(_radius)) {
                if (ENABLE_SURFACE) {
                    Vec3f hit_normal = is_same_side(normal(), ray.dir) ? -normal() : normal();
                    cp /= _radius;
                    Vec2f uv(cp.dot(_rot.u()), cp.dot(_rot.v()));
                    hit = Hit(t, hit_normal, hit_normal, true, uv, &_material);
                } else {
                    hit.t = t;
                }
            }
        }
    }
    __device__ Vec3f sample(const Vec3f &org, RandEngine &rng) const {
        Vec2f uv = rng.rand_in_disk(_radius);
        Vec3f target = _center + _rot.u() * uv.x() + _rot.v() * uv.y();
        return (target - org).normalized();
    }

    __device__ float pdf(const Ray &ray) const {
        Hit hit;
        intersect_impl<false>(ray, hit, nullptr);
        if (!hit.is_hit()) {
            return 0;
        }
        float area = M_PIf32 * square(_radius);
        return plane_pdf_value(hit.t, area, normal(), ray.dir);
    }

  private:
    __device__ const Vec3f &normal() const { return _rot.w(); }

  private:
    float _radius;
    Vec3f _center;
    Rotation _rot;
    Material _material;
    // derived
    float _dist;
};

struct AABB {
    __device__ AABB() : AABB(Vec3f::Zero(), Vec3f::Zero()) {}
    __device__ AABB(const Vec3f &minv, const Vec3f &maxv) : _minv(minv), _maxv(maxv) {}

#ifndef __NVCC__
    AABB(const cpu::AABB &cpu_aabb) : AABB(cpu_aabb.min(), cpu_aabb.max()) {}
#endif

    __device__ float intersect(const Ray &ray) const {
        float tmin = -INF;
        float tmax = INF;
        for (int a = 0; a < 3; a++) {
            float inv_dir = 1.f / ray.dir[a]; // inf!
            float t0 = (_minv[a] - ray.org[a]) * inv_dir;
            float t1 = (_maxv[a] - ray.org[a]) * inv_dir;
            if (inv_dir < 0.f) {
                dev_swap(t0, t1);
            }
            // don't use std::min or std::max since t0,t1 might be nan
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
        }
        if (tmax < tmin || tmax < EPS) {
            return INF;
        }
        return tmin;
    }

  private:
    Vec3f _minv;
    Vec3f _maxv;
};

struct TriangleMesh;

struct TriangleMeshFace : public Object {
    TriangleMeshFace(const Vec3f &a, const Vec3f &ab, const Vec3f &ac, int face_id, const TriangleMesh *mesh)
        : Object(TYPE_TRIANGLE_MESH_FACE), _a(a), _ab(ab), _ac(ac), _face_id(face_id), _mesh(mesh) {}

#ifndef __NVCC__
    TriangleMeshFace(const cpu::TriangleMeshFace &cpu_face, const TriangleMesh *mesh)
        : TriangleMeshFace(cpu_face._a, cpu_face._ab, cpu_face._ac, cpu_face._face_id, mesh) {}
#endif
    __device__ const Material &material() const;

    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }
    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;
    __device__ Vec3f sample(const Vec3f &org, RandEngine &rng) const {
        float sqrt_r1 = std::sqrt(rng.random());
        float r2 = rng.random();
        float u = sqrt_r1 * (1 - r2);
        float v = r2 * sqrt_r1;
        Vec3f target = _a + u * _ab + v * _ac;
        return (target - org).normalized();
    }
    __device__ float pdf(const Ray &ray) const {
        Hit hit;
        intersect_impl<false>(ray, hit, nullptr);
        if (!hit.is_hit()) {
            return 0;
        }
        Vec3f face_normal = _ac.cross(_ab);
        float area = .5f * face_normal.norm();
        face_normal.normalize();
        return plane_pdf_value(hit.t, area, face_normal, ray.dir);
    }

  private:
    Vec3f _a;
    Vec3f _ab;
    Vec3f _ac;
    int _face_id;
    const TriangleMesh *_mesh;
};

struct BVHTree {
    static constexpr int ROOT_ID = 1;

    BVHTree() : _half_size(0) {}

    BVHTree(const Array<AABB> &aabbs, const Array<int> &indices, const Array<const TriangleMeshFace *> &objects,
            int half_size)
        : _aabbs(aabbs), _indices(indices), _objects(objects), _half_size(half_size) {}

#ifndef __NVCC__
    static BVHTree create(const cpu::BVHTree &cpu_bvh, const Array<TriangleMeshFace> &dev_objects) {
        std::vector<AABB> vec_aabbs(1);
        std::vector<int> vec_indices;
        std::vector<const TriangleMeshFace *> vec_objects;

        std::queue<const cpu::BVHNode *> q;
        q.push(cpu_bvh.root());
        int start = 0;
        uint32_t half_size = 1;
        while (!q.empty()) {
            const auto cpu_node = q.front();
            q.pop();
            vec_aabbs.emplace_back(cpu_node->bbox);
            if (!cpu_node->is_leaf()) {
                q.push(cpu_node->left.get());
                q.push(cpu_node->right.get());
                half_size++;
            } else {
                vec_indices.push_back(start);
                start += cpu_node->objects.size();
                for (const auto *cpu_obj : cpu_node->objects) {
                    auto cpu_triangle = dynamic_cast<const cpu::TriangleMeshFace *>(cpu_obj);
                    TINYPT_CHECK(cpu_triangle != nullptr);
                    vec_objects.emplace_back(dev_objects.data() + cpu_triangle->face_id());
                }
            }
        }
        vec_indices.push_back(start);
        TINYPT_CHECK((half_size & (half_size - 1)) == 0); // half_size must be power of 2
        TINYPT_CHECK(vec_indices.size() == half_size + 1);
        TINYPT_CHECK(vec_aabbs.size() == 2 * half_size);

        auto aabbs = Array<AABB>::create(vec_aabbs);
        auto indices = Array<int>::create(vec_indices);
        auto objects = Array<const TriangleMeshFace *>::create(vec_objects);
        return BVHTree(aabbs, indices, objects, half_size);
    }

    static void destroy(BVHTree &bvh_tree) {
        Array<AABB>::destroy(bvh_tree._aabbs);
        Array<int>::destroy(bvh_tree._indices);
        Array<const TriangleMeshFace *>::destroy(bvh_tree._objects);
    }
#endif

    __device__ static int left(int id) { return id * 2; }
    __device__ static int right(int id) { return id * 2 + 1; }
    __device__ static int parent(int id) { return id / 2; }

    __device__ bool is_leaf(int id) const { return id >= _half_size; }
    __device__ int to_leaf(int id) const { return id - _half_size; }

    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }
    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
        struct {
            int id;
            float tmin;
        } stk[20];
        int top = 0;

        int curr_id = BVHTree::ROOT_ID;
        float curr_tmin = _aabbs[curr_id].intersect(ray);

        stk[top++] = {curr_id, curr_tmin};
        while (top > 0) {
            top--;
            curr_id = stk[top].id;
            curr_tmin = stk[top].tmin;
            if (hit.t <= curr_tmin) {
                continue;
            }

            while (!is_leaf(curr_id)) {
                int left_id = BVHTree::left(curr_id);
                int right_id = BVHTree::right(curr_id);
                float left_tmin = _aabbs[left_id].intersect(ray);
                float right_tmin = _aabbs[right_id].intersect(ray);
                if (left_tmin > right_tmin) {
                    dev_swap(left_id, right_id);
                    dev_swap(left_tmin, right_tmin);
                }
                if (right_tmin < hit.t) {
                    curr_id = left_id;
                    stk[top++] = {right_id, right_tmin};
                } else if (left_tmin < hit.t) {
                    curr_id = left_id;
                } else {
                    curr_id = 0;
                    break;
                }
            }

            if (curr_id > 0) {
                int leaf_id = to_leaf(curr_id);
                for (int i = _indices[leaf_id]; i < _indices[leaf_id + 1]; i++) {
                    const auto *obj = _objects[i];
                    obj->intersect_impl<ENABLE_SURFACE>(ray, hit, rng);
                }
            }
        }
    }

    Array<AABB> _aabbs;
    Array<int> _indices;
    Array<const TriangleMeshFace *> _objects;
    int _half_size;
};

struct VertexInfo {
    int vertex_index;
    int normal_index;
    int texcoord_index;

    VertexInfo() = default;
    VertexInfo(int vertex_index_, int normal_index_, int texcoord_index_)
        : vertex_index(vertex_index_), normal_index(normal_index_), texcoord_index(texcoord_index_) {}
};

struct TriangleMesh : public Object {
    Array<Vec3f> _vertices;
    Array<Vec3f> _vertex_normals;
    Array<Vec2f> _texture_coords;
    Array<VertexInfo[3]> _faces_index;
    Array<uint32_t> _material_ids;
    Array<Material> _materials;

    Array<TriangleMeshFace> _objects;
    BVHTree _bvh_tree;

    TriangleMesh() : Object(TYPE_TRIANGLE_MESH) {}
    TriangleMesh(const Array<Vec3f> &vertices, const Array<Vec3f> &vertex_normals, const Array<Vec2f> &texture_coords,
                 const Array<VertexInfo[3]> &faces_index, const Array<uint32_t> &material_ids,
                 const Array<Material> &materials, const Array<TriangleMeshFace> &objects, const BVHTree &bvh_tree)
        : Object(TYPE_TRIANGLE_MESH), _vertices(vertices), _vertex_normals(vertex_normals),
          _texture_coords(texture_coords), _faces_index(faces_index), _material_ids(material_ids),
          _materials(materials), _objects(objects), _bvh_tree(bvh_tree) {}

#ifndef __NVCC__
    static TriangleMesh create(const cpu::TriangleMesh &cpu_mesh, const TriangleMesh *dev_mesh) {
        Array<Vec3f> vertices;
        {
            std::vector<Vec3f> host_vertices(cpu_mesh._vertices.size());
            std::copy(cpu_mesh._vertices.begin(), cpu_mesh._vertices.end(), host_vertices.begin());
            vertices = Array<Vec3f>::create(host_vertices);
        }
        Array<Vec3f> vertex_normals;
        {
            std::vector<Vec3f> host_vertex_normals(cpu_mesh._vertex_normals.size());
            std::copy(cpu_mesh._vertex_normals.begin(), cpu_mesh._vertex_normals.end(), host_vertex_normals.begin());
            vertex_normals = Array<Vec3f>::create(host_vertex_normals);
        }
        Array<Vec2f> texture_coords;
        {
            std::vector<Vec2f> host_texture_coords(cpu_mesh._texture_coords.size());
            std::copy(cpu_mesh._texture_coords.begin(), cpu_mesh._texture_coords.end(), host_texture_coords.begin());
            texture_coords = Array<Vec2f>::create(host_texture_coords);
        }
        Array<VertexInfo[3]> faces_index;
        {
            std::vector<VertexInfo[3]> host_faces_index(cpu_mesh._faces_index.size());
            std::memcpy(host_faces_index.data(), cpu_mesh._faces_index.data(),
                        sizeof(VertexInfo[3]) * host_faces_index.size());
            faces_index = Array<VertexInfo[3]>::create(host_faces_index);
        }
        Array<uint32_t> material_ids = Array<uint32_t>::create(cpu_mesh._material_ids);
        Array<Material> materials;
        {
            std::vector<Material> host_materials;
            host_materials.reserve(cpu_mesh._materials.size());
            for (const auto &cpu_material : cpu_mesh._materials) {
                host_materials.emplace_back(Material::create(cpu_material));
            }
            materials = Array<Material>::create(host_materials);
        }
        Array<TriangleMeshFace> objects;
        BVHTree bvh_tree;
        {
            std::vector<std::shared_ptr<cpu::Object>> cpu_objects;
            for (const auto &cpu_obj : cpu_mesh._obj.objects()) {
                auto cpu_group = std::dynamic_pointer_cast<cpu::ObjectGroup>(cpu_obj);
                TINYPT_CHECK(cpu_group != nullptr);
                cpu_objects.insert(cpu_objects.end(), cpu_group->objects().begin(), cpu_group->objects().end());
            }
            // build all objects (triangles)
            std::vector<TriangleMeshFace> host_objects;
            host_objects.reserve(cpu_objects.size());
            for (const auto &cpu_obj : cpu_objects) {
                auto cpu_triangle = std::dynamic_pointer_cast<cpu::TriangleMeshFace>(cpu_obj);
                TINYPT_CHECK(cpu_triangle != nullptr);
                host_objects.emplace_back(*cpu_triangle, dev_mesh);
            }
            objects = Array<TriangleMeshFace>::create(host_objects);

            // rebuild 1-layer BVH tree
            cpu::ObjectGroup cpu_group(std::move(cpu_objects), 4);
            bvh_tree = BVHTree::create(cpu_group.bvh_tree(), objects);
        }

        return TriangleMesh(vertices, vertex_normals, texture_coords, faces_index, material_ids, materials, objects,
                            bvh_tree);
    }

    static void destroy(TriangleMesh &mesh) {
        Array<Vec3f>::destroy(mesh._vertices);
        Array<Vec3f>::destroy(mesh._vertex_normals);
        Array<Vec2f>::destroy(mesh._texture_coords);
        Array<VertexInfo[3]>::destroy(mesh._faces_index);
        Array<uint32_t>::destroy(mesh._material_ids);

        auto host_materials = mesh._materials.to_cpu();
        for (auto &host_material : host_materials) {
            Material::destroy(host_material);
        }
        Array<Material>::destroy(mesh._materials);

        Array<TriangleMeshFace>::destroy(mesh._objects);
        BVHTree::destroy(mesh._bvh_tree);
    }
#endif

    __device__ void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
    __device__ void intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }
    template <bool ENABLE_SURFACE = true>
    __device__ void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
        _bvh_tree.intersect_impl<ENABLE_SURFACE>(ray, hit, rng);
    }
};

__device__ inline const Material &TriangleMeshFace::material() const {
    int material_id = _mesh->_material_ids[_face_id];
    return _mesh->_materials[material_id];
}

template <bool ENABLE_SURFACE>
__device__ inline void TriangleMeshFace::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    float t, u, v;
    {
        Vec3f p = ray.dir.cross(_ac);
        float inv_det = 1.f / p.dot(_ab); // inf!
        Vec3f ao = ray.org - _a;
        u = p.dot(ao) * inv_det;
        Vec3f q = ao.cross(_ab);
        v = q.dot(ray.dir) * inv_det;
        if (!(0 <= u && 0 <= v && u + v <= 1)) {
            return;
        }
        t = q.dot(_ac) * inv_det;
    }

    if (EPS <= t && t < hit.t) {
        if (ENABLE_SURFACE) {
            const auto &face_index = _mesh->_faces_index[_face_id];
            const auto &idx_a = face_index[0];
            const auto &idx_b = face_index[1];
            const auto &idx_c = face_index[2];

            Vec2f uv;
            if (idx_a.texcoord_index >= 0) {
                // vertex texture coordinates
                const Vec2f &uv_a = _mesh->_texture_coords[idx_a.texcoord_index];
                const Vec2f &uv_b = _mesh->_texture_coords[idx_b.texcoord_index];
                const Vec2f &uv_c = _mesh->_texture_coords[idx_c.texcoord_index];
                uv = (1 - u - v) * uv_a + u * uv_b + v * uv_c;
            } else {
                uv = {u, v};
            }
            auto mat = &material();
            // transparent mask
            if (mat->alpha_texture().alpha_at(uv) < rng->random()) {
                return;
            }

            Vec3f normal;
            if (idx_a.normal_index >= 0) {
                // vertex normal
                const Vec3f &vn_a = _mesh->_vertex_normals[idx_a.normal_index];
                const Vec3f &vn_b = _mesh->_vertex_normals[idx_b.normal_index];
                const Vec3f &vn_c = _mesh->_vertex_normals[idx_c.normal_index];
                normal = ((1 - u - v) * vn_a + u * vn_b + v * vn_c).normalized();
            } else {
                // face normal
                normal = _ac.cross(_ab).normalized();
            }

            bool into = !is_same_side(normal, ray.dir);
            if (!into) {
                normal = -normal;
            }
            Vec3f shade_normal = normal;

            // TODO
            //            const BumpTexture &bump_tex = mat->bump_texture();
            //            if (!bump_tex.empty()) {
            //                // compute dpdu, dpdv
            //                Vec2f duv_ab, duv_ac;
            //                if (idx_a.texcoord_index >= 0) {
            //                    // vertex texture coordinates
            //                    const Vec2f &uv_a = _mesh->_texture_coords[idx_a.texcoord_index];
            //                    const Vec2f &uv_b = _mesh->_texture_coords[idx_b.texcoord_index];
            //                    const Vec2f &uv_c = _mesh->_texture_coords[idx_c.texcoord_index];
            //                    duv_ab = uv_b - uv_a;
            //                    duv_ac = uv_c - uv_a;
            //                } else {
            //                    duv_ab = {1, 0};
            //                    duv_ac = {0, 1};
            //                }
            //                float du_ab = duv_ab.x(), dv_ab = duv_ab.y();
            //                float du_ac = duv_ac.x(), dv_ac = duv_ac.y();
            //                float inv_det = 1.f / (du_ab * dv_ac - du_ac * dv_ab); // !inf
            //                if (std::isfinite(inv_det)) {
            //                    Vec3f dpdu = (dv_ac * _ab - dv_ab * _ac) * inv_det;
            //                    Vec3f dpdv = (-du_ac * _ab + du_ab * _ac) * inv_det;
            //                    shade_normal = bump_tex.bump_normal(uv, shade_normal, dpdu, dpdv);
            //                }
            //            }
            hit = Hit(t, normal, shade_normal, into, uv, mat);
        } else {
            hit.t = t;
        }
    }
}

__device__ inline Vec3f Object::sample(const Vec3f &org, RandEngine &rng) const {
    switch (_type) {
    case TYPE_SPHERE:
        return ((Sphere *)this)->sample(org, rng);
    case TYPE_RECTANGLE:
        return ((Rectangle *)this)->sample(org, rng);
    case TYPE_CIRCLE:
        return ((Circle *)this)->sample(org, rng);
    case TYPE_TRIANGLE_MESH_FACE:
        return ((TriangleMeshFace *)this)->sample(org, rng);
    case TYPE_TRIANGLE_MESH:
    default:
        printf("unreachable!\n");
        return Vec3f::Zero();
    }
}
__device__ inline float Object::pdf(const Ray &ray) const {
    switch (_type) {
    case TYPE_SPHERE:
        return ((Sphere *)this)->pdf(ray);
    case TYPE_RECTANGLE:
        return ((Rectangle *)this)->pdf(ray);
    case TYPE_CIRCLE:
        return ((Circle *)this)->pdf(ray);
    case TYPE_TRIANGLE_MESH_FACE:
        return ((TriangleMeshFace *)this)->pdf(ray);
    case TYPE_TRIANGLE_MESH:
    default:
        printf("unreachable!\n");
        return -1;
    }
}

} // namespace cuda
} // namespace tinypt