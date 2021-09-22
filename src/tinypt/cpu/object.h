#pragma once

#include "tinypt/cpu/bvh.h"
#include "tinypt/cpu/material.h"
#include "tinypt/cpu/ray.h"

#include <iostream>

namespace tinypt {

namespace cuda {
struct Sphere;
struct Rectangle;
struct Circle;
struct TriangleMesh;
struct TriangleMeshFace;
struct Scene;
} // namespace cuda

namespace cpu {

class Object {
  public:
    virtual ~Object() = default;

    virtual Vec3f location() const = 0;
    virtual void set_location(const Vec3f &location) = 0;
    virtual Mat3f rotation() const = 0;
    virtual void set_rotation(const Mat3f &rotation) = 0;
    virtual bool has_material() const = 0;
    virtual const Material &material() const = 0;
    virtual void set_material(Material material) = 0;

    virtual void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const = 0;
    virtual void intersect_t(const Ray &ray, Hit &hit) const = 0;
    virtual Vec3f sample(const Vec3f &org, RandEngine &rng) const = 0;
    virtual float pdf(const Ray &ray) const = 0;
    virtual AABB bounding_box() const = 0;
    virtual std::vector<Object *> children() const = 0;
};

class Sphere : public Object {
    friend tinypt::cuda::Sphere;

  public:
    Sphere() : _center(Vec3f::Zero()), _radius(1) {}

    Vec3f location() const override { return _center; }
    void set_location(const Vec3f &location) override { _center = location; }
    Mat3f rotation() const override { return Mat3f::Identity(); }
    void set_rotation(const Mat3f &rotation) override { LOG(WARNING) << "No need to set rotation for Sphere"; }
    float radius() const { return _radius; }
    void set_radius(float radius) { _radius = radius; }
    bool has_material() const override { return true; }
    const Material &material() const override { return _material; }
    void set_material(Material material) override { _material = std::move(material); }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override;
    void intersect_t(const Ray &ray, Hit &hit) const override;

    Vec3f sample(const Vec3f &org, RandEngine &rng) const override;
    float pdf(const Ray &ray) const override;
    AABB bounding_box() const override;
    std::vector<Object *> children() const override { return {}; }

  private:
    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

    float get_cos_theta_max(const Vec3f &org) const;

  private:
    Vec3f _center;
    float _radius;
    Material _material;
};

class Rectangle : public Object {
    friend tinypt::cuda::Rectangle;

  public:
    Rectangle() : Rectangle(Vec3f::Zero(), Mat3f::Identity(), Vec2f::Ones(), {}) {}

    Vec3f location() const override { return _center; }
    void set_location(const Vec3f &location) override {
        _center = location;
        _dist = calc_dist(_rotation, _center);
    }
    Mat3f rotation() const override { return _rotation.matrix(); }
    void set_rotation(const Mat3f &rotation) override {
        _rotation = rotation;
        _dist = calc_dist(_rotation, _center);
    }
    Vec2f dimension() const { return _dimension; }
    void set_dimension(const Vec2f &dimension) { _dimension = dimension; }
    bool has_material() const override { return true; }
    const Material &material() const override { return _material; }
    void set_material(Material material) override { _material = std::move(material); }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override;
    void intersect_t(const Ray &ray, Hit &hit) const override;

    Vec3f sample(const Vec3f &org, RandEngine &rng) const override;
    float pdf(const Ray &ray) const override;
    AABB bounding_box() const override;
    std::vector<Object *> children() const override { return {}; }

  private:
    Rectangle(const Vec3f &location, const Mat3f &rotation, const Vec2f &dimension, Material material);

    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

    const Vec3f &normal() const { return _rotation.w(); }

    static float calc_dist(const Rotation &rotation, const Vec3f &center) { return rotation.w().dot(center); }

  private:
    Vec3f _center;
    Rotation _rotation;
    Vec2f _dimension;
    Material _material;
    // derived
    float _dist;
};

class Circle : public Object {
    friend tinypt::cuda::Circle;

  public:
    Circle() : Circle(1, Vec3f::Zero(), Mat3f::Identity(), {}) {}

    Vec3f location() const override { return _center; }
    void set_location(const Vec3f &loc) override { *this = Circle(radius(), loc, rotation(), material()); }
    Mat3f rotation() const override { return _rot.matrix(); }
    void set_rotation(const Mat3f &rot) override { *this = Circle(radius(), location(), rot, material()); }
    float radius() const { return _radius; }
    void set_radius(float radius) { *this = Circle(radius, location(), rotation(), material()); }
    bool has_material() const override { return true; }
    const Material &material() const override { return _material; }
    void set_material(Material material) override { _material = std::move(material); }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override;
    void intersect_t(const Ray &ray, Hit &hit) const override;

    Vec3f sample(const Vec3f &org, RandEngine &rng) const override;
    float pdf(const Ray &ray) const override;
    AABB bounding_box() const override;
    std::vector<Object *> children() const override { return {}; }

  private:
    Circle(float radius, const Vec3f &location, const Mat3f &rotation, Material material);

    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

    const Vec3f &normal() const { return _rot.w(); }

  private:
    float _radius;
    Vec3f _center;
    Rotation _rot;
    Material _material;
    // derived
    float _dist;
};

class TriangleMesh;

class TriangleMeshFace : public Object {
    friend tinypt::cuda::TriangleMeshFace;

  public:
    TriangleMeshFace(const Vec3f &a, const Vec3f &b, const Vec3f &c, int face_id, const TriangleMesh *mesh);

    Vec3f location() const override { return Vec3f::Zero(); }
    void set_location(const Vec3f &location) override { LOG(WARNING) << "Cannot set location for TriangleMeshFace"; }
    Mat3f rotation() const override { return Mat3f::Identity(); }
    void set_rotation(const Mat3f &rotation) override { LOG(WARNING) << "cannot set rotation for TriangleMeshFace"; }
    bool has_material() const override { return true; }
    const Material &material() const override;
    void set_material(Material material) override { TINYPT_THROW << "Cannot set material of TriangleMeshFace"; }

    Vec3f va() const { return _a; }
    Vec3f vb() const { return _a + _ab; }
    Vec3f vc() const { return _a + _ac; }
    int face_id() const { return _face_id; }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override;
    void intersect_t(const Ray &ray, Hit &hit) const override;
    Vec3f sample(const Vec3f &org, RandEngine &rng) const override;
    float pdf(const Ray &ray) const override;
    AABB bounding_box() const override;
    std::vector<Object *> children() const override { return {}; }

  private:
    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

  private:
    Vec3f _a;
    Vec3f _ab;
    Vec3f _ac;
    int _face_id;
    const TriangleMesh *_mesh;
};

class ObjectGroup : public Object {
  public:
    ObjectGroup() = default;
    ObjectGroup(std::vector<std::shared_ptr<Object>> objects, int bvh_node_size = 0);

    Vec3f location() const override { return _mapping.location(); }
    void set_location(const Vec3f &location) override { _mapping.set_location(location); }
    Mat3f rotation() const override { return _mapping.rotation(); }
    void set_rotation(const Mat3f &rotation) override { _mapping.set_rotation(rotation); }
    bool has_material() const override { return false; }
    const Material &material() const override { TINYPT_THROW << "Cannot get material of ObjectGroup"; }
    void set_material(Material material) override { TINYPT_THROW << "Cannot set material of ObjectGroup"; }

    bool empty() const { return _objects.empty(); }
    const BVHTree &bvh_tree() const { return _bvh; }

    const std::vector<std::shared_ptr<Object>> &objects() const { return _objects; }
    std::vector<std::shared_ptr<Object>> &mutable_objects() { return _objects; }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override;
    void intersect_t(const Ray &ray, Hit &hit) const override;
    Vec3f sample(const Vec3f &org, RandEngine &rng) const override;
    float pdf(const Ray &ray) const override;
    AABB bounding_box() const override;
    std::vector<Object *> children() const override;

  private:
    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

  private:
    Mapping _mapping;
    std::vector<std::shared_ptr<Object>> _objects;
    BVHTree _bvh;
};

struct VertexInfo {
    int vertex_index;
    int normal_index;
    int texcoord_index;

    VertexInfo() = default;
    VertexInfo(int vertex_index_, int normal_index_, int texcoord_index_)
        : vertex_index(vertex_index_), normal_index(normal_index_), texcoord_index(texcoord_index_) {}
};

class TriangleMesh : public Object {
    friend TriangleMeshFace;
    friend tinypt::cuda::TriangleMesh;
    friend tinypt::cuda::Scene;

  public:
    TriangleMesh() = default;
    TriangleMesh(const TriangleMesh &other) = delete;
    TriangleMesh &operator=(const TriangleMesh &other) = delete;

    Vec3f location() const override { return _obj.location(); }
    void set_location(const Vec3f &location) override { _obj.set_location(location); }
    Mat3f rotation() const override { return _obj.rotation(); }
    void set_rotation(const Mat3f &rotation) override { _obj.set_rotation(rotation); }
    bool has_material() const override { return false; }
    const Material &material() const override { TINYPT_THROW << "Cannot get material of TriangleMesh"; }
    void set_material(Material material) override { TINYPT_THROW << "Cannot set material of TriangleMesh"; }

    static std::shared_ptr<TriangleMesh> from_obj(const std::string &path,
                                                  const std::shared_ptr<Material> &material = nullptr);

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const override { _obj.intersect(ray, hit, rng); }
    void intersect_t(const Ray &ray, Hit &hit) const override { _obj.intersect_t(ray, hit); }
    Vec3f sample(const Vec3f &org, RandEngine &rng) const override { return _obj.sample(org, rng); }
    float pdf(const Ray &ray) const override { return _obj.pdf(ray); }
    AABB bounding_box() const override { return _obj.bounding_box(); }
    std::vector<Object *> children() const override { return _obj.children(); }

  private:
    std::vector<Vec3f> _vertices;
    std::vector<Vec3f> _vertex_normals;
    std::vector<Vec2f> _texture_coords;
    std::vector<std::array<VertexInfo, 3>> _faces_index;
    std::vector<uint32_t> _material_ids;
    std::vector<Material> _materials;
    // TODO derived
    ObjectGroup _obj;
};

} // namespace cpu
} // namespace tinypt