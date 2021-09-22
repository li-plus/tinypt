#pragma once

#include "tinypt/cpu/rand.h"
#include "tinypt/cpu/ray.h"

namespace tinypt {
namespace cpu {

// Axis-Aligned Bounding Box
class AABB {
  public:
    AABB() = default;
    AABB(const AlignedBox3f &box) : _box(box) {}
    AABB(const Vec3f &minv, const Vec3f &maxv) : _box(minv, maxv) {}

    float intersect(const Ray &ray) const;

    bool empty() const { return _box.isEmpty(); }
    const Vec3f &min() const { return _box.min(); }
    const Vec3f &max() const { return _box.max(); }

    AABB &extend(const AABB &other) {
        _box.extend(other._box);
        return *this;
    }

    bool isApprox(const AABB &other, float prec = decltype(_box)::ScalarTraits::dummy_precision()) const {
        return _box.isApprox(other._box, prec);
    }

    friend std::ostream &operator<<(std::ostream &os, const AABB &self) {
        return os << "AABB(min={" << self.min().transpose() << "}, max={" << self.max().transpose() << "})";
    }

  private:
    AlignedBox3f _box;
};

class Object;

struct BVHNode {
    // data
    AABB bbox;
    std::vector<Object *> objects;
    // children
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;

    bool is_leaf() const { return left == nullptr; }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const;
    void intersect_t(const Ray &ray, Hit &hit) const;
};

class BVHTree {
  public:
    BVHTree() = default;
    BVHTree(std::vector<Object *> objects, size_t node_size);

    bool empty() const { return _root == nullptr; }
    const BVHNode *root() const { return _root.get(); }
    BVHNode *mutable_root() { return _root.get(); }

    void intersect(const Ray &ray, Hit &hit, RandEngine &rng) const;
    void intersect_t(const Ray &ray, Hit &hit) const;

    AABB bounding_box() const { return _root->bbox; }

    friend std::ostream &operator<<(std::ostream &os, const BVHTree &tree);

  private:
    template <bool ENABLE_SURFACE = true>
    void intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const;

    template <bool ENABLE_SURFACE = true>
    void intersect_node(const Ray &ray, const BVHNode *node, Hit &hit, RandEngine *rng) const;

    void build_tree(BVHNode *node, std::vector<Object *> &objects, int lo, int hi, int height) const;

    static AABB get_bbox(const std::vector<Object *> &objs, int lo, int hi);

    static std::ostream &serialize(std::ostream &os, const BVHNode *node, int offset);

  private:
    std::unique_ptr<BVHNode> _root;
};

} // namespace cpu
} // namespace tinypt
