#include "tinypt/cpu/bvh.h"

#include "tinypt/cpu/object.h"

namespace tinypt {
namespace cpu {

float AABB::intersect(const Ray &ray) const {
    float tmin = -INF;
    float tmax = INF;
    for (int a = 0; a < 3; a++) {
        float inv_dir = 1.f / ray.dir[a]; // inf!
        float t0 = (_box.min()[a] - ray.org[a]) * inv_dir;
        float t1 = (_box.max()[a] - ray.org[a]) * inv_dir;
        if (inv_dir < 0.f) {
            std::swap(t0, t1);
        }
        // don't use std::min or std::max since t0,t1 might be nan
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
    }
    if (tmax < tmin || tmax < EPS) {
        return INF;
    }
    DCHECK(std::isfinite(tmin));
    return tmin;
}

void BVHNode::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const {
    for (const auto *obj : objects) {
        obj->intersect(ray, hit, rng);
    }
}

void BVHNode::intersect_t(const Ray &ray, Hit &hit) const {
    for (const auto *obj : objects) {
        obj->intersect_t(ray, hit);
    }
}

BVHTree::BVHTree(std::vector<Object *> objects, size_t node_size) {
    TINYPT_CHECK(node_size > 0) << "got empty node";
    if (!objects.empty()) {
        int height = 1;
        while ((node_size << (height - 1)) < objects.size()) {
            height++;
        }
        _root = std::make_unique<BVHNode>();
        build_tree(_root.get(), objects, 0, objects.size(), height);
    }
}

void BVHTree::build_tree(BVHNode *node, std::vector<Object *> &objects, int lo, int hi, int height) const {
    height--;
    node->bbox = get_bbox(objects, lo, hi);
    if (height <= 0) {
        node->objects.assign(objects.begin() + lo, objects.begin() + hi);
        return;
    }

    // Split the axis with the largest range
    Vec3f axis_range = node->bbox.max() - node->bbox.min();
    int axis = (axis_range.x() > axis_range.y()) ? ((axis_range.x() > axis_range.z()) ? 0 : 2)
                                                 : ((axis_range.y() > axis_range.z()) ? 1 : 2);
    auto cmp_less = [&](const Object *x, const Object *y) {
        return x->bounding_box().min()[axis] < y->bounding_box().min()[axis];
    };
    int mid = (lo + hi) / 2;
    std::nth_element(objects.begin() + lo, objects.begin() + mid, objects.begin() + hi, cmp_less);

    // Build children nodes recursively
    node->left = std::make_unique<BVHNode>();
    build_tree(node->left.get(), objects, lo, mid, height);
    node->right = std::make_unique<BVHNode>();
    build_tree(node->right.get(), objects, mid, hi, height);
}

void BVHTree::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
void BVHTree::intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

template <bool ENABLE_SURFACE>
void BVHTree::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    float t = _root->bbox.intersect(ray);
    if (t < hit.t) {
        intersect_node<ENABLE_SURFACE>(ray, _root.get(), hit, rng);
    }
}

template <bool ENABLE_SURFACE>
void BVHTree::intersect_node(const Ray &ray, const BVHNode *node, Hit &hit, RandEngine *rng) const {
    if (node->is_leaf()) {
        // at leaf node, intersect every face
        if constexpr (ENABLE_SURFACE) {
            node->intersect(ray, hit, *rng);
        } else {
            node->intersect_t(ray, hit);
        }
        return;
    }

    float left_tmin = node->left->bbox.intersect(ray);
    float right_tmin = node->right->bbox.intersect(ray);
    if (left_tmin < right_tmin) {
        if (left_tmin < hit.t) {
            intersect_node<ENABLE_SURFACE>(ray, node->left.get(), hit, rng);
        }
        if (right_tmin < hit.t) {
            intersect_node<ENABLE_SURFACE>(ray, node->right.get(), hit, rng);
        }
    } else {
        if (right_tmin < hit.t) {
            intersect_node<ENABLE_SURFACE>(ray, node->right.get(), hit, rng);
        }
        if (left_tmin < hit.t) {
            intersect_node<ENABLE_SURFACE>(ray, node->left.get(), hit, rng);
        }
    }
}

AABB BVHTree::get_bbox(const std::vector<Object *> &objs, int lo, int hi) {
    TINYPT_CHECK(lo <= hi) << "invalid range [" << lo << "," << hi << "]";
    if (lo == hi) {
        return AABB(Vec3f::Zero(), Vec3f::Zero());
    }
    AABB bbox({INF, INF, INF}, {-INF, -INF, -INF});
    for (int i = lo; i < hi; i++) {
        bbox.extend(objs[i]->bounding_box());
    }
    return bbox;
}

std::ostream &operator<<(std::ostream &os, const BVHTree &tree) { return BVHTree::serialize(os, tree._root.get(), 0); }

std::ostream &BVHTree::serialize(std::ostream &os, const BVHNode *node, int offset) {
    if (node == nullptr) {
        return os;
    }
    // print left child
    serialize(os, node->left.get(), offset + 2);
    // print self
    os << std::string(offset, ' ');
    if (node->is_leaf()) {
        os << node->objects.size() << '\n';
    } else {
        os << "N\n";
    }
    // print right child
    return serialize(os, node->right.get(), offset + 2);
}

} // namespace cpu
} // namespace tinypt