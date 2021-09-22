#include "tinypt/cpu/object.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

TEST(BVH, BVHTree) {
    constexpr float x1 = 6;
    constexpr float y1 = 5;
    constexpr float z1 = 4;
    Vec3f px(x1, 0, 0);
    Vec3f py(0, y1, 0);
    Vec3f pz(0, 0, z1);

    TriangleMeshFace t1(px, py, pz, 0, nullptr);
    TriangleMeshFace t2(-px, py, pz, 0, nullptr);
    TriangleMeshFace t3(-px, -py, pz, 0, nullptr);
    TriangleMeshFace t4(px, -py, pz, 0, nullptr);
    TriangleMeshFace t5(px, py, -pz, 0, nullptr);
    TriangleMeshFace t6(-px, py, -pz, 0, nullptr);
    TriangleMeshFace t7(-px, -py, -pz, 0, nullptr);
    TriangleMeshFace t8(px, -py, -pz, 0, nullptr);

    {
        BVHTree bvh({}, 1);
        EXPECT_TRUE(bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4, &t5, &t6, &t7}, 1);
        EXPECT_TRUE(!bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4, &t5, &t6}, 1);
        EXPECT_TRUE(!bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4, &t5}, 1);
        EXPECT_TRUE(!bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4}, 1);
        EXPECT_TRUE(!bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4, &t5, &t6, &t7}, 2);
        EXPECT_TRUE(!bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4, &t5, &t6}, 2);
        EXPECT_TRUE(!bvh.empty());
        bvh = BVHTree({&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8}, 3);
        EXPECT_TRUE(!bvh.empty());
    }

    struct NodeInfo {
        BVHNode *node;
        AABB bbox;
        bool is_leaf;
        std::vector<Object *> objects;
    };

    // ===== 8 TRIANGLES & 8 LEAVES =====
    BVHTree bvh({&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8}, 1);
    EXPECT_TRUE(!bvh.empty());
    std::vector<NodeInfo> ans(16);

    ans[1].node = bvh.mutable_root();
    for (int i = 1; i < 8; i++) {
        ans[i * 2].node = ans[i].node->left.get();
        ans[i * 2 + 1].node = ans[i].node->right.get();
    }
    ans[1].bbox = AABB({-x1, -y1, -z1}, {x1, y1, z1});
    ans[2].bbox = AABB({-x1, -y1, -z1}, {0, y1, z1});
    ans[3].bbox = AABB({0, -y1, -z1}, {x1, y1, z1});
    ans[4].bbox = AABB({-x1, -y1, -z1}, {0, 0, z1});
    ans[5].bbox = AABB({-x1, 0, -z1}, {0, y1, z1});
    ans[6].bbox = AABB({0, -y1, -z1}, {x1, 0, z1});
    ans[7].bbox = AABB({0, 0, -z1}, {x1, y1, z1});
    ans[8].bbox = AABB({-x1, -y1, -z1}, {0, 0, 0});
    ans[9].bbox = AABB({-x1, -y1, 0}, {0, 0, z1});
    ans[10].bbox = AABB({-x1, 0, -z1}, {0, y1, 0});
    ans[11].bbox = AABB({-x1, 0, 0}, {0, y1, z1});
    ans[12].bbox = AABB({0, -y1, -z1}, {x1, 0, 0});
    ans[13].bbox = AABB({0, -y1, 0}, {x1, 0, z1});
    ans[14].bbox = AABB({0, 0, -z1}, {x1, y1, 0});
    ans[15].bbox = AABB({0, 0, 0}, {x1, y1, z1});

    for (int i = 1; i < 16; i++) {
        ans[i].is_leaf = i > 7;
    }

    ans[8].objects = {&t7};
    ans[9].objects = {&t3};
    ans[10].objects = {&t6};
    ans[11].objects = {&t2};
    ans[12].objects = {&t8};
    ans[13].objects = {&t4};
    ans[14].objects = {&t5};
    ans[15].objects = {&t1};

    ans.erase(ans.begin());
    for (const auto &rec : ans) {
        const auto *node = rec.node;
        EXPECT_TRUE(node->bbox.isApprox(rec.bbox));
        EXPECT_EQ(node->is_leaf(), rec.is_leaf);
        EXPECT_EQ(node->objects, rec.objects);
    }
}

} // namespace cpu
} // namespace tinypt