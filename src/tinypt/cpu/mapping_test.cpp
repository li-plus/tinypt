#include "tinypt/cpu/mapping.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

struct RotationCase {
    Rotation out;
    Rotation ans;

    RotationCase(const Rotation &out_, const Rotation &ans_) : out(out_), ans(ans_) {}
};

TEST(Rotation, basic) {
    RotationCase c1(Rotation::from_direction(Vec3f::UnitZ()), Rotation::identity());
    RotationCase c2(Rotation::from_direction(Vec3f::UnitY()), Rotation(Vec3f::UnitZ(), Vec3f::UnitX(), Vec3f::UnitY()));
    RotationCase c3(Rotation::from_euler({0, 1, 2}, {0, 0, 0}), Rotation::identity());
    RotationCase c4(Rotation::from_euler({0, 1, 2}, {90, 0, 0}, true),
                    Rotation(Vec3f::UnitX(), Vec3f::UnitZ(), -Vec3f::UnitY()));
    for (const auto &c : {c1, c2, c3, c4}) {
        EXPECT_TRUE(c.out.isApprox(c.ans)) << c.out << " vs " << c.ans;
    }
}

TEST(Mapping, map_bounding_box) {
    Mapping mapping;
    EXPECT_TRUE(mapping.location().isApprox(Vec3f::Zero()));
    EXPECT_TRUE(mapping.rotation().isApprox(Mat3f::Identity()));

    Vec3f loc(1, 2, 3);
    Mat3f rot = Rotation::from_euler({0, 1, 2}, {10, 20, 30}, true).matrix();
    mapping.set_location(loc);
    mapping.set_rotation(rot);
    EXPECT_TRUE(mapping.location().isApprox(loc));
    EXPECT_TRUE(mapping.rotation().isApprox(rot));

    Vec3f dir = Vec3f(3, 4, 5).normalized();
    EXPECT_TRUE(mapping.map_direction(dir).isApprox(rot * dir));
    EXPECT_TRUE(mapping.map_direction_inverse(dir).isApprox(rot.inverse() * dir));

    Vec3f pos(3, 4, 5);
    EXPECT_TRUE(mapping.map_point(pos).isApprox(rot * pos + loc));
    EXPECT_TRUE(mapping.map_point_inverse(pos).isApprox(rot.inverse() * (pos - loc)));
}

} // namespace cpu
} // namespace tinypt