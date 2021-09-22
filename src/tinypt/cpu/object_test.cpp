#include "tinypt/cpu/object.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

struct IntersectTestCase {
    Ray ray;
    Hit hit;

    void check(const Hit &out_hit) const {
        EXPECT_FLOAT_EQ(hit.t, out_hit.t);
        EXPECT_EQ(hit.into, out_hit.into);
        EXPECT_TRUE(hit.normal.isApprox(out_hit.normal))
            << "[" << hit.normal.transpose() << "] vs [" << out_hit.normal.transpose() << "]";
        EXPECT_TRUE(hit.uv.isApprox(out_hit.uv))
            << "[" << hit.uv.transpose() << "] vs [" << out_hit.uv.transpose() << "]";
    }
};

TEST(Sphere, basic) {
    RandEngine rng;

    Vec3f location(1, 2, 3);
    constexpr float radius = 2;
    Sphere sphere;
    sphere.set_location(location);
    sphere.set_radius(radius);
    EXPECT_TRUE(sphere.location().isApprox(location));
    EXPECT_FLOAT_EQ(sphere.radius(), radius);

    IntersectTestCase c1;
    c1.ray = Ray(location, Vec3f::Ones().normalized());
    c1.hit.t = radius;
    c1.hit.into = false;
    c1.hit.normal = -Vec3f::Ones().normalized();
    c1.hit.uv = Vec2f(0.625, std::acos(1 / std::sqrt(3)) / M_PIf32);

    IntersectTestCase c2;
    c2.ray = Ray(location + Vec3f(0, radius + 3, 0), -Vec3f::UnitY());
    c2.hit.t = 3;
    c2.hit.into = true;
    c2.hit.normal = Vec3f::UnitY();
    c2.hit.uv = Vec2f(0.75, 0.5);

    for (const auto &c : {c1, c2}) {
        Hit hit;
        sphere.intersect(c.ray, hit, rng);
        c.check(hit);
    }
}

TEST(Circle, bounding_box) {
    {
        Circle circle;
        float r = 2;
        circle.set_radius(r);
        AABB bbox({-r, -r, -EPS}, {r, r, EPS});
        EXPECT_TRUE(circle.bounding_box().isApprox(bbox));
    }
    {
        Circle circle;
        float r = 2;
        circle.set_radius(r);
        Vec3f loc(1, 2, 3);
        circle.set_location(loc);
        circle.set_rotation(AngleAxisf(radians(45), Vec3f::UnitX()).matrix());
        AABB bbox({loc.x() - r, loc.y() - M_SQRT2f32, loc.z() - M_SQRT2f32},
                  {loc.x() + r, loc.y() + M_SQRT2f32, loc.z() + M_SQRT2f32});
        EXPECT_TRUE(circle.bounding_box().isApprox(bbox, EPS)) << circle.bounding_box() << " vs " << bbox;
    }
}

TEST(Circle, sample) {
    RandEngine rng;

    Circle circle;
    circle.set_radius(2);

    for (int i = 0; i < 100; i++) {
        Vec3f org(1, 2, 3);
        Vec3f dir = circle.sample(org, rng);
        Ray ray(org, dir);
        Hit hit;
        circle.intersect_t(ray, hit);
        EXPECT_TRUE(hit.is_hit());
    }
}

TEST(Circle, basic) {
    RandEngine rng;

    Circle circle;
    circle.set_radius(2);
    EXPECT_FLOAT_EQ(circle.radius(), 2);

    IntersectTestCase c1;
    c1.ray = Ray({0, 0, 5}, -Vec3f::UnitZ());
    c1.hit.t = 5;
    c1.hit.into = true;
    c1.hit.normal = Vec3f::UnitZ();
    c1.hit.uv = {0, 0};

    IntersectTestCase c2;
    c2.ray = Ray({1, -0.5, 5}, -Vec3f::UnitZ());
    c2.hit.t = 5;
    c2.hit.into = true;
    c2.hit.normal = Vec3f::UnitZ();
    c2.hit.uv = {0.5, -0.25};

    for (const auto &c : {c1, c2}) {
        Hit hit;
        circle.intersect(c.ray, hit, rng);
        c.check(hit);
    }
}

TEST(Rectangle, basic) {
    RandEngine rng;

    // attributes
    {
        Rectangle rect;
        EXPECT_TRUE(rect.location().isApprox(Vec3f::Zero()));
        EXPECT_TRUE(rect.rotation().isApprox(Mat3f::Identity()));

        for (int i = 0; i < 100; i++) {
            Vec3f org(1, 2, 3);
            Vec3f dir = rect.sample(org, rng);
            Ray ray(org, dir);
            Hit hit;
            rect.intersect(ray, hit, rng);
            EXPECT_TRUE(hit.is_hit());
        }
    }

    // intersection test
    {
        Rectangle rect;
        rect.set_location(Vec3f::Zero());
        rect.set_dimension({1, 2});

        IntersectTestCase c1;
        c1.ray = Ray({1, 1, 1}, -Vec3f::Ones().normalized());
        c1.hit.t = std::sqrt(3);
        c1.hit.into = true;
        c1.hit.normal = Vec3f::UnitZ();
        c1.hit.uv = {0.5, 0.5};

        IntersectTestCase c2;
        c2.ray = Ray({-1 + 0.25, -1 - 0.5, -1}, Vec3f::Ones().normalized());
        c2.hit.t = std::sqrt(3);
        c2.hit.into = true;
        c2.hit.normal = -Vec3f::UnitZ();
        c2.hit.uv = {0.75, 0.25};

        for (const auto &c : {c1, c2}) {
            Hit hit;
            rect.intersect(c.ray, hit, rng);
            c.check(hit);
        }
    }

    // zero dimension
    {
        Rectangle rect;
        rect.set_dimension(Vec2f::Zero());
        IntersectTestCase c1;
        c1.ray = Ray({0, 0, 1}, {0, 0, -1});
        c1.hit.t = INF;
        for (const auto &c : {c1}) {
            Hit hit;
            rect.intersect(c.ray, hit, rng);
            c.check(hit);
        }
    }
}

} // namespace cpu
} // namespace tinypt