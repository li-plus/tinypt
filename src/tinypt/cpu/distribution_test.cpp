#include "tinypt/cpu/distribution.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

struct TestCase1D {
    float u;
    float x;
    float p;

    friend std::ostream &operator<<(std::ostream &os, const TestCase1D &self) {
        return os << "TestCase1D(u=" << self.u << ", x=" << self.x << ", p=" << self.p << ")";
    }
};

TEST(Distribution, Distribution1D) {
    auto check = [](const Distribution1D &dist, const std::vector<TestCase1D> &cases) {
        for (auto &c : cases) {
            EXPECT_FLOAT_EQ(dist.sample(c.u), c.x) << c;
            if (c.p >= 0) {
                EXPECT_FLOAT_EQ(dist.pdf(c.x), c.p) << c;
            }
        }
    };

    {
        std::vector<TestCase1D> cases{{0, 0, 1},       {0.25, 0.25, 1}, {0.33, 0.33, 1}, {0.5, 0.5, 1},
                                      {0.66, 0.66, 1}, {0.75, 0.75, 1}, {0.8, 0.8, 1},   {1, 1, 1}};
        Distribution1D dist;
        check(dist, cases);
        dist = Distribution1D({1});
        check(dist, cases);
    }
    {
        float p1 = 0.2 * 2;
        float p2 = 0.8 * 2;
        Distribution1D dist({2, 8});
        std::vector<TestCase1D> cases{{{0, 0, p1},
                                       {0.2, 0.5, p1},
                                       {1, 1, p2},
                                       {0.1, 0.25, p1},
                                       {0.15, 0.375, p1},
                                       {0.6, 0.75, p2},
                                       {0.8, 0.875, p2}}};
        check(dist, cases);
    }
    {
        float p1 = 0.2 * 3;
        float p2 = 0.3 * 3;
        float p3 = 0.5 * 3;
        Distribution1D dist({2, 3, 5});
        std::vector<TestCase1D> cases{
            {0, 0, p1},         {0.2, 1 / 3.f, p1}, {0.5, 2 / 3.f, p2},  {1, 1, p3},
            {0.1, 1 / 6.f, p1}, {0.35, 0.5, p2},    {0.75, 5 / 6.f, p3},
        };
        check(dist, cases);
    }
}

struct TestCase2D {
    Vec2f uv;
    Vec2f xy;
    float p;

    TestCase2D(const Vec2f &uv_, const Vec2f &xy_, float p_) : uv(uv_), xy(xy_), p(p_) {}

    friend std::ostream &operator<<(std::ostream &os, const TestCase2D &self) {
        return os << "TestCase2D(uv=" << self.uv.transpose() << ", xy=" << self.xy.transpose() << ", p=" << self.p
                  << ")";
    }
};

TEST(Distribution, Distribution2D) {
    auto check = [](const Distribution2D &dist, const std::vector<TestCase2D> &cases) {
        for (auto &c : cases) {
            EXPECT_EQ(dist.sample(c.uv), c.xy) << c;
            if (c.p >= 0) {
                EXPECT_FLOAT_EQ(dist.pdf(c.xy), c.p) << c;
            }
        }
    };

    {
        std::vector<std::vector<float>> weights{{1}};
        Distribution2D dist(weights);
        std::vector<TestCase2D> cases{
            TestCase2D({0, 0}, {0, 0}, 1), TestCase2D({0, 1}, {0, 1}, 1),     TestCase2D({1, 0}, {1, 0}, 1),
            TestCase2D({1, 1}, {1, 1}, 1), TestCase2D({0, 0.5}, {0, 0.5}, 1), TestCase2D({0.5, 0.5}, {0.5, 0.5}, 1),
        };
        check(dist, cases);
    }
    {
        std::vector<std::vector<float>> weights{{2, 8}, {15, 15}};
        float pdfy[2] = {0.25 * 2, 0.75 * 2};
        float pdfx[2][2] = {{0.2 * 2, 0.8 * 2}, {0.5 * 2, 0.5 * 2}};
        Distribution2D dist(weights);
        std::vector<TestCase2D> cases{
            TestCase2D({0, 0}, {0, 0}, pdfy[0] * pdfx[0][0]),
            TestCase2D({0.2, 0}, {0.5, 0}, pdfy[0] * pdfx[0][0]),
            TestCase2D({1, 0}, {1, 0}, pdfy[0] * pdfx[0][1]),
            TestCase2D({0, 0.25}, {0, 0.5}, pdfy[0] * pdfx[0][0]),
            TestCase2D({0.2, 0.25}, {0.5, 0.5}, pdfy[0] * pdfx[0][0]),
            TestCase2D({1, 0.25}, {1, 0.5}, pdfy[0] * pdfx[0][1]),
            TestCase2D({0, 1}, {0, 1}, pdfy[1] * pdfx[1][0]),
            TestCase2D({0.5, 1}, {0.5, 1}, pdfy[1] * pdfx[1][0]),
            TestCase2D({1, 1}, {1, 1}, pdfy[1] * pdfx[1][1]),
        };
        check(dist, cases);
    }
}

} // namespace cpu
} // namespace tinypt
