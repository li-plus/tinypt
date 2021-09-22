#include "tinypt/cpu/rand.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

TEST(RandEngine, basic) {
    RandEngine rng;

    for (int i = 0; i < 100; i++) {
        {
            float u;
            u = rng.random();
            EXPECT_TRUE(0 <= u && u <= 1) << u;
            u = rng.random(100);
            EXPECT_TRUE(0 <= u && u <= 100) << u;
            u = rng.random(-.5, .5);
            EXPECT_TRUE(-.5 <= u && u <= .5) << u;
        }
        {
            int u;
            u = rng.rand_int();
            EXPECT_TRUE(0 <= u && u <= RAND_MAX);
            u = rng.rand_int(100);
            EXPECT_TRUE(0 <= u && u <= 100);
            u = rng.rand_int(-100, 100);
            EXPECT_TRUE(-100 <= u && u <= 100);
        }
        {
            Vec2f uv;
            uv = rng.rand_on_disk(0.5f);
            EXPECT_FLOAT_EQ(uv.norm(), 0.5f);
            uv = rng.rand_in_disk(0.5f);
            EXPECT_LT(uv.norm(), 0.5f + EPS);
        }
        {
            Vec3f pos;
            pos = rng.rand_on_sphere(0.5f);
            EXPECT_FLOAT_EQ(pos.norm(), 0.5f);
            pos = rng.rand_in_sphere(0.5f);
            EXPECT_LT(pos.norm(), 0.5f + EPS);
        }
    }
}

} // namespace cpu
} // namespace tinypt