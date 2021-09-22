#include "tinypt/cpu/light.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

TEST(PointLightTest, basic) {
    Vec3f location({3, 4, 5});
    Vec3f color({0.4, 0.5, 0.6});
    float power = 2;
    PointLight light;
    light.set_location(location);
    light.set_color(color);
    light.set_power(power);
    EXPECT_TRUE(light.is_delta());
    EXPECT_TRUE(light.location().isApprox(location));
    EXPECT_TRUE(light.color().isApprox(color));
    EXPECT_FLOAT_EQ(light.power(), power);
    EXPECT_TRUE(light.emission().isApprox(color * power));

    RandEngine rng;
    Vec3f pos(rng.random(), rng.random(), rng.random());
    auto dir = light.sample(pos, rng);
    EXPECT_TRUE(dir.isApprox((location - pos).normalized()));
}

TEST(DistantLightTest, basic) {
    DistantLight light;
    auto ray_dir = Vec3f(1, 2, 3).normalized();
    light.set_rotation(Rotation::from_direction(-ray_dir).matrix());
    float power = 8;
    light.set_power(power);
    EXPECT_TRUE(light.is_delta());
    EXPECT_TRUE(light.color().isOnes());
    EXPECT_FLOAT_EQ(light.power(), power);
    EXPECT_TRUE(light.emission().isApprox(Vec3f::Ones() * power));

    RandEngine rng;
    Vec3f pos(rng.random(16), rng.random(16), rng.random(16));
    auto dir = light.sample(pos, rng);
    EXPECT_TRUE(dir.isApprox(-ray_dir));
}

} // namespace cpu
} // namespace tinypt