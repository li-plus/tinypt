#include "tinypt/cpu/texture.h"

#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

class EnvTextureTest : public ::testing::Test {
  public:
    static void test_basic() {
        std::vector<Vec2f> uvs;
        for (int i = 1; i < 10; i++) {
            for (int j = 1; j < 10; j++) {
                uvs.emplace_back(i / 10., j / 10.);
            }
        }
        for (const auto &uv : uvs) {
            Vec3f out_dir = EnvTexture::uv2dir(uv);
            Vec2f out_uv = EnvTexture::dir2uv(out_dir);
            EXPECT_TRUE(uv.isApprox(out_uv)) << " uv=" << uv.transpose() << " out_uv=" << out_uv.transpose();
        }
    }
};

TEST_F(EnvTextureTest, basic) { test_basic(); }

TEST(BumpTexture, basic) {
    cv::Mat mat = (cv::Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    cv::Mat grad_x = (cv::Mat_<float>(3, 3) << 4, 8, 4, 4, 8, 4, 4, 8, 4) / 8;
    cv::Mat grad_y = (cv::Mat_<float>(3, 3) << 12, 12, 12, 24, 24, 24, 12, 12, 12) / 8;
    BumpTexture bump(Image(std::move(mat)), 1);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            auto dbduv = bump.map().at<Vec2f>(x, y);
            EXPECT_FLOAT_EQ(dbduv.x(), grad_x.at<float>(y, x));
            EXPECT_FLOAT_EQ(dbduv.y(), grad_y.at<float>(y, x));
        }
    }
}

} // namespace cpu
} // namespace tinypt