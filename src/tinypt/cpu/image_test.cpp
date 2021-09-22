#include "tinypt/cpu/image.h"

#include <filesystem>
#include <gtest/gtest.h>

namespace tinypt {
namespace cpu {

namespace fs = std::filesystem;

TEST(Image, Image) {
    fs::path proj_root = fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    {
        auto img_path = proj_root / "resource/envmap/venice_sunset_4k.hdr";
        ASSERT_TRUE(fs::exists(img_path));
        auto rgba = Image::open(img_path, false);
        EXPECT_TRUE(!rgba.rgb().empty());
        EXPECT_EQ(rgba.rgb().channels(), 3);
        rgba.rgb().save("test_venice_sunset_4k.png");
        EXPECT_TRUE(rgba.alpha().empty());
    }
    {
        auto img_path = proj_root / "resource/fireplace_room/textures/leaf.png";
        ASSERT_TRUE(fs::exists(img_path));
        auto rgba = Image::open(proj_root / "resource/fireplace_room/textures/leaf.png", false);
        EXPECT_TRUE(!rgba.rgb().empty());
        EXPECT_EQ(rgba.rgb().channels(), 3);
        EXPECT_TRUE(!rgba.alpha().empty());
        EXPECT_EQ(rgba.alpha().channels(), 1);
        EXPECT_EQ(rgba.rgb().width(), rgba.alpha().width());
        EXPECT_EQ(rgba.rgb().height(), rgba.alpha().height());
        rgba.rgb().save("test_leaf_rgb.png");
        rgba.alpha().save("test_leaf_alpha.png");
    }
}

} // namespace cpu
} // namespace tinypt