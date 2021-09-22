#include "tinypt/cpu/image.h"

namespace tinypt {
namespace cpu {

RGBAImage Image::open(const std::string &path, bool grayscale) {
    auto read_mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED;
    cv::Mat mat = cv::imread(path, read_mode);
    TINYPT_CHECK(!mat.empty()) << "[cv::imread] cannot open image " << path;

    cv::flip(mat, mat, 0); // flip vertically

    // separate rgb & alpha channels
    cv::Mat rgb;
    cv::Mat alpha;
    if (grayscale) {
        alpha = mat;
    } else {
        if (mat.channels() == 4) {
            cv::Mat channels[4];
            cv::split(mat, channels);
            alpha = channels[3];
            cv::merge(channels, 3, rgb);
        } else if (mat.channels() == 1) {
            cv::cvtColor(mat, rgb, cv::COLOR_GRAY2BGR);
        } else {
            TINYPT_CHECK(mat.channels() == 3) << "invalid image channels: " << mat.channels();
            rgb = mat;
        }
    }

    // process alpha
    if (!alpha.empty()) {
        TINYPT_CHECK(alpha.channels() == 1) << "invalid alpha channels: " << alpha.channels();
        if (alpha.depth() == CV_8U) {
            alpha.convertTo(alpha, CV_32FC1, 1 / 255.);
        } else {
            TINYPT_CHECK(alpha.depth() == CV_32F) << "invalid alpha depth: " << alpha.depth();
        }
    }

    // process rgb
    if (!rgb.empty()) {
        TINYPT_CHECK(rgb.channels() == 3) << "invalid rgb channels " << rgb.channels();
        if (rgb.depth() == CV_8U) {
            rgb.convertTo(rgb, CV_32FC3, 1 / 255.);
        } else {
            TINYPT_CHECK(rgb.depth() == CV_32F) << "invalid rgb depth " << rgb.depth();
            auto tonemap = cv::createTonemapDrago(SRGB_GAMMA);
            tonemap->process(rgb, rgb);
            constexpr float LDR_SCALE = 4;
            rgb = cv::max(0, cv::min(rgb * LDR_SCALE, 1));
        }
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        cv::pow(rgb, SRGB_GAMMA, rgb); // gamma
    }

    return RGBAImage(std::move(rgb), std::move(alpha));
}

void Image::save(const std::string &path) const {
    cv::Mat mat = _mat.clone();
    cv::flip(mat, mat, 0); // flip vertically
    if (channels() == 3) {
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::pow(mat, 1 / SRGB_GAMMA, mat); // gamma
    } else {
        TINYPT_CHECK(channels() == 1) << "invalid image channels: " << channels();
    }
    mat.convertTo(mat, CV_8UC(mat.channels()), 255.);
    TINYPT_CHECK(cv::imwrite(path, mat)) << "[cv::imwrite] cannot save image to " << path;
}

} // namespace cpu
} // namespace tinypt
