#pragma once

#include "tinypt/cpu/defs.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace tinypt {
namespace cpu {

class RGBAImage;

class Image {
  public:
    Image() = default;
    Image(int width, int height, int channels) : _mat(height, width, CV_32FC(channels)) {}
    Image(cv::Mat mat) : _mat(std::move(mat)) {}
    Image(const cv::MatExpr &mat_expr) : _mat(mat_expr) {}

    cv::Mat &mat() { return _mat; }
    const cv::Mat &mat() const { return _mat; }

    bool empty() const { return _mat.empty(); }
    int width() const { return _mat.cols; }
    int height() const { return _mat.rows; }
    int channels() const { return _mat.channels(); }

    template <typename T>
    const T &at(int x, int y) const;
    template <typename T>
    T &at(int x, int y);

    static Image zeros(int width, int height, int channels) { return cv::Mat::zeros(height, width, CV_32FC(channels)); }
    static Image ones(int width, int height, int channels) { return cv::Mat::ones(height, width, CV_32FC(channels)); }

    const uint8_t *data() const { return _mat.data; }
    uint8_t *data() { return _mat.data; }

    friend Image operator*(float s, const Image &a) { return s * a._mat; }
    friend Image operator*(const Image &a, float s) { return a._mat * s; }
    Image &operator*=(float s) { return *this = *this * s; }

    // custom methods
    static RGBAImage open(const std::string &path, bool grayscale);
    void save(const std::string &path) const;

  private:
    template <typename T, typename CV_T>
    const T &at_impl(int x, int y) const {
        return *(const T *)&_mat.at<CV_T>(y, x);
    }
    template <typename T, typename CV_T>
    T &at_impl(int x, int y) {
        return *(T *)&_mat.at<CV_T>(y, x);
    }

  private:
    cv::Mat _mat;
};

template <>
inline const float &Image::at(int x, int y) const {
    return at_impl<float, float>(x, y);
}
template <>
inline float &Image::at(int x, int y) {
    return at_impl<float, float>(x, y);
}
template <>
inline const Vec2f &Image::at(int x, int y) const {
    return at_impl<Vec2f, cv::Vec2f>(x, y);
}
template <>
inline Vec2f &Image::at(int x, int y) {
    return at_impl<Vec2f, cv::Vec2f>(x, y);
}
template <>
inline const Vec3f &Image::at(int x, int y) const {
    return at_impl<Vec3f, cv::Vec3f>(x, y);
}
template <>
inline Vec3f &Image::at(int x, int y) {
    return at_impl<Vec3f, cv::Vec3f>(x, y);
}
template <>
inline const Vec4f &Image::at(int x, int y) const {
    return at_impl<Vec4f, cv::Vec4f>(x, y);
}
template <>
inline Vec4f &Image::at(int x, int y) {
    return at_impl<Vec4f, cv::Vec4f>(x, y);
}

class RGBAImage {
  public:
    RGBAImage() = default;
    RGBAImage(Image rgb, Image alpha) : _rgb(std::move(rgb)), _alpha(std::move(alpha)) {}

    const Image &rgb() const { return _rgb; }
    Image &rgb() { return _rgb; }

    const Image &alpha() const { return _alpha; }
    Image &alpha() { return _alpha; }

  private:
    Image _rgb;
    Image _alpha;
};

} // namespace cpu
} // namespace tinypt