#pragma once

#include "tinypt/cpu/camera.h"
#include "tinypt/cpu/image.h"
#include "tinypt/cpu/material.h"
#include "tinypt/cpu/scene.h"

namespace tinypt {

using cpu::degrees;
using cpu::radians;
using cpu::rgb_color;
using cpu::SRGB_GAMMA;

using cpu::Affine2f;
using cpu::Affine3f;
using cpu::AlignedBox3f;
using cpu::AngleAxisf;
using cpu::Mat2f;
using cpu::Mat3f;
using cpu::Mat4f;
using cpu::Vec2f;
using cpu::Vec2i;
using cpu::Vec3f;
using cpu::Vec3i;
using cpu::Vec4f;
using cpu::Vec4i;

using cpu::AABB;
using cpu::Camera;
using cpu::Hit;
using cpu::Image;
using cpu::RandEngine;
using cpu::Ray;
using cpu::Rotation;

using cpu::Circle;
using cpu::Object;
using cpu::Rectangle;
using cpu::Sphere;
using cpu::TriangleMesh;

using cpu::DistantLight;
using cpu::EnvLight;
using cpu::Light;
using cpu::ObjectLight;
using cpu::PointLight;

using cpu::AddBXDF;
using cpu::BXDF;
using cpu::Dielectric;
using cpu::Glossy;
using cpu::Lambertian;
using cpu::Material;
using cpu::Metal;

using cpu::AlphaTexture;
using cpu::BumpTexture;
using cpu::EnvTexture;
using cpu::RGBTexture;

class Device {
  public:
    enum DeviceType { DEVICE_CPU, DEVICE_CUDA };

    Device() : _device(DEVICE_CPU) {}
    Device(DeviceType device) : _device(device) {}
    Device(const std::string &device_str) {
        if (device_str == "cpu") {
            _device = DEVICE_CPU;
        } else if (device_str == "cuda") {
            _device = DEVICE_CUDA;
        } else {
            TINYPT_THROW_EX(std::invalid_argument) << "Unknown device " << device_str;
        }
    }

    bool operator==(const Device &other) const { return _device == other._device; }
    bool operator!=(const Device &other) const { return !(*this == other); }

    bool is_cpu() const { return _device == DEVICE_CPU; }
    bool is_cuda() const { return _device == DEVICE_CUDA; }

  private:
    DeviceType _device;
};

} // namespace tinypt