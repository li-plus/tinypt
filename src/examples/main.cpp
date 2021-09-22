#include "scene_factory.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(num_samples, 100, "Number of samples per pixel");
DEFINE_string(save_path, "scene.png", "Path to output image");
DEFINE_string(device, "cpu", "Backend device");
DEFINE_string(scene, "cornell_box", "Scene name");

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::SetUsageMessage("A simple path tracing renderer");
    google::InstallFailureSignalHandler();
    google::ParseCommandLineFlags(&argc, &argv, true);

    try {
        tinypt::Device device(FLAGS_device);

        SceneFactory factory;
        auto scene = factory.make_scene(FLAGS_scene).to(device);

        auto pt = tinypt::PathTracer(device);
        LOG(INFO) << "begin rendering. scene: " << FLAGS_scene << ", spp: " << FLAGS_num_samples;
        auto start = std::chrono::system_clock::now();
        tinypt::Image img = pt.render(scene, FLAGS_num_samples);
        auto end = std::chrono::system_clock::now();
        auto ms_cnt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        LOG(INFO) << "finish rendering. time: " << ms_cnt / 1000.f << "s, output: " << FLAGS_save_path;
        img.save(FLAGS_save_path);
    } catch (const std::exception &e) {
        LOG(ERROR) << e.what();
        exit(EXIT_FAILURE);
    }
    return 0;
}
