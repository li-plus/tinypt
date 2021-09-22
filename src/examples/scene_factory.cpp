#include "scene_factory.h"

SceneFactory::SceneFactory() {
    register_scene_builder("cornell_sphere", [] { return make_cornell_sphere(); });
    register_scene_builder("cornell_box", [] { return make_cornell_box(); });
    register_scene_builder("breakfast_room", [] { return make_breakfast_room(); });
    register_scene_builder("living_room", [] { return make_living_room(); });
    register_scene_builder("fireplace_room", [] { return make_fireplace_room(); });
    register_scene_builder("rungholt", [] { return make_rungholt(); });
    register_scene_builder("dabrovic_sponza", [] { return make_dabrovic_sponza(); });
    register_scene_builder("salle_de_bain", [] { return make_salle_de_bain(); });
    register_scene_builder("debug", [] { return make_debug(); });
    register_scene_builder("debug2", [] { return make_debug2(); });
}

void SceneFactory::register_scene_builder(const std::string &name, std::function<tinypt::Scene()> scene_builder) {
    _scene_builders[name] = std::move(scene_builder);
}

tinypt::Scene SceneFactory::make_scene(const std::string &name) const {
    auto it = _scene_builders.find(name);
    if (it == _scene_builders.end()) {
        throw std::invalid_argument("scene not found: " + name);
    }
    return it->second();
}

tinypt::Scene SceneFactory::make_cornell_box() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/CornellBox/CornellBox-Original.obj");

    tinypt::Camera camera;
    camera.set_location({0, -3.5, 1});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {90, 0, 0}, true).matrix());
    camera.set_resolution(1024, 1024);
    camera.set_fov(tinypt::radians(42));

    tinypt::Scene scene(camera, {mesh});
    return scene;
}

tinypt::Scene SceneFactory::make_cornell_sphere() {
    float x1 = 1, y1 = -170, z1 = 0;
    float x2 = 99, y2 = 0, z2 = 81.6;
    float xm = (x1 + x2) / 2, ym = (y1 + y2) / 2, zm = (z1 + z2) / 2;
    float xs = x2 - x1, ys = y2 - y1, zs = z2 - z1;

    auto left = std::make_shared<tinypt::Rectangle>();
    left->set_location({x1, ym, zm});
    left->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {0, 90, 0}, true).matrix());
    left->set_dimension({zs, ys});
    left->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .25, .25))));
    auto right = std::make_shared<tinypt::Rectangle>();
    right->set_location({x2, ym, zm});
    right->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {0, 90, 0}, true).matrix());
    right->set_dimension({zs, ys});
    right->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.25, .25, .75))));
    auto back = std::make_shared<tinypt::Rectangle>();
    back->set_location({xm, y2, zm});
    back->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {90, 0, 0}, true).matrix());
    back->set_dimension({xs, zs});
    back->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .75, .75))));
    auto bottom = std::make_shared<tinypt::Rectangle>();
    bottom->set_location({xm, ym, z1});
    bottom->set_dimension({xs, ys});
    bottom->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .75, .75))));
    auto top = std::make_shared<tinypt::Rectangle>();
    top->set_location({xm, ym, z2});
    top->set_dimension({xs, ys});
    top->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .75, .75))));
    auto mirror_sphere = std::make_shared<tinypt::Sphere>();
    mirror_sphere->set_location({27, -47, 16.5});
    mirror_sphere->set_radius(16.5);
    mirror_sphere->set_material(tinypt::Material(std::make_shared<tinypt::Metal>(tinypt::Vec3f(.999, .999, .999))));
    auto glass_sphere = std::make_shared<tinypt::Sphere>();
    glass_sphere->set_location({73, -78, 16.5});
    glass_sphere->set_radius(16.5);
    glass_sphere->set_material(
        tinypt::Material(std::make_shared<tinypt::Dielectric>(tinypt::Vec3f(.999, .999, .999), 1.5)));
    auto light = std::make_shared<tinypt::Circle>();
    light->set_radius(18);
    light->set_location({50, -81.6, 81.6 - 0.01});
    light->set_material(
        tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(0, 0, 0)), tinypt::Vec3f(12, 12, 12)));

    // camera
    auto w = tinypt::Vec3f(0, -1, 0.042612).normalized();
    auto u = tinypt::Vec3f::UnitX();
    auto v = w.cross(u);
    tinypt::Mat3f cam_rot;
    cam_rot << u, v, w;

    tinypt::Camera camera;
    camera.set_location({50, -295.6, 52});
    camera.set_rotation(cam_rot);
    camera.set_resolution(1024, 768);
    camera.set_fov(tinypt::radians(39.32));

    tinypt::Scene scene(camera, {left, right, back, bottom, top, mirror_sphere, glass_sphere, light});
    return scene;
}

tinypt::Scene SceneFactory::make_breakfast_room() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/breakfast_room/breakfast_room.obj");
    auto area_light = std::make_shared<tinypt::Rectangle>();
    area_light->set_location({-0.596747, 1.83138, 7.02496});
    area_light->set_dimension({11.3233, 6});
    area_light->set_material(
        tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(0, 0, 0)), tinypt::Vec3f(2, 2, 2)));
    auto sunlight = std::make_shared<tinypt::DistantLight>();
    sunlight->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {50.3857, -0.883569, 74.8117}, true).matrix());
    sunlight->set_power(8);
    sunlight->set_angle(tinypt::radians(2.2));

    tinypt::Camera camera;
    camera.set_location({-0.62, -7.59, 1.20});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {90, 0, 0}, true).matrix());
    camera.set_resolution(1024, 1024);
    camera.set_fov(tinypt::radians(49.1343));

    tinypt::Scene scene(camera, {mesh, area_light}, {sunlight});
    return scene;
}

tinypt::Scene SceneFactory::make_living_room() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/living_room/living_room.obj");

    tinypt::Camera camera;
    camera.set_location({2.2, -7.7, 1.9});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {82.6, 0, 20.7}, true).matrix());
    camera.set_resolution(1920, 1080);
    camera.set_fov(tinypt::radians(67));

    tinypt::Scene scene(camera, {mesh});
    return scene;
}

tinypt::Scene SceneFactory::make_fireplace_room() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/fireplace_room/fireplace_room.obj");

    tinypt::Camera camera;
    camera.set_location({5.0, 3.0, 1.1});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {90, 0, 113}, true).matrix());
    camera.set_resolution(1920, 1080);
    camera.set_fov(tinypt::radians(75));

    tinypt::Scene scene(camera, {mesh});
    return scene;
}

tinypt::Scene SceneFactory::make_rungholt() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/rungholt/rungholt.obj");

    auto background = tinypt::Image::open("../resource/envmap/venice_sunset_4k.hdr", false).rgb();
    auto env_rot = tinypt::Rotation::from_euler({0, 1, 2}, {0, 0, 220}, true).matrix();
    auto env = std::make_shared<tinypt::EnvTexture>(background, env_rot);

    auto sunlight = std::make_shared<tinypt::DistantLight>();
    sunlight->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {-65, 0, 0}, true).matrix());
    sunlight->set_color({1, 0.9, 0.27});
    sunlight->set_power(8);
    sunlight->set_angle(tinypt::radians(24));

    tinypt::Camera camera;
    camera.set_location({270, -209, 169});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {58, 0, 58}, true).matrix());
    camera.set_resolution(1920, 1080);
    camera.set_fov(tinypt::radians(75));

    tinypt::Scene scene(camera, {mesh}, {sunlight}, std::move(env));
    return scene;
}

tinypt::Scene SceneFactory::make_dabrovic_sponza() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/dabrovic_sponza/sponza.obj");

    auto sunlight = std::make_shared<tinypt::DistantLight>();
    sunlight->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {-10, -15, 98}, true).matrix());
    sunlight->set_power(10);
    sunlight->set_angle(tinypt::radians(20));

    auto env = std::make_shared<tinypt::EnvTexture>(tinypt::rgb_color(86, 137, 190));

    tinypt::Camera camera;
    camera.set_location({-12, 1.2, 1.4});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {105, 0, 261}, true).matrix());
    camera.set_resolution(1920, 1080);
    camera.set_fov(tinypt::radians(81.8));

    tinypt::Scene scene(camera, {mesh}, {sunlight}, std::move(env));
    return scene;
}

tinypt::Scene SceneFactory::make_salle_de_bain() {
    auto mesh = tinypt::TriangleMesh::from_obj("../resource/salle_de_bain/salle_de_bain.obj");

    tinypt::Camera camera;
    camera.set_location({7.8, -39, 15.5});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {87, 0, 21}, true).matrix());
    camera.set_resolution(1080, 1080);
    camera.set_fov(tinypt::radians(60));

    tinypt::Scene scene(camera, {mesh});
    return scene;
}

tinypt::Scene SceneFactory::make_debug() {
    auto env_map = tinypt::Image::open("../resource/envmap/venice_sunset_4k.hdr", false).rgb();
    auto env_rot = tinypt::Rotation::from_euler({0, 1, 2}, {-10, 10, 10}, true).matrix();
    auto sphere = std::make_shared<tinypt::Sphere>();
    sphere->set_location({0, 0, 0});
    sphere->set_radius(1);
    sphere->set_material(tinypt::Material(std::make_shared<tinypt::Metal>(tinypt::Vec3f(0.8, 0.8, 0.8))));
    auto env = std::make_shared<tinypt::EnvTexture>(std::move(env_map), env_rot);

    tinypt::Camera camera;
    camera.set_location({4, -6, 1});
    camera.set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {90, 0, 30}, true).matrix());
    camera.set_fov(tinypt::radians(70));
    camera.set_resolution(1920, 1080);

    tinypt::Scene scene(camera, {sphere}, {}, env);
    return scene;
}

tinypt::Scene SceneFactory::make_debug2() {
    float x1 = 1, y1 = -170, z1 = 0;
    float x2 = 99, y2 = 0, z2 = 81.6;
    float xm = (x1 + x2) / 2, ym = (y1 + y2) / 2, zm = (z1 + z2) / 2;
    float xs = x2 - x1, ys = y2 - y1, zs = z2 - z1;

    auto left = std::make_shared<tinypt::Rectangle>();
    left->set_location({x1, ym, zm});
    left->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {0, 90, 0}, true).matrix());
    left->set_dimension({zs, ys});
    left->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .25, .25))));
    auto right = std::make_shared<tinypt::Rectangle>();
    right->set_location({x2, ym, zm});
    right->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {0, 90, 0}, true).matrix());
    right->set_dimension({zs, ys});
    right->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.25, .25, .75))));
    auto back = std::make_shared<tinypt::Rectangle>();
    back->set_location({xm, y2, zm});
    back->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {90, 0, 0}, true).matrix());
    back->set_dimension({xs, zs});
    back->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .75, .75))));
    auto bottom = std::make_shared<tinypt::Rectangle>();
    bottom->set_location({xm, ym, z1});
    bottom->set_dimension({xs, ys});
    bottom->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .75, .75))));
    auto top = std::make_shared<tinypt::Rectangle>();
    top->set_location({xm, ym, z2});
    top->set_dimension({xs, ys});
    top->set_material(tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(.75, .75, .75))));
    auto mirror_sphere = std::make_shared<tinypt::Sphere>();
    mirror_sphere->set_location({27, 16.5, 47});
    mirror_sphere->set_radius(16.5);
    mirror_sphere->set_material(tinypt::Material(std::make_shared<tinypt::Metal>(tinypt::Vec3f(.999, .999, .999))));
    auto glass_sphere = std::make_shared<tinypt::Sphere>();
    glass_sphere->set_location({73, 16.5, 78});
    glass_sphere->set_radius(16.5);
    glass_sphere->set_material(
        tinypt::Material(std::make_shared<tinypt::Dielectric>(tinypt::Vec3f(.999, .999, .999), 1.5)));
    auto light = std::make_shared<tinypt::Circle>();
    light->set_radius(18);
    light->set_location({50, 81.6 - 0.01, 81.6});
    light->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {-90, 0, 0}, true).matrix());
    light->set_material(
        tinypt::Material(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(0, 0, 0)), tinypt::Vec3f(12, 12, 12)));

    auto box = tinypt::TriangleMesh::from_obj(
        "../resource/cube.obj",
        std::make_shared<tinypt::Material>(std::make_shared<tinypt::Lambertian>(tinypt::Vec3f(0, .8, 0))));
    box->set_location({20, 20, 20});
    box->set_rotation(tinypt::Rotation::from_euler({0, 1, 2}, {10, 20, 30}, true).matrix());

    // camera
    auto w = tinypt::Vec3f(0, 0.042612, 1).normalized();
    auto u = tinypt::Vec3f::UnitX();
    auto v = w.cross(u);
    tinypt::Mat3f cam_rot;
    cam_rot << u, v, w;

    tinypt::Camera camera;
    camera.set_location({50, 52, 295.6});
    camera.set_rotation(cam_rot);
    camera.set_resolution(1024, 768);
    camera.set_fov(tinypt::radians(39.32));

    tinypt::Scene scene(camera, {left, right, back, bottom, top, light, box});
    return scene;
}
