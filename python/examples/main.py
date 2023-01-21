import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

import tinypt

tinypt_root = Path(__file__).resolve().parent.parent.parent


def make_scene(name):
    world_makers = {
        "cornell_sphere": make_cornell_sphere,
        "cornell_box": make_cornell_box,
        "breakfast_room": make_breakfast_room,
        "living_room": make_living_room,
        "fireplace_room": make_fireplace_room,
        "rungholt": make_rungholt,
        "dabrovic_sponza": make_dabrovic_sponza,
        "salle_de_bain": make_salle_de_bain,
    }

    if name not in world_makers:
        raise ValueError(f"invalid scene name: {name}")

    return world_makers[name]()


def make_cornell_sphere():
    x1, y1, z1 = 1, -170, 0
    x2, y2, z2 = 99, 0, 81.6
    xm, ym, zm = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
    xs, ys, zs = x2 - x1, y2 - y1, z2 - z1

    yz_rot = R.from_euler("y", 90, degrees=True).as_matrix()
    left = tinypt.Rectangle(
        location=(x1, ym, zm),
        rotation=yz_rot,
        dimension=(zs, ys),
        surface=tinypt.Lambertian(texture=(0.75, 0.25, 0.25)),
    )
    right = tinypt.Rectangle(
        location=(x2, ym, zm),
        rotation=yz_rot,
        dimension=(zs, ys),
        surface=tinypt.Lambertian(texture=(0.25, 0.25, 0.75)),
    )
    xz_rot = R.from_euler("x", 90, degrees=True).as_matrix()
    back = tinypt.Rectangle(
        location=(xm, y2, zm),
        rotation=xz_rot,
        dimension=(xs, zs),
        surface=tinypt.Lambertian(texture=(0.75, 0.75, 0.75)),
    )
    bottom = tinypt.Rectangle(
        location=(xm, ym, z1),
        dimension=(xs, ys),
        surface=tinypt.Lambertian(texture=(0.75, 0.75, 0.75)),
    )
    top = tinypt.Rectangle(
        location=(xm, ym, z2),
        dimension=(xs, ys),
        surface=tinypt.Lambertian(texture=(0.75, 0.75, 0.75)),
    )
    mirror = tinypt.Sphere(
        radius=16.5, location=(27, -47, 16.5), surface=tinypt.Metal(texture=(1, 1, 1))
    )
    glass = tinypt.Sphere(
        location=(73, -78, 16.5),
        radius=16.5,
        surface=tinypt.Dielectric(texture=(1, 1, 1), ior=1.5),
    )
    light = tinypt.Circle(
        radius=18, location=(50, -81.6, 81.6 - 0.01), emission=(12, 12, 12)
    )
    objects = [left, right, back, bottom, top, mirror, glass, light]

    cam_w = np.array([0, -0.999093, 0.042573], dtype=np.float32)
    cam_u = np.array([1, 0, 0], dtype=np.float32)
    cam_v = np.cross(cam_w, cam_u)
    cam_rot = np.vstack((cam_u, cam_v, cam_w)).T
    camera = tinypt.Camera(
        location=(50, -295.6, 52),
        rotation=cam_rot,
        resolution=(1024, 768),
        fov=math.radians(39.32),
    )

    scene = tinypt.Scene(camera=camera, objects=objects)
    return scene


def make_cornell_box():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/CornellBox/CornellBox-Original.obj")
    )
    cam_rot = R.from_euler("x", 90, degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(0, -3.5, 1),
        rotation=cam_rot,
        resolution=(1024, 1024),
        fov=math.radians(42),
    )
    scene = tinypt.Scene(camera=camera, objects=[mesh])
    return scene


def make_breakfast_room():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/breakfast_room/breakfast_room.obj")
    )
    area_light = tinypt.Rectangle(
        location=(-0.596747, 1.83138, 7.02496),
        dimension=(11.3233, 6),
        emission=(2, 2, 2),
    )
    sunlight_rot = R.from_euler(
        "xyz", [50.3857, -0.883569, 74.8117], degrees=True
    ).as_matrix()
    sunlight = tinypt.DistantLight(
        rotation=sunlight_rot, power=8, angle=math.radians(2.2)
    )

    cam_rot = R.from_euler("x", 90, degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(-0.62, -7.59, 1.20),
        rotation=cam_rot,
        resolution=(1024, 1024),
        fov=math.radians(49.1343),
    )
    scene = tinypt.Scene(camera=camera, objects=[mesh, area_light], lights=[sunlight])
    return scene


def make_living_room():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/living_room/living_room.obj")
    )
    cam_rot = R.from_euler("xyz", (82.6, 0, 20.7), degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(2.2, -7.7, 1.9),
        rotation=cam_rot,
        resolution=(1920, 1080),
        fov=math.radians(67),
    )
    scene = tinypt.Scene(camera=camera, objects=[mesh])
    return scene


def make_fireplace_room():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/fireplace_room/fireplace_room.obj")
    )
    cam_rot = R.from_euler("xyz", (90, 0, 113), degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(5.0, 3.0, 1.1),
        rotation=cam_rot,
        resolution=(1920, 1080),
        fov=math.radians(75),
    )
    scene = tinypt.Scene(camera=camera, objects=[mesh])
    return scene


def make_rungholt():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/rungholt/rungholt.obj")
    )

    background = cv2.imread(
        str(tinypt_root / "resource/envmap/venice_sunset_4k.hdr"), cv2.IMREAD_UNCHANGED
    )
    tonemap = cv2.createTonemapDrago(gamma=2.2)
    background = (tonemap.process(background) * 4).clip(0, 1)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    env_rot = R.from_euler("z", 220, degrees=True).as_matrix()
    background = tinypt.EnvTexture(texture=background, rotation=env_rot)

    sunlight_rot = R.from_euler("x", -65, degrees=True).as_matrix()
    sunlight = tinypt.DistantLight(
        rotation=sunlight_rot, color=(1, 0.9, 0.27), power=8, angle=math.radians(24)
    )

    cam_rot = R.from_euler("xyz", (58, 0, 58), degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(270, -209, 169),
        rotation=cam_rot,
        resolution=(1920, 1080),
        fov=math.radians(75),
    )
    scene = tinypt.Scene(
        camera=camera, objects=[mesh], lights=[sunlight], background=background
    )
    return scene


def make_dabrovic_sponza():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/dabrovic_sponza/sponza.obj")
    )
    sunlight_rot = R.from_euler("xyz", [-10, -15, 98], degrees=True).as_matrix()
    sunlight = tinypt.DistantLight(
        rotation=sunlight_rot, power=10, angle=math.radians(20)
    )
    cam_rot = R.from_euler("xyz", (105, 0, 261), degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(-12, 1.2, 1.4),
        rotation=cam_rot,
        resolution=(1920, 1080),
        fov=math.radians(81.8),
    )
    scene = tinypt.Scene(
        camera=camera,
        objects=[mesh],
        lights=[sunlight],
        background=tinypt.EnvTexture(texture="#5689BE"),
    )
    return scene


def make_salle_de_bain():
    mesh = tinypt.TriangleMesh.from_obj(
        str(tinypt_root / "resource/salle_de_bain/salle_de_bain.obj")
    )
    cam_rot = R.from_euler("xyz", (87, 0, 21), degrees=True).as_matrix()
    camera = tinypt.Camera(
        location=(7.8, -39, 15.5),
        rotation=cam_rot,
        resolution=(1080, 1080),
        fov=math.radians(60),
    )
    scene = tinypt.Scene(camera=camera, objects=[mesh])
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="scene.png")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--scene",
        type=str,
        default="cornell_sphere",
        choices=[
            "cornell_sphere",
            "cornell_box",
            "breakfast_room",
            "living_room",
            "fireplace_room",
            "rungholt",
            "dabrovic_sponza",
            "salle_de_bain",
        ],
    )
    args = parser.parse_args()

    scene = make_scene(args.scene).to(args.device)

    engine = tinypt.PathTracer(args.device)
    start = time.time()
    image = engine.render(scene, args.num_samples).numpy()
    print(f"[Path Tracing] ({args.num_samples} spp) Time: {time.time() - start:.2f}s")
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(args.save_path)


if __name__ == "__main__":
    main()
