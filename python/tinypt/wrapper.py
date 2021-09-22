import warnings
import numpy as np
from PIL import ImageColor
from tinypt import _C


def _parse_color(color):
    if isinstance(color, str):
        if color.startswith('#'):
            color = ImageColor.getrgb(color)
            color = tuple((x / 255.) ** 2.2 for x in color)
        else:
            raise RuntimeError('not implemented')

    return _parse_vec3f(color)


def _parse_vec2f(v):
    arr = np.asarray(v, dtype=np.float32)
    if arr.shape != (2,):
        raise RuntimeError(f'Invalid Vec2f {v}')
    return arr


def _parse_vec3f(v):
    arr = np.array(v, dtype=np.float32)
    if arr.shape != (3,):
        raise RuntimeError(f'Invalid Vec3f {v}')
    return arr


def _parse_rgb_texture(texture):
    if texture is None:
        texture = RGBTexture()
    elif isinstance(texture, RGBTexture):
        pass
    elif isinstance(texture, np.ndarray) and texture.ndim == 3:
        texture = RGBTexture(image=texture)
    else:
        color = _parse_color(texture)
        texture = RGBTexture(color=color)
    return texture


class Object3d(object):
    pass


class PathTracer(object):
    def __init__(self, device='cpu') -> None:
        self._obj = _C.PathTracer(device)

    def render(self, scene, num_samples):
        return self._obj.render(scene._obj, num_samples)


class Scene(object):
    def __init__(self, camera, objects=[], lights=None, background=None, obj=None) -> None:
        if obj is not None:
            self._obj = obj
            return
        if background is None:
            background = EnvTexture(texture=(0, 0, 0))
        native_objs = [x._obj for x in objects]
        if lights is None:
            lights = []
        native_lights = [x._obj for x in lights]
        self._obj = _C.Scene(camera._obj, native_objs,
                             native_lights, background._obj)

    def to(self, device):
        return Scene(camera=None, obj=self._obj.to(device))


class Camera(object):
    def __init__(self, location, rotation, resolution, fov) -> None:
        self._obj = _C.Camera()
        self._obj.location = _parse_vec3f(location)
        self._obj.rotation = rotation
        self._obj.width, self._obj.height = resolution
        self._obj.fov = fov

    @property
    def location(self):
        return self._obj.location

    @property
    def rotation(self):
        return self._obj.rotation

    @property
    def width(self):
        return self._obj.width

    @property
    def height(self):
        return self._obj.height


class Sphere(Object3d):
    def __init__(self, radius, location, surface=None, emission=None, alpha=None, bump=None) -> None:
        self._obj = _C.Sphere()
        self._obj.radius = float(radius)
        self._obj.location = _parse_vec3f(location)
        self._obj.material = Material(surface, emission, alpha, bump)._obj

    @property
    def radius(self):
        return self._obj.radius

    @property
    def location(self):
        return self._obj.location


class Rectangle(Object3d):
    def __init__(self, location=None, rotation=None, dimension=None,
                 surface=None, emission=None, alpha=None, bump=None) -> None:
        self._obj = _C.Rectangle()
        if location is not None:
            self._obj.location = _parse_vec3f(location)
        if rotation is not None:
            self._obj.rotation = rotation
        if dimension is not None:
            self._obj.dimension = _parse_vec2f(dimension)

        self._obj.material = Material(surface, emission, alpha, bump)._obj

    @property
    def location(self):
        return self._obj.location

    @property
    def rotation(self):
        return self._obj.rotation

    @property
    def dimension(self):
        return self._obj.dimension

    @property
    def material(self):
        return self._obj.material


class Circle(Object3d):
    def __init__(self, radius=1, location=None, rotation=None, surface=None,
                 emission=None, alpha=None, bump=None) -> None:
        self._obj = _C.Circle()
        self._obj.radius = radius
        if location is None:
            location = (0, 0, 0)
        self._obj.location = _parse_vec3f(location)
        if rotation is not None:
            self._obj.rotation = rotation
        self._obj.material = Material(surface, emission, alpha, bump)._obj


class TriangleMesh(Object3d):
    @staticmethod
    def from_obj(path):
        mesh = TriangleMesh()
        mesh._obj = _C.TriangleMesh.from_obj(path, None)
        return mesh


class BXDF(object):
    pass


class Lambertian(BXDF):
    def __init__(self, texture=None) -> None:
        texture = _parse_rgb_texture(texture)
        self._obj = _C.Lambertian(texture._obj)


class Metal(BXDF):
    def __init__(self, texture=None) -> None:
        texture = _parse_rgb_texture(texture)
        self._obj = _C.Metal(texture._obj)


class Dielectric(BXDF):
    def __init__(self, texture=None, ior=1.5) -> None:
        texture = _parse_rgb_texture(texture)
        self._obj = _C.Dielectric(texture._obj, ior)


class RGBTexture(object):
    def __init__(self, color=None, image=None) -> None:
        self._obj = _C.RGBTexture()
        if image is not None:
            if color is not None:
                warnings.warn(
                    'both color and image are specified, using image as texture')
            self._obj.map = _C.Image(image)
        elif color is not None:
            self._obj.value = _parse_color(color)


class AlphaTexture(object):
    def __init__(self) -> None:
        self._obj = _C.AlphaTexture()


class BumpTexture(object):
    def __init__(self) -> None:
        self._obj = _C.BumpTexture()


class EnvTexture(object):
    def __init__(self, texture, rotation=None) -> None:
        texture = _parse_rgb_texture(texture)
        self._obj = _C.EnvTexture(texture._obj)
        if rotation is not None:
            self._obj.rotation = rotation


class Material(object):
    def __init__(self, surface=None, emission=None, alpha=None, bump=None) -> None:
        if surface is None:
            surface = Lambertian(texture=(0, 0, 0))
        self._obj = _C.Material(surface._obj)
        if emission is not None:
            emission = _parse_rgb_texture(emission)
            self._obj.emission_texture = emission._obj
        if alpha is not None:
            self._obj.alpha_texture = alpha._obj
        if bump is not None:
            self._obj.bump_texture = bump._obj


class DistantLight(object):
    def __init__(self, rotation=None, color=(1, 1, 1), power=1, angle=0) -> None:
        self._obj = _C.DistantLight()
        if rotation is not None:
            self._obj.rotation = rotation
        self._obj.color = color
        self._obj.power = power
        self._obj.angle = angle


class EnvLight(object):
    def __init__(self, env=None) -> None:
        if env is not None:
            self._obj = _C.EnvLight(env)
        else:
            self._obj = _C.EnvLight()
