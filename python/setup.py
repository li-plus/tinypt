import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# reference: https://github.com/pybind/cmake_example
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve().parent)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.getenv("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DTINYPT_ENABLE_PYBIND=ON",
            f"-DTINYPT_ENABLE_TEST=OFF",
            f"-DTINYPT_ENABLE_EXAMPLES=OFF",
        ]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_temp, check=True)


setup(
    name="tinypt",
    version="0.0.1",
    author="Jiahao Li",
    author_email="liplus17@163.com",
    maintainer="Jiahao Li",
    maintainer_email="liplus17@163.com",
    url="https://github.com/li-plus/tinypt",
    description="A tiny path tracing renderer",
    long_description="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords=["path tracing", "ray tracing", "rendering", "computer graphics"],
    license="MIT",
    python_requires=">=3.7",
    ext_modules=[CMakeExtension("tinypt._C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=[
        "numpy",
        "Pillow",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ]
    },
)
