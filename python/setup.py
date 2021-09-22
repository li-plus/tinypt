import setuptools
from tinypt.version import VERSION

setuptools.setup(
    name='tinypt',
    version=VERSION,
    author='Jiahao Li',
    author_email='liplus17@163.com',
    maintainer='Jiahao Li',
    maintainer_email='liplus17@163.com',
    url='https://github.com/li-plus/tinypt',
    description='A tiny path tracing renderer',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords=['path tracing', 'ray tracing', 'rendering', 'computer graphics'],
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'Pillow',
    ],
    extras_require={
        'dev': [
            'pytest',
        ]
    },
    include_package_data=True,
)
