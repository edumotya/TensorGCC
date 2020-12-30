from setuptools import find_packages, setup

setup(
    name="tensorgcc",
    version="0.1",
    install_requires="tensorflow>=2",
    description="TensorFlow implementation of the generalized cross-correlation (GCC).",
    author="Eduardo Cuesta",
    url="https://github.com/edumotya/TensorGCC",
    packages=find_packages(),
)