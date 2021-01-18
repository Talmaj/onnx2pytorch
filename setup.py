# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""


PACKAGES = find_packages(exclude=("tests.*", "tests"))

setup(
    name="onnx2pytorch",
    version="0.1.0",
    description="Library to transform onnx model to pytorch.",
    license="apache-2.0",
    author="Talmaj Marinc",
    packages=PACKAGES,
    install_requires=["torch>=1.4.0", "onnx>=1.6.0"],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
