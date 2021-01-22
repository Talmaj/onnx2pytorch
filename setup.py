# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except IOError:
    long_description = ""


PACKAGES = find_packages(exclude=("tests.*", "tests"))

setup(
    name="onnx2pytorch",
    version="0.1.1",
    description="Library to transform onnx model to pytorch.",
    license="apache-2.0",
    author="Talmaj Marinc",
    packages=PACKAGES,
    install_requires=["torch>=1.4.0", "onnx>=1.6.0"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ToriML/onnx2pytorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
