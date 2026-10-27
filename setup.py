# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except IOError:
    long_description = ""

# Extract version. Cannot import directly because of import error.
root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, "onnx2pytorch/__init__.py"), "r") as f:
    for line in f.readlines():
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
            break

PACKAGES = find_packages(exclude=("tests.*", "tests"))

setup(
    name="onnx2pytorch",
    version=version,
    description="onnx2pytorch converts ONNX models to PyTorch modules.",
    license="apache-2.0",
    author="Talmaj Marinc",
    packages=PACKAGES,
    install_requires=["torch>=2.0.0", "onnx>=1.16.0", "torchvision>=0.15.0"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ToriML/onnx2pytorch",
    project_urls={
        "Source": "https://github.com/Talmaj/onnx2pytorch",
        "Issues": "https://github.com/Talmaj/onnx2pytorch/issues",
        "Homepage": "https://github.com/Talmaj/onnx2pytorch",
    },
    keywords=[
        "onnx",
        "pytorch",
        "model conversion",
        "onnx-to-pytorch",
        "converter",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
)
