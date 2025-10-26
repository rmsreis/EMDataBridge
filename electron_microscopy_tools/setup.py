#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EMDataBridge",
    version="0.0.1",
    author="Roberto dos Reis",
    author_email="roberto.reis@northwestern.edu",
    description="A bridge between electron microscopy data formats with standardization, translation, and discovery capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rmsreis/EMDataBridge",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "EMDataBridge": ["data/*", "templates/*"]
    },
    extras_require={
        "dev": ["pytest>=7.0", "flake8>=6.0", "pre-commit>=3.0", "black>=23.7"],
        "ml": ["transformers>=4.18.0", "hyperspy>=1.7.0"],
        "io": ["pyem>=0.5.0", "mrcfile>=1.3.0", "h5py>=3.6.0", "zarr>=2.11.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "jsonschema>=4.4.0",
        "xmltodict>=0.13.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "networkx>=2.6",
        "matplotlib>=3.3",
    ],
    entry_points={
        "console_scripts": [
            "emdb=EMDataBridge.cli:main",
        ],
    },
)
