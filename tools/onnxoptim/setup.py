#!/usr/bin/env python3
from setuptools import setup

setup(
    name='onnxoptim',
    version='0.0.1',
    install_requires=[
        'numpy',
        'onnx',
    ],
    package_dir={'': 'onnxoptim'},
)
