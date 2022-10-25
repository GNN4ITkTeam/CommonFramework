import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

dependencies = [
    "tqdm",
    "ipykernel",
    "atlasify",
]

setup(
    name="gnn4itk-cf",
    version="0.0.1",
    description="A common framework for GNN4ITK",
    author="GNN4ITK Team",
    install_requires=dependencies,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "g4i-train = src.train_stage:main",
            "g4i-infer = src.infer_stage:main",
            "g4i-eval = src.eval_stage:main",
        ]
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    keywords=[
        "ATLAS",
        "track reconstruction",
        "graph networks",
        "machine learning",
    ],
    url="https://gitlab.cern.ch/gnn4itkteam/commonframework",
)
