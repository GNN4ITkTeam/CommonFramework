import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="acorn",
    version="0.0.1",
    description="A common framework for GNN4ITK",
    author="GNN4ITK Team",
    packages=find_packages(include=["acorn", "acorn.*"]),
    entry_points={
        "console_scripts": [
            "g4i-train = acorn.core.train_stage:main",
            "g4i-infer = acorn.core.infer_stage:main",
            "g4i-eval = acorn.core.eval_stage:main",
            "acorn = acorn.core.entrypoint_stage:cli",
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
    url="https://gitlab.cern.ch/gnn4itkteam/acorn",
)
