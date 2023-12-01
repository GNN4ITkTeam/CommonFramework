import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


dependencies = [
    "tqdm",
    "ipykernel",
    "atlasify",
    "trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3",
    "networkx",
    "seaborn",
    "pyyaml",
    "click",
    "pytorch-lightning===1.8.6",
    "pytest",
    "pytest-cov",
    "torch-geometric==2.2.0",
    "uproot",
    "class-resolver",
]

setup(
    name="acorn",
    version="0.0.1",
    description="A common framework for GNN4ITK",
    author="GNN4ITK Team",
    install_requires=dependencies,
    packages=find_packages(include=["acorn", "acorn.*"]),
    entry_points={
        "console_scripts": [
            "g4i-train = acorn.core.entrypoint_stage_cf:main",
            "g4i-infer = acorn.core.entrypoint_stage_cf:main",
            "g4i-eval = acorn.core.entrypoint_stage_cf:main",
            "acorn = acorn.core.entrypoint_stage:main",
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
