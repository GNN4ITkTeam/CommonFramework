# base image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update
RUN apt-get install -y git h5utils wget vim g++

# update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

# copy and install package
COPY . .
RUN python -m pip install -e .
