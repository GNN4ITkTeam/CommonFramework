# base image
FROM docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update && apt-get install -y git h5utils wget xrootd-client xrootd-server python3-xrootd zip unzip

COPY . ./acorn
RUN pip install -r acorn/requirements.txt && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
    cd acorn && pip install -e . && \
    pip cache purge --no-input
