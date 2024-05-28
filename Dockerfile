# base image
FROM docexoty/exatrkx:cuda12-pytorch2.1

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update
RUN apt-get install -y git h5utils wget vim g++

# copy and install package
COPY . .
RUN source activate gnn4itk && python -m pip install -e .
