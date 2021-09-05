# Use nvidia/cudagl rather than nvidia/cuda 
# https://www.ybliu.com/2020/06/docker-libGL-failed-swrast.html

FROM nvidia/cudagl:10.1-devel-ubuntu16.04

ARG nvidia_cudnn_version=7.6.4.38-1+cuda10.1

ENV NVIDIA_CUDNN_VERSION=${nvidia_cudnn_version}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=${NVIDIA_CUDNN_VERSION} \
    libcudnn7-dev=${NVIDIA_CUDNN_VERSION} \
    && apt-mark hold libcudnn7 \
    && rm -rf /var/lib/apt/lists/


RUN apt-get update && apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get -y install python3.6 python3.6-dev 

# since repo deadsnakes does not have pip we need to install default pip and upgrade it with python3.6
RUN apt-get install python3-pip -y
RUN python3.6 -m pip install --upgrade pip


# install matplotlib dependency
RUN apt-get install -y python3.6-tk

RUN echo 'alias python3=python3.6' >> ~/.bashrc

# install ros-kinetic
RUN apt-get update && apt-get install -y lsb-release && apt-get clean all

ENV UBUNTU_RELEASE=xenial
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $UBUNTU_RELEASE main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

# Install ROS bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    python-rosinstall-generator \
    python-wstool

RUN rosdep init && rosdep update

# install requirements
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app

RUN pip install -r requirements.txt
WORKDIR /




