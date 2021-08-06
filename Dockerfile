FROM nvidia/cuda:10.1-runtime-ubuntu16.04

RUN apt-get update

RUN apt-get update && apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get -y install python3.6 

# since repo deadsnakes does not have pip we need to install default pip and upgrade it with python3.6
RUN apt-get install python3-pip -y
RUN python3.6 -m pip install --upgrade pip

# install matplotlib dependency
RUN apt-get install -y python3.6-tk

RUN echo 'alias python3=python3.6' >> ~/.bashrc

# install ros-kinetic
RUN apt-get update && apt-get install -y lsb-release && apt-get clean all

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update

RUN apt install curl -y

RUN apt-get install ros-kinetic-desktop -y

RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

RUN apt -y install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential -y

RUN apt -y install python-rosdep -y

RUN rosdep init

RUN rosdep fix-permissions && rosdep update

RUN apt-get install -y ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control 

RUN pip install tensorflow-gpu==2.2.0

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN pip install -r requirements.txt

WORKDIR /

RUN apt-get -y install xauth

#remove the library provided by mesa
RUN rm /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1
RUN rm /usr/lib/x86_64-linux-gnu/mesa/libGL.so




