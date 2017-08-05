FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
	python \
        python3-dev \
        python3-pip \
        libcupti-dev \ 
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
	tcl-dev \
	tk-dev \	
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# upgrade pip install setuptools
RUN pip3 install --upgrade pip setuptools

# Install scipy packages
RUN pip3 install --no-cache-dir numpy \
        scipy \
        opencv-python \
        matplotlib \
        ipython \
        jupyter \
        pandas \
        sympy \
        nose

RUN pip3 install --no-cache-dir tensorflow-gpu keras



# workdir
RUN mkdir /workdir
WORKDIR "/workdir"

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# TensorBoard
EXPOSE 6006
# Flask Server
EXPOSE 4567
