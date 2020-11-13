FROM nvcr.io/nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && \
  apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip git
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN python -m pip install pip --upgrade

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64"
RUN pip install torch==0.4.1 tensorflow-gpu==1.5.0 librosa matplotlib


RUN apt-get install -y --no-install-recommends \
    libsndfile1

WORKDIR /work
RUN git clone https://github.com/r9y9/wavenet_vocoder.git -b v0.0.4
WORKDIR /work/wavenet_vocoder
RUN pip install -e ".[train]"
RUN pip install lws
