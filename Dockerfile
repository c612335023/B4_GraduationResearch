#Local

FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

COPY requirements.txt /tmp/

RUN apt update && apt install -y \
git \
wget \
curl \
libsndfile1 \
fluidsynth \
python3 \
python3-pip \
&& apt autoremove -y && apt clean && rm -rf /usr/local/src/* \
&& pip install --no-cache-dir -U pip setuptools wheel \
&& pip install --no-cache-dir -r /tmp/requirements.txt