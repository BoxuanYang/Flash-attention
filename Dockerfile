FROM ubuntu:22.04
COPY . /Flash-attention/
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    gcc \
    gdb \
    g++ \
    python3-pip \
    make \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /Flash-attention
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.5
RUN pip3 install torch


# 运行这个docker，只需要执行如下命令：
#docker build -t boxuan-flash-attention .
#docker run -it boxuan-flash-attention


# apt-get update
# apt-get install libcunit1 libcunit1-doc libcunit1-dev python3-dev  




