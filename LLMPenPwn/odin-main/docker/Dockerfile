FROM ubuntu:22.04 AS common

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash \
       coreutils \
       git \
       curl \
       python3 \
       python3-pip \
       python3-venv \
       grep \
       ripgrep \
       jq \
       wget \
       ca-certificates \
       file \
       python-is-python3 \
       tree \
       binutils \
       unzip \
       xxd \
       socat \
       openjdk-21-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
      bandit \
      semgrep \
      jupyter_kernel_gateway \
      jupyter_client \
      notebook \
      requests \
      numpy==1.26.4 \
      pillow \
      scikit-learn \
      matplotlib \
      scipy \
      z3-solver \
      pandas \
      pycryptodome \
      flask \
      pwntools \
      fastecdsa \
      pyelftools

ENV GHIDRA_HOME=/opt/ghidra

RUN curl -L "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.4.2_build/ghidra_11.4.2_PUBLIC_20250826.zip" \
    -o /tmp/ghidra.zip && \
    unzip /tmp/ghidra.zip -d /opt && \
    mv /opt/ghidra_11.4.2_PUBLIC $GHIDRA_HOME && \
    rm /tmp/ghidra.zip

RUN mkdir -p $GHIDRA_HOME/Decompile
COPY resources/DecompilerExplorer.java $GHIDRA_HOME/Decompile/DecompilerExplorer.java

RUN mkdir /working
WORKDIR /working

COPY entrypoint.sh /usr/local/bin/odin-entrypoint.sh
RUN chmod +x /usr/local/bin/odin-entrypoint.sh

RUN useradd -m -s /bin/bash odin
RUN chown -R odin:odin /working

ENV EG_WS_PING_INTERVAL_SECS=12000

# Python-only image
FROM common AS base
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/odin-entrypoint.sh"]

# SageMath-enabled image
FROM common AS sage

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       sagemath \
       sagemath-jupyter \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN sage -python -m sage.repl.ipython_kernel.install

RUN sage -pip install requests \
                      numpy \
                      pillow \
                      scikit-learn \
                      matplotlib \
                      scipy \
                      z3-solver \
                      pandas \
                      pycryptodome \
                      flask \
                      pwntools \
                      fastecdsa \
                      pyelftools

RUN mkdir /installer
RUN mkdir /resources
RUN (cd /installer && git clone https://github.com/maple3142/lll_cvp && cd lll_cvp && cp lll_cvp.py /resources && sage -pip install -e .)


RUN printf '%s\n' \
  "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe" \
  "deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe" \
  "deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe" \
  "deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe" \
  > /etc/apt/sources.list \
 && apt-get update

RUN apt-get install -y --no-install-recommends libgmp-dev libmpfr-dev fplll-tools libfplll-dev libeigen3-dev libopenblas-dev


RUN (cd /installer && git clone https://github.com/keeganryan/flatter && cd flatter && mkdir build && cd ./build && cmake .. && make && make install && ldconfig)

RUN (cd /resources && git clone https://github.com/kionactf/coppersmith)

EXPOSE 8888 8889
ENTRYPOINT ["/usr/local/bin/odin-entrypoint.sh"]
