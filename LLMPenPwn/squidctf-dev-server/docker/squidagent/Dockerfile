# Base Image
FROM ubuntu:22.04 AS base

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Enable i386 architecture for multilib
RUN dpkg --add-architecture i386

# Update and upgrade once in base layer
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get upgrade -y

# System Packages Stages
# Core System Tools (minimal dependencies, fast)
FROM base AS pkg-core
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    sudo curl unzip netcat-openbsd wget jq vim git \
    bsdmainutils file nasm

# Development Libraries (needed by many tools)
FROM base AS pkg-libs
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    libssl-dev libffi-dev libtbb2 libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libgmp-dev libmpfr-dev libcapstone-dev

# Crypto & Base Analysis Tools (no heavy dependencies)
FROM base AS pkg-crypto-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    p11-kit p11-kit-modules z3

# Build Tools (needs multilib setup)
FROM base AS pkg-build-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    libc6-i386 libc6-dev:i386

FROM pkg-build-base AS pkg-build
COPY --from=pkg-libs /usr/lib /usr/lib
COPY --from=pkg-libs /usr/include /usr/include
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    build-essential pkg-config gdb gdbserver \
    gcc-multilib g++-multilib

# Python Ecosystem (may need some libs)
FROM base AS pkg-python
COPY --from=pkg-libs /usr/lib /usr/lib
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    python2 python3-dev python3-pip python3-venv

# Language Runtimes (independent of each other but may need base libs)
FROM base AS pkg-runtimes
COPY --from=pkg-libs /usr/lib /usr/lib
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    ruby-full openjdk-17-jdk openjdk-17-jre \
    golang-go rustc cargo

# Debug/Analysis Tools (need build tools)
FROM pkg-build AS pkg-debug
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    ltrace strace valgrind

# QEMU (may need some build dependencies)
FROM pkg-build AS pkg-qemu
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y qemu

# Forensics Tools (may need libs)
FROM base AS pkg-forensics
COPY --from=pkg-libs /usr/lib /usr/lib
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    sleuthkit scalpel libimage-exiftool-perl \
    steghide binwalk tshark ngrep

# Security/CTF Tools Part 1 - Core (need python and libs)
FROM base AS pkg-security-core
COPY --from=pkg-python /usr /usr
COPY --from=pkg-libs /usr/lib /usr/lib
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    nmap sqlmap nikto gobuster

# Security/CTF Tools Part 2 - Java-based (need JDK)
FROM base AS pkg-security-java
COPY --from=pkg-runtimes /usr/lib/jvm /usr/lib/jvm
COPY --from=pkg-runtimes /usr/bin/java* /usr/bin/
COPY --from=pkg-runtimes /etc/java* /etc/
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y apktool

# Security/CTF Tools Part 3 - Crypto (need build tools)
FROM pkg-build AS pkg-security-crypto
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y \
    hashcat john

# SageMath (very heavy, needs python and lots of libs)
FROM base AS pkg-sagemath
COPY --from=pkg-python /usr /usr
COPY --from=pkg-libs /usr/lib /usr/lib
COPY --from=pkg-libs /usr/include /usr/include
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y sagemath

# Combine system package layers
FROM base AS system-packages

# Layer 1 - Foundation
COPY --from=pkg-core /usr /usr
COPY --from=pkg-core /var /var
COPY --from=pkg-core /etc /etc
COPY --from=pkg-libs /usr /usr
COPY --from=pkg-libs /var /var
COPY --from=pkg-crypto-base /usr /usr
COPY --from=pkg-crypto-base /var /var

# Layer 2 - Build on foundation
COPY --from=pkg-build /usr /usr
COPY --from=pkg-build /var /var
COPY --from=pkg-build /lib /lib
COPY --from=pkg-build /lib32 /lib32
COPY --from=pkg-build /lib64 /lib64
COPY --from=pkg-build /libx32 /libx32
COPY --from=pkg-python /usr /usr
COPY --from=pkg-python /var /var
COPY --from=pkg-runtimes /usr /usr
COPY --from=pkg-runtimes /var /var
COPY --from=pkg-runtimes /etc /etc

# Layer 3 - Tools depending on layer 2
COPY --from=pkg-debug /usr /usr
COPY --from=pkg-debug /var /var
COPY --from=pkg-qemu /usr /usr
COPY --from=pkg-qemu /var /var
COPY --from=pkg-forensics /usr /usr
COPY --from=pkg-forensics /var /var
COPY --from=pkg-security-core /usr /usr
COPY --from=pkg-security-core /var /var
COPY --from=pkg-security-java /usr /usr
COPY --from=pkg-security-java /var /var
COPY --from=pkg-security-crypto /usr /usr
COPY --from=pkg-security-crypto /var /var

# Layer 4 - Heavy packages
COPY --from=pkg-sagemath /usr /usr
COPY --from=pkg-sagemath /var /var
COPY --from=pkg-sagemath /opt /opt

# SecLists Download Stage
FROM base AS seclists-download

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y curl unzip

RUN curl -L https://github.com/danielmiessler/SecLists/archive/refs/heads/master.zip -o /tmp/seclists.zip && \
    unzip /tmp/seclists.zip -d /opt/ && \
    mv /opt/SecLists-master /opt/seclists && \
    rm -f /tmp/seclists.zip

# Jadx Download Stage
FROM base AS jadx-download

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y curl unzip

RUN curl -LO https://github.com/skylot/jadx/releases/download/v1.4.7/jadx-1.4.7.zip && \
    mkdir -p /opt/jadx && \
    unzip -d /opt/jadx jadx-1.4.7.zip && \
    rm -f jadx-1.4.7.zip && \
    chmod +x /opt/jadx/bin/jadx* || true

# Smuggler Cloning Stage
FROM base AS smuggler-clone

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y git

RUN git clone https://github.com/defparam/smuggler.git /opt/smuggler && \
    chmod +x /opt/smuggler/smuggler.py

# http2smugl Cloning Stage
FROM base AS http2smugl-clone

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y git

RUN git clone https://github.com/neex/http2smugl.git /opt/http2smugl && \
    cd /opt/http2smugl && \
    sed -i 's/go 1.23.4/go 1.18/' go.mod || true

# Ruby Gems Stage
FROM system-packages AS ruby-gems

RUN echo "gem: --no-document --bindir /usr/local/bin" > /etc/gemrc && \
    echo "gem: --no-document --bindir /usr/local/bin" > ~/.gemrc

# Install gems
RUN gem install one_gadget zsteg --bindir /sbin 

# Also add gem bin paths to PATH as backup
ENV PATH="/var/lib/gems/3.0.0/bin:/usr/local/bin:$PATH"
ENV PATH="/var/lib/gems/3.0.0/gems/one_gadget-1.9.0/bin:$PATH"

# MP3Stego Stage
FROM system-packages AS mp3stego-install

# Create a marker directory so COPY always has something to copy
RUN mkdir -p /tmp/mp3stego-marker

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y mp3stego || true

# Python Base Stage
FROM system-packages AS python-base

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel packaging>=23.1

# Python Core Stage
FROM python-base AS python-core

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir pyelftools || (python3 -m pip list --format=columns && false)

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir pwntools gmpy2 angr chepy openai urllib3 requests reedsolo || (python3 -m pip list --format=columns && false)

# Headless IDA Stage
FROM python-base AS python-optional

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir headless-ida || true

# PyTorch Install Stage
FROM python-base AS pytorch-install

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps torch torchvision torchaudio

# Anthropic Library Install Stage
FROM python-base AS anthropic-install

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir "anthropic==0.69.0"

# Merge System Packages and Ruby Gems
FROM system-packages AS merged-base

# Copy Ruby gems (only what exists)
COPY --from=ruby-gems /var/lib/gems /var/lib/gems

# Final Python Stage
FROM merged-base AS final-with-python

# Install Python packages sequentially on top of merged base
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel packaging>=23.1

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir pyelftools || (python3 -m pip list --format=columns && false)

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir pwntools gmpy2 angr chepy openai urllib3 requests reedsolo || (python3 -m pip list --format=columns && false)

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir headless-ida || true

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps torch torchvision torchaudio

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir "anthropic==0.69.0"

# Final Image
FROM final-with-python AS squidagent

# Try to install mp3stego if it exists in the optional stage
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y mp3stego || true

# Copy all static resources from parallel builds
COPY --from=seclists-download /opt/seclists /opt/seclists
COPY --from=jadx-download /opt/jadx /opt/jadx
COPY --from=smuggler-clone /opt/smuggler /opt/smuggler
COPY --from=http2smugl-clone /opt/http2smugl /opt/http2smugl

# Create jadx symlinks for easy access
RUN ln -sf /opt/jadx/bin/jadx /usr/local/bin/jadx && \
    ln -sf /opt/jadx/bin/jadx-gui /usr/local/bin/jadx-gui

# Create smuggler symlink
RUN ln -sf /opt/smuggler/smuggler.py /usr/local/bin/smuggler

# Build http2smugl (if it exists)
RUN cd /opt/http2smugl && \
    (go build -o http2smugl . && ln -s /opt/http2smugl/http2smugl /usr/local/bin/http2smugl) || \
    echo "Warning: http2smugl build failed, skipping..."

# Copy packaged tool
COPY tools/tool.zip /tmp/tool.zip
COPY tools/webhook.py /opt/webhook_site.py

# Copy helper scripts
COPY scripts/dogbolt /opt/dogbolt
COPY scripts/decompile_binary_with_ida.py /opt/decompile_binary_with_ida.py
COPY scripts/dump_from_offset.py /opt/dump_from_offset.py

# If sage exists, install pwntools via sage pip (optional)
RUN if command -v sage >/dev/null 2>&1; then sage --pip install pwntools || true; fi

# Force Chrome/Firefox to use system CA store
RUN ln -s -f /usr/lib/x86_64-linux-gnu/pkcs11/p11-kit-trust.so /usr/lib/x86_64-linux-gnu/nss/libnssckbi.so || true
RUN ln -s -f /usr/lib/x86_64-linux-gnu/pkcs11/p11-kit-trust.so /usr/lib/firefox/libnssckbi.so || true

# Install pwndbg non-interactively and make 'gdb' wrapper safely
# Configure symlink for gdb
RUN mkdir -p /usr/local/lib && \
    ln -s /lib64/ld-linux-x86-64.so.2 /usr/local/lib/ld-linux-x86-64.so.2

# Clean apt caches and temp files to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create runtime non-root user
ARG HOST_UID=1000
ARG USERNAME=ctfplayer
ARG USER_UID=$HOST_UID
ARG USER_GID=$USER_UID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Copy entrypoint while still root, set ownership & permissions
COPY entrypoint.sh /home/${USERNAME}/.entrypoint.sh
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.entrypoint.sh \
    && chmod +x /home/${USERNAME}/.entrypoint.sh

# Handle pwndbg
RUN git clone https://github.com/pwndbg/pwndbg.git
RUN cd /pwndbg && /pwndbg/setup.sh
RUN  ln -s /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0 /usr/lib/x86_64-linux-gnu/libopenblas.so.0
RUN ln -s /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 /usr/lib/x86_64-linux-gnu/libblas.so.3
RUN apt install nodejs -y
RUN echo 'export PATH="/var/lib/gems/3.0.0/bin:$PATH"' >> /home/ctfplayer/.bashrc && \
    echo 'export PATH="/var/lib/gems/3.0.0/bin:$PATH"' >> /home/ctfplayer/.profile
RUN echo 'export PATH="/var/lib/gems/3.0.0/bin:$PATH"' >> /etc/profile && \
    echo 'export PATH="/var/lib/gems/3.0.0/bin:$PATH"' >> /etc/bash.bashrc
RUN pip3 install scapy
RUN apt-get update && \
    apt-get install -y wget apt-transport-https && \
    wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-sdk-8.0
RUN dotnet tool install --tool-path /usr/local/bin ilspycmd    
# Switch to non-root user for runtime
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Final default command
CMD ["bash", "/home/ctfplayer/.entrypoint.sh"]
