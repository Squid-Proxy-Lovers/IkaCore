#!/usr/bin/env bash
networkname="ctfnet"
network_exists=$(docker network ls | grep $networkname)
# Create network
if [ -z "$network_exists" ]; then
    docker network create ctfnet
else
    echo "Network ${networkname} already exists, skip!"
fi

BASE=$(dirname $0)
echo "Base directory: $BASE"
echo "Current directory: $(pwd)"

# Build main docker image with no cache
echo "Building main CTF environment image (ctfenv:squidagent) with --no-cache..."
cd $BASE/docker/squidagent && docker build --no-cache --progress=plain -t ctfenv:squidagent .

cd ../..

# Always build IDA docker image with no cache
echo "Building IDA Pro environment image (ctfenv:idadocker) with --no-cache..."
echo "Changing to: $BASE/docker/idadocker"
cd $BASE/docker/idadocker && docker build --no-cache --progress=plain -t ctfenv:idadocker .

cd -
echo "Installing python package"
pip install --break-system-packages -r requirements.txt

echo "Setup complete! Both Docker images are ready (built with --no-cache):"
echo "  - ctfenv:squidagent (main CTF environment)"
echo "  - ctfenv:idadocker (IDA Pro environment)"