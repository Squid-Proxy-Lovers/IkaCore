#!/usr/bin/env bash

python -u -m http.server &> /tmp/ctf_web.log &
headless-ida-server /opt/ida-pro/idat localhost 1337 &

sleep infinity
