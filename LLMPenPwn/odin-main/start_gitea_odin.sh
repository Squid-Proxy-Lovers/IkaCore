#!/usr/bin/env bash

cd docker && ./build-odinbase.sh && cd ..
pip install -e .
cd integration/gitea && pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8888