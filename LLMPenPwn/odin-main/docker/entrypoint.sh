#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d /opt/resources ]]; then
  echo "[odin-entrypoint] /opt/resources mount missing" >&2
  exit 1
fi

echo "[odin-entrypoint] Starting Jupyter Kernel Gateway (python3) on 8888..." >&2
env NO_COLOR=1 CLICOLOR=0 CLICOLOR_FORCE=0 TERM=dumb KG_DEFAULT_KERNEL_NAME=python3 \
  jupyter kernelgateway \
    --KernelGatewayApp.ip=0.0.0.0 \
    --KernelGatewayApp.port=8888 \
    --KernelGatewayApp.allow_origin='*' \
    --KernelGatewayApp.ws_ping_interval=300 &

if command -v sage >/dev/null 2>&1 && jupyter kernelspec list 2>/dev/null | grep -qi sagemath; then
  echo "[odin-entrypoint] Starting Jupyter Kernel Gateway (sagemath) on 8889..." >&2
  env NO_COLOR=1 CLICOLOR=0 CLICOLOR_FORCE=0 TERM=dumb KG_DEFAULT_KERNEL_NAME=sagemath \
    jupyter kernelgateway \
      --KernelGatewayApp.ip=0.0.0.0 \
      --KernelGatewayApp.port=8889 \
      --KernelGatewayApp.allow_origin='*' \
      --KernelGatewayApp.ws_ping_interval=300 &
else
  echo "[odin-entrypoint] Sage kernel not found; skipping 8889." >&2
fi

echo "[odin-entrypoint] Ready. Sleeping..." >&2
exec tail -f /dev/null