#!/usr/bin/env bash
# First start: fetch the first-run model set into the /app/models volume.
# Already-installed rows are skipped, so this is a no-op once the volume is filled.
# ANIMA_SKIP_DOWNLOAD=1 skips it (e.g. a host models dir mounted read-only).
#
# PUBLIC_KEY set (RunPod injects it): authorize it for root and start sshd, so
# the pod's public 22/tcp takes ssh / rsync. With no TTY (a RunPod pod) the
# default `bash` would exit at once and restart the container; the container
# then idles on `sleep infinity` instead.
set -e

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    # sshd sessions start from a bare env: hand them the image's PATH / ANIMA_* /
    # HF_* (and RunPod's RUNPOD_*) through ~/.ssh/environment.
    env | grep -E '^(PATH|ANIMA_|HF_|UV_|CUDA|NVIDIA_|RUNPOD_)' > /root/.ssh/environment
    echo "PermitUserEnvironment yes" > /etc/ssh/sshd_config.d/anima.conf
    ssh-keygen -A >/dev/null
    /usr/sbin/sshd
fi

if [ -z "${ANIMA_SKIP_DOWNLOAD:-}" ] \
   && [ ! -f /app/models/diffusion_models/anima-base-v1.0.safetensors ]; then
    echo "==> models not found in /app/models — running download-models"
    python tasks.py download-models
fi

if [ "$*" = "bash" ] && [ ! -t 0 ]; then
    exec sleep infinity
fi
exec "$@"
