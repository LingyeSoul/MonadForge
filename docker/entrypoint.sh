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

# nproc reports the host's cores, not the container's CPU quota (cgroup v2
# cpu.max: "<quota> <period>"), so thread pools sized from it oversubscribe.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null || quota=max
    if [ "$quota" != "max" ]; then
        cpus=$(( quota / period - 1 ))
    else
        cpus=$(nproc)
    fi
    export OMP_NUM_THREADS=$(( cpus > 1 ? cpus : 1 ))
fi

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    # sshd sessions start from a bare env, and pam_env sets PATH from
    # /etc/environment after ~/.ssh/environment is read — so the image's PATH /
    # ANIMA_* / HF_* (and RunPod's RUNPOD_*) go into /etc/environment.
    env | grep -E '^(PATH|ANIMA_|HF_|UV_|CUDA|NVIDIA_|RUNPOD_|OMP_)' \
        | sed -E 's/^([^=]+)=(.*)$/\1="\2"/' > /etc/environment
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
