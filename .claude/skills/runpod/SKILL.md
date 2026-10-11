---
name: runpod
description: Operating a RunPod pod for a GPU run — settling the run and its input list before create, pushing code and inputs (rsync --mkpath, parallel streams, measured throughput), system packages and thread counts on the pod, launching detached, waiting with a background until loop, pulling back results and terminating. Load before renting, pushing to, running work on, or terminating a RunPod pod.
---

# RunPod pods

`make runpod ARGS="gpus|ls|create|wait|ssh|stop|terminate …"` (`anima_daemon/runpod/`)
rents a pod by hand; it is not a daemon job backend. What the CLI and the image do
(API split, key lookup, `create` defaults and flags, the entrypoint):
`anima_daemon/README.md` § RunPod pods.

Billing starts at `create`, so ask the user before running it, and do everything that
needs no GPU locally first.

## Order of a run

1. **Settle the run before `create`.** Write the config (`.toml`), any set or asset it
   reads, and the code change, and decide what the pod builds and what comes back.
2. **List the inputs**: everything the pod reads that the image lacks — gitignored
   assets (fonts, packs the catalog does not fetch), the `output/` trees a build reads,
   untracked files such as a new config. When a builder's reads are not obvious, run
   it once locally at a small fraction under a `sys.addaudithook` hook on `open` and
   keep what it opened under the repo. Paths recorded absolute on the local box (in a
   manifest or pool file) must resolve under `/app` on the pod; the reader re-anchors
   them.
3. **`create`, then push code and inputs together** (§ Moving data). Then `uv sync` on
   the pod (a no-op when the lock matches the image).
4. **Check the pod's system packages and thread count** (§ On the pod).
5. **Launch detached** (§ Launching).
6. **Wait with one background Bash `until` loop** (§ Waiting).
7. **Pull back the results only** — weights, logs, build manifests (tens of MB) — run
   the reads locally, then `make runpod ARGS="terminate <pod_id> --yes"`. A stopped
   pod's container disk is wiped anyway, so a finished pod is terminated, not stopped.

ssh answered 159 s (`CA-MTL-1`) and 233 s (`EU-SE-1`) after `create` (2026-10-11), most
of it the image pull.

## Moving data

`create --wait` prints `ssh root@<ip> -p <port>`; `ls` shows the same. Every rsync
below uses

```bash
SSH="ssh -p <port> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
rsync -a --mkpath -e "$SSH" <local>/ root@<ip>:/app/<same path>/
```

(Shell variables do not carry across Bash calls; set `SSH` in each command.)

- `--mkpath` is required for a path under `/app` that does not exist yet: rsync
  creates only the last directory level and otherwise fails with `mkdir "…" failed:
  No such file or directory (code 11)`.
- **Code**: the working tree's tracked files over `/app` (the image's `latest` may be
  from another ref):
  `git ls-files -z | rsync -a --mkpath --from0 --files-from=- -e "$SSH" ./ root@<ip>:/app/`.
  Without `-z`, git quotes non-ASCII names and rsync misses them. Untracked files are
  not in the list; push them as inputs.
- **Inputs**: `rsync -aL`, which copies symlink targets (e.g. `models/*` symlinked to
  another disk). When a directory itself sits behind a symlink, name the directory
  under it with a trailing slash so the local path resolves through the link;
  `rsync -a` on the symlink copies the link.
- **Parallel streams**: one rsync stream is latency-bound. Split the files over
  parallel streams, one ssh connection each (no ControlMaster, which multiplexes them
  onto one TCP connection). Shard by size — largest file first onto the lightest
  shard — write one list per shard, and run one `rsync --files-from=<list>` per list
  in parallel.

Measured 2026-10-11 from a residential line, 139 files / 418 MB to `EU-SE-1`:

| streams | time | MB/s |
|---|---|---|
| 1 | 133 s | 3.1 |
| 4 | 55 s | 7.7 |
| 8 | 47 s | 9.0 |

8 streams reached the line's upload ceiling (a `docker push` from it: 9.4 MB/s), so
1 GB takes ~2 min and 4 GB ~8 min. One stream to `CA-MTL-1` measured the same
3.1 MB/s. Code plus 1.1 GB of inputs over 9 streams: 98 s.

## On the pod

- **System packages**: text rendering needs `libraqm0` (Pillow's text layout) and
  `fonts-noto-cjk` / `fonts-noto-cjk-extra` (the system CJK faces); without them a
  rendered dataset draws other faces and placements. The current image installs them.
  A pod from an image built before that lacks them — check with `dpkg -s libraqm0
  fonts-noto-cjk fonts-noto-cjk-extra` and `apt-get install` what is missing.
- **Threads**: `nproc` reports the host's cores (128 on one 5090 pod), not the
  container's quota (`/sys/fs/cgroup/cpu.max` `1360000 100000` = 13.6 CPUs). The
  current image's entrypoint exports `OMP_NUM_THREADS` = the quota's whole CPUs less
  one into ssh sessions (12 there). If `echo $OMP_NUM_THREADS` over ssh is empty, the
  pod is on an older image; set it on the command.
- **Worker count**: a build whose output depends on its worker count takes the same
  `--workers` as the local run it is compared against.

## Launching

```bash
$SSH root@<ip> 'cd /app; setsid nohup <cmd> > /root/<run>.log 2>&1 < /dev/null & echo $! > /root/<run>.pid'
```

The launching ssh can stay open after the command returns, so run it as a background
Bash command.

## Waiting

One background Bash (`run_in_background: true`) `until` loop that checks the process
over ssh every minute and exits when it is gone. An ssh failure does not count as gone:

```bash
until [ "$($SSH root@<ip> 'kill -0 $(cat /root/<run>.pid) 2>/dev/null && echo up || echo gone')" = gone ]; do
  sleep 60
done
```

Not the Monitor tool: its deadline is capped at 30 min and it kills the command at
expiry (`daemon` skill § Waiting on a job). Read progress with
`$SSH root@<ip> tail -n 40 /root/<run>.log` when the user asks.
