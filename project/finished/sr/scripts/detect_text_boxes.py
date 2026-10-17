"""Precompute text bounding boxes over an HR pool with comictextdetector (CTD).

Runs the manga-image-translator CTD **ONNX** model — no torch model code, no
manga_translator package. Weights come from `make download-models` (MIT step,
models/mit/comictextdetector.pt.onnx).

Backend: onnxruntime CUDAExecutionProvider by default (~13 ms/img forward; decode+
letterbox runs in a thread pool feeding one session — the ~70x lever vs cv2.dnn,
which is CPU-only). --no-gpu (or ort/CUDA unavailable -> automatic fallback) uses
the original cv2.dnn multiprocessing path (~1 img/s on 8 workers).

We only need coarse text-region boxes to bias training crops (ArtSRDataset
text_crop_prob), not per-line polygons. Detection uses the YOLO **blk** head
(conf 0.4 + NMS 0.35 — CTD's own defaults) cross-checked against the seg-mask head
(each block must contain some text-stroke coverage). The seg mask ALONE is not
usable: it fires on halos / hair ornaments / decorative line art (measured: 22/30
false-positive files on the aak block of hr_pool), and component mean-confidence
doesn't separate — thin real strokes average LOWER than solid decorative curves.

Output JSON (consumed by sr/distill_rsd/data.py::ArtSRDataset):
    {"meta": {...}, "boxes": {"<relpath under --src>": [[x0,y0,x1,y1], ...], ...}}
Every processed file gets a key; an empty list means "processed, no text" (so
--resume can skip it and the dataset can build its text-file list cheaply).

    python sr/scripts/detect_text_boxes.py                        # hr_pool, GPU (root venv)
    ... --src <dir> --out <json> --workers N --limit 100 --resume --no-gpu
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SR = HERE.parent
REPO = SR.parents[2]
ONNX_DEFAULT = REPO / "models" / "mit" / "comictextdetector.pt.onnx"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CANVAS = 1024  # CTD input size


def _prep(job):
    """(relpath, abspath) -> (relpath, letterboxed RGB canvas | None, (r, w0, h0))."""
    rel, path = job
    im = cv2.imread(path)  # BGR; None on decode failure
    if im is None:
        return rel, None, None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h0, w0 = im.shape[:2]
    r = min(CANVAS / h0, CANVAS / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    canvas = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return rel, canvas, (r, w0, h0)


def _post(blk, seg, geom, conf_th, nms_th, seg_th, seg_cov):
    """blk (N,7) + seg (H,W) heads -> [[x0,y0,x1,y1], ...] in original coords."""
    r, w0, h0 = geom
    conf = blk[:, 4] * blk[:, 5:].max(axis=1)
    keep = conf > conf_th
    if not keep.any():
        return []
    b, c = blk[keep], conf[keep]
    xywh = np.concatenate([b[:, :2] - b[:, 2:4] / 2, b[:, 2:4]], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), c.tolist(), conf_th, nms_th)
    boxes = []
    for i in np.array(idx).flatten():
        x, y, w, h = xywh[i]
        cx0, cy0 = max(int(x), 0), max(int(y), 0)
        cx1, cy1 = min(int(x + w), CANVAS), min(int(y + h), CANVAS)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        # cross-check: a real text block contains text strokes; decorative line-art
        # blocks (halos etc.) that fool the yolo head are stroke-sparse inside their box
        if (seg[cy0:cy1, cx0:cx1] > seg_th).mean() < seg_cov:
            continue
        boxes.append([max(int(x / r), 0), max(int(y / r), 0),
                      min(int((x + w) / r), w0), min(int((y + h) / r), h0)])
    return boxes


# ---- cv2.dnn multiprocessing fallback ------------------------------------------
_net = None
_args = None


def _init_worker(onnx_path, conf_th, nms_th, seg_th, seg_cov):
    global _net, _args
    cv2.setNumThreads(1)  # workers parallelize across images, not within one
    _net = cv2.dnn.readNetFromONNX(str(onnx_path))
    _args = (conf_th, nms_th, seg_th, seg_cov, _net.getUnconnectedOutLayersNames())


def _detect_cpu(job):
    conf_th, nms_th, seg_th, seg_cov, uoln = _args
    rel, canvas, geom = _prep(job)
    if canvas is None:
        return rel, []
    _net.setInput(cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(CANVAS, CANVAS)))
    outs = _net.forward(uoln)
    blk = next(o for o in outs if o.ndim == 3)[0]                         # yolo head
    seg = next(o for o in outs if o.ndim == 4 and o.shape[1] == 1)[0, 0]  # stroke prob
    return rel, _post(blk, seg, geom, conf_th, nms_th, seg_th, seg_cov)


def _iter_cpu(jobs, args):
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.onnx, args.conf, args.nms,
                           args.seg_thresh, args.seg_cov)) as pool:
        yield from pool.imap_unordered(_detect_cpu, jobs, chunksize=4)


# ---- onnxruntime CUDA path -------------------------------------------------------
def _iter_gpu(jobs, args):
    """Threaded decode (bounded prefetch) feeding one CUDA session serially.

    Session setup happens EAGERLY (before the generator is returned) so main()'s
    try/except can fall back to CPU — errors inside a generator body would only
    surface at first next(), after the fallback point.
    """
    import onnxruntime as ort
    ort.preload_dlls()  # resolve cudnn/cublas from the venv's pip nvidia-* packages
    sess = ort.InferenceSession(str(args.onnx),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError("CUDAExecutionProvider unavailable")
    return _gen_gpu(sess, jobs, args)


def _gen_gpu(sess, jobs, args):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    inp = sess.get_inputs()[0].name
    with ThreadPoolExecutor(args.workers) as ex:
        pending = deque()
        it = iter(jobs)

        def submit_next():
            job = next(it, None)
            if job is not None:
                pending.append(ex.submit(_prep, job))

        for _ in range(args.workers * 2):  # bounded prefetch, not .map (RAM)
            submit_next()
        while pending:
            rel, canvas, geom = pending.popleft().result()
            submit_next()
            if canvas is None:
                yield rel, []
                continue
            blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            outs = sess.run(None, {inp: blob})
            blk = next(o for o in outs if o.ndim == 3)[0]
            seg = next(o for o in outs if o.ndim == 4 and o.shape[1] == 1)[0, 0]
            yield rel, _post(blk, seg, geom, args.conf, args.nms,
                             args.seg_thresh, args.seg_cov)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SR / "data" / "hr_pool"))
    ap.add_argument("--out", default=str(SR / "data" / "text_boxes.json"))
    ap.add_argument("--onnx", default=str(ONNX_DEFAULT))
    ap.add_argument("--conf", type=float, default=0.4, help="yolo blk confidence (CTD default)")
    ap.add_argument("--nms", type=float, default=0.35, help="NMS IoU threshold (CTD default)")
    ap.add_argument("--seg_thresh", type=float, default=0.3,
                    help="stroke-prob threshold for the seg cross-check")
    ap.add_argument("--seg_cov", type=float, default=0.03,
                    help="min fraction of stroke pixels inside a blk box to keep it")
    ap.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                    help="onnxruntime CUDA (falls back to cv2.dnn CPU if unavailable)")
    ap.add_argument("--workers", type=int, default=8,
                    help="decode threads (GPU) / detector processes (CPU)")
    ap.add_argument("--limit", type=int, default=0, help="cap images (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already present in --out")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out)
    if not Path(args.onnx).exists():
        sys.exit(f"CTD onnx missing: {args.onnx} — run `make download-models` (MIT step)")
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images under {src}")

    done = {}
    if args.resume and out.exists():
        done = json.loads(out.read_text())["boxes"]
        print(f"resume: {len(done)} already done")
    jobs = [(str(p.relative_to(src)), str(p)) for p in files
            if str(p.relative_to(src)) not in done]
    if args.limit:
        jobs = jobs[: args.limit]

    results = None
    if args.gpu:
        try:
            results = _iter_gpu(jobs, args)
        except Exception as e:  # noqa: BLE001
            print(f"GPU backend unavailable ({e}) — falling back to cv2.dnn CPU")
    if results is None:
        results = _iter_cpu(jobs, args)
    print(f"{len(jobs)} images to detect ({len(files)} total), workers={args.workers}, "
          f"backend={'gpu' if args.gpu else 'cpu'}")

    t0 = time.time()
    boxes = dict(done)
    n_text = sum(1 for v in done.values() if v)
    meta = {"src": str(src), "onnx": args.onnx, "conf": args.conf, "nms": args.nms,
            "seg_thresh": args.seg_thresh, "seg_cov": args.seg_cov, "canvas": CANVAS}
    for i, (rel, bx) in enumerate(results):
        boxes[rel] = bx
        n_text += bool(bx)
        if (i + 1) % 200 == 0 or i + 1 == len(jobs):
            rate = (i + 1) / (time.time() - t0)
            eta = (len(jobs) - i - 1) / max(rate, 1e-9)
            print(f"{i + 1}/{len(jobs)}  {rate:.1f} img/s  with-text={n_text}  "
                  f"eta {eta / 60:.1f} min", flush=True)
            out.write_text(json.dumps(  # periodic flush so --resume loses <=200
                {"meta": meta, "boxes": boxes}))
    out.write_text(json.dumps({"meta": meta, "boxes": boxes}))
    print(f"done: {len(boxes)} files, {sum(1 for v in boxes.values() if v)} with text "
          f"-> {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
