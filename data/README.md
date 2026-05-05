# Data

This folder is **not committed** (see top-level `.gitignore`). It exists on
each teammate's machine after running the download steps below.

## MOT17 — pedestrian tracking benchmark

**Origin:** <https://motchallenge.net/data/MOT17/>
**License:** CC BY-NC-SA 3.0 — non-commercial research use only.
**Citation:** Milan, A., Leal-Taixé, L., Reid, I., Roth, S. & Schindler, K.
"MOT16: A Benchmark for Multi-Object Tracking." arXiv:1603.00831 (2016).

### Mirror used

The official server `motchallenge.net` is unreachable from a number of
networks (TCP/443 timeouts, including this team's machines). We therefore
download from the **PaddleDetection mirror** on Baidu Cloud Object Storage:

```
https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip
```

| Property | Value |
|---|---|
| Mirror URL | `https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip` |
| Size | 2,388,186,946 bytes (~2.39 GB) |
| SHA-256 | `4253cf596550847a74f58859fee6a1263a03c5bd946ec9545c0119e8e5e5e800` |
| Verified | 2026-04-26 |

The PaddleDetection mirror differs from the official zip in two ways:

1. **Layout.** Members live at `MOT17/images/{train,test}/<sequence>/...`
   instead of `<sequence>/...` at the root. Our extractor strips the
   `MOT17/images/train/` prefix so files land under `data/mot17/train/`.
2. **Detector variants.** Only the **-SDP** variant of each sequence is
   shipped (`MOT17-04-SDP`, `MOT17-09-SDP`, …). The original zip ships
   DPM/FRCNN/SDP for each sequence, but those three variants share the
   **same images and ground truth** — only the public detection files
   differ, and we use YOLO instead. Since we never read the public
   detections, SDP is functionally equivalent to FRCNN for our pipeline,
   so we extract under the canonical `MOT17-04-FRCNN/` and
   `MOT17-09-FRCNN/` names the rest of the project refers to. The
   `seqinfo.ini` `name=` field is patched from `-SDP` to `-FRCNN` after
   extraction so downstream code agrees.

### How to download

The repo ships a partial-extraction script that pulls only the dev and
held-out sequences (≈350 MB on disk after extract, vs. the full 2.39 GB
zip):

```bash
bash scripts/download_mot17.sh
```

For the full benchmark (all train sequences and the test set), pass `--all`:

```bash
bash scripts/download_mot17.sh --all
```

### Pinned split (frozen for the rest of the project)

| Role      | Sequence            | Frames | Notes |
|-----------|---------------------|--------|-------|
| Dev       | `MOT17-04-FRCNN`    | 1050   | Busy daytime, static camera, strong GT |
| Held-out  | `MOT17-09-FRCNN`    |  525   | Different scene, also static daytime — comparable to dev |
| (unused)  | All other sequences | —      | Not used in this project |

### Verified post-extraction state (2026-04-26)

| Check | MOT17-04-FRCNN | MOT17-09-FRCNN |
|---|---|---|
| `img1/*.jpg` count | 1050 | 525 |
| First frame opens (`cv2.imread(...).shape`) | (1080, 1920, 3) | (1080, 1920, 3) |
| `gt/gt.txt` rows | 108,005 | 10,411 |
| `gt/gt.txt` rows after `class==1 & conf==1` | 47,557 | 5,325 |
| Unique pedestrian track IDs | 83 | 26 |
| Frame-1 valid GT boxes | 42 | 6 |
| `seqinfo.ini` frameRate / seqLength | 30 / 1050 | 30 / 525 |

`gt.txt` columns (MOTChallenge standard, comma-separated):
`frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility`.

For our use, **always filter `class == 1 AND conf == 1`** to get the
"evaluation-considered" pedestrian boxes (pitfall §7 in the plan).

### Final on-disk layout

```
data/mot17/
├── train/
│   ├── MOT17-04-FRCNN/   <-- DEV
│   │   ├── img1/000001.jpg ... 001050.jpg
│   │   ├── det/det.txt              (public SDP detections; unused)
│   │   ├── gt/gt.txt                (MOTChallenge standard format)
│   │   └── seqinfo.ini              (name patched to MOT17-04-FRCNN)
│   └── MOT17-09-FRCNN/   <-- HELD-OUT
│       ├── img1/000001.jpg ... 000525.jpg
│       ├── det/det.txt              (unused)
│       ├── gt/gt.txt
│       └── seqinfo.ini              (name patched to MOT17-09-FRCNN)
└── test/                            (empty — no GT — unused for our purposes)
```

### MOTS17 (segmentation GT) — added in Phase 4 only

Phase 4 needs MOTS17 (`https://motchallenge.net/data/MOTS/`) for the
mask-IoU experiment on every 5th frame of MOT17-04. Do **not** download it
yet; that's a Phase 4 concern. Pinned here so the choice isn't relitigated.

## Demo clip — non-MOT, qualitative only

`data/demo_clip/source.mp4` — see `data/demo_clip/SOURCE.md` for the
specific source URL and license.

`data/frames/demo_clip/` — extracted frames at 10 FPS (Step 6).

No ground-truth labels exist for the demo clip; `dataset_loader.load_demo_clip()`
yields `Frame.gt = None` for every frame.

## Disk budget

| Item                                | Size    |
|-------------------------------------|---------|
| MOT17 dev + held-out (extracted)    | ~350 MB |
| Full MOT17 zip (if `--all` used)    | ~2.4 GB |
| Full MOT17 extracted (if `--all`)   | ~5 GB   |
| Demo clip + frames                  | ~100 MB |
| venv                                | ~3 GB   |
| **Total (default partial extract)** | ~3.5 GB |
