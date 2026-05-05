# Demo clip — source & license

This file documents the single non-MOT video used for the qualitative
domain-generality demo. **No metrics are computed on the demo clip** — there
is no ground truth.

## Chosen clip

- **Title:** "People walking in the street in Japan"
- **Source page:** <https://mixkit.co/free-stock-video/people-walking-in-the-street-in-japan-4437/>
- **Direct .mp4:** <https://assets.mixkit.co/videos/4437/4437-720.mp4>
- **License:** [Mixkit Free Stock Video License](https://mixkit.co/license/) — free for commercial and personal use; no attribution required.
- **Author / source:** Mixkit (anonymous contributor; published on the Mixkit free library).
- **Local file:** `data/demo_clip/source.mp4`
- **Downloaded on:** 2026-04-26

### Verified file properties

| Property | Value |
|---|---|
| File size | 12,656,300 bytes (≈ 12 MB) |
| SHA-256 | `84da6ac4c563f94fc21e3e172b57680a1d5a3fb7a03715ddd5c5bb03b3fd5c4d` |
| Container | MP4 (ISO Media / iTunes Video) |
| Resolution | 1280 × 720 (HD ready, exactly at the ≤ 720p limit) |
| Frame rate | 25 fps |
| Frame count | 937 |
| Duration | 37.48 s (within the 30–60 s window) |
| First frame decode | OK (shape `(720, 1280, 3)`) |
| Mid frame decode | OK |

### Verified content properties (≥ 3 simultaneously visible people)

YOLOv8n run at `conf=0.25` on five sampled frames (frame indices 0, 234,
468, 702, 932):

| Sampled frame | YOLOv8n person count |
|---|---|
| 0 | 8 |
| 234 | 13 |
| 468 | 10 |
| 702 | 17 |
| 932 | 8 |

Min = 8, max = 17, mean = 11.2 — comfortably exceeds the ≥ 3 simultaneously
visible people requirement (Pitfall §4 in `Phase1_Implementation_Plan.md`).
This clip is a busy Japanese street scene with continuous foot traffic,
giving the proximity / dominant-region behavioral features real signal.

## How to (re)download

```bash
mkdir -p data/demo_clip
curl -L --fail -o data/demo_clip/source.mp4 \
  https://assets.mixkit.co/videos/4437/4437-720.mp4
```

To verify the SHA-256:

```bash
sha256sum data/demo_clip/source.mp4
# expected: 84da6ac4c563f94fc21e3e172b57680a1d5a3fb7a03715ddd5c5bb03b3fd5c4d
```

## Why Mixkit (not Pexels / YouTube CC)

From the team's network, the official `motchallenge.net` and the Pexels /
Pixabay CDNs (`https://www.pexels.com`, `https://cdn.pixabay.com`) all
return either TCP timeouts or HTTP 403 to programmatic clients. Mixkit's
`assets.mixkit.co` answers normally and does not require an account or
attribution. Wikimedia Commons is also reachable but did not have a
better-fitting pedestrian clip on a quick survey.

## Frame extraction (Step 6)

The downstream pipeline reads per-frame JPEGs, not the .mp4. After this
file is in place:

```bash
.venv/Scripts/python.exe -m src.frame_extractor \
  data/demo_clip/source.mp4 \
  data/frames/demo_clip --fps 10
```

At 25 fps source, downsampling to 10 fps yields ~`round(937 * 10 / 25)` ≈
**375 frames**, which is in the expected ~300–600 range from the plan.

## Status

✅ **Acquired and verified 2026-04-26.** The clip meets every Step 5
requirement (length, resolution, person count, license). Step 6 (frame
extraction) is unblocked.
