# Behavior Analysis — Docker Setup

Vision-based multi-object tracking and behavioral analysis Streamlit app. This README covers running the app with Docker only.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/) v2 (bundled with Docker Desktop)
- ~2 GB free disk space (image + weights + demo clip)
- (Optional, GPU) NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## Run the app (CPU)

```bash
git clone https://github.com/shyamsolanke/Msml-640-behavior-analysis.git
cd Msml-640-behavior-analysis
docker compose up
```

On first start the container automatically downloads:
- YOLOv8m weights (~50 MB) → `weights/yolov8m.pt`
- MobileSAM weights (~39 MB) → `weights/mobile_sam.pt`
- Demo clip (~12 MB) → `data/demo_clip/source.mp4`

These are persisted in bind-mounted folders, so subsequent starts are instant.

Once you see `Launching Streamlit on port 8501`, open:

**http://localhost:8501**

---

## Run the app (GPU)

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.
2. Open `docker-compose.yml` and uncomment the entire `app-gpu:` service block.
3. Start with the `gpu` profile:

```bash
docker compose --profile gpu up
```

Open **http://localhost:8501**.

---

## Optional: enable LLM summaries

Set `OPENAI_API_KEY` before starting the container (it is forwarded automatically):

```bash
# Linux / macOS
export OPENAI_API_KEY=sk-...
docker compose up

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
docker compose up
```

---

## Common commands

| Action | Command |
|---|---|
| Run in background | `docker compose up -d` |
| View logs | `docker compose logs -f` |
| Stop the app | `docker compose down` |
| Rebuild after code changes | `docker compose up --build` |
| Force fresh image | `docker compose build --no-cache` |
| Open shell in container | `docker compose exec app bash` |

---

## Troubleshooting

### Port 8501 already in use
Another process is using the port. Either stop it, or change the host-side port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"   # access at http://localhost:8502
```

### `docker: command not found` / `Cannot connect to the Docker daemon`
Docker Desktop is not running. Start Docker Desktop and wait until the whale icon is steady before retrying.

### Weights or demo clip fail to download on first run
The entrypoint pulls assets from the public internet. If you are behind a proxy or offline:
- Check connectivity: `docker compose exec app curl -I https://assets.mixkit.co`
- Manually drop the files into the bind-mounted folders on the host, then restart:
  - `weights/yolov8m.pt`
  - `weights/mobile_sam.pt`
  - `data/demo_clip/source.mp4`
- Restart with `docker compose restart`.

### Build fails on `pip install`
Usually a transient network error. Retry with:
```bash
docker compose build --no-cache
```

### Container exits immediately / `entrypoint.sh: not found` or `bad interpreter`
This happens on Windows when `entrypoint.sh` was checked out with CRLF line endings. The Dockerfile already strips them at build time, but if you edited the file locally on Windows, force a clean rebuild:
```bash
docker compose build --no-cache
docker compose up
```

### GPU not detected (`torch.cuda.is_available()` returns False)
- Verify the host sees the GPU: `nvidia-smi`
- Verify Docker can use it: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
- Confirm you started with `--profile gpu` AND that the `app-gpu` block is uncommented in `docker-compose.yml`.
- On WSL2, ensure NVIDIA drivers are installed on **Windows** (not inside WSL) and Docker Desktop's WSL integration is enabled.

### Permission errors on `weights/`, `data/`, or `outputs/` (Linux)
The container writes as root by default; bind-mounted folders inherit those permissions. Fix ownership on the host:
```bash
sudo chown -R $USER:$USER weights data outputs
```

### App runs but is very slow on CPU
YOLOv8m + MobileSAM is heavy on CPU (expect several seconds per frame). Use the GPU profile for real-time performance.

### Out of memory / container killed
Increase Docker Desktop's memory limit (Settings → Resources → Memory) to at least 4 GB; 8 GB recommended when running MobileSAM.

### Reset everything (re-download all assets)
```bash
docker compose down
rm -rf weights data/demo_clip outputs        # PowerShell: Remove-Item -Recurse -Force weights, data\demo_clip, outputs
docker compose up
```
