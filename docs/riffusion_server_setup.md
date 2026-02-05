# Riffusion server local changes (MGC compatibility)

This repo does not vend the riffusion server code. The local server lives in:

- /Users/admin/riffusion-hobby

The file `docs/riffusion_hobby_local.patch` captures local modifications made to get
riffusion running reliably with MGC. This keeps the changes reproducible without
requiring a fork or push to the upstream repo.

## Apply the patch

From the riffusion repo root:

```
cd /Users/admin/riffusion-hobby
patch -p1 < /Users/admin/music-generator-company/docs/riffusion_hobby_local.patch
```

To revert:

```
cd /Users/admin/riffusion-hobby
patch -p1 -R < /Users/admin/music-generator-company/docs/riffusion_hobby_local.patch
```

## What the patch does (high level)

- Lazy, thread-safe pipeline init for gunicorn workers.
- Env overrides for checkpoint/device/traced_unet.
- Adds `/health` and `/` endpoints for quick checks.
- More tolerant JSON parsing + better error handling.
- Minor seed image path handling + robustness tweaks.

## Runtime notes

Recommended env when launching gunicorn:

```
export RIFFUSION_DEVICE=cpu
export RIFFUSION_NO_TRACED_UNET=1
```

Example run:

```
gunicorn \
  -b 0.0.0.0:3013 \
  --workers 1 \
  --threads 1 \
  --timeout 300 \
  riffusion.server:app
```

## Local-only artifacts

The riffusion repo includes local virtualenvs (`.venv311/`, `.venv_riff/`) which are
not captured by the patch.
