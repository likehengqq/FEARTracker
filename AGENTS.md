# AGENTS.md

## Cursor Cloud specific instructions

This is the **FEAR visual tracker** research repo (PyTorch + PyTorch Lightning). It is not a web app — there are no servers, databases, or long-running services. Work happens through Python CLI scripts (see `README.md`).

### Python environment
- The project requires **Python 3.7** with the pinned old stack in `requirements.txt` (e.g. `torch==1.7.1`). It does NOT run on the system Python 3.12.
- A conda env named `py37fear` is created during setup and persisted in the VM snapshot. The startup update script refreshes deps into it.
- There is no `conda activate` in non-interactive shells. Use the env's interpreter directly:
  - Python: `~/miniconda3/envs/py37fear/bin/python`
  - pip: `~/miniconda3/envs/py37fear/bin/pip`
- Always run scripts from the repo root with `PYTHONPATH=.` (there is no installed package), e.g.
  `cd /workspace && PYTHONPATH=. ~/miniconda3/envs/py37fear/bin/python evaluate/macs_params.py`

### GPU / CUDA caveat (important)
- **This environment has no NVIDIA GPU.** `torch.cuda.is_available()` is `False`.
- Most code already falls back to CPU via `to_device()` in `model_training/utils/utils.py`, BUT two spots assume CUDA and must be worked around for CPU runs:
  - `demo_video.py` hard-codes `.cuda()` in `get_tracker()`.
  - `FEARTracker`/`BoxCoder` default to `cuda_id=0`, and `BoxCoder.to_device` does `tensor.to(0)` (interpreted as CUDA device 0). Instantiate the tracker with `cuda_id="cpu"` to run on CPU.
  - The bundled checkpoint `evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt` was saved with CUDA storages; load with `torch.load(..., map_location="cpu")` (i.e. pass `map_location="cpu"` to `load_from_lighting`).
- To run the demo on CPU without editing repo source, instantiate the tracker like the official `demo_video.py` flow but with `cuda_id="cpu"` and a CPU `map_location`. On a GPU machine, the documented command in `README.md` works as-is.

### Side effects of installing deps
- `requirements.txt` contains an editable git install (`-e git+...mobile-vision`). When pip runs from `/workspace`, it clones that package into `/workspace/src/` (gitignored). This is expected.
- Inference output videos go to `/workspace/outputs/` (gitignored).

### Useful commands (run from repo root with the py37fear interpreter)
- Count FLOPs / params (CPU-friendly, no GPU needed): `PYTHONPATH=. ~/miniconda3/envs/py37fear/bin/python evaluate/macs_params.py`
- Demo tracking inference: see `README.md` "Demo inference with Python" (needs GPU for the documented command, or the CPU workaround above).
- Training (`model_training/train.py`) and CoreML conversion (`evaluate/coreml_convert.py`) require external datasets / macOS+Xcode respectively and are out of scope for a basic CPU sanity run.

### Tests / lint
- There is no test suite, lint config, or build step in this repo. "Verification" means running the scripts above and confirming they execute end-to-end.

### iOS apps (optional, not runnable here)
- `evaluate/FEARDemo` and `evaluate/MeasurePerformance` are native iOS/Swift Xcode projects. They require macOS + Xcode + a physical iOS device and cannot be built in this Linux cloud environment.
