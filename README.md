# ZigVideo

**Local AI video generation for PCs that normally cannot run modern video models.**

ZigVideo is an experimental, hardware-aware video generation engine focused on
low-VRAM and older consumer GPUs. Instead of asking the user to understand every
model-specific memory flag, ZigVideo detects the machine and builds a generation
plan around the available GPU, VRAM, system RAM and CUDA generation.

> Status: early MVP. Version 0.1 implements hardware detection and the automatic
> generation planner. Inference backends are the next milestone.

## Why this project exists

Modern video diffusion models often assume powerful GPUs. The goal of ZigVideo is
not merely to wrap one model. It is to make multiple generation strategies behave
like one adaptive engine:

- low-resolution generation followed by temporal/spatial upscaling;
- FP16/BF16 selection based on the actual GPU generation;
- quantized weights where the hardware benefits from them;
- model/group/sequential CPU offload;
- VAE tiling and chunked decoding;
- temporal chunking with aggressive memory release between stages;
- disk spill when RAM is also constrained;
- progressive generation and resumable jobs;
- a hybrid keyframe + interpolation mode for hardware too weak for full video diffusion.

## Target hardware classes

| Mode | Typical machine | Initial strategy |
|---|---|---|
| CPU fallback | no usable CUDA GPU | sparse AI keyframes + interpolation |
| Ultra Lite | <4 GB VRAM | hybrid pipeline, tiny resolution, disk-first cache |
| Legacy 6 GB | GTX/Turing 4–6.5 GB | LTX 2B distilled GGUF quality experiment + staged/group offload; alternate FP16-native backend under evaluation |
| Low VRAM | RTX-class 6–8 GB | FramePack when compatible; otherwise quantized/offloaded Diffusers |
| Balanced | 8–12 GB | Wan/LTX with group offload |
| Quality | 12+ GB | larger windows/resolution and less aggressive offload |

The limits above are **project targets**, not guarantees. Every backend will be
benchmarked and the registry will eventually be driven by measured data.

## GTX 1660 / 6 GB is a first-class test target

A core challenge for ZigVideo is supporting 6 GB Turing-class cards that do not
make BF16/modern Tensor Core assumptions safe. The planned path is FP16-first,
small distilled models, temporal chunking, VAE tiling, memory release between
pipeline stages and post-generation interpolation/upscaling.

## MVP commands

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .

zigvideo doctor
zigvideo plan --task i2v --seconds 5 --fps 16
```

`doctor` prints the detected machine. `plan` chooses a generation strategy.

## Architecture

```text
Hardware Profiler
       |
       v
Capability / Memory Budgeter
       |
       v
Automatic Planner ----------------------+
       |                                |
       +--> Diffusers backend           |
       +--> FramePack backend           |
       +--> Hybrid keyframe backend     |
       +--> ComfyUI external adapter    |
                                        v
                          Chunk / Offload Scheduler
                                        |
                                        v
                              Decode / Interpolate
                                        |
                                        v
                                   FFmpeg output
```

## Roadmap

### M0 — Hardware-aware planner (current)
- GPU/RAM/VRAM detection
- legacy GPU detection
- automatic performance mode
- model/backend recommendation
- memory-safe resolution and chunk defaults

### M1 — First real video on 6 GB legacy NVIDIA
- LTX 2B distilled adapter
- FP16-only path
- sequential/group offload experiments
- VAE tiling and decode chunks
- OOM recovery with automatic parameter downgrade
- reproducible benchmark command

### M2 — Hybrid mode for 2–4 GB / CPU
- image/keyframe generator adapter
- optical-flow/frame interpolation adapter
- camera motion / Ken Burns stage
- optional lightweight upscaler

### M3 — RTX 6 GB progressive backend
- FramePack adapter when supported
- live progressive preview
- resume/cancel job support

### M4 — Smart model registry
- measured VRAM/latency database
- auto-download with license/size display
- model compatibility rules
- user-overridable profiles

### M5 — UI
- simple local web UI first
- optional node graph later
- one-click presets: Fast Preview / Low VRAM / Balanced / Max Quality

## Licensing strategy

ZigVideo core is intended to stay independent and Apache-2.0. ComfyUI is useful
as an optional external backend/reference, but its GPL-3.0 code should not be
copied into this core if we want to preserve a permissive core license.
Individual models and adapters keep their own upstream licenses and must be
checked before redistribution.

## Project principles

1. **Measure, do not guess.** Profiles graduate from experimental to stable only after benchmarks.
2. **Never crash first.** Catch OOM, downgrade settings, resume where possible.
3. **Weak hardware is a product requirement, not an edge case.**
4. **Preview cheaply, render expensively.**
5. **Separate generation from enhancement.** Low-res motion first, interpolation/upscale later.
6. **No silent quality tricks.** Caches/quantization that may alter output must be visible to the user.

## Upstream inspirations

- ComfyUI — modular graph execution, VRAM management and model offloading
- FramePack — progressive video generation with constant-size context packing
- Hugging Face Diffusers — quantization, group offloading and video pipeline ecosystem
- LTX-Video — small/distilled video models and multiscale workflows
- Wan — consumer-oriented video models and workflows


## Experimental first-video command

After `zigvideo ai-check` and `zigvideo torch-check` both pass, inspect the
first low-VRAM generation plan without downloading models:

```powershell
.\.venv\Scripts\zigvideo.exe generate --prompt "A small robot walks through a rainy futuristic city at night, cinematic camera movement" --preset ultra-safe --dry-run
```

Then run the first real generation:

```powershell
.\.venv\Scripts\zigvideo.exe generate --prompt "A small robot walks through a rainy futuristic city at night, cinematic camera movement" --preset ultra-safe --output outputs/first-zigvideo.mp4
```

The experimental backend uses an LTX-Video 2B distilled GGUF transformer with
FP16 compute, sequential CPU offload and VAE tiling. On CUDA OOM it automatically
retries progressively smaller resolutions. The first run downloads several GB of
third-party model components into `models/huggingface`.

The `ultra-safe` profile deliberately starts at 320x192, 9 frames and 8
inference steps. It is a compatibility experiment, not a quality preset.


## Legacy 6 GB output-quality checkpoint (2026-10)

The CogVideoX 2B FP16 backend runs end-to-end on GTX 1660 SUPER 6 GB.
Controlled tests kept prompt, seed (42), native 720x480 generation, 16 frames,
8 FPS, and guidance 6.0 constant. Changing only denoising steps resulted in:

| Steps | Total runtime | Observed visual result |
|---|---:|---|
| 30 | ~43.2 min | Clearly recognizable robot with strong contrast |
| 12 | ~20.9 min | Recognizable robot, but extremely dark |
| 8 | ~15.8 min | Near-solid black output, unusable |

Lowering steps did not preserve acceptable output quality in this test; do not
use 8-step CogVideoX as a validated preview profile. A runtime warning
(`invalid value encountered in cast`) appeared during conversion and is not
by itself proof of where the numerical fault occurred.

To diagnose this, ZigVideo now requests `output_type="pt"` from CogVideoX,
checks raw postprocessed frame tensors for NaN/Inf and low luminance, and writes
metrics plus a quality status into the generation JSON before image conversion.
These metrics do not replace reviewing the produced video. Suggested next
controlled experiment: test an intermediate 20-step run rather than assuming
the fastest run is usable.


### Numerical stage diagnosis

After the 20-step run produced a recognizable robot against a mostly black
background, the post-decode validation measured 18.33% non-finite output
values and 95.39% near-black pixels. This does **not** prove whether the
transformer denoising or the VAE introduced the non-finite values.

CogVideoX generation now inspects latent tensors at the first, middle and
last diffusion steps and records `latent_checks` plus
`quality.numerical_stage` in the generation JSON. The stage diagnostic
distinguishes non-finite values observed during denoising from non-finite
output observed only after decoding. Inspection adds small GPU tensor
reductions at only three checkpoints; it does not regenerate or repair
corrupted pixels. Do not assume FP32 VAE is the solution without this
stage evidence.


### FP32 VAE isolation experiment

The 8-step diagnostic found **0% non-finite latents** at steps 1, 5 and 8,
but **18.33% non-finite values after decode**, with all pixels almost black.
These sampled checks constrain, but do not conclusively localize, the fault.

A new optional `--vae-fp32` switch keeps CogVideoX transformer/text-encoder
in FP16 while loading the VAE in FP32 **before** installing CPU-offload hooks.
The CogVideoX backend now also records `raw_vae` statistics from the decoded
tensor before Diffusers' video postprocessor, as well as
`quality.decode_stage` in the JSON report. This is a diagnostic experiment,
not yet a validated quality fix. FP32 decode may require more memory/time.

```powershell
.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --prompt "A small friendly robot clearly visible in the center of the frame, full body, walking slowly through a rainy futuristic city street at night, detailed metallic body, neon reflections on wet pavement, cinematic lighting, realistic scene" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 8 `
  --seed 42 `
  --vae-fp32 `
  --output outputs\cogvideo-8steps-vae-fp32.mp4
```
