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



## First CogVideoX video on legacy 6 GB hardware

The currently validated **text-to-video** backend for GTX 1660 SUPER 6 GB is
CogVideoX-2B (native FP16), using sequential CPU offload. The legacy
`--backend auto` selector chooses CogVideoX. The LTX 2B GGUF path remains
experimental and produced unusable smeared output on this hardware.

```powershell
.\.venv\Scripts\zigvideo.exe generate `
  --prompt "A small friendly robot walking through a neon city, centered composition" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 30 `
  --output outputs\robot-quality.mp4
```

The model generates landscape 720x480 at 8 source fps, then ZigVideo reframes
the central composition to portrait 360x640. `zigvideo deliver` can use motion
interpolation and spatial scaling to produce a 720x1280, 24 fps delivery file.
This **does not** add true 720p source detail. For the measured 6 GB setup,
30 denoising steps have been a more reliable visual baseline than 8, 12, or
20 in the controlled robot-prompt benchmark below.

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
These metrics do not replace reviewing the produced video. The subsequent 20-step experiment also produced a very dark output; see the
stage-specific diagnostics below.


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
  --save-latents `
  --output outputs\cogvideo-8steps-vae-fp32.mp4
```


With `--save-latents`, the final latent tensor is saved alongside the video
as `outputs/cogvideo-8steps-vae-fp32.latents.safetensors`. This is a small
numeric artifact, not a playable video. It supports **decode-only**
experiments without rerunning the expensive transformer, using `zigvideo decode`. This diagnostic mode keeps the baseline
unchanged unless `--vae-fp32` and/or `--save-latents` are passed.


### Decode-only diagnostic (no new diffusion)

Once `--save-latents` has produced a `.latents.safetensors` file,
use `zigvideo decode` to load **only the CogVideoX VAE** in FP32 on CPU.
This avoids repeating transformer inference. It uses the exact CogVideoX
pipeline latent layout: `[batch, latent_frames, 16, latent_height, latent_width]`.

```powershell
.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\cogvideo-8steps-vae-fp32.latents.safetensors `
  --output outputs\cogvideo-8steps-cpu-decode.mp4 `
  --fps 8 `
  --aspect 9:16 `
  --dry-run
```

Remove `--dry-run` to decode the video. The standalone decoder reports
a sampled **raw VAE value distribution** and the final pixel-quality metrics,
including how often raw values lie at or below -1 (which the video
postprocessor maps to black). This path uses CPU/RAM, not the GTX, and may
take several minutes. It is a diagnostic mode, **not** an automatic repair
for the 8-step output.

The 8-step FP32-VAE experiment removed non-finite VAE values but still
produced a fully near-black video (maximum final pixel value 2/255). The
saved final latents were finite, so further work should focus on the raw
decoder distribution and whether 8 diffusion steps provide a useful
signal for this model. Do not infer from finite latents alone that all
internal operations of the diffusion model were numerically sound.


### Completed decode-only CPU/FP32 experiment

The same saved 8-step latent tensor was decoded again **without diffusion**
using only CogVideoX's VAE in FP32 on CPU (361.82 s). Results:

| Raw VAE sample metric | Observation |
|---|---:|
| Non-finite values | 0% |
| Sample minimum | -1.0540061 |
| Sample maximum | -0.9854688 |
| Sample mean | -1.0188351 |
| Share at or below -1 | 71.7488% |
| Final pixel maximum | 2/255 |
| Final video status | near_black |

The CogVideoX video postprocessor maps raw values from approximately [-1, 1]
to [0, 1] and clips outside that range. With virtually every sampled VAE
output value near or below -1, the image becomes near-black. **This occurs
on CPU and GPU with FP32 VAE output**, so repeatedly changing the VAE
offload method, MP4 encoding, or display scaling is not a promising remedy
for these saved 8-step latents. Finite latent values do not guarantee
that the denoising trajectory contains useful image signal. The exact root
cause of the poor 8-step latent distribution remains unconfirmed.

**Project decision:** the 8-step path is diagnostic only, not a quality
preset. Keep 30 steps as the reference on GTX 1660 SUPER for this prompt
while investigating speedups that do not simply reduce denoising steps.


## Experimental group offloading on GTX 1660 SUPER (diagnostic)

The established CogVideoX-2B pipeline uses sequential CPU offload and
typically takes 75-79 seconds per diffusion step on the tested GTX 1660
SUPER 6 GB. The default remains `--offload sequential`.

**First group experiment: failed.** On 2026-10-08, the one-step
`--offload group` test completed denoising in 134.26 seconds and
then raised `RuntimeError: Input type (torch.cuda.HalfTensor) and weight
type (torch.HalfTensor) should be the same` during VAE decoding.
This was **not** a CUDA OOM. It did **not** demonstrate improved speed.

**New implementation:** Instead of applying block-level offloading to the
entire pipeline (which left some VAE convolution weights on CPU while
inputs were on CUDA), the experimental mode configures each model
component according to the Diffusers 0.40 documentation:

- Transformer: block-level, one block per group, no streams.
- VAE: leaf-level; handles `vae.decode()` even though the VAE's
  root `forward()` method is not called.
- Text encoder (T5): leaf-level.
- The sequential baseline remains completely separate; hooks are not
  mixed on the same component.

The component-wise correction **was validated** on the GTX 1660 SUPER on
2026-10-08: 40 tests passed, the one-step run completed and produced MP4,
JSON and saved latents, with **peak CUDA allocated 3.78 GB**. However, the
single denoising step required 106.40 seconds and the complete pipeline
after model loading required 309.49 seconds. It generated near-black
content with 18.33% non-finite FP16 VAE output, which is not unexpected
with only one denoising step.

**Decision:** group offloading is supported as a diagnostic experiment but
is **not the recommended faster preset** for this GTX. In these
experiments, it offered no evidence of speedup over the established
75-79 s/step sequential baseline. Do not run a 30-step group benchmark
unless there is a specific new hypothesis or optimization to test.

The new stage-aware profiler reports:
- `prompt_setup_and_first_step_seconds` (includes T5 and setup)
- `subsequent_step_mean_seconds` (steady-state estimate, null with one step)
- `time_to_final_latents_seconds` (before VAE)
- `vae_decode_seconds` (VAE forward alone)
- `remaining_pipeline_seconds` (postprocessing and hooks)
- `pipeline_total_seconds`

This avoids the older misleading display of the **309.49-second**
full pipeline as `s/step`. The reported peak CUDA allocation is a
PyTorch allocation metric, **not** total hardware VRAM use.

### One-step diagnostic run (do not evaluate video quality)

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --prompt "A small friendly robot walking through a rainy futuristic city" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 1 `
  --offload group `
  --save-latents `
  --output outputs\group-offload-smoke-v2.mp4 `
  --dry-run
```

This experiment **has already been completed** on the GTX 1660 SUPER.
The saved `outputs/group-offload-smoke-v2.latents.safetensors` was
written successfully at the final denoising callback, before VAE decode.
The command is retained here for reproducibility, **not** as a request
to run it again. An intentionally single-step video is not a
meaningful quality result.

Stop after this smoke test and compare load time, inference time,
peak allocated GPU memory, and any error with the known sequential
baseline. If the run fails or is slower, keep the sequential
offload strategy and do not spend time on a 30-step group run.

References:
- https://huggingface.co/docs/diffusers/v0.40.0/optimization/memory
- https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/hooks/group_offloading.py


## VAE-only CPU versus CUDA FP32 benchmark (experimental, 2026-10)

The previous 8-step CogVideoX tensor was saved to
`outputs/cogvideo-8steps-vae-fp32.latents.safetensors`.
Its CPU FP32 decode-only reference took **361.82 seconds end-to-end**,
with `sample_mean=-1.018835` and final pixel maximum `2/255`.
The output is near-black because the saved **8-step** latent content was
insufficient; changing the decode device cannot restore missing details.

The `zigvideo decode` command now optionally accepts `--device cuda`.
The default **remains CPU FP32** (previously tested). On CUDA the VAE is
loaded in FP32 and uses **leaf-level CPU offloading** with tiling and
slicing. This avoids loading the transformer or performing diffusion,
and avoids the whole-pipeline group hooks that previously caused a
CPU/GPU convolution mismatch.

This is a **performance isolation experiment**, not a quality fix or a
guaranteed VRAM fit. On a 6 GB GTX card CUDA FP32 may be slower or
exceed free memory. Close other GPU applications if possible.

### Validate without loading model weights

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\cogvideo-8steps-vae-fp32.latents.safetensors `
  --output outputs\cogvideo-8steps-cuda-fp32.mp4 `
  --device cuda `
  --fps 8 `
  --aspect 9:16 `
  --dry-run
```

### Optional CUDA-only VAE experiment

Remove only `--dry-run` from the previous command. For comparable
results, keep the **same input latent file**, FP32, aspect and FPS.
No diffusion is run, so even if it is slow it avoids a 30-step job.
The resulting JSON reports:
- `load_seconds`: VAE model setup, including offload hooks.
- `decode_seconds`: the VAE decode itself, excluding postprocessing.
- `postprocess_seconds`: raw-value and frame quality analysis.
- `export_seconds`: reframe and MP4 encoding.
- `peak_cuda_allocated_gb`: measured PyTorch allocation (CUDA only).

Compare the CUDA `elapsed_seconds` against the earlier CPU-only total
**361.82s**, with the caveat that loading/cache conditions can differ.
The old CPU report did **not** break down individual stages. Only a
new CPU run with the new code can support an exact per-stage comparison;
it should not be necessary unless results are ambiguous.

Check that the candidate has no unexpected NaN/Inf and a broadly
consistent raw VAE output distribution. If CUDA is slower or fails,
simply omit `--device cuda` on future decodes. Do **not** repeat
30-step full generation until this measurement establishes an
advantage. FP32 VAE decode, even if accelerated, covers only part
of the full CogVideoX runtime.
