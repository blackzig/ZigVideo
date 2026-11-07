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


### Completed CPU-vs-CUDA FP32 VAE comparison (2026-10-10)

**Result: CUDA leaf-level VAE offload is not a performance win on the
tested GTX 1660 SUPER.** The same saved 8-step latents were decoded in
FP32 on both devices. Both produced a near-black video, maximum
2/255, **0% nonfinite output**, `sample_mean=-1.018835`, and
`sample_fraction_below_minus_one=0.717488`.

| Measurement | CPU FP32 | CUDA FP32 with leaf offload |
|---|---:|---:|
| Total elapsed | **361.82 s** | **375.84 s** |
| Load | Not separately measured | 2.59 s |
| VAE decode | Not separately measured | 369.79 s |
| Postprocess | Not separately measured | 0.63 s |
| Export | Not separately measured | 2.30 s |
| Peak PyTorch CUDA allocation | Not applicable | **10.574 GB** |

The CUDA total time was **14.02 s (about 3.9%) slower** in this
individual comparison. Different loading conditions mean this is not a
controlled estimate of universal CPU-vs-GPU speed, but it gives no
reason to adopt CUDA leaf offload for this tested hardware.

**Memory metric caveat:** the observed CUDA peak allocation was
**10.574 GB**, above the device's approximately **6 GB physical VRAM**.
PyTorch allocator accounting is **not the same as physically resident
VRAM**. Windows WDDM GPU-memory oversubscription/shared-memory paging
is one possible explanation; the experiment does not prove the exact
mechanism. The decode report now adds
`cuda_physical_vram_gb` and
`cuda_peak_exceeds_physical_vram`, and emits a warning when it
detects this discrepancy.

**Decision:** default CPU FP32 decode-only remains the safer diagnostic
choice for this machine. Do not repeat the CUDA benchmark or 30-step
full generation just to compare the same VAE offload path again.
Future performance work should be based on a different testable
hypothesis; improvements to VAE-only runtime do not imply equivalent
improvements to diffusion time.


## CogVideoX diffusion-only CFG cost benchmark (opt-in, 2026-10-10)

The 30-step, CFG=6, sequential CPU-offload generation remains the
**reference for known usable visuals** on the GTX 1660 SUPER 6 GB.
Tests with fewer denoising steps yielded dark video. The VAE FP32
CPU-vs-CUDA work above found no benefit from GPU leaf offloading.

To isolate **transformer** cost from slow VAE decoding, CogVideoX now
supports `generate --latents-only`. It asks the pinned Diffusers
pipeline for `output_type="latent"`, which skips VAE decoding and MP4
export entirely. Final latents are saved as
`outputs/<output-stem>.latents.safetensors` alongside a
`.latents.safetensors.json` performance report.
A `.mp4` output argument is used as the filename *stem*: **no MP4
is produced** in this mode.

This test adds an experimental `--cfg-scale` override. The
baseline is CFG=6. For CogVideoX, values greater than 1 use
classifier-free guidance with a doubled transformer input batch
(conditional + unconditional); CFG=1 skips this doubling.
This may reduce computation/memory but **changes the algorithm
and can affect subject identity, adherence and overall visual
quality**. CPU/GPU offload overhead may limit the actual gain.
No CFG=1 quality or speed benefit is currently validated.

### Step 1 — Update and check the plan only

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --prompt "A small friendly robot clearly visible in the center of the frame, full body, walking slowly through a rainy futuristic city street at night, detailed metallic body, neon reflections on wet pavement, cinematic lighting, realistic scene" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 2 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 1 `
  --latents-only `
  --output outputs\cfg1-2step-benchmark.mp4 `
  --dry-run
```

The dry run **does not use the GPU, load weights or generate latents**.
It should show `latents_only: true`,
`cfg_transformer_batch_factor: 1` and CFG 1 in
`native_generation`.

### Step 2 — Optional short diffusion-only benchmark

If the dry run and unit tests pass, repeat the command without
`--dry-run` when convenient. It runs **two steps** and writes
`outputs/cfg1-2step-benchmark.latents.safetensors` plus JSON.
It will still load the CogVideoX pipeline, so expect model-loading
and GPU inference cost, but **not** an additional expensive VAE decode.
No output quality can be judged from these latents alone.

The first checkpoint includes T5 prompt encoding, pipeline setup
and the first denoising step. The interval between the first and
second checkpoints is our approximate steady-state step measurement.
The result includes `stage_timings.subsequent_step_mean_seconds` and
`cfg_transformer_batch_factor`.

Only if that experiment is viable, make a like-for-like **CFG=6**
two-step reference with the *identical prompt, seed, resolution,
offload mode and steps*, but:
- `--cfg-scale 6`
- `--output outputs\cfg6-2step-benchmark.mp4`

Compare `stage_timings.subsequent_step_mean_seconds`,
`peak_cuda_allocated_gb`, and model-loading time.
Do **not** claim full 30-step speedups or acceptable visual
quality from the two-step data. Never evaluate quality by decoding
these severely under-denoised latent samples.

The normal `generate` command is unchanged unless
`--latents-only` or `--cfg-scale` is explicitly supplied.
`--vae-fp32` with `--latents-only` is rejected because the
VAE never runs.

Relevant pinned Diffusers implementation:
https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py


### Visual quality verdict: 30 steps, CFG 1 versus CFG 6 (2026-10-10)

A complete 30-step CogVideoX FP16 run was completed using the **same
original robot prompt and seed 42**, 720x480 native resolution, 16 frames
at 8 fps, sequential CPU offload, and 9:16 output. The only
intentional guidance change was **CFG 1 instead of CFG 6**.

| Metric | CFG 1 (new) | CFG 6 (earlier baseline) |
|---|---:|---:|
| Inference steps | 30 | 30 |
| CFG transformer batch factor | 1 | 2 |
| Measured steady step | 37.246 s | about 75-79 s (baseline) |
| Total wall-clock generation | **1433.62 s** (23m 53s) | **2593.89 s** (43m 14s) |
| Time difference | 1160.27 s saved (~44.7%) | reference |
| Prompt fidelity | **Poor**: robot not clearly identifiable | **Much better**: recognizable walking metal robot |
| Visual quality | Unacceptable as final output | Recognizable subject, but underlit |

Visual examination of the saved MP4s:
- `cfg1-30steps.mp4`: visually busy street-like forms with no clearly
  recognizable foreground robot. The image is generally brighter, but
  brightness **is not** prompt adherence.
- `cogvideo-shorts-test.mp4` (original CFG 6, 30 steps): a clearly
  identifiable silver robot walking against a very dark urban
  background; still requires quality/lighting improvements.

The CFG 1 raw FP16 VAE output had **18.333333% sampled nonfinite
values** despite finite denoising latents at steps 1, 16, and 30.
Those values were sanitized for export, so a successful MP4 is
**not** a clean numeric pass. Stage timings: 1152.21 s to last
latents, 177.59 s VAE, 1433.62 s total; peak CUDA allocated
4.028 GB.

**Decision:** do **not** switch the default to CFG 1. The near-halving
of transformer time comes with unacceptable prompt fidelity for this
scene. Keep **30 steps / CFG 6 / sequential offload** as the quality
reference, while acknowledging that it is slow and still too dark.
CFG 1 remains an opt-in diagnostic only. Two-step latent-only runs
establish throughput but **do not** establish image quality.

Further work should prioritize optimizations which preserve guidance,
or alternate backends / guided scheduling only if accompanied by a
new falsifiable hypothesis and *real* quality validation. Do not run
another 30-step job based solely on predicted speed, and do not
present faster-but-unrecognizable clips as a product improvement.


### Experimental: selective CFG steps (2026-10-10)

**Hypothesis:** keep normal CFG=6 on the **first N diffusion steps**
to preserve early prompt guidance, then run the conditional half of the
transformer only for the remaining steps. This is an opt-in experimental
cost optimization; **visual fidelity and runtime are not validated yet**.

**Why a simple changing CFG value is insufficient:** pinned Diffusers
0.40 computes `do_classifier_free_guidance` once, before the loop.
Changing `_guidance_scale` through a step callback would still send a
batch of 2 through the transformer on every step, so it would not
save the expensive forward computation.

The experimental `--cfg-guided-steps N` option uses temporary
PyTorch module input/output hooks only for the transformer:
- First N steps: unmodified CFG=6 conditional + unconditional batch.
- Remaining steps: send *only* the positive/conditional half to the
  real transformer. Duplicate that single prediction on return so
  the upstream CFG combination mathematically becomes the same
  conditional prediction as CFG=1.
- The original scheduler and Diffusers pipeline loop remain unchanged.
- Hooks are detached on successful return or failure.

**Restrictions:** the new flag requires `--backend cogvideox`,
`--offload sequential`, `--latents-only` and CFG strictly greater
than 1. Normal video generation **cannot** enable it until tested.
A single transformer call per denoising step and the upstream
`return_dict=False` tuple output are expected; incompatible
signatures fail explicitly. This is not a supported production preset.

**First experiment:** one CFG=6 step followed by one conditional-only
step, with no VAE or MP4. This tests actual GPU compatibility and
whether the second step approaches the previously measured CFG=1
steady time of **38.186 s**, rather than CFG=6's **75.293 s**.

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

# Check planned batch factors without loading the model
.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --prompt "A small friendly robot clearly visible in the center of the frame, full body, walking slowly through a rainy futuristic city street at night, detailed metallic body, neon reflections on wet pavement, cinematic lighting, realistic scene" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 2 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --cfg-guided-steps 1 `
  --latents-only `
  --output outputs\cfg-selective-1of2.mp4 `
  --dry-run
```

After tests and the dry run pass, remove **only** `--dry-run` for a
small real-GPU smoke test. It creates
`outputs/cfg-selective-1of2.latents.safetensors` and
`outputs/cfg-selective-1of2.latents.safetensors.json`, not an MP4.
The report's `cfg_schedule.actual_transformer_batch_factors`
should read `[2, 1]`; check the **measured**
`stage_timings.subsequent_step_mean_seconds` too.

**Stop if:** incompatible hook signature, model output shape mismatch,
GPU memory failure, step count mismatch or second step remains near
CFG=6 timing. Passing the two-step smoke only establishes that the
batch reduction works; it **does not** establish visual fidelity.
Do not launch a 30-step selective-CFG quality job before this check.

The previously validated baseline remains 30 steps, CFG=6,
sequential offload; do not change the default automatically.


### Selective CFG: real GTX 1660 SUPER proof and middle-window option (2026-10-10)

The 2-step **CFG 6 on first step / conditional-only on second**
experiment completed on the user's GTX 1660 SUPER 6 GB, with
Diffusers 0.40 and sequential CPU offload.

| Two-step latent-only experiment | CFG 1 always | CFG 6 always | CFG 6 then conditional-only |
|---|---:|---:|---:|
| Guidance transformer batch pattern | [1, 1] | [2, 2] | **[2, 1]** |
| Second denoising step | 38.186 s | 75.293 s | **37.401 s** |
| Full pipeline inference, including first step | 119.959 s | 192.537 s | **153.135 s** |
| Peak CUDA allocation | 0.509 GB | 0.763 GB | 0.757 GB |

The selective hook recorded **2 transformer calls**, one normal
guided call and one conditional-only call, with actual batch factors
**[2, 1]**. Both latent checkpoints reported **0% nonfinite**.
The measured second-step time is **50.3% lower** than always-guided
CFG 6 in these separate runs; total diffusion time is about
**20.5% lower**. This verifies the computational mechanism,
**not visual quality**. The unchanged peak CUDA usage across
the selective and full-CFG runs reflects the first guided step.

**Next research hypothesis — middle window:** studies of
stage-wise CFG schedules in *image diffusion*, including
[Jin, Shi & Gu, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f9e2800a251fa9107a008104f47c45d1-Abstract-Conference.html),
suggest that the timing of strong guidance affects quality and
diversity. This does **not** prove the best CogVideoX video schedule.
Rather than assuming that CFG belongs only at the beginning, we
now allow a guided interval anywhere.

With the added `--cfg-guided-start S` (zero-based), the
`--cfg-guided-steps N` option guides indices **[S, S+N)**,
and all other indices use the faster conditional-only transformer.
Default `--cfg-guided-start 0` preserves the already-tested
guided-prefix behavior. The saved safetensors metadata now records
`cfg_schedule=selective_window`, `cfg_guided_start` and
`cfg_guided_steps`.

**Do not run a 30-step quality test yet.** First verify that
the real model can move from conditional-only, to CFG, and
back to conditional-only. This is a three-step **latents-only**
smoke test, pattern **[1, 2, 1]**:

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --prompt "A small friendly robot clearly visible in the center of the frame, full body, walking slowly through a rainy futuristic city street at night, detailed metallic body, neon reflections on wet pavement, cinematic lighting, realistic scene" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 3 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --cfg-guided-start 1 `
  --cfg-guided-steps 1 `
  --latents-only `
  --output outputs\cfg-mid-1of3.mp4 `
  --dry-run
```

If and only if the tests and dry run pass, remove only
`--dry-run` to measure a three-step smoke test. No VAE and
no MP4 are produced. Expect `actual_transformer_batch_factors`
to equal **[1, 2, 1]**. The second step may run close to the
CFG 6 baseline, while the third should approach the CFG 1
baseline; neither is guaranteed by unit tests. Do not yet infer
image quality from these intermediate latents.

Guidance is still restricted to `--latents-only` and
`--offload sequential`, so regular MP4 generation and the
quality reference remain untouched.


### GTX 1660 SUPER: real middle-window CFG switching benchmark (2026-10-10)

A **three-step latents-only** CogVideoX-2B test succeeded with CFG=6,
`--cfg-guided-start 1 --cfg-guided-steps 1`, 720x480 native,
16 frames, sequential offload, seed 42. The *actual* transformer
batch-factor sequence was **[1, 2, 1]**: no CFG, CFG 6, no CFG.

| Step | Actual CFG mode | Elapsed checkpoint interval |
|---|---|---:|
| 1 | Conditional-only (batch 1) | 78.584 s, includes encoding/setup |
| 2 | CFG 6 (batch 2) | **72.167 s** |
| 3 | Conditional-only (batch 1) | **36.211 s** |

The last step was approximately **49.8% faster** than the guided
second step *within the same run*. Pipeline diffusion time was
**187.001 s**; loading took **99.33 s**; total elapsed was
**286.8 s**. Peak CUDA allocation was **0.763 GB**, reflecting
the guided step. Transformer calls: 3; guided: 1; conditional-only: 2.
All three sampled latent checkpoints were finite (0% nonfinite).
No VAE decode or MP4 export was performed, and image quality was
**not evaluated**. Measured values reflect this hardware/run only.

**Engineering verdict:** the hook can switch **both ways** between
the actual positive-only batch and doubled CFG batch inside one
real pipeline run, without observed numeric failures. The claim is
computational, **not** a claim about preserving a recognizable robot
or delivering usable video. The previous 30-step CFG=1 output
demonstrated that speed alone is insufficient.

**Next quality decision:** before any costly 30-step run, explicitly
select a middle-window or prefix schedule, preserve the original
CFG=6 benchmark prompt and seed for comparison, and add a guarded
experimental MP4 output path that always saves latents and reports
VAE numerical problems. Do not change the default CFG=6 generation
preset. No full-length selective-CFG video has been tested.


### Explicit experimental selective-CFG MP4 quality gate (2026-10-10)

After the real **[1, 2, 1]** mid-window GPU test succeeded, selective
CFG can now be evaluated visually with a **separate explicit safety
gate**: `--experimental-cfg-video`. No normal/default invocation
enables it. The normal 30-step CogVideoX **CFG=6 / sequential-offload**
quality reference is unchanged.

Without this flag, the ZigVideo CLI **still refuses** selective CFG
in video mode. With it, the requested guided window must mix guided
and non-guided steps, CFG must exceed 1 and offload must be
`sequential`. `--experimental-cfg-video` must **not** be combined
with `--latents-only` and requires `--cfg-guided-steps`.

Video mode **automatically saves final latents before VAE decoding**
(and saves the selective guided window in safetensors metadata).
The normal raw VAE/quality diagnostics and MP4 report are retained.
The `.mp4.json` report additionally records
`experimental_cfg_video=true` and `cfg_schedule` with real
transformer batch factors. This matters because our previous 30-step
CFG=1 run had **18.333333% sampled nonfinite FP16 VAE output** even
though denoising latents were finite. A generated MP4 does not imply
good image quality or clean numeric output.

**Quality test plan, NOT yet run:** retain the established
30-step CogVideoX prompt, seed 42, 720x480 native size, 16 frames
and 9:16 output. A candidate middle-window test is **20 guided
steps of 30**, guided zero-based indices **5..24**, with lighter
conditional-only transformer steps at indices 0..4 and 25..29.
This is **a hypothesis** about preserving recognizable structure,
not a validated good schedule. At the measured ~75s / ~37s
steady-step costs, 10 conditional-only steps might save around
6 minutes of denoising compared with all-CFG; this is a rough
estimate, not a promise about wall-clock runtime or quality.
The test may still take more than 30 minutes including loading
and VAE decoding.

Before incurring that cost, run only the unit tests and dry run:

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --prompt "A small friendly robot walking through a rainy futuristic city" `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 30 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --cfg-guided-start 5 `
  --cfg-guided-steps 20 `
  --experimental-cfg-video `
  --output outputs\cfg-mid-30steps-quality.mp4 `
  --dry-run
```

Dry run must say `experimental_cfg_video=true`,
`output_mode="experimental_cfg_mp4"`, `save_latents=true` and
`cfg_step_batch_factors` with five 1s, twenty 2s and five 1s.
Do not remove `--dry-run` until the new tests pass and the user
chooses to spend the generation time. Any later completed MP4
must be visually inspected and compared to the CFG=6 reference.


### Results: full 30-step selective-CFG MP4 (2026-10-10)

The first **full video** test of selective CFG completed on the
GTX 1660 SUPER (6 GB), CogVideoX-2B FP16, native 720x480,
16 frames at 8 FPS, 9:16 center-cropped to 360x640,
30 steps, seed 42 and sequential offload. The experiment
used `--cfg-scale 6 --cfg-guided-start 5 --cfg-guided-steps 20
--experimental-cfg-video`.

- **Actual GPU steps**: 5 conditional-only, 20 guided (CFG 6),
  then 5 conditional-only. Verified 30 transformer calls and actual
  batch factors `[1,1,1,1,1,2×20,1,1,1,1,1]`.
- **Mean guided step time**: **78.2905 s** (steps 6–25).
  First four post-setup conditional-only steps averaged
  **39.3453 s**; final five averaged **39.9148 s**.
- **Final latents**: reached in **2008.735 s**; sampled
  steps 1, 16 and 30 had **0% nonfinite** values.
- **VAE FP16**: **192.226 s**; sampled raw VAE output contained
  **18.333333% nonfinite values**, sanitized for export.
  Numeric status `nonfinite_detected` remains a quality warning.
- **Total**: **2310.6 s (38m 30.6s)**. Prior full CFG=6
  baseline had **2593.89 s (43m 13.89s)**; difference is
  **283.29 s (4m 43.29s), ~10.92%**. These are separate runs;
  the baseline report does not record prompt text, so exact
  prompt equality cannot be independently verified from those
  report files alone.
- **Video visual inspection**: selective CFG has a robot partly
  visible **at the right edge**, with a large vehicle occupying most
  of the delivered vertical frame. The baseline full CFG=6 video
  has a recognizable full-body robot centered in the vertical frame.
  CFG=1 full had no clearly recognizable main robot. Thus the
  selective 20/30 **does not pass the prompt/enquadramento quality
  goal**, despite working correctly and saving runtime.

**Decision:** leave default full CFG=6 unchanged, and keep the
selective-video option experimental. Do not call 20/30 "quality
validated", do not replace the standard preset, and do not
launch another 30-step video generation yet.

**Low-cost next diagnostic:** re-decode the **existing**
`outputs/cfg-mid-30steps-quality.latents.safetensors` with
the existing `zigvideo decode` command using `--device cpu`
(FP32 VAE) and `--aspect 16:9`. The current 9:16 result is
a center crop of 720x480 native diffusion. The wider view
may reveal whether the subject exists outside the portrait crop.
FP32 may also change VAE numerical behavior, but neither an
improvement nor a recognizable subject is guaranteed. This
reuses 30-step latents; it does **not** repeat the denoising pass.

```powershell
.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\cfg-mid-30steps-quality.latents.safetensors `
  --device cpu `
  --aspect 16:9 `
  --fps 8 `
  --output outputs\cfg-mid-30steps-wide-fp32.mp4 `
  --dry-run
```

If the dry-run validates the path/shape/metadata, remove just
`--dry-run`. The CPU-only FP32 VAE can be slow and use
substantial RAM; it does not run the transformer. Compare the
wider MP4 (and raw VAE nonfinite diagnostic JSON) with the
existing vertical export before changing the video crop
or diffusion schedule.


### VAE FP32 and portrait subject positioning: confirmed diagnosis (2026-10-10)

The **previously saved** 30-step selective-CFG latents
(`cfg-mid-30steps-quality.latents.safetensors`) were successfully
re-decoded using `zigvideo decode --device cpu --aspect 16:9`:

- FP32 VAE **0% sampled nonfinite**; preflight quality status
  `plausible`, 16 frames @ 8 FPS, 640×360 exported.
- FP32 CPU VAE time **339.93 seconds**; total **342.47 seconds**.
  The prior FP16 VAE in video generation took 192.23 s but had
  **18.333333% sampled nonfinite** outputs. These modes differ in
  both precision *and* device, so do not ascribe the whole
  runtime difference to FP32 alone.
- Visual inspection of the wide FP32 MP4 confirmed a recognizable
  gray/white robot with glowing eyes on the **right-hand side**
  of the native scene, with a large vehicle dominating the
  background. The original *center* 9:16 crop omitted much of
  this right-positioned subject. The robot moves its head/body
  but does **not** demonstrate a clear full-body walk; its lower
  legs are cut off by the scene framing.
- This corrects the preliminary interpretation: CFG selective
  20/30 **does produce a recognizable robot**, but composition
  and requested walking motion remain poorer than the prior
  full CFG=6 baseline.

A diagnostic 9:16 crop of the existing **wide** FP32 MP4,
shifted right, confirms that the robot can be centered **without
regenerating the video**. This diagnostic crop was made from the
640×360 horizontal file and therefore cannot restore vertical
pixels already removed by the wide aspect export.

**New opt-in feature**: `zigvideo decode --focus-x 0.70`.
This crops **directly from the native 720×480 VAE output** when
`--aspect 9:16`, rather than using the intermediate 16:9 file.
The normalized horizontal subject position is 0 (left edge),
0.5 (original center; default), 1 (right edge). With focus=0.70,
the native portrait window is approximately x=369..639
instead of the old center x=225..495. The value is
clamped to the source bounds and recorded in the new decode
JSON `focus_x` field. Values outside 0..1, NaN and Inf
are rejected. Non-default focus with `--aspect 16:9` is rejected.

**Validation workflow (no new diffusion):**

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\cfg-mid-30steps-quality.latents.safetensors `
  --device cpu `
  --aspect 9:16 `
  --focus-x 0.70 `
  --fps 8 `
  --output outputs\cfg-mid-30steps-vertical-focus70-fp32.mp4 `
  --dry-run
```

The tests and dry run should be checked before removing
`--dry-run`; the FP32 CPU VAE will again take several minutes.
This is optional since a cropped FP32 diagnostic MP4 already
shows the robot centered, but it may retain **more vertical
content** than cropping the horizontal MP4. Do not claim
full-body walking without visual verification. The full
CFG=6 generation baseline and all generation defaults are
unchanged.


### Final portrait FP32 decode and visual check (2026-10-10)

The new `--focus-x 0.70` decoder was validated by the user on
Windows with **95/95 passing tests** and a successful
`decode --dry-run`. The real decoded file was:

`outputs/cfg-mid-30steps-vertical-focus70-fp32.mp4`
with JSON report alongside it. This re-used the already-saved
`outputs/cfg-mid-30steps-quality.latents.safetensors`, without
running the transformer again.

- VAE: FP32 CPU, `--aspect 9:16 --focus-x 0.70 --fps 8`
- Result: **16 frames, 360×640, 2 seconds**
- Raw VAE sampled nonfinite fraction **0.0%**, quality
  preflight `plausible`; source FP16 VAE had 18.333333%
  sampled nonfinite values
- Load 1.45s; VAE 318.56s; postprocess 0.29s; export 0.42s;
  **total 320.72s (~5m21s)**

Visual review of frames 1, 3, 5, 7, 9, 11, 13 and 15 of the
user-supplied MP4:
**the robot is recognizably present and centered**, with visible
head, torso and arms through the short clip. There are subtle
postural changes and a more distinct head turn toward the end.
Its legs are still clipped by the framing and the large vehicle
remains prominent behind it. The clip does **not** depict a
convincing full-body walking cycle, and is only 2 seconds long.

**Engineering outcome:** numerical stability of sampled VAE output
is fixed in this CPU FP32 path; the subject-loss caused by centered
portrait cropping is addressed with optional horizontal focus.
The unchanged defaults remain as before (`focus_x=0.5`,
full-CFG reference generation). **Selective CFG is not promoted
to a default** or declared to have equivalent visual quality.
Further work should target **native composition and temporal
walking motion**, not additional step-time benchmarks.

The saved final MP4 and JSON constitute this checkpoint.
No additional denoising run is necessary to establish these
observations; any future 30+ step generation should have a
clearly defined quality hypothesis and estimated compute cost.


### Next quality hypothesis: camera composition and visible walking (2026-10-10)

The validated 2-second selective-CFG robot clip revealed two separate
problems: a full-body character was never composed in the native scene,
and the robot did **not** perform a convincing alternating-foot walk.
`--focus-x 0.70` recovered the existing subject during portrait
decoding but **cannot reconstruct missing legs or invent motion**.
Changing the output from 8 FPS to 24 FPS is delivery interpolation,
**not** new model-generated movement.

To isolate the next variable without changing successful settings,
ZigVideo now has an **opt-in prompt experiment**:

- `--scene-profile robot-walk`: a fixed positive/negative prompt
  targeting **one** small robot completely visible from head to both feet
  within the central portrait-safe region of the native 720×480 image.
  It explicitly describes alternating left/right footfalls, arm
  swing, tripod camera, and an unobstructed simple rainy street.
  The negative prompt discourages vehicles, clipped legs, zoom/pan,
  static poses, sliding feet, and extra limbs. These instructions are
  **hypotheses**, not guaranteed model capabilities.
- `--prompt` remains the ordinary custom-prompt option. Specify
  **either** `--prompt` **or** `--scene-profile`, not both.
  The latter is CogVideoX-only.
- `--negative-prompt` optionally overrides the chosen negative text
  for CogVideoX. Without it, the **exact previous default** is used
  for ordinary prompts.
- The **exact** positive prompt, negative prompt and optional profile
  name are now written to dry-run JSON, full MP4/latents-only JSON
  reports and (when saving latents) safetensors metadata. Previously
  missing prompt provenance made quality comparisons less rigorous.
  Reproduce a comparison using the exact reported prompt/seed/CFG
  and generation settings; reports do not imply visual quality.
- All old CogVideoX defaults, including 30 steps, CFG=6,
  16 native source frames at 8 FPS, sequential offload, FP16 VAE
  and center crop, remain unchanged. Selective CFG remains
  experimental; the new profile does **not** enable it.
  For the already observed VAE FP16 nonfinites, preserve latents
  with `--save-latents` if eventually doing a real run, then use
  separate FP32 CPU decode for quality inspection.

**Do the zero-GPU preflight first:**

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --scene-profile robot-walk `
  --preset ultra-safe `
  --aspect 9:16 `
  --steps 30 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --save-latents `
  --output outputs\robot-fullbody-walk-test.mp4 `
  --dry-run
```

The dry run must show `scene_profile="robot-walk"`, the
exact effective prompt and negative prompt, `save_latents=true`,
and **normal full CFG=6** (no `cfg_schedule`). It doesn't load
models or perform any inference.

**Do NOT remove `--dry-run` until unit tests and profile output
are checked.** If/when the user elects the GPU cost, a 30-step
run could again require roughly **40+ minutes** and subsequent
CPU FP32 VAE decoding ~5 minutes. A new video is warranted only
to test whether the revised composition/motion prompt actually
improves body visibility and gait against the earlier CFG=6
reference; do not assume that a 16-frame, 2-second clip can show
a natural extended walk or a 15-second short. Longer content will
require a separate temporal-generation plan.

**Acceptance criteria:** one robot is centered within the native
portrait-safe area *before* cropping; head and both feet remain
visible across the 16 frames; alternate foot placement is clear
rather than foot sliding; VAE FP32 decoded result has no sampled
nonfinite values; any failed criterion is recorded, not hidden
by postprocessing.


### Result: robot-walk full CFG6 / 30-step quality test (2026-10-10)

**Executed and user-verified.** The first `--scene-profile robot-walk`
generation (after 107/107 local unit tests passed) used
CogVideoX-2B FP16, 720×480 native, 16 frames at 8 FPS,
seed=42, sequential CPU offload, and **full CFG=6** in all
30 denoising steps. It used `--latents-only`, preserving
the exact positive/negative prompt, `scene_profile=robot-walk`
and CFG metadata for comparison.

- 30/30 steps finished; `cfg_schedule=null` (no selective CFG).
- Latent checkpoints 1, 16 and 30 sampled **0% nonfinite**.
  Native latent shape: [1, 4, 16, 60, 90].
- Loading **112.38 s**; inference **2321.63 s**; complete
  generation **2434.69 s (40m 34.69s)**. Mean subsequent
  denoising step **76.061 s**; peak CUDA allocation **0.763 GB**
  (this was latent-only, excluding video/VAE generation).
- The saved latents were independently decoded with CogVideoX
  FP32 VAE on **CPU**, `--aspect 9:16 --focus-x 0.50 --fps 8`
  to `outputs/robot-fullbody-walk-fp32.mp4`.
  CPU VAE **372.21 s**, decode total **376.63 s (6m 16.63s)**.
  Combined generation + decode **2811.32 s (46m 51.32s)**.
- FP32 raw VAE sampling: **0.0% nonfinite**, sample mean
  **-0.69675076**, sample fraction below -1 **0.012399**;
  output preflight `plausible`, final 360×640, **16 frames /
  2 seconds**. Numerically plausible does not imply motion
  quality or realistic physics.

**Human visual inspection of the actual uploaded 16-frame MP4**
(8-frame and 16-frame contact sheets): one clearly recognizable
humanoid robot, **fully visible from head to both feet**, within
the vertical frame and approximately centered. No vehicle obscures
the subject. The blue-neon wet paving is relatively uncluttered.
Foot and arm positions vary over the sequence, including some
leg alternation toward the latter frames, but the walk remains
**somewhat stiff** and an extended natural gait is **not yet
established** in the very short 2-second output.

**Interpretation:** this is a successful composition experiment,
not proof that increasing steps/frames or selective CFG will improve
walking naturally. In particular, a different prompt was used
from the older selective-CFG experiment, so its visual superiority
**must not** be attributed solely to full CFG=6.

**Next priority:** retain the profile and all generation defaults,
do not rerun expensive 30-step tests for speed alone. Plan a
measured, opt-in **temporal-duration and motion** experiment
separately, with explicit memory/time expectations and comparison
criteria. Delivering at 24 FPS interpolates existing source
frames and cannot invent missing gait or extend the 2-second
narrative. The saved current MP4 is a useful reference.


### Next experimental stage: 33 *native* frames with unchanged 30-step CFG6 (2026-10-10)

The previous `robot-walk` quality checkpoint passed the **single,
fully framed robot** composition objective at 16 frames (8 FPS,
2 seconds), but the walking cycle was still stiff. To isolate
temporal length as the only generation variable, a new
CogVideoX-only `generate --frames` **optional override** accepts
the model's already configured frame buckets `16`, `33`, `49`.
It does **not** change any defaults or any preset definitions.
Previously `--preset safe` selected 33 frames **and 40 steps**;
now `--preset ultra-safe --frames 33 --steps 30` keeps all
settings but the number of native frames equal to the successful
16-frame reference.

**Changes and safeguards:**
- `--frames 33` selects **33 genuine generated frames at 8 FPS
  (4.125 seconds)**, not duplicated or interpolated frames.
- The default stays 16 frames, 30 steps, native 720x480,
  8 FPS, CFG6 and sequential offload. The `safe` preset
  still selects 33 frames and 40 steps unless overridden.
- Unsupported frame counts are rejected before model loading,
  and `--frames` is rejected for the experimental LTX backend.
- CogVideoX dry-run now reports `source_duration_seconds`.
  Explicit long frame overrides include
  `temporal_experiment` with `quality_validated=false`,
  `time_validated=false` and a warning that memory and runtime
  may increase nonlinearly. **Dry-run cannot check actual GPU RAM
  or render visual quality.** The generated report/latent metadata
  already saves the true selected number of frames.
- The 33-frame result is **NOT YET GPU VALIDATED** on the GTX
  1660 SUPER 6 GB. The previously measured 16-frame latents-only
  run took **40m34.69s**, and CPU FP32 decode took **6m16.63s**.
  A simplistic linear extrapolation using 33/16 suggests ~80 min
  for diffusion alone and ~13 min for FP32 decode, plus load.
  This is **only a budgeting illustration**, *not* a benchmark:
  actual cost and peak RAM/VRAM can be significantly different
  because sequence-length effects are not necessarily linear.
  Allow well over an hour for a possible later full-length test
  and be prepared for an OOM, even if the 16-frame test worked.

**Phase A — zero-GPU validation:**

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --scene-profile robot-walk `
  --preset ultra-safe `
  --frames 33 `
  --aspect 9:16 `
  --steps 30 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --latents-only `
  --output outputs\robot-walk-33frames.mp4 `
  --dry-run
```

Check that `native_generation` is 720×480,
`num_frames=33`, `num_inference_steps=30`, `fps=8`,
`guidance_scale=6.0`, `source_duration_seconds=4.125`,
`cfg_schedule=null`, and no VAE is requested. The profile
prompt/negative prompt should exactly match the 16-frame
checkpoint.

**Phase B — optional one-step hardware/OOM smoke test
(only after passing the unit tests and dry-run):**

```powershell
.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --scene-profile robot-walk `
  --preset ultra-safe `
  --frames 33 `
  --aspect 9:16 `
  --steps 1 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --latents-only `
  --output outputs\robot-walk-33frames-smoke.mp4
```

This must **not** be used for video-quality evaluation, as a
1-step image is expected to be poor. Check actual peak CUDA
allocation, true transformer step execution, and errors/OOM,
before deciding whether to spend roughly an hour or more on
the full 30-step test. The smoke run is not a guarantee that
a longer run and VAE decode will succeed. The full 30-step
command above should only be considered **without --dry-run**
after the user reviews those data and elects the GPU cost.

**Quality criteria if full test is later authorized:** 33 frames
from one native generation; robot fully visible in the same
9:16 crop; recognizable alternating foot placement with no
abrupt scene changes; stable relative camera/subject framing;
no sampled nonfinite FP32 VAE output. Compare all 33 frames
against the established 16-frame full-CFG6 result. Interpolating
to 24 FPS does not improve the underlying native gait and
does not extend runtime.


### Confirmed 33-frame hardware smoke test (2026-10-10)

**Phase A and B have now been run by the user**, on the
Windows GTX 1660 SUPER 6GB target:

- `git pull` current; **124/124 pytest tests passed in
  28.24 seconds**.
- The 33-frame, 30-step full-CFG6 `robot-walk` **dry run**
  returned 720×480, 33 native frames, 8 FPS, duration
  **4.125 seconds**, `--latents-only`, `cfg_guided_steps=null`,
  `quality_validated=false`, `time_validated=false`.
  The effective prompt and negative prompt matched the 16-frame
  profile test.
- The real `--frames 33 --steps 1 --cfg-scale 6 --latents-only`
  smoke run completed 1/1 transformer step without OOM using
  sequential CPU offload. No VAE and no MP4 were involved.
- Saved `outputs/robot-walk-33frames-smoke.latents.safetensors`
  with shape **[1,9,16,60,90]**, matching 33 native frames after
  CogVideoX temporal compression. The sampled final latent
  tensor was **0% nonfinite**.
- Peak **PyTorch CUDA allocated**: **1.336 GiB**, vs 0.763 GiB
  in the earlier 16-frame 30-step latents-only run. This metric
  is *not total device residency* and should not be treated as
  a VRAM/OOM guarantee across 30 steps.
- Model load **107.84s**; one-step pipeline (including setup)
  **264.17s**; total **372.52s (~6m13s)**. The progress bar
  showed ~247.21s, but first-step timing includes setup and
  cannot be multiplied by 30 as a reliable runtime estimate.
- Current conclusion: **33 native frames are feasible for one
  full-CFG6 transformer step on this GPU**. Video quality,
  30-step completion, VAE FP32 RAM and runtime remain untested.

**Next quality decision (separate, opt-in, expensive):**
A complete 33-frame/30-step run is now technically justified
*for measuring gait continuity*, but it may take **well over one
hour** for diffusion and several additional minutes for CPU
FP32 decoding. If approved, run the same Phase A command
**without `--dry-run`** and keep `--latents-only`; do not
decode smoke-run latents for quality. This will save
`outputs/robot-walk-33frames.latents.safetensors` and its JSON
without VAE, allowing decoding later. Check real temporal shape,
timing, numerical checkpoints and failures before deciding to
decode. Keep the successful 16-frame MP4 as the quality baseline.

No new generation default, frame preset, offloading strategy, or
selective CFG status is changed by this result.


### Confirmed full 33-frame diffusion on GTX 1660 SUPER (2026-10-10)

The user completed the previously planned **33-frame / 30-step**
`robot-walk` experiment with full CFG=6 (no selective CFG), seed 42,
720×480 native, 8 FPS, sequential CPU offload and
`--latents-only`. This was **the same positive/negative scene
profile** as the 16-frame quality reference; temporal length was
the controlled change.

**Actual measured outcome:**
- 30/30 transformer steps completed, without an OOM or reported
  execution error. `cfg_schedule=null`.
- Saved
  `outputs/robot-walk-33frames.latents.safetensors`
  and `outputs/robot-walk-33frames.latents.safetensors.json`.
  Latent shape **[1,9,16,60,90]**, correct for 33 native frames.
- Latent checkpoints **1, 16, 30 sampled 0% nonfinite**.
  This is not a VAE result; no visual output or MP4 exists yet.
- Prompt setup/first step **254.783s**. Later 29 steps averaged
  **210.925s**; latent completion **6371.615s**.
- Load **120.71s**, inference **6371.65s**,
  overall **6493.03s (1h48m13s)**.
- Peak PyTorch CUDA allocated **1.348 GiB** (not equivalent to
  total VRAM resident/used). No VAE ran.
- Earlier same-profile 16-frame CFG6/30-step latent generation:
  **2434.69s**, mean subsequent step **76.061s**,
  peak CUDA allocation **0.763 GiB**.
  Doubling temporal length 16→33 (2.0625×) increased
  measured generation wall time ~**2.667×**;
  average post-initial step ~**2.773×**.
  This provides an actual non-linear cost observation rather than
  the previous extrapolation. No model upgrade or kernel speedup
  is demonstrated by these numbers.

**Current conclusion:** the 33-frame native diffusion completed
on the tested 6 GB system and produced finite sampled latents.
The resulting 4.125-second video and quality of gait are **not
yet validated**; the 1-step smoke test and the 30-step latent
report have no playable video. Keep the successful 16-frame
`robot-fullbody-walk-fp32.mp4` as visual reference.

**Next safe action, no further expensive diffusion:**

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\robot-walk-33frames.latents.safetensors `
  --device cpu `
  --aspect 9:16 `
  --focus-x 0.50 `
  --fps 8 `
  --output outputs\robot-walk-33frames-fp32.mp4 `
  --dry-run
```

If the tests and preflight succeed, run the same decode command
**without `--dry-run`**. This invokes the FP32 VAE only,
not the transformer. The previous 16-frame CPU FP32 VAE took
376.63s total; at 33 frames expect greater CPU RAM/time but
do **not** assert any new runtime or memory guarantee.
FP32 is preferred to the previously measured FP16 VAE output,
which had nonfinite values. Inspect the 33-frame MP4 and its
numerical-quality JSON before considering 49 frames, CFG
schedules, or artificial delivery interpolation.

**Visual evaluation:** compare 16-frame and 33-frame MP4s for
one centered fully visible robot, recognizable alternating feet
and arms, stability of camera and robot identity, background
continuity, and absence of new malformed motion. A longer
duration is not proof of an improved walking cycle. No changes
to production defaults or generation settings have been made.


### Final quality result: 33-frame full CFG6 FP32 video (2026-10-10)

The user completed the 33-frame saved-latents VAE FP32 CPU decode
after **126/126 local pytest tests passed (31.46 s)** and a successful
decode dry run verifying saved shape **[1,9,16,60,90]**, metadata
`frames=33`, seed 42, CFG 6, and `scene_profile=robot-walk`.
The final files are:

- `outputs/robot-walk-33frames-fp32.mp4`
- `outputs/robot-walk-33frames-fp32.mp4.json`

**Measured decode:** CogVideoX FP32 VAE on CPU, native decode
`[1,3,33,480,720]`, output 360×640 (9:16 portrait,
`focus_x=0.50`), **33 frames at 8 FPS = 4.125 s**.
VAE decode **658.60 s**; total decode **662.64 s (11m02.64s)**.
Previously established 33-frame denoising was **6493.03 s**,
so combined diffusion + decode took **7155.67 s
(1h59m15.67s)** across the separate operations.
No transformer inference was repeated during CPU decoding.

**Numerical checks:** raw FP32 VAE tensor sampled
**0% nonfinite**, raw sample mean **-0.70037675**, sampled
fraction **below -1 = 0.115066 (11.5066%)**.
Output preflight: `status=plausible`, 0% sampled nonfinite,
mean sampled pixel value **38.333/255**.
For the prior 16-frame FP32 clip, raw sample fraction below -1
was **0.012399 (1.2399%)**, while sample mean was ~-0.69675.
These are *sampled tensor diagnostics*, not a measurement of
the black subject's total pixel area or a causal proof of
why the robot is a silhouette. Values outside the display
range can be clipped by postprocessing.

**Visual inspection of all 33 decoded MP4 frames**, compared
against the 16-frame full-CFG6 robot-walk MP4:

- A single full-body robot is consistently centered; head,
  arms, legs and feet are generally within the portrait crop.
  No nearby vehicle occludes it. The saturated blue wet
  pavement remains consistent across the sequence.
- Across 4.125 seconds there is clearly more ongoing movement
  than in the original 2-second version, with visible alternating
  leg positions and arm motion. The gait is **not fully natural**;
  hand/arm pose looks exaggerated in some early frames, and
  no objective guarantee of smooth foot contacts is available.
- **Important quality regression:** the robot is now a
  nearly black silhouette, with very little face or metallic
  detail; the 16-frame version has much better mechanical
  detail and a visibly illuminated head. More native frames
  did not preserve the desired subject lighting and identity.
- A *post-export diagnostic gamma lift* of frames 1, 9, 17,
  25 and 33 brightened the paving substantially and exposed
  only coarse arm/leg shapes, not the missing mechanical
  textures/illuminated face. This does not prove the decoded
  latent image has no hidden detail, but simple exposure
  correction is not a satisfactory restoration.

**Decision:** 33 native frames are technically feasible on
this GTX 1660 SUPER 6 GB and do offer more time for motion,
but the **visual result does not graduate as a new quality
baseline**. Keep the more detailed 16-frame CFG6 robot-walk
clip as the **appearance/reference image quality baseline**,
and retain 33 frames as a separate **duration/movement**
comparison. Do not switch generation defaults; do not
automatically promote selective CFG; do not launch 49
frames merely to gain duration.

**Next quality experiment, only after a new plan and user
approval:** retain the validated 16-frame / 30-step /
full-CFG6 configuration and test one clearly delimited change
to subject lighting/composition in the positive prompt
(e.g. bright, front-lit robot with visible metallic facial
details and blue accents, avoiding silhouette/backlighting).
Prefer a new opt-in profile rather than overwriting the
working `robot-walk` reference. Save latents, use VAE FP32 CPU,
and inspect detail *before* spending ~2 hours on another
33-frame test. Text prompting is not a guarantee of outcome;
future success criteria must include robot identity, facial
detail and visible alternating footfalls independently.


### New opt-in lighting-only composition experiment: `robot-walk-lit` (2026-10-10)

The most recent 33-frame / 30-step full-CFG6 robot-walk video
achieved **4.125 seconds of native motion**, but visual review
found a **nearly black robot silhouette** with diminished metallic
and facial detail. A simple gamma/brightness lift brightened the
environment more than it restored the character. The previous
16-frame robot-walk full-CFG6 video remains the most detailed
appearance baseline, with body and face correctly centered.

A new **separate opt-in** `--scene-profile robot-walk-lit` was
added to `zigvideo.scene_profiles`; the existing `robot-walk`
profile remains **byte-for-byte unchanged**. The new positive
prompt inherits the original robot's full-body framing,
alternating left/right steps, stationary camera, wet night
street and neon background, and makes **only a lighting and
metal-detail semantic change**: soft diffused frontal key light,
visible silver metallic body panels, and illuminated blue LED
eyes are requested early in the prompt (reducing potential
end-of-prompt truncation), with balanced exposure and lit legs
reiterated at the end. The negative prompt extends the
original with bans on silhouette, blacked-out robot detail,
unlit face and excessive backlighting. No changes to
transformer weights or inference scheduling were made.

The added CPU-only tests check that the original positive
prompt can be reconstructed *exactly* from the lit version
by reversing its two lighting edits; all previous movement
and framing phrases remain untouched. They also check CLI
opt-in behavior, incompatible options, no-LTX enforcement,
effective dry-run prompts, CFG6/30-step/16-frame settings
and safetensors metadata. **This profile has not been
visually tested, and lighting is not guaranteed by text.**

**No-GPU validation step (compare only 16 frames first):**

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --scene-profile robot-walk-lit `
  --preset ultra-safe `
  --frames 16 `
  --aspect 9:16 `
  --steps 30 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --latents-only `
  --output outputs\robot-walk-lit-16frames.mp4 `
  --dry-run
```

The dry run should show `scene_profile="robot-walk-lit"`,
early `diffused frontal key light` and `silver metallic body
panels`, source `num_frames=16`, `num_inference_steps=30`,
`guidance_scale=6.0`, 8 FPS / 2 seconds, no selective CFG,
`latents_only=true`, and `save_latents=true`. The existing
`robot-walk` profile remains available, as does normal
`--prompt` usage.

**Do not remove `--dry-run` automatically.** A later real
16-frame/30-step CFG6 run would take approximately 40 minutes
for diffusion **based on the earlier measured reference**;
the separate CPU FP32 decoder previously took ~6 minutes.
Timing and visual improvement are not guaranteed. If a real
run is later authorized, generate **latents only** first,
then decode with FP32 CPU, 9:16 and `focus_x=0.50`.
Compare the new clip directly against the old
`robot-fullbody-walk-fp32.mp4` on the same 16-frame duration
and setting set: face/eye legibility, silver mechanical
detail, no silhouette, feet visible, camera stability and
walking motion. If lit 16 frames fail to improve appearance,
**do not** spend another ~2 hours on lit 33 frames or
jump to 49 frames.

This is a prompt-control experiment, not a change to
application defaults or an approval of selective CFG.


### Result: robot-walk-lit 16-frame/30-step full CFG6 latents (2026-10-10)

The user ran the planned `--scene-profile robot-walk-lit` experiment
on Windows with GTX 1660 SUPER 6 GB, preserving the established
16-frame reference settings: CogVideoX-2B native FP16 transformer,
720×480, 16 frames at 8 FPS, seed 42, full CFG=6, 30 steps,
sequential CPU offload, `--latents-only`. Only the profile's
lighting-focused positive and negative prompts were changed from
`robot-walk`.

- All **30/30 transformer steps completed**, with no reported OOM.
- Checkpoints at steps **1, 16, and 30** sampled **0% nonfinite**
  latents, shape **[1,4,16,60,90]**, final mean absolute latent
  value **1.15937209**.
- Model loading **111.20 s**; inference **2247.59 s**;
  elapsed total **2359.20 s (39m19.20s)**.
- Mean subsequent step time **73.335 s**; PyTorch CUDA peak
  allocated **0.763 GiB** (not total device memory residency).
- Final output is *only* the reusable latent tensor at
  `outputs/robot-walk-lit-16frames.latents.safetensors`
  and its report at
  `outputs/robot-walk-lit-16frames.latents.safetensors.json`.
  The report's quality field remains
  `not_evaluated_latents_only`.

**This does not yet demonstrate a visual lighting improvement.**
Numerically finite latents are not decoded frames; the robot,
face and lighting cannot be assessed until CPU FP32 VAE output is
visually inspected. Keep the earlier CFG6 `robot-walk` 16-frame
MP4 as the reference.

The next step avoids rerunning diffusion and uses the
previously successful CPU FP32 VAE path:

```powershell
.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\robot-walk-lit-16frames.latents.safetensors `
  --device cpu `
  --aspect 9:16 `
  --focus-x 0.50 `
  --fps 8 `
  --output outputs\robot-walk-lit-16frames-fp32.mp4
```

Compare the new 2-second MP4 against
`outputs/robot-fullbody-walk-fp32.mp4` for head and facial
details, blue eyes, illuminated metal body, framing and gait.
Only after examining video and JSON numerical metrics can the
lighting hypothesis be accepted or rejected. Do not expand to
33 or 49 frames before that decision.


### robot-walk-lit 16-frame FP32 CPU decode: numerical result (2026-11-06)

After the validated 16-frame/30-step/full-CFG6 latents-only
`robot-walk-lit` generation, the user completed a **decode-only**
CogVideoX FP32 VAE run on CPU, without repeating diffusion:

```powershell
.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\robot-walk-lit-16frames.latents.safetensors `
  --device cpu `
  --aspect 9:16 `
  --focus-x 0.50 `
  --fps 8 `
  --output outputs\robot-walk-lit-16frames-fp32.mp4
```

Confirmed outputs (user's terminal report):
- `outputs/robot-walk-lit-16frames-fp32.mp4`
- `outputs/robot-walk-lit-16frames-fp32.mp4.json`
- **16 frames at 8 FPS**, 9:16 center crop, expected
  360×640 portrait export, **2.0 seconds**.
- Raw FP32 VAE shape `[1,3,16,480,720]`, sample count
  **65,569**, **0.0% sampled nonfinite**, sample mean
  **−0.5818799138**, sampled fraction below −1
  **0.003447 (0.3447%)**.
- Postprocessed numerical quality `plausible`,
  **0.0% nonfinite** in the quality check,
  mean pixel value **53.301/255**, maximum 255/255,
  `near_black_pixel_fraction=0.0`.
- VAE **391.11 s**; model load 3.43 s,
  postprocess 0.90 s, export 3.01 s,
  decode total **398.99 s (6m38.99s)**.
  Combined with the prior `robot-walk-lit` latents-only
  generation of 2359.20 s, this experiment took
  **2758.19 s (45m58.19s)** across two completed runs.

**Same-length 16-frame comparison with `robot-walk` (not lit):**
Earlier FP32 VAE raw sampled mean was **−0.69675076**,
fraction below −1 **0.012399 (1.2399%)**, and
postprocessed average pixel value **38.779/255**.
The lit variant's postprocessed image mean is
**14.522/255 higher (~37.45%)**, while the below−1
raw fraction fell to 0.3447%. These are whole-video
numeric diagnostics, not per-subject brightness or texture
measurements. They cannot demonstrate that the eyes,
face, metal plates or footfalls actually improved.

**Current quality verdict is PENDING VISUAL INSPECTION.**
Only the terminal log and numeric report have been supplied
for this run; **the newly decoded MP4 has not yet been
uploaded or visually inspected**. Do not claim that the
lit profile passes the appearance acceptance criteria yet.
Request the actual `robot-walk-lit-16frames-fp32.mp4`
(and optionally its JSON) and compare its frames side by
side with the original `robot-fullbody-walk-fp32.mp4`.
Do not begin an expensive 33-frame lit experiment before
that visual gate. The Hugging Face unauthenticated/deprecated
symlink warnings were nonfatal in this completed decode.


### Visual verdict — uploaded robot-walk-lit 16-frame FP32 MP4 (2026-11-06)

The user uploaded the actual decoded
`robot-walk-lit-16frames-fp32.mp4`. We extracted all **16 frames**
at **360×640, 8 FPS, 2.0 seconds**, and compared corresponding
frame indices (1, 4, 8, 12, 16, plus a nine-frame temporal strip)
with the previously uploaded 16-frame reference
`robot-fullbody-walk-fp32.mp4`, using the same displayed frame
size and frame indices. This supersedes the earlier
**PENDING VISUAL INSPECTION** note.

**Visually confirmed wins:**
- A single clearly recognizable silver humanoid robot with
  strongly lit metallic face/helmet, torso, chest plating,
  joints and legs. Blue illuminated facial elements and
  glowing blue hands are clearly visible. Do not claim that
  *two distinct* individual blue eyes are identifiable.
- The black-silhouette issue seen in the 33-frame
  `robot-walk` test is absent in the 16-frame lit clip.
  The numeric image mean change from 38.779 to 53.301/255
  is consistent with a brighter clip, but is **not itself
  a subject-only illumination metric**, especially because
  subject size changed.
- One character remains visually consistent across the
  16 frames; visible limb motion. No vehicles obstruct it.
  The wet neon street background remains recognizable.

**Quality regressions / limitations:**
- **Framing/scale regressed materially.** The original
  16-frame `robot-walk` depicts a smaller full-body
  robot with substantial surrounding pavement and margins.
  The lit profile generates a much larger, near-full-height
  robot with feet touching or almost touching the **bottom
  9:16 frame edge**, especially in the first frames.
  This undermines the full-body portrait safe-zone goal
  and leaves little room for a longer walk.
- Motion remains somewhat mechanical/stiff. Alternating
  leg placement and arm motion are present, but natural
  heel-to-toe ground contacts and continuous physically
  plausible gait have **not** been demonstrated in 2 seconds.
- The prompt describes `two blue LED eyes`, but the
  visual output supports **bright blue facial illumination**,
  not a certain count of two distinct eye lights.
- Neither the 16-frame reference nor `robot-walk-lit`
  should be called a complete final-quality Short; the
  reference is the framing baseline and the lit video
  is now the **robot appearance/lighting baseline**.

**Decision:** The lighting hypothesis **passed visually**,
but the new profile **did not preserve the original wider
framing**, despite unchanged framing text. This means
the 16-frame lit video is a successful lighting experiment,
not acceptance of the whole scene. Do **not** blindly
promote it to 33 frames (~2 hours based on previous
measurement) or try 49 frames.

**Suggested next controlled experiment (not yet generated):**
Create a *separate* opt-in `robot-walk-lit-wide` profile,
preserving the successful frontal lighting and metallic
detail instructions while more strongly requesting a
wide fixed-camera view, a modest-sized robot, substantial
visible pavement below both feet, and headroom. Test
again at **16 frames / 30 steps / CFG6 / seed42 / sequential
offload**, saving latents and decoding in CPU FP32 as before.
Compare both **face detail** and **top/bottom margins**;
only proceed to 33 frames if *both* criteria pass.
Text-based framing is a hypothesis, not a guarantee.
No such new 30-step GPU job has been authorized or started.


### New opt-in `robot-walk-lit-wide` framing experiment (2026-11-06)

After inspecting the uploaded 16-frame `robot-walk-lit` MP4,
we confirmed the frontal key-light and metallic-face appearance
improved, **but the robot grew to nearly full frame height** and
the feet almost touched the bottom edge. This is a composition
regression compared with the smaller, well-framed original
`robot-walk`. The lit video is therefore the lighting baseline;
the original remains the framing baseline.

**A third opt-in `--scene-profile robot-walk-lit-wide` is now
available.** No existing prompt, preset, model, CFG setting,
frame default, offloading option, or VAE path changed.
The new prompt is derived mechanically from the *already
successful* `robot-walk-lit` prompt via exactly **two edits**:
(1) the early subject phrase asks for a **wide static camera
shot** instead of a general shot, retaining the front key-light,
silver metallic panels, and two requested blue LED eyes;
(2) the earlier `Full-body wide shot` phrase now explicitly
requests the robot at **about half the image height**, viewed
from several meters away, with generous headroom and a clear
strip of wet pavement below the shoes.

Everything else, including the alternating left/right
footsteps, center-of-native-frame 9:16 safe-zone language,
fixed tripod camera, and backlit neon wet street, stays as
in `robot-walk-lit`. The negative prompt only *extends* the
lit profile with exclusions for oversized subjects, tight
framing, missing headroom, and feet on the bottom edge.
A reversible textual test confirms that undoing the two
framing changes reproduces the previous lit positive
prompt exactly. This design **isolates the intended prompt
variable, not the actual stochastic model response**.

**Safety/acceptance gate:** first check 16 native frames,
30 steps, full CFG6, seed 42, 8 FPS, sequential CPU offload,
9:16 crop. Save final latents (no FP16 VAE); decode later
in CPU FP32. Target **one silver, readable, front-lit robot
occupying roughly half the portrait height, with both feet
well above the frame bottom and headroom**, still showing
alternating steps without subject/car obstructions.
Do not accept better framing alone if facial/metallic detail
collapses. Text prompting cannot guarantee either criterion.

**Run the inexpensive validation first (no models loaded):**

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --scene-profile robot-walk-lit-wide `
  --preset ultra-safe `
  --frames 16 `
  --aspect 9:16 `
  --steps 30 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --latents-only `
  --output outputs\robot-walk-lit-wide-16frames.mp4 `
  --dry-run
```

Expected: `scene_profile=robot-walk-lit-wide`,
`native_generation={width:720,height:480,num_frames:16,
fps:8,num_inference_steps:30,guidance_scale:6.0}`,
`source_duration_seconds=2.0`, `latents_only=true`,
`save_latents=true`, `cfg_guided_steps=null`,
`cfg_schedule` absent/not enabled. This phase performs
**zero diffusion/GPU work**.

**Do not automatically remove `--dry-run`**. A full
16-frame/30-step run of this third prompt would again
consume roughly **40 minutes for latents plus 6–7 minutes
for CPU FP32 VAE**, based on prior real runs, with no
guarantee of quality. After successful local tests and
preflight, a later full run is a separate deliberate
decision, not part of implementation. Decode any saved
latents with `zigvideo decode --device cpu --aspect 9:16
--focus-x 0.50 --fps 8` to a *new* `...-fp32.mp4`
filename. Keep both existing reference MP4s unchanged.

Do **not** spend another ~2 hours on 33-frame output or
attempt 49 frames until a 16-frame lit-wide result passes
both framing and appearance criteria. More frames may
produce different composition; 16-frame success still
doesn't guarantee 33-frame quality.


### Local validation complete: robot-walk-lit-wide (2026-11-06)

The user pulled commit `370e987` on Windows and completed
the entire local pytest suite: **139 passed in 11.77 seconds**
(no failures). All seven newly introduced `test_wide_scene_profile.py`
tests passed.

The user also ran the exact planned `robot-walk-lit-wide` CogVideoX
`--dry-run`: **16 native frames, 8 FPS (2.0 seconds), 720×480,
30 inference steps, full CFG6, seed 42, sequential CPU offload,
latents-only and save-latents enabled**, no selective CFG. The
preview contained explicit silver metallic panels, frontal
key-light, 9:16 centered portrait-safe crop, robot at roughly
half picture height, and wet pavement below both feet. It selected
output `outputs/robot-walk-lit-wide-16frames.mp4` as the
latents-only output stem.

**Status:** the option and parameter plumbing are validated.
The `dry-run` did *not* generate a video, check GPU consumption,
or verify visual framing and illumination. A future real run
must deliberately omit `--dry-run`, and will save
`outputs/robot-walk-lit-wide-16frames.latents.safetensors` and
its JSON; it **will not** produce MP4 until the separate CPU FP32
VAE decode. The displayed `vae_precision=float16` in the
generation preview is non-operative in `latents-only` mode;
the separate decode remains float32 by design. The tested
16-frame baseline suggests roughly 40 minutes GPU diffusion,
plus roughly 6–7 minutes CPU FP32 decoding, without a guarantee.
Do not repeat the 139 tests or one-step smoke check purely for
routine; the remaining question is visually whether the
robot can be both smaller and brightly detailed.


### Full robot-walk-lit-wide 16-frame generation: measured result (2026-11-06)

The user completed the planned full 30-step generation on Windows
GTX 1660 SUPER 6GB using `--scene-profile robot-walk-lit-wide`,
CogVideoX-2B native FP16 transformer, 720×480, 16 frames at 8 FPS,
full CFG6, seed 42, sequential CPU offload and `--latents-only`.

**Numerical/runtime result:**
- All **30/30** diffusion steps completed without a reported OOM.
  Latent checkpoints **1, 16, 30** each sampled **0% nonfinite**;
  final shape **[1,4,16,60,90]**.
- Output:
  `outputs/robot-walk-lit-wide-16frames.latents.safetensors`
  and `outputs/robot-walk-lit-wide-16frames.latents.safetensors.json`.
  **No MP4 was generated; VAE not run.**
- Model load **115.19s**, diffusion **4635.46s**, total
  **4751.44s (1h19m11.44s)**; later-step mean
  **148.917s**; PyTorch peak CUDA allocated **0.763 GiB**
  (not total device residency).
- Previous otherwise-matching `robot-walk-lit` 16-frame
  full CFG6 run took **2359.20s (39m19.20s)**, including
  **2247.59s** inference and **73.335s** mean later-step.
  The wide run was ~**2.01× slower overall**.
  Large variations in individual step durations
  (including several >300s) mean the source of the
  slowdown **cannot be concluded from these logs alone**.

**Important discovered prompt-limit issue:**
During actual generation, transformers warned:
`Token indices sequence length is longer than the specified
maximum sequence length for this model (241 > 226)`, and
diffusers confirmed that the final prompt suffix was truncated:
`background, balanced exposure with face, mechanical details
and both legs clearly lit.`.
The saved prompt string in the JSON is the original *requested*
full string, **not proof that the text encoder processed all
its tokens**. The early positive instructions for distant
wide framing, robot size, pavement below feet, frontal light,
silver panels and blue LED eyes appear *before* the truncated
tail. So this completed experiment still tests those early
instructions, but **must be labeled as a partially truncated
prompt**, and should not be claimed to validate balanced
exposure wording that was dropped. The 0% nonfinite latent
check does not say anything about visual fidelity.

**Next action (no repeat diffusion):**

```powershell
.\.venv\Scripts\zigvideo.exe decode `
  --latents outputs\robot-walk-lit-wide-16frames.latents.safetensors `
  --device cpu `
  --aspect 9:16 `
  --focus-x 0.50 `
  --fps 8 `
  --output outputs\robot-walk-lit-wide-16frames-fp32.mp4
```

The previous 16-frame CPU FP32 VAE took about 6–7 minutes,
but actual runtime may vary. Examine the resulting MP4 and
numeric JSON against both the original `robot-walk` (small,
well-framed robot) and `robot-walk-lit` (bright metal face,
near-edge feet). Evaluate size, headroom, pavement below feet,
face/body visibility and foot placement across all 16 frames.

**Future fix, not a reason to discard these saved latents:**
introduce tokenizer-aware preflight visibility or shorten
the wide profile under the real model's 226-token budget,
with regression tests, **before any further expensive
prompt-dependent generation**. Avoid silently changing
the profile used for this already saved experiment; preserve
its exact text as provenance. Do not advance to 33 or 49
frames before the decoded 16-frame result is visually reviewed.


### Visual assessment of uploaded robot-walk-lit-wide FP32 MP4 (2026-11-07)

The user completed the CPU FP32 VAE decode, uploaded the real
`robot-walk-lit-wide-16frames-fp32.mp4`, and supplied its
numerical report. We inspected every one of its 16 frames
(360×640, 8 FPS, 2 seconds) beside the two previously decoded
16-frame controls:
`robot-fullbody-walk-fp32.mp4` (original `robot-walk`) and
`robot-walk-lit-16frames-fp32.mp4` (lighting-only version).

**Decode observations:**
- Native FP32 VAE tensor `[1,3,16,480,720]`;
  sampled nonfinite **0%**, sampled raw mean
  **−0.6512075663**, sampled raw fraction below −1
  **0.158551 (15.8551%)** (not a subject quality metric).
- `quality.status=plausible` is numerical only;
  sampled mean pixel value `44.686/255`. For comparison,
  prior `robot-walk-lit` was `53.301/255` and
  original `robot-walk` was `38.779/255`, but the
  differing subject scales prevent these whole-video means
  from being treated as subject-specific brightness scores.
- VAE decode **363.06s**; total decode **367.09s
  (6m07.09s)**, using CPU with `focus_x=0.50`.
  Full prior generation **4751.44s**, combined run time
  **5118.53s (~1h25m18.53s)**.

**Visual verdict: silver robot preserved, wide shot NOT achieved.**
All 16 frames consistently show one clear front-lit
blue/silver metallic robot; face and mechanical details
are visible, blue face lights present, no vehicle occlusion,
subject stays near the center and feet are visible.
Compared with the near-full-frame `robot-walk-lit`,
`robot-walk-lit-wide` allows slightly more apparent room at
some moments, but the subject still occupies **most of the
portrait height**, with only a narrow margin under feet
in several frames. It is not the requested roughly half-height
long shot. The original `robot-walk` is still far smaller
and has generous head and foot room. Gait has changing leg
positions and arm movement, but still looks stiff; a
well-resolved continuous footfall pattern is not confirmed.
Thus neither the requested scale nor a high-quality walking
cycle was established. The overall brightness is between the
two earlier profiles, and the face and metal are legible.

**Prompt provenance caveat:** this is the run where positive
prompt was **241 tokens vs model limit 226**, truncating the
last phrase about balanced exposure. The early wide-shot
and frontal-light requests were before the cut, but cannot
be presumed to have guaranteed composition. The exact
original text remains unchanged in the profile and reports
for reproducibility. Do not overinterpret the 15.8551%
raw-VAE-below−1 sample fraction as the cause of zoom or
cropping.

**Decision:** preserve the original 16-frame clip as
framing baseline and `robot-walk-lit` as best bright-robot
appearance baseline. The 16-frame lit-wide clip is a
useful comparison but **fails the target 50%-frame-height
framing gate**. Do not spend another ~2 hours on 33 frames
with this profile or jump to 49. Correct the prompt
token-limit workflow and simplify prompts before any
further expensive generation.

### CogVideoX exact 226-token T5 preflight (2026-11-07)

Added `zigvideo/prompt_preflight.py` to guard against the
actual 226-token text encoder cutoff in the upstream
CogVideoX Diffusers pipeline. It uses the **locally cached
THUDM/CogVideoX-2b T5Tokenizer** with
`add_special_tokens=True, truncation=False, padding=False`.
Both the positive and negative prompts must fit. It counts
model tokens, **not English words or characters**; it
rejects 227+ tokens with an actionable error before loading
the video pipeline. This also prevents a silent mismatch
between the full prompt recorded in latent metadata and
the truncated text actually provided to the model.

**Real CogVideoX `generate` commands now enforce this
check automatically before loading the transformer/VAE
weights.** `generate --dry-run` continues to work without
the tokenizer or any model components for backward
compatibility. Add `--validate-prompt-tokens` to a
CogVideoX dry-run to opt in to the exact offline check.
This fetches **no model weights and makes no internet
requests**; it requires the tokenizer to already exist in
`--cache-dir`. If the tokenizer isn't in the specified cache,
the checked dry-run / real generation fails closed with
an error explaining the missing tokenizer. This is
intentional to avoid a costly truncated generation.

To inspect the existing *already tested* (truncated)
wide profile, use **only the checked dry run**:

```powershell
git pull
.\.venv\Scripts\python.exe -m pytest -v

.\.venv\Scripts\zigvideo.exe generate `
  --backend cogvideox `
  --scene-profile robot-walk-lit-wide `
  --preset ultra-safe `
  --frames 16 `
  --aspect 9:16 `
  --steps 30 `
  --seed 42 `
  --offload sequential `
  --cfg-scale 6 `
  --latents-only `
  --output outputs\robot-walk-lit-wide-token-check.mp4 `
  --validate-prompt-tokens `
  --dry-run
```

**Expected for this old wide prompt: fail before any model
load with a positive prompt length beyond 226**. The
previous actual 30-step run warned `241 > 226`; the
checked tokenizer should reproduce that length if the
same cached tokenizer revision is used. A failing
checked dry run is the *desired safety result* for this
historically oversized prompt. The already generated
latents and MP4 remain valid and untouched.

Tests are CPU-only and use fake tokenizers to exercise
boundary (226 vs 227), negative prompt truncation,
missing-cache fail-closed behavior, zero-GPU checked
dry run, legacy dry-run compatibility and prevention
of weight loading on too-long real prompts. No tokenizer
downloads or GPU work is performed by the tests.
Do not claim that a new compact prompt fits until
it passes the real tokenizer checked dry run.
