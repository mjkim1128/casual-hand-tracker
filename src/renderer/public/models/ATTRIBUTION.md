# Third-party assets in this folder

## VRM1_Constraint_Twist_Sample.vrm

- **Title**: VRM1_Constraint_Twist_Sample
- **Author**: pixiv Inc. (2022)
- **Source**: [vrm-c/vrm-specification](https://github.com/vrm-c/vrm-specification/tree/master/samples/VRM1_Constraint_Twist_Sample) — the official VRM Consortium's own spec-conformance sample set
- **Downloaded from**: `https://raw.githubusercontent.com/vrm-c/vrm-specification/master/samples/VRM1_Constraint_Twist_Sample/vrm/VRM1_Constraint_Twist_Sample.vrm`
- **License**: VRM Public License 1.0 (full terms: https://vrm.dev/licenses/1.0/). This is a *configurable* license — permissions like commercial use, modification, and redistribution are set per-model in embedded "License Settings" metadata rather than being fixed by the license name alone. This file's exact settings haven't been individually parsed; it's used here only as a **local development/testing asset**, not redistributed or shipped in any build. Re-check the embedded settings before this project's build output (as opposed to its source repo) is ever distributed to anyone.
- **Modifications**: none — used as downloaded.

### Why this model

Built by pixiv specifically to test VRM bone *twist constraints* — it wears a short-sleeve T-shirt, so the wrist and hand stay fully visible in any arm pose. Replaced the previous placeholder, AvatarSample_B (removed), whose jacket sleeves hid the hands during wrist-rotation debugging — exactly the problem this project needed to see clearly.

Used purely as a placeholder test avatar to validate the tracking → VMC pipeline end-to-end — not shipped as part of any released build.

## depth-anything-v2-small/ (config.json, preprocessor_config.json, onnx/model_quantized.onnx)

- **Title**: Depth Anything V2 — Small
- **Author**: DepthAnything team (Lihe Yang et al.); this ONNX export republished by the `onnx-community` Hugging Face org
- **Source**: [depth-anything/Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) (original weights); [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small) (this ONNX conversion, downloaded from)
- **License**: Apache License 2.0 — the Small checkpoint specifically (Base/Large/Giant checkpoints of the same model are CC-BY-NC-4.0 and are *not* used here)
- **Modifications**: none — used as downloaded (dynamic-quantized `model_quantized.onnx` variant, ~27MB, chosen for CPU/WASM inference speed over the full-precision ~100MB `model.onnx`)

### Why this model

Monocular depth estimation, run directly on the webcam frame — used to supplement MediaPipe's own pose Z-depth estimate (a known weak point, see main.ts) with an independent, image-based depth signal for the tracked hand/wrist, particularly for a hand reaching toward the camera.
