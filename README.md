# Casual Hand Tracking → VMC

Webcam-only arm/hand tracking that outputs the VMC Protocol, for plugging into
Warudo / VNyan / OBSKUR as an isolated hand+arm tracker (no glove, no VR
controller). Full design rationale lives in the original planning doc; this
README tracks what's actually implemented.

## Stack

- Electron + `electron-vite` + TypeScript (main / preload / renderer split)
- `@mediapipe/tasks-vision` `HolisticLandmarker` (WASM, GPU delegate) for pose+hand keypoints
- Kalidokit (pose/hand keypoints → bone rotations) — not wired in yet
- Self-implemented VMC/OSC sender over `dgram` (Node has no UDP in the renderer,
  so this lives in the main process) — not wired in yet

## Status vs. build plan

Mapped to the original doc's §7.3 stage numbers:

- [x] **1. Webcam → keypoints**: `HolisticLandmarker` runs in the renderer,
      draws pose + both hands on a canvas overlay, and shows live FPS /
      landmark counts (`src/renderer/main.ts`).
- [ ] 2. Kalidokit: keypoints → bone rotations
- [ ] 3. VMC sender (UDP/OSC) from the main process
- [ ] 4. Drive a test VRM avatar in Warudo/VNyan
- [ ] 5. Occlusion / confidence logging
- [ ] 6. Confidence-based fallback (hold last pose)
- [ ] 7. Joint-angle clamping
- [ ] 8. Hardcoded gesture snapping (2–3 gestures)
- [ ] 9. (optional) Calibration onboarding flow

## Run it

```bash
npm install
npm run dev
```

The main process auto-grants the OS camera permission prompt
(`session.defaultSession.setPermissionRequestHandler`) — the renderer just
calls `getUserMedia` directly. Model + WASM are fetched from Google's/jsDelivr's
CDN at runtime, so the first launch needs network access.

## Structure

```
src/
  main/       Electron main process (window, will host the VMC/UDP sender)
  preload/    contextBridge surface for main <-> renderer IPC (empty for now)
  renderer/   webcam capture + MediaPipe inference + canvas overlay
```

## License notes (carried over from the design doc)

- MediaPipe Tasks Vision: Apache 2.0 — OK commercially.
- Kalidokit: MIT — OK commercially.
- VMC protocol itself is just an OSC message spec; the sender is being
  hand-rolled on top of a permissively-licensed OSC library instead of reusing
  `python_vmc`/`vmcp`, which are AGPL-3.0.
- WiLoR/HaMeR (SOTA occlusion-robust models) are CC-BY-NC-ND — techniques only,
  not the weights, if this ever needs a stronger model.
