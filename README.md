# Casual Hand Tracking

Webcam-only arm/hand tracking that drives a VRM avatar in real time — no
glove, no VR controller, no depth camera. One `HolisticLandmarker` (pose +
hands) and one standalone `HandLandmarker` read a single 2D webcam feed;
a bone-length-constrained two-bone IK solver turns that into stable
shoulder/elbow/wrist joints; a small rigging layer aims the VRM's bones at
those joints.

The tracking core (`src/renderer/armTracking.ts`) has no dependency on
Three.js, VRM, or Electron — it is a plain TypeScript module that takes one
frame of MediaPipe landmark data in and returns resolved 3D joint positions
out. See [Using the tracking core as a module](#using-the-tracking-core-as-a-module).

## How it works

- **Direction from the image, length from the world.** MediaPipe's world
  landmarks carry accurate metric bone *length* but a noisier in-plane
  *direction* than the 2D image landmarks. Each upper-arm/forearm segment's
  on-screen direction is taken from the image landmarks; its length comes
  from the world landmarks (`withImageXY`).
- **Two-bone IK with the person's own bone lengths enforced exactly.** The
  wrist target (direction + distance from the shoulder) drives a
  law-of-cosines solve against the person's measured upper/lower arm
  lengths, so the elbow can never end up at a physically impossible distance
  from the shoulder no matter how noisy the raw landmark is. The one
  remaining degree of freedom — which way the elbow bends around the
  shoulder-wrist axis — comes from the raw elbow landmark, used only as a
  bend-direction hint (the "pole"), never as a trusted position.
- **Ambient self-calibration.** Arm lengths and shoulder width start from
  rough adult averages and drift toward this person's own measurements as
  they move, with an explicit T-pose measurement (the "T포즈 측정" button, or
  holding a genuine full reach and pressing `R`) available to lock in exact
  values immediately instead of waiting for the ambient estimate to
  converge.
- **Confidence-gated holding, everywhere.** A joint below its visibility
  floor holds its last good position rather than jumping to a fresh noisy
  reading; a hand-model wrist reading has to agree with itself across two
  consecutive frames before it's trusted (see `resolveTrustedHandWrist`); the
  bend-plane ("pole") direction is rotated toward its new target at a capped
  angular speed, scaled down further the lower the current confidence is, so
  a single bad frame nudges it only slightly instead of snapping.
- **A standalone hand model, independent of pose.** Holistic's own hand
  branch derives its crop region from the *pose* model's wrist landmark, so
  it goes blind in lockstep whenever that estimate drifts. A separate
  `HandLandmarker` runs directly on the image and is matched to left/right by
  proximity to the pose wrists (not by MediaPipe's own `handedness`, which
  assumes a convention this app's mirrored left/right crossing violates).

## History

- **Scaffold** — Electron + `electron-vite` + TypeScript, `HolisticLandmarker`
  wired up for live pose+hand keypoints on a canvas overlay.
- **Guided calibration + pinhole depth reconstruction** — a puppeted
  calibration flow (T-pose / overhead / arms-forward) measured per-person arm
  lengths, a Z-axis tracking-bias correction, and a two-point hand-size-based
  depth model; arm depth was reconstructed via `sqrt(realLength² - xyLength²)`
  from a calibrated bone length and the on-screen X/Y length, with an
  external "Depth Anything V2 Small" monocular depth model as an independent
  depth signal. **Abandoned**: the reconstruction's sensitivity to landmark
  noise blows up near its own singularity (a segment nearly broadside to the
  camera, `xyLength ≈ realLength`) — worst exactly during ordinary lateral
  arm movement, the opposite of the case it was built for.
- **Self-calibration** — replaced the guided flow with the ambient
  auto-converging estimate described above, and dropped the depth
  reconstruction pipeline (including the external depth model) for a lighter
  approach: trust MediaPipe's own world Z directly, damped rather than
  independently reconstructed.
- **Two-bone IK rework** (this project's largest change) — replaced
  Kalidokit's angle-heuristic pose/hand rigging with the bone-length-enforced
  IK solve, the standalone `HandLandmarker`, and a series of measured-live
  stability fixes: an angular-velocity cap on the pole blend (replacing a
  percentage-per-second blend that let a single bad frame move the elbow a
  large absolute amount), a hand-wrist corroboration gate (a
  freshly-reacquired hand-model wrist has to agree with itself across two
  frames before it's trusted, since the first reading after a tracking gap
  was measured to be the least reliable one), and a direction check on the
  "forced full reach" heuristic (only apply it when the raw target is
  actually displaced toward the camera, not just low-confidence and
  small-looking — see below).
- **Kalidokit, revisited and dropped again.** Kalidokit computes each
  bone's rotation straight from its own pair of raw landmarks (shoulder→elbow
  for the upper arm, elbow→wrist for the forearm) with no bone-length
  enforcement at all — the opposite tradeoff from this project's IK. It was
  reintroduced briefly as a live-toggleable alternate mode to test a
  hypothesis: that the IK's coupling of wrist distance into elbow position
  was *why* a specific pose (upper arm at the side, forearm raised) made the
  upper arm visibly swing. Measured side-by-side, both modes were equally
  unstable in that exact pose — proving the instability is in the *raw
  landmark data* (MediaPipe's wrist/elbow confidence genuinely collapses in
  that pose) and not in which algorithm processes it. The comparison mode was
  removed once it had answered the question; the original handedness-crossing
  bug (Kalidokit's hand solver assumes `side` matches the raw landmark's own
  chirality, which this app's deliberate left/right mirroring violates) was
  never re-solved, since it stopped being relevant.

## The "상완 쏠림" investigation, briefly

A running theme this project spent a long time on: raising the forearm with
the upper arm held at the side made the upper arm visibly swing/lean. In
order, what turned out **not** to be the cause: MediaPipe's Z axis being
specifically misread as "forward" (measured directly — the drift direction
didn't match), the pole vector's world x/y/z mixing, the front-of-torso clamp
(measured to push the elbow by only ~1-2cm, negligible), and the IK
architecture itself (Kalidokit's decoupled approach was equally unstable).
What it actually traced to: the pose model's **wrist landmark confidence**
collapses specifically in that pose (occlusion/foreshortening against the
torso), and that low confidence was being misread by the "forced full reach"
heuristic (meant for a hand pushed right up against the camera) as exactly
that case — snapping the arm to near-full-extension. Requiring the raw
target to actually be displaced toward the camera before that heuristic can
fire (`towardCamera` / `FORWARD_REACH_MIN_Z_RATIO`) fixed that specific snap.
The remaining, smaller residual instability in that pose is the raw landmark
noise itself, which no downstream algorithm change can fully absorb — the
next lever, not yet pulled, is holding the raw wrist/elbow readings more
aggressively once their own visibility drops this low (traded off against a
past regression where doing that froze a genuine camera-facing reach at a
bent-looking position).

## Using the tracking core as a module

`src/renderer/armTracking.ts` exports a self-contained API: feed it one
MediaPipe `HolisticLandmarker` result (plus, optionally, per-hand
`HandLandmarker` results) each frame, get back stable 3D joints. Nothing in
it references Three.js, VRM, or Electron, so it can be reused by any
renderer that wants "MediaPipe landmarks in, stable shoulder/elbow/wrist out"
without pulling in this project's avatar-rendering code.

```ts
import {
  trackingDataFromResult,
  withImageXY,
  resolveTrackingFrame,
  setTrackingDebug,
  setTrackingStatusHandler,
  requestTPoseMeasurement
} from './armTracking'

setTrackingDebug(true) // optional: emit [ik]/[tpose]/[glitch] console diagnostics
setTrackingStatusHandler((text) => console.log(text)) // optional: T-pose/reach-capture progress text

// Per frame:
const holisticResult = holisticLandmarker.detectForVideo(video, now)
let data = trackingDataFromResult(holisticResult)
data = withImageXY(data, video.videoWidth / video.videoHeight)
data.leftHandLandmarks = /* optional: an independent HandLandmarker's left-hand 2D landmarks */
data.leftHandWorldLandmarks = /* ...and its world (metric) landmarks */
// (same for rightHand*)

const dt = (now - lastNow) / 1000
const resolved = resolveTrackingFrame(data, dt)
// resolved.left / resolved.right: ArmJoints | null — { shoulder, elbow, wrist } in world meters
// resolved.leftHand / resolved.rightHand: Vec3Like[21] | null — smoothed hand landmarks
// resolved.left/rightConfident: was this frame's wrist a trustworthy live reading, or held?

// Optional: trigger auto T-pose calibration (locks in exact arm lengths from
// a 2-second held T-pose) instead of waiting for the ambient estimate to converge.
requestTPoseMeasurement()
```

Everything else in the module (visibility gating, bone-length outlier
rejection, the hand-wrist corroboration gate, the reach-forward heuristic,
ambient self-calibration) runs automatically inside `resolveTrackingFrame` —
there is no separate calibration object to manage.

## Strengths and limitations

**Strengths**

- Single 2D webcam, no depth sensor, no VR hardware, no manual measurement
  step (self-calibrating, with an optional explicit T-pose for instant exact
  values).
- Bone lengths are always physically consistent by construction — the elbow
  can never render at an impossible distance from the shoulder, however
  noisy the raw landmark.
- Confidence-gated holding and corroboration gates make brief tracking
  dropouts and hand-model reacquisition largely invisible in the output.
- The tracking core is a dependency-free module, reusable outside this
  project's specific Electron/VRM setup.

**Limitations**

- MediaPipe's Z axis (depth from the camera) is the least reliable of the
  three on a single 2D webcam — this is a hardware-fundamental limit, not
  something any post-processing fully removes.
- The two-bone IK derives the elbow's bend *angle* purely from how far the
  wrist target is from the shoulder — so wrist-landmark noise can still
  visibly move the whole arm even when the elbow's own landmark is fine.
  Measured directly against Kalidokit's decoupled (no bone-length
  enforcement) approach, this isn't actually an IK-specific weakness — both
  architectures were equally unstable under the same low-confidence input —
  but it means the fix has to be "trust the raw landmark less when its own
  visibility is low," not "process it differently downstream."
- Certain poses (forearm raised with the upper arm held against the torso)
  make MediaPipe's own wrist confidence collapse — a model characteristic,
  not a bug in this codebase — with visible knock-on instability that hold/
  corroboration logic only partially absorbs.
- VMC/OSC output (to plug into Warudo/VNyan/OBSKUR) is not implemented — this
  is currently a self-contained webcam→VRM-avatar preview, not yet a
  streaming tracker for other apps.

## Run it

```bash
npm install
npm run dev
```

The main process auto-grants the OS camera permission prompt
(`session.defaultSession.setPermissionRequestHandler`) — the renderer just
calls `getUserMedia` directly. The MediaPipe models + WASM runtime are
fetched from Google's/jsDelivr's CDN at runtime, so the first launch needs
network access.

```bash
npm run typecheck   # tsc -b across main/preload/renderer
npm run build        # electron-vite build
```

While running, `T포즈 측정` (or holding both arms out to the side for ~2s)
measures exact arm lengths; holding one arm at a genuine full reach and
pressing `R` measures how far short of the true reach distance MediaPipe's
raw estimate falls, to correct for it. `setTrackingDebug(true)` (on by
default in `main.ts`) prints `[ik]`/`[tpose]`/`[glitch]`/`[ratchet]`
diagnostics to the terminal running `npm run dev`.

## Structure

```
src/
  main/       Electron main process (window, camera permission grant)
  preload/    contextBridge surface for main <-> renderer IPC (placeholder)
  renderer/
    armTracking.ts   the tracking core — see "Using the tracking core as a module"
    main.ts           webcam capture, MediaPipe inference, VRM rigging, debug overlays
```

## License notes

- MediaPipe Tasks Vision: Apache 2.0 — OK commercially.
- The bundled placeholder avatar (`public/models/VRM1_Constraint_Twist_Sample.vrm`)
  is the VRM Consortium's own spec-conformance sample, used here only as a
  local development/testing asset — see `public/models/ATTRIBUTION.md` before
  distributing any build that includes it.
