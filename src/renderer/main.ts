import {
  FilesetResolver,
  HolisticLandmarker,
  DrawingUtils,
  type HolisticLandmarkerResult,
  type Landmark
} from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils, VRMHumanBoneName, type VRM } from '@pixiv/three-vrm'
import { startDepthWorker, sampleDepth, type DepthFrame } from './depthModel'

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task'
const VRM_URL = './models/VRM1_Constraint_Twist_Sample.vrm'

// This app only tracks arms/hands (see the design doc) — hips and legs are
// out of scope, and on a desk webcam they're usually off-frame anyway, which
// makes MediaPipe's hip-relative pose estimate for them unreliable. Shoulders/
// elbows/wrists stay in frame and are used everywhere instead: for the 2D
// overlay, the 3D debug skeleton, and as the anchor point (shoulder midpoint)
// in place of the hips.
const UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16] as const // L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist
const UPPER_BODY_CONNECTIONS: Array<{ start: number; end: number }> = [
  { start: 0, end: 1 }, // shoulder to shoulder
  { start: 0, end: 2 }, // left shoulder -> elbow
  { start: 2, end: 4 }, // left elbow -> wrist
  { start: 1, end: 3 }, // right shoulder -> elbow
  { start: 3, end: 5 } // right elbow -> wrist
]

// Each finger as [wrist, joint1, joint2, joint3] MediaPipe hand-landmark
// indices. The true fingertip (4/8/12/16/20) is skipped: VRM's *Distal bones
// have no child bone to represent it, so there's nothing on the avatar to
// match it 1:1 against.
type FingerName = 'Thumb' | 'Index' | 'Middle' | 'Ring' | 'Little'
const FINGERS: FingerName[] = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
const FINGER_LANDMARKS: Record<FingerName, [number, number, number, number]> = {
  Thumb: [0, 1, 2, 3],
  Index: [0, 5, 6, 7],
  Middle: [0, 9, 10, 11],
  Ring: [0, 13, 14, 15],
  Little: [0, 17, 18, 19]
}
// The true fingertip landmark, used only for rigging (the last segment's aim
// target) — not for the debug skeleton, which has no avatar bone to compare it to.
const FINGER_TIP_LANDMARK: Record<FingerName, number> = {
  Thumb: 4,
  Index: 8,
  Middle: 12,
  Ring: 16,
  Little: 20
}

// 5 fingers x [wrist, joint1, joint2, joint3] = 20 points/hand, connected in
// 3 segments per finger (wrist->joint1->joint2->joint3).
const HAND_POINT_COUNT = FINGERS.length * 4
const HAND_CONNECTIONS_CALIBRATED: Array<{ start: number; end: number }> = FINGERS.flatMap((_, f) => [
  { start: f * 4, end: f * 4 + 1 },
  { start: f * 4 + 1, end: f * 4 + 2 },
  { start: f * 4 + 2, end: f * 4 + 3 }
])

const video = document.getElementById('webcam') as HTMLVideoElement
const canvas = document.getElementById('overlay') as HTMLCanvasElement
const avatarDebugCanvas = document.getElementById('avatar-debug-canvas') as HTMLCanvasElement
const avatarCleanCanvas = document.getElementById('avatar-clean-canvas') as HTMLCanvasElement
const avatarTopCanvas = document.getElementById('avatar-top-canvas') as HTMLCanvasElement
const avatarSideCanvas = document.getElementById('avatar-side-canvas') as HTMLCanvasElement
const statusEl = document.getElementById('status') as HTMLElement
const readoutEl = document.getElementById('readout') as HTMLUListElement
const cameraSelect = document.getElementById('camera-select') as HTMLSelectElement
const calibrateButton = document.getElementById('calibrate-button') as HTMLButtonElement
const calibrationPromptEl = document.getElementById('calibration-prompt') as HTMLElement
const ctx = canvas.getContext('2d')!

function setStatus(text: string): void {
  statusEl.textContent = text
}

type Vec3Like = { x: number; y: number; z: number }

// MediaPipe occasionally emits a landmark with a NaN/Infinity coordinate on
// a degenerate frame. Used to reject those at the point they'd otherwise
// enter held/accumulated state — see holdJoint and resolveTrackingFrame.
function isFiniteVec3(v: Vec3Like): boolean {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z)
}

// A plain object carrying just the landmark arrays applyPoseToVRM/the debug
// skeleton need — decoupled from MediaPipe's own result type so both can be
// driven by one frame's live `HolisticLandmarkerResult` via trackingDataFromResult.
interface TrackingData {
  poseLandmarks?: Vec3Like[] // normalized image-space (0-1) — for on-screen body position, see computeBodyOffset
  poseWorldLandmarks?: Landmark[] // keeps `visibility` — needed for confidence gating, see resolveTrackingFrame
  leftHandWorldLandmarks?: Vec3Like[]
  rightHandWorldLandmarks?: Vec3Like[]
  // Normalized image-space (0-1, perspective-projected) hand landmarks —
  // unlike the world ones above, these DO shrink/grow with distance from
  // the camera, which is exactly the size-based depth cue resolveArm uses
  // (see handSize).
  leftHandLandmarks?: Vec3Like[]
  rightHandLandmarks?: Vec3Like[]
}

function trackingDataFromResult(result: HolisticLandmarkerResult): TrackingData {
  return {
    poseLandmarks: result.poseLandmarks[0],
    poseWorldLandmarks: result.poseWorldLandmarks[0],
    leftHandWorldLandmarks: result.leftHandWorldLandmarks[0],
    rightHandWorldLandmarks: result.rightHandWorldLandmarks[0],
    leftHandLandmarks: result.leftHandLandmarks[0],
    rightHandLandmarks: result.rightHandLandmarks[0]
  }
}

// ---------- Guided body-measurement calibration ----------
//
// Passively hoping the user happens to move through a few good calibration
// poses during an ambient window was unreliable (no guarantee of a clean
// broadside arm extension, no signal at all for "arm close to the body")
// and non-obvious to the user (silently measuring in the background gave no
// visible feedback about what's being measured or when it's safe to move
// normally). Guiding through 3 explicit, held poses via a button-triggered
// sequence instead:
//
// - 'tpose': arms straight out to the sides — the same broadside-length
//   measurement done before, just guaranteed instead of hoped-for (see
//   PersonArmLengths).
// - 'overhead': elbow kept down against the torso (not lifted away from
//   the body) while the forearm curls up to raise the hand near the head —
//   a bicep-curl shape, not a full overhead raise, specifically so the
//   elbow stays genuinely close to the body and the raised hand stays
//   roughly above it rather than leaning forward — i.e. at the *same
//   depth* as the shoulder line, just much higher up. That "same depth as
//   the body" property is the whole point: whatever the hand's apparent
//   size looks like on-screen right now is this person's reference size
//   for "hand at body depth," which 'arms-forward' below can be compared
//   against (see handSize) — a reach toward the camera makes the hand look
//   *bigger* than this reference, independent of and complementary to the
//   bone-length reasoning that drives the Z correction itself.
// - 'arms-forward': both arms extended straight at the camera — a guided,
//   known instance of the exact reach the depth reconstruction exists to
//   handle, so its result can be checked in the readout under a controlled
//   pose rather than an incidental one, and so the hand-size reference from
//   'overhead' has something to compare against.
//
// Each pose gets its own "-prep" stage first (get into position, nothing
// measured yet) before the actual measured hold — moving straight from one
// held pose into measuring the next with no transition time was too rushed
// to comfortably get into position for it.
type CalibrationPose = 'tpose' | 'overhead' | 'arms-forward'
type CalibrationStage = 'idle' | `${CalibrationPose}-prep` | CalibrationPose | 'done'

const CALIBRATION_POSE_ORDER: CalibrationPose[] = ['tpose', 'overhead', 'arms-forward']
const CALIBRATION_PREP_MS = 3000
const CALIBRATION_HOLD_MS: Record<CalibrationPose, number> = {
  tpose: 5000,
  overhead: 4000,
  'arms-forward': 4000
}
const CALIBRATION_PROMPT: Record<Exclude<CalibrationStage, 'idle' | 'done'>, string> = {
  'tpose-prep': '준비하세요: 아바타처럼 T포즈를 따라해주세요',
  tpose: '아바타와 같은 자세로 유지해주세요',
  'overhead-prep': '준비하세요: 아바타처럼 팔꿈치는 몸통 옆에 붙인 채 팔을 굽혀 손을 위로 올려주세요 (손이 어깨보다 앞으로 나가지 않게)',
  overhead: '아바타와 같은 자세로 유지해주세요 (팔꿈치는 몸통 옆에, 손이 어깨보다 앞으로 나가지 않게)',
  'arms-forward-prep': '준비하세요: 아바타처럼 양팔을 카메라 쪽으로 쭉 뻗어주세요',
  'arms-forward': '아바타와 같은 자세로 유지해주세요'
}

// Drives the avatar into each calibration pose directly (rather than
// leaving it under live tracking control) so the user has an unambiguous
// visual target to copy instead of having to interpret a text description —
// exactly the "손을 반대편 어깨" vs. "같은 쪽 어깨" mix-up a text-only
// instruction just caused. `null` means "reset to rest pose" (T-pose is
// already close to the VRM normalized rig's own rest orientation); the
// other two are world-space directions for aimBoneToward, both arms aimed
// identically since the pose is symmetric. Approximate on purpose — these
// only need to look like a recognizable target pose, not be biomechanically
// exact, since what's actually measured comes from the live camera
// tracking of the *person* copying the pose, not from these numbers.
const CALIBRATION_DEMO_DIRECTION: Record<CalibrationPose, { upper: THREE.Vector3; lower: THREE.Vector3 } | null> = {
  tpose: null,
  // Upper arm stays DOWN against the torso (elbow tucked at the side, not
  // lifted) while only the forearm curls up to raise the hand — a bicep-
  // curl shape, not a full overhead raise, so the elbow genuinely stays
  // close to the body instead of swinging the whole arm up and away from it.
  overhead: { upper: new THREE.Vector3(0, -1, 0), lower: new THREE.Vector3(0, 1, 0) },
  'arms-forward': { upper: new THREE.Vector3(0, 0, 1), lower: new THREE.Vector3(0, 0, 1) }
}

let calibrationStage: CalibrationStage = 'idle'
let calibrationStageDeadline = 0

// ---------- Tracking-frame resolution: confidence gating + bone-length-
// constrained depth reconstruction ----------
//
// Two of the tracking problems (joints jumping to a wrong spot when an
// arm/hand is barely visible, and forward-reach toward the camera not
// registering at all) share a fix that has to happen in one place, before
// mirroring/rigging, on the same raw shoulder/elbow/wrist points — this runs
// once per frame and its result feeds both the actual rigging
// (applyPoseToVRM/applyHandToVRM) and the debug skeleton, so the debug view
// always shows exactly what's driving the avatar rather than a separate,
// unfiltered copy.

// MediaPipe's own visibility score for pose landmarks (hand landmarks don't
// carry a meaningful one of their own). Below this, a given joint's reading
// this frame is untrustworthy, and the last frame where it *was*
// trustworthy keeps being used instead ("hold last good pose") rather than
// letting a noisy low-confidence reading move it somewhere arbitrary.
// Matches minPoseDetectionConfidence below — no reason for this gate to be
// stricter than the detector's own confidence floor.
const MIN_ARM_VISIBILITY = 0.5

interface ArmJoints {
  shoulder: Vec3Like
  elbow: Vec3Like
  wrist: Vec3Like
}

interface JointHold {
  current: Vec3Like | null
}

// Holds and gates ONE joint independently of its neighbors. Shoulder, elbow
// and wrist each drop below the confidence threshold at different times —
// a raised wrist is very commonly less visible than the shoulder it's
// attached to — and freezing the *whole* arm just because the wrist alone
// went unreliable was making the shoulder reference point itself visibly
// drift/skew away from the shoulder's own (still perfectly good) tracking.
// `corroborated`: an independent signal (see resolveArm's `handTracked`)
// that this joint is genuinely present even if its own visibility score is
// low — without it, a wrist/elbow reaching toward the camera gets treated
// exactly like a *badly* tracked one, since both look identical from
// visibility alone (low score), even though a self-occluded-by-your-own-hand
// wrist is precisely the case depth reconstruction exists to handle, not a
// case to freeze and throw that data away.
function holdJoint(
  poseWorldLm: Landmark[] | undefined,
  idx: number,
  held: JointHold,
  corroborated = false
): { point: Vec3Like | null; live: boolean } {
  const lm = poseWorldLm?.[idx]
  // isFiniteVec3 rejects an occasional degenerate NaN/Infinity reading —
  // without this, one bad frame permanently poisons the running max-length
  // estimate below (Math.max with a NaN argument returns NaN forever after),
  // which then corrupts that arm's bone rotation via slerp and breaks the
  // avatar's rendering for the rest of the session.
  const live = !!lm && (lm.visibility >= MIN_ARM_VISIBILITY || corroborated) && isFiniteVec3(lm)
  if (live && lm) held.current = lm
  return { point: held.current, live }
}

// A person's real bone lengths don't change during a tracking session, so
// rather than continuously adapting forever, this is measured once, during
// the guided 'tpose' calibration stage: the largest on-screen (XY-only —
// deliberately ignoring MediaPipe's own unreliable Z) distance seen between
// two joints while T-posing is tracked as the estimate of the real length
// between them (exact for a clean broadside extension), then it's frozen
// for the rest of the session so depth reconstruction below is always
// computed against one fixed, known measurement rather than a value that
// keeps drifting. Seeded from the avatar's own (bilaterally averaged) bone
// lengths in main() as a reasonable starting prior in case calibration is
// never run at all, or the T-pose stage produced too few good samples.
interface PersonArmLengths {
  upper: number
  lower: number
}
const personArmLengths: { left: PersonArmLengths; right: PersonArmLengths } = {
  left: { upper: 0, lower: 0 },
  right: { upper: 0, lower: 0 }
}

// Every valid (see resolveArm) length observed during the 'tpose' stage,
// collected so the final locked-in value can be a robust statistic instead
// of a raw running maximum. A raw max has a real weakness: a normal
// reading can only ever *underestimate* the true bone length, but a single
// bad frame — a motion-blurred elbow, one mistracked landmark — CAN read as
// *longer* than the truth, and a running max bakes that one bad frame in
// permanently the moment it's frozen, with no way for later good frames to
// correct it. See finalizeArmLength.
interface ArmLengthSamples {
  upper: number[]
  lower: number[]
}
const armLengthSamples: { left: ArmLengthSamples; right: ArmLengthSamples } = {
  left: { upper: [], lower: [] },
  right: { upper: [], lower: [] }
}

// Below this many valid samples for a segment, the calibration window
// wasn't informative enough to trust (e.g. that arm stayed still, or
// occluded, the whole time) — finalizeArmLength leaves whatever value is
// already there (the avatar-bone-length seed, or a partial live estimate)
// rather than locking in a statistic computed from too little data.
const MIN_CALIBRATION_SAMPLES = 30
// Using the 90th percentile instead of the maximum means at least 10% of
// *all* samples collected have to agree a length is at least this large
// before it can become the locked-in value — a single outlier frame can't
// move it, only a real, repeated broadside pose can.
const CALIBRATION_PERCENTILE = 0.9

function percentile(samples: number[], p: number): number {
  const sorted = [...samples].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))
  return sorted[idx]
}

// Called exactly once, when the 'tpose' stage ends (see
// advanceCalibrationStage) — turns this stage's collected samples into the
// final, permanently locked-in per-segment length.
function finalizeArmLength(estimate: PersonArmLengths, samples: ArmLengthSamples): void {
  if (samples.upper.length >= MIN_CALIBRATION_SAMPLES) estimate.upper = percentile(samples.upper, CALIBRATION_PERCENTILE)
  if (samples.lower.length >= MIN_CALIBRATION_SAMPLES) estimate.lower = percentile(samples.lower, CALIBRATION_PERCENTILE)
}

// Diagnostic-only numbers surfaced in the on-screen readout so the
// measured-length-based depth correction can be sanity-checked against what
// the same reach would produce using the avatar's own (fixed) bone length
// instead — see computeBlendedDepth.
interface ArmDiagnostics {
  upperDepth: number
  lowerDepth: number
  avatarUpperDepth: number
  avatarLowerDepth: number
  // Raw (pre-correction) X/Y/Z displacement for each segment, straight from
  // MediaPipe's world landmarks — added specifically to check, with actual
  // numbers instead of eyeballing a screenshot, how much of a segment's
  // movement is really vertical (Y) vs. depth (Z) rather than assuming.
  rawUpperDelta: Vec3Like
  rawLowerDelta: Vec3Like
}
const ZERO_DELTA: Vec3Like = { x: 0, y: 0, z: 0 }
const NO_ARM_DIAGNOSTICS: ArmDiagnostics = {
  upperDepth: 0,
  lowerDepth: 0,
  avatarUpperDepth: 0,
  avatarLowerDepth: 0,
  rawUpperDelta: ZERO_DELTA,
  rawLowerDelta: ZERO_DELTA
}

// Below this fraction of the known real bone length, the on-screen (XY)
// length is short enough that the segment is plausibly reaching toward or
// away from the camera rather than just being tracking noise, so depth
// reconstruction starts blending in. Blending smoothly instead of switching
// on/off at a hard cutoff avoids a visible snap right at the boundary, and
// keeps the correction's own sign ambiguity (see below) confined to the
// region where the correction is smallest anyway.
const RECON_BLEND_START_RATIO = 0.6

// Apparent hand size in the (perspective-projected) normalized image —
// distance from the wrist to the middle-fingertip — which, unlike the
// metric world landmarks used everywhere else, genuinely shrinks and grows
// with distance from the camera. Used as an independent depth cue: compare
// this against handSizeReference (captured during the 'overhead'
// calibration pose, where the hand is deliberately held at the same depth
// as the body) to tell whether the hand is currently closer or farther than
// that reference — see handDepthSignHint.
function handSize(landmarks: Vec3Like[]): number {
  const wrist = landmarks[0]
  const middleTip = landmarks[12]
  return Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y)
}

// This person's own hand size (see handSize) while holding a hand at body
// depth, measured once during the 'overhead' calibration pose and then
// frozen — same measure-then-freeze shape as PersonArmLengths, and for the
// same reason: an size that kept adapting forever couldn't be trusted as a
// stable "this is what body-depth looks like" reference.
const handSizeReference: { left: number; right: number } = { left: 0, right: 0 }
const handSizeSamples: { left: number[]; right: number[] } = { left: [], right: [] }
const MIN_HAND_SIZE_SAMPLES = 15

function finalizeHandSizeReference(): void {
  if (handSizeSamples.left.length >= MIN_HAND_SIZE_SAMPLES) {
    handSizeReference.left = percentile(handSizeSamples.left, 0.5)
  }
  if (handSizeSamples.right.length >= MIN_HAND_SIZE_SAMPLES) {
    handSizeReference.right = percentile(handSizeSamples.right, 0.5)
  }
}

// A second hand-size sample, this time collected during 'arms-forward' —
// both arms extended straight at the camera by the full, already-measured
// T-pose arm length. Combined with handSizeReference (hand size at body
// depth, zero reach), this is now a size measured at *two* known depths
// instead of one, which is enough to solve perspective's size ∝
// 1/distance relationship completely (see finalizeHandSizeDepthModel) —
// turning the hand-size cue from a same-signed hint into an actual
// estimated depth in meters.
const handSizeFullReachSamples: { left: number[]; right: number[] } = { left: [], right: [] }
const handSizeAtFullReach: { left: number; right: number } = { left: 0, right: 0 }
// The camera-to-body distance implied by the two calibrated hand sizes and
// the known reach distance (armFullLength) between them — solving
// S0 = k/D0 and S1 = k/(D0 - armFullLength) for D0. Used at runtime to
// convert a live hand size back into an estimated depth — see resolveArm.
const handSizeBodyDistance: { left: number; right: number } = { left: 0, right: 0 }

function finalizeHandSizeDepthModel(side: 'left' | 'right', armFullLength: number): void {
  const samples = handSizeFullReachSamples[side]
  if (samples.length < MIN_HAND_SIZE_SAMPLES || handSizeReference[side] < 1e-6 || armFullLength < 1e-4) return
  const fullReachSize = percentile(samples, 0.5)
  // The reach has to have actually looked bigger than the body-depth
  // reference for the two-point solve below to make sense (a well-formed,
  // increasing size-vs-closeness relationship) — if it didn't, this
  // person's 'arms-forward' pose likely wasn't a clean, deliberate reach,
  // and the size-based depth estimate is left disabled rather than solved
  // from a degenerate or inverted pair of points.
  if (fullReachSize <= handSizeReference[side]) return
  handSizeAtFullReach[side] = fullReachSize
  handSizeBodyDistance[side] = (fullReachSize * armFullLength) / (fullReachSize - handSizeReference[side])
}

// ---------- External depth model (Depth Anything V2 Small) ----------
//
// A monocular depth-estimation network run directly on the webcam frame
// (see depthModel.ts) — a signal completely independent of MediaPipe's own
// pose Z estimate, sensing depth straight from the image rather than
// inferring it from bone-length or hand-size geometry. Its raw output
// isn't metric on its own, so it's calibrated the same two-known-points way
// as the hand-size cue: sampled at the wrist during 'overhead' (known: body
// depth) and during 'arms-forward' (known: body depth minus the full,
// T-pose-measured arm length), which is enough to convert a live raw
// sample into an estimated meters offset from the body.
interface ExternalDepthCalibration {
  bodyDepthRaw: number
  fullReachRaw: number
  calibrated: boolean
}
const externalDepthCalibration: { left: ExternalDepthCalibration; right: ExternalDepthCalibration } = {
  left: { bodyDepthRaw: 0, fullReachRaw: 0, calibrated: false },
  right: { bodyDepthRaw: 0, fullReachRaw: 0, calibrated: false }
}
const externalDepthReferenceSamples: { left: number[]; right: number[] } = { left: [], right: [] } // 'overhead'
const externalDepthFullReachSamples: { left: number[]; right: number[] } = { left: [], right: [] } // 'arms-forward'
// This signal updates far less often than a normal frame (it comes from the
// depth worker's own inference rate — see depthWorker.requestFrame in the
// main loop below), so demanding as many samples as the other calibrations
// (MIN_HAND_SIZE_SAMPLES etc.) would mean requiring an unreasonably long
// hold — a smaller minimum is an acceptable trade against robustness here.
const MIN_EXTERNAL_DEPTH_SAMPLES = 5

function finalizeExternalDepthReference(side: 'left' | 'right'): void {
  const samples = externalDepthReferenceSamples[side]
  if (samples.length < MIN_EXTERNAL_DEPTH_SAMPLES) return
  externalDepthCalibration[side].bodyDepthRaw = percentile(samples, 0.5)
}

function finalizeExternalDepthModel(side: 'left' | 'right'): void {
  const samples = externalDepthFullReachSamples[side]
  const calib = externalDepthCalibration[side]
  if (samples.length < MIN_EXTERNAL_DEPTH_SAMPLES) return
  const fullReachRaw = percentile(samples, 0.5)
  if (fullReachRaw === calib.bodyDepthRaw) return // no usable slope between the two points
  calib.fullReachRaw = fullReachRaw
  calib.calibrated = true
}

// Bundles everything resolveArm needs from the external depth model for one
// side, so it doesn't need half a dozen more loose parameters — the live
// depth frame (produced on a dedicated Worker, see depthModel.ts, and
// polled into latestDepthFrame once per render-loop tick) plus this side's
// wrist pixel position (to sample it at) and its calibration/sample state.
interface ExternalDepthContext {
  frame: DepthFrame | null
  wristNormalized: Vec3Like | undefined // normalized image-space, un-mirrored — matches DepthFrame's own coordinate convention
  calibration: ExternalDepthCalibration
  referenceSamples: number[]
  fullReachSamples: number[]
}

// Polled from the depth worker once per render-loop tick (see main()) —
// resolveArm only ever reads whatever's currently here, since it runs
// synchronously every tracking frame and can't itself wait on a
// several-hundred-millisecond model inference.
let latestDepthFrame: DepthFrame | null = null
let depthModelStatus = '로딩 중...'

// Beyond this much relative deviation from the body-depth reference size,
// the hand-size cue is trusted enough to override computeBlendedDepth's own
// sign guess — small deviations are left alone since normal hand-size
// tracking noise (finger spread, viewing angle) is easily within this band
// even at a constant true depth.
const HAND_SIZE_SIGN_MARGIN = 0.1

// Compares this frame's apparent hand size against the calibrated body-depth
// reference to say, independently of computeBlendedDepth's own raw-Z-based
// guess, whether the hand is currently closer to the camera (bigger than
// the reference) or farther (smaller) — `undefined` when uncalibrated or
// too close to the reference to tell confidently, in which case
// computeBlendedDepth falls back to its own raw-Z sign guess.
function handDepthSignHint(landmarks: Vec3Like[] | undefined, reference: number): number | undefined {
  if (!landmarks || reference < 1e-6) return undefined
  const ratio = handSize(landmarks) / reference
  if (ratio > 1 + HAND_SIZE_SIGN_MARGIN) return -1 // bigger than reference — closer to the camera
  if (ratio < 1 - HAND_SIZE_SIGN_MARGIN) return 1 // smaller than reference — farther from the camera
  return undefined
}

// MediaPipe's own raw Z has a systematic per-segment bias, not just random
// noise — visible during the 'overhead' pose, where the shoulder, elbow and
// wrist are all deliberately held at the *same* real depth (see
// CalibrationPose) so the true dz is known to be ~0, yet the tracked elbow/
// wrist still read as noticeably forward of the shoulder in the debug
// skeleton. Measuring that offset while the true answer is known and
// subtracting it from every future reading (see computeBlendedDepth) is a
// zero-point calibration — the same idea as taring a scale before weighing
// anything — done once, then frozen the same way as PersonArmLengths.
interface ArmZBias {
  upper: number
  lower: number
}
const armZBias: { left: ArmZBias; right: ArmZBias } = {
  left: { upper: 0, lower: 0 },
  right: { upper: 0, lower: 0 }
}
const armZBiasSamples: { left: ArmLengthSamples; right: ArmLengthSamples } = {
  left: { upper: [], lower: [] },
  right: { upper: [], lower: [] }
}

function finalizeArmZBias(bias: ArmZBias, samples: ArmLengthSamples): void {
  if (samples.upper.length >= MIN_CALIBRATION_SAMPLES) bias.upper = percentile(samples.upper, 0.5)
  if (samples.lower.length >= MIN_CALIBRATION_SAMPLES) bias.lower = percentile(samples.lower, 0.5)
}

// Computes the Z offset (parent -> child) from the constraint that their
// real distance apart is `realLength` (Pythagorean: dz = sqrt(realLength^2
// - xyLength^2)) — ALWAYS, not just when the segment looks foreshortened
// enough to plausibly be a forward/backward reach. Originally this only
// reconstructed below a foreshortening ratio and otherwise trusted
// MediaPipe's own (zBias-corrected) raw Z outright; that let a segment held
// nearly perpendicular to the camera (e.g. hanging straight down at the
// side) fall through to raw Z untouched, and a real test standing at
// attention showed that raw Z — even bias-corrected — produced a ~25-40deg
// fake elbow bend there. zBias itself turned out to be the biggest offender:
// it was measured from one specific pose ('overhead') where dz is known to
// be ~0, so it's only valid near that pose's arm configuration, and applying
// it as a flat constant everywhere actively introduced error rather than
// removing it. Reliable X/Y tracking plus a precisely measured real bone
// length is a better depth source in *every* pose than a weak, pose-
// dependent axis ever was, so it's now used unconditionally; `zBias` and
// `signHint` only ever get a vote on which *sign* (toward vs. away from the
// camera) the reconstructed magnitude should take, never on magnitude
// itself. `signHint` comes from handDepthSignHint (a genuinely independent
// cue) when available, falling back to the (bias-corrected) raw Z's own
// sign otherwise — and the cases where that sign guess is least reliable
// (segment near broadside, dz close to 0) are also the cases where the
// reconstructed magnitude is smallest, so a wrong sign there barely moves
// the point anyway. Called twice per segment: once with the measured length
// (the value actually used for rigging) and once with the avatar's own bone
// length (diagnostic-only, for the readout's side-by-side comparison).
function computeBlendedDepth(parent: Vec3Like, child: Vec3Like, realLength: number, zBias: number, signHint?: number): number {
  const rawDz = child.z - parent.z - zBias
  if (realLength < 1e-4) return rawDz // no calibration data yet — trust MediaPipe's own Z as-is

  const dx = child.x - parent.x
  const dy = child.y - parent.y
  const xyLength = Math.hypot(dx, dy)
  const reconstructedDepth = Math.sqrt(Math.max(realLength * realLength - xyLength * xyLength, 0))
  const sign = signHint ?? (rawDz === 0 ? 1 : Math.sign(rawDz))
  return sign * reconstructedDepth
}

// Resolves one side's shoulder/elbow/wrist independently (see holdJoint),
// measures the real-bone-length estimate during the calibration window from
// whichever segments were genuinely observed simultaneously this frame,
// then chains depth reconstruction outward from the shoulder (the wrist's
// correction builds on the already-corrected elbow, not the raw one).
// `confident` reports the *wrist's* own live status — the signal the hand
// landmarks below piggyback their own confidence gating on, since they
// don't have a usable one of their own. `avatarLengths` is used only to
// compute the diagnostic comparison numbers, never for the actual rigging.
function resolveArm(
  poseWorldLm: Landmark[] | undefined,
  shoulderIdx: number,
  elbowIdx: number,
  wristIdx: number,
  estimate: PersonArmLengths,
  samples: ArmLengthSamples,
  avatarLengths: ArmBoneLengths,
  stage: CalibrationStage,
  handLandmarks: Vec3Like[] | undefined, // normalized image-space — see handDepthSignHint
  handSizeRef: number,
  handSizeSampleList: number[],
  handSizeFullReachSampleList: number[],
  handSizeBodyDist: number,
  externalDepth: ExternalDepthContext,
  zBias: ArmZBias,
  zBiasSamples: ArmLengthSamples,
  heldShoulder: JointHold,
  heldElbow: JointHold,
  heldWrist: JointHold
): { arm: ArmJoints | null; confident: boolean; diagnostics: ArmDiagnostics } {
  // A hand the Hand Landmarker is still confidently tracking is independent
  // corroborating evidence that this arm's elbow/wrist region genuinely is
  // there — even when the Pose Landmarker's own visibility score for them
  // drops, which it will while pointing straight at the camera even though
  // nothing is actually wrong (see holdJoint). Not applied to the shoulder,
  // which a hand being tracked says nothing about.
  const handTracked = !!handLandmarks
  const shoulder = holdJoint(poseWorldLm, shoulderIdx, heldShoulder)
  const elbow = holdJoint(poseWorldLm, elbowIdx, heldElbow, handTracked)
  const wrist = holdJoint(poseWorldLm, wristIdx, heldWrist, handTracked)

  // 'overhead' calibration: the hand is deliberately held at body depth
  // (see the CalibrationPose doc comment), so whatever it looks like
  // on-screen right now becomes this person's reference for "hand at body
  // depth" — see finalizeHandSizeReference. Also true depth: the shoulder,
  // elbow and wrist are all at the same real depth during this pose, so any
  // raw Z difference observed between them right now is pure tracking
  // bias — see finalizeArmZBias.
  if (stage === 'overhead' && handLandmarks) {
    handSizeSampleList.push(handSize(handLandmarks))
  }
  // 'arms-forward' calibration: the hand is now at a *known* depth too —
  // body depth minus the full (T-pose-measured) arm length — giving a
  // second calibrated size at a second known distance. See
  // finalizeHandSizeDepthModel.
  if (stage === 'arms-forward' && handLandmarks) {
    handSizeFullReachSampleList.push(handSize(handLandmarks))
  }
  // Same two calibration points, sampled from the external depth model's
  // raw output at the wrist's pixel position instead — see
  // finalizeExternalDepthReference/finalizeExternalDepthModel.
  if (externalDepth.frame && externalDepth.wristNormalized) {
    const raw = sampleDepth(externalDepth.frame, externalDepth.wristNormalized.x, externalDepth.wristNormalized.y)
    if (stage === 'overhead') externalDepth.referenceSamples.push(raw)
    if (stage === 'arms-forward') externalDepth.fullReachSamples.push(raw)
  }
  if (stage === 'overhead' && shoulder.live && elbow.live && shoulder.point && elbow.point) {
    zBiasSamples.upper.push(elbow.point.z - shoulder.point.z)
  }
  if (stage === 'overhead' && elbow.live && wrist.live && elbow.point && wrist.point) {
    zBiasSamples.lower.push(wrist.point.z - elbow.point.z)
  }

  // Only during the 'tpose' calibration stage, and only from segments whose
  // endpoints are both live *this same frame* — mixing a live joint with a
  // stale held one would report a distance that was never actually
  // observed simultaneously, corrupting the measurement. `estimate` is
  // updated live here too (running max) purely so the readout has
  // something to show growing during the stage — the value that actually
  // gets locked in comes from finalizeArmLength's robust statistic over
  // `samples`, computed once the stage ends.
  if (stage === 'tpose' && shoulder.live && elbow.live && shoulder.point && elbow.point) {
    const upperXY = Math.hypot(elbow.point.x - shoulder.point.x, elbow.point.y - shoulder.point.y)
    estimate.upper = Math.max(estimate.upper, upperXY)
    samples.upper.push(upperXY)
  }
  if (stage === 'tpose' && elbow.live && wrist.live && elbow.point && wrist.point) {
    const lowerXY = Math.hypot(wrist.point.x - elbow.point.x, wrist.point.y - elbow.point.y)
    estimate.lower = Math.max(estimate.lower, lowerXY)
    samples.lower.push(lowerXY)
  }

  if (!shoulder.point || !elbow.point || !wrist.point) {
    return { arm: null, confident: false, diagnostics: NO_ARM_DIAGNOSTICS }
  }

  const signHint = handDepthSignHint(handLandmarks, handSizeRef)

  const upperDepth = computeBlendedDepth(shoulder.point, elbow.point, estimate.upper, zBias.upper, signHint)
  const correctedElbow: Vec3Like = { x: elbow.point.x, y: elbow.point.y, z: shoulder.point.z + upperDepth }
  const lowerDepth = computeBlendedDepth(correctedElbow, wrist.point, estimate.lower, zBias.lower, signHint)
  const correctedWrist: Vec3Like = { x: wrist.point.x, y: wrist.point.y, z: correctedElbow.z + lowerDepth }

  const avatarUpperDepth = computeBlendedDepth(shoulder.point, elbow.point, avatarLengths.upperArm, zBias.upper, signHint)
  const avatarLowerDepth = computeBlendedDepth(correctedElbow, wrist.point, avatarLengths.lowerArm, zBias.lower, signHint)

  // When *both* segments individually look heavily foreshortened on-screen
  // — not just the shoulder-to-wrist span, which shrinks in 2D just from a
  // bent elbow even with zero camera-axis foreshortening at all, and so
  // can't by itself tell "bent" apart from "reaching at the camera" — the
  // whole arm is pointing close to the camera's own viewing axis. In that
  // case the elbow is very likely hidden behind the hand/forearm from the
  // camera's viewpoint, so its raw X/Y (never corrected above, only its Z
  // was) is an extrapolated guess rather than a real observation, and tends
  // to slide away from where the elbow actually is (e.g. downward),
  // producing a visibly bent-looking arm even after Z correction — even
  // though the person is holding it straight. Requiring *both* segments to
  // be individually foreshortened (rather than just the overall span) is
  // what tells a straight arm pointed at the camera apart from a normally
  // bent elbow raised in front of the body, where usually at most one
  // segment is foreshortened, not both.
  const upperXY = Math.hypot(elbow.point.x - shoulder.point.x, elbow.point.y - shoulder.point.y)
  const lowerXY = Math.hypot(wrist.point.x - correctedElbow.x, wrist.point.y - correctedElbow.y)
  const upperRatio = estimate.upper < 1e-4 ? 1 : upperXY / estimate.upper
  const lowerRatio = estimate.lower < 1e-4 ? 1 : lowerXY / estimate.lower

  const upperBlendForStraightening = upperRatio >= RECON_BLEND_START_RATIO ? 0 : 1 - upperRatio / RECON_BLEND_START_RATIO
  const lowerBlendForStraightening = lowerRatio >= RECON_BLEND_START_RATIO ? 0 : 1 - lowerRatio / RECON_BLEND_START_RATIO

  const wholeArmLength = estimate.upper + estimate.lower

  // Hand size (see handSize) and the external depth model (see
  // depthModel.ts) each give a second, independent depth estimate for the
  // wrist specifically, in actual meters rather than just a sign (see
  // handSizeBodyDistance / ExternalDepthContext) — averaged together when
  // more than one is available, then blended in only as much as the
  // bone-length method is already engaging for this same segment (reusing
  // lowerBlendForStraightening as the weight) so none of this can introduce
  // noise into an ordinary, non-reaching pose where nothing needs correcting.
  if (lowerBlendForStraightening > 0) {
    const wristZCandidates: number[] = []
    if (handLandmarks && handSizeBodyDist > 1e-6 && handSizeRef > 1e-6) {
      const liveSize = handSize(handLandmarks)
      if (liveSize > 1e-6) {
        const depthFromBody = handSizeBodyDist * (1 - handSizeRef / liveSize)
        wristZCandidates.push(shoulder.point.z - depthFromBody)
      }
    }
    if (externalDepth.calibration.calibrated && externalDepth.frame && externalDepth.wristNormalized && wholeArmLength > 1e-4) {
      const liveRaw = sampleDepth(externalDepth.frame, externalDepth.wristNormalized.x, externalDepth.wristNormalized.y)
      const denom = externalDepth.calibration.fullReachRaw - externalDepth.calibration.bodyDepthRaw
      const t = (liveRaw - externalDepth.calibration.bodyDepthRaw) / denom // 0 at body depth, 1 at full reach
      const depthFromBody = t * wholeArmLength
      wristZCandidates.push(shoulder.point.z - depthFromBody)
    }
    if (wristZCandidates.length > 0) {
      const combinedWristZ = wristZCandidates.reduce((sum, z) => sum + z, 0) / wristZCandidates.length
      correctedWrist.z += (combinedWristZ - correctedWrist.z) * lowerBlendForStraightening
    }
  }

  const straightBlend = Math.min(upperBlendForStraightening, lowerBlendForStraightening)
  if (straightBlend > 0 && wholeArmLength > 1e-4) {
    const f = estimate.upper / wholeArmLength
    correctedElbow.x += (shoulder.point.x + f * (correctedWrist.x - shoulder.point.x) - correctedElbow.x) * straightBlend
    correctedElbow.y += (shoulder.point.y + f * (correctedWrist.y - shoulder.point.y) - correctedElbow.y) * straightBlend
    correctedElbow.z += (shoulder.point.z + f * (correctedWrist.z - shoulder.point.z) - correctedElbow.z) * straightBlend
  }

  const rawUpperDelta: Vec3Like = {
    x: elbow.point.x - shoulder.point.x,
    y: elbow.point.y - shoulder.point.y,
    z: elbow.point.z - shoulder.point.z
  }
  const rawLowerDelta: Vec3Like = {
    x: wrist.point.x - elbow.point.x,
    y: wrist.point.y - elbow.point.y,
    z: wrist.point.z - elbow.point.z
  }

  return {
    arm: { shoulder: shoulder.point, elbow: correctedElbow, wrist: correctedWrist },
    confident: wrist.live,
    diagnostics: { upperDepth, lowerDepth, avatarUpperDepth, avatarLowerDepth, rawUpperDelta, rawLowerDelta }
  }
}

// TEMPORARY (diagnostics only): angle between two vectors, in degrees — used
// to measure how "bent" an elbow looks (0deg = perfectly straight, upper and
// lower segment vectors pointing the same way) independent of which
// coordinate frame/handedness they're expressed in, so a resolveArm-space
// vector and a three.js-world-space vector can be compared on equal footing.
function angleBetweenDeg(a: Vec3Like, b: Vec3Like): number {
  const dot = a.x * b.x + a.y * b.y + a.z * b.z
  const magA = Math.hypot(a.x, a.y, a.z)
  const magB = Math.hypot(b.x, b.y, b.z)
  if (magA < 1e-6 || magB < 1e-6) return NaN
  const cos = Math.min(1, Math.max(-1, dot / (magA * magB)))
  return (Math.acos(cos) * 180) / Math.PI
}

const heldLeftShoulder: JointHold = { current: null } // person-left (11)
const heldLeftElbow: JointHold = { current: null } // person-left (13)
const heldLeftWrist: JointHold = { current: null } // person-left (15)
const heldRightShoulder: JointHold = { current: null } // person-right (12)
const heldRightElbow: JointHold = { current: null } // person-right (14)
const heldRightWrist: JointHold = { current: null } // person-right (16)
const heldLeftHand: { current: Vec3Like[] | null } = { current: null }
const heldRightHand: { current: Vec3Like[] | null } = { current: null }

interface ResolvedFrame {
  left: ArmJoints | null // person-left arm, depth-reconstructed
  right: ArmJoints | null // person-right arm, depth-reconstructed
  leftConfident: boolean // for the readout only — was THIS frame's left wrist trustworthy, or held?
  rightConfident: boolean
  leftHand: Vec3Like[] | null // person-left hand, held across low-confidence frames
  rightHand: Vec3Like[] | null
  leftDiagnostics: ArmDiagnostics
  rightDiagnostics: ArmDiagnostics
  calibratingArmLength: boolean // for the readout only — currently in the 'tpose' stage?
}

// Advances the guided calibration state machine (see CalibrationStage) once
// its current stage's timer runs out, running whatever one-time
// finalization that stage's measurement needs exactly at the transition.
// Called once per frame from the main loop, before resolveTrackingFrame.
function advanceCalibrationStage(now: number): void {
  if (calibrationStage === 'idle' || calibrationStage === 'done') return
  if (now < calibrationStageDeadline) return

  for (let i = 0; i < CALIBRATION_POSE_ORDER.length; i++) {
    const pose = CALIBRATION_POSE_ORDER[i]
    if (calibrationStage === `${pose}-prep`) {
      // Prep time is up — start the actual measured hold for this pose.
      calibrationStage = pose
      calibrationStageDeadline = now + CALIBRATION_HOLD_MS[pose]
      return
    }
    if (calibrationStage === pose) {
      // Hold time is up — finalize whatever this pose measures, then move
      // on to the next pose's prep (or finish, after the last one).
      if (pose === 'tpose') {
        finalizeArmLength(personArmLengths.left, armLengthSamples.left)
        finalizeArmLength(personArmLengths.right, armLengthSamples.right)
      } else if (pose === 'overhead') {
        finalizeHandSizeReference()
        finalizeArmZBias(armZBias.left, armZBiasSamples.left)
        finalizeArmZBias(armZBias.right, armZBiasSamples.right)
        finalizeExternalDepthReference('left')
        finalizeExternalDepthReference('right')
      } else if (pose === 'arms-forward') {
        finalizeHandSizeDepthModel('left', personArmLengths.left.upper + personArmLengths.left.lower)
        finalizeHandSizeDepthModel('right', personArmLengths.right.upper + personArmLengths.right.lower)
        finalizeExternalDepthModel('left')
        finalizeExternalDepthModel('right')
      }
      const next = CALIBRATION_POSE_ORDER[i + 1]
      if (next) {
        calibrationStage = `${next}-prep`
        calibrationStageDeadline = now + CALIBRATION_PREP_MS
      } else {
        calibrationStage = 'done'
      }
      return
    }
  }
}

function resolveTrackingFrame(data: TrackingData, avatarLeft: ArmBoneLengths, avatarRight: ArmBoneLengths): ResolvedFrame {
  const externalDepthLeft: ExternalDepthContext = {
    frame: latestDepthFrame,
    wristNormalized: data.poseLandmarks?.[15],
    calibration: externalDepthCalibration.left,
    referenceSamples: externalDepthReferenceSamples.left,
    fullReachSamples: externalDepthFullReachSamples.left
  }
  const externalDepthRight: ExternalDepthContext = {
    frame: latestDepthFrame,
    wristNormalized: data.poseLandmarks?.[16],
    calibration: externalDepthCalibration.right,
    referenceSamples: externalDepthReferenceSamples.right,
    fullReachSamples: externalDepthFullReachSamples.right
  }

  const left = resolveArm(
    data.poseWorldLandmarks,
    11,
    13,
    15,
    personArmLengths.left,
    armLengthSamples.left,
    avatarLeft,
    calibrationStage,
    data.leftHandLandmarks,
    handSizeReference.left,
    handSizeSamples.left,
    handSizeFullReachSamples.left,
    handSizeBodyDistance.left,
    externalDepthLeft,
    armZBias.left,
    armZBiasSamples.left,
    heldLeftShoulder,
    heldLeftElbow,
    heldLeftWrist
  )
  const right = resolveArm(
    data.poseWorldLandmarks,
    12,
    14,
    16,
    personArmLengths.right,
    armLengthSamples.right,
    avatarRight,
    calibrationStage,
    data.rightHandLandmarks,
    handSizeReference.right,
    handSizeSamples.right,
    handSizeFullReachSamples.right,
    handSizeBodyDistance.right,
    externalDepthRight,
    armZBias.right,
    armZBiasSamples.right,
    heldRightShoulder,
    heldRightElbow,
    heldRightWrist
  )

  // Hands don't carry a usable visibility score of their own — gated
  // instead on the same side's wrist confidence, just computed above (plus
  // the same finite-value check as holdJoint, for the same reason).
  if (data.leftHandWorldLandmarks && left.confident && data.leftHandWorldLandmarks.every(isFiniteVec3)) {
    heldLeftHand.current = data.leftHandWorldLandmarks
  }
  if (data.rightHandWorldLandmarks && right.confident && data.rightHandWorldLandmarks.every(isFiniteVec3)) {
    heldRightHand.current = data.rightHandWorldLandmarks
  }

  return {
    left: left.arm,
    right: right.arm,
    leftConfident: left.confident,
    rightConfident: right.confident,
    leftHand: heldLeftHand.current,
    rightHand: heldRightHand.current,
    leftDiagnostics: left.diagnostics,
    rightDiagnostics: right.diagnostics,
    calibratingArmLength: calibrationStage === 'tpose'
  }
}

// poseWorldLandmarks (used for arm/hand rigging) are re-centered on the hips
// every single frame, so they can never reveal whole-body movement — only
// joint-relative pose. The normalized image-space landmarks are the only
// signal for "where in frame is this person right now", so on-screen body
// movement is driven off of those instead: track how far the shoulder
// midpoint has drifted (in normalized 0-1 units) from a baseline center
// point, and turn that into a small world-space offset.
const BODY_OFFSET_SCALE = { x: 1.4, y: 1.4 }
const BODY_OFFSET_LERP = 0.2

let bodyOffsetBaseline: { x: number; y: number } | null = null
let bodyOffsetBaselineSumX = 0
let bodyOffsetBaselineSumY = 0
let bodyOffsetBaselineSamples = 0

function computeBodyOffsetTarget(poseLandmarks: Vec3Like[] | undefined): THREE.Vector3 | null {
  if (!poseLandmarks) return null
  const shoulderMid = {
    x: (poseLandmarks[11].x + poseLandmarks[12].x) / 2,
    y: (poseLandmarks[11].y + poseLandmarks[12].y) / 2
  }

  // Same measure-during-a-stage-then-freeze approach as the arm length,
  // piggybacking on the 'tpose' stage (standing normally, facing the
  // camera) and averaged over every frame of it — using just wherever the
  // person happened to be standing on a single first detected frame was
  // producing a persistent pull toward whatever direction that one frame
  // was off-center from center-frame.
  if (calibrationStage === 'tpose') {
    bodyOffsetBaselineSumX += shoulderMid.x
    bodyOffsetBaselineSumY += shoulderMid.y
    bodyOffsetBaselineSamples++
    bodyOffsetBaseline = { x: bodyOffsetBaselineSumX / bodyOffsetBaselineSamples, y: bodyOffsetBaselineSumY / bodyOffsetBaselineSamples }
  } else if (!bodyOffsetBaseline) {
    bodyOffsetBaseline = shoulderMid // calibration never run — fall back rather than divide by zero
  }

  const dx = shoulderMid.x - bodyOffsetBaseline.x
  const dy = shoulderMid.y - bodyOffsetBaseline.y
  // Mirror x (matches the mirrored webcam pane) and flip y (image-space y
  // increases downward, three.js world y increases upward).
  return new THREE.Vector3(-dx * BODY_OFFSET_SCALE.x, -dy * BODY_OFFSET_SCALE.y, 0)
}

let currentStream: MediaStream | null = null

async function startWebcamStream(deviceId?: string): Promise<void> {
  if (currentStream) {
    for (const track of currentStream.getTracks()) track.stop()
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
    audio: false
  })
  currentStream = stream
  video.srcObject = stream
  await new Promise<void>((resolve) => {
    video.onloadedmetadata = () => resolve()
  })
  await video.play()
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
}

// Device labels are only populated by the browser after a getUserMedia call has
// already been granted for this origin — so this has to run after the first
// startWebcamStream(), not before.
async function populateCameraList(): Promise<void> {
  const devices = await navigator.mediaDevices.enumerateDevices()
  const cameras = devices.filter((d) => d.kind === 'videoinput')

  cameraSelect.innerHTML = ''
  cameras.forEach((cam, i) => {
    const option = document.createElement('option')
    option.value = cam.deviceId
    option.textContent = cam.label || `카메라 ${i + 1}`
    cameraSelect.appendChild(option)
  })

  const activeDeviceId = currentStream?.getVideoTracks()[0]?.getSettings().deviceId
  if (activeDeviceId) cameraSelect.value = activeDeviceId
}

cameraSelect.addEventListener('change', () => {
  setStatus('카메라 전환 중...')
  startWebcamStream(cameraSelect.value)
    .then(() => setStatus('추적 중'))
    .catch((err) => setStatus(`오류: ${(err as Error).message}`))
})

async function setupHolistic(): Promise<HolisticLandmarker> {
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE)
  return HolisticLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    minPoseDetectionConfidence: 0.5,
    minHandLandmarksConfidence: 0.5
  })
}

function drawResult(result: HolisticLandmarkerResult, drawer: DrawingUtils): void {
  ctx.save()
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Only shoulders/elbows/wrists — hips and below are out of scope (see
  // UPPER_BODY_INDICES) and mostly off-frame on a desk webcam anyway.
  for (const landmarks of result.poseLandmarks) {
    const upperBody = UPPER_BODY_INDICES.map((i) => landmarks[i])
    drawer.drawConnectors(upperBody, UPPER_BODY_CONNECTIONS, {
      color: '#00d0ff',
      lineWidth: 3
    })
    drawer.drawLandmarks(upperBody, { color: '#00d0ff', radius: 3 })
  }
  for (const landmarks of result.leftHandLandmarks) {
    drawer.drawConnectors(landmarks, HolisticLandmarker.HAND_CONNECTIONS, {
      color: '#ff5f7e',
      lineWidth: 2
    })
    drawer.drawLandmarks(landmarks, { color: '#ff5f7e', radius: 2 })
  }
  for (const landmarks of result.rightHandLandmarks) {
    drawer.drawConnectors(landmarks, HolisticLandmarker.HAND_CONNECTIONS, {
      color: '#ffd25f',
      lineWidth: 2
    })
    drawer.drawLandmarks(landmarks, { color: '#ffd25f', radius: 2 })
  }
  ctx.restore()
}

function updateReadout(
  result: HolisticLandmarkerResult,
  fps: number,
  resolved: ResolvedFrame,
  calibration: AvatarCalibration
): void {
  readoutEl.innerHTML = ''
  const measured = personArmLengths
  const avatar = { left: calibration.left, right: calibration.right }
  const calibTag = calibrationStage === 'idle' ? '[대기]' : resolved.calibratingArmLength ? '[측정 중]' : '[고정]'
  const fmt = (upper: number, lower: number): string => `${upper.toFixed(3)}/${lower.toFixed(3)}/${(upper + lower).toFixed(3)}`
  const fmtDelta = (d: Vec3Like): string => `x${d.x.toFixed(3)}/y${d.y.toFixed(3)}/z${d.z.toFixed(3)}`
  const lines = [
    `fps: ${fps.toFixed(1)}`,
    `pose landmarks: ${result.poseLandmarks[0]?.length ?? 0}`,
    `left hand landmarks: ${result.leftHandLandmarks[0]?.length ?? 0}`,
    `right hand landmarks: ${result.rightHandLandmarks[0]?.length ?? 0}`,
    `보정 단계: ${calibrationStage}`,
    `팔 신뢰도 — L: ${resolved.leftConfident ? 'OK' : 'HOLD'} / R: ${resolved.rightConfident ? 'OK' : 'HOLD'}`,
    `사용자 측정 팔길이(상완/전완/전체) ${calibTag} L: ${fmt(measured.left.upper, measured.left.lower)} R: ${fmt(measured.right.upper, measured.right.lower)}`,
    `손 크기 기준값(몸 깊이) L: ${handSizeReference.left.toFixed(4)} R: ${handSizeReference.right.toFixed(4)}`,
    `Z축 보정값(상완/전완) L: ${armZBias.left.upper.toFixed(3)}/${armZBias.left.lower.toFixed(3)} R: ${armZBias.right.upper.toFixed(3)}/${armZBias.right.lower.toFixed(3)}`,
    `손크기-깊이 모델(카메라~몸 거리, m) L: ${handSizeBodyDistance.left.toFixed(3)} R: ${handSizeBodyDistance.right.toFixed(3)}`,
    `외부 깊이 모델 — ${depthModelStatus} (프레임: ${latestDepthFrame ? `${latestDepthFrame.width}x${latestDepthFrame.height}` : '없음'}, L 보정: ${externalDepthCalibration.left.calibrated ? 'OK' : '-'} R 보정: ${externalDepthCalibration.right.calibrated ? 'OK' : '-'})`,
    `아바타 보정값(상완/전완/전체) L: ${fmt(avatar.left.upperArm, avatar.left.lowerArm)} R: ${fmt(avatar.right.upperArm, avatar.right.lowerArm)}`,
    `뻗기 깊이-측정기준(상완/전완) L: ${resolved.leftDiagnostics.upperDepth.toFixed(3)}/${resolved.leftDiagnostics.lowerDepth.toFixed(3)} R: ${resolved.rightDiagnostics.upperDepth.toFixed(3)}/${resolved.rightDiagnostics.lowerDepth.toFixed(3)}`,
    `뻗기 깊이-아바타기준(상완/전완) L: ${resolved.leftDiagnostics.avatarUpperDepth.toFixed(3)}/${resolved.leftDiagnostics.avatarLowerDepth.toFixed(3)} R: ${resolved.rightDiagnostics.avatarUpperDepth.toFixed(3)}/${resolved.rightDiagnostics.avatarLowerDepth.toFixed(3)}`,
    `원본 변위-상완(보정 전) L: ${fmtDelta(resolved.leftDiagnostics.rawUpperDelta)} R: ${fmtDelta(resolved.rightDiagnostics.rawUpperDelta)}`,
    `원본 변위-전완(보정 전) L: ${fmtDelta(resolved.leftDiagnostics.rawLowerDelta)} R: ${fmtDelta(resolved.rightDiagnostics.rawLowerDelta)}`
  ]
  for (const line of lines) {
    const li = document.createElement('li')
    li.textContent = line
    readoutEl.appendChild(li)
  }
}

// ---------- Avatar scene ----------
// One scene/VRM/rig is shared by both avatar panes — only the render target
// (camera + renderer, one pair per canvas) differs, so the debug-skeleton and
// clean panes always show the exact same pose with zero duplicated rigging work.

function setupAvatarScene(): THREE.Scene {
  const scene = new THREE.Scene()
  scene.add(new THREE.AmbientLight(0xffffff, 1.5))
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0)
  dirLight.position.set(1, 1, 1)
  scene.add(dirLight)
  return scene
}

// 'front' is the original single view; 'top' and 'side' exist purely to make
// depth (toward/away from the camera) visible — a front view alone collapses
// depth almost entirely, which is exactly the axis a forward reach moves
// along, so checking whether depth reconstruction is working requires
// looking at the same skeletons from an angle where depth isn't collapsed.
type AvatarViewKind = 'front' | 'top' | 'side'

// A hardcoded absolute focus point only frames whichever avatar happened to
// have its shoulders sitting at exactly that height — a taller or shorter
// model (or one with an off-origin root) ends up off-center with no way to
// correct it. Positioning relative to `focus` instead means main() can pass
// the avatar's own real anchor point (see calibrateToAvatar) once it's
// known, and every view re-centers on whatever avatar is actually loaded.
function positionDebugCamera(camera: THREE.PerspectiveCamera, kind: AvatarViewKind, focus: THREE.Vector3): void {
  if (kind === 'front') {
    camera.position.set(focus.x, focus.y + 0.2, focus.z + 2.2)
    camera.up.set(0, 1, 0)
  } else if (kind === 'top') {
    // Looking straight down is degenerate with the default (0,1,0) up
    // vector (parallel to the view direction), so this points camera.up
    // along -Z (screen-up = away from the webcam) instead for a stable,
    // consistently-oriented top-down view.
    camera.position.set(focus.x, focus.y + 2.4, focus.z)
    camera.up.set(0, 0, -1)
  } else {
    camera.position.set(focus.x + 2.2, focus.y, focus.z)
    camera.up.set(0, 1, 0)
  }
  camera.lookAt(focus)
}

// Default focus used before the avatar has loaded/calibrated — see
// repositionDebugCameras, called once calibration.anchor is known, for the
// real one.
const DEFAULT_AVATAR_FOCUS = new THREE.Vector3(0, 1.1, 0)

function setupAvatarRenderTarget(
  canvasEl: HTMLCanvasElement,
  kind: AvatarViewKind
): {
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  kind: AvatarViewKind
} {
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20)
  positionDebugCamera(camera, kind, DEFAULT_AVATAR_FOCUS)

  const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, alpha: true, antialias: true })

  const resize = (): void => {
    const width = canvasEl.clientWidth || 1
    const height = canvasEl.clientHeight || 1
    camera.aspect = width / height
    camera.updateProjectionMatrix()
    renderer.setSize(width, height, false)
  }
  resize()
  window.addEventListener('resize', resize)

  return { camera, renderer, kind }
}

// Called once the avatar has loaded and calibrated — re-centers every debug
// camera on the avatar's own real anchor point (see calibrateToAvatar)
// instead of the DEFAULT_AVATAR_FOCUS guess they were created with, so the
// framing matches whatever avatar is actually loaded rather than an
// arbitrary height/position guess.
function repositionDebugCameras(
  targets: ReadonlyArray<{ camera: THREE.PerspectiveCamera; kind: AvatarViewKind }>,
  anchor: THREE.Vector3
): void {
  // Slightly below the shoulder midpoint so the framing centers on the
  // torso rather than clipping it at the very top of the frame.
  const focus = anchor.clone().add(new THREE.Vector3(0, -0.15, 0))
  for (const target of targets) positionDebugCamera(target.camera, target.kind, focus)
}

// ---------- Debug 3D skeleton (drawn in the avatar-debug window, for
// comparing raw tracking against what the rigged avatar is actually doing) ----------

// Mirror X/Z to match the mirrored webcam view in the tracking window.
// MediaPipe's world-landmark Y increases *downward* (like image coordinates),
// the opposite of three.js's Y-up, so it also needs to be negated.
function mirrorScale(lm: Vec3Like, scale: number): Vec3Like {
  return { x: -lm.x * scale, y: -lm.y * scale, z: -lm.z * scale }
}

class LandmarkSkeleton {
  private readonly points: THREE.Points
  private readonly lines: THREE.LineSegments
  private readonly maxPoints: number
  private readonly connections: ReadonlyArray<{ start: number; end: number }>

  constructor(maxPoints: number, connections: ReadonlyArray<{ start: number; end: number }>, color: number) {
    this.maxPoints = maxPoints
    this.connections = connections

    const pointGeom = new THREE.BufferGeometry()
    pointGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxPoints * 3), 3))
    // depthTest: false so these draw on top of the avatar's mesh — the actual
    // bones often sit inside loose clothing/sleeve geometry and would
    // otherwise be entirely hidden behind it.
    this.points = new THREE.Points(
      pointGeom,
      new THREE.PointsMaterial({ color, size: 0.03, sizeAttenuation: true, depthTest: false })
    )

    const lineGeom = new THREE.BufferGeometry()
    lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(connections.length * 2 * 3), 3))
    this.lines = new THREE.LineSegments(lineGeom, new THREE.LineBasicMaterial({ color, depthTest: false }))

    this.points.renderOrder = 999
    this.lines.renderOrder = 999
    this.points.visible = false
    this.lines.visible = false
  }

  addTo(group: THREE.Group): void {
    group.add(this.points, this.lines)
  }

  // `landmarks` must already be in final, mirrored, avatar-scaled local space —
  // callers do that transform themselves (see reconstructChain / getAvatarActual*),
  // since it now depends on avatar bone lengths rather than being a fixed
  // mirror+scale this class can do on its own.
  update(landmarks: Vec3Like[] | undefined, offset: Vec3Like = { x: 0, y: 0, z: 0 }): void {
    const visible = !!landmarks && landmarks.length > 0
    this.points.visible = visible
    this.lines.visible = visible
    if (!landmarks) return

    const toWorld = (lm: Vec3Like): [number, number, number] => [lm.x + offset.x, lm.y + offset.y, lm.z + offset.z]

    const posAttr = this.points.geometry.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < this.maxPoints; i++) {
      const lm = landmarks[i]
      const [x, y, z] = lm ? toWorld(lm) : [0, 0, 0]
      posAttr.setXYZ(i, x, y, z)
    }
    posAttr.needsUpdate = true

    const lineAttr = this.lines.geometry.attributes.position as THREE.BufferAttribute
    this.connections.forEach((conn, i) => {
      const a = landmarks[conn.start]
      const b = landmarks[conn.end]
      const [ax, ay, az] = a ? toWorld(a) : [0, 0, 0]
      const [bx, by, bz] = b ? toWorld(b) : [0, 0, 0]
      lineAttr.setXYZ(i * 2, ax, ay, az)
      lineAttr.setXYZ(i * 2 + 1, bx, by, bz)
    })
    lineAttr.needsUpdate = true
  }
}

interface ArmBoneLengths {
  upperArm: number
  lowerArm: number
}

// What a bone needs in order to be pointed at a new direction later: its rest
// direction (bone -> child bone, in world space) and its rest world rotation
// — see aimBoneToward.
interface AimCalibration {
  restDirWorld: THREE.Vector3
  restWorldQuat: THREE.Quaternion
}

interface ArmAim {
  upper: AimCalibration
  lower: AimCalibration
}

// Which bone carries torso rotation (Chest if the model has one, else Spine)
// plus its basis calibration. A single shoulder-line vector isn't enough on
// its own — it leaves the twist *around* that line unconstrained, and the
// arbitrary way setFromUnitVectors resolves that ambiguity frame to frame
// was exactly what made the torso twist and shake; a second vector (toward
// the head) pins it down fully, the same fix already used for the wrist.
// Rotating this bone before the arms each frame (aimBoneToward/aimBasisToward
// always read the parent's *current* world rotation) means the arms
// automatically keep pointing the right way in world space regardless of how
// the torso turns — no separate compensation needed.
interface TorsoAim {
  boneName: VRMHumanBoneName
  calib: BasisAimCalibration
}

// One AimCalibration per finger bone (joint1, joint2, joint3 — see fingerBoneNames).
type FingerAim = [AimCalibration, AimCalibration, AimCalibration]

// The wrist needs a full orientation match, not just one aim direction — a
// single vector leaves twist/roll around that axis undefined, which is what
// was producing a distorted hand whenever the wrist rotated (Kalidokit's
// Hand.solve was used for this before; its internal index/little-finger
// landmark selection assumes `side` reflects the real handedness of the data,
// which this app's crossed left/right mapping violates — see applyHandToVRM).
// Two vectors — forward (wrist -> middle finger) and the palm's face normal —
// fully constrain it instead.
interface BasisAimCalibration {
  restForwardWorld: THREE.Vector3
  restNormalWorld: THREE.Vector3
  restWorldQuat: THREE.Quaternion
}

interface HandCalibration {
  fingers: Record<FingerName, [number, number, number]> // wrist->j1, j1->j2, j2->j3 lengths, for the debug skeleton
  fingerAim: Record<FingerName, FingerAim>
  wristAim: BasisAimCalibration
}

interface AvatarCalibration {
  anchor: THREE.Vector3 // world position the debug skeleton group sits at (avatar's shoulder midpoint)
  leftShoulderLocal: Vec3Like // avatar's actual left/right shoulder joints, relative to anchor
  rightShoulderLocal: Vec3Like
  left: ArmBoneLengths
  right: ArmBoneLengths
  leftArmAim: ArmAim
  rightArmAim: ArmAim
  leftHand: HandCalibration
  rightHand: HandCalibration
  torso: TorsoAim
}

function getBoneWorldPos(vrm: VRM, name: VRMHumanBoneName): THREE.Vector3 {
  const pos = new THREE.Vector3()
  vrm.humanoid.getRawBoneNode(name)?.getWorldPosition(pos)
  return pos
}

function getNormalizedBoneWorldPos(vrm: VRM, name: VRMHumanBoneName): THREE.Vector3 {
  const pos = new THREE.Vector3()
  vrm.humanoid.getNormalizedBoneNode(name)?.getWorldPosition(pos)
  return pos
}

// Captures what `boneName` needs in order to later be re-aimed toward a new
// target direction (see aimBoneToward): where it pointed at rest (bone's own
// position -> `childBoneName`'s position, in world space) and its rest world
// rotation. Uses the *normalized* bone rig throughout — normalized bones (not
// raw) are what aimBoneToward rotates, and the two rigs can have different
// rest orientations for the same bone.
function calibrateAim(vrm: VRM, boneName: VRMHumanBoneName, childBoneName: VRMHumanBoneName): AimCalibration {
  const bone = vrm.humanoid.getNormalizedBoneNode(boneName)!
  const bonePos = getNormalizedBoneWorldPos(vrm, boneName)
  const childPos = getNormalizedBoneWorldPos(vrm, childBoneName)
  const restWorldQuat = new THREE.Quaternion()
  bone.getWorldQuaternion(restWorldQuat)
  return { restDirWorld: childPos.clone().sub(bonePos).normalize(), restWorldQuat }
}

// Rotates `boneName` so it points toward `targetDirWorld` instead of
// wherever it currently points — a pure direction (swing) match, no twist
// around that direction is constrained (there's no second reference vector
// to pin it, unlike aimBasisToward).`calib` must come from calibrateAim() on
// this same bone, captured before any rig rotation was ever applied.
//
// Blends two candidate rotations every frame instead of using either alone:
//
// - "rest-relative": the minimal rotation from the original REST direction
//   straight to the target (what this function used to compute exclusively).
//   Always exactly correctable — computed fresh from a fixed reference every
//   frame, so it can never accumulate drift — but `setFromUnitVectors`
//   always picks the *minimal* rotation, and which twist that minimal
//   rotation happens to carry can jump noticeably between two nearby
//   targets once the swing away from rest gets large. A fully-extended limb
//   swept through a wide arc (a straight arm raised overhead and lowered)
//   would visibly twist/warp mid-motion from this alone.
// - "continuous": a small step from wherever the bone currently is toward
//   the target, composed onto its current world rotation. Never jumps for
//   exactly that reason, but composing many such "shortest path" steps over
//   time has a real geometric-phase effect (the same phenomenon behind
//   Foucault's pendulum): moving the pointing direction around any loop that
//   encloses nonzero solid angle — completely ordinary arm movement over a
//   session — leaves behind a net twist that this candidate alone has no
//   way to ever undo, drifting further every time.
//
// Blending a small, fixed fraction back toward the drift-free rest-relative
// candidate every frame gives a continuous correction (never a snap, since
// the pulled fraction is tiny) that keeps the drift-prone continuous
// candidate from ever wandering far, while the continuous candidate still
// dominates moment-to-moment, so a single large sweep still can't visibly
// twist the way the rest-relative candidate alone would.
const AIM_DRIFT_CORRECTION_RATE = 0.04

function aimBoneToward(
  vrm: VRM,
  boneName: VRMHumanBoneName,
  calib: AimCalibration,
  targetDirWorld: THREE.Vector3,
  lerpAmount = 0.5
): void {
  const bone = vrm.humanoid.getNormalizedBoneNode(boneName)
  if (!bone || !bone.parent) return
  // The isFiniteVec3 check is defense-in-depth: every known source of a bad
  // vector is already filtered upstream (see holdJoint), but a NaN slipping
  // through into bone.quaternion.slerp() would corrupt that bone permanently
  // for the rest of the session (see the comment on holdJoint), so this
  // path guards against it too rather than trusting callers exclusively.
  if (targetDirWorld.lengthSq() < 1e-8 || !isFiniteVec3(targetDirWorld)) return

  const target = targetDirWorld.clone().normalize()

  const restDelta = new THREE.Quaternion().setFromUnitVectors(calib.restDirWorld, target)
  const restRelativeWorldQuat = restDelta.multiply(calib.restWorldQuat)

  const currentWorldQuat = new THREE.Quaternion()
  bone.getWorldQuaternion(currentWorldQuat)
  // Recovers "what direction does this bone currently effectively point",
  // purely from its live world quaternion — the same relationship the
  // rest-relative formula above uses in reverse (there, `delta *
  // restWorldQuat` is defined as the desired world quat; here we invert
  // that to read back the direction implied by whatever world quat is
  // already live).
  const currentDeltaQuat = currentWorldQuat.clone().multiply(calib.restWorldQuat.clone().invert())
  const currentPointingDir = calib.restDirWorld.clone().applyQuaternion(currentDeltaQuat).normalize()

  const incrementalDelta = new THREE.Quaternion().setFromUnitVectors(currentPointingDir, target)
  const continuousWorldQuat = incrementalDelta.multiply(currentWorldQuat)

  const desiredWorldQuat = continuousWorldQuat.slerp(restRelativeWorldQuat, AIM_DRIFT_CORRECTION_RATE)

  const parentWorldQuat = new THREE.Quaternion()
  bone.parent.getWorldQuaternion(parentWorldQuat)

  const desiredLocalQuat = parentWorldQuat.invert().multiply(desiredWorldQuat)
  bone.quaternion.slerp(desiredLocalQuat, lerpAmount)
  // aimBoneToward is called parent-before-child within a frame (upper arm
  // before lower arm); refresh this bone's own world matrix immediately so
  // the child's call sees this rotation as its parent's *current* transform,
  // not last frame's.
  bone.updateWorldMatrix(false, false)
}

// Kalidokit's 3-joint-per-finger output (Proximal/Intermediate/Distal) maps onto
// VRM's thumb, which only has 3 bones too but named Metacarpal/Proximal/Distal —
// shift the chain down one: Kalidokit's "Proximal" is the knuckle closest to the
// wrist, which corresponds to the VRM thumb's "Metacarpal". Same mapping is used
// both for rigging (applyHandToVRM) and here for the debug skeleton.
function fingerBoneNames(
  side: 'Left' | 'Right',
  finger: FingerName
): [VRMHumanBoneName, VRMHumanBoneName, VRMHumanBoneName] {
  const B = VRMHumanBoneName
  if (finger === 'Thumb') {
    return side === 'Left'
      ? [B.LeftThumbMetacarpal, B.LeftThumbProximal, B.LeftThumbDistal]
      : [B.RightThumbMetacarpal, B.RightThumbProximal, B.RightThumbDistal]
  }
  return [B[`${side}${finger}Proximal`], B[`${side}${finger}Intermediate`], B[`${side}${finger}Distal`]]
}

// b3 (the Distal bone) has no child bone to compute a "points toward" rest
// direction from, so its aim baseline instead reuses b2's rest direction —
// approximating that the finger continues in the same straight line it was
// already heading in, which holds at rest (fingers are straight, not curled).
function calibrateFingerAim(vrm: VRM, side: 'Left' | 'Right', finger: FingerName): FingerAim {
  const [b1, b2, b3] = fingerBoneNames(side, finger)
  const aim1 = calibrateAim(vrm, b1, b2)
  const aim2 = calibrateAim(vrm, b2, b3)
  const bone3 = vrm.humanoid.getNormalizedBoneNode(b3)!
  const restWorldQuat3 = new THREE.Quaternion()
  bone3.getWorldQuaternion(restWorldQuat3)
  const aim3: AimCalibration = { restDirWorld: aim2.restDirWorld.clone(), restWorldQuat: restWorldQuat3 }
  return [aim1, aim2, aim3]
}

// General-purpose 2-vector orientation helper, not hand-specific despite the
// wrist being its first use: forward = far point -> origin's "away" point
// (e.g. wrist -> middle finger, or shoulder-mid -> head); normal = cross of
// forward and an "across" vector (e.g. index-to-little across the palm, or
// left-to-right across the shoulders). Together they fully define an
// orientation. Used identically for the avatar's rest-pose bones
// (calibration) and for live tracked landmarks (at runtime) — see aimBasisToward.
function computeBasis(
  origin: Vec3Like,
  acrossA: Vec3Like,
  acrossB: Vec3Like,
  forwardPoint: Vec3Like
): { forward: THREE.Vector3; normal: THREE.Vector3 } {
  const o = new THREE.Vector3(origin.x, origin.y, origin.z)
  const forward = new THREE.Vector3(forwardPoint.x, forwardPoint.y, forwardPoint.z).sub(o).normalize()
  const across = new THREE.Vector3(acrossA.x, acrossA.y, acrossA.z).sub(
    new THREE.Vector3(acrossB.x, acrossB.y, acrossB.z)
  )
  const normal = new THREE.Vector3().crossVectors(forward, across).normalize()
  return { forward, normal }
}

function basisToQuaternion(forward: THREE.Vector3, normal: THREE.Vector3): THREE.Quaternion {
  const f = forward.clone().normalize()
  const right = new THREE.Vector3().crossVectors(normal, f).normalize()
  const up = new THREE.Vector3().crossVectors(f, right).normalize()
  const m = new THREE.Matrix4().makeBasis(right, up, f)
  return new THREE.Quaternion().setFromRotationMatrix(m)
}

function calibrateWristAim(vrm: VRM, side: 'Left' | 'Right'): BasisAimCalibration {
  const B = VRMHumanBoneName
  const hand = side === 'Left' ? B.LeftHand : B.RightHand
  const indexProx = side === 'Left' ? B.LeftIndexProximal : B.RightIndexProximal
  const littleProx = side === 'Left' ? B.LeftLittleProximal : B.RightLittleProximal
  const middleProx = side === 'Left' ? B.LeftMiddleProximal : B.RightMiddleProximal

  const wristPos = getNormalizedBoneWorldPos(vrm, hand)
  const { forward, normal } = computeBasis(
    wristPos,
    getNormalizedBoneWorldPos(vrm, indexProx),
    getNormalizedBoneWorldPos(vrm, littleProx),
    getNormalizedBoneWorldPos(vrm, middleProx)
  )

  const bone = vrm.humanoid.getNormalizedBoneNode(hand)!
  const restWorldQuat = new THREE.Quaternion()
  bone.getWorldQuaternion(restWorldQuat)

  return { restForwardWorld: forward, restNormalWorld: normal, restWorldQuat }
}

// Same idea as aimBoneToward, but matching a full 2-vector basis instead of
// one direction, since the wrist needs its twist/roll constrained too.
function aimBasisToward(
  vrm: VRM,
  boneName: VRMHumanBoneName,
  calib: BasisAimCalibration,
  targetForward: THREE.Vector3,
  targetNormal: THREE.Vector3,
  lerpAmount = 0.5
): void {
  const bone = vrm.humanoid.getNormalizedBoneNode(boneName)
  if (!bone || !bone.parent) return
  // See the matching comment in aimBoneToward — same defense-in-depth reasoning.
  if (
    targetForward.lengthSq() < 1e-8 ||
    targetNormal.lengthSq() < 1e-8 ||
    !isFiniteVec3(targetForward) ||
    !isFiniteVec3(targetNormal)
  )
    return

  const restBasis = basisToQuaternion(calib.restForwardWorld, calib.restNormalWorld)
  const targetBasis = basisToQuaternion(targetForward, targetNormal)
  const delta = targetBasis.multiply(restBasis.invert())
  const desiredWorldQuat = delta.multiply(calib.restWorldQuat)

  const parentWorldQuat = new THREE.Quaternion()
  bone.parent.getWorldQuaternion(parentWorldQuat)

  const desiredLocalQuat = parentWorldQuat.invert().multiply(desiredWorldQuat)
  bone.quaternion.slerp(desiredLocalQuat, lerpAmount)
  bone.updateWorldMatrix(false, false)
}

function calibrateHand(vrm: VRM, side: 'Left' | 'Right', wrist: THREE.Vector3): HandCalibration {
  const fingers = {} as Record<FingerName, [number, number, number]>
  const fingerAim = {} as Record<FingerName, FingerAim>
  for (const finger of FINGERS) {
    const [b1, b2, b3] = fingerBoneNames(side, finger)
    const p1 = getBoneWorldPos(vrm, b1)
    const p2 = getBoneWorldPos(vrm, b2)
    const p3 = getBoneWorldPos(vrm, b3)
    fingers[finger] = [wrist.distanceTo(p1), p1.distanceTo(p2), p2.distanceTo(p3)]
    fingerAim[finger] = calibrateFingerAim(vrm, side, finger)
  }
  return { fingers, fingerAim, wristAim: calibrateWristAim(vrm, side) }
}

// Reads the avatar's own rest-pose bone positions so the debug skeleton can be
// drawn at the avatar's actual shoulder position and bone lengths instead of
// guesses — must run before any rig rotation has been applied to the VRM.
// Chest isn't a required VRM bone — fall back to Spine if this model doesn't have one.
//
// "forward" is derived purely from the shoulder line + a fixed world-up —
// *not* shoulder-mid -> head/nose (tried first, reverted): the head sits
// mostly *above* the shoulders, not in front of them, so that vector was
// really measuring spine tilt, not facing direction, and it measures a
// different thing at rest (calibration) than in a tilted head pose while
// tracking — the mismatch between those two is what was twisting the torso
// and flipping the avatar to face backward. cross(worldUp, shoulderLine) is
// always exactly horizontal and perpendicular to the shoulders, so rest and
// live values are directly comparable, and head tilt/nod no longer leaks
// into torso rotation at all. This does mean forward/backward lean (pitch)
// isn't captured — only yaw (turning) and roll (leaning sideways) are.
const WORLD_UP = new THREE.Vector3(0, 1, 0)

function calibrateTorsoAim(vrm: VRM): TorsoAim {
  const B = VRMHumanBoneName
  const boneName = vrm.humanoid.getNormalizedBoneNode(B.Chest) ? B.Chest : B.Spine

  const leftShoulder = getNormalizedBoneWorldPos(vrm, B.LeftUpperArm)
  const rightShoulder = getNormalizedBoneWorldPos(vrm, B.RightUpperArm)
  const across = rightShoulder.clone().sub(leftShoulder)
  const forward = new THREE.Vector3().crossVectors(WORLD_UP, across).normalize()

  const bone = vrm.humanoid.getNormalizedBoneNode(boneName)!
  const restWorldQuat = new THREE.Quaternion()
  bone.getWorldQuaternion(restWorldQuat)

  return { boneName, calib: { restForwardWorld: forward, restNormalWorld: WORLD_UP.clone(), restWorldQuat } }
}

function calibrateToAvatar(vrm: VRM): AvatarCalibration {
  vrm.scene.updateMatrixWorld(true)
  const B = VRMHumanBoneName
  const lShoulder = getBoneWorldPos(vrm, B.LeftUpperArm)
  const lElbow = getBoneWorldPos(vrm, B.LeftLowerArm)
  const lWrist = getBoneWorldPos(vrm, B.LeftHand)
  const rShoulder = getBoneWorldPos(vrm, B.RightUpperArm)
  const rElbow = getBoneWorldPos(vrm, B.RightLowerArm)
  const rWrist = getBoneWorldPos(vrm, B.RightHand)

  const anchor = lShoulder.clone().add(rShoulder).multiplyScalar(0.5)

  return {
    anchor,
    leftShoulderLocal: { x: lShoulder.x - anchor.x, y: lShoulder.y - anchor.y, z: lShoulder.z - anchor.z },
    rightShoulderLocal: { x: rShoulder.x - anchor.x, y: rShoulder.y - anchor.y, z: rShoulder.z - anchor.z },
    left: { upperArm: lShoulder.distanceTo(lElbow), lowerArm: lElbow.distanceTo(lWrist) },
    right: { upperArm: rShoulder.distanceTo(rElbow), lowerArm: rElbow.distanceTo(rWrist) },
    leftArmAim: {
      upper: calibrateAim(vrm, B.LeftUpperArm, B.LeftLowerArm),
      lower: calibrateAim(vrm, B.LeftLowerArm, B.LeftHand)
    },
    rightArmAim: {
      upper: calibrateAim(vrm, B.RightUpperArm, B.RightLowerArm),
      lower: calibrateAim(vrm, B.RightLowerArm, B.RightHand)
    },
    leftHand: calibrateHand(vrm, 'Left', lWrist),
    rightHand: calibrateHand(vrm, 'Right', rWrist),
    torso: calibrateTorsoAim(vrm)
  }
}

// Takes a chain of MediaPipe raw points, keeps only the *directions* between
// consecutive points, and walks them out from `startLocal` using fixed
// `lengths` (taken from the avatar's own bones) instead of MediaPipe's — so
// the debug skeleton always matches the avatar's actual proportions rather
// than the tracked person's. Returns `lengths.length + 1` points, starting
// with `startLocal` itself.
function reconstructChain(startLocal: Vec3Like, rawPoints: Vec3Like[], lengths: number[]): Vec3Like[] {
  const points: Vec3Like[] = [startLocal]
  let prev = startLocal
  for (let i = 0; i < lengths.length; i++) {
    const a = mirrorScale(rawPoints[i], 1)
    const b = mirrorScale(rawPoints[i + 1], 1)
    const dir = new THREE.Vector3(b.x - a.x, b.y - a.y, b.z - a.z)
    if (dir.lengthSq() > 1e-8) dir.normalize()
    const next: Vec3Like = {
      x: prev.x + dir.x * lengths[i],
      y: prev.y + dir.y * lengths[i],
      z: prev.z + dir.z * lengths[i]
    }
    points.push(next)
    prev = next
  }
  return points
}

function reconstructHandPoints(
  wristLocal: Vec3Like,
  handLandmarks: Vec3Like[] | undefined,
  handCalibration: HandCalibration
): Vec3Like[] | undefined {
  if (!handLandmarks) return undefined
  const points: Vec3Like[] = []
  for (const finger of FINGERS) {
    const idxs = FINGER_LANDMARKS[finger]
    const raw = idxs.map((i) => handLandmarks[i])
    points.push(...reconstructChain(wristLocal, raw, handCalibration.fingers[finger]).slice(0, 4))
  }
  return points
}

// Live query of where the avatar's own bones actually ended up this frame
// (after rigging), in the same anchor-local space as the reconstruction
// above — lets the two be compared joint-for-joint.
function getAvatarActualArmPoints(vrm: VRM, anchor: THREE.Vector3): Vec3Like[] {
  const B = VRMHumanBoneName
  const toLocal = (p: THREE.Vector3): Vec3Like => ({ x: p.x - anchor.x, y: p.y - anchor.y, z: p.z - anchor.z })
  return [
    toLocal(getBoneWorldPos(vrm, B.LeftUpperArm)),
    toLocal(getBoneWorldPos(vrm, B.RightUpperArm)),
    toLocal(getBoneWorldPos(vrm, B.LeftLowerArm)),
    toLocal(getBoneWorldPos(vrm, B.RightLowerArm)),
    toLocal(getBoneWorldPos(vrm, B.LeftHand)),
    toLocal(getBoneWorldPos(vrm, B.RightHand))
  ]
}

function getAvatarActualHandPoints(vrm: VRM, side: 'Left' | 'Right', anchor: THREE.Vector3): Vec3Like[] {
  const toLocal = (p: THREE.Vector3): Vec3Like => ({ x: p.x - anchor.x, y: p.y - anchor.y, z: p.z - anchor.z })
  const wristLocal = toLocal(getBoneWorldPos(vrm, side === 'Left' ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand))
  const points: Vec3Like[] = []
  for (const finger of FINGERS) {
    const [b1, b2, b3] = fingerBoneNames(side, finger)
    points.push(wristLocal, toLocal(getBoneWorldPos(vrm, b1)), toLocal(getBoneWorldPos(vrm, b2)), toLocal(getBoneWorldPos(vrm, b3)))
  }
  return points
}

interface DebugSkeleton {
  group: THREE.Group
  tracked: { pose: LandmarkSkeleton; leftHand: LandmarkSkeleton; rightHand: LandmarkSkeleton }
  avatarActual: { arms: LandmarkSkeleton; leftHand: LandmarkSkeleton; rightHand: LandmarkSkeleton }
}

function setupDebugSkeleton(scene: THREE.Scene, calibration: AvatarCalibration): DebugSkeleton {
  const group = new THREE.Group()
  group.position.copy(calibration.anchor)

  // Tracked: reconstructed from the webcam, scaled to the avatar's bone lengths.
  const pose = new LandmarkSkeleton(UPPER_BODY_INDICES.length, UPPER_BODY_CONNECTIONS, 0x00d0ff)
  const leftHand = new LandmarkSkeleton(HAND_POINT_COUNT, HAND_CONNECTIONS_CALIBRATED, 0xff5f7e)
  const rightHand = new LandmarkSkeleton(HAND_POINT_COUNT, HAND_CONNECTIONS_CALIBRATED, 0xffd25f)

  // Avatar-actual: where the rigged avatar's own bones ended up this frame —
  // for a direct, joint-by-joint check against the tracked skeleton above.
  const armsActual = new LandmarkSkeleton(UPPER_BODY_INDICES.length, UPPER_BODY_CONNECTIONS, 0x39ff6a)
  const leftHandActual = new LandmarkSkeleton(HAND_POINT_COUNT, HAND_CONNECTIONS_CALIBRATED, 0x39ff6a)
  const rightHandActual = new LandmarkSkeleton(HAND_POINT_COUNT, HAND_CONNECTIONS_CALIBRATED, 0x39ff6a)

  for (const skel of [pose, leftHand, rightHand, armsActual, leftHandActual, rightHandActual]) skel.addTo(group)

  scene.add(group)
  return {
    group,
    tracked: { pose, leftHand, rightHand },
    avatarActual: { arms: armsActual, leftHand: leftHandActual, rightHand: rightHandActual }
  }
}

// Recomputes every point in the debug skeleton for this frame's tracking data
// — only meaningful in the avatar-debug window (the group stays invisible,
// and this is skipped, in avatar-clean).
function updateDebugSkeleton(
  debugSkeleton: DebugSkeleton,
  resolved: ResolvedFrame,
  calibration: AvatarCalibration,
  vrm: VRM
): void {
  let upperBody: Vec3Like[] | undefined
  let leftWristRel: Vec3Like = calibration.leftShoulderLocal
  let rightWristRel: Vec3Like = calibration.rightShoulderLocal
  if (resolved.left && resolved.right) {
    const avatarRight = reconstructChain(
      calibration.rightShoulderLocal,
      [resolved.left.shoulder, resolved.left.elbow, resolved.left.wrist],
      [calibration.right.upperArm, calibration.right.lowerArm]
    )
    const avatarLeft = reconstructChain(
      calibration.leftShoulderLocal,
      [resolved.right.shoulder, resolved.right.elbow, resolved.right.wrist],
      [calibration.left.upperArm, calibration.left.lowerArm]
    )
    // each = [shoulder, elbow, wrist]
    upperBody = [avatarLeft[0], avatarRight[0], avatarLeft[1], avatarRight[1], avatarLeft[2], avatarRight[2]]
    leftWristRel = avatarLeft[2]
    rightWristRel = avatarRight[2]
  }
  debugSkeleton.tracked.pose.update(upperBody)
  // Crossed the same way as applyHandToVRM: the avatar-left wrist is driven
  // by the person's right arm, so it's the person's right hand that has to
  // attach there too.
  debugSkeleton.tracked.leftHand.update(
    reconstructHandPoints(leftWristRel, resolved.rightHand ?? undefined, calibration.leftHand)
  )
  debugSkeleton.tracked.rightHand.update(
    reconstructHandPoints(rightWristRel, resolved.leftHand ?? undefined, calibration.rightHand)
  )

  // Avatar-actual: where the rigged avatar's bones actually ended up this
  // frame, for a direct joint-by-joint comparison against the tracked skeleton.
  // Uses the *live* group position (calibration.anchor + this frame's body
  // offset — see computeBodyOffsetTarget), not the static calibration anchor:
  // the bones' real world positions already include the body offset (the
  // whole avatar moved), so subtracting anything other than where the group
  // itself currently sits would double-count that offset.
  const liveAnchor = debugSkeleton.group.position
  debugSkeleton.avatarActual.arms.update(getAvatarActualArmPoints(vrm, liveAnchor))
  debugSkeleton.avatarActual.leftHand.update(getAvatarActualHandPoints(vrm, 'Left', liveAnchor))
  debugSkeleton.avatarActual.rightHand.update(getAvatarActualHandPoints(vrm, 'Right', liveAnchor))
}

async function loadAvatar(scene: THREE.Scene): Promise<VRM> {
  const loader = new GLTFLoader()
  loader.register((parser) => new VRMLoaderPlugin(parser))
  const gltf = await loader.loadAsync(VRM_URL)
  const vrm = gltf.userData.vrm as VRM
  VRMUtils.rotateVRM0(vrm)
  scene.add(vrm.scene)
  return vrm
}

// ---------- Adaptive smoothing (One Euro filter) ----------
//
// Applied to the final tracked direction/basis vectors (after depth
// reconstruction above, right before aimBoneToward/aimBasisToward) — not to
// the raw landmarks, since smoothing those first would feed the depth
// reconstruction stale, lagged data. A One Euro filter is used instead of a
// fixed-strength lerp because it adapts its own strength to how fast the
// tracked value is moving: nearly still, it damps hard (kills jitter);
// moving fast, it damps almost not at all (stays responsive) — a constant
// lerp factor can't do both without either leaving jitter in or adding lag
// to fast motion.
class OneEuroFilter {
  private hasPrev = false
  private xPrev = 0
  private dxPrev = 0

  constructor(
    private readonly minCutoff: number,
    private readonly beta: number,
    private readonly dCutoff = 1.0
  ) {}

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff)
    return 1 / (1 + tau / dt)
  }

  filter(x: number, dt: number): number {
    if (!this.hasPrev || dt <= 0) {
      this.hasPrev = true
      this.xPrev = x
      this.dxPrev = 0
      return x
    }
    const dx = (x - this.xPrev) / dt
    const dAlpha = this.alpha(this.dCutoff, dt)
    const dxHat = this.dxPrev + dAlpha * (dx - this.dxPrev)
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat)
    const a = this.alpha(cutoff, dt)
    const xHat = this.xPrev + a * (x - this.xPrev)
    this.xPrev = xHat
    this.dxPrev = dxHat
    return xHat
  }

  // Forces the next filter() call to treat its input as a fresh first
  // sample instead of comparing it against a stale xPrev — see
  // Vector3OneEuroFilter.reset() for why this matters after a tracking gap.
  reset(): void {
    this.hasPrev = false
  }
}

// Filters a THREE.Vector3 as 3 independent scalar channels — the direction
// vectors filtered here don't need cross-axis coupling, just per-axis
// jitter removal.
class Vector3OneEuroFilter {
  private readonly fx: OneEuroFilter
  private readonly fy: OneEuroFilter
  private readonly fz: OneEuroFilter

  constructor(minCutoff: number, beta: number, dCutoff = 1.0) {
    this.fx = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fy = new OneEuroFilter(minCutoff, beta, dCutoff)
    this.fz = new OneEuroFilter(minCutoff, beta, dCutoff)
  }

  filter(v: THREE.Vector3, dt: number): THREE.Vector3 {
    return new THREE.Vector3(this.fx.filter(v.x, dt), this.fy.filter(v.y, dt), this.fz.filter(v.z, dt))
  }

  // Call when the value being filtered just came back after a gap in
  // updates (e.g. a hand that left frame and reappeared) rather than a
  // continuous frame-to-frame reading. Without this, the filter compares the
  // new value against however-old xPrev over just one frame's dt, reads
  // that as an implausibly fast motion, and lets it straight through
  // unfiltered — exactly backwards from the smoothing this is meant to add.
  reset(): void {
    this.fx.reset()
    this.fy.reset()
    this.fz.reset()
  }
}

// minCutoff: baseline smoothing strength when the tracked point is roughly
// still (lower = smoother but more lag at low speed). beta: how fast that
// smoothing backs off as speed increases (higher = snappier during fast
// motion). Fingers are deliberately left unfiltered — they're already tuned
// snappy via FINGER_LERP, and the jitter this is fixing was reported on the
// arms/torso, not the fingers.
const ARM_FILTER_MIN_CUTOFF = 0.8
const ARM_FILTER_BETA = 0.4
const TORSO_FILTER_MIN_CUTOFF = 0.8
const TORSO_FILTER_BETA = 0.4
const WRIST_FILTER_MIN_CUTOFF = 1.0
const WRIST_FILTER_BETA = 0.4

// Keyed by *person* side for the arms/torso (matching the personLeft/
// personRight variables in applyPoseToVRM — the continuous physical signal
// is "this person's left arm", regardless of which avatar bone it's crossed
// onto) and by *avatar* side for the wrist (matching applyHandToVRM's own
// `side` parameter). Either choice is fine as long as it's used consistently
// frame to frame, since the person<->avatar crossing itself never changes
// mid-session.
const directionFilters = {
  leftUpper: new Vector3OneEuroFilter(ARM_FILTER_MIN_CUTOFF, ARM_FILTER_BETA),
  leftLower: new Vector3OneEuroFilter(ARM_FILTER_MIN_CUTOFF, ARM_FILTER_BETA),
  rightUpper: new Vector3OneEuroFilter(ARM_FILTER_MIN_CUTOFF, ARM_FILTER_BETA),
  rightLower: new Vector3OneEuroFilter(ARM_FILTER_MIN_CUTOFF, ARM_FILTER_BETA),
  torsoForward: new Vector3OneEuroFilter(TORSO_FILTER_MIN_CUTOFF, TORSO_FILTER_BETA),
  leftWristForward: new Vector3OneEuroFilter(WRIST_FILTER_MIN_CUTOFF, WRIST_FILTER_BETA),
  leftWristNormal: new Vector3OneEuroFilter(WRIST_FILTER_MIN_CUTOFF, WRIST_FILTER_BETA),
  rightWristForward: new Vector3OneEuroFilter(WRIST_FILTER_MIN_CUTOFF, WRIST_FILTER_BETA),
  rightWristNormal: new Vector3OneEuroFilter(WRIST_FILTER_MIN_CUTOFF, WRIST_FILTER_BETA)
}

// ---------- Hand reacquisition easing ----------
//
// When a hand leaves frame or gets occluded, applyHandToVRM already leaves
// its bones exactly where they were (it just returns early) — that alone
// gives "hold the last shape". The part that was still missing: the moment
// tracking picks the hand back up, aimBasisToward/applyFingerAim's normal
// per-frame lerp amounts (tuned for smoothing *continuous* tracking, not
// for closing a large one-off gap) would move most of the way to the new
// pose within a couple of frames — which reads as a snap, especially for
// fingers (FINGER_LERP is deliberately snappy). Ramping the lerp amount
// itself from small to normal over a short window after reacquisition, and
// resetting the wrist's One Euro filters (see their .reset() calls) so they
// don't mistake the reacquisition jump for real high-speed motion, spreads
// that same movement across many frames instead — the "hold, then ease into
// the next real movement" behavior being asked for.
const HAND_REACQUIRE_RAMP_FRAMES = 20
const HAND_REACQUIRE_MIN_LERP = 0.08

interface ReacquireState {
  wasTracked: boolean
  framesSinceReacquire: number
}
const handReacquireState: Record<'Left' | 'Right', ReacquireState> = {
  Left: { wasTracked: false, framesSinceReacquire: 0 },
  Right: { wasTracked: false, framesSinceReacquire: 0 }
}

// computeBasis's normal is cross(forward, index-to-little) — as the wrist
// rotates normally, that vector rotates smoothly right along with it, but
// at certain viewing angles (roughly edge-on to the camera) the index/little
// MCP landmarks become harder for MediaPipe to place accurately, and a
// slightly wrong "across" reading can put the computed normal on the wrong
// side of the forward axis for a frame or two. Since forward/normal fully
// determine a rigid orientation, that reads as the whole hand instantly
// flipping/twisting itself the rest of the way around the forearm — the
// "tangled" look — rather than as ordinary jitter, and the One Euro filter
// alone doesn't stop it (a flip that lands cleanly on the far side of the
// axis isn't fast *jitter* by the filter's own velocity-based test).
// Fixed by simply never letting the normal flip to the opposite side of
// whatever it was last frame: if this frame's raw normal points away from
// the last one, use its mirror image instead, which is an equally valid
// choice geometrically (only the *sign* is ambiguous from these 4 points —
// see computeBasis) but keeps the sign continuous over time. Cleared to
// null on every reacquisition (see applyHandToVRM) so a stale reading from
// before a tracking gap can't bias the fresh one.
const lastWristNormal: Record<'Left' | 'Right', THREE.Vector3 | null> = { Left: null, Right: null }

// Fingers get a much higher steady-state lerp than the arm/wrist default —
// small, fast joints read as laggy/disconnected-from-tracking at heavier
// smoothing, and they don't need as much damping as a big arm swing does.
const FINGER_LERP = 0.9

// Points one finger's 3 bones at the current tracked joint directions —
// same direction-matching technique as the arms, using `worldLandmarks`
// (metric 3D, not the perspective-distorted normalized ones) for the actual
// vectors. b3 (Distal) aims at the true fingertip landmark, which has no
// corresponding avatar bone but still makes a perfectly good aim target.
function applyFingerAim(
  vrm: VRM,
  side: 'Left' | 'Right',
  finger: FingerName,
  worldLandmarks: Vec3Like[],
  aim: FingerAim,
  lerpAmount: number
): void {
  const [b1, b2, b3] = fingerBoneNames(side, finger)
  const [, j1Idx, j2Idx, j3Idx] = FINGER_LANDMARKS[finger]
  const tipIdx = FINGER_TIP_LANDMARK[finger]

  const j1 = mirrorScale(worldLandmarks[j1Idx], 1)
  const j2 = mirrorScale(worldLandmarks[j2Idx], 1)
  const j3 = mirrorScale(worldLandmarks[j3Idx], 1)
  const tip = mirrorScale(worldLandmarks[tipIdx], 1)

  aimBoneToward(vrm, b1, aim[0], new THREE.Vector3(j2.x - j1.x, j2.y - j1.y, j2.z - j1.z), lerpAmount)
  aimBoneToward(vrm, b2, aim[1], new THREE.Vector3(j3.x - j2.x, j3.y - j2.y, j3.z - j2.z), lerpAmount)
  aimBoneToward(vrm, b3, aim[2], new THREE.Vector3(tip.x - j3.x, tip.y - j3.y, tip.z - j3.z), lerpAmount)
}

function applyHandToVRM(
  vrm: VRM,
  worldLandmarks: Vec3Like[] | null | undefined,
  side: 'Left' | 'Right',
  handCalibration: HandCalibration,
  dt: number
): void {
  const reacquire = handReacquireState[side]
  if (!worldLandmarks) {
    reacquire.wasTracked = false
    return
  }

  const B = VRMHumanBoneName
  const hand = side === 'Left' ? B.LeftHand : B.RightHand
  const forwardFilter = side === 'Left' ? directionFilters.leftWristForward : directionFilters.rightWristForward
  const normalFilter = side === 'Left' ? directionFilters.leftWristNormal : directionFilters.rightWristNormal

  if (!reacquire.wasTracked) {
    forwardFilter.reset()
    normalFilter.reset()
    lastWristNormal[side] = null
    reacquire.framesSinceReacquire = 0
  }
  reacquire.wasTracked = true
  const rampT = Math.min(reacquire.framesSinceReacquire / HAND_REACQUIRE_RAMP_FRAMES, 1)
  reacquire.framesSinceReacquire++
  const wristLerp = HAND_REACQUIRE_MIN_LERP + (0.5 - HAND_REACQUIRE_MIN_LERP) * rampT
  const fingerLerp = HAND_REACQUIRE_MIN_LERP + (FINGER_LERP - HAND_REACQUIRE_MIN_LERP) * rampT

  // Wrist: full basis match (see BasisAimCalibration) using wrist(0),
  // index-MCP(5), little-MCP(17), and middle-MCP(9) — mirrored the same way
  // as everything else so it stays consistent with the crossed left/right
  // hand assignment below.
  const wrist = mirrorScale(worldLandmarks[0], 1)
  const indexBase = mirrorScale(worldLandmarks[5], 1)
  const littleBase = mirrorScale(worldLandmarks[17], 1)
  const middleBase = mirrorScale(worldLandmarks[9], 1)
  const rawBasis = computeBasis(wrist, indexBase, littleBase, middleBase)
  const lastNormal = lastWristNormal[side]
  if (lastNormal && lastNormal.dot(rawBasis.normal) < 0) rawBasis.normal.negate()
  lastWristNormal[side] = rawBasis.normal.clone()

  const forward = forwardFilter.filter(rawBasis.forward, dt)
  const normal = normalFilter.filter(rawBasis.normal, dt)
  aimBasisToward(vrm, hand, handCalibration.wristAim, forward, normal, wristLerp)
  // Refresh the wrist's world matrix so the finger aims below (whose parent
  // is the wrist) see this frame's wrist rotation, not last frame's.
  vrm.humanoid.getNormalizedBoneNode(hand)?.updateWorldMatrix(false, false)

  for (const finger of FINGERS) {
    applyFingerAim(vrm, side, finger, worldLandmarks, handCalibration.fingerAim[finger], fingerLerp)
  }
}

function resetBoneToRest(vrm: VRM, boneName: VRMHumanBoneName): void {
  vrm.humanoid.getNormalizedBoneNode(boneName)?.quaternion.identity()
}

// Puppets the avatar's torso + arms into whichever calibration pose is
// currently active (see CALIBRATION_DEMO_DIRECTION), overriding live
// tracking entirely for those bones while calibration is running — see the
// call site in main()'s loop. `lerpAmount = 1` (instant, not eased) since
// this is a reference target to copy, not something that needs to feel
// responsive frame-to-frame.
function applyCalibrationDemoPose(vrm: VRM, calibration: AvatarCalibration, stage: CalibrationStage): void {
  const B = VRMHumanBoneName
  // Neutral facing, regardless of whatever direction the user is actually
  // standing — otherwise the demo pose would be skewed by their live torso
  // rotation instead of giving a clean, consistent target to copy.
  resetBoneToRest(vrm, calibration.torso.boneName)

  const pose = CALIBRATION_POSE_ORDER.find((p) => stage === p || stage === `${p}-prep`)
  const demo = pose ? CALIBRATION_DEMO_DIRECTION[pose] : null

  if (!demo) {
    resetBoneToRest(vrm, B.LeftUpperArm)
    resetBoneToRest(vrm, B.LeftLowerArm)
    resetBoneToRest(vrm, B.RightUpperArm)
    resetBoneToRest(vrm, B.RightLowerArm)
    return
  }

  aimBoneToward(vrm, B.LeftUpperArm, calibration.leftArmAim.upper, demo.upper, 1)
  aimBoneToward(vrm, B.LeftLowerArm, calibration.leftArmAim.lower, demo.lower, 1)
  aimBoneToward(vrm, B.RightUpperArm, calibration.rightArmAim.upper, demo.upper, 1)
  aimBoneToward(vrm, B.RightLowerArm, calibration.rightArmAim.lower, demo.lower, 1)
}

// Arms are rigged by directly aiming each bone at the tracked direction
// (see aimBoneToward) rather than through Kalidokit's angle heuristics —
// this is the same direction-matching the debug skeleton already uses, so
// the actual avatar now moves in lockstep with it, and it sidesteps
// Kalidokit's hip-relative "is the wrist offscreen" check entirely (which
// was misfiring against Pose.solve() previously).
//
// Left/right is deliberately crossed: the tracked person's own left
// shoulder/elbow/wrist (MediaPipe landmarks 11/13/15) drive the avatar's
// RIGHT arm bones, and the person's right (12/14/16) drives the avatar's
// LEFT — matching the mirror convention already used for the tracking
// window's 2D pane (`transform: scaleX(-1)` in style.css) and Kalidokit's
// own Pose.calcArms, which crossed the same way internally.
function applyPoseToVRM(vrm: VRM, resolved: ResolvedFrame, calibration: AvatarCalibration, dt: number): void {
  if (resolved.left && resolved.right) {
    const B = VRMHumanBoneName

    const personLeftShoulder = mirrorScale(resolved.left.shoulder, 1)
    const personLeftElbow = mirrorScale(resolved.left.elbow, 1)
    const personLeftWrist = mirrorScale(resolved.left.wrist, 1)
    const personRightShoulder = mirrorScale(resolved.right.shoulder, 1)
    const personRightElbow = mirrorScale(resolved.right.elbow, 1)
    const personRightWrist = mirrorScale(resolved.right.wrist, 1)

    // Torso rotation first — the arms below read their parent bone's
    // *current* world rotation each frame, so as long as this runs before
    // them, they automatically keep pointing the right way in world space
    // no matter how the torso turns underneath them.
    // Same cross(worldUp, shoulderLine) formula as calibrateTorsoAim, so
    // target and rest are directly comparable — which means this has to be
    // built in *avatar* right/left terms too, same as the arms just below:
    // personLeftShoulder is spatially where the avatar's right shoulder is
    // (person-left drives avatar-right), so "avatar right - avatar left" is
    // personLeftShoulder - personRightShoulder, not the other way around.
    // Getting this backwards is what was turning the avatar to face away.
    const personAcross = new THREE.Vector3(
      personLeftShoulder.x - personRightShoulder.x,
      personLeftShoulder.y - personRightShoulder.y,
      personLeftShoulder.z - personRightShoulder.z
    )
    const torsoTargetForwardRaw = new THREE.Vector3().crossVectors(WORLD_UP, personAcross).normalize()
    const torsoTargetForward = directionFilters.torsoForward.filter(torsoTargetForwardRaw, dt)
    aimBasisToward(vrm, calibration.torso.boneName, calibration.torso.calib, torsoTargetForward, WORLD_UP, 0.4)

    aimBoneToward(
      vrm,
      B.RightUpperArm,
      calibration.rightArmAim.upper,
      directionFilters.leftUpper.filter(
        new THREE.Vector3(
          personLeftElbow.x - personLeftShoulder.x,
          personLeftElbow.y - personLeftShoulder.y,
          personLeftElbow.z - personLeftShoulder.z
        ),
        dt
      )
    )
    aimBoneToward(
      vrm,
      B.RightLowerArm,
      calibration.rightArmAim.lower,
      directionFilters.leftLower.filter(
        new THREE.Vector3(
          personLeftWrist.x - personLeftElbow.x,
          personLeftWrist.y - personLeftElbow.y,
          personLeftWrist.z - personLeftElbow.z
        ),
        dt
      )
    )
    aimBoneToward(
      vrm,
      B.LeftUpperArm,
      calibration.leftArmAim.upper,
      directionFilters.rightUpper.filter(
        new THREE.Vector3(
          personRightElbow.x - personRightShoulder.x,
          personRightElbow.y - personRightShoulder.y,
          personRightElbow.z - personRightShoulder.z
        ),
        dt
      )
    )
    aimBoneToward(
      vrm,
      B.LeftLowerArm,
      calibration.leftArmAim.lower,
      directionFilters.rightLower.filter(
        new THREE.Vector3(
          personRightWrist.x - personRightElbow.x,
          personRightWrist.y - personRightElbow.y,
          personRightWrist.z - personRightElbow.z
        ),
        dt
      )
    )
  }

  // Crossed the same way as the arms just above — the person's left hand is
  // attached to the avatar's right arm, so it has to drive the avatar's right
  // hand bones too, or the fingers would visibly belong to the other arm.
  applyHandToVRM(vrm, resolved.leftHand, 'Right', calibration.rightHand, dt)
  applyHandToVRM(vrm, resolved.rightHand, 'Left', calibration.leftHand, dt)
}

async function main(): Promise<void> {
  setStatus('웹캠 요청 중...')

  // Runs on a dedicated Worker (see depthModel.ts) so its per-frame WASM
  // inference — a few hundred ms of synchronous CPU work — can't stall the
  // main thread's requestAnimationFrame loop below. Never awaited here:
  // this is a supplementary signal (see ExternalDepthContext), not
  // something the rest of the app should be blocked on starting up over.
  const depthWorker = startDepthWorker()

  const scene = setupAvatarScene()
  const debugTarget = setupAvatarRenderTarget(avatarDebugCanvas, 'front')
  const cleanTarget = setupAvatarRenderTarget(avatarCleanCanvas, 'front')
  const topTarget = setupAvatarRenderTarget(avatarTopCanvas, 'top')
  const sideTarget = setupAvatarRenderTarget(avatarSideCanvas, 'side')

  const [, holistic, vrm] = await Promise.all([
    startWebcamStream(),
    (async () => {
      setStatus('MediaPipe 모델 로딩 중...')
      return setupHolistic()
    })(),
    (async () => {
      setStatus('아바타 로딩 중...')
      return loadAvatar(scene)
    })()
  ])

  // Must happen after the avatar has loaded (needs its rest-pose bones) and
  // before any rig rotation is applied to it.
  const calibration = calibrateToAvatar(vrm)
  const debugSkeleton = setupDebugSkeleton(scene, calibration)
  repositionDebugCameras([debugTarget, cleanTarget, topTarget, sideTarget], calibration.anchor)

  // Seed the person-arm-length measurement (see PersonArmLengths) with the
  // avatar's own bone lengths, averaged across both sides — a reasonable
  // adult-proportions starting point in case the guided calibration below
  // is never run, or its 'tpose' stage produces too few good samples.
  // Reapplied every time the button below restarts calibration, so a
  // second run doesn't start from whatever the first run measured.
  const avgUpperArmLength = (calibration.left.upperArm + calibration.right.upperArm) / 2
  const avgLowerArmLength = (calibration.left.lowerArm + calibration.right.lowerArm) / 2
  personArmLengths.left.upper = avgUpperArmLength
  personArmLengths.left.lower = avgLowerArmLength
  personArmLengths.right.upper = avgUpperArmLength
  personArmLengths.right.lower = avgLowerArmLength

  calibrateButton.disabled = false
  calibrateButton.addEventListener('click', () => {
    personArmLengths.left.upper = avgUpperArmLength
    personArmLengths.left.lower = avgLowerArmLength
    personArmLengths.right.upper = avgUpperArmLength
    personArmLengths.right.lower = avgLowerArmLength
    armLengthSamples.left = { upper: [], lower: [] }
    armLengthSamples.right = { upper: [], lower: [] }
    handSizeSamples.left = []
    handSizeSamples.right = []
    handSizeReference.left = 0
    handSizeReference.right = 0
    handSizeFullReachSamples.left = []
    handSizeFullReachSamples.right = []
    handSizeAtFullReach.left = 0
    handSizeAtFullReach.right = 0
    handSizeBodyDistance.left = 0
    handSizeBodyDistance.right = 0
    armZBias.left = { upper: 0, lower: 0 }
    armZBias.right = { upper: 0, lower: 0 }
    armZBiasSamples.left = { upper: [], lower: [] }
    armZBiasSamples.right = { upper: [], lower: [] }
    externalDepthCalibration.left = { bodyDepthRaw: 0, fullReachRaw: 0, calibrated: false }
    externalDepthCalibration.right = { bodyDepthRaw: 0, fullReachRaw: 0, calibrated: false }
    externalDepthReferenceSamples.left = []
    externalDepthReferenceSamples.right = []
    externalDepthFullReachSamples.left = []
    externalDepthFullReachSamples.right = []
    bodyOffsetBaseline = null
    bodyOffsetBaselineSumX = 0
    bodyOffsetBaselineSumY = 0
    bodyOffsetBaselineSamples = 0
    calibrationStage = 'tpose-prep'
    calibrationStageDeadline = performance.now() + CALIBRATION_PREP_MS
  })

  await populateCameraList()

  const drawer = new DrawingUtils(ctx)
  setStatus('추적 중')

  let lastFrameTime = performance.now()
  let lastDiagnosticLogSecond = -1
  let lastActualLogSecond = -1
  const bodyOffset = new THREE.Vector3()

  const loop = (): void => {
    const now = performance.now()
    const deltaSeconds = (now - lastFrameTime) / 1000
    const fps = 1 / deltaSeconds
    lastFrameTime = now

    // Fire-and-forget: no-ops while the worker is still busy with the
    // previous frame, so this naturally throttles to inference speed.
    depthWorker.requestFrame(video)
    latestDepthFrame = depthWorker.getLatestFrame()
    depthModelStatus = depthWorker.getStatus()

    advanceCalibrationStage(now)
    if (calibrationStage === 'idle' || calibrationStage === 'done') {
      calibrationPromptEl.hidden = true
    } else {
      calibrationPromptEl.hidden = false
      const secondsLeft = Math.max(0, Math.ceil((calibrationStageDeadline - now) / 1000))
      calibrationPromptEl.textContent = `${CALIBRATION_PROMPT[calibrationStage]} (${secondsLeft})`
    }

    const result = holistic.detectForVideo(video, now)
    drawResult(result, drawer)

    const data = trackingDataFromResult(result)
    // Confidence-gated hold-last-good-pose + bone-length depth reconstruction,
    // shared by the actual rigging below and the debug skeleton.
    const resolved = resolveTrackingFrame(data, calibration.left, calibration.right)
    updateReadout(result, fps, resolved, calibration)

    // TEMPORARY: diagnosing a reported bent-elbow-while-standing-straight
    // symptom — dumps once/sec so it can be read off the forwarded console
    // log without needing a screenshot. Remove once that's resolved.
    const nowSecond = Math.floor(now / 1000)
    if (nowSecond !== lastDiagnosticLogSecond) {
      lastDiagnosticLogSecond = nowSecond
      console.log('[diag]', JSON.stringify({
        stage: calibrationStage,
        measuredArmLength: { left: personArmLengths.left, right: personArmLengths.right },
        zBias: { left: armZBias.left, right: armZBias.right },
        left: resolved.left
          ? {
              confident: resolved.leftConfident,
              targetBendAngleDeg: angleBetweenDeg(
                { x: resolved.left.elbow.x - resolved.left.shoulder.x, y: resolved.left.elbow.y - resolved.left.shoulder.y, z: resolved.left.elbow.z - resolved.left.shoulder.z },
                { x: resolved.left.wrist.x - resolved.left.elbow.x, y: resolved.left.wrist.y - resolved.left.elbow.y, z: resolved.left.wrist.z - resolved.left.elbow.z }
              ),
              // What the same angle would be with the zBias subtraction
              // skipped entirely — i.e. straight off MediaPipe's own raw Z,
              // no correction at all. Doesn't change any actual rigging,
              // just answers "would removing zBias help" with real numbers.
              targetBendAngleDegNoZBias: angleBetweenDeg(resolved.leftDiagnostics.rawUpperDelta, resolved.leftDiagnostics.rawLowerDelta),
              rawUpperDelta: resolved.leftDiagnostics.rawUpperDelta,
              rawLowerDelta: resolved.leftDiagnostics.rawLowerDelta,
              correctedElbow: resolved.left.elbow,
              correctedWrist: resolved.left.wrist,
              shoulder: resolved.left.shoulder
            }
          : null,
        right: resolved.right
          ? {
              confident: resolved.rightConfident,
              targetBendAngleDeg: angleBetweenDeg(
                { x: resolved.right.elbow.x - resolved.right.shoulder.x, y: resolved.right.elbow.y - resolved.right.shoulder.y, z: resolved.right.elbow.z - resolved.right.shoulder.z },
                { x: resolved.right.wrist.x - resolved.right.elbow.x, y: resolved.right.wrist.y - resolved.right.elbow.y, z: resolved.right.wrist.z - resolved.right.elbow.z }
              ),
              targetBendAngleDegNoZBias: angleBetweenDeg(resolved.rightDiagnostics.rawUpperDelta, resolved.rightDiagnostics.rawLowerDelta),
              rawUpperDelta: resolved.rightDiagnostics.rawUpperDelta,
              rawLowerDelta: resolved.rightDiagnostics.rawLowerDelta,
              correctedElbow: resolved.right.elbow,
              correctedWrist: resolved.right.wrist,
              shoulder: resolved.right.shoulder
            }
          : null
      }))
    }

    // Whole-avatar position follows the tracked shoulder line moving around
    // the frame, same as the 2D skeleton in the camera pane does.
    const bodyOffsetTarget = computeBodyOffsetTarget(data.poseLandmarks)
    if (bodyOffsetTarget) bodyOffset.lerp(bodyOffsetTarget, BODY_OFFSET_LERP)
    vrm.scene.position.copy(bodyOffset)
    debugSkeleton.group.position.copy(calibration.anchor).add(bodyOffset)

    // During calibration the avatar demonstrates the pose instead of
    // following live tracking — the user is copying the avatar, not the
    // other way around (see applyCalibrationDemoPose). Hands/fingers still
    // follow live tracking either way; only the torso/arms are puppeted.
    if (calibrationStage === 'idle' || calibrationStage === 'done') {
      applyPoseToVRM(vrm, resolved, calibration, deltaSeconds)
    } else {
      applyCalibrationDemoPose(vrm, calibration, calibrationStage)
      applyHandToVRM(vrm, resolved.leftHand, 'Right', calibration.rightHand, deltaSeconds)
      applyHandToVRM(vrm, resolved.rightHand, 'Left', calibration.leftHand, deltaSeconds)
    }
    vrm.update(deltaSeconds)
    // Bone world matrices are otherwise only refreshed inside renderer.render();
    // force it now so the avatar-actual query below reads this frame's pose,
    // not last frame's.
    vrm.scene.updateMatrixWorld(true)
    updateDebugSkeleton(debugSkeleton, resolved, calibration, vrm)

    // TEMPORARY: same diagnosis as the [diag] log above, but reading back
    // what the rig actually ended up rendering this frame — see if
    // resolveArm's already-verified-straight target direction survives
    // aimBoneToward, or gets bent somewhere in the rig application itself.
    if (nowSecond !== lastActualLogSecond) {
      lastActualLogSecond = nowSecond
      const actual = getAvatarActualArmPoints(vrm, debugSkeleton.group.position)
      const [leftUpperArm, rightUpperArm, leftLowerArm, rightLowerArm, leftHand, rightHand] = actual
      const vecBetween = (from: Vec3Like, to: Vec3Like): Vec3Like => ({ x: to.x - from.x, y: to.y - from.y, z: to.z - from.z })
      console.log('[diag-actual]', JSON.stringify({
        // avatar-Right = driven by person-left; avatar-Left = driven by person-right
        avatarRight: {
          upperArm: rightUpperArm,
          lowerArm: rightLowerArm,
          hand: rightHand,
          actualBendAngleDeg: angleBetweenDeg(vecBetween(rightUpperArm, rightLowerArm), vecBetween(rightLowerArm, rightHand))
        },
        avatarLeft: {
          upperArm: leftUpperArm,
          lowerArm: leftLowerArm,
          hand: leftHand,
          actualBendAngleDeg: angleBetweenDeg(vecBetween(leftUpperArm, leftLowerArm), vecBetween(leftLowerArm, leftHand))
        }
      }))
    }

    // Same scene, same pose, rendered twice — once with the debug skeleton
    // on (for the debug pane), once with it off (for the clean pane).
    debugSkeleton.group.visible = true
    debugTarget.renderer.render(scene, debugTarget.camera)
    topTarget.renderer.render(scene, topTarget.camera)
    sideTarget.renderer.render(scene, sideTarget.camera)
    debugSkeleton.group.visible = false
    cleanTarget.renderer.render(scene, cleanTarget.camera)

    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}

main().catch((err) => {
  console.error(err)
  setStatus(`오류: ${(err as Error).message}`)
})
