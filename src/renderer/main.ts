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
const measureButton = document.getElementById('measure-arm-button') as HTMLButtonElement
const measureStatusEl = document.getElementById('measure-arm-status') as HTMLElement
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
  // unlike the world ones above, these shrink/grow with distance from the
  // camera and share the Pose Landmarker's own image-space coordinate
  // convention, which is exactly what handOcclusionBlend compares against.
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

// Persists the last resolved forward/backward sign for one arm segment's
// depth reconstruction (see computeBlendedDepth) — `0` means "never resolved
// yet". The held value is reused instead of re-deriving a sign fresh from
// MediaPipe's own noisy raw Z every frame, which could flip on a single
// noisy frame and — now that reconstruction runs unconditionally, so its
// magnitude can be as large as the whole bone length — made the arm visibly
// snap from reaching forward to folded backward and back.
//
// Resolving it once and then holding it *forever*, with no way to ever
// revisit that first guess, turned out to have its own failure mode: at
// rest, an arm's real depth barely changes, so raw Z there is mostly noise
// — whichever way that noise happened to point on the very first live frame
// gets locked in permanently, for the rest of the session, regardless of
// how the arm actually moves afterward (seen in practice as one arm always
// resolving in front of the body and the other always behind, with no
// per-frame flicker at all — a static bias, not the jitter this was
// originally built to prevent). `disagreeStreak` lets the hold still
// self-correct: a *single* confident-but-wrong frame is still ignored (same
// protection as before), but several confident, consistently-disagreeing
// frames in a row are treated as real evidence the initial guess was wrong.
interface SignHold {
  current: number
  disagreeStreak: number
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

// A person's real bone lengths don't change during a tracking session, but
// there's no longer a guided measurement step to determine them from this
// person's actual body — resolveArm's bone-length depth reconstruction
// (see computeBlendedDepth) uses the avatar's own bone lengths instead
// (seeded once in main() and never changed after), which is only ever an
// approximation of this specific person's real proportions.
interface PersonArmLengths {
  upper: number
  lower: number
}
const personArmLengths: { left: PersonArmLengths; right: PersonArmLengths } = {
  left: { upper: 0, lower: 0 },
  right: { upper: 0, lower: 0 }
}

// Diagnostic-only numbers surfaced in the on-screen readout — see
// computeBlendedDepth.
interface ArmDiagnostics {
  upperDepth: number
  lowerDepth: number
  // Raw (pre-correction) X/Y/Z displacement for each segment, straight from
  // MediaPipe's world landmarks — added specifically to check, with actual
  // numbers instead of eyeballing a screenshot, how much of a segment's
  // movement is really vertical (Y) vs. depth (Z) rather than assuming.
  rawUpperDelta: Vec3Like
  rawLowerDelta: Vec3Like
  // The actual (span-ratio vs. hand-occlusion, whichever's bigger) blend
  // weight used for the lower-arm straightening correction this frame — see
  // lowerBlendForStraightening and handOcclusionBlend.
  lowerBlendForStraightening: number
  // On-screen (X/Y only) segment length, pre-straightening — see
  // startArmCalibration, which is the only other consumer of these.
  upperXY: number
  lowerXY: number
}
const ZERO_DELTA: Vec3Like = { x: 0, y: 0, z: 0 }
const NO_ARM_DIAGNOSTICS: ArmDiagnostics = {
  upperDepth: 0,
  lowerDepth: 0,
  rawUpperDelta: ZERO_DELTA,
  rawLowerDelta: ZERO_DELTA,
  lowerBlendForStraightening: 0,
  upperXY: 0,
  lowerXY: 0
}

// Below this fraction of the known real bone length, the on-screen (XY)
// length is short enough that the segment is plausibly reaching toward or
// away from the camera rather than just being tracking noise, so depth
// reconstruction starts blending in. Blending smoothly instead of switching
// on/off at a hard cutoff avoids a visible snap right at the boundary, and
// keeps the correction's own sign ambiguity (see below) confined to the
// region where the correction is smallest anyway.
const RECON_BLEND_START_RATIO = 0.6

// Below this on-screen distance between the elbow (Pose Landmarker) and the
// hand's own wrist point (Hand Landmarker — a completely separate model),
// relative to this frame's own upper-arm on-screen length (so it isn't
// fooled by the person standing closer to or farther from the camera), the
// two models are effectively reporting the elbow and the hand at the same
// spot in the actual camera image. See handOcclusionBlend.
const HAND_OCCLUSION_START_RATIO = 0.5

// Direct, independent-of-computeBlendedDepth evidence that the hand is
// visually hiding the elbow from the camera — two separate models (Pose
// Landmarker's elbow, Hand Landmarker's wrist) agreeing they're in the same
// place on-screen is much stronger than inferring it indirectly from how
// foreshortened the shoulder-to-wrist span looks (see the comment above
// upperBlendForStraightening/lowerBlendForStraightening), and unlike that
// span-based ratio, it doesn't require the *upper* arm to also look
// foreshortened before it can fire. `shoulderImg`/`elbowImg` must be the
// Pose Landmarker's normalized image-space positions (not world) — this is
// a purely on-screen check, in the same coordinate space handLandmarks is
// already in.
function handOcclusionBlend(shoulderImg: Vec3Like | undefined, elbowImg: Vec3Like | undefined, handLandmarks: Vec3Like[] | undefined): number {
  if (!shoulderImg || !elbowImg || !handLandmarks) return 0
  const upperArmImgLength = Math.hypot(elbowImg.x - shoulderImg.x, elbowImg.y - shoulderImg.y)
  if (upperArmImgLength < 1e-4) return 0
  const handWrist = handLandmarks[0]
  const dist = Math.hypot(elbowImg.x - handWrist.x, elbowImg.y - handWrist.y)
  const ratio = dist / upperArmImgLength
  if (ratio >= HAND_OCCLUSION_START_RATIO) return 0
  return 1 - ratio / HAND_OCCLUSION_START_RATIO
}

// See SignHold.
const armSignHold: { left: { upper: SignHold; lower: SignHold }; right: { upper: SignHold; lower: SignHold } } = {
  left: { upper: { current: 0, disagreeStreak: 0 }, lower: { current: 0, disagreeStreak: 0 } },
  right: { upper: { current: 0, disagreeStreak: 0 }, lower: { current: 0, disagreeStreak: 0 } }
}

// Computes the Z offset (parent -> child) from the constraint that their
// real distance apart is `realLength` (Pythagorean: dz = sqrt(realLength^2
// - xyLength^2)) — for almost every pose, not just when the segment looks
// foreshortened enough to plausibly be a forward/backward reach. Reliable
// X/Y tracking plus a known real bone length is a better depth source in
// almost every pose than MediaPipe's own weak Z axis ever was (a real test
// standing at attention with the old ratio-gated version, which trusted raw
// Z outright above a foreshortening threshold, showed a ~25-40deg fake
// elbow bend even standing still).
//
// "Almost every" excludes one region on purpose — see
// RECON_DEAD_ZONE_RATIO below — because this formula's own sensitivity to
// xyLength blows up right where a standing-still arm actually lives.
// d/d(xyLength)[sqrt(realLength^2 - xyLength^2)] = -xyLength /
// sqrt(realLength^2 - xyLength^2), which diverges as xyLength ->
// realLength. Concretely, at realLength=0.25m: a 1mm tracking-noise wobble
// in xyLength at xyLength=0.20m only moves the computed depth by ~0.3mm,
// but the *same* 1mm wobble at xyLength=0.249m (an arm held almost, but not
// perfectly, straight — i.e. an ordinary standing pose, not a special
// case) moves it by over 22mm. Left unguarded, that's a visibly trembling
// avatar arm from nothing but MediaPipe's normal per-frame landmark jitter,
// even when the person is holding perfectly still.
//
// Raw Z only ever gets a vote on which *sign* (toward vs. away from the
// camera) the reconstructed magnitude should take, never on magnitude
// itself. The *held* sign from a previous frame is reused (see SignHold)
// rather than re-derived from raw Z's own noisy sign every frame —
// re-deriving it fresh let a single noisy frame flip the sign, and since
// the magnitude can be as large as the whole bone length, a flip showed up
// as the arm visibly snapping from reaching forward to folded backward and
// back. But raw Z near rest is mostly noise, small in magnitude — below
// SIGN_CONFIDENCE_THRESHOLD, it's never trusted even to resolve the sign for
// the first time, let alone override an already-held one. Above it,
// SIGN_FLIP_HYSTERESIS_FRAMES consecutive disagreeing frames (not just one)
// are required to actually flip an existing hold — see SignHold.
const SIGN_CONFIDENCE_THRESHOLD = 0.03 // meters
const SIGN_FLIP_HYSTERESIS_FRAMES = 8
// Above this xyLength/realLength ratio, the segment is close enough to
// fully straight that the sensitivity blow-up described above makes the
// Pythagorean formula's own output less trustworthy than just calling the
// depth zero — an everyday near-straight pose (standing still, an arm
// raised to head height, etc.), not a deliberate reach, is exactly what
// lives in this zone. Below it, the full reconstruction fades back in
// smoothly (not a hard on/off switch, to avoid its own visible snap at the
// boundary) as the segment looks more and more genuinely foreshortened.
const RECON_DEAD_ZONE_RATIO = 0.9
function computeBlendedDepth(parent: Vec3Like, child: Vec3Like, realLength: number, signHold: SignHold): number {
  const rawDz = child.z - parent.z
  if (realLength < 1e-4) return rawDz // no bone length known yet — trust MediaPipe's own Z as-is

  const dx = child.x - parent.x
  const dy = child.y - parent.y
  const xyLength = Math.hypot(dx, dy)
  const straightnessRatio = xyLength / realLength
  const reconstructedDepth =
    straightnessRatio >= RECON_DEAD_ZONE_RATIO
      ? 0
      : Math.sqrt(Math.max(realLength * realLength - xyLength * xyLength, 0)) * (1 - straightnessRatio / RECON_DEAD_ZONE_RATIO)

  if (Math.abs(rawDz) >= SIGN_CONFIDENCE_THRESHOLD) {
    const rawSign = Math.sign(rawDz)
    if (signHold.current === 0) {
      signHold.current = rawSign // first-ever confident resolution for this segment — nothing held yet
    } else if (rawSign !== signHold.current) {
      signHold.disagreeStreak += 1
      if (signHold.disagreeStreak >= SIGN_FLIP_HYSTERESIS_FRAMES) {
        signHold.current = rawSign
        signHold.disagreeStreak = 0
      }
    } else {
      signHold.disagreeStreak = 0
    }
  }
  return signHold.current * reconstructedDepth
}

// Resolves one side's shoulder/elbow/wrist independently (see holdJoint),
// then chains depth reconstruction outward from the shoulder (the wrist's
// correction builds on the already-corrected elbow, not the raw one).
// `confident` reports the *wrist's* own live status — the signal the hand
// landmarks below piggyback their own confidence gating on, since they
// don't have a usable one of their own.
function resolveArm(
  poseWorldLm: Landmark[] | undefined,
  poseImageLm: Vec3Like[] | undefined, // normalized image-space — see handOcclusionBlend
  shoulderIdx: number,
  elbowIdx: number,
  wristIdx: number,
  estimate: PersonArmLengths,
  handLandmarks: Vec3Like[] | undefined, // normalized image-space — see handOcclusionBlend
  signHold: { upper: SignHold; lower: SignHold },
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

  if (!shoulder.point || !elbow.point || !wrist.point) {
    return { arm: null, confident: false, diagnostics: NO_ARM_DIAGNOSTICS }
  }

  const upperDepth = computeBlendedDepth(shoulder.point, elbow.point, estimate.upper, signHold.upper)
  const correctedElbow: Vec3Like = { x: elbow.point.x, y: elbow.point.y, z: shoulder.point.z + upperDepth }
  const lowerDepth = computeBlendedDepth(correctedElbow, wrist.point, estimate.lower, signHold.lower)
  const correctedWrist: Vec3Like = { x: wrist.point.x, y: wrist.point.y, z: correctedElbow.z + lowerDepth }

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
  const wholeArmLength = estimate.upper + estimate.lower

  const spanBasedUpperBlend = upperRatio >= RECON_BLEND_START_RATIO ? 0 : 1 - upperRatio / RECON_BLEND_START_RATIO
  // Two independent signals for "the lower arm is foreshortened/hidden
  // toward the camera": the span-based ratio above, and direct on-screen
  // hand-vs-elbow proximity (see handOcclusionBlend) — whichever is more
  // confident wins, since either alone is sufficient evidence.
  const spanBasedLowerBlend = lowerRatio >= RECON_BLEND_START_RATIO ? 0 : 1 - lowerRatio / RECON_BLEND_START_RATIO
  const handOcclusionBlendRaw = handOcclusionBlend(poseImageLm?.[shoulderIdx], poseImageLm?.[elbowIdx], handLandmarks)

  // Both signals above go through the elbow's own on-screen position —
  // upperRatio directly, handOcclusionBlend by comparing it against the
  // hand. When the elbow is genuinely hidden, MediaPipe doesn't report a
  // small-but-honest position there, it extrapolates one from scratch, and
  // that guess can land anywhere (typically drooping down) — including
  // somewhere that looks *not* foreshortened and isn't close to the hand
  // either, fooling both checks into never firing at all even though the
  // arm really is fully extended. Shoulder-to-wrist span never touches the
  // elbow (the wrist is corroborated by the tracked hand, see holdJoint),
  // so it can't be fooled the same way — but on its own it's genuinely
  // ambiguous (see the comment above), so it only counts once there's also
  // real, substantial camera-axis depth between wrist and shoulder (not
  // just SIGN_CONFIDENCE_THRESHOLD-level noise), which a merely bent elbow
  // (hand pulled toward the shoulder without actually reaching forward)
  // never has.
  const wholeArmXY = Math.hypot(wrist.point.x - shoulder.point.x, wrist.point.y - shoulder.point.y)
  const wholeArmRatio = wholeArmLength < 1e-4 ? 1 : wholeArmXY / wholeArmLength
  const wholeArmHasRealDepth = Math.abs(wrist.point.z - shoulder.point.z) >= SIGN_CONFIDENCE_THRESHOLD
  const wholeArmBlendRaw =
    wholeArmHasRealDepth && wholeArmRatio < RECON_BLEND_START_RATIO ? 1 - wholeArmRatio / RECON_BLEND_START_RATIO : 0

  const upperBlendForStraightening = Math.max(spanBasedUpperBlend, wholeArmBlendRaw)
  const lowerBlendForStraightening = Math.max(spanBasedLowerBlend, handOcclusionBlendRaw, wholeArmBlendRaw)

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
    diagnostics: { upperDepth, lowerDepth, rawUpperDelta, rawLowerDelta, lowerBlendForStraightening, upperXY, lowerXY }
  }
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
}

function resolveTrackingFrame(data: TrackingData): ResolvedFrame {
  const left = resolveArm(
    data.poseWorldLandmarks,
    data.poseLandmarks,
    11,
    13,
    15,
    personArmLengths.left,
    data.leftHandLandmarks,
    armSignHold.left,
    heldLeftShoulder,
    heldLeftElbow,
    heldLeftWrist
  )
  const right = resolveArm(
    data.poseWorldLandmarks,
    data.poseLandmarks,
    12,
    14,
    16,
    personArmLengths.right,
    data.rightHandLandmarks,
    armSignHold.right,
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
    rightDiagnostics: right.diagnostics
  }
}

// ---------- Guided arm-length self-calibration (see PersonArmLengths) ----------
// personArmLengths is seeded from the avatar's own bone lengths, which are
// only ever an approximation of this specific person's real proportions —
// any mismatch injects a spurious depth offset via computeBlendedDepth's
// sqrt(realLength^2 - xyLength^2) in *every* pose, not just reaches. This
// briefly guides the person through a continuous motion (rather than one
// static pose) to measure their real upper/lower arm length instead.
//
// The motion is chosen so the on-screen (X/Y) segment length gets close to
// the segment's true length for a good stretch of it: raising the arms
// from the side up overhead, and spreading them from a T-pose out to
// pointing forward, both sweep through angles that stay mostly in the
// camera's own image plane before ending pointed at the camera. The final
// "pointed at the camera" instant of each sweep — where the hand can hide
// the elbow and the on-screen reading is least trustworthy — is
// deliberately never singled out or specially handled: a high percentile
// of everything sampled during the sweep already sits close to the true
// length (most of the sweep is near it), while that one foreshortened tail
// end naturally falls to the *low* end of the sample distribution and gets
// left out of a high percentile on its own, with no separate case needed.
interface CalibrationPhase {
  instruction: string
  durationMs: number
  sample: boolean
}
const CALIBRATION_PHASES: CalibrationPhase[] = [
  { instruction: '차렷 자세로 서주세요', durationMs: 2000, sample: true },
  { instruction: '천천히 팔을 만세 자세까지 들어올리세요', durationMs: 4000, sample: true },
  { instruction: '이제 팔을 T자세(옆으로 쭉)로 벌려주세요', durationMs: 2000, sample: true },
  { instruction: '천천히 팔을 앞으로 나란히 자세까지 뻗어주세요', durationMs: 4000, sample: true },
  { instruction: '측정 완료', durationMs: 0, sample: false }
]

interface ArmLengthSamples {
  upper: number[]
  lower: number[]
}
let calibrationSamples: { left: ArmLengthSamples; right: ArmLengthSamples } | null = null
let calibrationPhaseIndex = -1
let calibrationPhaseStartedAt = 0
let calibrationActive = false

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))))
  return sorted[idx]
}

// A single spike frame must not corrupt the estimate the way a plain
// running max would (see holdJoint's isFiniteVec3 comment for the same
// class of one-bad-frame problem). A high percentile of everything sampled
// during the sweep converges on close to the same value a max would find —
// most of the sweep already sits near the true length — but one outlier
// frame is diluted by the rest of the distribution instead of permanently
// overriding it.
const CALIBRATION_PERCENTILE = 90
const MIN_CALIBRATION_SAMPLES = 20

function startArmCalibration(): void {
  calibrationSamples = { left: { upper: [], lower: [] }, right: { upper: [], lower: [] } }
  calibrationPhaseIndex = 0
  calibrationPhaseStartedAt = performance.now()
  calibrationActive = true
  measureButton.disabled = true
  measureStatusEl.textContent = CALIBRATION_PHASES[0].instruction
}

// Called once per frame from the main loop while a calibration is running.
function updateArmCalibration(resolved: ResolvedFrame): void {
  if (!calibrationActive || !calibrationSamples) return
  const phase = CALIBRATION_PHASES[calibrationPhaseIndex]
  if (phase.sample) {
    if (resolved.leftConfident) {
      calibrationSamples.left.upper.push(resolved.leftDiagnostics.upperXY)
      calibrationSamples.left.lower.push(resolved.leftDiagnostics.lowerXY)
    }
    if (resolved.rightConfident) {
      calibrationSamples.right.upper.push(resolved.rightDiagnostics.upperXY)
      calibrationSamples.right.lower.push(resolved.rightDiagnostics.lowerXY)
    }
  }

  if (performance.now() - calibrationPhaseStartedAt < phase.durationMs) return
  calibrationPhaseIndex += 1
  if (calibrationPhaseIndex >= CALIBRATION_PHASES.length) {
    finishArmCalibration()
    return
  }
  calibrationPhaseStartedAt = performance.now()
  measureStatusEl.textContent = CALIBRATION_PHASES[calibrationPhaseIndex].instruction
}

function finishArmCalibration(): void {
  const samples = calibrationSamples
  calibrationActive = false
  calibrationSamples = null
  measureButton.disabled = false
  if (!samples) return

  const apply = (side: PersonArmLengths, s: ArmLengthSamples): void => {
    if (s.upper.length >= MIN_CALIBRATION_SAMPLES) side.upper = percentile(s.upper, CALIBRATION_PERCENTILE)
    if (s.lower.length >= MIN_CALIBRATION_SAMPLES) side.lower = percentile(s.lower, CALIBRATION_PERCENTILE)
  }
  apply(personArmLengths.left, samples.left)
  apply(personArmLengths.right, samples.right)

  measureStatusEl.textContent = '측정 완료'
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
// How many good frames to average into the baseline before locking it —
// using just wherever the person happened to be standing on a single first
// detected frame produced a persistent pull toward whatever direction that
// one frame was off-center from center-frame.
const BODY_OFFSET_BASELINE_SAMPLES_NEEDED = 60

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

  if (bodyOffsetBaselineSamples < BODY_OFFSET_BASELINE_SAMPLES_NEEDED) {
    bodyOffsetBaselineSumX += shoulderMid.x
    bodyOffsetBaselineSumY += shoulderMid.y
    bodyOffsetBaselineSamples++
    bodyOffsetBaseline = { x: bodyOffsetBaselineSumX / bodyOffsetBaselineSamples, y: bodyOffsetBaselineSumY / bodyOffsetBaselineSamples }
  }

  const baseline = bodyOffsetBaseline ?? shoulderMid // always set above on the very first call
  const dx = shoulderMid.x - baseline.x
  const dy = shoulderMid.y - baseline.y
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

measureButton.addEventListener('click', startArmCalibration)

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
  const avatar = { left: calibration.left, right: calibration.right }
  const fmt = (upper: number, lower: number): string => `${upper.toFixed(3)}/${lower.toFixed(3)}/${(upper + lower).toFixed(3)}`
  const fmtDelta = (d: Vec3Like): string => `x${d.x.toFixed(3)}/y${d.y.toFixed(3)}/z${d.z.toFixed(3)}`
  const lines = [
    `fps: ${fps.toFixed(1)}`,
    `pose landmarks: ${result.poseLandmarks[0]?.length ?? 0}`,
    `left hand landmarks: ${result.leftHandLandmarks[0]?.length ?? 0}`,
    `right hand landmarks: ${result.rightHandLandmarks[0]?.length ?? 0}`,
    `팔 신뢰도 — L: ${resolved.leftConfident ? 'OK' : 'HOLD'} / R: ${resolved.rightConfident ? 'OK' : 'HOLD'}`,
    `아바타 뼈 길이(상완/전완/전체) L: ${fmt(avatar.left.upperArm, avatar.left.lowerArm)} R: ${fmt(avatar.right.upperArm, avatar.right.lowerArm)}`,
    `뻗기 깊이(상완/전완) L: ${resolved.leftDiagnostics.upperDepth.toFixed(3)}/${resolved.leftDiagnostics.lowerDepth.toFixed(3)} R: ${resolved.rightDiagnostics.upperDepth.toFixed(3)}/${resolved.rightDiagnostics.lowerDepth.toFixed(3)}`,
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

  // Seed the depth-reconstruction bone lengths (see PersonArmLengths) with
  // the avatar's own bone lengths, averaged across both sides — there's no
  // per-person measurement step, so this approximation is all resolveArm
  // ever has to work with.
  const avgUpperArmLength = (calibration.left.upperArm + calibration.right.upperArm) / 2
  const avgLowerArmLength = (calibration.left.lowerArm + calibration.right.lowerArm) / 2
  personArmLengths.left.upper = avgUpperArmLength
  personArmLengths.left.lower = avgLowerArmLength
  personArmLengths.right.upper = avgUpperArmLength
  personArmLengths.right.lower = avgLowerArmLength

  await populateCameraList()

  const drawer = new DrawingUtils(ctx)
  setStatus('추적 중')
  measureButton.disabled = false

  let lastFrameTime = performance.now()
  const bodyOffset = new THREE.Vector3()

  const loop = (): void => {
    const now = performance.now()
    const deltaSeconds = (now - lastFrameTime) / 1000
    const fps = 1 / deltaSeconds
    lastFrameTime = now

    const result = holistic.detectForVideo(video, now)
    drawResult(result, drawer)

    const data = trackingDataFromResult(result)
    // Confidence-gated hold-last-good-pose + bone-length depth reconstruction,
    // shared by the actual rigging below and the debug skeleton.
    const resolved = resolveTrackingFrame(data)
    updateArmCalibration(resolved)
    updateReadout(result, fps, resolved, calibration)

    // Whole-avatar position follows the tracked shoulder line moving around
    // the frame, same as the 2D skeleton in the camera pane does.
    const bodyOffsetTarget = computeBodyOffsetTarget(data.poseLandmarks)
    if (bodyOffsetTarget) bodyOffset.lerp(bodyOffsetTarget, BODY_OFFSET_LERP)
    vrm.scene.position.copy(bodyOffset)
    debugSkeleton.group.position.copy(calibration.anchor).add(bodyOffset)

    applyPoseToVRM(vrm, resolved, calibration, deltaSeconds)
    vrm.update(deltaSeconds)
    // Bone world matrices are otherwise only refreshed inside renderer.render();
    // force it now so the avatar-actual query below reads this frame's pose,
    // not last frame's.
    vrm.scene.updateMatrixWorld(true)
    updateDebugSkeleton(debugSkeleton, resolved, calibration, vrm)

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
