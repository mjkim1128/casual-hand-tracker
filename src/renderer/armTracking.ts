// Arm tracking pipeline of the avatar app (main.ts): turns one frame of MediaPipe Holistic output into
// shoulder/elbow/wrist positions for both arms, using a two-bone IK solve instead of reconstructing
// per-bone camera depth.
//
//  - xy of every joint is the picture: the direction of each bone comes from the image
//    landmarks, its length from the world landmarks (withImageXY).
//  - The wrist's raw world position (direction + distance from the shoulder) is the IK target;
//    the person's own bone lengths (measured once in a T-pose, auto-detected, persisted in
//    localStorage) are enforced exactly via the law of cosines, so the elbow can never end up
//    at an impossible distance from the shoulder no matter how noisy the raw landmark is. The
//    one remaining degree of freedom (which way the elbow points around the shoulder-wrist
//    axis) comes from the raw elbow landmark, used only as a bend-direction hint - never as a
//    trusted position - and is held across frames where that hint is missing or degenerate.
//  - Confidence gating (hold-last-good joint), bone-length outlier rejection and elbow/wrist
//    glitch handling still run on the raw landmarks before the IK solve sees them.
//
// Module state is per window (one person, one camera).
import type { Landmark } from '@mediapipe/tasks-vision'
import type { HolisticLandmarkerResult } from '@mediapipe/tasks-vision'

// Status text (T-pose calibration progress) goes to whichever UI registered here.
let statusSink: (text: string) => void = () => {}
export function setTrackingStatusHandler(handler: (text: string) => void): void {
  statusSink = handler
}
function setStatus(text: string): void {
  statusSink(text)
}

// Diagnostic logging ([ik], [tpose], [glitch], ...) - main.ts turns it on.
let debugEnabled = false
export function setTrackingDebug(enabled: boolean): void {
  debugEnabled = enabled
}
function debugLog(message: string): void {
  if (debugEnabled) console.log(message)
}

export type Vec3Like = { x: number; y: number; z: number }

export function isFiniteVec3(v: Vec3Like): boolean {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z)
}

export interface TrackingData {
  poseWorldLandmarks?: Landmark[]
  leftHandLandmarks?: Vec3Like[]
  rightHandLandmarks?: Vec3Like[]
  // Metric (meters) 3D hand landmarks - paired with the image ones above to
  // measure how strongly the hand is magnified by its distance to the camera.
  leftHandWorldLandmarks?: Vec3Like[]
  rightHandWorldLandmarks?: Vec3Like[]
  poseLandmarks?: Vec3Like[]
}

export function trackingDataFromResult(result: HolisticLandmarkerResult): TrackingData {
  return {
    poseWorldLandmarks: result.poseWorldLandmarks[0],
    leftHandLandmarks: result.leftHandLandmarks[0],
    rightHandLandmarks: result.rightHandLandmarks[0],
    leftHandWorldLandmarks: result.leftHandWorldLandmarks[0],
    rightHandWorldLandmarks: result.rightHandWorldLandmarks[0],
    poseLandmarks: result.poseLandmarks[0]
  }
}

// ---------- Image-space direction, world-space length ----------
// MediaPipe's world landmarks' x/y are a separately regressed output whose
// in-plane direction was measured to disagree with the image landmarks' by
// ~10° on average (tens of degrees when the arm is poorly visible), so each
// segment's on-screen DIRECTION is taken from the image landmarks. World x/y
// is in meters with no perspective magnification, so the LENGTH comes from
// there. Each segment = image direction * world length, chained from the
// world shoulder. Z stays world Z.
let rawShoulderWidthXY: number | null = null
let currentAspect = 16 / 9

const IMAGE_XY_CHAINS: Array<[number, number]> = [
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16]
]

export function withImageXY(data: TrackingData, aspect: number): TrackingData {
  const world = data.poseWorldLandmarks
  const image = data.poseLandmarks
  if (!world || !image || world.length < 17 || image.length < 17) return data

  currentAspect = aspect
  // Raw world shoulder width, kept for the ambient shoulder-width estimate.
  const worldShoulderDist = Math.hypot(world[11].x - world[12].x, world[11].y - world[12].y)
  rawShoulderWidthXY = worldShoulderDist > 1e-3 ? worldShoulderDist : null

  const rebuilt = world.slice()
  const placed = new Map<number, { x: number; y: number }>([
    [11, { x: world[11].x, y: world[11].y }],
    [12, { x: world[12].x, y: world[12].y }]
  ])
  for (const [parentIdx, childIdx] of IMAGE_XY_CHAINS) {
    const parent = placed.get(parentIdx)
    if (!parent) continue
    const worldDx = world[childIdx].x - world[parentIdx].x
    const worldDy = world[childIdx].y - world[parentIdx].y
    const worldLength = Math.hypot(worldDx, worldDy)
    const imageDx = (image[childIdx].x - image[parentIdx].x) * aspect
    const imageDy = image[childIdx].y - image[parentIdx].y
    const imageLength = Math.hypot(imageDx, imageDy)
    // A segment with no on-screen extent has no image direction to use.
    const pos =
      imageLength > 1e-6
        ? { x: parent.x + (imageDx / imageLength) * worldLength, y: parent.y + (imageDy / imageLength) * worldLength }
        : { x: parent.x + worldDx, y: parent.y + worldDy }
    placed.set(childIdx, pos)
    rebuilt[childIdx] = { ...world[childIdx], x: pos.x, y: pos.y }
  }
  return { ...data, poseWorldLandmarks: rebuilt }
}

// ---------- Tracking-frame resolution — copied from main.ts, see the header
// comment above for why this isn't a shared module (yet). ----------

const MIN_ARM_VISIBILITY = 0.5
// The wrist's visibility score drops hard while the arm is stretched up or
// toward the camera even though its position is still roughly right; the
// bone-length outlier check (BONE_OUTLIER_TOLERANCE) is what rejects the
// actually-bad readings, so the visibility bar can be much lower here.
// Tried lower (0.05) after live testing showed a reach held in front of the
// torso freezing at a bent-looking position - reverted: the low-confidence
// reading there isn't just unused, it's itself biased short (see the
// reach-toward-camera-in-front-of-face case), so accepting it "live" didn't
// fix the bend either. The real fix has to go in the front-reach
// post-processing, not the accept/freeze threshold.
const MIN_WRIST_VISIBILITY = 0.2
const BONE_OUTLIER_TOLERANCE = 1.3
// The 3D distance elbow<->wrist is the forearm's length, whatever the pose (even a
// forearm pointing at the camera has a depth difference between the two). If the
// elbow point lands on the wrist point, that distance collapses toward 0 - so a
// distance under this fraction of the forearm length means one of the two points
// is a glitch.
const ELBOW_WRIST_MIN_RATIO = 0.4
const ELBOW_WRIST_OVERLAP_RATIO = 0.15
const ELBOW_WRIST_JUMP_MIN = 0.08
let lastGlitchLogAt = 0
const BONE_OUTLIER_MAX_FRAMES = 20

export interface ArmJoints {
  shoulder: Vec3Like
  elbow: Vec3Like
  wrist: Vec3Like
}

interface JointHold {
  current: Vec3Like | null
  rejectStreak?: number
}

// The hold in holdJoint only lasts BONE_OUTLIER_MAX_FRAMES, after which a
// glitch that persists is accepted as real (seen: wrist xy at 3x the bone
// length for ~1s). Even then no segment is allowed on-screen length beyond the
// tolerance: keep the direction, cap the length. Leaves plausible points alone.
function clampSegmentXY(parent: Vec3Like, child: Vec3Like, maxLength: number): Vec3Like {
  if (maxLength < 1e-4) return child
  const dx = child.x - parent.x
  const dy = child.y - parent.y
  const length = Math.hypot(dx, dy)
  if (length <= maxLength || length < 1e-9) return child
  const k = maxLength / length
  return { x: parent.x + dx * k, y: parent.y + dy * k, z: child.z }
}

function holdJoint(poseWorldLm: Landmark[] | undefined, idx: number, held: JointHold, corroborated = false, plausible = true, minVisibility = MIN_ARM_VISIBILITY): { point: Vec3Like | null; live: boolean } {
  const lm = poseWorldLm?.[idx]
  // A reading implying a bone longer than the person's own bone (see
  // BONE_OUTLIER_TOLERANCE) is a tracking glitch, not real motion — hold the
  // last good point instead, but not forever: after enough consecutive
  // rejections assume the glitch is actually where the joint is now.
  let accepted = plausible
  if (!plausible) {
    held.rejectStreak = (held.rejectStreak ?? 0) + 1
    if (held.rejectStreak >= BONE_OUTLIER_MAX_FRAMES) accepted = true
  } else {
    held.rejectStreak = 0
  }
  const live = accepted && !!lm && (lm.visibility >= minVisibility || corroborated) && isFiniteVec3(lm)
  if (live && lm) held.current = lm
  return { point: held.current, live }
}

interface PersonArmLengths {
  upper: number
  lower: number
}
// No avatar here to seed these from (see main.ts's calibrateToAvatar) — used
// rough adult anthropometric averages as the starting point instead. See the
// ambient self-calibration block below (ported from main.ts) for how these
// converge toward this specific person's own measurements from there.
const personArmLengths: { left: PersonArmLengths; right: PersonArmLengths } = {
  left: { upper: 0.28, lower: 0.25 },
  right: { upper: 0.28, lower: 0.25 }
}
let personShoulderWidth = 0.38 // meters, rough adult average

// ---------- Ambient self-calibration — ported from main.ts, same reasoning
// (see the "Ambient self-calibration" comment there). Kept as its own copy
// rather than a shared module, same as the rest of this file. ----------
interface AmbientSample {
  value: number
  at: number
}
interface AmbientEstimator {
  samples: AmbientSample[]
  target: number | null
  lastRecomputeAt: number
}
function newAmbientEstimator(): AmbientEstimator {
  return { samples: [], target: null, lastRecomputeAt: 0 }
}

const AMBIENT_WINDOW_MS = 8000
const AMBIENT_RECOMPUTE_INTERVAL_MS = 1000
const AMBIENT_PERCENTILE = 90
const AMBIENT_MIN_SAMPLES = 30

function ambientSample(estimator: AmbientEstimator, value: number, now: number): void {
  estimator.samples.push({ value, at: now })
  const cutoff = now - AMBIENT_WINDOW_MS
  while (estimator.samples.length > 0 && estimator.samples[0].at < cutoff) estimator.samples.shift()

  if (now - estimator.lastRecomputeAt < AMBIENT_RECOMPUTE_INTERVAL_MS) return
  estimator.lastRecomputeAt = now
  if (estimator.samples.length < AMBIENT_MIN_SAMPLES) return
  const sorted = estimator.samples.map((s) => s.value).sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((AMBIENT_PERCENTILE / 100) * (sorted.length - 1))))
  estimator.target = sorted[idx]
}

const AMBIENT_LERP_PER_SECOND = 0.15
function ambientLerp(current: number, target: number, dt: number): number {
  const amount = Math.min(1, AMBIENT_LERP_PER_SECOND * dt)
  return current + (target - current) * amount
}

const shoulderWidthEstimator = newAmbientEstimator()
const armLengthEstimators: {
  left: { upper: AmbientEstimator; lower: AmbientEstimator }
  right: { upper: AmbientEstimator; lower: AmbientEstimator }
} = {
  left: { upper: newAmbientEstimator(), lower: newAmbientEstimator() },
  right: { upper: newAmbientEstimator(), lower: newAmbientEstimator() }
}

const ARM_UPPER_TO_SHOULDER_WIDTH_RATIO = 0.75
const ARM_LOWER_TO_SHOULDER_WIDTH_RATIO = 0.65
const ARM_LENGTH_MAX_OVER_RATIO = 1.2

function currentArmLengthTarget(shoulderWidth: number, ratio: number, direct: AmbientEstimator): number | null {
  if (shoulderWidth < 1e-4) return direct.target
  const ratioBased = ratio * shoulderWidth
  // The direct estimate only ever sees on-screen (xy) length, which is at most
  // the true length and much shorter whenever the arm points toward the camera,
  // so it can only be trusted to raise the estimate above the ratio-based one,
  // never to pull it below.
  // Also capped: image-space lengths of a limb reaching toward the camera are
  // magnified by perspective (seen: samples of 0.40 for a ~0.25 bone), and an
  // uncapped estimate then feeds itself — a bigger L means bigger
  // reconstructed depth.
  if (direct.target !== null) return Math.min(Math.max(direct.target, ratioBased), ratioBased * ARM_LENGTH_MAX_OVER_RATIO)
  return ratioBased
}

// See main.ts's matching OneEuroFilter for the full reasoning. Used here for
// per-hand-landmark smoothing (see smoothHandLandmarks) and the torso/shoulder
// depth filter below.
export class OneEuroFilter {
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

  // What filter() would compute as this channel's own smoothed speed
  // estimate (|dxHat|) if called right now, without actually committing any
  // state change — see Vector3OneEuroFilter.peekSpeed for why this needs to
  // be readable separately from filter() itself.
  peekSpeed(x: number, dt: number): number {
    if (!this.hasPrev || dt <= 0) return 0
    const dx = (x - this.xPrev) / dt
    const dAlpha = this.alpha(this.dCutoff, dt)
    return Math.abs(this.dxPrev + dAlpha * (dx - this.dxPrev))
  }

  // `speedOverride`, when given, replaces this channel's own |dxHat| for
  // picking the cutoff — see Vector3OneEuroFilter.filter for why a caller
  // would want that.
  filter(x: number, dt: number, speedOverride?: number): number {
    if (!this.hasPrev || dt <= 0) {
      this.hasPrev = true
      this.xPrev = x
      this.dxPrev = 0
      return x
    }
    const dx = (x - this.xPrev) / dt
    const dAlpha = this.alpha(this.dCutoff, dt)
    const dxHat = this.dxPrev + dAlpha * (dx - this.dxPrev)
    const speed = speedOverride ?? Math.abs(dxHat)
    const cutoff = this.minCutoff + this.beta * speed
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

const DEPTH_FILTER_MIN_CUTOFF = 0.5
const DEPTH_FILTER_BETA = 0.6
const SHOULDER_DEPTH_MARGIN = 0.85
const SIGN_CONFIDENCE_THRESHOLD = 0.03

const shoulderDepthFilter = new OneEuroFilter(DEPTH_FILTER_MIN_CUTOFF, DEPTH_FILTER_BETA)
function computeShoulderDepth(rightShoulder: Vec3Like, leftShoulder: Vec3Like, realWidth: number, dt: number): number {
  const rawDz = leftShoulder.z - rightShoulder.z
  if (realWidth < 1e-4) return rawDz

  const dx = leftShoulder.x - rightShoulder.x
  const dy = leftShoulder.y - rightShoulder.y
  const xyLength = Math.hypot(dx, dy)
  // Same cliff as an arm segment (xy of the shoulder line is ~= the real
  // width whenever the torso faces the camera), but here it shows up as the
  // whole shoulder reference line slowly rotating while the person sits still:
  // a ±10% wobble in the image→world scale is ±25° of apparent torso yaw.
  // SHOULDER_DEPTH_MARGIN turns that into a dead zone (xy above 85% of the
  // width reads as no rotation at all).
  const effectiveWidth = realWidth * SHOULDER_DEPTH_MARGIN
  const rawReconstructedDepth = Math.sqrt(Math.max(effectiveWidth * effectiveWidth - xyLength * xyLength, 0))
  const reconstructedDepth = shoulderDepthFilter.filter(rawReconstructedDepth, dt)

  const sign = Math.abs(rawDz) >= SIGN_CONFIDENCE_THRESHOLD ? Math.sign(rawDz) : 0
  return sign * reconstructedDepth
}

function jointAngleDeg(a: Vec3Like, b: Vec3Like, c: Vec3Like): string {
  const v1 = [a.x - b.x, a.y - b.y, a.z - b.z]
  const v2 = [c.x - b.x, c.y - b.y, c.z - b.z]
  const n1 = Math.hypot(v1[0], v1[1], v1[2])
  const n2 = Math.hypot(v2[0], v2[1], v2[2])
  if (n1 < 1e-6 || n2 < 1e-6) return 'n/a'
  const cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
  return Math.round((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI) + '°'
}

function clampNum(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

// See POLE_MAX_ANGULAR_SPEED_DEG_PER_SEC - rotates a unit vector toward
// another by at most maxAngleRad, instead of by a fixed fraction of the
// remaining angle (the difference matters most for a large remaining
// angle: a fraction lets a big jump still move a lot in one frame, while a
// fixed max-per-frame angle bounds it the same regardless of how big the
// jump is).
function rotateTowardUnit(from: Vec3Like, to: Vec3Like, maxAngleRad: number): Vec3Like {
  const dot = clampNum(from.x * to.x + from.y * to.y + from.z * to.z, -1, 1)
  const angle = Math.acos(dot)
  if (angle < 1e-6) return from
  const t = clampNum(maxAngleRad / angle, 0, 1)
  if (t >= 1) return to
  const sinAngle = Math.sin(angle)
  const wa = Math.sin((1 - t) * angle) / sinAngle
  const wb = Math.sin(t * angle) / sinAngle
  return { x: from.x * wa + to.x * wb, y: from.y * wa + to.y * wb, z: from.z * wa + to.z * wb }
}

// ---------- Two-bone IK ----------
// The elbow's depth is never reconstructed from the picture: the raw wrist world
// landmark (direction + distance from the shoulder) is trusted directly as the IK
// target, and the elbow is placed by the law of cosines on the shoulder-elbow-wrist
// triangle using the person's own T-pose-measured bone lengths - so the elbow can
// never land at an impossible distance from the shoulder, whatever the raw
// landmark's noise. The target distance is clamped to what the two bones can
// actually reach, which is what turns a reach toward the camera into a straight
// arm automatically (no separate "is this a straight reach" detector needed).
//
// The triangle leaves exactly one degree of freedom open: which way the elbow
// points around the shoulder-wrist axis (the same freedom a shoulder's own
// rotation would use). That comes from the raw elbow landmark, but only its
// component perpendicular to the shoulder-wrist axis - never its distance - so a
// wrong guess there can only rotate the elbow around the axis, never move it off
// the sphere of valid elbow positions. Held across frames where the hint is
// missing or degenerate (arm dead straight, or the landmark unavailable).
// Small enough that it only guards the exact degenerate triangle (division by
// ~0 at dist == upper+lower), not visible straightness: at 0.02 a full reach's
// elbow angle capped at ~149° instead of opening up toward 180° (measured live).
const MIN_TRIANGLE_SPAN = 0.002
// See solveTwoBoneIK's use of these, at the pole vector: below this fraction
// of the upper arm's own length, the raw elbow landmark's sideways offset is
// too small to trust as a bend direction (measured live: right as the arm
// straightens, or whenever visibility dropped, this offset shrank toward 0
// and a tiny noisy value normalized into a direction pointing anywhere -
// "the elbow popping outward"). Above it, the held direction still only
// blends toward the new one at this rate per second, not an instant snap.
const POLE_MIN_PERP_RATIO = 0.15
// The cutoff above was a hard on/off gate: anything above it was blended in
// at full trust, no matter how close to the cutoff (and therefore how
// noise-dominated) it still was. Measured live: as the arm approaches full
// reach, the perpendicular offset shrinks continuously toward that cutoff -
// right in that approach, a still-mostly-noise reading just above the gate
// got treated identically to a clean, well-above-threshold one, and its
// chaotic direction (small numerator noise / small pLen denominator) swung
// the elbow - and therefore the whole upper arm's rendered direction - out
// to whichever way the noise happened to point ("상완이 쏠림"). Above this
// higher ratio the hint is trusted fully same as before; between the two,
// trust (and therefore how fast the pole blends toward it) ramps up
// linearly instead of snapping to full trust right at the gate.
const POLE_TRUST_FULL_RATIO = 0.4
// Replaces the old POLE_BLEND_PER_SECOND fractional blend (new = old + (target
// - old) * fraction). That fraction closes the SAME % of the gap every frame
// regardless of how big the gap is, so a large single-frame misread (a
// re-estimate, not noise - see chat) still moved the pole a large ABSOLUTE
// amount in that one frame, just less than snapping outright. A max angular
// speed instead bounds the ABSOLUTE rotation per frame no matter how big the
// jump behind it is: a genuine, gradual bend stays well under this and passes
// through basically unrestricted, while a one-frame bad reading only nudges
// the pole a couple of degrees before (if it was noise) the next frame pulls
// back - instead of both cases getting the same-percentage, visibly large jump
// they got before. A real, sustained multi-second bias still eventually gets
// there, just gradually instead of snapping.
const POLE_MAX_ANGULAR_SPEED_DEG_PER_SEC = 180
// MediaPipe's z (depth from the camera) is the least reliable of the three
// axes - it's the same unreliability the old pinhole depth pipeline was
// scrapped over. The pole vector still reads it directly off the raw elbow
// landmark, and when the bend is near 90° (sin(theta)~1, an arm out to the
// side with the forearm folded up) that raw z rides almost unscaled into the
// elbow's rendered position - measured live: xyUp alone already covered 96%
// of the upper arm's true length, meaning the landmark's own z offset was
// small, yet the pole came out z=0.97 and pushed the whole upper arm forward.
// Damping z before the perpendicular vector is even normalized biases the
// bend direction back toward the x/y plane, which the model actually tracks
// well, instead of trusting its weakest axis at full strength.
const POLE_Z_DAMPING = 0.35
// See solveTwoBoneIK's use of this: the reach gain only kicks in above this
// fraction of the arm's own max span, so an ordinary bent or resting arm is
// never stretched - only a reading already close to a genuine full reach is.
const REACH_GAIN_KNEE = 0.75
// The blend below reaches full gain at this fraction, not at 1.0 - measured
// live: a real forward reach's rawDist only climbed to ~80-85% of maxSpan
// even at good confidence, so requiring literal 100% before the gain ever
// reached full strength meant the correction was weakest exactly where it
// was needed most. Lower = more aggressive (narrows the room between KNEE
// and here), which also raises the risk of stretching an arm that was only
// ever held at ~80% on purpose (not reaching) into looking fully extended.
const REACH_GAIN_FULL_AT = 0.85
// The model's raw z visibly lags for a few frames whenever the arm actually
// moves, even while it stays just as reached (measured live: sweeping a
// fully reached arm sideways bent it every time, because the raw distance
// briefly dipped as the landmark caught up). A held reach distance releases
// toward the live one at this rate (fraction of the arm's own length per
// second) instead of following it instantly, so a dip shorter than the
// release takes to drain never reaches the elbow. A real, sustained drop
// (actually relaxing the arm) still gets there within a fraction of a
// second - this is a bridge over brief lag, not a long-term smoother.
const REACH_HOLD_RELEASE_PER_SECOND = 2.5
// Moving a reached arm in front of the face/torso isn't a brief lag, it's a
// sustained several-second stretch of low wrist visibility where the raw
// reading itself sits biased short the whole time (measured live: HandLandmarker
// loses the hand entirely there too, so there's no better signal to switch
// to - see the chat). The release rate above is scaled down by how low the
// current wrist visibility is, so the hold survives that whole stretch
// instead of draining within its first fraction of a second; it never fully
// stops, so a genuine multi-second relax still eventually shows.
// Cubed, not squared (measured live again): confidence doesn't drop straight
// to near-zero, it drains from ~0.9 to ~0.15 over roughly a second, and even
// a squared curve still gave the 0.3-0.6 band on the way down 20-40% speed -
// enough to empty the hold before the pass ever reached the truly low part.
// Cubing (plus a lower floor and a higher "confident" cutoff) keeps anything
// under ~0.7 down near the floor, so the hold survives the whole gradual
// decline and only lets go once confidence is genuinely settled back up.
const RELEASE_CONFIDENCE_LOW = 0.2
const RELEASE_CONFIDENCE_HIGH = 0.9
const RELEASE_MIN_FACTOR = 0.02
// The release above only guards the downward side - an upward jump in
// `stretched` was applied to heldReachDist instantly, no matter how low
// confidence was. Measured live: a single 80ms frame where rawDist spiked
// 0.108 -> 0.350 while wristVis sat at ~0.4 (a landmark glitch, confirmed by
// a [glitch] collapse log firing right around the same frames) popped the arm
// immediately. Scaling the attack by the same confidence curve fixes that
// without slowing a genuine reach: this rate is high enough that at good
// confidence (factor ~1) it still closes any realistic single-frame reach
// delta in one frame, and only throttles when confidence says the reading
// is untrustworthy.
const ATTACK_HOLD_RATE_PER_SECOND = 8
// A hand pointed straight at the camera foreshortens the whole arm behind it -
// the classic "reaching for the lens" look. When that happens while confidence
// is already low, the raw wrist reading is exactly the "biased short" case
// this file has fought all session, so instead of accepting it, treat it as
// a forward reach and push the distance toward the physical max. Below this
// wrist confidence only - a well-tracked short arm (resting, gesturing) must
// never be reinterpreted as a reach.
const FORWARD_REACH_CONFIDENCE_MAX = 0.5
// forceFullReach used to push all the way to maxSpan (100%) - dialed back
// slightly so a forced reach doesn't look quite as rigidly, artificially
// straight; still overrides the raw (biased-short) reading by a lot.
const FORCE_FULL_REACH_FRACTION = 0.92
// The whole arm (shoulder-to-wrist) has to read as no bigger than this many
// hand-lengths (wrist-to-middle-knuckle) to count as "swallowed" by the hand -
// a real human arm is 4-6x a hand's length, so anything under ~1.3x is only
// plausible as foreshortening, not a genuinely short/bent arm.
const FORWARD_REACH_HAND_MARGIN = 1.3
// `enclosed` alone (whether via the hand-size check or the hand-independent
// elbow/wrist collapse fallback) can't tell a genuine reach toward the camera
// apart from ordinary tracking noise elsewhere on the body - measured live
// (see chat): raising the forearm with the upper arm pinned to the side made
// the collapse fallback fire from pure noise, snapping the arm to full
// extension even though the raw shoulder->wrist direction wasn't pointing at
// the camera at all (z ratio +0.42 and +0.03 - see towardCamera below, not
// negative). A genuine "hand at the lens" reach has the wrist sitting
// clearly closer to the camera than the shoulder (MediaPipe's z: smaller/more
// negative is closer), so requiring that too rejects a same-side noise spike
// while still letting an actual forward reach through.
const FORWARD_REACH_MIN_Z_RATIO = 0.3
// A held hand reading is only trusted this long after it was last actually
// seen - long enough to bridge a brief drop, not so long that a hand that
// truly left frame keeps forcing a reach based on stale data. Measured live:
// a low-confidence forward reach can lose the hand model for several
// consecutive seconds (no brief recoveries at all, unlike normal tracking
// flicker, which recovers within a couple hundred ms on its own) - 400ms
// wasn't close to bridging that, so ikWrist fell back to the pose model's
// own (biased) wrist estimate for the whole reach instead of the hand's.
const HAND_SEEN_STALE_MS = 2000
// See the ikWrist blend in resolveArm - full pose trust above this
// confidence, full hand-model trust below it, linear in between.
const WRIST_SOURCE_CONFIDENCE_LOW = 0.3
const WRIST_SOURCE_CONFIDENCE_HIGH = 0.7

function solveTwoBoneIK(
  shoulder: Vec3Like,
  target: Vec3Like,
  poleHint: Vec3Like | null,
  upper: number,
  lower: number,
  heldPole: { current: Vec3Like | null },
  // See the reach-calibration block below: raw distance under-reports a
  // genuine full reach by a measured 5-10%, and the law of cosines is
  // sharply sensitive to exactly that last stretch (95-98% of the true
  // reach distance still only opened the elbow to ~140-157°, not the
  // ~170°+ a fully straight arm needs to look right). Applied to the
  // magnitude only - the direction below still comes from the raw vector.
  // Blended in only above REACH_GAIN_KNEE (see there) - applied flat, it
  // also stretched every ordinary, not-reaching pose forward (measured
  // live: the resting arm couldn't sit close to the body any more).
  gain: number,
  dt: number,
  heldReachDist: { value: number },
  // Current wrist visibility (0-1) - see RELEASE_CONFIDENCE_LOW/HIGH above.
  wristConfidence: number,
  // See FORWARD_REACH_CONFIDENCE_MAX/FORWARD_REACH_HAND_MARGIN above - set by
  // the caller when the arm looks swallowed by the hand while confidence is
  // low. This is a structural inference, not a single frame's raw reading,
  // so it drives the attack at full rate instead of the confidence-scaled one.
  forceFullReach: boolean
): { elbow: Vec3Like; wrist: Vec3Like; rawDist: number } {
  const tx = target.x - shoulder.x
  const ty = target.y - shoulder.y
  const tz = target.z - shoulder.z
  const rawDist = Math.hypot(tx, ty, tz)
  if (rawDist < 1e-4 || upper < 1e-4 || lower < 1e-4) {
    return { elbow: { ...shoulder }, wrist: { ...shoulder }, rawDist }
  }
  const ux = tx / rawDist
  const uy = ty / rawDist
  const uz = tz / rawDist
  const minSpan = Math.max(Math.abs(upper - lower), MIN_TRIANGLE_SPAN)
  const maxSpan = Math.max(minSpan, upper + lower - MIN_TRIANGLE_SPAN)
  // Below the knee (an ordinary bent arm, a gesture, resting at the side) the
  // gain is fully off - only a reading already this close to the theoretical
  // max gets stretched the rest of the way, ramping the gain in linearly so
  // there's no snap right at the boundary.
  const reachFraction = clampNum(rawDist / maxSpan, 0, 1)
  const gainBlend =
    reachFraction <= REACH_GAIN_KNEE
      ? 0
      : clampNum((reachFraction - REACH_GAIN_KNEE) / (REACH_GAIN_FULL_AT - REACH_GAIN_KNEE), 0, 1)
  const effectiveGain = 1 + (gain - 1) * gainBlend
  const stretched = forceFullReach ? maxSpan * FORCE_FULL_REACH_FRACTION : rawDist * effectiveGain
  const confidenceT = clampNum((wristConfidence - RELEASE_CONFIDENCE_LOW) / (RELEASE_CONFIDENCE_HIGH - RELEASE_CONFIDENCE_LOW), 0, 1)
  const confidenceFactor = RELEASE_MIN_FACTOR + (1 - RELEASE_MIN_FACTOR) * confidenceT * confidenceT * confidenceT
  const release = (upper + lower) * REACH_HOLD_RELEASE_PER_SECOND * confidenceFactor * Math.max(0, dt)
  const attackRate = forceFullReach ? ATTACK_HOLD_RATE_PER_SECOND : ATTACK_HOLD_RATE_PER_SECOND * confidenceFactor
  const attack = (upper + lower) * attackRate * Math.max(0, dt)
  // Capped at maxSpan here, not just when reading it into `dist` below: letting
  // the stored value itself overshoot past the arm's own physical ceiling
  // (measured live: gain + a raw reading already near maxSpan pushed it to
  // 0.51 against a 0.466 maxSpan) meant it had to decay through that whole
  // overshoot before `dist` ever moved off the ceiling - the arm stayed
  // rendered fully straight for many frames after the person had already
  // started pulling back, looking like the forearm was stuck too far forward.
  const nextHeld =
    stretched >= heldReachDist.value
      ? Math.min(stretched, heldReachDist.value + attack)
      : Math.max(stretched, heldReachDist.value - release)
  heldReachDist.value = Math.min(nextHeld, maxSpan)
  const dist = clampNum(heldReachDist.value, minSpan, maxSpan)
  const cosShoulder = clampNum((upper * upper + dist * dist - lower * lower) / (2 * upper * dist), -1, 1)
  const theta = Math.acos(cosShoulder)

  let pole = heldPole.current ?? { x: 0, y: -1, z: 0 }
  if (poleHint) {
    const px0 = poleHint.x - shoulder.x
    const py0 = poleHint.y - shoulder.y
    const pz0 = poleHint.z - shoulder.z
    const along = px0 * ux + py0 * uy + pz0 * uz
    const px = px0 - along * ux
    const py = py0 - along * uy
    const pz = (pz0 - along * uz) * POLE_Z_DAMPING
    const pLen = Math.hypot(px, py, pz)
    // The raw elbow landmark's perpendicular offset shrinks toward 0 as the arm
    // straightens (the elbow sits almost exactly on the shoulder-target axis) -
    // measured live: right at that point, and whenever the landmark itself was
    // noisy (low visibility), a nearly-zero-length offset got normalized into a
    // full-length direction that pointed anywhere, snapping the elbow sideways
    // frame to frame ("popping out"). Below this fraction of the upper arm's own
    // length, the hint is too small to trust - keep the held direction. Above
    // it, rotate toward the new one at up to POLE_MAX_ANGULAR_SPEED_DEG_PER_SEC
    // instead of snapping straight to it, so an accepted but still-somewhat-noisy
    // hint doesn't itself pop the elbow in one frame.
    // That speed used to apply no matter how low wristConfidence was -
    // measured live: at a middling ~0.65-0.70 (not low enough to hold, not
    // high enough to trust), pole.y drifted steadily from 0.87 toward 0.02
    // over about a second, reading as the elbow sagging downward. Scaling
    // it by the same confidence curve as the reach distance's attack/release
    // slows exactly that middling-confidence drift, while still rotating
    // quickly at good confidence.
    if (pLen > upper * POLE_MIN_PERP_RATIO) {
      const newPole = { x: px / pLen, y: py / pLen, z: pz / pLen }
      const perpRatio = pLen / upper
      const perpTrust = clampNum(
        (perpRatio - POLE_MIN_PERP_RATIO) / (POLE_TRUST_FULL_RATIO - POLE_MIN_PERP_RATIO),
        0,
        1
      )
      const maxAngleRad =
        ((POLE_MAX_ANGULAR_SPEED_DEG_PER_SEC * Math.PI) / 180) * confidenceFactor * perpTrust * Math.max(0, dt)
      pole = rotateTowardUnit(pole, newPole, maxAngleRad)
      heldPole.current = pole
    }
  }

  const elbow: Vec3Like = {
    x: shoulder.x + upper * (Math.cos(theta) * ux + Math.sin(theta) * pole.x),
    y: shoulder.y + upper * (Math.cos(theta) * uy + Math.sin(theta) * pole.y),
    z: shoulder.z + upper * (Math.cos(theta) * uz + Math.sin(theta) * pole.z)
  }
  const wrist: Vec3Like = {
    x: shoulder.x + dist * ux,
    y: shoulder.y + dist * uy,
    z: shoulder.z + dist * uz
  }
  return { elbow, wrist, rawDist }
}

function resolveArm(
  poseWorldLm: Landmark[] | undefined,
  shoulderIdx: number,
  elbowIdx: number,
  wristIdx: number,
  estimate: PersonArmLengths,
  handLandmarks: Vec3Like[] | undefined,
  handWorldLandmarks: Vec3Like[] | undefined,
  heldShoulder: JointHold,
  heldElbow: JointHold,
  heldWrist: JointHold,
  heldPole: { current: Vec3Like | null },
  reachGain: number,
  dt: number,
  heldReachDist: { value: number },
  heldHandSeen: HandSeen,
  heldHandWristTrust: HandWristTrust
): {
  arm: ArmJoints | null
  confident: boolean
  upperXY: number
  lowerXY: number
  wholeXY: number
  wristDy: number
  rawReachDist: number
  forceFullReach: boolean
} {
  const handTracked = !!handLandmarks
  const shoulder = holdJoint(poseWorldLm, shoulderIdx, heldShoulder)
  const rawElbow = poseWorldLm?.[elbowIdx]
  const rawWrist = poseWorldLm?.[wristIdx]
  let elbowCollapsed = false
  let wristCollapsed = false
  if (rawElbow && rawWrist && estimate.lower > 1e-4) {
    const forearm3D = Math.hypot(rawWrist.x - rawElbow.x, rawWrist.y - rawElbow.y, rawWrist.z - rawElbow.z)
    // Blame whichever of the two jumped further from where it was last frame.
    const prevElbow = heldElbow.current
    const prevWrist = heldWrist.current
    const elbowMoved = prevElbow ? Math.hypot(rawElbow.x - prevElbow.x, rawElbow.y - prevElbow.y, rawElbow.z - prevElbow.z) : 0
    const wristMoved = prevWrist ? Math.hypot(rawWrist.x - prevWrist.x, rawWrist.y - prevWrist.y, rawWrist.z - prevWrist.z) : 0
    const jump = Math.max(elbowMoved, wristMoved)
    // A short elbow<->wrist distance alone is NOT enough: the model's 3D bone
    // lengths swing widely with pose, and a forearm folded up against the upper arm
    // legitimately measures ~0.09 against a 0.24 bone (a 0.4 cutoff froze exactly
    // that motion). It has to be a sudden jump, or a near-total overlap.
    const collapsed =
      forearm3D < ELBOW_WRIST_OVERLAP_RATIO * estimate.lower ||
      (forearm3D < ELBOW_WRIST_MIN_RATIO * estimate.lower && jump >= ELBOW_WRIST_JUMP_MIN)
    if (collapsed) {
      if (elbowMoved >= wristMoved) elbowCollapsed = true
      else wristCollapsed = true
      const nowMs = performance.now()
      if (nowMs - lastGlitchLogAt > 1000) {
        lastGlitchLogAt = nowMs
        debugLog(`[glitch] elbow-wrist collapsed forearm3D=${forearm3D.toFixed(3)} (L=${estimate.lower.toFixed(3)}) blamed=${elbowCollapsed ? 'elbow' : 'wrist'}`)
      }
    }
  }
  const elbowPlausible =
    !elbowCollapsed &&
    (!rawElbow || !shoulder.point || estimate.upper < 1e-4 ||
      Math.hypot(rawElbow.x - shoulder.point.x, rawElbow.y - shoulder.point.y) <= estimate.upper * BONE_OUTLIER_TOLERANCE)
  const elbow = holdJoint(poseWorldLm, elbowIdx, heldElbow, handTracked, elbowPlausible)
  if (elbow.point && shoulder.point) elbow.point = clampSegmentXY(shoulder.point, elbow.point, estimate.upper * BONE_OUTLIER_TOLERANCE)
  const wristPlausible =
    !wristCollapsed &&
    (!rawWrist || !elbow.point || estimate.lower < 1e-4 ||
      Math.hypot(rawWrist.x - elbow.point.x, rawWrist.y - elbow.point.y) <= estimate.lower * BONE_OUTLIER_TOLERANCE)
  const wrist = holdJoint(poseWorldLm, wristIdx, heldWrist, handTracked, wristPlausible, MIN_WRIST_VISIBILITY)
  if (wrist.point && elbow.point) wrist.point = clampSegmentXY(elbow.point, wrist.point, estimate.lower * BONE_OUTLIER_TOLERANCE)

  if (!shoulder.point || !elbow.point || !wrist.point) {
    return { arm: null, confident: false, upperXY: 0, lowerXY: 0, wholeXY: 0, wristDy: 0, rawReachDist: 0, forceFullReach: false }
  }

  // Measured on the raw (held) landmarks, before the IK solve - these feed the
  // T-pose detector, the ambient bone-length estimate and the length ratchet,
  // all of which want the picture's own on-screen bone length regardless of
  // how the final joint positions get computed.
  const upperXY = Math.hypot(elbow.point.x - shoulder.point.x, elbow.point.y - shoulder.point.y)
  const lowerXY = Math.hypot(wrist.point.x - elbow.point.x, wrist.point.y - elbow.point.y)
  const wholeXY = Math.hypot(wrist.point.x - shoulder.point.x, wrist.point.y - shoulder.point.y)
  const wristDy = wrist.point.y - shoulder.point.y

  const wristConfidence = rawWrist?.visibility ?? 1
  const nowMs = performance.now()
  // Held in the hand model's own WORLD (metric) coordinates - see
  // HAND_SEEN_STALE_MS. Deliberately the world landmarks, not the normalized
  // image ones handTracked/wristGap use below: this feeds handSize and the
  // wrist-follow blend, both compared directly against world-space arm
  // lengths, and mixing normalized image units into that comparison was a
  // real bug caught while re-checking this (handSize came out two orders of
  // magnitude smaller than wholeXYZ, so enclosedByHand could never fire).
  if (handWorldLandmarks) {
    heldHandSeen.landmarks = handWorldLandmarks
    heldHandSeen.at = nowMs
  }
  const handRecent = heldHandSeen.landmarks && nowMs - heldHandSeen.at < HAND_SEEN_STALE_MS ? heldHandSeen.landmarks : null
  if (!handRecent) {
    // Stale for long enough that anything could have happened in the gap -
    // the next reading has to re-earn trust from scratch (see
    // resolveTrustedHandWrist), not slot straight in as if tracking had
    // continued uninterrupted.
    heldHandWristTrust.confirmed = null
    heldHandWristTrust.pendingCandidate = null
    heldHandWristTrust.pendingCount = 0
  }
  // Wrist-to-middle-knuckle, in the hand model's own coordinates - a stable
  // "how big does the hand look" reference regardless of finger curl.
  const handSize = handRecent
    ? Math.hypot(handRecent[9].x - handRecent[0].x, handRecent[9].y - handRecent[0].y, handRecent[9].z - handRecent[0].z)
    : null
  const wholeXYZ = Math.hypot(wrist.point.x - shoulder.point.x, wrist.point.y - shoulder.point.y, wrist.point.z - shoulder.point.z)
  const enclosedByHand = handSize != null && handSize > 1e-4 && wholeXYZ <= handSize * FORWARD_REACH_HAND_MARGIN
  // No hand data recent enough to size against - fall back to the
  // hand-independent collapse check already computed above.
  const enclosed = handRecent ? enclosedByHand : elbowCollapsed || wristCollapsed
  // See FORWARD_REACH_MIN_Z_RATIO above.
  const targetZRatio = wholeXYZ > 1e-4 ? (wrist.point.z - shoulder.point.z) / wholeXYZ : 0
  const towardCamera = targetZRatio < -FORWARD_REACH_MIN_Z_RATIO
  const forceFullReach = wristConfidence < FORWARD_REACH_CONFIDENCE_MAX && enclosed && towardCamera

  // Measured live (see chat): the pose model's and hand model's own wrist
  // estimates agree to within a few cm even at decent confidence, with no
  // fixed offset - the same coordinate frame, just each with its own noise.
  // So below WRIST_SOURCE_CONFIDENCE_HIGH, blend toward the hand model's
  // wrist instead of trusting the pose model's alone; below
  // WRIST_SOURCE_CONFIDENCE_LOW, trust it fully. Only feeds the IK target -
  // upperXY/lowerXY/wholeXY above stay on the pose model's own reading, since
  // those calibrate off "the picture's own bone length" and haven't been
  // checked against this substitution.
  let ikWrist = wrist.point
  if (handRecent && handSize != null) {
    const trustedHandWrist = resolveTrustedHandWrist(handRecent[0], handSize, heldHandWristTrust)
    if (trustedHandWrist) {
      const t = clampNum(
        (wristConfidence - WRIST_SOURCE_CONFIDENCE_LOW) / (WRIST_SOURCE_CONFIDENCE_HIGH - WRIST_SOURCE_CONFIDENCE_LOW),
        0,
        1
      )
      const h = trustedHandWrist
      ikWrist = {
        x: h.x + (wrist.point.x - h.x) * t,
        y: h.y + (wrist.point.y - h.y) * t,
        z: h.z + (wrist.point.z - h.z) * t
      }
    }
  }

  const solved = solveTwoBoneIK(
    shoulder.point,
    ikWrist,
    elbow.point,
    estimate.upper,
    estimate.lower,
    heldPole,
    reachGain,
    dt,
    heldReachDist,
    wristConfidence,
    forceFullReach
  )

  return {
    arm: { shoulder: shoulder.point, elbow: solved.elbow, wrist: solved.wrist },
    confident: wrist.live,
    upperXY,
    lowerXY,
    wholeXY,
    wristDy,
    rawReachDist: solved.rawDist,
    forceFullReach
  }
}

const TORSO_HALF_THICKNESS = 0.09
// clampInFrontOfTorso's x-range used to be the full span between the two
// shoulders - but an arm just resting or moving at its OWN side naturally
// sits with its elbow/wrist x near THAT shoulder's own edge (one end of that
// span, not the middle), and its natural z there is legitimately at-or-behind
// the torso (the shoulder joint itself isn't out in front of the chest) - see
// chat: measured live, elbowRelZ stayed positive (behind) the whole time
// during "arm at the side, raise the forearm," and the clamp (which runs
// after that log, so it wasn't visible there) was snapping it forward anyway
// because that resting position's x still fell inside the full shoulder-to-
// shoulder span. Shrinking the band to the inner fraction around the body's
// midline means only a limb that's genuinely crossed toward/past center -
// reaching across the front of one's own torso, the actual clipping case
// this clamp exists for - trips it; a limb still near its own shoulder's
// edge doesn't.
const CROSS_BODY_INSET = 0.5

function clampInFrontOfTorso(point: Vec3Like, torsoMinX: number, torsoMaxX: number, torsoZ: number): Vec3Like {
  if (point.x < torsoMinX || point.x > torsoMaxX) return point
  const maxZ = torsoZ - TORSO_HALF_THICKNESS
  if (point.z <= maxZ) return point
  return { x: point.x, y: point.y, z: maxZ }
}

const heldLeftShoulder: JointHold = { current: null }
const heldLeftElbow: JointHold = { current: null }
const heldLeftWrist: JointHold = { current: null }
const heldRightShoulder: JointHold = { current: null }
const heldRightElbow: JointHold = { current: null }
const heldRightWrist: JointHold = { current: null }
const heldLeftPole: { current: Vec3Like | null } = { current: null }
const heldRightPole: { current: Vec3Like | null } = { current: null }
const heldLeftReachDist: { value: number } = { value: 0 }
const heldRightReachDist: { value: number } = { value: 0 }
// See FORWARD_REACH_CONFIDENCE_MAX/HAND_SEEN_STALE_MS - a hand reading kept
// around briefly after it's last actually seen, so a one-frame drop doesn't
// throw away the only signal that tells the reach-forward heuristic apart
// from a genuinely short/bent arm.
interface HandSeen {
  landmarks: Vec3Like[] | null
  at: number
}
const heldLeftHandSeen: HandSeen = { landmarks: null, at: 0 }
const heldRightHandSeen: HandSeen = { landmarks: null, at: 0 }

// The hand model's own wrist point (landmark 0) is the same physical joint as
// the pose model's wrist - any gap between them is measurement error, not a
// real difference, so it's tempting to just always snap to the hand model's
// reading. Measured live, though: the frame a hand reading first reappears
// after a drop is exactly when it's least trustworthy (one case measured a
// fresh reading 0.5m from the pose wrist, immediately blended in and popping
// the arm - see chat). So a fresh hand-wrist reading has to first agree with
// itself across a couple of frames before it's trusted at all; a single
// reading that jumps too far from the last CONFIRMED one is held as a
// "pending" candidate instead of used immediately, and only promoted to
// confirmed once a following frame lands close to it too. The tolerance
// scales with the hand's own measured size (wrist-to-middle-knuckle), not a
// fixed distance, so it stays meaningful whether the hand is close to the
// camera (large) or far (small).
const HAND_WRIST_JUMP_TOLERANCE = 0.6
const HAND_WRIST_CORROBORATION_FRAMES = 2
interface HandWristTrust {
  confirmed: Vec3Like | null
  pendingCandidate: Vec3Like | null
  pendingCount: number
}
const heldLeftHandWristTrust: HandWristTrust = { confirmed: null, pendingCandidate: null, pendingCount: 0 }
const heldRightHandWristTrust: HandWristTrust = { confirmed: null, pendingCandidate: null, pendingCount: 0 }

// Runs the corroboration gate described above and returns the wrist position
// to actually trust this frame - the newly-confirmed reading, the last
// already-confirmed one, or null if nothing's been confirmed yet.
function resolveTrustedHandWrist(raw: Vec3Like, handSize: number, trust: HandWristTrust): Vec3Like | null {
  if (handSize <= 1e-4) return trust.confirmed
  const tolerance = handSize * HAND_WRIST_JUMP_TOLERANCE
  const dist = (a: Vec3Like, b: Vec3Like): number => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
  if (trust.confirmed && dist(raw, trust.confirmed) <= tolerance) {
    trust.confirmed = raw
    trust.pendingCandidate = null
    trust.pendingCount = 0
    return trust.confirmed
  }
  // Either there's no confirmed baseline yet, or this reading jumped too far
  // from it - either way, it needs to agree with a pending candidate first.
  if (trust.pendingCandidate && dist(raw, trust.pendingCandidate) <= tolerance) {
    trust.pendingCount++
    trust.pendingCandidate = raw
    if (trust.pendingCount >= HAND_WRIST_CORROBORATION_FRAMES) {
      trust.confirmed = raw
      trust.pendingCandidate = null
      trust.pendingCount = 0
    }
  } else {
    trust.pendingCandidate = raw
    trust.pendingCount = 1
  }
  return trust.confirmed
}

// Depth bookkeeping the avatar app shows in its readout.
export interface ArmDiagnostics {
  upperDepth: number
  lowerDepth: number
  // Raw (pre-correction) displacement of each segment, straight from MediaPipe's world landmarks.
  rawUpperDelta: Vec3Like
  rawLowerDelta: Vec3Like
}

export interface ResolvedFrame {
  left: ArmJoints | null // person-left arm
  right: ArmJoints | null // person-right arm
  leftConfident: boolean // was THIS frame's left wrist trustworthy, or held?
  rightConfident: boolean
  leftHand: Vec3Like[] | null // person-left hand, held across low-confidence frames
  rightHand: Vec3Like[] | null
  leftDiagnostics: ArmDiagnostics
  rightDiagnostics: ArmDiagnostics
}

// ---------- T-pose auto-calibration ----------
// Arm length has to be known to within a few percent: it is enforced exactly by
// the IK solve, so an error in it shows up directly as a wrong elbow bend. The
// ambient/ratio estimates above can't get that close (they depend on the
// image→world scale, the shoulder-width estimate and how the person's
// proportions differ from the ratio). But when an arm is held straight out
// sideways, its on-screen (xy) segment lengths ARE its real bone lengths,
// measured in exactly the units the rest of the pipeline uses — so that moment
// is measured directly and overrides the estimates. Detected automatically,
// no button.
const TPOSE_MIN_STRAIGHTNESS = 0.95 // shoulder→wrist xy / (upper xy + lower xy)
const TPOSE_MAX_DROOP_RATIO = 0.25 // |wrist.y - shoulder.y| / shoulder→wrist xy
const TPOSE_HOLD_MS = 2000
const TPOSE_MIN_SAMPLES = 20
const tPoseCalibrated = { left: false, right: false }
interface TPoseHold {
  startedAt: number | null
  upper: number[]
  lower: number[]
}
const tPoseHold: { left: TPoseHold; right: TPoseHold } = {
  left: { startedAt: null, upper: [], lower: [] },
  right: { startedAt: null, upper: [], lower: [] }
}

// Once any arm is T-pose measured, its lengths are in units set by the current
// personShoulderWidth (the image→world scale is pw / image shoulder width), so
// pw must stop drifting from then on or the stored L and live xy would fall
// out of step. Persisted together so a reload doesn't throw the measurement away.
let shoulderWidthFrozen = false
const TPOSE_STORAGE_KEY = 'casual-hand-tracker:viewer-tpose:v4'
interface StoredTPose {
  shoulderWidth: number
  left: { upper: number; lower: number } | null
  right: { upper: number; lower: number } | null
  reachGain?: { left: number; right: number }
}

function saveTPoseProfile(): void {
  try {
    const profile: StoredTPose = {
      shoulderWidth: personShoulderWidth,
      left: tPoseCalibrated.left ? { ...personArmLengths.left } : null,
      right: tPoseCalibrated.right ? { ...personArmLengths.right } : null,
      reachGain: { ...reachGain }
    }
    localStorage.setItem(TPOSE_STORAGE_KEY, JSON.stringify(profile))
  } catch {
    // storage unavailable — measurement just won't survive a reload
  }
}

function loadTPoseProfile(): void {
  try {
    const raw = localStorage.getItem(TPOSE_STORAGE_KEY)
    if (!raw) return
    const p = JSON.parse(raw) as StoredTPose
    const ok = (v: unknown, lo: number, hi: number): v is number => typeof v === 'number' && Number.isFinite(v) && v > lo && v < hi
    if (!ok(p.shoulderWidth, 0.2, 0.6)) return
    personShoulderWidth = p.shoulderWidth
    shoulderWidthFrozen = true
    debugLog(`[profile] loaded: pw=${p.shoulderWidth}`)
    for (const side of ['left', 'right'] as const) {
      const arm = p[side]
      if (arm && ok(arm.upper, 0.1, 0.6) && ok(arm.lower, 0.1, 0.6)) {
        personArmLengths[side].upper = arm.upper
        personArmLengths[side].lower = arm.lower
        tPoseCalibrated[side] = true
      }
    }
    if (p.reachGain) {
      for (const side of ['left', 'right'] as const) {
        const g = p.reachGain[side]
        if (ok(g, 0.99, REACH_MAX_GAIN + 0.01)) reachGain[side] = g
      }
    }
  } catch {
    // corrupted or unavailable — fall back to ambient estimation
  }
}

// ---------- Reach calibration ----------
// The raw wrist-to-shoulder distance under-reports a genuine full reach by a
// measured 5-10% (MediaPipe's world z compresses near full extension), and
// the law of cosines the IK solve uses is sharply sensitive to exactly that
// last stretch: 95-98% of the true reach distance still only opened the
// elbow to ~140-157°, not the ~170°+ a fully straight arm needs to look right
// (measured live). Press R while holding a genuine full reach once to
// measure how far short of the person's own arm length the raw distance
// falls, and stretch every later reading by that same ratio before it's
// clamped in solveTwoBoneIK - so the clamp's ceiling is actually reachable.
const reachGain = { left: 1, right: 1 }
const REACH_DELAY_MS = 4000
const REACH_HOLD_MS = 1500
const REACH_MIN_RAW_FRACTION = 0.5 // of the arm's own length — below this, the arm wasn't reaching at all
const REACH_MAX_GAIN = 1.3
const reachCapture: { until: number | null; startAt: number; lastShownSecond: number; left: number[]; right: number[] } = {
  until: null,
  startAt: 0,
  lastShownSecond: -1,
  left: [],
  right: []
}

window.addEventListener('keydown', (event) => {
  if (event.key !== 'r' && event.key !== 'R') return
  const pressedAt = performance.now()
  reachCapture.startAt = pressedAt + REACH_DELAY_MS
  reachCapture.until = reachCapture.startAt + REACH_HOLD_MS
  reachCapture.lastShownSecond = -1
  reachCapture.left.length = 0
  reachCapture.right.length = 0
  setStatus(`뻗기 측정 준비 — ${Math.ceil(REACH_DELAY_MS / 1000)}초 안에 팔을 앞으로 쭉 뻗고 유지하세요`)
})

function updateReachCapture(leftRawDist: number, rightRawDist: number, now: number): void {
  if (reachCapture.until === null) return
  if (now < reachCapture.startAt) {
    const secondsLeft = Math.ceil((reachCapture.startAt - now) / 1000)
    if (secondsLeft !== reachCapture.lastShownSecond) {
      reachCapture.lastShownSecond = secondsLeft
      setStatus(`뻗기 측정 준비 — ${secondsLeft}초 후 측정 시작, 팔을 쭉 뻗은 채 유지하세요`)
    }
    return
  }
  if (reachCapture.lastShownSecond !== 0) {
    reachCapture.lastShownSecond = 0
    setStatus('뻗기 측정 중... 그대로 유지하세요')
  }
  if (leftRawDist > 0) reachCapture.left.push(leftRawDist)
  if (rightRawDist > 0) reachCapture.right.push(rightRawDist)
  if (now < reachCapture.until) return

  const parts: string[] = []
  for (const side of ['left', 'right'] as const) {
    const samples = reachCapture[side]
    const label = side === 'left' ? '왼팔' : '오른팔'
    if (samples.length < TPOSE_MIN_SAMPLES) {
      parts.push(`${label} 데이터 부족`)
      continue
    }
    const measured = median(samples)
    const trueSpan = personArmLengths[side].upper + personArmLengths[side].lower
    if (measured < trueSpan * REACH_MIN_RAW_FRACTION) {
      parts.push(`${label} 뻗지 않음`)
      continue
    }
    reachGain[side] = Math.min(REACH_MAX_GAIN, Math.max(1, trueSpan / measured))
    debugLog(`[reach] ${side} measured=${measured.toFixed(3)} trueSpan=${trueSpan.toFixed(3)} gain=${reachGain[side].toFixed(3)}`)
    parts.push(`${label} 게인 ${reachGain[side].toFixed(2)}`)
  }
  reachCapture.until = null
  saveTPoseProfile()
  setStatus(`뻗기 측정 완료 — ${parts.join(', ')}`)
}

// Raw world shoulder widths sampled while both arms are held out: the torso
// faces the camera then, so this is the true frontal width. Re-measured every
// T-pose, because the shoulder-rotation dead zone is relative to it and a stale
// value (e.g. saved under a different pose model) reads as a tilted shoulder line.
const tPoseShoulderWidths: number[] = []

// Largest value of the short moving average. The on-screen length of a bone can
// only ever fall short of its real length (bent, tilted, foreshortened), so during
// a T-pose the biggest sustained value is the closest to the truth and the median
// would sit below it; averaging over a few frames first keeps a single frame of
// upward landmark noise from being taken as the length.
function smoothedMax(values: number[], window = 5): number {
  if (values.length < window) return median(values)
  let best = 0
  for (let i = 0; i + window <= values.length; i++) {
    let sum = 0
    for (let j = 0; j < window; j++) sum += values[i + j]
    best = Math.max(best, sum / window)
  }
  return best
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

// ---------- Bone length self-correction (upward only) ----------
// A bone's on-screen length can never exceed its real length. So if the length
// measured on screen sits above the stored L for seconds on end (measured: 1.2-1.3x
// with L = 0.178 stored from a T-pose that was held slightly tilted), L is too
// short — and the IK solve then bends the elbow more than it should, since the
// target distance keeps clamping to a too-short reach. L is raised toward what
// the arm actually shows, never lowered: shrinking would need the opposite
// evidence, which noise can't provide safely.
// Window widened and the percentile lowered from 0.9 to 0.7 (live-tested): a
// short burst of a few hundred ms of glitchy/fast-motion xy could dominate the
// top 10% of a 4s window and ratchet the length up by 20%+ in a few seconds,
// after which a genuine full reach could no longer reach the (now too long)
// target and looked permanently bent. Requiring a much larger share of the
// window to be elevated means only a sustained underestimate — not a brief
// spike — can still trigger it.
const RATCHET_WINDOW_MS = 8000
const RATCHET_CHECK_MS = 500
const RATCHET_PERCENTILE = 0.7
const RATCHET_TRIGGER = 1.05
const RATCHET_TARGET_FRACTION = 0.97
const RATCHET_EASE = 0.5
interface RatchetTrack {
  samples: { v: number; at: number }[]
  lastCheckAt: number
}
const ratchetTracks = {
  left: { upper: { samples: [], lastCheckAt: 0 } as RatchetTrack, lower: { samples: [], lastCheckAt: 0 } as RatchetTrack },
  right: { upper: { samples: [], lastCheckAt: 0 } as RatchetTrack, lower: { samples: [], lastCheckAt: 0 } as RatchetTrack }
}

function updateLengthRatchet(side: 'left' | 'right', part: 'upper' | 'lower', xyLength: number, now: number): void {
  const track = ratchetTracks[side][part]
  track.samples.push({ v: xyLength, at: now })
  const cutoff = now - RATCHET_WINDOW_MS
  while (track.samples.length > 0 && track.samples[0].at < cutoff) track.samples.shift()
  if (now - track.lastCheckAt < RATCHET_CHECK_MS) return
  track.lastCheckAt = now
  if (track.samples.length < TPOSE_MIN_SAMPLES * 2) return

  const sorted = track.samples.map((sample) => sample.v).sort((a, b) => a - b)
  const high = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * RATCHET_PERCENTILE))]
  const current = personArmLengths[side][part]
  if (current < 1e-4 || high < current * RATCHET_TRIGGER) return

  const target = high * RATCHET_TARGET_FRACTION
  personArmLengths[side][part] = current + (target - current) * RATCHET_EASE
  debugLog(`[ratchet] ${side} ${part}: L ${current.toFixed(3)} -> ${personArmLengths[side][part].toFixed(3)} (p90 xy=${high.toFixed(3)})`)
  saveTPoseProfile()
}

type ResolvedArmResult = { arm: ArmJoints | null; confident: boolean; upperXY: number; lowerXY: number; wholeXY: number; wristDy: number }

function isTPosePose(resolved: ResolvedArmResult): boolean {
  const arm = resolved.arm
  if (!arm || !resolved.confident) return false
  const wholeXY = resolved.wholeXY
  const segmentsXY = resolved.upperXY + resolved.lowerXY
  return segmentsXY > 0.3 && wholeXY >= TPOSE_MIN_STRAIGHTNESS * segmentsXY && Math.abs(resolved.wristDy) <= TPOSE_MAX_DROOP_RATIO * wholeXY
}

// BOTH arms have to be held out at once. Raising one arm sideways passes
// through "straight and horizontal" all the time in ordinary movement, and an
// earlier per-arm trigger kept re-measuring (and re-locking the scale) in the
// middle of normal use, each time with a slightly different value.
// The T-pose is measured only on request (the app's measure button): held out automatically
// it kept re-measuring, and re-locking the scale, in the middle of normal use.
const TPOSE_REQUEST_TIMEOUT_MS = 20000
let tPoseRequestUntil: number | null = null
let tPoseRequestLastSecond = -1
export function requestTPoseMeasurement(): void {
  tPoseRequestUntil = performance.now() + TPOSE_REQUEST_TIMEOUT_MS
  tPoseRequestLastSecond = -1
  setStatus('T포즈 측정 대기 — 양팔을 옆으로 쭉 펴고 2초 유지하세요')
}

function updateTPoseCalibration(left: ResolvedArmResult, right: ResolvedArmResult, now: number): void {
  if (tPoseRequestUntil === null) return
  if (now > tPoseRequestUntil) {
    tPoseRequestUntil = null
    setStatus('T포즈 측정 실패 — 양팔을 화면 안에서 수평으로 쭉 펴고 다시 눌러주세요')
    return
  }
  const sides = [
    { side: 'left' as const, resolved: left },
    { side: 'right' as const, resolved: right }
  ]
  const bothInPose = isTPosePose(left) && isTPosePose(right)
  if (!bothInPose) {
    tPoseShoulderWidths.length = 0
    for (const { side } of sides) {
      tPoseHold[side].startedAt = null
      tPoseHold[side].upper.length = 0
      tPoseHold[side].lower.length = 0
    }
    const waited = Math.floor((now - (tPoseRequestUntil - TPOSE_REQUEST_TIMEOUT_MS)) / 5000)
    if (waited !== tPoseRequestLastSecond) {
      tPoseRequestLastSecond = waited
      setStatus('T포즈 측정 대기 — 양팔을 옆으로 쭉 펴고 2초 유지하세요')
    }
    return
  }
  if (rawShoulderWidthXY !== null) tPoseShoulderWidths.push(rawShoulderWidthXY)
  for (const { side, resolved } of sides) {
    const hold = tPoseHold[side]
    if (hold.startedAt === null) hold.startedAt = now
    hold.upper.push(resolved.upperXY)
    hold.lower.push(resolved.lowerXY)
  }
  const first = tPoseHold.left
  if (first.startedAt === null || now - first.startedAt < TPOSE_HOLD_MS || first.upper.length < TPOSE_MIN_SAMPLES) return

  for (const { side } of sides) {
    const hold = tPoseHold[side]
    personArmLengths[side].upper = smoothedMax(hold.upper)
    personArmLengths[side].lower = smoothedMax(hold.lower)
    tPoseCalibrated[side] = true
    debugLog(`[tpose] ${side} measured: upper=${personArmLengths[side].upper.toFixed(3)} lower=${personArmLengths[side].lower.toFixed(3)}`)
    hold.startedAt = null
    hold.upper.length = 0
    hold.lower.length = 0
  }
  if (tPoseShoulderWidths.length >= TPOSE_MIN_SAMPLES) {
    personShoulderWidth = Math.min(0.6, Math.max(0.2, median(tPoseShoulderWidths)))
  }
  tPoseShoulderWidths.length = 0
  shoulderWidthFrozen = true
  tPoseRequestUntil = null
  saveTPoseProfile()
  setStatus(
    `T포즈 측정 완료 (어깨 ${personShoulderWidth.toFixed(2)}) — 왼팔 ${personArmLengths.left.upper.toFixed(2)}/${personArmLengths.left.lower.toFixed(2)}, ` +
      `오른팔 ${personArmLengths.right.upper.toFixed(2)}/${personArmLengths.right.lower.toFixed(2)}`
  )
}

loadTPoseProfile()

let lastDebugLogAt = 0
// TEMP debug: the reconstructed elbow's own world position from the last
// logged frame, per side - lets the log show how far the FINAL elbow
// position actually moved this frame (elbowMove below), which is what's
// actually visible on screen, instead of the rotation-angle proxies (pole
// vs u) that turned out not to isolate noise cleanly - see chat.
const lastLoggedElbow: { left: Vec3Like | null; right: Vec3Like | null } = { left: null, right: null }

export function resolveTrackingFrame(data: TrackingData, dt: number): ResolvedFrame {
  const left = resolveArm(
    data.poseWorldLandmarks,
    11,
    13,
    15,
    personArmLengths.left,
    data.leftHandLandmarks,
    data.leftHandWorldLandmarks,
    heldLeftShoulder,
    heldLeftElbow,
    heldLeftWrist,
    heldLeftPole,
    reachGain.left,
    dt,
    heldLeftReachDist,
    heldLeftHandSeen,
    heldLeftHandWristTrust
  )
  const right = resolveArm(
    data.poseWorldLandmarks,
    12,
    14,
    16,
    personArmLengths.right,
    data.rightHandLandmarks,
    data.rightHandWorldLandmarks,
    heldRightShoulder,
    heldRightElbow,
    heldRightWrist,
    heldRightPole,
    reachGain.right,
    dt,
    heldRightReachDist,
    heldRightHandSeen,
    heldRightHandWristTrust
  )
  updateReachCapture(left.rawReachDist, right.rawReachDist, performance.now())

  // TEMP debug: the IK solve's inputs/outputs — raw target distance vs what the
  // triangle actually reached, the drawn elbow angle and the raw landmark visibility.
  const dbgNow = performance.now()
  if (debugEnabled && data.poseWorldLandmarks && dbgNow - lastDebugLogAt > 80) {
    lastDebugLogAt = dbgNow
    const w = data.poseWorldLandmarks
    const img = data.poseLandmarks
    const sides = [
      {
        name: 'L',
        r: left,
        el: 13,
        wr: 15,
        lengths: personArmLengths.left,
        gain: reachGain.left,
        handLm: data.leftHandLandmarks,
        handWorldLm: data.leftHandWorldLandmarks,
        held: heldLeftReachDist,
        pole: heldLeftPole,
        trust: heldLeftHandWristTrust
      },
      {
        name: 'R',
        r: right,
        el: 14,
        wr: 16,
        lengths: personArmLengths.right,
        gain: reachGain.right,
        handLm: data.rightHandLandmarks,
        handWorldLm: data.rightHandWorldLandmarks,
        held: heldRightReachDist,
        pole: heldRightPole,
        trust: heldRightHandWristTrust
      }
    ]
    for (const { name, r, el, wr, lengths, gain, handLm, handWorldLm, held, pole, trust } of sides) {
      if (!r.arm) continue
      const a = r.arm
      const solvedDist = Math.hypot(a.wrist.x - a.shoulder.x, a.wrist.y - a.shoulder.y, a.wrist.z - a.shoulder.z)
      // TEMP debug: the bend-direction hint actually being used (held.pole), and
      // the raw elbow landmark's own perpendicular offset this frame (rawPoleDeg) -
      // checking whether the elbow "popping outward" (reported live, lowering a
      // reached arm below the chest) traces back to this rotating unexpectedly.
      const p = pole.current
      const poleStr = p ? `${p.x.toFixed(2)}/${p.y.toFixed(2)}/${p.z.toFixed(2)}` : 'n/a'
      // TEMP debug: the shoulder->wrist direction (u) this frame - pole is
      // defined relative to u, so when u rotates (the wrist moving, e.g.
      // raising the forearm), pole rotating by roughly the SAME amount is
      // expected/self-cancelling, not drift. Logging u lets that expected
      // rotation be told apart from whatever's left over (see chat: is the
      // pole swing during a forearm raise just tracking u, or more than u).
      const uDist = solvedDist > 1e-4 ? solvedDist : 1
      const uStr = `${((a.wrist.x - a.shoulder.x) / uDist).toFixed(2)}/${((a.wrist.y - a.shoulder.y) / uDist).toFixed(2)}/${(
        (a.wrist.z - a.shoulder.z) / uDist
      ).toFixed(2)}`
      // TEMP debug: how far the reconstructed elbow's world position actually
      // moved since the last logged frame - the thing actually visible
      // on-screen, unlike the pole/u rotation angles above.
      const lastElbowKey = name === 'L' ? 'left' : 'right'
      const lastElbow = lastLoggedElbow[lastElbowKey]
      const elbowMove = lastElbow
        ? Math.hypot(a.elbow.x - lastElbow.x, a.elbow.y - lastElbow.y, a.elbow.z - lastElbow.z).toFixed(3)
        : 'n/a'
      lastLoggedElbow[lastElbowKey] = { x: a.elbow.x, y: a.elbow.y, z: a.elbow.z }
      // TEMP debug: the elbow's depth relative to the shoulder, isolated from
      // x/y - directly answers "did it actually move toward the camera"
      // instead of inferring that from pole/u's world x/y/z mix (see chat:
      // pole.x/pole.y are world-frame, not screen-frame, so a swap there
      // doesn't by itself say whether the elbow moved toward the camera).
      const elbowRelZ = (a.elbow.z - a.shoulder.z).toFixed(3)
      // TEMP debug: the actual forearm SEGMENT direction (elbow->wrist,
      // normalized) - see chat: "raise the forearm" was reported as not
      // ending up vertical (y-dominant) even after the torso-clamp fix
      // turned out to be a red herring (it only ever pushed elbow.z by
      // 1-2cm). elbowRelZ/pole/u above all describe the shoulder-side of the
      // triangle; this is the one thing that directly says which way the
      // rendered forearm bone actually points.
      const forearmLen = Math.hypot(a.wrist.x - a.elbow.x, a.wrist.y - a.elbow.y, a.wrist.z - a.elbow.z)
      const forearmDirStr =
        forearmLen > 1e-4
          ? `${((a.wrist.x - a.elbow.x) / forearmLen).toFixed(2)}/${((a.wrist.y - a.elbow.y) / forearmLen).toFixed(2)}/${(
              (a.wrist.z - a.elbow.z) / forearmLen
            ).toFixed(2)}`
          : 'n/a'
      // TEMP debug: pose model's own wrist vs the hand model's wrist, image distance in
      // shoulder widths - checking whether the hand model tracks the wrist more reliably
      // than the pose model in the low-wristVis "in front of face/torso" case (see chat).
      let wristGap = 'n/a'
      if (img && img[11] && img[12] && img[wr] && handLm && handLm[0]) {
        const imgShoulderW = Math.hypot((img[11].x - img[12].x) * currentAspect, img[11].y - img[12].y)
        if (imgShoulderW > 1e-4) {
          wristGap = (Math.hypot((img[wr].x - handLm[0].x) * currentAspect, img[wr].y - handLm[0].y) / imgShoulderW).toFixed(2)
        }
      }
      // TEMP debug: pose model's wrist vs hand model's wrist in WORLD coordinates
      // (not image space, see wristGap/sw above) - checking whether the two
      // actually share one coordinate frame in this unified Holistic model,
      // before trusting the hand model's wrist as a fallback for the pose
      // model's. Only meaningful when wristVis is high (both should agree).
      let worldWristGap = 'n/a'
      if (handWorldLm && handWorldLm[0]) {
        worldWristGap = Math.hypot(w[wr].x - handWorldLm[0].x, w[wr].y - handWorldLm[0].y, w[wr].z - handWorldLm[0].z).toFixed(3)
      }
      // TEMP debug: same frame, 2D hand landmarks present vs 3D world hand
      // landmarks present - checking whether these two ever diverge (2D
      // detected but world missing), which would mean the wrist-follow
      // fallback is losing a hand the on-screen skeleton still shows.
      const hand2D = handLm && handLm[0] ? 'y' : 'n'
      const hand3D = handWorldLm && handWorldLm[0] ? 'y' : 'n'
      // TEMP debug: a cheap fingerprint of the RIGGED finger shape actually fed
      // to the VRM (heldLeftHand/heldRightHand.current) - sum of every finger
      // landmark's x+y+z. If this number stops changing across consecutive
      // lines while hand3D=n, the finger pose really is frozen on-screen; if it
      // keeps changing, something other than the raw per-frame hand landmarks
      // is still driving it.
      const heldHand = name === 'L' ? heldLeftHand.current : heldRightHand.current
      const handShapeSum = heldHand ? heldHand.reduce((s, p) => s + p.x + p.y + p.z, 0).toFixed(4) : 'n/a'
      // TEMP debug: the hand-wrist corroboration gate's state - see
      // resolveTrustedHandWrist. 'confirmed' means the ikWrist blend is using
      // the hand model's wrist this frame; 'pending'/'none' mean it isn't yet
      // (or fell back after a stale gap) and the pose model's own wrist is
      // carrying the target alone until a corroborating frame lands.
      const wristTrust = trust.confirmed ? 'confirmed' : trust.pendingCandidate ? 'pending' : 'none'
      debugLog(
        `[ik ${name}] L=${lengths.upper.toFixed(3)}/${lengths.lower.toFixed(3)} rawDist=${r.rawReachDist.toFixed(3)} gain=${gain.toFixed(2)} held=${held.value.toFixed(3)} solvedDist=${solvedDist.toFixed(3)} ` +
          `elbowAngle=${jointAngleDeg(a.shoulder, a.elbow, a.wrist)} elbVis=${w[el].visibility.toFixed(2)} wristVis=${w[wr].visibility.toFixed(2)} ` +
          `xyUp=${r.upperXY.toFixed(3)} xyLo=${r.lowerXY.toFixed(3)} wristGap/sw=${wristGap} worldWristGap=${worldWristGap} hand2D=${hand2D} hand3D=${hand3D} handShapeSum=${handShapeSum} wristTrust=${wristTrust} pole=${poleStr} u=${uStr} elbowMove=${elbowMove} elbowRelZ=${elbowRelZ} forearmDir=${forearmDirStr} forceFullReach=${r.forceFullReach}`
      )
    }
  }

  // See main.ts's matching block in resolveTrackingFrame for the reasoning.
  const now = performance.now()
  if (left.arm && right.arm) {
    if (rawShoulderWidthXY !== null) ambientSample(shoulderWidthEstimator, rawShoulderWidthXY, now)
  }
  {
    const ratchetNow = performance.now()
    if (left.arm && left.confident) {
      updateLengthRatchet('left', 'upper', left.upperXY, ratchetNow)
      updateLengthRatchet('left', 'lower', left.lowerXY, ratchetNow)
    }
    if (right.arm && right.confident) {
      updateLengthRatchet('right', 'upper', right.upperXY, ratchetNow)
      updateLengthRatchet('right', 'lower', right.lowerXY, ratchetNow)
    }
  }
  // T-pose auto-calibration — see the block above resolveTrackingFrame.
  updateTPoseCalibration(left, right, now)

  if (left.confident) {
    ambientSample(armLengthEstimators.left.upper, left.upperXY, now)
    ambientSample(armLengthEstimators.left.lower, left.lowerXY, now)
  }
  if (right.confident) {
    ambientSample(armLengthEstimators.right.upper, right.upperXY, now)
    ambientSample(armLengthEstimators.right.lower, right.lowerXY, now)
  }
  if (shoulderWidthEstimator.target !== null && !shoulderWidthFrozen) {
    personShoulderWidth = Math.min(0.5, Math.max(0.25, ambientLerp(personShoulderWidth, shoulderWidthEstimator.target, dt)))
  }
  const leftUpperTarget = currentArmLengthTarget(personShoulderWidth, ARM_UPPER_TO_SHOULDER_WIDTH_RATIO, armLengthEstimators.left.upper)
  const leftLowerTarget = currentArmLengthTarget(personShoulderWidth, ARM_LOWER_TO_SHOULDER_WIDTH_RATIO, armLengthEstimators.left.lower)
  const rightUpperTarget = currentArmLengthTarget(personShoulderWidth, ARM_UPPER_TO_SHOULDER_WIDTH_RATIO, armLengthEstimators.right.upper)
  const rightLowerTarget = currentArmLengthTarget(personShoulderWidth, ARM_LOWER_TO_SHOULDER_WIDTH_RATIO, armLengthEstimators.right.lower)
  if (leftUpperTarget !== null && !tPoseCalibrated.left) personArmLengths.left.upper = ambientLerp(personArmLengths.left.upper, leftUpperTarget, dt)
  if (leftLowerTarget !== null && !tPoseCalibrated.left) personArmLengths.left.lower = ambientLerp(personArmLengths.left.lower, leftLowerTarget, dt)
  if (rightUpperTarget !== null && !tPoseCalibrated.right) personArmLengths.right.upper = ambientLerp(personArmLengths.right.upper, rightUpperTarget, dt)
  if (rightLowerTarget !== null && !tPoseCalibrated.right) personArmLengths.right.lower = ambientLerp(personArmLengths.right.lower, rightLowerTarget, dt)

  if (left.arm && right.arm) {
    const rawLeftShoulder = left.arm.shoulder
    const rawRightShoulder = right.arm.shoulder
    const shoulderDz = computeShoulderDepth(rawRightShoulder, rawLeftShoulder, personShoulderWidth, dt)
    const avgShoulderZ = (rawLeftShoulder.z + rawRightShoulder.z) / 2
    const leftShoulderDelta = avgShoulderZ + shoulderDz / 2 - rawLeftShoulder.z
    const rightShoulderDelta = avgShoulderZ - shoulderDz / 2 - rawRightShoulder.z

    left.arm.shoulder = { x: rawLeftShoulder.x, y: rawLeftShoulder.y, z: rawLeftShoulder.z + leftShoulderDelta }
    left.arm.elbow = { x: left.arm.elbow.x, y: left.arm.elbow.y, z: left.arm.elbow.z + leftShoulderDelta }
    left.arm.wrist = { x: left.arm.wrist.x, y: left.arm.wrist.y, z: left.arm.wrist.z + leftShoulderDelta }
    right.arm.shoulder = { x: rawRightShoulder.x, y: rawRightShoulder.y, z: rawRightShoulder.z + rightShoulderDelta }
    right.arm.elbow = { x: right.arm.elbow.x, y: right.arm.elbow.y, z: right.arm.elbow.z + rightShoulderDelta }
    right.arm.wrist = { x: right.arm.wrist.x, y: right.arm.wrist.y, z: right.arm.wrist.z + rightShoulderDelta }
  }

  if (left.arm && right.arm) {
    const shoulderMinX = Math.min(left.arm.shoulder.x, right.arm.shoulder.x)
    const shoulderMaxX = Math.max(left.arm.shoulder.x, right.arm.shoulder.x)
    const bodyMidX = (shoulderMinX + shoulderMaxX) / 2
    const shoulderHalfWidth = (shoulderMaxX - shoulderMinX) / 2
    // See CROSS_BODY_INSET above.
    const torsoMinX = bodyMidX - shoulderHalfWidth * CROSS_BODY_INSET
    const torsoMaxX = bodyMidX + shoulderHalfWidth * CROSS_BODY_INSET
    const torsoZ = (left.arm.shoulder.z + right.arm.shoulder.z) / 2
    left.arm.elbow = clampInFrontOfTorso(left.arm.elbow, torsoMinX, torsoMaxX, torsoZ)
    left.arm.wrist = clampInFrontOfTorso(left.arm.wrist, torsoMinX, torsoMaxX, torsoZ)
    right.arm.elbow = clampInFrontOfTorso(right.arm.elbow, torsoMinX, torsoMaxX, torsoZ)
    right.arm.wrist = clampInFrontOfTorso(right.arm.wrist, torsoMinX, torsoMaxX, torsoZ)
  }

  // Hands carry no usable visibility score of their own: gated on the same side's wrist
  // confidence (plus a finite check, like holdJoint), and held across low-confidence frames.
  // The raw landmarks are smoothed first (see smoothHandLandmarks) - a hand pointed straight
  // at the camera is the model's worst case (fingers overlap, little depth cue between them)
  // and it visibly guesses a different finger shape most frames even when the hand is still.
  if (data.leftHandWorldLandmarks && left.confident && data.leftHandWorldLandmarks.every(isFiniteVec3)) {
    heldLeftHand.current = smoothHandLandmarks('left', data.leftHandWorldLandmarks, dt)
  } else {
    resetHandLandmarkFilter('left')
  }
  if (data.rightHandWorldLandmarks && right.confident && data.rightHandWorldLandmarks.every(isFiniteVec3)) {
    heldRightHand.current = smoothHandLandmarks('right', data.rightHandWorldLandmarks, dt)
  } else {
    resetHandLandmarkFilter('right')
  }

  return {
    left: left.arm,
    right: right.arm,
    leftConfident: left.confident,
    rightConfident: right.confident,
    leftHand: heldLeftHand.current,
    rightHand: heldRightHand.current,
    leftDiagnostics: armDiagnostics(left.arm, data.poseWorldLandmarks, 11, 13, 15),
    rightDiagnostics: armDiagnostics(right.arm, data.poseWorldLandmarks, 12, 14, 16)
  }
}

const heldLeftHand: { current: Vec3Like[] | null } = { current: null }
const heldRightHand: { current: Vec3Like[] | null } = { current: null }

// Per-landmark (21 points x/y/z) smoothing for the hand model's own output - see the call
// site above. Same filter class and similar tuning to the arm segments (main.ts's
// ARM_FILTER_MIN_CUTOFF/BETA); finger points are smaller/faster-moving on purpose (a quick
// fist should still keep up), so the cutoff is a bit higher.
const HAND_FILTER_MIN_CUTOFF = 1.0
const HAND_FILTER_BETA = 0.5
type HandLandmarkFilters = { x: OneEuroFilter; y: OneEuroFilter; z: OneEuroFilter }[]
const handLandmarkFilters: { left: HandLandmarkFilters; right: HandLandmarkFilters } = { left: [], right: [] }

function smoothHandLandmarks(side: 'left' | 'right', landmarks: Vec3Like[], dt: number): Vec3Like[] {
  const filters = handLandmarkFilters[side]
  return landmarks.map((p, i) => {
    let f = filters[i]
    if (!f) f = filters[i] = { x: new OneEuroFilter(HAND_FILTER_MIN_CUTOFF, HAND_FILTER_BETA), y: new OneEuroFilter(HAND_FILTER_MIN_CUTOFF, HAND_FILTER_BETA), z: new OneEuroFilter(HAND_FILTER_MIN_CUTOFF, HAND_FILTER_BETA) }
    return { x: f.x.filter(p.x, dt), y: f.y.filter(p.y, dt), z: f.z.filter(p.z, dt) }
  })
}

function resetHandLandmarkFilter(side: 'left' | 'right'): void {
  for (const f of handLandmarkFilters[side]) {
    f.x.reset()
    f.y.reset()
    f.z.reset()
  }
}

const ZERO_DELTA: Vec3Like = { x: 0, y: 0, z: 0 }
function armDiagnostics(arm: ArmJoints | null, world: Landmark[] | undefined, shoulderIdx: number, elbowIdx: number, wristIdx: number): ArmDiagnostics {
  if (!arm) return { upperDepth: 0, lowerDepth: 0, rawUpperDelta: ZERO_DELTA, rawLowerDelta: ZERO_DELTA }
  const delta = (a: number, b: number): Vec3Like =>
    world?.[a] && world?.[b] ? { x: world[b].x - world[a].x, y: world[b].y - world[a].y, z: world[b].z - world[a].z } : ZERO_DELTA
  return {
    upperDepth: arm.elbow.z - arm.shoulder.z,
    lowerDepth: arm.wrist.z - arm.elbow.z,
    rawUpperDelta: delta(shoulderIdx, elbowIdx),
    rawLowerDelta: delta(elbowIdx, wristIdx)
  }
}
