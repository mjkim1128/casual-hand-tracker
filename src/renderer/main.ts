import {
  FilesetResolver,
  HolisticLandmarker,
  HandLandmarker,
  DrawingUtils,
  type HolisticLandmarkerResult,
  type HandLandmarkerResult,
  type Landmark,
  type NormalizedLandmark
} from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils, VRMHumanBoneName, type VRM } from '@pixiv/three-vrm'
import {
  isFiniteVec3,
  OneEuroFilter,
  requestTPoseMeasurement,
  resolveTrackingFrame,
  setTrackingDebug,
  setTrackingStatusHandler,
  trackingDataFromResult,
  withImageXY,
  type ArmDiagnostics,
  type ArmJoints,
  type ResolvedFrame,
  type TrackingData,
  type Vec3Like
} from './armTracking'

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task'
// Holistic's own hand branch derives its crop region from the POSE model's
// wrist landmark, so it goes blind in lockstep whenever the pose wrist
// estimate drifts (measured live: hand2D/hand3D dropped to nothing for 5+
// seconds, exactly matching the frames where wristVis collapsed) - even
// though the real hand was still plainly in frame. A standalone HandLandmarker
// detects hands directly from the image, independent of the pose branch, so
// it can keep tracking through exactly that failure mode. Its own output is
// then reassigned to left/right by matching against the pose wrists (see
// assignHandsToSides) rather than trusting its `handedness` classification,
// since that assumes a mirrored/selfie image and MediaPipe's own crossing
// convention there doesn't necessarily match Holistic's.
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
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
const ctx = canvas.getContext('2d')!

function setStatus(text: string): void {
  statusEl.textContent = text
}
setTrackingStatusHandler(setStatus)
// [ik]/[tpose]/[glitch]/[ratchet] diagnostics go to the terminal running `npm run dev` (see mirrorConsole).
setTrackingDebug(true)
document.getElementById('measure-button')!.addEventListener('click', () => requestTPoseMeasurement())


// poseWorldLandmarks (used for arm/hand rigging) are re-centered on the hips
// every single frame, so they can never reveal whole-body movement — only
// joint-relative pose. The normalized image-space landmarks are the only
// signal for "where in frame is this person right now", so on-screen body
// movement is driven off of those instead: track how far the shoulder
// midpoint has drifted (in normalized 0-1 units) from a fixed reference
// point, and turn that into a small world-space offset.
const BODY_OFFSET_SCALE = { x: 1.4, y: 1.4 }
const BODY_OFFSET_LERP = 0.2
// Normalized image-space point that maps to "no offset" (avatar rendered
// at its own neutral position). This used to be averaged from wherever the
// person happened to be standing during the first ~60 tracked frames after
// startup, which meant "centered" silently became whatever position that
// was — off-frame if the person wasn't already in place in time — and
// needed a fresh manual recenter practically every run. A fixed point tied
// to the camera frame itself (dead center) instead makes "stand in the
// middle of your own webcam view" reliably produce a centered avatar,
// every session, with no first-few-seconds dependency. Tune these two
// constants (not the mechanism) if testing shows the avatar sits off-center
// even when the person is centered in frame — that just means this
// project's typical camera framing doesn't put the shoulder midpoint at
// exactly (0.5, 0.5).
const BODY_OFFSET_CENTER_X = 0.5
const BODY_OFFSET_CENTER_Y = 0.5

function computeBodyOffsetTarget(poseLandmarks: Vec3Like[] | undefined): THREE.Vector3 | null {
  if (!poseLandmarks) return null
  const shoulderMid = {
    x: (poseLandmarks[11].x + poseLandmarks[12].x) / 2,
    y: (poseLandmarks[11].y + poseLandmarks[12].y) / 2
  }

  const dx = shoulderMid.x - BODY_OFFSET_CENTER_X
  const dy = shoulderMid.y - BODY_OFFSET_CENTER_Y
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
    // Lowered from the 0.5 default: the hand model dropped the hand for the whole of every
    // forward reach - see whether a laxer threshold keeps it.
    minHandLandmarksConfidence: 0.2
  })
}

// See HAND_MODEL_URL - independent of Holistic's pose-derived hand crop.
// Measured live: this returned rawCandidates=0 on every single frame under
// both GPU and CPU delegate, with no exception thrown from detectForVideo
// either way - ruling out the GPU-context-conflict theory tried earlier.
// Back on GPU (CPU only made it slower for no benefit). The one-time log
// right after creation, plus the try/catch around detectForVideo below,
// are there to catch whatever's actually failing instead of guessing again.
async function setupHandLandmarker(): Promise<HandLandmarker> {
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE)
  const landmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.2,
    minHandPresenceConfidence: 0.2,
    minTrackingConfidence: 0.2
  })
  console.log('[hand] HandLandmarker created successfully')
  return landmarker
}

interface AssignedHand {
  landmarks: NormalizedLandmark[]
  worldLandmarks: Landmark[]
}
interface AssignedHands {
  left: AssignedHand | null
  right: AssignedHand | null
}

// Matches each independently-detected hand to the person's left/right arm by
// image-space proximity to the POSE model's own wrist landmarks (11/13/15 =
// left, 12/14/16 = right), rather than trusting HandLandmarker's own
// `handedness` classification - that assumes a mirrored/selfie input and
// isn't guaranteed to line up with which side Holistic's pose skeleton calls
// "left". Two independent detections both racing for the same side (hands
// crossed/overlapping) resolve to whichever is closer; the loser is dropped
// rather than double-assigned.
function assignHandsToSides(handResult: HandLandmarkerResult, poseLandmarks: Vec3Like[] | undefined): AssignedHands {
  const result: AssignedHands = { left: null, right: null }
  if (!poseLandmarks || !poseLandmarks[15] || !poseLandmarks[16]) return result
  const leftWrist = poseLandmarks[15]
  const rightWrist = poseLandmarks[16]

  const candidates = handResult.landmarks
    .map((landmarks, i) => ({ landmarks, worldLandmarks: handResult.worldLandmarks[i], wrist: landmarks[0] }))
    .filter((c) => c.wrist)

  let bestLeft: { dist: number; hand: AssignedHand } | null = null
  let bestRight: { dist: number; hand: AssignedHand } | null = null
  for (const c of candidates) {
    const dLeft = Math.hypot(c.wrist.x - leftWrist.x, c.wrist.y - leftWrist.y)
    const dRight = Math.hypot(c.wrist.x - rightWrist.x, c.wrist.y - rightWrist.y)
    const hand: AssignedHand = { landmarks: c.landmarks, worldLandmarks: c.worldLandmarks }
    if (dLeft <= dRight) {
      if (!bestLeft || dLeft < bestLeft.dist) bestLeft = { dist: dLeft, hand }
    } else {
      if (!bestRight || dRight < bestRight.dist) bestRight = { dist: dRight, hand }
    }
  }
  result.left = bestLeft?.hand ?? null
  result.right = bestRight?.hand ?? null
  return result
}

function drawResult(result: HolisticLandmarkerResult, hands: AssignedHands, drawer: DrawingUtils): void {
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
  // Drawn from the independent HandLandmarker's assignment (see
  // assignHandsToSides), not result.leftHandLandmarks/rightHandLandmarks -
  // this overlay should show exactly what's actually driving the rig.
  if (hands.left) {
    drawer.drawConnectors(hands.left.landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: '#ff5f7e',
      lineWidth: 2
    })
    drawer.drawLandmarks(hands.left.landmarks, { color: '#ff5f7e', radius: 2 })
  }
  if (hands.right) {
    drawer.drawConnectors(hands.right.landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: '#ffd25f',
      lineWidth: 2
    })
    drawer.drawLandmarks(hands.right.landmarks, { color: '#ffd25f', radius: 2 })
  }
  ctx.restore()
}

function updateReadout(
  result: HolisticLandmarkerResult,
  hands: AssignedHands,
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
    `left hand landmarks: ${hands.left?.landmarks.length ?? 0}`,
    `right hand landmarks: ${hands.right?.landmarks.length ?? 0}`,
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
// OneEuroFilter itself is exported from armTracking.ts (used there for the
// hand-landmark and shoulder-depth smoothing). This section covers everything
// else it's used for: the final tracked direction/basis vectors, filtered
// right before aimBoneToward/aimBasisToward (not the raw landmarks — smoothing
// those first would feed the IK solve stale, lagged data).
//
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

  // See peekSpeed — pass a shared speedOverride when this vector needs to
  // stay in lockstep with another filter (e.g. a sibling bone) rather than
  // each picking its own cutoff independently.
  filter(v: THREE.Vector3, dt: number, speedOverride?: number): THREE.Vector3 {
    return new THREE.Vector3(
      this.fx.filter(v.x, dt, speedOverride),
      this.fy.filter(v.y, dt, speedOverride),
      this.fz.filter(v.z, dt, speedOverride)
    )
  }

  // Highest of the 3 channels' own (not-yet-committed) speed estimate —
  // used to synchronize two *different* Vector3OneEuroFilters (e.g. this
  // arm's upper- and lower-segment direction) so they lag by the same
  // amount at every instant instead of each adapting to its own signal's
  // speed independently. Two segments of a genuinely straight, moving arm
  // don't necessarily read the same *raw* speed moment to moment (their
  // directions rotate at related but not identical rates), so without this,
  // one segment's filter output can lag visibly behind the other's during a
  // fast sweep — which looks exactly like the elbow briefly bending and
  // straightening back out, even though the unfiltered tracking data never
  // left the straight line the whole time.
  peekSpeed(v: THREE.Vector3, dt: number): number {
    return Math.max(this.fx.peekSpeed(v.x, dt), this.fy.peekSpeed(v.y, dt), this.fz.peekSpeed(v.z, dt))
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

    // A genuinely straight, moving arm's upper/lower segments don't
    // necessarily read the same *raw* speed moment to moment, so each
    // filter picking its own cutoff independently can lag one segment
    // behind the other during a fast sweep — visibly, an elbow that briefly
    // bends and straightens back out again, even though the underlying
    // tracked positions were on the straight line the entire time. Peeking
    // each segment's own speed first and sharing the faster of the two
    // (see Vector3OneEuroFilter.peekSpeed) keeps both segments' cutoffs in
    // lockstep instead.
    const leftUpperRawDir = new THREE.Vector3(
      personLeftElbow.x - personLeftShoulder.x,
      personLeftElbow.y - personLeftShoulder.y,
      personLeftElbow.z - personLeftShoulder.z
    )
    const leftLowerRawDir = new THREE.Vector3(
      personLeftWrist.x - personLeftElbow.x,
      personLeftWrist.y - personLeftElbow.y,
      personLeftWrist.z - personLeftElbow.z
    )
    const leftArmSpeed = Math.max(
      directionFilters.leftUpper.peekSpeed(leftUpperRawDir, dt),
      directionFilters.leftLower.peekSpeed(leftLowerRawDir, dt)
    )
    const rightUpperRawDir = new THREE.Vector3(
      personRightElbow.x - personRightShoulder.x,
      personRightElbow.y - personRightShoulder.y,
      personRightElbow.z - personRightShoulder.z
    )
    const rightLowerRawDir = new THREE.Vector3(
      personRightWrist.x - personRightElbow.x,
      personRightWrist.y - personRightElbow.y,
      personRightWrist.z - personRightElbow.z
    )
    const rightArmSpeed = Math.max(
      directionFilters.rightUpper.peekSpeed(rightUpperRawDir, dt),
      directionFilters.rightLower.peekSpeed(rightLowerRawDir, dt)
    )

    aimBoneToward(
      vrm,
      B.RightUpperArm,
      calibration.rightArmAim.upper,
      directionFilters.leftUpper.filter(leftUpperRawDir, dt, leftArmSpeed)
    )
    aimBoneToward(
      vrm,
      B.RightLowerArm,
      calibration.rightArmAim.lower,
      directionFilters.leftLower.filter(leftLowerRawDir, dt, leftArmSpeed)
    )
    aimBoneToward(
      vrm,
      B.LeftUpperArm,
      calibration.leftArmAim.upper,
      directionFilters.rightUpper.filter(rightUpperRawDir, dt, rightArmSpeed)
    )
    aimBoneToward(
      vrm,
      B.LeftLowerArm,
      calibration.leftArmAim.lower,
      directionFilters.rightLower.filter(rightLowerRawDir, dt, rightArmSpeed)
    )
  }

  // Crossed the same way as the arms just above — the person's left hand is
  // attached to the avatar's right arm, so it has to drive the avatar's right
  // hand bones too, or the fingers would visibly belong to the other arm.
  applyHandToVRM(vrm, resolved.leftHand, 'Right', calibration.rightHand, dt)
  applyHandToVRM(vrm, resolved.rightHand, 'Left', calibration.leftHand, dt)
}

let lastHandDebugAt = 0

async function main(): Promise<void> {
  setStatus('웹캠 요청 중...')

  const scene = setupAvatarScene()
  const debugTarget = setupAvatarRenderTarget(avatarDebugCanvas, 'front')
  const cleanTarget = setupAvatarRenderTarget(avatarCleanCanvas, 'front')
  const topTarget = setupAvatarRenderTarget(avatarTopCanvas, 'top')
  const sideTarget = setupAvatarRenderTarget(avatarSideCanvas, 'side')

  const [, holistic, handLandmarker, vrm] = await Promise.all([
    startWebcamStream(),
    (async () => {
      setStatus('MediaPipe 모델 로딩 중...')
      return setupHolistic()
    })(),
    setupHandLandmarker(),
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

  // The person's own bone lengths come from the T-pose measured by the tracking pipeline
  // (armTracking.ts, saved in localStorage) - not from the avatar, whose proportions say nothing
  // about the person in front of the camera. Only directions are retargeted onto the avatar.

  await populateCameraList()

  const drawer = new DrawingUtils(ctx)
  setStatus('추적 중')

  let lastFrameTime = performance.now()
  const bodyOffset = new THREE.Vector3()

  // requestAnimationFrame runs on the display's refresh rate, not the camera's -
  // a source with irregular delivery (measured live: DroidCam over WiFi) let this
  // loop re-run full inference plus four scene renders on the same stale video
  // texture multiple times before the next real frame arrived, burning exactly
  // the headroom needed to catch up smoothly. requestVideoFrameCallback only
  // fires once per frame the video element actually decoded.
  const scheduleNext = (cb: () => void): void => {
    if (typeof video.requestVideoFrameCallback === 'function') {
      video.requestVideoFrameCallback(cb)
    } else {
      requestAnimationFrame(cb)
    }
  }

  const loop = (): void => {
    const now = performance.now()
    const deltaSeconds = (now - lastFrameTime) / 1000
    const fps = 1 / deltaSeconds
    lastFrameTime = now

    const result = holistic.detectForVideo(video, now)
    // TEMP debug: rawCandidates has stayed 0 on every single frame across
    // both GPU and CPU delegate, with the rest of the app running fine
    // around it (no thrown exception reaching here) - wrapping the call
    // itself to catch anything being swallowed, and logging video readiness
    // once in a while in case detectForVideo is silently seeing a frame it
    // doesn't like (e.g. a size it never gets from Holistic's own call).
    let handResult: HandLandmarkerResult | null = null
    try {
      handResult = handLandmarker.detectForVideo(video, now)
    } catch (err) {
      console.log(`[hand] detectForVideo threw: ${(err as Error).message}`)
    }
    const assignedHands = handResult ? assignHandsToSides(handResult, result.poseLandmarks[0]) : { left: null, right: null }
    if (now - lastHandDebugAt > 200) {
      lastHandDebugAt = now
      console.log(
        `[hand] rawCandidates=${handResult?.landmarks.length ?? 'n/a'} assignedLeft=${assignedHands.left ? 'y' : 'n'} assignedRight=${assignedHands.right ? 'y' : 'n'} videoSize=${video.videoWidth}x${video.videoHeight} readyState=${video.readyState}`
      )
    }
    drawResult(result, assignedHands, drawer)

    const data = trackingDataFromResult(result)
    // Holistic's own hand branch is pose-derived and goes blind whenever the
    // pose wrist drifts (see HAND_MODEL_URL) - overridden here with the
    // independent HandLandmarker's assignment instead.
    data.leftHandLandmarks = assignedHands.left?.landmarks
    data.rightHandLandmarks = assignedHands.right?.landmarks
    data.leftHandWorldLandmarks = assignedHands.left?.worldLandmarks
    data.rightHandWorldLandmarks = assignedHands.right?.worldLandmarks
    // Confidence-gated hold-last-good-pose + bone-length depth reconstruction,
    // shared by the actual rigging below and the debug skeleton.
    const aspect = video.videoWidth > 0 && video.videoHeight > 0 ? video.videoWidth / video.videoHeight : 16 / 9
    const resolved = resolveTrackingFrame(withImageXY(data, aspect), deltaSeconds)
    updateReadout(result, assignedHands, fps, resolved, calibration)

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

    scheduleNext(loop)
  }
  scheduleNext(loop)
}

main().catch((err) => {
  console.error(err)
  setStatus(`오류: ${(err as Error).message}`)
})
