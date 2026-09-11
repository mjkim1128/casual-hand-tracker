import {
  FilesetResolver,
  HolisticLandmarker,
  DrawingUtils,
  type HolisticLandmarkerResult
} from '@mediapipe/tasks-vision'

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task'

const video = document.getElementById('webcam') as HTMLVideoElement
const canvas = document.getElementById('overlay') as HTMLCanvasElement
const statusEl = document.getElementById('status') as HTMLElement
const readoutEl = document.getElementById('readout') as HTMLUListElement
const ctx = canvas.getContext('2d')!

function setStatus(text: string): void {
  statusEl.textContent = text
}

async function setupWebcam(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 800, height: 600 },
    audio: false
  })
  video.srcObject = stream
  await new Promise<void>((resolve) => {
    video.onloadedmetadata = () => resolve()
  })
  await video.play()
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
}

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

  for (const landmarks of result.poseLandmarks) {
    drawer.drawConnectors(landmarks, HolisticLandmarker.POSE_CONNECTIONS, {
      color: '#00d0ff',
      lineWidth: 3
    })
    drawer.drawLandmarks(landmarks, { color: '#00d0ff', radius: 3 })
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

function updateReadout(result: HolisticLandmarkerResult, fps: number): void {
  readoutEl.innerHTML = ''
  const lines = [
    `fps: ${fps.toFixed(1)}`,
    `pose landmarks: ${result.poseLandmarks[0]?.length ?? 0}`,
    `left hand landmarks: ${result.leftHandLandmarks[0]?.length ?? 0}`,
    `right hand landmarks: ${result.rightHandLandmarks[0]?.length ?? 0}`
  ]
  for (const line of lines) {
    const li = document.createElement('li')
    li.textContent = line
    readoutEl.appendChild(li)
  }
}

async function main(): Promise<void> {
  setStatus('웹캠 요청 중...')
  await setupWebcam()

  setStatus('MediaPipe 모델 로딩 중...')
  const holistic = await setupHolistic()
  const drawer = new DrawingUtils(ctx)

  setStatus('추적 중')

  let lastFrameTime = performance.now()

  const loop = (): void => {
    const now = performance.now()
    const fps = 1000 / (now - lastFrameTime)
    lastFrameTime = now

    const result = holistic.detectForVideo(video, now)
    drawResult(result, drawer)
    updateReadout(result, fps)

    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}

main().catch((err) => {
  console.error(err)
  setStatus(`오류: ${(err as Error).message}`)
})
