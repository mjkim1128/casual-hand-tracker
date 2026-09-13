import type { DepthWorkerInMessage, DepthWorkerOutMessage } from './depthWorker'

export interface DepthFrame {
  width: number
  height: number
  data: Float32Array // raw model output units — NOT metric, and NOT normalized per-frame (unlike the `depth` visualization image the pipeline also returns) — see calibrateExternalDepth in main.ts for turning this into an actual depth estimate.
}

export interface DepthWorkerHandle {
  /** Human-readable status for the readout — see main.ts's depthModelStatus usage. */
  getStatus(): string
  getLatestFrame(): DepthFrame | null
  /**
   * Grabs the current video frame and sends it to the worker, unless a
   * previous frame is still being processed or the model isn't ready yet —
   * call this once per render-loop tick and it self-throttles to whatever
   * rate inference actually keeps up with.
   */
  requestFrame(video: HTMLVideoElement): void
}

// Depth Anything V2 (Small, Apache-2.0 — see public/models/ATTRIBUTION.md).
// A monocular depth model run directly on the webcam frame, independent of
// MediaPipe's own pose Z estimate (a known weak point) — supplements it
// with a second, image-based depth signal.
//
// Runs on a dedicated Worker rather than the main thread: the model's WASM
// inference is synchronous CPU work taking a few hundred ms per frame, and
// running that inline with the main render/tracking loop starved
// requestAnimationFrame, making pose/hand tracking itself look slow even
// though MediaPipe's own detection speed was unaffected.
export function startDepthWorker(): DepthWorkerHandle {
  const worker = new Worker(new URL('./depthWorker.ts', import.meta.url), { type: 'module' })

  let status = '로딩 중'
  let latestFrame: DepthFrame | null = null
  let inFlight = false

  worker.onmessage = (event: MessageEvent<DepthWorkerOutMessage>) => {
    const message = event.data
    if (message.type === 'ready') {
      status = '준비됨'
      console.log('depth model ready')
    } else if (message.type === 'error') {
      status = `로드 실패: ${message.message}`
      inFlight = false
      console.error('depth worker error', message.message)
    } else {
      latestFrame = { width: message.width, height: message.height, data: message.data }
      inFlight = false
    }
  }
  worker.onerror = (event: ErrorEvent) => {
    status = `로드 실패: ${event.message}`
    inFlight = false
    console.error('depth worker crashed', event.message)
  }

  return {
    getStatus: () => status,
    getLatestFrame: () => latestFrame,
    requestFrame: (video) => {
      if (inFlight || status !== '준비됨') return
      if (video.videoWidth === 0 || video.videoHeight === 0) return
      inFlight = true
      createImageBitmap(video)
        .then((bitmap) => {
          const message: DepthWorkerInMessage = { type: 'frame', bitmap }
          worker.postMessage(message, [bitmap])
        })
        .catch(() => {
          inFlight = false
        })
    }
  }
}

// Nearest-neighbor is plenty — the depth map's own resolution is already
// far lower than the source video, so bilinear wouldn't meaningfully help.
// `normalizedX`/`normalizedY` use MediaPipe's own 0-1, un-mirrored
// image-space convention, matching `poseLandmarks`.
export function sampleDepth(frame: DepthFrame, normalizedX: number, normalizedY: number): number {
  const x = Math.min(frame.width - 1, Math.max(0, Math.round(normalizedX * frame.width)))
  const y = Math.min(frame.height - 1, Math.max(0, Math.round(normalizedY * frame.height)))
  return frame.data[y * frame.width + x]
}
