/// <reference lib="webworker" />

// Runs the external depth model (see depthModel.ts) on a dedicated thread.
// The model's WASM inference is synchronous CPU work that takes a few
// hundred ms per frame — running it on the main thread stalled
// requestAnimationFrame, which made pose/hand tracking itself look slow
// even though MediaPipe's own detection was unaffected. Moving it here
// keeps that cost off the render/tracking loop entirely.

import { pipeline, env, RawImage, type Tensor } from '@huggingface/transformers'

env.allowRemoteModels = false
env.allowLocalModels = true
// Absolute from the origin root — a relative path resolves against this
// worker script's own URL (src/renderer/...), not the page's, and would
// point at the wrong place. The model files are served from public/models/
// at the site root regardless of which document/worker is asking.
env.localModelPath = '/models/'
env.backends.onnx.wasm!.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/'
env.backends.onnx.wasm!.numThreads = 1

type DepthEstimationPipeline = (image: RawImage) => Promise<{ predicted_depth: Tensor }>

let pipelinePromise: Promise<DepthEstimationPipeline> | null = null

function loadDepthModel(): Promise<DepthEstimationPipeline> {
  if (!pipelinePromise) {
    pipelinePromise = pipeline('depth-estimation', 'depth-anything-v2-small', {
      dtype: 'q8'
    }) as unknown as Promise<DepthEstimationPipeline>
  }
  return pipelinePromise
}

export type DepthWorkerOutMessage =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  | { type: 'depth'; width: number; height: number; data: Float32Array }

export interface DepthWorkerInMessage {
  type: 'frame'
  bitmap: ImageBitmap
}

loadDepthModel()
  .then(() => postMessage({ type: 'ready' } satisfies DepthWorkerOutMessage))
  .catch((err: unknown) => {
    postMessage({ type: 'error', message: (err as Error).message } satisfies DepthWorkerOutMessage)
  })

// Frames are submitted one at a time by depthModel.ts, which waits for a
// reply before sending the next — no queueing needed here.
self.onmessage = async (event: MessageEvent<DepthWorkerInMessage>) => {
  const { bitmap } = event.data
  try {
    const depthPipe = await loadDepthModel()
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(bitmap, 0, 0)

    const image = await RawImage.fromCanvas(canvas)
    const { predicted_depth } = await depthPipe(image)
    const dims = predicted_depth.dims
    const height = dims[dims.length - 2]
    const width = dims[dims.length - 1]
    // Copied out of the tensor's own buffer before transferring — that
    // buffer may be a view shared with other tensors, so transferring it
    // directly could hand ownership of memory still in use elsewhere.
    const data = new Float32Array(predicted_depth.data as Float32Array)

    const message: DepthWorkerOutMessage = { type: 'depth', width, height, data }
    ;(postMessage as (message: DepthWorkerOutMessage, transfer: Transferable[]) => void)(message, [data.buffer])
  } catch (err) {
    postMessage({ type: 'error', message: (err as Error).message } satisfies DepthWorkerOutMessage)
  } finally {
    bitmap.close()
  }
}
