import { resolve } from 'path'
import { defineConfig } from 'electron-vite'

export default defineConfig({
  main: {
    build: {
      rollupOptions: {
        input: resolve(__dirname, 'src/main/index.ts')
      }
    }
  },
  preload: {
    build: {
      rollupOptions: {
        input: resolve(__dirname, 'src/preload/index.ts')
      }
    }
  },
  renderer: {
    root: resolve(__dirname, 'src/renderer'),
    build: {
      rollupOptions: {
        input: resolve(__dirname, 'src/renderer/index.html')
      }
    },
    // onnxruntime-web (used by @huggingface/transformers) dynamically
    // imports its own .mjs WASM loader at runtime — Vite's dependency
    // pre-bundler doesn't carry that file along correctly, producing a
    // "file does not exist in the optimize deps directory" error. Excluding
    // it makes Vite serve it straight from node_modules instead, where the
    // dynamic import resolves correctly.
    optimizeDeps: {
      exclude: ['onnxruntime-web', '@huggingface/transformers']
    }
  }
})
