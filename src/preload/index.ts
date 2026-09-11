import { contextBridge } from 'electron'

// Placeholder bridge — the calibration UI and VMC start/stop controls (stage 3+ of
// the build plan) will hang IPC calls off `window.api` here.
const api = {}

contextBridge.exposeInMainWorld('api', api)
