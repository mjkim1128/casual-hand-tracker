import { app, BrowserWindow, session } from 'electron'
import { join } from 'path'

const LEVEL_NAMES = ['LOG', 'WARN', 'ERROR']

// Renderer console.* calls (and uncaught errors) otherwise only show up in
// that window's own DevTools — mirrored to this process's stdout too so
// they show up in the same terminal running `npm run dev`, useful for
// debugging without needing to open DevTools separately.
function mirrorConsole(window: BrowserWindow, label: string): void {
  window.webContents.on('console-message', (_event, level, message, line, sourceId) => {
    console.log(`[${label}:${LEVEL_NAMES[level] ?? level}] ${message} (${sourceId}:${line})`)
  })
}

function loadPage(window: BrowserWindow, htmlFile: string): void {
  if (!app.isPackaged && process.env['ELECTRON_RENDERER_URL']) {
    window.loadURL(`${process.env['ELECTRON_RENDERER_URL']}/${htmlFile}`)
  } else {
    window.loadFile(join(__dirname, `../renderer/${htmlFile}`))
  }
}

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1500,
    height: 980,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })
  mirrorConsole(mainWindow, 'renderer')
  loadPage(mainWindow, 'index.html')
}

app.whenReady().then(() => {
  // The renderer needs the OS webcam permission prompt (getUserMedia); grant it
  // automatically instead of leaving Electron's default (silent deny in packaged apps).
  //
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(permission === 'media')
  })

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
        }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
