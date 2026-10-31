"""Windows platform layer.

The shell, injector, and hotkey are pure ctypes. Microphone capture
(`wasapi`) uses sounddevice, which the package already depends on, and
imports it only when a stream is opened.

- injector: SendInput keystrokes (Unicode packets; Enter/Tab as real keys)
- hotkey: a low-level keyboard hook (hold-to-talk, bare Right Ctrl)
- clipboard / selection: the native clipboard, and copy-the-selection
- shell: the overlay pill, the Kai orb, and the tray icon
- app: the windowless tray app that hosts the daemon
  (`pythonw -m voice_keyboard.windows`)
- wasapi: low-latency microphone capture. Shared mode is the default.
  Exclusive is opt-in and falls back to shared if the open fails.
  sounddevice is already a package dependency; this module imports it
  only when a stream is opened.

The injector/hotkey factories select these automatically on win32.
"""
