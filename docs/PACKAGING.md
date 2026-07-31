# Phase 9: Packaging as a Windows .exe

I can't build a real Windows `.exe` for you from here — I'm running in a
Linux sandbox, and PyInstaller only cross-compiles for the OS it's
actually running on (a Linux box can't produce a working `.exe`). You'll
need to run this step yourself on a Windows machine, but it's just two
commands once you're there.

## 1. Set up on the Windows machine

```bat
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

## 2. Build

MediaPipe ships model files (hand landmark models etc.) as data files
that PyInstaller doesn't pick up automatically, so they must be added
explicitly with `--collect-data`. Same idea for your own `models/` folder
(the `.pkl` files) — `--add-data` bundles them into the exe.

For the GUI app (recommended as the shipped product):

```bat
pyinstaller --onefile --windowed ^
  --name ASLTranslator ^
  --collect-data mediapipe ^
  --add-data "models;models" ^
  backend\gui_app.py
```

For the OpenCV/CLI version instead:

```bat
pyinstaller --onefile --console ^
  --name ASLTranslator_CLI ^
  --collect-data mediapipe ^
  --add-data "models;models" ^
  backend\realtime_recognition.py
```

- `--windowed` hides the console window (use this for the GUI build so
  no black terminal pops up behind it).
- `--onefile` bundles everything into one `.exe`; drop it if you'd rather
  have a folder of files (faster startup, easier to debug if something's
  missing).
- The `;` in `--add-data "models;models"` is Windows syntax (it's `:` on
  macOS/Linux, in case you ever build there too).

The finished executable lands in `dist\ASLTranslator.exe`.

## 3. Common issues

- **"Could not find mediapipe data"** at runtime → you skipped
  `--collect-data mediapipe`, or the mediapipe version bundled differs
  from what's installed. Rebuild with the flag.
- **"Required file not found: models\svm_model.pkl"** → the `--add-data`
  path is wrong or wasn't included; check `dist\` for a `models` folder
  next to the exe (or inside it, if `--onefile`).
- **Antivirus flags the exe** → this happens a lot with PyInstaller
  single-file builds because of how they self-extract; it's a false
  positive but worth mentioning to whoever you send it to.
- **No microphone/camera permission dialog on first run** → Windows may
  silently block webcam access for an unsigned exe; check
  Settings → Privacy → Camera.

## 4. If you don't have a Windows machine handy

GitHub Actions can build it for you for free using a `windows-latest`
runner — a workflow that checks out the repo, installs requirements, and
runs the same `pyinstaller` command above, then uploads `dist\*.exe` as a
build artifact. Ask if you'd like that workflow file written out.
