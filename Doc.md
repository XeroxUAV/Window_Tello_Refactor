me
📘 README.md — Complete GitHub Setup & Push Guide
Copy everything below into a file named README.md at the root of your project.
Then follow the steps inside to get your repo pushed to GitHub.

markdown
# 🪟 Window PASS — Tello + YOLO Autonomous Window Traversal

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-red?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/Drone-DJI%20Tello-black?style=for-the-badge" alt="Tello">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>An autonomous vision-based system that guides a DJI Tello drone through a window using YOLO segmentation, PID alignment, and a finite-state control pipeline.</b>
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [State Machine](#-state-machine)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [Control Theory](#-control-theory)
- [Tuning Guide](#-tuning-guide)
- [Safety](#-safety)
- [Troubleshooting](#-troubleshooting)
- [Testing](#-testing)
- [GitHub Setup & Push Guide](#-github-setup--push-guide)
- [Troubleshooting Git Push Errors](#-troubleshooting-git-push-errors)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

**Window PASS** is an autonomous drone system that:

1. **Searches** 360° for a window using a spinning + ascending scan pattern.
2. **Aligns** the drone using a **decoupled Yaw + Roll** controller (geometry-based, not just center offsets).
3. **Centers vertically** using a throttle PID on the camera's optical-axis offset.
4. **Approaches** the window while maintaining alignment.
5. **Passes** through it by detecting the characteristic area *increase → drop* signature.
6. **Lands** cleanly after a `DONE` state.

The system runs entirely onboard a laptop connected to the Tello over Wi-Fi — no companion computer required.

---

## 🏗 System Architecture
┌──────────────────────────────────────────────────────────────┐
│ WindowPassApp │
│ (app.py / main loop) │
└──────┬───────────────┬──────────────┬──────────────┬─────────┘
│ │ │ │
▼ ▼ ▼ ▼
┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
│ Detector │ │ Aligner │ │ HUD │ │ Drone │
│ (YOLO) │──▶│ (FSM+PID) │─▶│ (draw) │ │ (Tello) │
└──────────┘ └────────────┘ └──────────┘ └──────────┘
▲ ▲
│ │
┌─────────┐ ┌──────────┐
│geometry │ │ PID │
└─────────┘ └──────────┘

text

| Module | Responsibility |
|---|---|
| `config.py` | All tunable parameters as nested `@dataclass` trees |
| `pid.py` | Generic PID with integral clamping + output smoothing |
| `geometry.py` | Pure functions: corner ordering, mask → quad, error math |
| `detector.py` | YOLO wrapper → returns a typed `Detection` object |
| `aligner.py` | Finite-state machine that produces `AlignCommand` |
| `hud.py` | All `cv2` drawing in one class |
| `drone.py` | Tello wrapper: connect, RC, hover-kick, land, shutdown |
| `app.py` | Wires everything together + main loop |
| `main.py` | Entry point |

---

## 🔄 State Machine

```mermaid
stateDiagram-v2
    [*] --> SEARCH_360
    SEARCH_360 --> SEARCH_UP : 360° complete
    SEARCH_UP --> SEARCH_360 : timeout (restart)
    SEARCH_360 --> YAW_ROLL : window detected
    SEARCH_UP --> YAW_ROLL : window detected
    YAW_ROLL --> THROTTLE : yaw+roll aligned
    THROTTLE --> APPROACH : throttle centered
    APPROACH --> PASS : area_ratio ≥ 0.35 or timeout
    PASS --> DONE : pass detected or timeout
    DONE --> [*]

    YAW_ROLL --> LOST : no detection
    THROTTLE --> LOST : no detection
    APPROACH --> LOST : no detection
    LOST --> SEARCH_360 : > MAX_LOST_FRAMES
    LOST --> YAW_ROLL : reacquired
Stage	What happens	Exit condition
SEARCH 360	Rotate yaw at fixed speed, alternating direction each cycle	Timer expires → SEARCH UP
SEARCH UP	Ascend briefly, then resume 360	Timer expires or window found
YAW+ROLL	Simultaneous yaw (corner-length asymmetry) + roll (top-edge tilt) alignment	Both errors < threshold for stable_count frames
THROTTLE	Vertical centering on camera's optical axis	Error < THROTTLE_THRESHOLD
APPROACH	Fly forward while re-centering	Area ratio ≥ AREA_RATIO_THRESHOLD or timeout
PASS	Fly straight through; detect area peak→drop	Pass signature or timeout
DONE	Stop, land	Terminal
LOST	Track missing detection for MAX_LOST_FRAMES	Reacquire or fall to SEARCH
✨ Features
✅ Decoupled Yaw & Roll — Yaw uses corner-length asymmetry, Roll uses top-edge tilt angle (degrees). No cross-coupling.

✅ Non-blocking control loop — No time.sleep() inside the loop; hover kick is a one-shot startup routine.

✅ Configurable PID smoothing — Low-pass output filter to reduce jitter (PID_OUTPUT_SMOOTHING).

✅ Integral anti-windup — All PID integrators are clamped.

✅ Real LOST handling — Missed detections feed a counter that triggers SEARCH after a configurable budget.

✅ Mask-area enforcement — Detections under MIN_MASK_AREA are discarded.

✅ Camera optical-axis offset — Corrects for the Tello camera being physically above the frame center.

✅ Half-precision on CUDA — Automatic FP16 when GPU is available.

✅ Modular & testable — Every component is independently unit-testable.

✅ Emergency stop — Press e to abort, q to quit, r to reset the state machine.

✅ Battery safety — Refuses to take off below MIN_BATTERY.

📁 Project Structure
text
window_tello/
├── __init__.py
├── main.py            # Entry point
├── app.py             # WindowPassApp — main loop
├── config.py          # AppConfig dataclasses
├── pid.py             # PID controller
├── geometry.py        # Pure geometry helpers
├── detector.py        # YOLO wrapper
├── aligner.py         # FSM + PID coordination
├── hud.py             # OpenCV drawing
└── drone.py           # Tello wrapper
🛠 Installation
Prerequisites
Python 3.9 – 3.11

DJI Tello drone (or Tello EDU)

Laptop with Wi-Fi (must be able to connect to Tello's AP)

(Optional) NVIDIA GPU + CUDA for faster inference

Steps
bash
# 1. Clone
git clone https://github.com/XeroxUAV/Window_Tello.git
cd Window_Tello

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
requirements.txt
txt
ultralytics>=8.0.0
djitellopy>=2.5.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
keyboard>=0.13.5
.gitignore
gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/
.pytest_cache/
.mypy_cache/

# Models & data (too large for Git)
*.pt
*.onnx
*.pth
weights/
runs/
datasets/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp

# Logs
*.log
Trained model
Place your YOLO segmentation weights at the path specified in config.py:

python
MODEL_PATH = r"D:\Window_Tello\best.pt"
The model must be trained with instance segmentation (masks, not just boxes) — the roll/yaw geometry depends on polygon corners.

⚠️ Do not commit best.pt — it's in .gitignore. Host it via GitHub Releases instead (see GitHub Setup).

⚙️ Configuration
All parameters live in config.py as nested dataclasses. Override programmatically:

python
from window_tello import AppConfig
from window_tello.app import WindowPassApp

cfg = AppConfig()
cfg.model.path = "models/window_v2.pt"
cfg.yaw.kp = 0.5
cfg.roll.enabled = True
cfg.passing.forward_speed = 50

app = WindowPassApp(cfg)
app.run()
Parameter reference (highlights)
Parameter	Default	Description
model.confidence	0.5	YOLO confidence threshold
model.inference_size	640	YOLO input resolution
yaw.kp / ki / kd	0.4 / 0 / 0	Yaw PID gains (error in %)
roll.kp / ki / kd	1.5 / 0 / 0	Roll PID gains (error in degrees)
throttle.kp / ki / kd	0.4 / 0.05 / 0	Throttle PID gains (error in px)
approach.speed	12	Forward RC command during approach
passing.forward_speed	40	Forward RC command during pass
passing.extra_time	1.25 s	Extra forward push after pass detected
camera.offset_y	130	Optical axis offset from frame center (px)
runtime.min_mask_area	2000	Ignore masks smaller than this
runtime.max_lost_frames	25	Frames before LOST → SEARCH
🚀 Usage
1. Power on the Tello
Turn on the drone.

Connect your laptop to the Tello's Wi-Fi AP (TELLO-XXXXXX).

2. Run
bash
python -m window_tello.main
3. Keyboard controls
Key	Action
q	Quit gracefully (lands)
r	Reset the state machine to YAW
e	Emergency stop (lands)
4. Dry-run (no flight)
Set cfg.runtime.flight_mode = False to test the vision + FSM pipeline without sending RC commands.

🧮 Control Theory
Yaw error — corner-length asymmetry
Given the ordered quad [TL, TR, BR, BL]:

text
left_len  = ‖BL − TL‖
right_len = ‖BR − TR‖
yaw_err   = 100 · (right_len − left_len) / avg_len
Zero when the drone faces the window head-on.

Roll error — top-edge tilt
text
roll_err_deg = degrees(atan2(TR_y − TL_y, TR_x − TL_x))
Normalized to [-90°, +90°]. Zero when the window's top edge is horizontal.

Throttle error — camera offset
text
thr_err = window_center_y − camera_optical_axis_y
Positive error → window is below axis → drone must ascend.

Pass detection
Growth phase — area > previous_area × 1.05

Drop phase — area < initial_area × 0.45 OR area is None

Extra push — fly forward for PASS_EXTRA_TIME seconds.

🎛 Tuning Guide
Symptom	Fix
Drone oscillates around yaw target	↓ yaw.kp, ↑ yaw.deadzone
Drone rolls the wrong way	roll.invert = True
Drone yaws the wrong way	yaw.geometry_invert = True
Never reaches throttle center	↓ throttle.threshold or ↓ camera.offset_y
Pass triggers too early	↑ passing.area_drop
Pass never triggers	↓ passing.area_drop
RC commands ignored by Tello	↑ min_speeds.* values
Jittery RC output	↑ runtime.pid_output_smoothing
Slow to reacquire after loss	↓ runtime.max_lost_frames
🛡 Safety
⚠️ Read before flying.

Propellers spin on takeoff. Keep hands, hair, and cables clear.

Fly in an open, indoor space with soft flooring for early tests.

Battery threshold — The app refuses to fly below MIN_BATTERY (default 20%).

Emergency stop — Hold e at any time to abort and land.

Wi-Fi interference — Keep the laptop within ~10 m.

Outdoor flight — NOT supported. Tello has no GPS.

Pass phase is irreversible — Ensure the far side of the window is clear.

🩺 Troubleshooting
Problem	Likely cause	Solution
Tello not found	Not connected to Tello Wi-Fi	Reconnect; check TELLO-XXXX AP
Frame is None	Stream not started	Ensure streamon() succeeded
YOLO warm-up skipped	CUDA not available	Fine on CPU, just slower
[SAFETY] Battery low	Battery < 20%	Charge the drone
Land error	Connection dropped mid-flight	Tello will auto-land on signal loss
Roll never centers	Bad corner order from mask	Check MIN_MASK_AREA; retrain model
Constant LOST	Model not detecting window	Lower confidence to 0.3 temporarily
🧪 Testing
bash
pytest tests/test_geometry.py
pytest tests/test_pid.py
pytest tests/test_aligner.py
Example:

python
from window_tello.geometry import order_quad_corners
import numpy as np

def test_order_quad():
    pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]])
    ordered = order_quad_corners(pts)
    assert np.allclose(ordered[0], [10, 10])   # TL
    assert np.allclose(ordered[2], [100, 100]) # BR
🚀 GitHub Setup & Push Guide
This section walks you through pushing your code to GitHub from scratch.

Step 1: Prepare your local folder
Create this structure on your computer:

text
Window_Tello/
├── README.md              ← this file
├── requirements.txt
├── LICENSE
├── .gitignore
├── window_tello/          ← your refactored code
│   ├── __init__.py
│   ├── main.py
│   ├── app.py
│   ├── config.py
│   ├── pid.py
│   ├── geometry.py
│   ├── detector.py
│   ├── aligner.py
│   ├── hud.py
│   └── drone.py
└── tests/
    └── test_geometry.py
Step 2: Create required auxiliary files
requirements.txt — see Installation section above.

.gitignore — see Installation section above.

LICENSE — if you chose MIT, create LICENSE with the standard MIT text. Generate one at choosealicense.com.

Step 3: Initialize Git locally
Open a terminal in the Window_Tello/ folder:

bash
git init
git add .
git commit -m "Initial commit: Window PASS autonomous drone system"
git branch -M main
Step 4: Create the repo on GitHub
Go to https://github.com/new

Repository name: Window_Tello

Description: Autonomous DJI Tello drone that flies through windows using YOLO segmentation + PID control

Visibility: Public (or Private)

❗ DO NOT check "Add a README", "Add .gitignore", or "Choose a license" — you already have them locally.

Click Create repository.

Step 5: Push
bash
git remote add origin https://github.com/XeroxUAV/Window_Tello.git
git push -u origin main
Done! Refresh the GitHub page — your README renders automatically. 🎉

Adding the model via GitHub Releases
Since best.pt is in .gitignore (GitHub rejects files > 100 MB):

Go to your repo → Releases → Draft a new release

Tag: v0.1.0, Title: Initial release

Upload best.pt as a release asset.

Add to README:

markdown
## 📥 Model Weights
Download `best.pt` from the [latest release](https://github.com/XeroxUAV/Window_Tello/releases/latest).
🔧 Troubleshooting Git Push Errors
❌ ! [rejected] master -> master (fetch first)
Cause: The GitHub repo already has commits (usually an auto-generated README) that your local repo doesn't have.

Fix (recommended — your local wins):

bash
cd /path/to/Window_Tello
git branch -M main
git push -u origin main --force-with-lease
If --force-with-lease fails, use --force:

bash
git push -u origin main --force
Then on GitHub:

Settings → Branches → Default branch → main

Delete the old master branch if it exists

Safer alternative (keep remote's work):

bash
git fetch origin
git pull --rebase origin master
git push origin master
Resolve any conflicts (e.g., duplicate README.md) and continue.

Backup remote first (if worried about losing data):

bash
git fetch origin
git branch remote-backup origin/main
❌ fatal: remote origin already exists
bash
git remote set-url origin https://github.com/XeroxUAV/Window_Tello.git
❌ remote: Support for password authentication was removed
Use a Personal Access Token (PAT) instead of your password:

GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

Generate new token with repo scope.

When Git asks for a password, paste the token.

Or switch to SSH:

bash
git remote set-url origin git@github.com:XeroxUAV/Window_Tello.git
❌ file is 123.45 MB; this exceeds GitHub's file size limit of 100.00 MB
You accidentally committed a model. Remove it:

bash
git rm --cached path/to/best.pt
echo "*.pt" >> .gitignore
git add .gitignore
git commit -m "Remove large model, add to gitignore"
git push
If it's already in history, use git-filter-repo or BFG Repo-Cleaner.

❌ Updates were rejected because the tip of your current branch is behind
Same as the "fetch first" case — remote is ahead of you.

bash
git pull --rebase origin main
git push origin main
Or if you're sure your local is correct:

bash
git push origin main --force-with-lease
❌ error: failed to push some refs
Generic catch-all. Usually one of the above. Run:

bash
git fetch origin
git status
To see what's different, then pick the appropriate fix.

🧠 Quick Reference Table
Situation	Command
Remote has stuff you want to keep	git pull --rebase origin main
Remote has junk, overwrite it	git push origin main --force-with-lease
Unrelated histories, merge them	git pull origin main --allow-unrelated-histories
Renamed to main + want clean slate	git push -u origin main --force
Scared to lose remote data	git branch remote-backup origin/main first
Wrong remote URL	git remote set-url origin <new-url>
🗺 Roadmap
□ YAML config loading
□ Web dashboard (Flask + WebSocket)
□ Onboard Raspberry Pi deployment
□ Kalman filter for smoother corner tracking
□ Multiple windows / door detection
□ ROS 2 node wrapper
□ Automated pass-time measurement (for racing)
📄 License
MIT License — see LICENSE for details.

🙏 Acknowledgements
Ultralytics YOLOv8 — segmentation backbone

djitellopy — Tello SDK wrapper

OpenCV — vision primitives

DJI Tello community for SDK documentation

<p align="center"> <i>Built for autonomous indoor flight. Fly safe. 🛩️</i> </p> ```
✅ How To Use This File
Option 1 — Use it as your actual README.md
Copy everything inside the code block above (from # 🪟 Window PASS... to the final </p>).

Save it as README.md at the root of Window_Tello/.

Push to GitHub — it renders automatically.

Option 2 — Keep it as DOCS.md (no homepage render)
Same copy/paste as above, but save as DOCS.md.

Delete any existing README.md:

bash
rm -f README.md
git rm --cached README.md 2>/dev/null || true
Push. The docs stay in the repo but won't clutter the homepage.

Option 3 — Split into multiple docs
If you prefer separation:

File	Content
README.md	Overview + Installation + Usage
DOCS.md	Architecture + State Machine + Control Theory
TUNING.md	Tuning guide + PID reference
GIT.md	GitHub setup + push troubleshooting
🚀 Push It Now
After saving the file:

bash
cd /path/to/Window_Tello

# Stage everything
git add .

# Commit
git commit -m "docs: add README with full setup and GitHub guide"

# Rename branch to main
git branch -M main

# Push (force if remote already has a README)
git push -u origin main --force-with-lease
