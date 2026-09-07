# AirDraw AI

Write or draw in the air using just your index finger — a webcam and computer
vision turn your fingertip into a pen, then AI figures out what you made and
shows you a matching image.

## How it works

1. Point your webcam at yourself and raise your index finger (other fingers curled).
2. "Write" a word (e.g. `SUN`) or sketch an object (e.g. a simple flower shape) in the air.
3. Press `s` — the app recognizes whether you wrote text or drew a shape, and
   fetches/generates a matching image right in the same window.

No mode-switching needed — the app decides between handwriting and drawing
recognition automatically.

## Features

- Real-time hand tracking via MediaPipe (HandLandmarker)
- Smooth, pen-like fingertip tracking using a One Euro Filter
- Air canvas overlaid live on the camera feed, with clear/undo
- Handwriting recognition (EasyOCR)
- Object drawing recognition (CLIP, zero-shot — no training required)
- Automatic classifier that decides handwriting vs. drawing based on confidence
- AI image generation for the recognized result, with a local-image fallback

## Tech stack

Python · OpenCV · MediaPipe · EasyOCR · CLIP (Transformers/PyTorch) · Pollinations.ai

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Download the hand landmark model:
```bash
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

Run it:
```bash
python app.py
```

First run downloads the OCR and CLIP models (~650MB total) — this only happens once.

## Controls

| Key | Action |
|-----|--------|
| Point (index up, others curled) | Draw |
| `c` | Clear canvas |
| `z` | Undo last stroke |
| `s` | Recognize + generate image |
| `q` | Quit |

## Project structure
