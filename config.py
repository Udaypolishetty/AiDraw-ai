"""
Central configuration for AirDraw AI.
Keeping all tunable values here means no hardcoded magic numbers
scattered across the codebase — required per project guidelines.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HAND_LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

# --- Webcam ---
CAMERA_INDEX = 0          # change to 1 if you have multiple cameras
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Hand tracking ---
MAX_NUM_HANDS = 1          # AirDraw only needs one drawing hand
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# MediaPipe hand landmark indices (from the official 21-point hand model)
WRIST = 0
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6

# --- Air Canvas (Phase 2) ---
CANVAS_WIDTH = FRAME_WIDTH
CANVAS_HEIGHT = FRAME_HEIGHT
CANVAS_BACKGROUND_COLOR = (0, 0, 0)      # black background, BGR
STROKE_COLOR = (0, 255, 255)           # white ink, BGR
STROKE_THICKNESS = 6

# Exponential smoothing factor for fingertip position (0-1).
# Lower = smoother but laggier, higher = snappier but jittery.
SMOOTHING_ALPHA = 0.5

# --- Air Canvas noise control (Phase 2 fix) ---
MIN_DRAW_DISTANCE = 3       # ignore movement smaller than this (pixels) — treat as jitter
MAX_JUMP_DISTANCE = 120     # if fingertip jumps farther than this in one frame, don't connect it

# --- Recognition (Phase 4/5/6) ---
MIN_CONFIDENCE_THRESHOLD = 40.0    # below this for BOTH engines -> treat as unrecognized
OCR_PREFERENCE_MARGIN = 10.0       # OCR must beat drawing confidence by this much to "win"

OCR_TRUST_THRESHOLD = 65.0   # if OCR is this confident, trust it outright — don't compare to CLIP


# --- Image Provider (Phase 7) ---
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "assets", "sample_images")
IMAGE_GENERATION_TIMEOUT_SECONDS = 20
RESULT_PANEL_SIZE = 512