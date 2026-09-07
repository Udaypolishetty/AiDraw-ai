"""
HandTracker wraps MediaPipe's HandLandmarker (Tasks API).

Computer Vision / AI/ML component.
This module is intentionally the ONLY place that knows about MediaPipe
internals, so later phases (air_canvas.py, gesture_controller.py) just
call simple methods like get_index_fingertip() without caring how
detection actually happens.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


@dataclass
class HandLandmarks:
    """Holds one detected hand's landmarks in pixel coordinates."""
    points: List[Tuple[int, int]]  # 21 (x, y) points in pixel space


class HandTracker:
    """Detects a hand in a video frame and exposes fingertip helpers."""

    def __init__(self) -> None:
        if not os.path.exists(config.HAND_LANDMARKER_MODEL_PATH):
            raise FileNotFoundError(
                f"Hand landmarker model not found at "
                f"{config.HAND_LANDMARKER_MODEL_PATH}.\n"
                "Download it first — see the README setup steps."
            )

        base_options = mp_python.BaseOptions(
            model_asset_path=config.HAND_LANDMARKER_MODEL_PATH
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> Optional[HandLandmarks]:
        """
        Runs hand detection on a single BGR frame (as OpenCV gives us).
        Returns pixel-space landmarks for the first detected hand, or None.
        """
        h, w, _ = frame_bgr.shape

        frame_rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.hand_landmarks:
            return None

        first_hand = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in first_hand]
        return HandLandmarks(points=points)

    @staticmethod
    def get_index_fingertip(hand: HandLandmarks) -> Tuple[int, int]:
        """Returns the (x, y) pixel position of the index fingertip."""
        return hand.points[config.INDEX_FINGER_TIP]

    @staticmethod
    def is_index_finger_up(hand: HandLandmarks) -> bool:
        """
        True only when the hand is actually 'pointing': index finger
        extended AND middle/ring/pinky curled down. This is stricter than
        just checking the index tip, so incidental hand movement doesn't
        get mistaken for an intentional draw gesture.
        """
        def finger_extended(tip_idx: int, pip_idx: int) -> bool:
            return hand.points[tip_idx][1] < hand.points[pip_idx][1]

        index_up = finger_extended(8, 6)
        middle_down = not finger_extended(12, 10)
        ring_down = not finger_extended(16, 14)
        pinky_down = not finger_extended(20, 18)

        return index_up and middle_down and ring_down and pinky_down

    def close(self) -> None:
        self._landmarker.close()