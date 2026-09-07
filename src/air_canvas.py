"""
AirCanvas: stores drawn strokes and can render them either overlaid on
the live camera frame (what the user sees) or on a clean black canvas
(what later recognition phases will use).

Storing strokes as a list of point-lists (instead of a flat pixel canvas)
is what makes Undo trivial — we just drop the last stroke and re-render.
"""
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from src.one_euro_filter import PointFilter

Point = Tuple[int, int]
Stroke = List[Point]


class AirCanvas:
    def __init__(self, width: int = config.CANVAS_WIDTH,
                 height: int = config.CANVAS_HEIGHT) -> None:
        self._width = width
        self._height = height
        self._strokes: List[Stroke] = []
        self._current_stroke: Stroke = []
        self._point_filter: Optional[PointFilter] = None

    def update(self, fingertip: Point, pen_down: bool) -> None:
        if not pen_down:
            self._end_current_stroke()
            return

        if self._point_filter is None:
            self._point_filter = PointFilter()

        smoothed = self._point_filter.filter(fingertip)

        if self._current_stroke:
            last = self._current_stroke[-1]
            distance = math.hypot(smoothed[0] - last[0], smoothed[1] - last[1])
            if distance < config.MIN_DRAW_DISTANCE:
                return
            if distance > config.MAX_JUMP_DISTANCE:
                # Glitch jump — end this stroke, start a new one at the new point.
                self._end_current_stroke()
                self._current_stroke.append(smoothed)
                return

        self._current_stroke.append(smoothed)

    def _end_current_stroke(self) -> None:
        if len(self._current_stroke) > 1:
            self._strokes.append(self._current_stroke)
        self._current_stroke = []
        self._point_filter = None

    def undo(self) -> None:
        """Removes the most recently completed stroke."""
        if self._current_stroke:
            self._current_stroke = []
            self._point_filter = None
        elif self._strokes:
            self._strokes.pop()

    def clear(self) -> None:
        self._strokes = []
        self._current_stroke = []
        self._point_filter = None

    def has_content(self) -> bool:
        return bool(self._strokes) or len(self._current_stroke) > 1

    def _draw_strokes(self, image: np.ndarray, color: Tuple[int, int, int]) -> None:
        all_strokes = self._strokes + ([self._current_stroke] if self._current_stroke else [])
        for stroke in all_strokes:
            for i in range(1, len(stroke)):
                cv2.line(image, stroke[i - 1], stroke[i], color, config.STROKE_THICKNESS)

    def render_overlay(self, camera_frame: np.ndarray) -> np.ndarray:
        """Draws strokes directly on top of the live camera frame."""
        overlay = camera_frame.copy()
        self._draw_strokes(overlay, config.STROKE_COLOR)
        return overlay

    def get_canvas(self) -> np.ndarray:
        """Clean black-and-white canvas — what recognition phases will use."""
        canvas = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        canvas[:] = config.CANVAS_BACKGROUND_COLOR
        self._draw_strokes(canvas, (255, 255, 255))
        return canvas