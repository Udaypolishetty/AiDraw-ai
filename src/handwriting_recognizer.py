"""
HandwritingRecognizer: turns an air-canvas trace into a recognized word
using EasyOCR, a pretrained OCR model.

OCR component. Kept behind a fixed recognize() -> OCRResult interface
so the underlying engine can be swapped later without touching callers.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import easyocr
import numpy as np


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0-100


class HandwritingRecognizer:
    def __init__(self, languages: Optional[List[str]] = None):
        # Downloads the pretrained model to ~/.EasyOCR the first time.
        self._reader = easyocr.Reader(languages or ["en"], gpu=False)

    def _prepare_image(self, canvas_bgr: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)

        # Crop to the bounding box of the ink so OCR isn't scanning mostly-blank canvas.
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        pad = 20
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        cropped = gray[y0:y1, x0:x1]

        # OCR expects dark text on a light background; our canvas is
        # white ink on black, so invert it.
        return cv2.bitwise_not(cropped)

    def recognize(self, canvas_bgr: np.ndarray) -> OCRResult:
        image = self._prepare_image(canvas_bgr)
        if image is None:
            return OCRResult(text="", confidence=0.0)

        results = self._reader.readtext(image, detail=1, paragraph=False)
        if not results:
            return OCRResult(text="", confidence=0.0)

        best = max(results, key=lambda r: r[2])  # r = (bbox, text, confidence)
        return OCRResult(text=best[1].strip().upper(), confidence=round(best[2] * 100, 1))