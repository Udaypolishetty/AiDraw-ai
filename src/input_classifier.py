"""
InputClassifier: decides whether an air-canvas trace is handwriting or
an object drawing, by running BOTH recognizers and comparing confidence.

Deliberately simple and explicit about being a heuristic, not a
trained decision model — do not oversell its accuracy.
"""
from dataclasses import dataclass

import numpy as np

import config
from src.drawing_recognizer import DrawingRecognizer, DrawingResult
from src.handwriting_recognizer import HandwritingRecognizer, OCRResult


@dataclass
class ClassificationResult:
    input_type: str  # "HANDWRITING" | "DRAWING" | "UNKNOWN"
    label: str
    confidence: float
    ocr_result: OCRResult
    drawing_result: DrawingResult


class InputClassifier:
    def __init__(self):
        self._ocr = HandwritingRecognizer()
        self._drawing = DrawingRecognizer()

    def classify(self, canvas_bgr: np.ndarray) -> ClassificationResult:
        ocr_result = self._ocr.recognize(canvas_bgr)
        drawing_result = self._drawing.recognize(canvas_bgr)

        ocr_conf = ocr_result.confidence if ocr_result.text else 0.0
        drawing_conf = drawing_result.confidence

        # OCR confidence is a genuine uncertainty measure; CLIP confidence is
        # a forced-choice softmax over only 14 classes and is always inflated.
        # So a strongly confident OCR read is trusted outright rather than
        # compared numerically against CLIP's score.
        if ocr_conf >= config.OCR_TRUST_THRESHOLD:
            return ClassificationResult("HANDWRITING", ocr_result.text, ocr_conf,
                                         ocr_result, drawing_result)

        if drawing_conf >= config.MIN_CONFIDENCE_THRESHOLD:
            return ClassificationResult("DRAWING", drawing_result.label, drawing_conf,
                                         ocr_result, drawing_result)

        if ocr_conf >= config.MIN_CONFIDENCE_THRESHOLD:
            return ClassificationResult("HANDWRITING", ocr_result.text, ocr_conf,
                                         ocr_result, drawing_result)

        return ClassificationResult("UNKNOWN", "", 0.0, ocr_result, drawing_result)