"""
DrawingRecognizer: classifies an air-drawn sketch against a fixed set of
known object classes using CLIP in zero-shot mode — no training needed.

AI/ML component. Kept behind a fixed recognize() -> DrawingResult
interface so this could be swapped for a dedicated sketch classifier later.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

KNOWN_CLASSES = [
    "sun", "cat", "dog", "car", "tree", "house", "apple",
    "star", "cloud", "flower", "heart", "bird", "fish", "ball",
]


@dataclass
class DrawingResult:
    label: str
    confidence: float  # 0-100


class DrawingRecognizer:
    def __init__(self, classes: Optional[List[str]] = None):
        self._classes = classes or KNOWN_CLASSES
        # Downloads pretrained CLIP weights to the HF cache the first time.
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._prompts = [f"a simple black and white sketch of a {c}" for c in self._classes]

    def _prepare_image(self, canvas_bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def recognize(self, canvas_bgr: np.ndarray) -> DrawingResult:
        image = self._prepare_image(canvas_bgr)
        inputs = self._processor(text=self._prompts, images=image,
                                  return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        best_idx = int(torch.argmax(probs))
        return DrawingResult(label=self._classes[best_idx].upper(),
                              confidence=round(float(probs[best_idx]) * 100, 1))