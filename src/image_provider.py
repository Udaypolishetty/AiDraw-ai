"""
ImageProvider abstraction: swap between local sample images, image
search, or image generation without changing recognition code.

Image Generation/Retrieval component.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import cv2
import numpy as np
import requests

import config


class ImageProvider(ABC):
    """Base interface: given a recognized label, return a BGR image or None."""

    @abstractmethod
    def get_image(self, label: str) -> Optional[np.ndarray]:
        ...


class LocalImageProvider(ImageProvider):
    """Loads a pre-supplied sample image matching the label. No network needed."""

    def __init__(self, folder: Optional[Path] = None):
        self._folder = folder or Path(config.SAMPLE_IMAGES_DIR)

    def get_image(self, label: str) -> Optional[np.ndarray]:
        label_lower = label.strip().lower()
        for ext in (".png", ".jpg", ".jpeg"):
            path = self._folder / f"{label_lower}{ext}"
            if path.exists():
                image = cv2.imread(str(path))
                if image is not None:
                    return image
        return None


class ImageGenerationProvider(ImageProvider):
    """
    Generates an image via Pollinations.ai's free public API — no API key
    required. Fails gracefully (returns None) on any network issue, so
    callers should always pair this with a LocalImageProvider fallback.
    """

    def __init__(self, timeout_seconds: int = config.IMAGE_GENERATION_TIMEOUT_SECONDS):
        self._timeout = timeout_seconds

    def get_image(self, label: str) -> Optional[np.ndarray]:
        prompt = f"a simple, colorful, clean illustration of a {label}, white background"
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        try:
            response = requests.get(
                url, timeout=self._timeout,
                params={"width": 512, "height": 512, "nologo": "true"},
            )
            response.raise_for_status()
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"WARNING: Image generation failed ({e}). Falling back to local images.")
            return None


class FallbackImageProvider(ImageProvider):
    """Tries providers in order, returns the first successful non-None result."""

    def __init__(self, providers: List[ImageProvider]):
        self._providers = providers

    def get_image(self, label: str) -> Optional[np.ndarray]:
        for provider in self._providers:
            image = provider.get_image(label)
            if image is not None:
                return image
        return None