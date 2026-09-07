"""
One Euro Filter: reduces jitter on slow movements while staying
responsive on fast movements — the standard filter for pen/stylus-style
tracking. Reference: Casiez et al., 'The One Euro Filter' (2012).
"""
import math


class LowPassFilter:
    def __init__(self):
        self._initialized = False
        self._value = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value = alpha * value + (1 - alpha) * self._value
        return self._value


class OneEuroFilter:
    """Filters a single scalar value. Use two instances for (x, y)."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.3, d_cutoff: float = 1.0):
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_value = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self._freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, value: float) -> float:
        if self._last_value is None:
            dx = 0.0
        else:
            dx = (value - self._last_value) * self._freq
        self._last_value = value

        edx = self._dx_filter.filter(dx, self._alpha(self._d_cutoff))
        cutoff = self._min_cutoff + self._beta * abs(edx)
        return self._x_filter.filter(value, self._alpha(cutoff))


class PointFilter:
    """Convenience wrapper: filters an (x, y) point using two OneEuroFilters."""

    def __init__(self):
        self._fx = OneEuroFilter()
        self._fy = OneEuroFilter()

    def filter(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        return int(self._fx.filter(x)), int(self._fy.filter(y))