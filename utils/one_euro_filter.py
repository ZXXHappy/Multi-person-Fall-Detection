"""Scalar One Euro smoothing; timestamps are measured in seconds.

Algorithm reference: https://gery.casiez.net/1euro/
"""

import math


# Central settings for pose experiments. Applied when filters are created.
ONE_EURO_FILTER_PARAMS = {
    "min_cutoff": 1.0,
    "beta": 0.007,
    "d_cutoff": 1.0,
}


class OneEuroFilter:
    """Adaptive low-pass filter with independent state for one scalar signal."""

    def __init__(
        self,
        min_cutoff=ONE_EURO_FILTER_PARAMS["min_cutoff"],
        beta=ONE_EURO_FILTER_PARAMS["beta"],
        d_cutoff=ONE_EURO_FILTER_PARAMS["d_cutoff"],
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        if not all(math.isfinite(p) for p in (self.min_cutoff, self.beta, self.d_cutoff)):
            raise ValueError("Filter parameters must be finite")
        if self.min_cutoff <= 0 or self.d_cutoff <= 0 or self.beta < 0:
            raise ValueError("Cutoffs must be positive and beta must be nonnegative")
        self._timestamp = None
        self._raw_value = None
        self._filtered_value = None
        self._filtered_derivative = 0.0

    @staticmethod
    def _alpha(cutoff, elapsed):
        rate = 2.0 * math.pi * cutoff * elapsed
        return rate / (1.0 + rate)

    def __call__(self, value, timestamp):
        """Filter a sample; the first sample initializes state without smoothing.

        Repeated or backward timestamps leave the previous state unchanged.
        Use one increasing timestamp per pose for all its coordinate filters.
        """
        value = float(value)
        timestamp = float(timestamp)
        if not math.isfinite(value) or not math.isfinite(timestamp):
            raise ValueError("Sample and timestamp must be finite")

        if self._timestamp is None:
            self._timestamp = timestamp
            self._raw_value = value
            self._filtered_value = value
            return value

        elapsed = timestamp - self._timestamp
        if elapsed <= 0:
            return self._filtered_value

        # Smooth the raw sample derivative before adapting the position cutoff.
        derivative = (value - self._raw_value) / elapsed
        derivative_alpha = self._alpha(self.d_cutoff, elapsed)
        self._filtered_derivative += derivative_alpha * (
            derivative - self._filtered_derivative
        )
        cutoff = self.min_cutoff + self.beta * abs(self._filtered_derivative)
        alpha = self._alpha(cutoff, elapsed)
        self._filtered_value += alpha * (value - self._filtered_value)
        self._raw_value = value
        self._timestamp = timestamp
        return self._filtered_value
