import numpy as np
from scipy import ndimage

from ..utils import identify_steps


def find_amp_steps(y, period):
    y_range = ndimage.percentile_filter(y, 99, size=period) - ndimage.percentile_filter(y, 1, size=period)
    return identify_steps(y_range, allow_consecutive=False)

# TODO:
#  1. Find amplitude steps
#  2. Write step function
#  3. Write sin-step function
#  4. Fit step times, amplitudes, frequency, phase