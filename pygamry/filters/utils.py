import numpy as np
from scipy import ndimage


def masked_filter(a, mask, filter_func=None, **filter_kw):
    """
    Perform a masked/normalized filter operation on a. Only valid for linear filters
    :param ndarray a: array to filter
    :param ndarray mask: mask array indicating weight of each pixel in x_in; must match shape of x_in
    :param filter_func: filter function to apply. Defaults to gaussian_filter
    :param filter_kw: keyword args to pass to filter_func
    :return:
    """
    if filter_kw is None:
        if filter_func is None:
            sigma = np.ones(np.ndim(a))
            sigma[-1] = 0
            filter_kw = {'sigma': sigma}
        else:
            filter_kw = None
    if filter_func is None:
        filter_func = ndimage.gaussian_filter

    x_filt = filter_func(a * mask, **filter_kw)
    mask_filt = filter_func(mask, **filter_kw)

    return x_filt / mask_filt


def rms_filter(a, size, empty=False, **kw):
    # Get mean of squared deviations
    a2 = a ** 2
    a2_mean = ndimage.uniform_filter(a2, size, **kw)

    if empty:
        # Determine kernel volume
        if np.isscalar(size):
            ndim = np.ndim(a)
            n = size ** ndim
        else:
            n = np.prod(size)
        a2_mean -= a2 / n
        a2_mean *= n / (n - 1)

    # Small negatives may arise due to precision loss
    a2_mean[a2_mean < 0] = 0

    return a2_mean ** 0.5


def std_filter(a, size, mask=None, **kw):
    if mask is None:
        a_mean = ndimage.uniform_filter(a, size, **kw)
        var = ndimage.uniform_filter((a - a_mean) ** 2, size, **kw)
    else:
        a_mean = masked_filter(a, mask, ndimage.uniform_filter, size=size, **kw)
        var = masked_filter((a - a_mean) ** 2, mask, ndimage.uniform_filter, size=size, **kw)

    return var ** 0.5


def iqr_filter(a, size, **kw):
    q1 = ndimage.percentile_filter(a, 25, size=size, **kw)
    q3 = ndimage.percentile_filter(a, 75, size=size, **kw)
    return q3 - q1