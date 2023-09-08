import comtypes
import numpy as np

time_format_code = "%m/%d/%Y %H:%M:%S"


def nanround(x, digits):
    """
    Round scalar to specified number of digits. Pass NaNs through without attempting to round
    :param x:
    :param digits:
    :return:
    """
    if np.isnan(x):
        return x
    else:
        return round(x, digits)


# def rel_round(x, precision):
#     """
#     Round to relative precision
#     :param ndarray x: array of numbers to round
#     :param int precision : number of digits to keep
#     :return: rounded array
#     """
#     if precision is not None:
#         try:
#             # attempt to cast as float
#             x = np.array(x, dtype=float)
#             # add 1e-30 for safety in case of zeros in x
#             x_scale = np.floor(np.log10(np.array(np.abs(x)) + 1e-30))
#             digits = (precision - x_scale).astype(int)
#             # TODO: rewrite to use np.round rather than round in listcomp
#             # Scale all entries by x_scale, then np.round(precision), then rescale
#             # print(digits)
#             if np.shape(x) == ():
#                 x_round = nanround(x[()], digits)
#             else:
#                 shape = x.shape
#                 x_round = np.array([nanround(xi, di) for xi, di in zip(x.flatten(), digits.flatten())])
#                 x_round = np.reshape(x_round, shape)
#             return x_round
#         except (ValueError, TypeError) as err:
#             print(err)
#             return x
#     else:
#         return x


def rel_round(x, precision):
    """
    Round to relative precision
    :param ndarray x: array of numbers to round
    :param int precision : number of digits to keep
    :return: rounded array
    """
    if precision is not None:
        try:
            # attempt to cast as float
            x = np.array(x, dtype=float)
            # Scale all entries to order 1
            # add 1e-30 for safety in case of zeros in x
            x_order = np.floor(np.log10(np.array(np.abs(x)) + 1e-30))
            x_scaled = x / (10 ** x_order)
            if np.shape(x) == ():
                # Round scaled value
                xs_round = nanround(x_scaled[()], precision)
                # Rescale
                x_round = xs_round * (10 ** x_order)
            else:
                # Round scaled values
                xs_round = np.round(x_scaled, precision)
                # Rescale
                x_round = xs_round * (10 ** x_order)
            return x_round
        except (ValueError, TypeError) as err:
            print(err)
            return x
    else:
        return x
    

def nearest_index(x_array, x_val, constraint=None):
    """
    Get index of x_array corresponding to value closest to x_val
    :param ndarray x_array: Array to index
    :param float x_val: Value to match
    :param int constraint: If -1, find the nearest index for which x_array <= x_val. If 1, find the nearest index for
    which x_array >= x_val. If None, find the closest index regardless of direction
    :return:
    """
    if constraint is None:
        def func(arr, x):
            return np.abs(arr - x)
    elif constraint in [-1, 1]:
        def func(arr, x):
            out = np.zeros_like(arr) + np.inf
            constraint_index = constraint * arr >= constraint * x
            out[constraint_index] = constraint * (arr - x)[constraint_index]
            return out
    else:
        raise ValueError(f'Invalid constraint argument {constraint}. Options: None, -1, 1')

    obj_func = func(x_array, x_val)
    index = np.argmin(obj_func)

    # Validate index
    if obj_func[index] == np.inf:
        if constraint == -1:
            min_val = np.min(x_array)
            raise ValueError(f'No index satisfying {constraint} constraint: minimum array value {min_val} '
                             f'exceeds target value {x_val}')
        else:
            max_val = np.max(x_array)
            raise ValueError(f'No index satisfying {constraint} constraint: maximum array value {max_val} '
                             f'is less than target value {x_val}')

    return index


# Error handling
# -----------------
class GamryCOMError(Exception):
    pass


def gamry_error_decoder(e):
    if isinstance(e, comtypes.COMError):
        hresult = 2 ** 32 + e.args[0]
        if hresult & 0x20000000:
            return GamryCOMError('0x{0:08x}: {1}'.format(2**32+e.args[0], e.args[1]))
    return e


# Check functions
# -----------------
def check_write_mode(write_mode):
    options = ['continuous', 'once', 'interval']
    if write_mode not in options:
        raise ValueError(f'Invalid write_mode {write_mode}. Options: {options}')


def check_control_mode(control_mode):
    options = ['galv', 'pot']
    if control_mode not in options:
        raise ValueError(f'Invalid write_mode {control_mode}. Options: {options}')


# Data processing and prep
# -------------------------
def get_eis_frequencies(max_freq, min_freq, ppd):
    num_decades = np.log10(max_freq) - np.log10(min_freq)
    num_freq = int(ppd * num_decades) + 1
    eis_freq = np.logspace(np.log10(max_freq), np.log10(min_freq), num_freq)
    return eis_freq


def identify_steps(y, allow_consecutive=True, rthresh=50):
    """
    Identify steps in signal
    :param ndarray y: signal
    :param bool allow_consecutive: if False, do not allow consecutive steps
    :param float rthresh: relative threshold for identifying steps
    :return: step indices
    """
    dy = np.diff(y)
    # Identify indices where diff exceeds threshold. Add small number to threshold in case median = 0
    step_idx = np.where(np.abs(dy) >= np.median(np.abs(dy)) * rthresh + 1e-10)[0] + 1

    if not allow_consecutive:
        # eliminate consecutive steps - these arise due to finite rise time and do not represent distinct steps
        idx_diff = np.diff(step_idx)
        # idx_diff = np.concatenate(([2], idx_diff))
        idx_diff = np.insert(idx_diff, 0, 2)
        print(step_idx, idx_diff)
        step_idx = step_idx[idx_diff > 1]

    print('identify_steps step_idx:', step_idx)

    return step_idx


def split_steps(x, step_index):
    """
    Split x by step indices
    :param ndarray x: array to split
    :param ndarray step_index: step indices
    :return:
    """
    step_index = np.array(step_index)
    # Add start and end indices
    if step_index[0] > 0:
        step_index = np.insert(step_index, 0, 0)
    if step_index[-1] < len(x):
        step_index = np.append(step_index, len(x))

    return [x[start:end] for start, end in zip(step_index[:-1], step_index[1:])]


def robust_std(x):
    """Estimate standard deviation from interquartile range"""
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return (q3 - q1) / 1.349


def pdf_normal(x, loc, scale):
    return 1 / (scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - loc) ** 2 / scale ** 2)


def get_quantile_limits(y, qr_size=0.5, qr_thresh=1.5):
    q_lo = np.percentile(y, 50 - 100 * qr_size / 2)
    q_hi = np.percentile(y, 50 + 100 * qr_size / 2)
    qr = q_hi - q_lo
    y_min = q_lo - qr * qr_thresh
    y_max = q_hi + qr * qr_thresh

    return y_min, y_max


def identify_extreme_values(y, qr_size=0.8, qr_thresh=1.5):
    y_min, y_max = get_quantile_limits(y, qr_size, qr_thresh)

    return (y < y_min) | (y > y_max)




