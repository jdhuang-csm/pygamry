import numpy as np
# import pandas as pd


# def get_mfc_channel(channel_config, mfc_id, port=None):
#     if port is None:
#         # Assume all MFCs on same port
#         return dict(zip(channel_config['ID'].values, channel_config['Channel'].values)).get(mfc_id, None)
#     else:
#         mfc_df = channel_config[(channel_config['ID'] == mfc_id) & (channel_config['Port'] == port)]
#         if len(mfc_df) == 1:
#             return mfc_df['Channel'].values[0]
#         elif len(mfc_df) == 0:
#             return None
#         elif len(mfc_df) > 1:
#             raise ValueError(f'Multiple entries found for mfc_id {mfc_id} and port {port}')


# Miscellaneous functions
# -----------------------
def check_equality(a, b):
    """
    Convenience function for testing equality of arrays or dictionaries containing arrays
    :param dict or ndarray a: First object
    :param dict or ndarray b: Second object
    :return: bool
    """
    out = True
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        out = False

    return out
    
    
def rel_round(x, precision):
    """
    Round to relative precision
    :param ndarray x: array of numbers to round
    :param int precision : number of digits to keep
    :return: rounded array
    """
    try:
        # add 1e-30 for safety in case of zeros in x
        x_scale = np.floor(np.log10(np.array(np.abs(x)) + 1e-30))
        digits = (precision - x_scale).astype(int)
        # print(digits)
        if type(x) in (list, np.ndarray):
            if type(x) == list:
                x = np.array(x)
            shape = x.shape
            x_round = np.array([round(xi, di) for xi, di in zip(x.flatten(), digits.flatten())])
            x_round = np.reshape(x_round, shape)
        else:
            x_round = round(x, digits)
        return x_round
    except TypeError:
        return x
    
   
def is_subset(x, y, precision=10):
    """
    Check if x is a subset of y
    :param ndarray x: candidate subset array
    :param ndarray y: candidate superset array
    :param int precision: number of digits to compare. If None, compare exact values (or non-numeric values)
    :return: bool
    """
    if precision is None:
        # # Compare exact or non-numeric values
        # return np.min([xi in y for xi in x])
        set_x = set(x)
        set_y = set(y)
        return set_x.issubset(set_y)
    else:
        # Compare rounded values
        set_x = set(rel_round(x, precision))
        set_y = set(rel_round(y, precision))
        return set_x.issubset(set_y)


def eq_ph2o(temp):
    """
    Get approximate equilibrium water vapor pressure using Buck Equation
    :param float temp: temperature in C
    :return: vapor pressure in atm
    """
    p = 0.61121*np.exp((18.678 - temp/234.5)*(temp/(257.14+temp)))
    p /= 101.325
    return p
