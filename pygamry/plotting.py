import numpy as np
from collections import OrderedDict
# import matplotlib.pyplot as plt


# time_unit_dict = OrderedDict(s=1, min=60, h=3600, d=3600 * 24)
#
# def get_time_units(duration):
#     num_intervals = duration / scale

def get_nyquist_limits(ax, z_data):
    fig = ax.get_figure()

    x_data = z_data.real
    y_data = -z_data.imag

    # If data extends beyond current axis limits, adjust to capture all data
    axis_limits = {}
    axis_floor = {}
    for data, axis in zip([x_data, y_data], ['x', 'y']):
        data_range = np.max(data) - np.min(data)
        # Set axis floor (if data doesn't go negative, don't let axis go negative)
        if np.min(data) >= 0:
            axis_floor[axis] = 0
        else:
            axis_floor[axis] = -np.inf

        if np.min(data) < getattr(ax, f'get_{axis}lim')()[0]:
            # Extend lower axis limit to capture data
            axmin = max(axis_floor[axis], np.min(data) - data_range * 0.1)
        else:
            # Current lower limit ok
            axmin = getattr(ax, f'get_{axis}lim')()[0]

        if np.max(data) > getattr(ax, f'get_{axis}lim')()[1]:
            # Extend upper axis limit to capture data
            axmax = np.max(data) + data_range * 0.1
        else:
            # Current upper limit ok
            axmax = getattr(ax, f'get_{axis}lim')()[1]
        axis_limits[axis] = (axmin, axmax)

    # get range of each axis
    yrng = axis_limits['y'][1] - axis_limits['y'][0]
    xrng = axis_limits['x'][1] - axis_limits['x'][0]

    # get physical axis dimensions
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    yscale = yrng / height
    xscale = xrng / width

    if yscale > xscale:
        # Expand the x axis
        diff = (yscale - xscale) * width
        xmin = max(axis_floor['x'], axis_limits['x'][0] - diff / 2)
        xmin_delta = axis_limits['x'][0] - xmin
        xmax = axis_limits['x'][1] + diff - xmin_delta

        # ax.set_xlim(xmin, xmax)
        axis_limits['x'] = (xmin, xmax)
    elif xscale > yscale:
        # Expand the y axis
        diff = (xscale - yscale) * height

        ymin = max(axis_floor['y'], axis_limits['y'][0] - diff / 2)
        ymin_delta = axis_limits['y'][0] - ymin
        ymax = axis_limits['y'][1] + diff - ymin_delta

        # ax.set_ylim(ymin, ymax)
        axis_limits['y'] = (ymin, ymax)

    return axis_limits