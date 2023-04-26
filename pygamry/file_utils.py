import calendar
import time
import os
import numpy as np
import pandas as pd

from .utils import time_format_code


def get_file_time(file):
    with open(file, 'r') as f:
        txt = f.read()

    date_start = txt.find('DATE')
    date_end = txt[date_start:].find('\n') + date_start
    date_line = txt[date_start:date_end]
    date_str = date_line.split('\t')[2]

    time_start = txt.find('TIME')
    time_end = txt[time_start:].find('\n') + time_start
    time_line = txt[time_start:time_end]
    time_str = time_line.split('\t')[2]
    # Separate fractional seconds
    time_str, frac_seconds = time_str.split('.')

    dt_str = date_str + ' ' + time_str
    file_time = time.strptime(dt_str, time_format_code)

    return float(calendar.timegm(file_time)) + float('0.' + frac_seconds)


def read_last_line(file):
    with open(file, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:  # catch OSError in case of a one line file
            f.seek(0)
        last_line = f.readline().decode()

    return last_line


def get_decimation_index(times, step_times, t_sample, prestep_points, decimation_interval, decimation_factor,
                         max_t_sample):
    # Get evenly spaced samples from pre-step period
    prestep_times = times[times < np.min(step_times)]
    prestep_index = np.linspace(0, len(prestep_times) - 1, prestep_points).round(0).astype(int)

    # Determine index of first sample time after each step
    def pos_delta(x, x0):
        out = np.empty(len(x))
        out[x < x0] = np.inf
        out[x >= x0] = x[x >= x0] - x0
        return out

    step_index = [np.argmin(pos_delta(times, st)) for st in step_times]

    # Limit sample interval to max_t_sample
    if max_t_sample is None:
        max_sample_interval = np.inf
    else:
        max_sample_interval = int(max_t_sample / t_sample)

    # Build array of indices to keep
    keep_indices = [prestep_index]
    for i, start_index in enumerate(step_index):
        # Decimate samples after each step
        if start_index == step_index[-1]:
            next_step_index = len(times)
        else:
            next_step_index = step_index[i + 1]

        # Keep first decimation_interval points without decimation
        undec_index = np.arange(start_index, min(start_index + decimation_interval + 1, next_step_index), dtype=int)

        keep_indices.append(undec_index)
        sample_interval = 1
        last_index = undec_index[-1]
        while last_index < next_step_index - 1:
            # Increment sample_interval
            sample_interval = min(int(sample_interval * decimation_factor), max_sample_interval)

            if sample_interval == max_sample_interval:
                # Sample interval has reached maximum. Continue through end of step
                interval_end_index = next_step_index
            else:
                # Continue with current sampling rate until decimation_interval points acquired
                interval_end_index = min(last_index + decimation_interval * sample_interval + 1,
                                         next_step_index)

            keep_index = np.arange(last_index + sample_interval, interval_end_index, sample_interval, dtype=int)

            if len(keep_index) == 0:
                # sample_interval too large - runs past end of step. Keep last sample
                keep_index = [interval_end_index]

            # If this is the final interval, ensure that last point before next step is included
            if interval_end_index == next_step_index and keep_index[-1] < next_step_index - 1:
                keep_index = np.append(keep_index, next_step_index - 1)

            keep_indices.append(keep_index)

            # Increment last_index
            last_index = keep_index[-1]

    decimate_index = np.unique(np.concatenate(keep_indices))

    return decimate_index


def read_curve_data(file):
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    # find start of curve data
    cidx = txt.find('CURVE\tTABLE')

    # preceding text
    pretxt = txt[:cidx]

    # curve data
    ctable = txt[cidx:]

    # column headers are next line after CURVE TABLE line
    header_start = ctable.find('\n') + 1
    header_end = header_start + ctable[header_start:].find('\n')
    header = ctable[header_start:header_end].split('\t')

    # # units are next line after column headers
    # unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
    # units = ctable[header_end + 1:unit_end].split('\t')

    # determine # of rows to skip by counting line breaks in preceding text
    skiprows = len(pretxt.split('\n')) + 2

    # if table is indented, ignore empty left column
    if header[0] == '':
        usecols = header[1:]
    else:
        usecols = header
    # read data to DataFrame
    data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols)

    return data