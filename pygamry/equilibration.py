import numpy as np
import time
from collections import deque
from scipy.ndimage import median_filter, gaussian_filter1d

from .dtaq import DtaqPstatic, DtaqGstatic


def signal_is_stable(times, values, slope_thresh, filter_values=True):
    if filter_values:
        values = gaussian_filter1d(median_filter(values, 5), 1)

    # Get slope
    fit = np.polyfit(times, values, deg=1)
    slope = fit[0]
    # print('slope: {:.6f} units/min'.format(slope))

    # If slope is below threshold, signal is stable
    if abs(slope) <= slope_thresh:
        return True
    else:
        return False


class EquilWrapper(object):
    def __init__(self, equil_window_seconds, slope_thresh,
                 breakin_window_seconds=2, min_wait_time_minutes=None, **init_kw):
        self.is_equilibrated = None
        self.equil_window_seconds = equil_window_seconds
        self.equil_window_points = None
        self.slope_thresh = slope_thresh
        self.breakin_window_seconds = breakin_window_seconds
        self.breakin_window_points = None
        self.equil_status_history = None

        if min_wait_time_minutes is None:
            min_wait_time_minutes = 0
        self.min_wait_time_minutes = min_wait_time_minutes

        super().__init__(**init_kw)

    def check_equilibration_status(self):
        status = False
        elapsed_minutes = (time.time() - self.file_start_time) / 60

        # If min_wait_time has elapsed, start checking for equilibration
        if elapsed_minutes >= self.min_wait_time_minutes:
            # If enough points are available, check slope
            if self.num_points - self.breakin_window_points >= self.equil_window_points:
                t_window, s_window = self.get_slope_data()

                # Get slope in % per minute
                # If slope is below threshold, sample is equilibrated
                status = signal_is_stable(t_window, s_window, self.slope_thresh)

        return status

    def get_slope_data(self):
        # Fill in at subclass level
        return None, None

    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        super()._IGamryDtaqEvents_OnDataAvailable(this)

        # Check equilibration status
        self.equil_status_history.append(self.check_equilibration_status())

        # If all recent statuses are True, sample is equilibrated
        self.is_equilibrated = (np.sum(self.equil_status_history) == self.equil_status_history.maxlen)

        # If equilibrated, close connection to terminate PumpEvents
        if self.is_equilibrated:
            # Fudge new_count for final write - only matters for write_mode continuous, in which case new_count will be
            # equal to total_points - last_write_index
            self.write_to_files(self.total_points - self._last_write_index, True)  # final write
            self.close_connection()

    def run(self, pstat, s_const, duration, t_sample, max_iter=1, require_consecutive_statuses=10, **kw):
        # Determine equilibration window in number of points
        self.equil_window_points = int(np.ceil(self.equil_window_seconds / t_sample))
        self.breakin_window_points = int(np.ceil(self.breakin_window_seconds / t_sample))
        print('equil_window_points:', self.equil_window_points)

        # Set equilibration status
        self.is_equilibrated = False
        self.equil_status_history = deque(maxlen=require_consecutive_statuses)

        # Iterate until equilibrated
        iteration = 0
        while not self.is_equilibrated:
            if iteration == 0:
                append_to_file = False
            else:
                append_to_file = True

            super().run(pstat, s_const, duration, t_sample, append_to_file=append_to_file, **kw)

            iteration += 1
            if iteration >= max_iter:
                break

        return self.is_equilibrated


class DtaqPstaticEquil(EquilWrapper, DtaqPstatic):
    def __init__(self, equil_window_seconds, slope_thresh_pct_per_minute=0.5,
                 breakin_window_seconds=2, min_wait_time_minutes=None, **init_kw):
        super().__init__(equil_window_seconds, slope_thresh_pct_per_minute, breakin_window_seconds,
                         min_wait_time_minutes, **init_kw)

    def get_slope_data(self):
        data = self.data_array[-self.equil_window_points:]
        i_window = data[:, self.cook_columns.index('Im')]
        t_window = data[:, self.cook_columns.index('Time')]

        # Normalize current to median or 3 sigma range, whichever is larger
        i_med = np.median(i_window)
        i_deno = np.sign(i_med) * max(abs(i_med), 6 * np.std(i_window))  # Minimum normalization current
        # print('i_deno: {:.6f} mA'.format(i_deno))
        i_norm = 100 * i_window / i_deno  # percent
        t_norm = t_window / 60  # minutes

        return t_norm, i_norm

    def run(self, pstat, v_equil, duration, t_sample, max_iter=1, require_consecutive_statuses=10, **kw):
        super().run(pstat, v_equil, duration, t_sample, max_iter=max_iter,
                    require_consecutive_statuses=require_consecutive_statuses,
                    **kw)


class DtaqGstaticEquil(EquilWrapper, DtaqGstatic):
    def __init__(self, equil_window_seconds, slope_thresh_mv_per_minute=0.5,
                 breakin_window_seconds=2, min_wait_time_minutes=None, **init_kw):
        super().__init__(equil_window_seconds, slope_thresh_mv_per_minute, breakin_window_seconds,
                         min_wait_time_minutes, **init_kw)

    def get_slope_data(self):
        data = self.data_array[-self.equil_window_points:]
        v_window = data[:, self.cook_columns.index('Vf')]
        t_window = data[:, self.cook_columns.index('Time')]

        v_norm = v_window * 1000  # mV
        t_norm = t_window / 60  # minutes

        return t_norm, v_norm

    def run(self, pstat, i_equil, duration, t_sample, max_iter=1, require_consecutive_statuses=10, **kw):
        super().run(pstat, i_equil, duration, t_sample, max_iter=max_iter,
                    require_consecutive_statuses=require_consecutive_statuses,
                    **kw)


# class DtaqPstaticEquil(DtaqPstatic):
#     def __init__(self, equil_window_seconds, slope_thresh_pct_per_minute=0.5,
#                  breakin_window_seconds=10,
#                  min_wait_time_minutes=None, **init_kw):
#         self.is_equilibrated = None
#         self.equil_window_seconds = equil_window_seconds
#         self.equil_window_points = None
#         self.slope_thresh_pct_per_minute = slope_thresh_pct_per_minute
#         self.breakin_window_seconds = breakin_window_seconds
#         self.breakin_window_points = None
#         self.equil_status_history = None
#
#         if min_wait_time_minutes is None:
#             min_wait_time_minutes = 0
#         self.min_wait_time_minutes = min_wait_time_minutes
#
#         # if max_wait_time_minutes is None:
#         #     max_wait_time_minutes = np.inf
#         # self.max_wait_time_minutes = max_wait_time_minutes
#
#         super().__init__(**init_kw)
#
#     def check_equilibration_status(self):
#         status = False
#         elapsed_minutes = (time.time() - self.file_start_time) / 60
#
#         # If min_wait_time has elapsed, start checking for equilibration
#         if elapsed_minutes >= self.min_wait_time_minutes:
#             # If enough points are available, check slope
#             if self.num_points - self.breakin_window_points >= self.equil_window_points:
#                 data = self.data_array[-self.equil_window_points:]
#                 i_window = data[:, self.cook_columns.index('Im')]
#                 t_window = data[:, self.cook_columns.index('Time')]
#
#                 # Normalize current to median or 3 sigma range, whichever is larger
#                 i_med = np.median(i_window)
#                 i_deno = np.sign(i_med) * max(abs(i_med), 6 * np.std(i_window))  # Minimum normalization current
#                 # print('i_deno: {:.6f} mA'.format(i_deno))
#                 i_norm = 100 * i_window / i_deno  # percent
#                 t_norm = t_window / 60  # minutes
#
#                 # Get slope in % per minute
#                 # If slope is below threshold, sample is equilibrated
#                 status = signal_is_stable(t_norm, i_norm, self.slope_thresh_pct_per_minute)
#
#         return status
#
#     def _IGamryDtaqEvents_OnDataAvailable(self, this):
#         super()._IGamryDtaqEvents_OnDataAvailable(this)
#
#         # Check equilibration status
#         self.equil_status_history.append(self.check_equilibration_status())
#
#         # If all recent statuses are True, sample is equilibrated
#         self.is_equilibrated = (np.sum(self.equil_status_history) == self.equil_status_history.maxlen)
#
#         # If equilibrated, close connection to terminate PumpEvents
#         if self.is_equilibrated:
#             # Fudge new_count for final write - only matters for write_mode continuous, in which case new_count will be
#             # equal to total_points - last_write_index
#             self.write_to_files(self.total_points - self._last_write_index, True)  # final write
#             self.close_connection()
#
#     def run(self, pstat, v, duration, t_sample, max_iter=3, require_consecutive_statuses=10, **kw):
#         # Determine equilibration window in number of points
#         self.equil_window_points = int(np.ceil(self.equil_window_seconds / t_sample))
#         self.breakin_window_points = int(np.ceil(self.breakin_window_seconds / t_sample))
#         print('equil_window_points:', self.equil_window_points)
#
#         # Set equilibration status
#         self.is_equilibrated = False
#         self.equil_status_history = deque(maxlen=require_consecutive_statuses)
#
#         # Iterate until equilibrated
#         iteration = 0
#         while not self.is_equilibrated:
#             if iteration == 0:
#                 append_to_file = False
#             else:
#                 append_to_file = True
#
#             super().run(pstat, v, duration, t_sample, append_to_file=append_to_file, **kw)
#
#             iteration += 1
#             if iteration >= max_iter:
#                 break
#
#         return self.is_equilibrated
#
#
# class DtaqGstaticEquil(DtaqGstatic):
#     def __init__(self, equil_window_seconds, slope_thresh_mv_per_minute=0.5,
#                  breakin_window_seconds=10,
#                  min_wait_time_minutes=None, **init_kw):
#         self.is_equilibrated = None
#         self.equil_window_seconds = equil_window_seconds
#         self.equil_window_points = None
#         self.breakin_window_seconds = breakin_window_seconds
#         self.breakin_window_points = None
#         self.slope_thresh_mv_per_minute = slope_thresh_mv_per_minute
#         self.equil_status_history = None
#
#         if min_wait_time_minutes is None:
#             min_wait_time_minutes = 0
#         self.min_wait_time_minutes = min_wait_time_minutes
#
#         # if max_wait_time_minutes is None:
#         #     max_wait_time_minutes = np.inf
#         # self.max_wait_time_minutes = max_wait_time_minutes
#
#         super().__init__(**init_kw)
#
#     def check_equilibration_status(self):
#         status = False
#         elapsed_minutes = (time.time() - self.file_start_time) / 60
#
#         # If min_wait_time has elapsed, start checking for equilibration
#         if elapsed_minutes >= self.min_wait_time_minutes:
#             # If enough points are available, check slope
#             if self.num_points - self.breakin_window_points >= self.equil_window_points:
#                 data = self.data_array[-self.equil_window_points:]
#                 v_window = data[:, self.cook_columns.index('Vf')]
#                 t_window = data[:, self.cook_columns.index('Time')]
#
#                 v_norm = v_window * 1000  # mV
#                 t_norm = t_window / 60  # minutes
#
#                 # Get slope in mV per minute
#                 # If slope is below threshold, sample is equilibrated
#                 status = signal_is_stable(t_norm, v_norm, self.slope_thresh_mv_per_minute)
#
#         return status
#
#     def _IGamryDtaqEvents_OnDataAvailable(self, this):
#         super()._IGamryDtaqEvents_OnDataAvailable(this)
#
#         # Check equilibration status
#         self.equil_status_history.append(self.check_equilibration_status())
#
#         # If all recent statuses are True, sample is equilibrated
#         self.is_equilibrated = (np.sum(self.equil_status_history) == self.equil_status_history.maxlen)
#
#         # If equilibrated, close connection to terminate PumpEvents
#         if self.is_equilibrated:
#             # Fudge new_count for final write - only matters for write_mode continuous, in which case new_count will be
#             # equal to total_points - last_write_index
#             self.write_to_files(self.total_points - self._last_write_index, True)  # final write
#             self.close_connection()
#
#     def run(self, pstat, i, duration, t_sample, max_iter=3, require_consecutive_statuses=10, **kw):
#         # Determine equilibration window in number of points
#         self.equil_window_points = int(np.ceil(self.equil_window_seconds / t_sample))
#         self.breakin_window_points = int(np.ceil(self.breakin_window_seconds / t_sample))
#         print('equil_window_points:', self.equil_window_points)
#
#         # Set equilibration status
#         self.is_equilibrated = False
#         self.equil_status_history = deque(maxlen=require_consecutive_statuses)
#
#         # Iterate until equilibrated
#         iteration = 0
#         while not self.is_equilibrated:
#             if iteration == 0:
#                 append_to_file = False
#             else:
#                 append_to_file = True
#
#             super().run(pstat, i, duration, t_sample, append_to_file=append_to_file, **kw)
#
#             iteration += 1
#             if iteration >= max_iter:
#                 break
#
#         return self.is_equilibrated
