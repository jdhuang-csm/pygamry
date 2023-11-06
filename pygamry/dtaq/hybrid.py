import numpy as np
import os
import time
import pandas as pd
import warnings
from copy import deepcopy

from .readz import DtaqReadZ
from .chrono import DtaqChrono
from .config import GamryCOM

from ..utils import gamry_error_decoder
from ..file_utils import read_curve_data
from .. import polarization as pol


# ====================================
# Hybrid
# ====================================
class HybridSequencer:
    def __init__(self, chrono_mode='galv', eis_mode='galv', update_step_size=True, exp_notes=None):
        self.chrono_mode = chrono_mode
        self.eis_mode = eis_mode
        self.update_step_size = update_step_size

        if eis_mode == 'pot':
            warnings.warn('If eis_mode is set to pot, EIS parameters must be configured manually '
                          'by calling configure_eis after configuring the chrono step')

        # Initialize dtaqs. Don't turn cell off - handle this within run
        self.dt_chrono = DtaqChrono(chrono_mode, write_mode='once', exp_notes=exp_notes,
                                    start_with_cell_off=False, leave_cell_on=True)
        self.dt_eis = DtaqReadZ(eis_mode, write_mode='interval', write_interval=1, 
                                exp_notes=exp_notes,
                                start_with_cell_off=False, leave_cell_on=True)  #, readzspeed='ReadZSpeedFast')

        self.dstep_args = None
        self.mstep_args = None
        self.triplestep_args = None
        self.geostep_args = None
        self.eis_args = None
        self.measurement_type = None

        self.staircase_step_type = None
        self.staircase_num_steps = None
        self.staircase_v_limits = None
        self.staircase_args = None
        self.staircase_geo_kwargs = None
        self.staircase_v_rms_target = None
        self.staircase_jv_data = None
        self.staircase_i_hist = None
        self.staircase_v_hist = None

        self.r_tot_est = None
        self.meas_v_min = None
        self.meas_v_max = None
        self.meas_v_start = None
        self.meas_v_end = None
        self.meas_i_min = None
        self.meas_i_max = None
        self.meas_i_start = None
        self.meas_i_end = None

    def configure_eis(self, frequencies, dc_amp, ac_amp, z_guess=None):
        # Determine z_guess from s_rms
        if z_guess is None:
            if self.eis_mode == 'galv':
                # Assume step size is set to obtain v_rms of 10 mV
                z_guess = 0.01 / ac_amp
            elif self.eis_mode == 'pot':
                # Arbitrary guess
                z_guess = 1

        self.eis_args = [frequencies, dc_amp, ac_amp, z_guess]

    def configure_triple_step(self, s_init, s_rms, t_init, t_step, t_sample,
                              frequencies, z_guess=None):
        "half step - full step - half step"
        # # Chrono step is sqrt(2) * s_rms
        # s_half_step = s_rms * np.sqrt(2)
        #
        # # Dstep first: half step up, full step down
        # self.dstep_args = [s_init, s_init + init_sign * s_half_step, s_init - init_sign * s_half_step,
        #                    t_init, t_step, t_step, t_sample]
        # # Single step second: half step up back to s_init
        # self.mstep_args = [s_init - init_sign * s_half_step, init_sign * s_half_step,
        #                    t_init, t_step, t_sample, 1]

        self.triplestep_args = [s_init, s_rms, t_init, t_step, t_sample]

        # Ensure EIS will be centered at s_init and cover same range as steps
        dc_amp = s_init
        ac_amp = abs(s_rms)

        self.configure_eis(frequencies, dc_amp, ac_amp, z_guess=z_guess)

        # # Determine z_guess from s_rms
        # if z_guess is None:
        #     if self.mode == 'galv':
        #         # Assume step size is set to obtain v_rms of 10 mV
        #         z_guess = 0.01 / ac_amp
        #     elif self.mode == 'pot':
        #         # Arbitrary guess
        #         z_guess = 1
        #
        # self.eis_args = [frequencies, dc_amp, ac_amp, z_guess]

        self.measurement_type = 'triple_step'

    def configure_geo_step(self, s_init, s_rms, t_init, t_short, t_long, t_sample, num_scales, steps_per_scale,
                           frequencies, z_guess=None, end_at_init=False, end_time=None):
        """Geometric step centered at s_init"""
        # Set range based on s_rms
        s_half_step = s_rms * np.sqrt(2)
        s_max = s_init + abs(s_half_step)
        s_min = s_init - abs(s_half_step)

        # Determine end value
        if steps_per_scale >= 3:
            # 3 steps ensures full magnitude steps in both directions. End at initial value
            s_final = s_init
        else:
            # Ensure that the final step will be of full magnitude
            num_steps = num_scales * steps_per_scale
            sign_switch = (-1) ** (num_steps - 1)
            if sign_switch == -1:
                s_final = s_min
            else:
                s_final = s_max

        self.geostep_args = [
            s_init, s_final, s_min, s_max, t_init, t_sample, t_short, t_long, num_scales, steps_per_scale, 0.05,
            end_at_init, end_time
        ]

        # Ensure EIS will be centered at s_init and cover same range as steps
        dc_amp = s_init
        ac_amp = abs(s_rms)
        self.configure_eis(frequencies, dc_amp, ac_amp, z_guess=z_guess)

        self.measurement_type = 'geo_step'

    def configure_gzh_step(self, s_init, s_rms, t_init, t_short, t_long, t_sample, num_scales, steps_per_scale,
                           frequencies, z_guess=None):
        """Geo step ending at range center - EIS - half step to final value"""

        # Set range based on s_rms
        s_half_step = s_rms * np.sqrt(2)

        # Configure geostep
        s_center = s_init + s_half_step
        if s_rms < 0:
            s_max = s_init
            s_min = s_init + 2 * s_half_step
        else:
            s_min = s_init
            s_max = s_init + 2 * s_half_step

        self.geostep_args = [
            s_init, s_center, s_min, s_max, t_init, t_sample, t_short, t_long, num_scales, steps_per_scale, 0.05
        ]

        # Configure half step (mstep)
        # Start from center value and make a half step to final value. Use t_long from geo config
        self.mstep_args = [s_center, s_half_step, t_init, t_long, t_sample, 1]

        # Ensure EIS will be centered at s_init and cover same range as steps
        dc_amp = s_center
        ac_amp = abs(s_rms)
        self.configure_eis(frequencies, dc_amp, ac_amp, z_guess=z_guess)

        self.measurement_type = 'gzh'

    def configure_hzh_step(self, s_init, s_rms, t_init, t_step, t_sample,
                           frequencies, z_guess=None):
        "half step - EIS - haf step"
        # Chrono step is sqrt(2) * s_rms
        s_half_step = s_rms * np.sqrt(2)

        # Need two msteps: one before EIS, one after
        self.mstep_args = []
        # First step: half step to eis dc_amp
        self.mstep_args.append([s_init, s_half_step, t_init, t_step, t_sample, 1])
        # Second step: eis dc_amp plus half step
        self.mstep_args.append([s_init + s_half_step, s_half_step, t_init, t_step, t_sample, 1])

        # Ensure EIS will cover same range as steps
        dc_amp = s_init + s_half_step
        ac_amp = abs(s_rms)
        self.configure_eis(frequencies, dc_amp, ac_amp, z_guess=z_guess)

        self.measurement_type = 'hzh'

    def configure_fhzh_step(self, s_init, s_rms, t_init, t_step, t_sample,
                            frequencies, z_guess=None):
        "full step up - half step down - EIS - half step up"
        # Chrono step is sqrt(2) * s_rms
        s_half_step = s_rms * np.sqrt(2)

        # Dstep, EIS, Mstep

        # First step: Dstep - full step up, then back down one half step to eis dc_amp
        self.dstep_args = [s_init, s_init + 2 * s_half_step, s_init + s_half_step, t_init, t_step, t_step, t_sample]

        # Second step: Mstep from eis dc_amp plus half step
        self.mstep_args = [s_init + s_half_step, s_half_step, t_init, t_step, t_sample, 1]

        # Ensure EIS will cover same range as steps
        dc_amp = s_init + s_half_step
        ac_amp = abs(s_rms)
        self.configure_eis(frequencies, dc_amp, ac_amp, z_guess=z_guess)

        self.measurement_type = 'fhzh'

    def configure_fz_step(self, s_init, s_rms, t_init, t_step, t_sample,
                          frequencies, z_guess=None):
        "full step - EIS"
        # Chrono step is sqrt(2) * s_rms
        s_half_step = s_rms * np.sqrt(2)

        # Mstep: full step from s_init to eis dc_amp
        self.mstep_args = [s_init, 2 * s_half_step, t_init, t_step, t_sample, 1]

        # EIS starts at 2 * s_half_step
        dc_amp = s_init + 2 * s_half_step
        ac_amp = abs(s_rms)
        self.configure_eis(frequencies, dc_amp, ac_amp, z_guess=z_guess)

        self.measurement_type = 'fz'

    def configure_staircase(self, s_init, s_rms, t_init, t_step, t_sample,
                            frequencies, num_steps, v_limits=None, v_rms_target=None, step_type='hzh',
                            z_guess=None, geo_kwargs=None):
        step_options = ['fz', 'hzh', 'fhzh', 'gzh']
        if step_type not in step_options:
            raise ValueError(f'Invalid step_type {step_type} for staircase. Options: {step_options}')

        # Validate geo_kwargs
        if step_type == 'gzh':
            if geo_kwargs is None:
                raise ValueError('geo_kwargs must be provided for staircase step_type gzh')
            else:
                expected_keys = ['t_short', 'num_scales', 'steps_per_scale']
                for key in expected_keys:
                    if geo_kwargs.get(key, None) is None:
                        raise ValueError(f'geo_kwargs is missing key {key}')

        # if num_steps is None and v_limits is None:
        #     raise ValueError('Either num_steps or v_limits must be provided to determine end of staircase experiment')
        if v_limits is None:
            v_limits = (-np.inf, np.inf)

        self.staircase_step_type = step_type
        self.staircase_num_steps = num_steps
        self.staircase_v_limits = v_limits
        self.staircase_v_rms_target = v_rms_target
        self.staircase_args = [s_init, s_rms, t_init, t_step, t_sample, frequencies, z_guess]
        self.staircase_geo_kwargs = geo_kwargs

    def configure_staircase_from_jv(self, i_init, v_rms_target, t_init, t_step, t_sample, jv_data,
                                    frequencies, num_steps, v_limits=None, step_type='gzh', geo_kwargs=None):
        # Store jv_data
        if jv_data is not None:
            if type(jv_data) == str:
                jv_data = read_curve_data(jv_data)
            elif type(jv_data) != pd.DataFrame:
                raise ValueError(f'Expected jv_data to be a path or DataFrame; got type {type(jv_data)} instead')

        # Check whether ImExpected should be used in place of Im
        if 'ImExpected' in jv_data.columns:
            sort_index = np.argsort(jv_data['ImExpected'].values)
            i_meas = jv_data['Im'].values[sort_index]
            i_expected = jv_data['ImExpected'].values[sort_index]
            i_step = np.mean(np.diff(i_expected))
            max_err = np.max(np.abs(i_meas - i_expected)) / abs(i_step)
            if max_err > 0.5:
                i_col = 'ImExpected'
            else:
                i_col = 'Im'
        else:
            i_col = 'Im'

        # Pad jv data to allow extrapolation beyond measured range
        # Set min and max current well beyond measured range
        i_range = jv_data[i_col].max() - jv_data[i_col].min()
        i_low = jv_data[i_col].min() - 10 * i_range
        i_high = jv_data[i_col].max() + 10 * i_range

        # Sort by current
        sort_index = np.argsort(jv_data[i_col].values)
        i_sort = jv_data[i_col].values[sort_index]
        v_sort = jv_data['Vf'].values[sort_index]

        if len(i_sort) < 2:
            warnings.warn('Too few points in jv_data to use for staircase')
            self.staircase_jv_data = None
            # Start with 1 mA
            i_rms_init = 0.001
            self.configure_staircase(i_init, i_rms_init, t_init, t_step, t_sample, frequencies,
                                     num_steps, v_limits=v_limits, step_type=step_type, v_rms_target=v_rms_target,
                                     geo_kwargs=geo_kwargs)
        else:
            # Fit each end of curve for extrapolation
            if len(i_sort) >= 6:
                # Exclude endpoints due to voltage cutoff
                extrap_points = 4  # number of points to use for extrapolation
                i_sort = i_sort[1:-1]
                v_sort = v_sort[1:-1]
            else:
                # Very few points. Include endpoints
                extrap_points = min(4, len(i_sort))  # number of points to use for extrapolation
            low_fit = np.polyfit(i_sort[:extrap_points], v_sort[:extrap_points], deg=1)
            high_fit = np.polyfit(i_sort[-extrap_points:], v_sort[-extrap_points:], deg=1)
            v_low = np.polyval(low_fit, i_low)
            v_high = np.polyval(high_fit, i_high)
            i_extrap = np.concatenate([[i_low], i_sort, [i_high]])
            v_extrap = np.concatenate([[v_low], v_sort, [v_high]])

            # Store dataframe with extrapolated values
            jv_data_extrap = pd.DataFrame(np.vstack([i_extrap, v_extrap]).T, columns=['Im', 'Vf'])
            self.staircase_jv_data = jv_data_extrap

            # Determine initial i_rms
            i_rms_init = self.get_next_i_rms(i_init, v_rms_target)
            print('i_rms_init: {:.2e} A'.format(i_rms_init))

            self.configure_staircase(i_init, i_rms_init, t_init, t_step, t_sample, frequencies,
                                     num_steps, v_limits=v_limits, step_type=step_type, v_rms_target=v_rms_target,
                                     geo_kwargs=geo_kwargs)

    def configure_decimation(self, decimate_during, prestep_points, decimation_interval,
                             decimation_factor=None, max_t_sample=None):
        self.dt_chrono.configure_decimation(decimate_during, prestep_points, decimation_interval,
                                            decimation_factor, max_t_sample)

    def get_next_i_rms(self, i_init, v_rms_target):
        # Update current step size to maintain desired v_rms
        if self.staircase_jv_data is not None:
            # Extract i and v for interpolation
            im = self.staircase_jv_data['Im'].values
            vf = self.staircase_jv_data['Vf'].values

            # jv_data is sorted at time of intake - shouldn't need to sort here
            # sort_index = np.argsort(im)
            # im = im[sort_index]
            # vf = vf[sort_index]

            # Interpolate starting voltage
            v_init = np.interp(i_init, im, vf)

            # Interpolate current required to obtain desired end voltage
            v_full_step = 2 * v_rms_target * np.sqrt(2)
            i_end = np.interp(v_init + v_full_step, vf, im)

            # Get rms current
            i_full_step = i_end - i_init
            i_rms = i_full_step / (2 * np.sqrt(2))
        else:
            # Estimate based on last step
            # last_v_step = self.meas_v_max - self.meas_v_min
            # last_i_step = self.meas_i_max - self.meas_i_min
            # i_rms = last_i_step * v_rms_target / last_v_step
            i_next = pol.estimate_next_i(self.staircase_i_hist, self.staircase_v_hist,
                                         penalty_vec=[0, 0, 0.1],
                                         v_step=v_rms_target * 2 * np.sqrt(2),
                                         deg=2, num_points=3,
                                         prev_step_prior=1e-2, i_offset=None, v_offset=0, i_lambda=0)
            i_step = i_next - self.staircase_i_hist[-1]
            i_rms = i_step / (2 * np.sqrt(2))

        return i_rms

    def run(self, pstat, decimate=True, eis_first=True, data_path=None, kst_path=None, file_suffix='', rest_time=0,
            start_with_cell_off=True, show_plot=False, eis_max_passes=5,
            leave_cell_on=False, filter_response=False):

        if self.eis_args is None:
            raise RuntimeError('Measurement must be configured before calling run')

        eis_tag = f'EIS{self.eis_mode.upper()}'
        if self.chrono_mode == 'galv':
            chrono_tag = 'CHRONOP'
        else:
            chrono_tag = 'CHRONOA'

        if decimate and self.dt_chrono.decimate_args is None:
            raise RuntimeError('Decimation must be configured prior to calling run if decimate=True')

        # Open pstat
        pstat.Open()

        # Convenience functions
        # Rest function
        def rest():
            if rest_time > 0:
                print(f'Resting for {rest_time} s between measurements...')
                time.sleep(rest_time)

        def run_eis():
            print('Running EIS')
            self.dt_eis.run(pstat, *self.eis_args, result_file=eis_file, kst_file=eis_kst_file,
                            show_plot=show_plot, max_passes=eis_max_passes)

        if start_with_cell_off:
            # Start with cell off
            pstat.SetCell(GamryCOM.CellOff)

        if kst_path is not None:
            eis_kst_file = os.path.join(kst_path, 'Kst_EIS.DTA')
            chrono_kst_file = os.path.join(kst_path, 'Kst_IVT.DTA')
        else:
            eis_kst_file = None
            chrono_kst_file = None

        try:
            r_tot_values = []
            v_limits = []
            i_limits = []

            if self.measurement_type in ['triple_step', 'geo_step']:
                # Basic hybrid patterns
                if data_path is not None:
                    eis_file = os.path.join(data_path, '{}_{}.DTA'.format(eis_tag, file_suffix))
                    chrono_file = os.path.join(data_path, '{}_{}.DTA'.format(chrono_tag, file_suffix))
                else:
                    eis_file = None
                    chrono_file = None

                # Make chrono function for convenience
                def run_chrono():
                    # Configure step
                    if self.measurement_type == 'triple_step':
                        print('Running triplestep')
                        self.dt_chrono.configure_triplestep_signal(*self.triplestep_args)
                    else:
                        print('Running geostep')
                        self.dt_chrono.configure_geostep_signal(*self.geostep_args)
                    # Run step
                    self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file,
                                       kst_file=chrono_kst_file, show_plot=show_plot,
                                       filter_response=False)

                    # TODO: remove when done testing
                    # Write a separate file with the filtered data
                    if filter_response and decimate:
                        self.dt_chrono.filter_response = True
                        self.dt_chrono.result_file = chrono_file.replace('.DTA', '_Filtered.DTA')
                        self.dt_chrono.kst_file = None

                        # Write header
                        with open(self.dt_chrono.result_file, 'w+') as f:
                            f.write(self.dt_chrono.generate_header_text())

                        self.dt_chrono.write_to_files(1, True)
                        self.dt_chrono.filter_response = False

                    # Append R and v values
                    r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
                    i_limits.append(self.dt_chrono.get_i_limits(percentiles=[1, 99]))
                    self.meas_v_start = self.dt_chrono.get_v_init()
                    self.meas_i_start = self.dt_chrono.get_i_init()

                if eis_first:
                    # Run EIS
                    # print('Running EIS')
                    # self.dt_eis.run(pstat, *self.eis_args, result_file=eis_file, kst_file=eis_kst_file,
                    #                 show_plot=show_plot)
                    run_eis()

                    rest()

                    run_chrono()

                    # # Run Dstep
                    # print('Running Dstep')
                    # self.dt_chrono.configure_dstep_signal(*self.dstep_args)
                    # self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file,
                    #                    kst_file=chrono_kst_file, show_plot=show_plot)
                    # r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    # v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
                    #
                    # # Run Mstep. Append to Dstep file
                    # print('Running Mstep')
                    # self.dt_chrono.configure_mstep_signal(*self.mstep_args)
                    # self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file, kst_file=chrono_kst_file,
                    #                    append_to_file=True, show_plot=show_plot)
                    # r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    # v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))

                else:
                    run_chrono()

                    # # Run Dstep
                    # print('Running Dstep')
                    # self.dt_chrono.configure_dstep_signal(*self.dstep_args)
                    # self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file, kst_file=chrono_kst_file,
                    #                    show_plot=show_plot)
                    # r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    # v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
                    #
                    # # Run Mstep. Append to Dstep file
                    # print('Running Mstep')
                    # self.dt_chrono.configure_mstep_signal(*self.mstep_args)
                    # self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file, kst_file=chrono_kst_file,
                    #                    append_to_file=True, show_plot=show_plot)
                    # r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    # v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))

                    rest()

                    # Run EIS
                    # print('Running EIS')
                    # self.dt_eis.run(pstat, *self.eis_args, result_file=eis_file, kst_file=eis_kst_file,
                    #                 show_plot=show_plot)
                    run_eis()

            elif self.measurement_type in ['hzh', 'fhzh', 'gzh']:
                # Chrono-EIS-chrono patterns for staircase
                if data_path is not None:
                    eis_file = os.path.join(data_path, '{}_{}.DTA'.format(eis_tag, file_suffix))
                    chrono_file_pre = os.path.join(data_path, '{}_{}-a.DTA'.format(chrono_tag, file_suffix))
                    chrono_file_post = os.path.join(data_path, '{}_{}-b.DTA'.format(chrono_tag, file_suffix))
                else:
                    eis_file = None
                    chrono_file_pre = None
                    chrono_file_post = None

                # Make chrono functions for clarity
                def run_chrono_a():
                    print('Running step a')
                    # Configure step
                    if self.measurement_type == 'hzh':
                        self.dt_chrono.configure_mstep_signal(*self.mstep_args[0])
                    elif self.measurement_type == 'fhzh':
                        self.dt_chrono.configure_dstep_signal(*self.dstep_args)
                    else:
                        self.dt_chrono.configure_geostep_signal(*self.geostep_args)

                    # Run step
                    self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file_pre,
                                       kst_file=chrono_kst_file, show_plot=show_plot,
                                       filter_response=False)

                    # TODO: remove when done testing
                    if filter_response and decimate:
                        self.dt_chrono.filter_response = True
                        self.dt_chrono.result_file = chrono_file_pre.replace('.DTA', '_Filtered.DTA')
                        self.dt_chrono.kst_file = None

                        # Write header
                        with open(self.dt_chrono.result_file, 'w+') as f:
                            f.write(self.dt_chrono.generate_header_text())

                        self.dt_chrono.write_to_files(1, True)
                        self.dt_chrono.filter_response = False

                    # Append R and v values
                    self.meas_v_start = self.dt_chrono.get_v_init()
                    self.meas_i_start = self.dt_chrono.get_i_init()
                    r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
                    i_limits.append(self.dt_chrono.get_i_limits(percentiles=[1, 99]))

                def run_chrono_b():
                    print('Running step b')
                    # Configure step
                    if self.measurement_type == 'hzh':
                        self.dt_chrono.configure_mstep_signal(*self.mstep_args[1])
                    else:
                        self.dt_chrono.configure_mstep_signal(*self.mstep_args)

                    # Run step
                    self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file_post,
                                       kst_file=chrono_kst_file, show_plot=show_plot,
                                       filter_response=False)

                    # # TODO: remove when done testing
                    # if filter_response:
                    #     self.dt_chrono.filter_response = True
                    #     self.dt_chrono.result_file = chrono_file_post.replace('.DTA', '_Filtered.DTA')
                    #     self.dt_chrono.kst_file = None
                    #
                    #     # Write header
                    #     with open(self.dt_chrono.result_file, 'w+') as f:
                    #         f.write(self.dt_chrono.generate_header_text())
                    #
                    #     self.dt_chrono.write_to_files(1, True)
                    #     self.dt_chrono.filter_response = False

                    # Append R and v values
                    r_tot_values.append(self.dt_chrono.estimate_r_tot())
                    v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
                    i_limits.append(self.dt_chrono.get_i_limits(percentiles=[1, 99]))

                # # Run Mstep a
                # print('Running Mstep a')
                # self.dt_chrono.configure_mstep_signal(*self.mstep_args[0])
                # self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file_pre, kst_file=chrono_kst_file,
                #                    show_plot=show_plot)
                # r_tot_values.append(self.dt_chrono.estimate_r_tot())
                # v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))

                run_chrono_a()

                rest()

                # Run EIS
                # print('Running EIS')
                # self.dt_eis.run(pstat, *self.eis_args, result_file=eis_file, kst_file=eis_kst_file,
                #                 show_plot=show_plot)
                run_eis()

                rest()

                run_chrono_b()

                # # Run Mstep b
                # print('Running Mstep b')
                # self.dt_chrono.configure_mstep_signal(*self.mstep_args[1])
                # self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file_post, kst_file=chrono_kst_file,
                #                    show_plot=show_plot)
                # r_tot_values.append(self.dt_chrono.estimate_r_tot())
                # v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))

            # elif self.measurement_type == 'fhzh':
            #     if data_path is not None:
            #         eis_file = os.path.join(data_path, '{}_{}.DTA'.format(eis_tag, file_suffix))
            #         chrono_file_pre = os.path.join(data_path, '{}_{}-a.DTA'.format(chrono_tag, file_suffix))
            #         chrono_file_post = os.path.join(data_path, '{}_{}-b.DTA'.format(chrono_tag, file_suffix))
            #     else:
            #         eis_file = None
            #         chrono_file_pre = None
            #         chrono_file_post = None
            #
            #     # Run Dstep
            #     print('Running Dstep a')
            #     self.dt_chrono.configure_dstep_signal(*self.dstep_args)
            #     self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file_pre, kst_file=chrono_kst_file,
            #                        show_plot=show_plot)
            #     r_tot_values.append(self.dt_chrono.estimate_r_tot())
            #     v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
            #
            #     rest()
            #
            #     # Run EIS
            #     # self.dt_eis.run(pstat, *self.eis_args, result_file=eis_file, kst_file=eis_kst_file,
            #     #                 show_plot=show_plot)
            #     run_eis()
            #
            #     rest()
            #
            #     # Run Mstep b
            #     print('Running Mstep b')
            #     self.dt_chrono.configure_mstep_signal(*self.mstep_args)
            #     self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file_post, kst_file=chrono_kst_file,
            #                        show_plot=show_plot)
            #     r_tot_values.append(self.dt_chrono.estimate_r_tot())
            #     v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))

            elif self.measurement_type == 'fz':
                # Chrono-EIS pattern for staircase. Obsolete
                if data_path is not None:
                    eis_file = os.path.join(data_path, '{}_{}.DTA'.format(eis_tag, file_suffix))
                    chrono_file = os.path.join(data_path, '{}_{}.DTA'.format(chrono_tag, file_suffix))
                else:
                    eis_file = None
                    chrono_file = None

                # Run Mstep
                print('Running Mstep')
                self.dt_chrono.configure_mstep_signal(*self.mstep_args)
                # print('dt_chrono.i_max:', self.dt_chrono.i_max)
                # print('dt_chrono.mode:', self.dt_chrono.mode)
                self.dt_chrono.run(pstat, decimate=decimate, result_file=chrono_file, kst_file=chrono_kst_file,
                                   show_plot=show_plot, filter_response=filter_response)
                r_tot_values.append(self.dt_chrono.estimate_r_tot())
                v_limits.append(self.dt_chrono.get_v_limits(percentiles=[1, 99]))
                i_limits.append(self.dt_chrono.get_i_limits(percentiles=[1, 99]))

                rest()

                # Run EIS
                # print('Running EIS')
                # self.dt_eis.run(pstat, *self.eis_args, result_file=eis_file, kst_file=eis_kst_file,
                #                 show_plot=show_plot)
                run_eis()

            # Store estimated resistance and voltage thresholds
            self.r_tot_est = np.mean(r_tot_values)
            self.meas_v_min = np.min([vlim[0] for vlim in v_limits])
            self.meas_v_max = np.max([vlim[1] for vlim in v_limits])
            self.meas_v_end = self.dt_chrono.get_v_final()
            self.meas_i_min = np.min([ilim[0] for ilim in i_limits])
            self.meas_i_max = np.max([ilim[1] for ilim in i_limits])
            self.meas_i_end = self.dt_chrono.get_i_final()

            if not leave_cell_on:
                # Turn cell off and close pstat
                pstat.SetCell(GamryCOM.CellOff)
                pstat.Close()
        except Exception as e:
            # If any exception thrown, turn cell off and close pstat
            if pstat.TestIsOpen():
                pstat.SetCell(GamryCOM.CellOff)
                pstat.Close()
            raise gamry_error_decoder(e)

    def run_staircase(self, pstat, decimate=True, eis_first=True, data_path=None, kst_path=None, file_suffix='',
                      equil_time=0, rest_time=0, run_full_eis_pre=False, run_full_eis_post=False, full_frequencies=None,
                      start_with_cell_off=True, leave_cell_on=False, filter_response=False,
                      show_plot=False):

        if self.eis_mode == 'pot':
            raise ValueError('Hybrid staircase is not implemented for potentiostatic EIS')

        if self.staircase_args is None:
            raise RuntimeError('Measurement must be configured before calling run_staircase')

        # Open pstat
        pstat.Open()

        if start_with_cell_off:
            # Start with cell off
            pstat.SetCell(GamryCOM.CellOff)

        # Equilibrate before starting staircase
        # If start_with_cell_off=True, equilibration will take place with cell off (at OCV)
        # If start_with_cell_off=False, this equilibration step will take place at whatever the previous cell state was
        print(f'Equilibrating for {equil_time} s before running staircase...')
        time.sleep(equil_time)

        # Extract staircase args
        s_init, s_rms, t_init, t_step, t_sample, frequencies, z_guess = self.staircase_args
        # Chrono step is sqrt(2) * s_rms
        s_half_step = s_rms * np.sqrt(2)

        # Initialize history
        self.staircase_i_hist = None
        self.staircase_v_hist = None

        if run_full_eis_post or run_full_eis_pre:
            eis_tag = f'EIS{self.eis_mode.upper()}'

            if z_guess is None:
                if self.eis_mode == 'galv':
                    # Assume step size is set to obtain v_rms of 10 mV
                    z_guess = 0.01 / abs(s_rms)
                elif self.eis_mode == 'pot':
                    # Arbitrary guess
                    z_guess = 1

        v_rms_target = self.staircase_v_rms_target

        try:
            if run_full_eis_pre:
                print('Running pre-staircase EIS')
                eis_file = os.path.join(data_path, '{}_{}_Pre.DTA'.format(eis_tag, file_suffix))
                eis_kst_file = os.path.join(kst_path, 'Kst_EIS.DTA')
                self.dt_eis.run(pstat, full_frequencies, s_init, abs(s_rms), z_guess, show_plot=show_plot,
                                result_file=eis_file, kst_file=eis_kst_file, timeout=1000)

                # Update z_guess
                z_guess = self.dt_eis.z_dataframe['Zmod'].values[0]

            for step in range(self.staircase_num_steps):
                print(f'Running staircase step {step}\n-----------------------')
                step_start = time.time()

                # Add step index to suffix
                step_suffix = '{}_{}'.format(file_suffix, step)

                # Configure current step
                if self.staircase_step_type in ['hzh', 'fz', 'fhzh']:
                    getattr(self, f'configure_{self.staircase_step_type}_step')(s_init, s_rms, t_init, t_step,
                                                                                t_sample, frequencies, z_guess)
                # elif self.staircase_step_type == 'fz':
                #     self.configure_fz_step(s_init, s_rms, t_init, t_step, t_sample, frequencies, z_guess)
                # elif self.staircase_step_type == 'fhzh':
                #     self.configure_fhzh_step(s_init, s_rms, t_init, t_step, t_sample, frequencies, z_guess)
                elif self.staircase_step_type == 'gzh':
                    t_long = t_step
                    t_short = self.staircase_geo_kwargs['t_short']
                    num_scales = self.staircase_geo_kwargs['num_scales']
                    steps_per_scale = self.staircase_geo_kwargs['steps_per_scale']
                    self.configure_gzh_step(s_init, s_rms, t_init, t_short, t_long, t_sample, num_scales,
                                            steps_per_scale, frequencies, z_guess)
                else:
                    raise ValueError(f'Invalid step_type {self.staircase_step_type} for run_staircase')

                # Run current step. Must leave cell on between measurements
                self.run(pstat, decimate, eis_first, data_path, kst_path, step_suffix, rest_time=rest_time,
                         start_with_cell_off=False, leave_cell_on=True, filter_response=filter_response,
                         show_plot=show_plot)

                print('Step {} time: {:.2f} s'.format(step, time.time() - step_start))

                # Increment s_init
                s_init = s_init + 2 * s_half_step

                # Record i and v in history
                if step == 0:
                    self.staircase_i_hist = [self.meas_i_start, self.meas_i_end]
                    self.staircase_v_hist = [self.meas_v_start, self.meas_v_end]
                else:
                    # This assumes that start values are same as previous end values.
                    # This should be valid for staircase step types (hzh, gzh) since meas_y_end
                    # is recorded after 2nd chrono step
                    self.staircase_i_hist.append(self.meas_i_end)
                    self.staircase_v_hist.append(self.meas_v_end)

                # Determine next step size
                if self.chrono_mode == 'galv' and self.update_step_size:
                    # Determine desired v_rms
                    if v_rms_target is None:
                        v_rms_target = (self.meas_v_max - self.meas_v_min) / (2 * np.sqrt(2))

                    print(v_rms_target)
                    print('Desired v_rms: {:.1f} mV'.format(1000 * v_rms_target))

                    s_rms = self.get_next_i_rms(s_init, v_rms_target)
                    s_half_step = s_rms * np.sqrt(2)
                    print('Last v_full_step: {:.3f} mV'.format(1000 * (self.meas_v_max - self.meas_v_min)))
                    print('New i_half_step: {:.3f} mA'.format(1000 * s_half_step))

                # Update z_guess
                z_guess = self.dt_eis.z_dataframe['Zmod'].values[0]

                # Check if voltage limits were exceeded
                if self.meas_v_min <= self.staircase_v_limits[0]:
                    print('Lower voltage limit reached: {:.3f} (measured) <= {:.3f} (threshold)'.format(
                        self.meas_v_min, self.staircase_v_limits[0])
                    )
                    break
                elif self.meas_v_max > self.staircase_v_limits[1]:
                    print('Upper voltage limit reached: {:.3f} (measured) >= {:.3f} (threshold)'.format(
                        self.meas_v_max, self.staircase_v_limits[1])
                    )
                    break

            if run_full_eis_post:
                print('Running post-staircase EIS')
                eis_file = os.path.join(data_path, '{}_{}_Post.DTA'.format(eis_tag, file_suffix))
                eis_kst_file = os.path.join(kst_path, 'Kst_EIS.DTA')
                self.dt_eis.run(pstat, full_frequencies, s_init, abs(s_rms), z_guess, show_plot=show_plot,
                                result_file=eis_file, kst_file=eis_kst_file, timeout=1000)

            if not leave_cell_on:
                # Turn cell off and close pstat
                pstat.SetCell(GamryCOM.CellOff)
                pstat.Close()
        except Exception as e:
            # If any exception thrown, turn cell off and close pstat
            pstat.SetCell(GamryCOM.CellOff)
            pstat.Close()
            raise gamry_error_decoder(e)
