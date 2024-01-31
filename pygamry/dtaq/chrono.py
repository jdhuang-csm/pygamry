from .. import signals
from .config import GamryCOM
from .eventsink import GamryDtaqEventSink
from ..utils import gamry_error_decoder, rel_round, check_control_mode, identify_steps
from ..file_utils import get_decimation_index
from ..filters.antialiasing import filter_chrono_signal

import comtypes.client as client
import numpy as np
import warnings


class DtaqChrono(GamryDtaqEventSink):
    def __init__(self, mode, write_mode='once', **init_kw):
        # Set enums based on specified mode
        check_control_mode(mode)
        self.mode = mode
        if mode == 'galv':
            self.gc_dtaqchrono_type = 'ChronoPot'
            self.gc_ctrlmode = 'GstatMode'
            self.input_column = 'Im'
            self.response_column = 'Vf'
            tag = 'CHRONOP'
            test_id = 'Chronopotentiometry Scan'
        else:
            self.gc_dtaqchrono_type = 'ChronoAmp'
            self.gc_ctrlmode = 'PstatMode'
            self.input_column = 'Vf'
            self.response_column = 'Im'
            tag = 'CHRONOA'
            test_id = 'Chronoamperometry Scan'

        # Define axes for real-time plotting
        axes_def = [
            {'title': 'Potential',
             'type': 'y(t)',
             'y_column': 'Vf',
             'y_label': '$V$ (V)'
             },
            {'title': 'Current',
             'type': 'y(t)',
             'y_column': 'Im',
             'y_label': '$i$ (A)'
             }
        ]

        # Initialize subclass attributes
        self.signal_args = None
        self.decimate = None
        self.decimate_during = None
        self.decimate_args = None
        self.filter_response = None

        # self.expected_duration = None
        self.v_oc = None
        self.i_max = None

        super().__init__('GamryDtaqChrono', tag, axes_def, test_id=test_id,
                         write_mode=write_mode, **init_kw)

    # ---------------------------------
    # Initialization and configuration
    # ---------------------------------
    def initialize_pstat(self):

        # Set pstat settings
        self.pstat.SetCtrlMode(getattr(GamryCOM, self.gc_ctrlmode))
        # self.pstat.SetCell(GamryCOM.CellOff)
        self.pstat.SetIEStability(GamryCOM.StabilityFast)
        self.pstat.SetCASpeed(2)  # CASpeedNorm
        self.pstat.SetSenseSpeedMode(True)
        # self.pstat.SetIConvention(GamryCOM.PHE200_IConvention)
        self.pstat.SetGround(GamryCOM.Float)
        # Dan Cook 11/30/22: Ich ranging can be disabled and range set to 3.0 for most DC experiments
        self.pstat.SetIchRange(3.0)
        self.pstat.SetIchRangeMode(False)
        self.pstat.SetIchOffsetEnable(False)
        self.pstat.SetIchFilter(2 / self.signal_params['t_sample'])
        # self.pstat.SetVchRange(10.0)
        self.pstat.SetVchRangeMode(False)
        self.pstat.SetVchOffsetEnable(False)
        self.pstat.SetVchFilter(2 / self.signal_params['t_sample'])
        # self.pstat.SetAchRange(3.0)
        # # self.pstat.SetIERangeLowerLimit(NIL)
        # self.pstat.SetIERange(0.03)  # This causes multi-step experiments to start at wrong voltage
        # self.pstat.SetIERangeMode(False)
        # self.pstat.SetAnalogOut(0.0)

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat, getattr(GamryCOM, self.gc_dtaqchrono_type))

        # Set run-time decimation
        if self.decimate_during == 'run' and self.decimate:
            self.dtaq.SetDecimation(True, *self.decimate_args)
        else:
            self.dtaq.SetDecimation(False, 10, 10)

        # Set signal
        self.pstat.SetSignal(self.signal)

        print(f'Chrono i_max: {self.i_max}')
        if self.i_max is not None:
            # Set IERange based on expected max current
            ie_range = self.pstat.TestIERange(self.i_max)
            self.pstat.SetIERange(ie_range)
            # Fixed current range
            self.pstat.SetIERangeMode(False)

            if self.mode == 'galv':
                # Must set voltage to obtain initial requested current based on IERange
                ie_resistor = self.pstat.IEResistor(ie_range)
                s_init = self.signal_params['s_init']
                v_init = s_init * ie_resistor
                # self.pstat.SetIchRange(ie_range)
                self.pstat.SetVoltage(v_init)
            print(f'Chrono IERange: {ie_range}')
        else:
            # Set to auto-range
            self.pstat.SetIERangeMode(True)

        # Measure OCV with cell off
        self.v_oc = 0
        if self.start_with_cell_off:
            if self.mode == 'pot':
                self.v_oc = self.pstat.MeasureV()
                self.pstat.SetVoltage(self.v_oc)  # Set voltage to OCV
            else:
                self.pstat.SetVoltage(0)  # Set voltage to 0, corresponding to zero current:

        # Turn cell on
        self.pstat.SetCell(GamryCOM.CellOn)

        if self.mode == 'galv':
            # Find Vch range
            print('Finding Vch range...')
            self.pstat.FindVchRange()

            # If lowest voltage range is selected, increment by 1 to prevent voltage truncation
            if self.pstat.VchRange() == 0:
                self.pstat.SetVchRange(1)
            print('Vch range:', self.pstat.VchRange())

    def configure_mstep_signal(self, s_init, s_stepsize, t_init, t_step, t_sample, n_steps):
        # Regular step seems to be broken - use Mstep as workaround
        self.signal = client.CreateObject('GamryCOM.GamrySignalMstep')
        
        # Round durations to sample interval
        t_init, t_step = round_to_sample_interval(t_sample, [t_init, t_step])

        # Store signal args for later user - signal.Init cannot be called until pstat specified
        self.signal_args = [s_init, s_stepsize, t_init, t_step, n_steps, t_sample, getattr(GamryCOM, self.gc_ctrlmode)]

        # Store signal parameters for reference
        self.signal_params = {
            'signal_class': 'Mstep',
            's_init': s_init,
            's_stepsize': s_stepsize,
            't_init': t_init,
            't_step': t_step,
            't_sample': t_sample,
            'n_steps': n_steps
        }

        self.expected_duration = t_init + t_step * n_steps
        # Mstep starts at 0, not -t_init
        self.data_time_offset = 0

        if self.mode == 'galv':
            self.i_req = s_init + s_stepsize * n_steps  # final current
            self.v_req = None
            self.i_max = max(abs(s_init), abs(self.i_req)) * 1.05  # 5% buffer
        elif self.mode == 'pot':
            self.i_req = None
            self.v_req = s_init + s_stepsize * n_steps  # final voltage

    def configure_dstep_signal(self, s_init, s_step1, s_step2, t_init, t_step1, t_step2, t_sample):
        self.signal = client.CreateObject('GamryCOM.GamrySignalDstep')

        # Round durations to sample interval
        t_init, t_step1, t_step2 = round_to_sample_interval(t_sample, [t_init, t_step1, t_step2])
        
        # Store signal args for later user - signal.Init cannot be called until pstat specified
        self.signal_args = [s_init, t_init, s_step1, t_step1, s_step2, t_step2, t_sample,
                            getattr(GamryCOM, self.gc_ctrlmode)]

        # Store signal parameters for reference
        self.signal_params = {
            'signal_class': 'Dstep',
            's_init': s_init,
            's_step1': s_step1,
            's_step2': s_step2,
            't_init': t_init,
            't_step1': t_step1,
            't_step2': t_step2,
            't_sample': t_sample,
        }

        self.expected_duration = t_init + t_step1 + t_step2
        # Set data time offset - measurement time starts at -t_init, first step is at time zero
        self.data_time_offset = -t_init

        if self.mode == 'galv':
            self.i_req = s_step2  # final current
            self.v_req = None
            self.i_max = np.max(np.abs([s_init, s_step1, s_step2])) * 1.05  # 5% buffer
        elif self.mode == 'pot':
            self.i_req = None
            self.v_req = s_step2  # final voltage

    def configure_triplestep_signal(self, s_init, s_rms, t_init, t_step, t_sample):
        self.signal = client.CreateObject('GamryCOM.GamrySignalArray')

        # Round durations to sample interval
        t_init, t_step = round_to_sample_interval(t_sample, [t_init, t_step])
        
        # Build the signal array
        times, signal, step_times = signals.make_triplestep_signal(s_init, s_rms, t_init, t_step, t_sample)

        # Store signal args for later user - signal.Init cannot be called until pstat specified
        self.signal_args = [1, t_sample, len(signal), signal.tolist(), getattr(GamryCOM, self.gc_ctrlmode)]

        # Store signal parameters for reference
        self.signal_params = {
            'signal_class': 'Array',
            's_init': s_init,
            's_rms': s_rms,
            't_init': t_init,
            't_sample': t_sample,
            'step_times': step_times
        }

        self.expected_duration = times[-1]
        # Set data time offset - measurement time starts at 0
        self.data_time_offset = 0

        if self.mode == 'galv':
            self.i_req = s_init  # final current
            self.v_req = None
            self.i_max = np.max(np.abs(signal)) * 1.05  # 5% buffer
        elif self.mode == 'pot':
            self.i_req = None
            self.v_req = s_init  # final voltage

    def configure_geostep_signal(self, s_init, s_final, s_min, s_max, t_init, t_sample, t_short, t_long,
                               num_scales, steps_per_scale, flex_thresh=0.05, end_at_init=False, end_time=None):
        self.signal = client.CreateObject('GamryCOM.GamrySignalArray')
        
        # Round durations to sample interval
        t_init, t_short, t_long = round_to_sample_interval(t_sample, [t_init, t_short, t_long])

        # Build the signal array
        times, signal, step_times = signals.make_geostep_signal(s_init, s_final, s_min, s_max,
                                                                t_init, t_sample, t_short, t_long,
                                                                num_scales, steps_per_scale,
                                                                flex_thresh=flex_thresh,
                                                                end_at_init=end_at_init, end_time=end_time)

        # Store signal args for later user - signal.Init cannot be called until pstat specified
        self.signal_args = [1, t_sample, len(signal), signal.tolist(), getattr(GamryCOM, self.gc_ctrlmode)]

        # Store signal parameters for reference
        self.signal_params = {
            'signal_class': 'Array',
            's_init': s_init,
            's_final': s_final,
            's_min': s_min,
            's_max': s_max,
            't_init': t_init,
            't_short': t_short,
            't_long': t_long,
            't_sample': t_sample,
            'num_scales': num_scales,
            'steps_per_scale': steps_per_scale,
            'step_times': step_times
        }

        self.expected_duration = times[-1]
        # Set data time offset - measurement time starts at 0
        self.data_time_offset = 0

        if self.mode == 'galv':
            self.i_req = s_final  # final current
            self.v_req = None
            self.i_max = np.max(np.abs(signal)) * 1.05  # 5% buffer
        elif self.mode == 'pot':
            self.i_req = None
            self.v_req = s_final  # final voltage

    def configure_decimation(self, decimate_during, prestep_points, decimation_interval,
                             decimation_factor=None, max_t_sample=None):
        """
        Configure decimation. Decimation may be applied either at runtime by adjusting the potentiostat's sampling rate
        or at time of file writing by subsampling the collected data.
        :param str decimate_during: method by which to achieve decimation. Options: run, write
        :param int prestep_points: number of points prior to first step to keep
        :param int decimation_interval: number of points after which to increment decimation
        :param float decimation_factor: factor by which to change sample period at each decimation interval.
        Only applies when decimate_during=write
        :param float max_t_sample: maximum time between samples. Decimation will be applied until the sample period
        reaches max_t_sample, after which the sample period will be constant. Only applies when decimate_during=write
        :return:
        """
        # Check method
        if decimate_during not in ('write', 'run'):
            raise ValueError(f'Invalid decimate_during {decimate_during}. Options: run, write')
        elif decimate_during == 'write' and self.write_mode != 'once':
            raise ValueError('Decimation during write is only available when write_mode=''once''')
        elif decimate_during == 'write' and decimation_factor is None:
            raise ValueError('Decimation factor must be specified when decimate_during=''write''')
        elif decimate_during == 'run' and self.signal_params.get('t_sample', 1) < 1e-4:
            warnings.warn('Runtime decimation is not functional when sampling period is shorter than 1e-4 s')

        self.decimate_during = decimate_during

        if decimate_during == 'run':
            self.decimate_args = [prestep_points, decimation_interval]
        elif decimate_during == 'write':
            self.decimate_args = [prestep_points, decimation_interval, decimation_factor, max_t_sample]

    def get_step_times(self, include_erroneous: bool = True):
        """Determine step times from signal"""
        if self.signal_params['signal_class'] == 'Dstep':
            # Step occurs 1 sample period AFTER programmed time
            step_times = np.array([0, self.signal_params['t_step1']]) + 0.99 * self.signal_params['t_sample']
        elif self.signal_params['signal_class'] == 'Mstep':
            step_times = self.signal_params['t_init'] + \
                         np.arange(0, self.signal_params['n_steps']) * self.signal_params['t_step']
            # # Step occurs 1 sample period AFTER programmed time (? need to check if also true for mstep)
            # step_times += self.signal_params['t_sample']
        elif self.signal_params['signal_class'] == 'Array':
            step_times = self.signal_params['step_times']
        else:
            step_times = None
            
        if include_erroneous:
            # Check for any erroneous (unprogrammed) steps 
            # (these sometimes arise when configured current is very small)
            meas_step_index = identify_steps(
                self.data_array[:, self.cook_columns.index(self.input_column)], 
                allow_consecutive=False
            )
            
            meas_step_times = self.data_array[meas_step_index, self.cook_columns.index('Time')]
            
            if step_times is None:
                step_times = []
                
            # Get the superset of programmed and erroneous steps
            step_times = np.unique(np.concatenate([step_times, meas_step_times]))

        return step_times

    def get_step_index(self):
        step_times = self.get_step_times()
        times = self.data_array[:, self.cook_columns.index('Time')]

        def pos_delta(x, x0):
            out = np.empty(len(x))
            out[x < x0] = np.inf
            out[x >= x0] = x[x >= x0] - x0
            return out

        return np.unique([np.argmin(pos_delta(times, st)) for st in step_times])

    def set_signal(self):
        try:
            self.signal.Init(self.pstat, *self.signal_args)
        except Exception as e:
            raise gamry_error_decoder(e)

    # ---------------------------------
    # Run
    # ---------------------------------
    def run(self, pstat, timeout=None, decimate=False, i_max=None,
            result_file=None, kst_file=None, append_to_file=False,
            show_plot=False, plot_interval=None, filter_response=False, **run_kw):

        # Checks
        if self.signal_args is None:
            raise RuntimeError('Signal must be configured before calling run')

        if decimate and self.decimate_during is None:
            raise RuntimeError('Decimation must be configured prior to calling run if decimate=True')
        elif decimate and self.decimate_during == 'run' and self.signal_params.get('t_sample', 1) < 1e-4:
            warnings.warn('Runtime decimation is not functional when sampling period is shorter than 1e-4 s')

        if filter_response and not decimate:
            warnings.warn('Filtering can only be applied with decimate=True')

        if i_max is not None and self.mode == 'galv':
            warnings.warn('User-specified i_max will be ignored in galvanostatic mode')
        elif i_max is None and self.mode == 'pot':
            warnings.warn('For best results in potentiostatic mode, expected max current should be supplied to i_max')

        if self.signal_params['t_sample'] < 1e-2 and self.write_mode != 'once':
            warnings.warn("When chrono sample period is short, write_mode should be set to 'once' "
                          "to ensure that data collection is not interrupted")

        # Store expected max absolute current in potentiostatic mode.
        # Ignored in galvanostatic mode - max current is determined from signal configuration
        if self.mode == 'pot':
            self.i_max = i_max

        self.pstat = pstat
        self.set_signal()

        self.decimate = decimate
        self.filter_response = filter_response

        if timeout is None:
            timeout = self.expected_duration + 30

        if plot_interval is None:
            # By default, set plot interval such that plot will render AFTER measurement complete
            # This avoids interfering with measurement when sampling rate is fast
            plot_interval = self.expected_duration + 10

        super().run_main(pstat, result_file, kst_file, append_to_file, timeout=timeout, show_plot=show_plot,
                         plot_interval=plot_interval, **run_kw)

    # --------------------
    # Header
    # --------------------
    def get_dtaq_header(self):
        text = 'CONVENTION\tIQUANT\t1\tCurrent Convention\n' + \
               f'EOC\tQUANT\t{rel_round(self.v_oc, self.write_precision)}\tOpen Circuit(V)\n'

        if self.mode == 'pot':
            s_label = 'V'
        else:
            s_label = 'I'

        add_info = []
        if self.signal_params['signal_class'] == 'Dstep':
            add_info += [
                ['TPRESTEP', 'QUANT', '{:.4e}'.format(self.signal_params['t_init']), 'Pre-step Delay Time (s)'],
                [f'{s_label}PRESTEP', 'QUANT', '{:.4e}'.format(self.signal_params['s_init']),
                 'Pre-step Delay Time (s)'],
                ['TSTEP1', 'QUANT', '{:.4e}'.format(self.signal_params['t_step1']), 'Step 1 Time (s)'],
                [f'{s_label}STEP1', 'QUANT', '{:.4e}'.format(self.signal_params['s_step1']), 'Step 1 value'],
                ['TSTEP2', 'QUANT', '{:.4e}'.format(self.signal_params['t_step2']), 'Step 2 Time (s)'],
                [f'{s_label}STEP2', 'QUANT', '{:.4e}'.format(self.signal_params['s_step2']), 'Step 2 value'],
            ]
        elif self.signal_params['signal_class'] == 'Mstep':
            add_info += [
                ['TPRESTEP', 'QUANT', '{:.4e}'.format(self.signal_params['t_init']), 'Pre-step Delay Time (s)'],
                [f'{s_label}PRESTEP', 'QUANT', '{:.4e}'.format(self.signal_params['s_init']),
                 'Pre-step Delay Time (s)'],
                ['TSTEP', 'QUANT', '{:.4e}'.format(self.signal_params['t_step']), 'Step Time (s)'],
                [f'{s_label}STEP', 'QUANT', '{:.4e}'.format(self.signal_params['s_stepsize']), 'Step size'],
            ]

        text += '\n'.join(['\t'.join(ai) for ai in add_info]) + '\n'

        return text

    # -----------------------------
    # File writing with decimation
    # ----------------------------
    @property
    def decimate_index(self):
        # Get measurement times and step times
        times = self.data_array[:, 0]
        step_times = self.get_step_times()

        # Get decimated data indices
        decimate_index = get_decimation_index(times, step_times, self.signal_params['t_sample'],
                                              *self.decimate_args)

        return decimate_index

    @property
    def decimated_data_array(self):
        return self.data_array[self.decimate_index]

    @property
    def decimated_dataframe(self):
        return self.dataframe.loc[self.decimate_index, :]

    def write_to_file(self, data_func, destination_file, active_file, last_write_index,
                      new_count, is_final_write, indent):
        if destination_file is not None and is_final_write and \
                self.decimate and self.decimate_during == 'write' and self.write_mode == 'once':
            # Write decimated data to file
            decimate_index = self.decimate_index

            # Get full dataframe and then decimate
            df = data_func(0, self.total_points)

            # Filter response signal
            if self.filter_response:
                df = df.copy()
                t = df['Time'].values
                # input_signal = df[self.input_column].values
                y = df[self.response_column].values
                y_filt = filter_chrono_signal(t, y, step_index=self.get_step_index(), decimate_index=decimate_index,
                                              remove_outliers=True, max_sigma=100, sigma_factor=0.01)
                df[self.response_column] = y_filt

            df = df.loc[decimate_index, :]
            if self.append_to_file:
                # Apply offsets
                df = df.set_index(df.index + self.start_index_from)
                df['Time'] += self.file_time_offset

            data_string = df.to_csv(None, sep='\t', header=False, lineterminator='\n',
                                    float_format=f'%.{self.write_precision + 1}g')

            # Pad left with tabs
            if indent:
                data_string = '\t' + data_string.replace('\n', '\n\t')[:-1]

            with open(destination_file, 'a') as f:
                f.write(data_string)
            last_write_index += self.total_points
            return last_write_index
        else:
            return super().write_to_file(data_func, destination_file, active_file, last_write_index,
                                         new_count, is_final_write, indent)

    # Analysis
    # ---------
    def estimate_r_tot(self, window=None):
        """
        Estimate total resistance from step data
        :return:
        """
        vf = self.data_array[:, self.cook_columns.index('Vf')]
        im = self.data_array[:, self.cook_columns.index('Im')]
        times = self.data_array[:, self.cook_columns.index('Time')]

        step_v = []
        step_i = []

        start_time = times[0]
        for end_time in np.concatenate([self.get_step_times(), [times[-1] + 1e-10]]):
            if window is None:
                # Take the last 5% of the step duration
                step_duration = end_time - start_time
                window_index = np.where((times >= start_time + step_duration * 0.95) & (times < end_time))
                if len(window_index[0]) == 0:
                    window_index = np.where(times < end_time)[0][-1]
                # print(f'window length: {len(window_index[0])}')
                step_v.append(np.median(vf[window_index]))
                step_i.append(np.median(im[window_index]))
            else:
                # Take the last <window> points of each step
                step_index = np.where((times >= start_time) & (times < end_time))
                step_v.append(np.median(vf[step_index][-window:]))
                step_i.append(np.median(im[step_index][-window:]))
            start_time = end_time

        print('step_v:', step_v)
        print('step_i:', step_i)

        delta_v = np.diff(step_v)
        delta_i = np.diff(step_i)
        r_est = np.median(delta_v / delta_i)
        print('R_est: {:.3f} ohms'.format(r_est))

        # v_high = np.percentile(vf, 97.5)
        # v_low = np.percentile(vf, 2.5)
        # i_high = np.percentile(im, 97.5)
        # i_low = np.percentile(im, 2.5)
        # delta_v = v_high - v_low
        # delta_i = i_high - i_low

        # vf = np.sort(self.data_array[:, self.cook_columns.index('Vf')])
        # im = np.sort(self.data_array[:, self.cook_columns.index('Im')])
        # delta_v = np.mean(vf[-window:]) - np.mean(vf[:3])
        # delta_i = np.mean(im[-3:]) - np.mean(im[:3])

        return r_est

    def _get_cook_values(self, name):
        try:
            return self.data_array[:, self.cook_columns.index(name)]
        except ValueError:
            raise ValueError(f'Invalid value name {name}. Must be one of dtaq cook columns')

    def _get_step_end_vals(self, name, window):
        times = self._get_cook_values('Time')
        y = self._get_cook_values(name)
        step_y = []

        start_time = times[0]
        for end_time in np.concatenate([self.get_step_times(), [times[-1] + 1e-10]]):
            if window is None:
                # Take the last 5% of the step duration
                step_duration = end_time - start_time
                window_index = np.where((times >= start_time + step_duration * 0.95) & (times < end_time))
                if len(window_index[0]) == 0:
                    window_index = np.where(times < end_time)[0][-1]
                # print(f'window length: {len(window_index[0])}')
                step_y.append(np.median(y[window_index]))
            else:
                # Take the last <window> points of each step
                step_index = np.where((times >= start_time) & (times < end_time))
                step_y.append(np.median(y[step_index][-window:]))
            start_time = end_time

        return step_y

    def _get_val_init(self, name, window):
        y = self._get_cook_values(name)

        if window is None:
            # Get median voltage from last 5% of final step duration
            times = self.data_array[:, self.cook_columns.index('Time')]
            first_step_start = self.get_step_times()[0]
            return np.median(y[times < first_step_start])
        else:
            return np.median(y[:window])

    def _get_val_final(self, name, window):
        y = self._get_cook_values(name)

        if window is None:
            # Get median voltage from last 5% of final step duration
            times = self.data_array[:, self.cook_columns.index('Time')]
            last_step_start = self.get_step_times()[-1]
            last_step_duration = times[-1] - last_step_start
            return np.median(y[times >= last_step_start + 0.95 * last_step_duration])
        else:
            return np.median(y[-window:])

    def _get_val_limits(self, name, percentiles):
        y = self._get_cook_values(name)

        if percentiles is not None:
            v_min = np.percentile(y, percentiles[0])
            v_max = np.percentile(y, percentiles[1])
        else:
            v_min = np.min(y)
            v_max = np.max(y)

        return v_min, v_max

    def get_v_limits(self, percentiles=None):
        return self._get_val_limits('Vf', percentiles)

    def get_i_limits(self, percentiles=None):
        return self._get_val_limits('Im', percentiles)

    def get_v_init(self, window=None):
        return self._get_val_init('Vf', window)

    def get_v_final(self, window=None):
        return self._get_val_final('Vf', window)

    def get_i_init(self, window=None):
        return self._get_val_init('Im', window)

    def get_i_final(self, window=None):
        return self._get_val_final('Im', window)

    def get_i_step_end(self, window=None):
        return self._get_step_end_vals('Im', window)

    def get_v_step_end(self, window=None):
        return self._get_step_end_vals('Vf', window)
    
    
def round_to_sample_interval(t_sample, time_list):
    return [round(t / t_sample, 0) * t_sample for t in time_list]
        