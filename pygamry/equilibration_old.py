import numpy as np

from sqeis.square_drt import DRT
from sqeis.matrices import construct_response_matrix
from sqeis.preprocessing import get_step_info, generate_model_signal

from .dtaq import GamryDtaqEventSink


def identify_response_step(times, response_signal, offset_step_time):
    # Estimate step time from observed response
    # Start with large rthresh for step identification. Decrease as necessary to identify step
    rthresh = 20
    while True:
        try:
            step_times, step_sizes = get_step_info(times, response_signal, allow_consecutive=False,
                                                   offset_step_times=offset_step_time, rthresh=rthresh)
            break
        except IndexError:
            rthresh /= 2

    # Only keep first step
    step_time = step_times[0]
    step_size = step_sizes[0]

    # # Assume that input_signal step occurred 1 sample before it was observed in the response
    # if offset_step_time:
    #     t_sample = np.min(np.diff(times))
    #     step_time -= t_sample

    return step_time, step_size

# def generate_dummy_signal(times, response_signal):
#     # Estimate step time from observed response
#     step_times, response_step_sizes = get_step_info(times, response_signal, allow_consecutive=False)
#     # Only keep first step
#     step_times = step_times[0:1]
#
#     # Set a dummy input_signal step size with same sign as response step
#     # This ensures that the relaxation can be fitted with a non-negative DRT
#     step_sizes = [1.0] * np.sign(response_step_sizes[0])
#
#     # Generate ideal dummy signal
#     dummy_signal = generate_model_signal(times, step_times, step_sizes, None, 'ideal')
#
#     return dummy_signal, step_times


def get_downsample_kwargs(times, step_time, max_num_samples, prestep_samples=5, spacing='log'):
    """
    Get index of samples to use for fit given limited number of samples
    :param times:
    :param step_times:
    :param max_num_samples:
    :param int prestep_samples: Number of samples prior to first step to include. Default: 5
    :param str spacing: 'log' or 'linear'
    :return:
    """
    # Since we are more concerned with the end value and less concerned with short-timescale relaxations,
    # use linearly spaced samples rather than logarithmically spaced samples.
    poststep_samples = max_num_samples - prestep_samples

    if spacing == 'log':
        min_tdelta = times[times > step_time][0] - step_time
        max_tdelta = times[times > step_time][-1] - step_time
        ideal_times = np.logspace(np.log10(min_tdelta), np.log10(max_tdelta), poststep_samples - 1)
        ideal_times = np.concatenate([[0], ideal_times])  # include 0
    elif spacing == 'linear':
        min_tdelta = times[times >= step_time][0] - step_time
        max_tdelta = times[times >= step_time][-1] - step_time
        ideal_times = np.linspace(min_tdelta, max_tdelta, poststep_samples)
    else:
        raise ValueError(f"Invalid spacing option {spacing}. Options are 'log', 'linear'")

    return {'prestep_samples': prestep_samples, 'ideal_times': ideal_times}


class EquilibrationCheckDRT(DRT):
    def __init__(self, basis_tau=None, tau_basis_type='Cole-Cole', tau_epsilon=0.995,
                 step_model='ideal', op_mode='galv',
                 fit_inductance=False, time_precision=3):

        super().__init__(basis_tau, tau_basis_type, tau_epsilon,
                         step_model, op_mode, fit_inductance, time_precision)

        self.dummy_input_signal = None

    def check_for_equilibration(self, times, response_signal, rtol, check_window=10, max_num_samples=50,
                                prestep_samples=5,
                                offset_response_step_time=True,
                                **ridge_kw):

        # Create dummy input signal
        step_time, response_step_size = identify_response_step(times, response_signal, offset_response_step_time)
        # Set the dummy step size with same sign as response step
        # This ensures that the relaxation can be fitted with a non-negative DRT
        dummy_step_sizes = [1.0 * np.sign(response_step_size)]
        self.dummy_input_signal = generate_model_signal(times, [step_time], dummy_step_sizes, None, 'ideal')

        # Set i_signal and v_signal based on mode
        if self.op_mode == 'galv':
            i_signal = self.dummy_input_signal.copy()
            v_signal = response_signal.copy()
        elif self.op_mode == 'pot':
            i_signal = response_signal.copy()
            v_signal = self.dummy_input_signal.copy()

        # Set basis_tau values based on times measured so far
        self.basis_tau = self.get_tau_from_times(np.concatenate([times, times[-1:] * 10]), [step_time], ppd=50)

        # Generate kwargs to limit number of data points used in fit
        ds_kwargs = get_downsample_kwargs(times, step_time, max_num_samples, prestep_samples=prestep_samples)

        # Perform ridge fit
        ridge_defaults = dict(hyper_l2_lambda=True, nonneg=True, l2_lambda_0=1e-20, hl_l2_beta=2.5)
        ridge_defaults.update(ridge_kw)
        self.ridge_fit(times, i_signal, v_signal,
                           downsample=True, downsample_kw=ds_kwargs,
                           **ridge_defaults)

        # Get projected end value
        projected_end_value = self.predict_response(np.array([1e6]))[0]

        # Get mean measured value over last check_window measurements
        check_value = np.mean(response_signal[-check_window:])

        # Get projected change once fully equilibrated
        projected_delta = projected_end_value - np.mean(response_signal[times < step_time][-5:])

        # Check if current mean measured value is within rtol of projected end value
        if np.abs((projected_end_value - check_value) / projected_delta) < rtol:
            check_result = True
        else:
            check_result = False

        return check_result, projected_end_value, projected_delta

    def predict_equilibration_time(self, rtol):
        """
        Estimate projected end time
        :param rtol:
        :return:
        """
        # Identify first (shortest) time constant for which the sum of all previous coefficients is
        # within rtol of the total Rp
        coef = self.fit_parameters['x']
        R_tot = np.sum(coef) + self.fit_parameters['R_inf']
        cum_frac = (self.fit_parameters['R_inf'] + np.cumsum(coef)) / R_tot
        end_tau_index = np.where(cum_frac >= 1 - rtol)[0][0]
        tau_end = self.basis_tau[end_tau_index]

        return tau_end * 6  # when time = 6 * tau, should be within ~ 0.1% of end value


# --------------------------------------------------------------------
# Equilibration Dtaq
# --------------------------------------------------------------------
class DtaqEquilibration(GamryDtaqEventSink):
    def __init__(self, mode='galv'):

        # Set enums based on specified mode
        self.mode = mode
        if mode == 'galv':
            self.gc_dtaqchrono_type = 'ChronoPot'
            self.gc_ctrlmode = 'GstatMode'
            self.input_column = 'Im'
            self.response_column = 'Vf'
            tag = 'CHRONOP'
        elif mode == 'pot':
            self.gc_dtaqchrono_type = 'ChronoAmp'
            self.gc_ctrlmode = 'PstatMode'
            self.input_column = 'Vf'
            self.response_column = 'Im'
            tag = 'CHRONOA'

        # Create DRT instance for fitting equilibration curve
        # self.drt = DRT(tau_basis_type='zga', time_precision=3, tau_epsilon=4.34, op_mode=mode)
        self.drt = EquilibrationCheckDRT(tau_basis_type='Cole-Cole', tau_epsilon=0.995, op_mode=mode)
        # self.drt.set_zga_params(num_bases=7, nonneg=True)

        # Subclass-specific attributes
        self.first_check_time = None
        self.check_interval = None
        self.last_check_time = None
        self.max_num_samples = None
        self.check_window = None
        self.consecutive_checks = None
        self.check_history = None
        self.rtol = None

        self.projected_end_value = None
        self.projected_delta = None
        self.projection_times = None
        self.projected_signal = None

        super().__init__('GamryDtaqChrono', tag)

    @property
    def response_label_units(self):
        if self.mode == 'galv':
            return '$v$', 'V'
        elif self.mode == 'pot':
            return '$i$', 'A'

    @property
    def input_label_units(self):
        if self.mode == 'galv':
            return '$i$', 'A'
        elif self.mode == 'pot':
            return '$v$', 'V'

    def set_signal(self, pstat, s_init=0, t_init=0.1, s_final=1e-3, t_final=10, t_sample=0.1):
        # Regular step seems to be broken - use Mstep as workaround
        self.signal = client.CreateObject('GamryCOM.GamrySignalMstep')
        self.signal.Init(pstat,
                         s_init,  # Sinit
                         s_final,  # Sstep
                         t_init,  # Tinit
                         t_final,  # Tstep
                         1,  # Nstep
                         t_sample,  # SampleRate
                         getattr(GamryCOM, self.gc_ctrlmode),  # CtrlMode
                         )

        # Store signal parameters for reference
        self.signal_params = {
            's_init': s_init,
            's_final': s_final,
            't_init': t_init,
            't_final': t_final,
            't_sample': t_sample
        }

    # Subclass-specific methods
    def initialize_pstat(self):
        # Set pstat settings
        self.pstat.SetCtrlMode(getattr(GamryCOM, self.gc_ctrlmode))
        self.pstat.SetIEStability(GamryCOM.StabilityFast)
        self.pstat.SetCASpeed(2)  # CASpeedNorm
        self.pstat.SetSenseSpeedMode(True)
        # self.pstat.SetIConvention(GamryCOM.PHE200_IConvention)
        self.pstat.SetGround(GamryCOM.Float)
        self.pstat.SetIchRange(3.0)
        self.pstat.SetIchRangeMode(True)
        self.pstat.SetIchOffsetEnable(False)
        self.pstat.SetIchFilter(1 / self.signal_params['t_sample'])
        self.pstat.SetVchRange(10.0)
        self.pstat.SetVchRangeMode(True)
        self.pstat.SetVchOffsetEnable(False)
        self.pstat.SetVchFilter(1 / self.signal_params['t_sample'])
        self.pstat.SetAchRange(3.0)
        # self.pstat.SetIERangeLowerLimit(NIL)
        self.pstat.SetIERange(0.03)
        self.pstat.SetIERangeMode(False)
        self.pstat.SetAnalogOut(0.0)
        self.pstat.SetVoltage(0.0)

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat, getattr(GamryCOM, self.gc_dtaqchrono_type))

        # Set signal
        self.pstat.SetSignal(self.signal)

        # Turn cell on
        self.pstat.SetCell(GamryCOM.CellOn)

    def run_plot_animation(self, plot_interval=0.5):
        # Make figure and LiveAxes instances
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        lax1 = LiveAxes(axes[0])
        lax2 = LiveAxes(axes[1], fixed_ylim=(-5, 105))

        # Set input signal axis limits based on signal parameters
        s_range = self.signal_params['s_final'] - self.signal_params['s_init']
        lax1.ax.set_ylim(self.signal_params['s_init'] - s_range * 0.1,
                         self.signal_params['s_final'] + s_range * 0.1)

        # Define functions to update data for each artist
        # def update_input_signal(frame):
        #     x = self.data_array[:, self.cook_columns.index('Time')]
        #     y = self.data_array[:, self.cook_columns.index(self.input_column)]
        #     return x, y

        def update_response_signal(frame):
            x = self.data_array[:, self.cook_columns.index('Time')]
            y = self.data_array[:, self.cook_columns.index(self.response_column)]
            return x, y

        def update_projected_signal(frame):
            x = self.projection_times
            y = self.projected_signal
            return x, y

        def update_pct_to_equil(frame):
            x = self.data_array[:, self.cook_columns.index('Time')]
            y_meas = self.data_array[:, self.cook_columns.index(self.response_column)]
            y_proj = self.projected_end_value
            deno = self.projected_delta
            y = 100 * (y_proj - y_meas) / deno
            return x, y

        # Initialize projected signal so that response plot will render prior to first check
        self.projection_times = [0]
        self.projected_signal = [0]
        self.projected_end_value = np.nan
        self.projected_delta = np.nan

        # add artists to LiveAxes
        # lax1.add_line_artist('input', update_input_signal, marker='.', ms=10, alpha=0.5, ls='')
        # lax1.ax.set_xlabel('Time (s)')
        # lax1.ax.set_ylabel('{} ({})'.format(*self.input_label_units))

        lax1.add_line_artist('response', update_response_signal, marker='.', ms=10, alpha=0.5, ls='', label='Data')
        lax1.add_line_artist('projection', update_projected_signal, c='k', label='Projected')
        lax1.ax.set_xlabel('Time (s)')
        lax1.ax.set_ylabel('{} ({})'.format(*self.response_label_units))

        lax2.add_line_artist('pct_to_equil', update_pct_to_equil, marker='.', ms=10, alpha=0.5, ls='')
        lax2.ax.set_xlabel('Time (s)')
        lax2.ax.axhline(100 * self.rtol, c='k', ls='--', lw=1, zorder=-10)
        lax2.ax.set_ylabel('$v_{\mathrm{projected}} - v_{\mathrm{meas}}$ (%)')

        fig.tight_layout()

        lfig = LiveFigure([lax1, lax2])

        return lfig.run()

    def run(self, pstat, first_check_time=10, check_interval=5, max_num_samples=50, check_window=10,
            consecutive_checks=3, rtol=5e-3, show_plot=False, plot_interval=0.5):

        # Store check parameters as attributes so that they can be accessed by IGamryDtaqEvents_OnDataAvailable
        self.first_check_time = first_check_time
        self.last_check_time = 0
        self.check_interval = check_interval
        self.max_num_samples = max_num_samples
        self.check_window = check_window
        self.consecutive_checks = consecutive_checks
        self.rtol = rtol

        # Create check history
        self.check_history = []

        # Infer timeout from signal duration
        timeout = self.signal_params['t_final'] + 30

        super().run(pstat, timeout=timeout, show_plot=show_plot, plot_interval=plot_interval)

    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        # print('Data Available')
        new_count = self.cook()

        if self.show_plot:
            # Must incorporate plt.pause to run GUI event loop and allow artists to be drawn
            plt.pause(1e-5)

        # Check for equilibration
        last_measured_time = self.data_array[-1, 0]  # np.array(self.acquired_points)[-1, 0]
        if last_measured_time > self.first_check_time \
                and last_measured_time - self.last_check_time >= self.check_interval:
            print('Check at time {:.1f} s'.format(last_measured_time))
            self.check_for_equilibration()
            self.last_check_time = last_measured_time

    def check_for_equilibration(self):
        start = time.time()

        # Get current data
        data = self.data_array
        times = data[:, self.cook_columns.index('Time')]
        i_signal = data[:, self.cook_columns.index('Im')]
        v_signal = data[:, self.cook_columns.index('Vf')]

        if self.mode == 'galv':
            response_signal = v_signal
        elif self.mode == 'pot':
            response_signal = i_signal

        check_result, projected_end_value, projected_delta = \
            self.drt.check_for_equilibration(times, response_signal, self.rtol, self.check_window, self.max_num_samples)

        self.projected_end_value = projected_end_value
        self.projected_delta = projected_delta

        # Append current check status to history
        self.check_history.append(check_result)

        # If last consecutive_checks checks have all been successful, consider sample to be equilibrated
        if np.sum(self.check_history[-self.consecutive_checks:]) == self.consecutive_checks:
            equilibrated = True
        else:
            equilibrated = False

        # Estimate projected end time
        # Identify first (shortest) time constant for which the sum of all previous coefficients is
        # within rtol of the total Rp
        equil_time = self.drt.predict_equilibration_time(self.rtol)
        equil_time = max(equil_time, 5)  # Don't allow extremely short equilibration times

        equil_time += self.drt.step_times[0]
        # If estimated equil time is less than last measured time, use last measured time
        equil_time = max(equil_time, times[-1])

        if self.show_plot:
            # Calculate projected signal to update plots
            self.projection_times = np.concatenate(
                [times[0:1],  # first time
                 self.drt.step_times[0:1] - 1e-3,  # immediately before step
                 self.drt.step_times[0:1],  # step time
                 self.drt.step_times[0] + np.logspace(-1, np.log10(equil_time), 50)]  # log-spaced post-step times
            )
            self.projected_signal = self.drt.predict_response(self.projection_times)

        print('------------------------------')
        print('Predicted end value: {:.3e}'.format(projected_end_value))
        # print('Current value: {:.3e}'.format(check_val))
        print('equil time:', equil_time)
        print(f'check_result={check_result}')
        print(f'Equilibrated={equilibrated}')
        print('Check run time: {:.2f}'.format(time.time() - start))
        print('------------------------------')

        if equilibrated:
            self.terminate()




