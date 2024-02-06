from .config import GamryCOM
from .eventsink import GamryDtaqEventSink
from ..utils import gamry_error_decoder, rel_round, check_control_mode


import comtypes.client as client
import numpy as np


class DtaqOcv(GamryDtaqEventSink):
    def __init__(self, **init_kw):
        # Define axes for real-time plotting
        axes_def = [
            {'title': 'OCV',
             'type': 'y(t)',
             'y_column': 'Vf',
             'y_label': 'OCV (V)'
             }
        ]

        # Initialize subclass attributes
        # self.expected_duration = None

        super().__init__('GamryDtaqOcv', 'CORPOT', axes_def,
                         test_id='Open Circuit Potential', **init_kw)

    # ---------------------------------
    # Initialization and configuration
    # ---------------------------------
    def initialize_pstat(self):
        self.pstat.SetCtrlMode(GamryCOM.PstatMode)
        # self.pstat.SetCell(GamryCOM.CellOff)
        self.pstat.SetIEStability(GamryCOM.StabilityNorm)

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat)

        # Set signal
        self.pstat.SetSignal(self.signal)

        # Leave cell off for OCV measurement
        self.pstat.SetCell(GamryCOM.CellOff)

    def set_signal(self, duration, t_sample):
        """
        Create signal object
        :param duration:
        :param t_sample:
        :return:
        """
        try:
            self.signal = client.CreateObject('GamryCOM.GamrySignalConst')
            self.signal.Init(self.pstat,
                             0,  # voltage
                             duration,
                             t_sample,
                             GamryCOM.PstatMode,  # CtrlMode
                             )

            # Store signal parameters for reference
            self.signal_params = {
                'duration': duration,
                't_sample': t_sample
            }
        except Exception as e:
            raise gamry_error_decoder(e)

    # ---------------------------------
    # Run
    # ---------------------------------
    def run(self, pstat, duration, t_sample, timeout=None, result_file=None, kst_file=None, append_to_file=False,
            show_plot=False, plot_interval=None):
        self.pstat = pstat
        self.set_signal(duration, t_sample)

        self.expected_duration = duration

        if timeout is None:
            timeout = duration + 30

        if plot_interval is None:
            plot_interval = t_sample * 0.9

        super().run_main(pstat, result_file, kst_file, append_to_file, timeout=timeout, show_plot=show_plot,
                         plot_interval=plot_interval)

    def get_ocv(self, window=10):
        """
        Get mean OCV from last [window] data points
        """
        return np.mean(self.data_array[-window:, self.cook_columns.index('Vf')])