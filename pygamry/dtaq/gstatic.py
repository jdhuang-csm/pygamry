from .config import GamryCOM
from .eventsink import GamryDtaqEventSink
from ..utils import gamry_error_decoder, rel_round, check_control_mode


import comtypes.client as client


class DtaqGstatic(GamryDtaqEventSink):
    def __init__(self, **init_kw):
        # Define axes for real-time plotting
        axes_def = [
            {'title': 'Voltage',
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

        # Subclass-specific attributes
        self.v_oc = None
        self.v_min = None
        self.v_max = None

        super().__init__('GamryDtaqCiiv', 'GALVANOSTATIC', axes_def,
                         test_id='Galvanostatic Scan', **init_kw)

    # ---------------------------------
    # Initialization and configuration
    # ---------------------------------
    def initialize_pstat(self):
        self.pstat.SetCtrlMode(GamryCOM.GstatMode)
        self.pstat.SetIEStability(GamryCOM.StabilityNorm)

        # Set IE range based on requested current
        self.pstat.SetIERange(self.pstat.TestIERange(abs(self.i_req) * 1.05))
        self.pstat.SetIERangeMode(False)

        # Dan Cook 11/30/22: Ich ranging can be disabled and range set to 3.0 for most DC experiments
        self.pstat.SetIchRange(3.0)
        self.pstat.SetIchRangeMode(False)
        self.pstat.SetIchOffsetEnable(False)
        self.pstat.SetIchFilter(2 / self.signal_params['t_sample'])
        self.pstat.SetVchRangeMode(False)
        self.pstat.SetVchOffsetEnable(False)
        self.pstat.SetVchFilter(2 / self.signal_params['t_sample'])

        # Measure OCV with cell off
        if self.start_with_cell_off:
            self.v_oc = self.pstat.MeasureV()
        else:
            self.v_oc = 0

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat)

        # Set signal
        self.pstat.SetSignal(self.signal)

        # Set voltage limits
        if self.v_min is not None:
            self.dtaq.SetThreshVMin(True, self.v_min)
            self.dtaq.SetStopVMin(True, self.v_min)
        if self.v_max is not None:
            self.dtaq.SetThreshVMax(True, self.v_max)
            self.dtaq.SetStopVMax(True, self.v_max)

        # Turn cell on
        self.pstat.SetCell(GamryCOM.CellOn)

        # Find Vch range
        print('Finding Vch range...')
        self.pstat.FindVchRange()

    def set_signal(self, i, duration, t_sample):
        """
        Create signal object
        :param float i: constant current in ampls
        :param float duration: duration in seconds
        :param float t_sample: sample period in seconds
        :return:
        """
        self.signal = client.CreateObject('GamryCOM.GamrySignalConst')
        self.signal.Init(self.pstat,
                         i,  # current
                         duration,
                         t_sample,
                         GamryCOM.GstatMode,  # CtrlMode
                         )

        # Store signal parameters for reference
        self.signal_params = {
            'i': i,
            'duration': duration,
            't_sample': t_sample
        }

        self.i_req = i

    # ---------------------------------
    # Run
    # ---------------------------------
    def run(self, pstat, i, duration, t_sample, timeout=None, v_min=None, v_max=None,
            result_file=None, kst_file=None, append_to_file=False,
            show_plot=False, plot_interval=None):
        self.pstat = pstat
        self.set_signal(i, duration, t_sample)

        self.expected_duration = duration

        self.v_min = v_min
        self.v_max = v_max

        if timeout is None:
            timeout = duration + 30

        if plot_interval is None:
            plot_interval = t_sample * 0.9

        super().run_main(pstat, result_file, kst_file, append_to_file, timeout=timeout, show_plot=show_plot,
                         plot_interval=plot_interval)

    # --------------------
    # Header
    # --------------------
    def get_dtaq_header(self):
        text = 'AREA\tQUANT\t1.0\tSample Area (cm^2)\n' + \
               f'EOC\tQUANT\t{rel_round(self.v_oc, self.write_precision)}\tOpen Circuit (V)\n'
        return text