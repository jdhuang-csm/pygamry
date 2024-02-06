from .config import GamryCOM
from .eventsink import GamryDtaqEventSink
from ..utils import gamry_error_decoder, rel_round, check_control_mode


import comtypes.client as client


class DtaqPstatic(GamryDtaqEventSink):
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
        self.i_min = None
        self.i_max = None

        super().__init__('GamryDtaqCpiv', 'POTENTIOSTATIC', axes_def,
                         test_id='Potentiostatic Scan', **init_kw)

    # ---------------------------------
    # Initialization and configuration
    # ---------------------------------
    def initialize_pstat(self):
        self.pstat.SetCtrlMode(GamryCOM.PstatMode)
        self.pstat.SetIEStability(GamryCOM.StabilityNorm)
        self.pstat.SetIERangeMode(True)

        # Measure OCV with cell off
        if self.start_with_cell_off:
            self.v_oc = self.pstat.MeasureV()
        else:
            self.v_oc = 0

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat)

        # Set signal
        self.pstat.SetSignal(self.signal)

        # Set current limits
        if self.i_min is not None:
            self.dtaq.SetThreshIMin(True, self.i_min)
            self.dtaq.SetStopIMin(True, self.i_min)
        if self.i_max is not None:
            self.dtaq.SetThreshIMax(True, self.i_max)
            self.dtaq.SetStopIMax(True, self.i_max)

        # Turn cell on
        self.pstat.SetVoltage(self.v_req)
        self.pstat.SetCell(GamryCOM.CellOn)

        # Dan Cook 11/30/22: Ich ranging can be disabled and range set to 3.0 for most DC experiments
        self.pstat.SetIchRangeMode(False)
        self.pstat.SetIchRange(3.0)
        self.pstat.SetIchOffsetEnable(False)

        # Find current range
        print('Finding IE range...')
        self.pstat.FindIERange()

        print('IE range:', self.pstat.IERange())

    def set_signal(self, v, duration, t_sample):
        """
        Create signal object
        :param float v: constant potential to hold in volts
        :param float duration: duration in seconds
        :param float t_sample: sample period in seconds
        :return:
        """
        self.signal = client.CreateObject('GamryCOM.GamrySignalConst')
        self.signal.Init(self.pstat,
                         v,  # voltage
                         duration,
                         t_sample,
                         GamryCOM.PstatMode,  # CtrlMode
                         )

        # Store signal parameters for reference
        self.signal_params = {
            'v': v,
            'duration': duration,
            't_sample': t_sample
        }

        self.v_req = v

    # ---------------------------------
    # Run
    # ---------------------------------
    def run(self, pstat, v, duration, t_sample, timeout=None, i_min=None, i_max=None,
            result_file=None, kst_file=None, append_to_file=False,
            show_plot=False, plot_interval=None):
        self.pstat = pstat
        self.set_signal(v, duration, t_sample)

        self.expected_duration = duration

        self.i_min = i_min
        self.i_max = i_max

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