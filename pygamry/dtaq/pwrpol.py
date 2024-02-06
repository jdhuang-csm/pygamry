from .config import GamryCOM, dtaq_header_info
from .eventsink import GamryDtaqEventSink
from ..utils import gamry_error_decoder, rel_round, check_control_mode


import comtypes.client as client
import numpy as np
import pandas as pd


class DtaqPwrPol(GamryDtaqEventSink):
    """
    Class for current-controlled polarization curve
    """

    def __init__(self, mode='CurrentDischarge', **init_kw):

        mode_signs = {'CurrentDischarge': -1, 'CurrentCharge': 1}
        if mode not in mode_signs.keys():
            raise ValueError(f'Invalid mode argument {mode}. Options: {mode_signs.keys()}')

        self.mode = mode

        # Define axes for real-time plotting
        axes_def = [
            {'title': 'Polarization Curve',
             'type': 'y(x)',
             'x_column': 'Im',
             'y_column': 'Vf',
             'x_label': '$i$ (A)',
             'y_label': '$V$ (V)'
             },
            {'title': 'Power',
             'type': 'y(x)',
             'x_column': 'Im',
             'y_column': 'Pwr',
             'x_label': '$i$ (A)',
             'y_label': '$P$ (W)'
             },
        ]

        # Initialize subclass attributes
        self.v_oc = None
        self.v_min = None
        self.v_max = None

        super().__init__('GamryDtaqPwr', 'PWR800_POLARIZATION', axes_def,
                         test_id='Constant Current Polarization Curve', **init_kw)

    # ---------------------------------
    # Initialization and configuration
    # ---------------------------------
    @property
    def mode_sign(self):
        mode_signs = {'CurrentDischarge': -1, 'CurrentCharge': 1}
        return mode_signs[self.mode]

    def initialize_pstat(self):
        print('IERange 0:', self.pstat.IERange())
        self.pstat.SetCtrlMode(GamryCOM.GstatMode)
        self.pstat.SetIEStability(GamryCOM.StabilityNorm)

        # Measure OCV with cell off
        if self.start_with_cell_off:
            self.v_oc = self.pstat.MeasureV()
        else:
            self.v_oc = 0

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat)

        # Set IE range based on final current and disable auto-range
        self.pstat.SetIERange(self.pstat.TestIERange(self.signal_params['i_final']))
        self.pstat.SetIERangeMode(False)

        # Per Dan Cook 11/30/22: Disable Ich ranging, set IchRange to 3, disable Ich and Vch offsets
        self.pstat.SetIchRangeMode(False)
        self.pstat.SetIchRange(3.0)
        self.pstat.SetIchOffsetEnable(False)
        self.pstat.SetVchOffsetEnable(False)

        print('IchRange:', self.pstat.IchRange())

        # Set signal
        self.pstat.SetSignal(self.signal)

        # Set voltage threshold
        if self.v_min is not None:
            self.dtaq.SetThreshVMin(True, self.v_min)
            self.dtaq.SetStopVMin(True, self.v_min)
        if self.v_max is not None:
            self.dtaq.SetThreshVMax(True, self.v_max)
            self.dtaq.SetStopVMax(True, self.v_max)

        # Turn cell on
        self.pstat.SetCell(GamryCOM.CellOn)

    def set_signal(self, i_final, scan_rate, t_sample):
        self.signal = client.CreateObject('GamryCOM.GamrySignalPwrRamp')
        # Default values obtained from pwr800.exp
        self.signal.Init(self.pstat,
                         0.0,  # ValueInit
                         float(i_final),  # ValueFinal
                         float(scan_rate),  # ScanRate
                         0,  # LimitValue - unused for CurrentCharge and CurrentDischarge
                         0.05,  # Gain: PWR800_DEFAULT_CV_CP_GAIN default = 0.05
                         0.15,  # MinDif: PWR800_DEFAULT_MINIMUM_DIFFERENCE default = 0.15
                         0.05,  # MaxStep: PWR800_DEFAULT_MAXIMUM_STEP = 0.05
                         float(t_sample),  # SamplePeriod
                         0.01,  # PerturbationRate: PWR800_DEFAULT_PERTURBATION_RATE default = 0.01
                         0.003333,  # PerturbationPulseWidth: PWR800_DEFAULT_PERTURBATION_WIDTH default = 0.003333
                         0.0016666666,  # TimerRes: PWR800_DEFAULT_TIMER_RESOLUTION default = 0.0016666666
                         getattr(GamryCOM, self.mode),  # PWRSIGNAL Mode
                         True,  # WorkingPositive
                         )

        # Store signal parameters for reference
        self.signal_params = {
            'i_final': i_final,
            'scan_rate': scan_rate,
            't_sample': t_sample
        }

        self.expected_duration = i_final / scan_rate

    # ---------------------------------
    # Run
    # ---------------------------------
    def run(self, pstat, i_final, scan_rate, t_sample, timeout=None, v_min=None, v_max=None,
            result_file=None, kst_file=None, append_to_file=False,
            show_plot=False, plot_interval=None):
        self.pstat = pstat
        self.set_signal(i_final, scan_rate, t_sample)

        self.v_min = v_min
        self.v_max = v_max

        if timeout is None:
            timeout = self.expected_duration * 1.5

        if plot_interval is None:
            plot_interval = t_sample * 0.9

        super().run_main(pstat, result_file, kst_file, append_to_file, timeout=timeout, show_plot=show_plot,
                         plot_interval=plot_interval)

    # --------------------
    # Header
    # --------------------
    def get_dtaq_header(self):
        text = 'CAPACITY\tQUANT\t1.0\tCapacity (A-hr)\n' + \
               'CELLTYPE\tSELECTOR\t0\tCell Type\n' + \
               'WORKINGCONNECTION\tSELECTOR\t0\tWorking Connection\n' + \
               f'DISCHARGEMODE\tDROPDOWN\t{getattr(GamryCOM, self.mode)}\tDischarge Mode\tConstant Current\n' + \
               'SCANRATE\tQUANT\t{:.5e}\tScan Rate (A/s)\n'.format(self.signal_params['scan_rate']) + \
               'SAMPLETIME\tQUANT\t{:.5e}\tSample Period (s)\n'.format(self.signal_params['t_sample']) + \
               f'EOC\tQUANT\t{rel_round(self.v_oc, self.write_precision)}\tOpen Circuit(V)\n'
        return text

    def get_dataframe_to_write(self, start_index, end_index):
        """
        DataFrame containing only the columns to be written to result file
        :return:
        """
        cook_cols = dtaq_header_info[self.dtaq_class]['cook_columns'][1:]  # Skip INDEX
        try:
            write_data = self.data_array[start_index:end_index, self.column_index_to_write].astype(float)
        except TypeError:
            write_data = self.data_array[start_index:end_index, self.column_index_to_write]

        # Add a column for expected current in case of range errors that result in current cutoff
        cook_cols.append('Im_expected')
        # Get expected current
        i_step = self.signal_params['t_sample'] * self.signal_params['scan_rate'] * self.mode_sign
        i_exp = np.arange(start_index, end_index) * i_step
        write_data = np.insert(write_data, write_data.shape[1], i_exp, axis=1)

        return pd.DataFrame(write_data, index=pd.RangeIndex(start_index, end_index), columns=np.array(cook_cols))