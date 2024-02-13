import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
import time
import warnings

from .config import GamryCOM, dtaq_header_info

from ..animation import LiveAxes, LiveFigure, LiveNyquist
from .eventsink import GamryDtaqEventSink
from ..plotting import get_nyquist_limits
from ..utils import check_control_mode, gamry_error_decoder


def estimate_eis_duration(frequencies):
    cycle_time = 1 / frequencies
    max_cycles = get_max_eis_cycles(frequencies)
    freq_times = np.maximum(cycle_time * max_cycles, 1)
    return np.sum(freq_times)



class DtaqReadZ(GamryDtaqEventSink):
    """
    Class for EIS measurements. Some methods and attributes of GamryDtaqEventSink are used,
    but much of the structure must be rewritten to handle the fundamentally different
    EIS measurement procedure.
    """

    def __init__(self, mode='galv', readzspeed='ReadZSpeedNorm',
                 ac_ierange: bool = None,
                 **init_kw):
        # Set enums based on specified mode
        check_control_mode(mode)

        # Initialize user_ac_ierange as None prior to mode setting
        self.user_ac_ierange = None
        self.mode = mode

        # Then set ac_ierange
        self.ac_ierange = ac_ierange

        # Set ReadZ speed
        self.gc_readzspeed = readzspeed

        # Subclass-specific attributes
        self.frequencies = None
        self.frequency_index = None
        self.dc_amp_req = None
        self.ac_amp_req = None
        self.z_guess = None
        self.max_passes = None

        self.gc_readzstatus = None
        self.passes = 0

        self.z_data = None
        self.plot_type = None

        super().__init__('GamryReadZ', self.tag, None, test_id=self.test_id, **init_kw)

    def set_mode(self, mode: str):
        check_control_mode(mode)
        self._mode = mode

        # Set dependent attributes
        if mode == 'galv':
            self.gc_ctrlmode = 'GstatMode'
            self.input_column = 'I'
            self.response_column = 'V'
            self.tag = 'GALVEIS'
            self.test_id = 'Galvanostatic EIS'
        else:
            self.gc_ctrlmode = 'PstatMode'
            self.input_column = 'V'
            self.response_column = 'I'
            self.tag = 'EISPOT'
            self.test_id = 'Potentiostatic EIS'

        # Check ac_ierange
        rec_ac_ierange = (mode != 'galv')

        if self.user_ac_ierange is not None:
            # If user specified value, don't overwrite it.
            # Instead, check and raise warning if not recommended
            if self.user_ac_ierange != rec_ac_ierange:
                warnings.warn(f'For {mode} mode, the recommended value '
                              f'of ac_ierange is {rec_ac_ierange}.')
        else:
            # No user-specified value. Set to recommended value
            self.ac_ierange = rec_ac_ierange


    def get_mode(self) -> str:
        return self._mode

    mode = property(get_mode, set_mode)

    def set_ac_ierange(self, ac_ierange: bool | None):
        # Store user-provided value
        self.user_ac_ierange = ac_ierange

        # Get recommended value based on mode
        if self.mode == 'galv':
            recommended = False
        else:
            # In theory should be True, but works better with False
            recommended = False

        if ac_ierange is not None:
            # Check the provided value and set
            if ac_ierange != recommended:
                warnings.warn(f'For {self.mode} mode, the recommended value '
                              f'of ac_ierange is {recommended}.')
            self._ac_ierange = ac_ierange
        else:
            # No value provided - use the recommended value
            self._ac_ierange = recommended

    def get_ac_ierange(self) -> bool:
        return self._ac_ierange

    ac_ierange = property(get_ac_ierange, set_ac_ierange)


    # COM event handling
    # --------------------------
    def cook(self, num_points=1024):
        # Override base class cook so that total_points is not incremented
        count = 1
        tot_count = 0
        while count > 0:
            count, points = self.dtaq.Cook(num_points)
            # The columns exposed by GamryDtaq.Cook vary by dtaq and are
            # documented in the Toolkit Reference Manual.
            self.acquired_points.extend(zip(*points))

            tot_count += count

        return tot_count

    def set_ie_range(self, frequency: float, z_guess: float, s_dc_max: float = 1.0):
        # Get IE range
        if self.ac_ierange:
            # "Correct" way: need to account for frequency to select IERange.
            # Seems to yield worse results than using DC IERange for galv measurements.
            # Args to TestIERangeAC are: i_ac_max, v_ac_max, i_dc_max, v_dc_max, freq

            if self.mode == 'galv':
                v_ac_max = self.ac_amp_req *  z_guess * 2
                IERange = self.pstat.TestIERangeAC(self.ac_amp_req, v_ac_max,
                                                self.dc_amp_req, s_dc_max,
                                                frequency)
            else:
                i_ac_max = 2 * self.ac_amp_req / z_guess
                IERange = self.pstat.TestIERangeAC(i_ac_max, self.ac_amp_req,
                                                   s_dc_max, self.dc_amp_req,
                                                   frequency)
        else:
            # "Incorrect" way: just use DC IERange.
            # Seems to work better for galv mode.
            if self.mode == 'galv':
                # Get max current amplitude with 5% buffer
                i_max = 1.05 * (abs(self.dc_amp_req) + (2 ** 0.5) * abs(self.ac_amp_req))
            else:
                # Estimate (very roughly) max current
                i_max = 2 * (abs(self.dc_amp_req) + (2 ** 0.5) * self.ac_amp_req) / self.z_guess

            IERange = self.pstat.TestIERange(i_max)
            
        print('IERange:', IERange)

        # Set IERange
        self.pstat.SetIERange(IERange)

        if self.mode == 'galv':
            # In galv mode, we need to set the internal voltage to the correct value
            # to produce the requested DC current
            # Get internal resistance
            Rm = self.pstat.IEResistor(IERange)

            # Get internal voltage amplitude to produce requested current
            v_dc_internal = Rm * self.dc_amp_req

            # Set IERange and internal voltage
            self.pstat.SetVoltage(v_dc_internal)


    def _IGamryReadZEvents_OnDataAvailable(self, this):
        new_count = self.cook(1024)
        # print('Data Available')

    def _IGamryReadZEvents_OnDataDone(self, this, gc_readzstatus):
        # print(f'DataDone for frequency_index {self.frequency_index}')
        data_timedelta = time.time() - self.start_time

        # Store status
        self.gc_readzstatus = gc_readzstatus

        # Increment passes
        self.passes += 1

        frequency = self.frequencies[self.frequency_index]
        # print('At frequency {:.2e} status is {}'.format(frequency, self.dtaq.StatusMessage()))

        # Check for statuses that should trigger a retry
        # Qualifying statuses depend on readz speed
        # Manually set status to retry
        retry_reasons = {
            'ReadZSpeedFast': [],
            'ReadZSpeedNorm': [],
            'ReadZSpeedLow': ['Invalid Eis Result thought to be good.', 'Cycle Limit.']
        }
        if self.gc_readzstatus == GamryCOM.ReadZStatusOk and \
                self.dtaq.StatusMessage() in retry_reasons[self.gc_readzspeed]:
            self.gc_readzstatus = GamryCOM.ReadZStatusRetry

        # Check measurement status
        if self.gc_readzstatus == GamryCOM.ReadZStatusRetry and self.passes < self.max_passes:
            # Pstat settings need adjustment to obtain good data.
            # Print message, clear acquired_points, and retry (ReadZ will handle settings adjustments)
            print('Retry at frequency {:.2e} Hz for reason: {}'.format(frequency, self.dtaq.StatusMessage()))
            if self.mode == 'galv':
                # If operating in galvanostatic mode, ReadZ will not adjust the IERange.
                # This makes sense since the current is controlled and we should be able to set the correct range before
                # starting the measurement.
                # However, pstat seems to occasionally report current out of range during EISGALV
                # (is this because of gain?)
                # Therefore: manually adjust IERange in galv mode if current is too large or too small.
                print('Idc:', self.dtaq.Idc())
                print('Gain:', self.dtaq.Gain())
                # if self.dtaq.StatusMessage() in ('Iac too big.', 'Iac too small.'):
                #     # Update IE range
                #     old_ie_range = self.pstat.IERange()
                #     if self.dtaq.StatusMessage() == 'Iac too big.':
                #         new_ie_range = old_ie_range + 1
                #         # print('Incremented IERange to {}'.format(new_ie_range))
                #     else:
                #         new_ie_range = old_ie_range - 1
                #         # print('Decremented IERange to {}'.format(new_ie_range))
                #
                #     # Attempt to set new IE range
                #     self.pstat.SetIERange(new_ie_range)
                #
                #     # Check which IE range was actually set (may hit top or bottom limit)
                #     new_ie_range = self.pstat.IERange()
                #     if new_ie_range != old_ie_range:
                #         print('Adjusted IERange to {}'.format(new_ie_range))
                #
                #     # Update the DC voltage based on the new IE range
                #     ie_resistor = self.pstat.IEResistor(new_ie_range)
                #     v_dc = ie_resistor * self.dc_amp_req  # Voltage signal which generates I signal
                #     # Set new IE range
                #     self.pstat.SetIERange(new_ie_range)
                #     # Set voltage to generate requested DC current
                #     self.pstat.SetVoltage(v_dc)
                #
                #     if new_ie_range != old_ie_range:
                #         print('New Rm, Sdc: {:.3f}, {:.3f}'.format(ie_resistor, v_dc))
                    # self.dtaq.SetGain(1.0)

                # elif self.dtaq.StatusMessage() == 'Iac too small.':
                #     # Update IE range
                #     new_ie_range = self.pstat.IERange() - 1
                #     ie_resistor = self.pstat.IEResistor(new_ie_range)
                #     v_dc = ie_resistor * self.dc_amp_req  # Voltage signal which generates I signal
                #     # Set new IE range
                #     self.pstat.SetIERange(new_ie_range)
                #     # Set voltage to generate requested DC current
                #     self.pstat.SetVoltage(v_dc)
                #
                #     print('Decremented IERange to {}'.format(new_ie_range))

            self.acquired_points = []
            self.dtaq.Measure(frequency, self.ac_amp_req)  # self.measure_point would reset passes
        else:
            # if self.gc_readzstatus == GamryCOM.ReadZStatusOk \
            #     or self.passes == self.max_passes \
            #     or self.gc_readzstatus == GamryCOM.ReadZStatusError:
            # Done with current frequency (either due to successful measurement or failure to measure)
            # Store data and move to next point
            if self.gc_readzstatus == GamryCOM.ReadZStatusOk:
                # Good measurement
                # print('Frequency: {:.1e}'.format(frequency))
                # print('Idc:', self.dtaq.Idc())
                # print('Gain:', self.dtaq.Gain())
                # Store data
                self.z_data[self.frequency_index] = [data_timedelta] + self.get_current_zdata()
            else:
                # Measurement failed
                if self.passes == self.max_passes:
                    print('Measurement at {:.2e} Hz unsuccessful after {} passes'.format(
                        self.frequencies[self.frequency_index], self.passes)
                    )
                    # Store data anyway
                    self.z_data[self.frequency_index] = [data_timedelta] + self.get_current_zdata()
                elif self.gc_readzstatus == GamryCOM.ReadZStatusError:
                    print('Measurement at {:.2e} Hz failed with reason: {}'.format(
                        self.frequencies[self.frequency_index], self.dtaq.StatusMessage())
                    )
                    # Store nans in z_data row
                    self.z_data[self.frequency_index] = [data_timedelta, self.frequencies[self.frequency_index]] + \
                                                        [np.nan] * (self.z_data.shape[1] - 2)

            # Increment total points
            self.total_points += 1

            # Write to file
            self.write_to_files(1, False)

            # Increment frequency index
            self.frequency_index += 1

            # Move to next step
            if self.frequency_index == len(self.frequencies):
                # All requested frequencies measured
                # Set measurement status to complete
                self.measurement_complete = True

                # Final write - moved to run_main to ensure that file is written even if
                # measurement times out or measurement error occurs
                # self.write_to_files(True)

                # Close handle to terminate PumpEvents
                self.close_connection()
            else:
                if self.ac_ierange and self.mode == 'galv':
                    # If running in galv mode and we want to use AC IERange,
                    # need to update IERange based on frequency.
                    # In pot mode, IERange is updated automatically by the pstat.
                    # Get estimated z modulus
                    z_guess = self.z_guess
                    if self.frequency_index > 0:
                        # Use last measured point if successful
                        if not np.isnan(self.z_data[self.frequency_index - 1,
                                                    self.zdata_columns.index('Zmod')]):
                            z_guess = self.z_data[self.frequency_index - 1,
                                                self.zdata_columns.index('Zmod')]

                    # Update IE range    
                    self.set_ie_range(self.frequencies[self.frequency_index], z_guess)


                # Measure next point
                self.measure_point(self.frequencies[self.frequency_index])
        # elif self.passes == self.max_passes or self.gc_readzstatus == GamryCOM.ReadZStatusError:
        #     # Hit max number of passes OR error without successful result
        #     # Record nans and skip to next point
        #     if self.passes == self.max_passes:
        #         print('Measurement at {:.2e} Hz unsuccessful after {} passes'.format(
        #             self.frequencies[self.frequency_index], self.passes
        #         )
        #         )
        #     else:
        #         print('Measurement at {:.2e} Hz failed with reason: {}'.format(self.frequencies[self.frequency_index],
        #                                                                        self.dtaq.StatusMessage())
        #               )
        #
        #     # Increment total points
        #     self.total_points += 1
        #
        #     # Store nans in z_data row
        #     self.z_data[self.frequency_index] = [data_timedelta, self.frequencies[self.frequency_index]] + \
        #                                         [np.nan] * (self.z_data.shape[1] - 2)
        #
        #     # Write to file
        #     self.write_to_file(False)
        #
        #     # Increment frequency index
        #     self.frequency_index += 1
        #
        #     # Move to next step
        #     if self.frequency_index == len(self.frequencies):
        #         # Final write - moved to run_main to ensure that file is written even if
        #         # measurement times out or measurement error occurs
        #         # self.write_to_file(True)
        #
        #         # All requested frequencies measured. Close handle to terminate PumpEvents
        #         self.close_connection()
        #     else:
        #         # Measure next point
        #         self.measure_point(self.frequencies[self.frequency_index])

        # elif self.gc_readzstatus == GamryCOM.ReadZStatusError:
        #     # Error. Print message and close handle
        #     # Skip to next point
        #     print(self.dtaq.StatusMessage())
        #     self.close_handle()

    # Subclass-specific methods
    # ---------------------------
    def initialize_pstat(self):
        # Set pstat settings - taken from Framework EIS.exp files
        # self.pstat.SetCell(GamryCOM.CellOff)

        self.pstat.SetAchSelect(GamryCOM.GND)
        self.pstat.SetCtrlMode(getattr(GamryCOM, self.gc_ctrlmode))
        self.pstat.SetIEStability(GamryCOM.StabilityFast)
        self.pstat.SetSenseSpeedMode(True)
        self.pstat.SetIConvention(GamryCOM.Anodic)
        self.pstat.SetGround(GamryCOM.Float)

        # Dan Cook 11/30/22: Do not explicitly set ranging or offsets for EIS - let system determine
        # self.pstat.SetIchRange(3.0)
        # self.pstat.SetIchRangeMode(False)
        # self.pstat.SetIchFilter(2.5)  # 5?
        # self.pstat.SetVchRange(3.0)
        # self.pstat.SetVchRangeMode(False)

        # Offset enable seems to cause issues with high-frequency measurements
        # 1/30/23 - re-enabled IchOffset to fix errors encountered when measuring in galv mode
        #  with a small negative DC offset (e.g. -10 mA)
        # 10/16/23 - encountering issues with potentiostatic measurements at 
        #  small DC offset (50-100 mV). Disable IchOffset for pot mode
        # 11/3/23 - now encountering issues with galv measurements at 
        #  very small DC offset (~1e-6 A) for high-impedance thin films. 
        #  Disable IchOffset for galv mode as well
        if self.mode == 'galv':
            self.pstat.SetIchOffsetEnable(False)
        else:
            self.pstat.SetIchOffsetEnable(False)
        # if self.dc_amp_req < 0:
        
        # This seems to be necessary at least in the case that mode='pot', v_dc < -0.024
        self.pstat.SetVchOffsetEnable(True)

        # self.pstat.SetVchFilter(2.5)  # Causes "Invalid EIS result thought to be good" status at first frequency
        # self.pstat.SetAchRange(3.0)

        # self.pstat.SetIERange(0.03)
        self.pstat.SetIERangeMode(False)
        # self.pstat.SetIERangeLowerLimit(NIL)
        self.pstat.SetAnalogOut(0.0)

        self.pstat.SetPosFeedEnable(False)
        self.pstat.SetIruptMode(GamryCOM.IruptOff)

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat)
        self.dtaq.SetSpeed(getattr(GamryCOM, self.gc_readzspeed))
        self.dtaq.SetGain(1.0)
        self.dtaq.SetINoise(0.0)
        self.dtaq.SetVNoise(0.0)
        self.dtaq.SetIENoise(0.0)
        self.dtaq.SetZmod(self.z_guess)

        if self.mode == 'galv':
            self.pstat.SetCASpeed(3)  # Switched from 2 to 3 as recommended by Abe Krebs 1/30/23
            self.dtaq.SetIdc(self.dc_amp_req)
            print('dc_amp_req: {:.2e} A'.format(self.dc_amp_req))
            print('ac_amp_req: {:.2e} A'.format(self.ac_amp_req))
            # print('Idc:', self.dtaq.Idc())

            # Set IERange
            # i_max = 1.05 * (abs(self.dc_amp_req) + (2 ** 0.5) * abs(self.ac_amp_req))  # 5% buffer
            # print('Max current:', i_max)
            # # Old code: DC
            # IERange = self.pstat.TestIERange(i_max)  # Max current

            # # New code: AC
            # # v_ac_max = self.ac_amp_req *  self.z_guess * 50
            # # v_dc_max = 3
            # # IERange = self.pstat.TestIERangeAC(self.ac_amp_req, v_ac_max, self.dc_amp_req, v_dc_max,
            # #                                    self.frequencies[0])

            # Rm = self.pstat.IEResistor(IERange)
            # print('IERange, Rm:', IERange, Rm)
            # Sdc = Rm * self.dc_amp_req  # Voltage signal which generates I signal
            # print('Sdc (internal): {:.3e} V'.format(Sdc))

            # self.pstat.SetIERange(IERange)
            # # Set voltage to generate requested DC current
            # self.pstat.SetVoltage(Sdc)

            self.set_ie_range(self.frequencies[0], self.z_guess)

            # find the DC Voltage for VGS
            print("Finding Vch Range...")
            if self.start_with_cell_off:
                self.pstat.SetCell(GamryCOM.CellOn)  # turn the cell on
                time.sleep(3)  # Let sample equilibrate

            self.pstat.FindVchRange()
            # print('Sdc:', Sdc)
            # print('Measured v:', self.pstat.MeasureV())
            # self.pstat.SetVoltage(self.pstat.MeasureV())
        elif self.mode == 'pot':
            print('dc_amp_req: {:.2e} V'.format(self.dc_amp_req))
            print('ac_amp_req: {:.2e} V'.format(self.ac_amp_req))
            
            # In pot mode, we know the voltage a priori
            # -> Set the VchRange accordingly.
            # The pstat seems to do this on its own when vdc > 0, 
            # but does not seem to handle this for vdc < 0, 
            # which can cause Vdc to be truncated wherever the cached VchRange ends.
            # To avoid such issues, we set VchRange explicitly here.
            v_max = abs(self.dc_amp_req) + np.sqrt(2) * abs(self.ac_amp_req)
            self.pstat.SetVchRange(self.pstat.TestVchRange(v_max))

            self.pstat.SetCASpeed(3)  # MedFast
            self.pstat.SetVoltage(self.dc_amp_req)  # Set DC voltage

            # Estimate current range
            # IERange = self.pstat.TestIERange((abs(self.dc_amp_req) + (2 ** 0.5) * self.ac_amp_req) / self.z_guess)
            # print('IERange:', IERange)
            # self.pstat.SetIERange(IERange)
            self.set_ie_range(self.frequencies[0], self.z_guess)

            if self.start_with_cell_off:
                # Get DC current
                self.pstat.SetCell(GamryCOM.CellOn)
                time.sleep(1)
                self.dtaq.SetIdc(self.pstat.MeasureI())
                
        print('VchRange:', self.pstat.VchRange())

        # With everything ready, turn cell on
        self.pstat.SetCell(GamryCOM.CellOn)

    def set_cycle_lim(self, frequency):
        # From LVEIS example
        if frequency > 3e4:
            cycle_lim = (10, 20)
        elif frequency > 1e3:
            cycle_lim = (8, 12)
        elif frequency > 30:
            cycle_lim = (4, 8)
        elif frequency > 1:
            cycle_lim = (3, 6)
        else:
            cycle_lim = (2, 4)

        self.dtaq.SetCycleLim(*cycle_lim)

    def measure_point(self, frequency):
        # Set cycle limits based on frequency
        self.set_cycle_lim(frequency)

        # Clear acquired points
        self.acquired_points = []

        # Reset passes
        self.passes = 0

        self.dtaq.Measure(frequency, self.ac_amp_req)

    def run(self, pstat, frequencies, dc_amp, ac_amp, z_guess, max_passes=10, timeout=None,
            result_file=None, kst_file=None, append_to_file=False, condition_time=0,
            show_plot=False, plot_interval=1, plot_type='all'):
        # Store measurement parameters
        self.frequencies = frequencies
        self.dc_amp_req = dc_amp
        self.ac_amp_req = ac_amp
        self.z_guess = z_guess
        self.max_passes = max_passes
        self.plot_type = plot_type

        if self.mode == 'galv':
            self.i_req = dc_amp
            self.v_req = None
        elif self.mode == 'pot':
            self.i_req = None
            self.v_req = dc_amp

        self.pstat = pstat

        self.result_file = result_file
        self.kst_file = kst_file
        self.frequency_index = 0
        self._last_write_index = 0
        self._last_kst_write_index = 0
        self.total_points = 0

        # Reset measurement status
        self.measurement_complete = False

        if timeout is None:
            timeout = 30 + 3 * estimate_eis_duration(frequencies)
            print('EIS timeout limit: {:.2f} min'.format(timeout / 60))

        # Initialize data array
        self.z_data = np.zeros((len(frequencies), 10))

        # Create client connection and open pstat
        self.open_connection()

        try:
            # Set pstat settings and initialize dtaq
            self.initialize_pstat()

            if condition_time > 0:
                print('Conditioning...')
                time.sleep(condition_time)

            print('running')

            self.start_time = time.time()

            # Start measurement at first point. All subsequent points are triggered by DataDone event
            self.measure_point(frequencies[self.frequency_index])

            # Create result file
            self.get_file_offset_info(append_to_file, result_file)
            if self.result_file is not None:
                # Write header
                if not append_to_file:
                    with open(self.result_file, 'w+') as f:
                        f.write(self.generate_header_text())

                if self.write_mode == 'continuous':
                    self._active_file = open(self.result_file, 'a')
                else:
                    self._active_file = None

            # Create Kst file
            if self.kst_file is not None and not append_to_file:
                # Write header
                with open(self.kst_file, 'w+') as f:
                    f.write(self.generate_kst_header())

            # Pump events
            if show_plot:
                # For real-time plotting, pump events in intervals and place plotting pauses in between
                while self.frequency_index == 0:
                    print('frequency index:', self.frequency_index)
                    self.PumpEvents(plot_interval)
                    time.sleep(0.1)

                    # Check for timeout
                    if time.time() - self.start_time > timeout:
                        print('Timed out')
                        break

                # Initialize plots once at least one data point has been collected
                self.initialize_figure()
                self.ani = self.run_plot_animation(plot_interval)

                while self.connection_active:
                    self.PumpEvents(plot_interval)
                    plt.pause(plot_interval / 100)

                    # Check for timeout
                    if time.time() - self.start_time > timeout:
                        print('Timed out')
                        break
            else:
                while self.connection_active:
                    self.PumpEvents(1.0)
                    time.sleep(0.1)

                    # Check for timeout
                    if time.time() - self.start_time > timeout:
                        print('Timed out')
                        break
            print('PumpEvents done')

            # Final write
            self.write_to_files(0, True)

        except Exception as e:
            self.terminate()
            # Final write
            self.write_to_files(0, True)
            raise gamry_error_decoder(e)

        # Close connection if still active (may have timed out)
        if self.connection_active:
            self.close_connection()

        # Stop plot animation once data collection complete
        if show_plot:
            self.ani.pause()

        # Garbage collection
        gc.collect()

        print('Run time: {:.2f} s'.format(time.time() - self.start_time))

    # --------------------------
    # Plotting
    # --------------------------
    def initialize_figure(self):
        print('initializing figure')
        if self.plot_type == 'all':
            self.fig, self.axes = plt.subplots(1, 3, figsize=(10, 3.25))
            nyquist_ax = self.axes[0]
            bode_axes = self.axes[1:]
        elif self.plot_type == 'nyquist':
            self.fig, self.axes = plt.subplots(figsize=(4, 3.25))
            nyquist_ax = self.axes
            bode_axes = None
        elif self.plot_type == 'bode':
            self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 3.25))
            nyquist_ax = None
            bode_axes = self.axes
        else:
            raise ValueError(f'Invalid plot_type {self.plot_type}')

        self.axes = np.atleast_1d(self.axes)

        # Set scilimits
        for ax in self.axes:
            ax.ticklabel_format(scilimits=(-3, 3))

        live_axes = []
        if self.plot_type in ('all', 'nyquist'):
            # Title
            nyquist_ax.set_title('Nyquist')
            # Initialize axis limits. Aspect ratio set later once all subplots ready and tight_layout applied
            nyquist_ax.set_xlim(0, self.z_guess)
            nyquist_ax.set_ylim(0, self.z_guess * 0.1)

            # Axis labels
            nyquist_ax.set_xlabel('$Z^\prime$ ($\Omega$)')
            nyquist_ax.set_ylabel('$-Z^{\prime\prime}$ ($\Omega$)')

            # Set axis_extend_ratios to same value to maintain aspect ratio
            lax_n = LiveNyquist(nyquist_ax)  # , axis_extend_ratio={'x': 0.25, 'y': 0.25})

            # Data update function
            def update_nyquist(frame):
                x = self.z_data[:self.frequency_index, 2]
                y = -self.z_data[:self.frequency_index, 3]
                return x, y

            # Create artist
            lax_n.add_line_artist('z', update_nyquist, marker='.', ms=4, alpha=0.6, ls='')

            live_axes.append(lax_n)

        if self.plot_type in ('all', 'bode'):
            # Titles
            bode_axes[0].set_title('Modulus')
            bode_axes[1].set_title('Phase')

            # Initialize axis limits
            xlim = (np.min(self.frequencies) / 5, np.max(self.frequencies) * 5)  # passed to LiveAxes below
            for ax in bode_axes:
                ax.set_xscale('log')
            bode_axes[0].set_ylim(0, self.z_guess)  # modulus
            bode_axes[1].set_ylim(0, 10)  # phase

            # Data update functions
            def update_modulus(frame):
                x = self.z_data[:self.frequency_index, 1]
                y = self.z_data[:self.frequency_index, 5]
                return x, y

            def update_phase(frame):
                x = self.z_data[:self.frequency_index, 1]
                y = -self.z_data[:self.frequency_index, 6]
                return x, y

            # Create live axes
            lax_b1 = LiveAxes(bode_axes[0], fixed_xlim=xlim)
            lax_b2 = LiveAxes(bode_axes[1], fixed_xlim=xlim)

            # Create artists
            lax_b1.add_line_artist('modulus', update_modulus, marker='.', ms=4, alpha=0.6, ls='')
            lax_b2.add_line_artist('phase', update_phase, marker='.', ms=4, alpha=0.6, ls='')

            # Axis labels
            for ax in bode_axes:
                ax.set_xlabel('$f$ (Hz)')
            bode_axes[0].set_ylabel('$|Z|$ ($\Omega$)')
            bode_axes[1].set_ylabel(r'$-\theta$ ($^\circ$)')

            live_axes = live_axes + [lax_b1, lax_b2]

        # Add status text
        def update_status(frame):
            if self.frequency_index < len(self.frequencies):
                text = 'Measuring Z at {:.2e} Hz'.format(self.frequencies[self.frequency_index])
            else:
                text = 'Measurement complete'
            return text

        live_axes[0].add_text_artist('status', 0.025, 0.975, update_status, ha='left', va='top',
                                     transform=live_axes[0].ax.transAxes)

        self.fig.tight_layout()

        if self.plot_type in ('all', 'nyquist'):
            # Set Nyquist aspect ratio once all subplots have been defined and tight_layout has been applied
            axis_limits = get_nyquist_limits(nyquist_ax, np.array([self.z_guess * 0.1, self.z_guess * (0.9 - 0.5j)]))
            nyquist_ax.set_xlim(axis_limits['x'])
            nyquist_ax.set_ylim(axis_limits['y'])

        self.live_fig = LiveFigure(live_axes)

    # ----------------
    # Data
    # ----------------
    def get_current_zdata(self):
        """
        Get impedance data for current frequency
        :return:
        """
        readz_attributes = ['Zfreq', 'Zreal', 'Zimag', 'Zsig', 'Zmod', 'Zphz', 'Idc', 'Vdc', 'IERange']
        zdata = [getattr(self.dtaq, attr)() for attr in readz_attributes]

        return zdata

    @property
    def zdata_columns(self):
        return dtaq_header_info[self.dtaq_class]['columns'][1:]  # exclude index

    @property
    def z_dataframe(self):
        if self.z_data is not None:
            return pd.DataFrame(self.z_data, columns=self.zdata_columns)
        else:
            return None

    def get_dataframe_to_write(self, start_index, end_index):
        """
        DataFrame containing only the columns to be written to result file. Used by parent
        :return:
        """
        return pd.DataFrame(self.z_data[start_index:end_index],
                            index=pd.RangeIndex(start_index, end_index),
                            columns=self.zdata_columns
                            )

    @property
    def kst_column_index(self):
        return [self.zdata_columns.index(col) for col in self.kst_columns]

    def get_kst_dataframe(self, start_index, end_index):
        """
        DataFrame formatted for Kst
        :return:
        """
        try:
            kst_data = self.z_data[start_index:end_index, self.kst_column_index].astype(float)
        except TypeError:
            kst_data = self.z_data[start_index:end_index, self.kst_column_index]
        return pd.DataFrame(kst_data, index=pd.RangeIndex(start_index, end_index), columns=self.kst_columns)

    # -----------------------------
    # File writing
    # -----------------------------
    # def get_row_string(self, frequency_index):
    #     row_data = [frequency_index] + list(rel_round(self.z_data[frequency_index], self.write_precision))
    #     row_string = '\t' + '\t'.join([str(x) for x in row_data]) + '\n'
    #
    #     return row_string

    def write_to_file(self, data_func, destination_file, active_file, last_write_index, new_count,
                      is_final_write, indent):
        if destination_file is not None:
            if is_final_write:
                if self.write_mode == 'continuous':
                    # Last data point already written, just need to close file
                    if active_file is not None:
                        active_file.close()
                if self.write_mode == 'interval':
                    # Write all unwritten data points to file
                    num_available = self.total_points - last_write_index
                    if num_available > 0:
                        data_string = self.generate_data_string(data_func, last_write_index,
                                                                last_write_index + num_available, indent)
                        with open(destination_file, 'a') as f:
                            f.write(data_string)
                        last_write_index += num_available
                elif self.write_mode == 'once':
                    # Write all data to file
                    data_string = self.generate_data_string(data_func, 0, self.total_points, indent)
                    with open(destination_file, 'a') as f:
                        f.write(data_string)
                    last_write_index = self.total_points
            else:
                if self.write_mode == 'continuous':
                    # Write new data point
                    # row_string = self.get_row_string(self.frequency_index)
                    data_string = self.generate_data_string(data_func, self.frequency_index, self.frequency_index + 1,
                                                            indent)
                    if active_file is not None:
                        active_file.write(data_string)
                    else:
                        with open(destination_file, 'a') as f:
                            f.write(data_string)
                    last_write_index += 1
                elif self.write_mode == 'interval':
                    # If a sufficient number of new points are available, write them all to the file
                    num_available = self.total_points - last_write_index
                    if num_available >= self.write_interval:
                        data_string = self.generate_data_string(data_func, last_write_index,
                                                                last_write_index + num_available, indent)
                        with open(destination_file, 'a') as f:
                            f.write(data_string)
                        last_write_index += num_available
            return last_write_index


def get_max_eis_cycles(frequencies):
    cycle_max = [(3e4, 12), (1e3, 8), (30, 6), (1, 4)]
    cycles = np.zeros(len(frequencies), dtype=int) + 20
    for f, c in cycle_max:
        cycles[frequencies <= f] = c

    return cycles