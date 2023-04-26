import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import comtypes.client as client
# from comtypes.client._events import _handles_type, ctypes
import gc

from .file_utils import get_file_time, read_last_line
from .utils import gamry_error_decoder, check_write_mode, check_control_mode, rel_round, time_format_code
from .animation import LiveAxes, LiveFigure
from .plotting import get_nyquist_limits

GamryCOM = client.GetModule(['{BD962F0D-A990-4823-9CF5-284D1CDD9C6D}', 1, 0])


def get_pstat(family='Interface'):
    """
    Get potentiostat
    :param str family: potentiostat family. Options: 'Interface', 'Reference'
    :return:
    """
    devices = client.CreateObject('GamryCOM.GamryDeviceList')
    print(devices.EnumSections())

    if family == 'Interface':
        obj_string = 'GamryCOM.GamryPC6Pstat'
    elif family == 'Reference':
        obj_string = 'GamryCOM.GamryPC5Pstat'

    pstat = client.CreateObject(obj_string)
    pstat.Init(devices.EnumSections()[0])

    return pstat


class GamryDtaqEventSink(object):
    def __init__(self, dtaq_class, tag, leave_cell_on=False,
                 write_precision=6, write_mode='continuous', write_interval=100):

        self.dtaq_class = dtaq_class
        self.tag = tag

        # Create dtaq source
        self.dtaq = client.CreateObject(f'GamryCOM.{dtaq_class}')

        self.pstat = None
        self.connection = None
        self.connection_active = False
        self.leave_cell_on = leave_cell_on

        self.signal = None
        self.signal_params = None

        # Create data containers
        self.acquired_points = []
        self.total_points = None
        # self.counts = []
        # self.count_times = []

        self.start_time = None
        self.show_plot = None
        self.ani = None

        # Event handling
        self.handler_routine = None
        self.hevt = None  # Handle for PumpEvents

        # Result file and write options
        self.result_file = None
        self.append_to_file = None
        self.write_mode = write_mode
        self._active_file = None
        self.write_precision = write_precision
        self.write_interval = write_interval
        self._last_write_index = None

    # COM methods
    # --------------------------
    def cook(self, num_points=1024):
        count = 1
        tot_count = 0
        while count > 0:
            count, points = self.dtaq.Cook(num_points)
            # The columns exposed by GamryDtaq.Cook vary by dtaq and are
            # documented in the Toolkit Reference Manual.
            self.acquired_points.extend(zip(*points))

            # self.counts.append(count)
            # if count > 0:
            #     self.count_times.append(points[0][-1])
            # else:
            #     self.count_times.append(self.count_times[-1])

            tot_count += count

        self.total_points += tot_count

        return tot_count

    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        print('DataAvailable fired')
        new_count = self.cook()

        if new_count > 0 and self.result_file is not None:
            if self.write_mode == 'continuous':
                # Write all newly received data points
                data_string = self.generate_data_string(self._last_write_index, self._last_write_index + new_count)
                self._active_file.write(data_string)
                self._last_write_index += new_count
            elif self.write_mode == 'interval':
                # If a sufficient number of new points are available, write them all to the file
                num_available = self.total_points - self._last_write_index
                if num_available >= self.write_interval:
                    data_string = self.generate_data_string(self._last_write_index,
                                                            self._last_write_index + num_available)
                    with open(self.result_file, 'a') as f:
                        f.write(data_string)
                    self._last_write_index += num_available

        # If running plot animation, pause to plot new point(s)
        if self.show_plot:
            # Must incorporate plt.pause to run GUI event loop and allow artists to be drawn
            plt.pause(1e-5)
        print('OnDataAvailable finished')

    def _IGamryDtaqEvents_OnDataDone(self, this):
        new_count = self.cook()  # a final cook

        print('DataDone')

        # Final write
        if self.result_file is not None:
            if self.write_mode == 'continuous':
                # Write all newly received data points
                if new_count > 0:
                    data_string = self.generate_data_string(self._last_write_index, self._last_write_index + new_count)
                    self._active_file.write(data_string)
                # Close file
                self._active_file.close()
                self._active_file = None
                self._last_write_index += new_count
            elif self.write_mode == 'interval':
                # Write all unwritten data points to file
                num_available = self.total_points - self._last_write_index
                if num_available > 0:
                    data_string = self.generate_data_string(self._last_write_index,
                                                            self._last_write_index + num_available)
                    with open(self.result_file, 'a') as f:
                        f.write(data_string)
                    self._last_write_index += num_available
            elif self.write_mode == 'once':
                # Write all data to file
                with open(self.result_file, 'a') as f:
                    data_strings = [self.get_row_string(i) for i in range(self.total_points)]
                    data_string = ''.join(data_strings)
                    f.write(data_string)

        # Close handle to terminate PumpEvents
        # self.close_handle()
        self.close_connection()

    # Event control
    # --------------------------------
    # def PumpEvents(self, timeout):
    #     """
    #     Code based on comtypes.client._events.PumpEvents
    #     Modified to enable early termination when desired conditions are met (e.g. data collection finished)
    #     """
    #
    #     # Store handle as instance attribute to enable other methods to close handle prior to timeout
    #     self.hevt = ctypes.windll.kernel32.CreateEventA(None, True, False, None)
    #     handles = _handles_type(self.hevt)
    #     RPC_S_CALLPENDING = -2147417835
    #
    #     def handler_routine(self, dwCtrlType):
    #         if dwCtrlType == 0:  # CTRL+C
    #             ctypes.windll.kernel32.SetEvent(self.hevt)
    #             return 1
    #         return 0
    #
    #     self.handler_routine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(handler_routine)
    #     ctypes.windll.kernel32.SetConsoleCtrlHandler(self.handler_routine, 1)
    #
    #     try:
    #         try:
    #             res = ctypes.oledll.ole32.CoWaitForMultipleHandles(0,
    #                                                                int(timeout * 1000),
    #                                                                len(handles), handles,
    #                                                                ctypes.byref(ctypes.c_ulong()))
    #             print(res)
    #         except WindowsError as details:
    #             if details.winerror != RPC_S_CALLPENDING:  # timeout expired
    #                 raise
    #         else:
    #             raise KeyboardInterrupt
    #     finally:
    #         self.close_handle()
    #
    # def close_handle(self):
    #     """
    #     Method to close handle and terminate PumpEvents
    #     :return:
    #     """
    #     ctypes.windll.kernel32.CloseHandle(self.hevt)
    #     ctypes.windll.kernel32.SetConsoleCtrlHandler(self.handler_routine, 0)

    def run(self, pstat, result_file=None, append_to_file=False, pump_interval=5, timeout=None,
            show_plot=False, **plot_kw):

        self.pstat = pstat
        self.show_plot = show_plot

        # Clear acquired points
        self.acquired_points = []
        self.total_points = 0

        # Reset last write index
        self._last_write_index = 0

        # File params
        self.result_file = result_file
        self.append_to_file = append_to_file

        # Open connection and pstat
        self.open_connection()

        try:
            self.initialize_pstat()
            # Run
            print('running')

            if show_plot:
                self.ani = self.run_plot_animation(**plot_kw)

            # Record start time
            self.start_time = time.time()
            self.get_file_offset_info(append_to_file, result_file)

            # Create result file
            if result_file is not None:
                # Write header
                if not append_to_file:
                    with open(result_file, 'w+') as f:
                        f.write(self.generate_header_text())

                if self.write_mode == 'continuous':
                    # Keep file open for continuous writing
                    self._active_file = open(result_file, 'a')
                else:
                    self._active_file = None

            self.dtaq.Run()

        except Exception as e:
            self.terminate()
            raise gamry_error_decoder(e)

        while self.connection_active:
            client.PumpEvents(pump_interval)
            time.sleep(pump_interval / 100)

            # Stop if timeout exceeded
            if timeout is not None:
                if time.time() - self.start_time > timeout:
                    print('Measurement timed out')
                    break

        # Close connection if still active (may have been terminated prematurely)
        if self.connection_active:
            self.close_connection()

        # Stop plot animation once data collection complete
        if self.show_plot:
            self.ani.pause()

        print('Run time: {:.2f} s'.format(time.time() - self.start_time))

    def open_connection(self):
        """
        Create connection and open pstat
        :return:
        """
        self.connection = client.GetEvents(self.dtaq, self)
        self.pstat.Open()
        self.connection_active = True

    def close_connection(self):
        """
        Delete connection and close pstat
        :return:
        """
        # Turn cell off unless directed to leave on
        if not self.leave_cell_on:
            self.pstat.SetCell(GamryCOM.CellOff)

        self.pstat.Close()
        self.connection_active = False

        # Close handle - terminates PumpEvents
        # self.close_handle()

        # Delete connection
        del self.connection
        self.connection = None

        # Garbage collection
        gc.collect()

    def terminate(self):
        """Terminate prematurely"""
        # Stop measurement
        self.dtaq.Stop()
        self.pstat.SetCell(GamryCOM.CellOff)

        self.close_connection()

    # ------------------------------------
    # Subclass method placeholders
    # ------------------------------------
    def run_plot_animation(self, **kw):
        # This method should be written at the subclass level depending on the measurement type
        return None

    def initialize_pstat(self):
        pass

    # ------------------------------------
    # Data management
    # ------------------------------------
    @property
    def cook_columns(self):
        return dtaq_cook_columns.get(self.dtaq_class, None)

    @property
    def data_array(self):
        # Place acquired points in array
        return np.array(self.acquired_points)

    @property
    def dataframe(self):
        # Get data columns based on dtaq_class
        data_columns = self.cook_columns
        if data_columns is None:
            data_columns = list(np.arange(self.data_array.shape[1], dtype=int).astype(str))

        # Exclude ignored columns
        keep_index = np.where(np.array(data_columns) != 'Ignore')

        # Construct DataFrame
        return pd.DataFrame(self.data_array[:, keep_index[0]], columns=np.array(data_columns)[keep_index])

    # ---------------------------------
    # File writing
    # ---------------------------------
    def format_start_date_time(self):
        gmtime = time.gmtime(self.start_time)
        # Get date string
        date_string = time.strftime(time_format_code.split(' ')[0], gmtime)
        # Get time string
        time_dec = self.start_time % 1  # decimal seconds
        time_string = time.strftime('{}.{}'.format(
            time_format_code.split(' ')[1],
            str(round(time_dec, self.write_precision))[2:2 + self.write_precision]
        ),
            gmtime
        )

        return date_string, time_string

    def get_date_time_text(self):
        """Get formatted start date/time text to write to file header"""
        date_string, time_string = self.format_start_date_time()
        text = f'DATE\tLABEL\t{date_string}\tDate\n'
        text += f'TIME\tLABEL\t{time_string}\tTime\n'

        return text

    def get_row_string(self, row_index):
        # Apply offsets
        # Start from last index previously written to file
        write_index = row_index + self.start_index_from
        # Offset times by time delta
        write_array = self.data_array.copy()
        write_array[:, 0] += self.file_time_offset

        col_index = self.column_index_to_write()
        row_data = [write_index] + list(rel_round(write_array[row_index, col_index], self.write_precision))
        row_string = '\t' + '\t'.join([str(x) for x in row_data]) + '\n'

        return row_string

    def generate_data_string(self, start_index, end_index):
        data_strings = [self.get_row_string(i) for i in range(start_index, end_index)]
        data_string = ''.join(data_strings)
        return data_string

    def get_data_header(self):
        header_info = dtaq_header_info[self.dtaq_class]
        name_row = '\t' + '\t'.join(header_info['columns']) + '\n'
        unit_row = '\t' + '\t'.join(header_info['units']) + '\n'
        header_text = header_info['preceding'] + '\n' + name_row + unit_row

        return header_text

    def column_index_to_write(self):
        """Get indices of cook columns to write to result file"""
        cook_cols = dtaq_header_info[self.dtaq_class]['cook_columns'][1:]  # Skip INDEX
        index = [self.cook_columns.index(col) for col in cook_cols]

        return np.array(index)

    def generate_header_text(self):
        text = 'EXPLAIN\n' + f'TAG\t{self.tag}\n' + 'TITLE\tLABEL\tTitle\tTest Identifier\n' + \
               self.get_date_time_text() + self.get_data_header()
        return text

    def get_file_offset_info(self, append_to_file, result_file):
        if append_to_file:
            # Get start time from file timestamp
            self.file_start_time = get_file_time(result_file)
            # Get last index written to file
            last_line = read_last_line(result_file)
            self.start_index_from = int(last_line.split('\t')[1]) + 1
        else:
            self.file_start_time = self.start_time
            self.start_index_from = 0
        self.file_time_offset = self.file_start_time - self.start_time

    # ----------------------
    # Getters and setters
    # ----------------------
    def get_write_mode(self):
        return self._write_mode

    def set_write_mode(self, write_mode):
        check_write_mode(write_mode)
        self._write_mode = write_mode

    write_mode = property(get_write_mode, set_write_mode)


# ============================================
# Subclasses
# ============================================

# =========================================================
# OCV
# =========================================================
class DtaqOcv(GamryDtaqEventSink):
    def __init__(self, **init_kw):
        super().__init__('GamryDtaqOcv', 'CORPOT', **init_kw)

    # ---------------------------------
    # Initialization and configuration
    # ---------------------------------
    def initialize_pstat(self):
        self.pstat.SetCtrlMode(GamryCOM.PstatMode)
        self.pstat.SetCell(GamryCOM.CellOff)
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

    # ---------------------------------
    # Run
    # ---------------------------------
    def run(self, pstat, duration, t_sample, result_file=None, append_to_file=False, show_plot=False,
            plot_interval=None):
        self.pstat = pstat
        self.set_signal(duration, t_sample)

        if plot_interval is None:
            plot_interval = t_sample

        super().run(pstat, result_file, append_to_file, timeout=duration + 30, show_plot=show_plot,
                    plot_interval=plot_interval)

    # ------------------------------------
    # Plotting
    # ------------------------------------
    def run_plot_animation(self, plot_interval):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_xlabel('$t$ (s)')
        ax.set_ylabel('OCV (V)')

        # initialize y limits
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, min(3600, self.signal_params['duration']))

        lax = LiveAxes(ax)  # fixed_xlim=(0, self.signal_params['duration']))

        def update_ocv(frame):
            x = self.data_array[:, 0]
            y = self.data_array[:, 1]
            return x, y

        lax.add_line_artist('ocv', update_ocv, marker='.', ms=6, alpha=0.5, ls='')

        fig.tight_layout()

        return lax.run(frames=int(self.signal_params['duration'] / plot_interval) + 100, interval=plot_interval)


# =========================================================
# ReadZ
# =========================================================
class DtaqReadZ(GamryDtaqEventSink):
    def __init__(self, mode='galvanostatic', readzspeed='ReadZSpeedNorm', **init_kw):
        # Set enums based on specified mode
        check_control_mode(mode)
        self.mode = mode
        if mode == 'galvanostatic':
            self.gc_ctrlmode = 'GstatMode'
            self.input_column = 'I'
            self.response_column = 'V'
            tag = 'EISGALV'
        elif mode == 'potentiostatic':
            self.gc_ctrlmode = 'PstatMode'
            self.input_column = 'V'
            self.response_column = 'I'
            tag = 'EISPOT'

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

        super().__init__('GamryReadZ', tag, **init_kw)

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

    def _IGamryReadZEvents_OnDataAvailable(self, this):
        new_count = self.cook(1024)
        # print('Data Available')

    def _IGamryReadZEvents_OnDataDone(self, this, gc_readzstatus):
        data_timedelta = time.time() - self.start_time

        # Store status
        self.gc_readzstatus = gc_readzstatus

        # Increment passes
        self.passes += 1

        frequency = self.frequencies[self.frequency_index]
        print('At frequency {:.2e} status is {}'.format(frequency, self.dtaq.StatusMessage()))

        # Check for status 'Invalid Eis Result thought to be good.'
        # This indicates a bad measurement, but is associated with OK status
        # Manually set status to retry
        if self.gc_readzstatus == GamryCOM.ReadZStatusOk and \
                self.dtaq.StatusMessage() == 'Invalid Eis Result thought to be good.':
            self.gc_readzstatus = GamryCOM.ReadZStatusRetry

        # Check measurement status
        if self.gc_readzstatus == GamryCOM.ReadZStatusOk:
            # Good measurement
            # Reset passes
            self.passes = 0

            # Increment total points
            self.total_points += 1

            # Store data
            self.z_data[self.frequency_index] = [data_timedelta] + self.get_current_zdata()

            # # If writing line-by-line, write new data point to file
            # if self.result_file is not None
            #     if self.write_mode == 'continuous':
            #     row_string = self.get_row_string(self.frequency_index)
            #     self._active_file.write(row_string)

            if self.result_file is not None:
                if self.write_mode == 'continuous':
                    # Write new data point
                    row_string = self.get_row_string(self.frequency_index)
                    self._active_file.write(row_string)
                    self._last_write_index += 1
                elif self.write_mode == 'interval':
                    # If a sufficient number of new points are available, write them all to the file
                    num_available = self.total_points - self._last_write_index
                    if num_available >= self.write_interval:
                        data_string = self.generate_data_string(self._last_write_index,
                                                                self._last_write_index + num_available)
                        with open(self.result_file, 'a') as f:
                            f.write(data_string)
                        self._last_write_index += num_available

            # Increment frequency index
            self.frequency_index += 1

            # If running plot animation, pause to plot new point(s)
            if self.show_plot:
                # Must incorporate plt.pause to run GUI event loop and allow artists to be drawn
                plt.pause(1e-5)

            # Move to next step
            if self.frequency_index == len(self.frequencies):
                # All requested frequencies measured. Close handle to terminate PumpEvents
                self.close_handle()

                # Final write
                if self.result_file is not None:
                    if self.write_mode == 'interval':
                        # Write all unwritten data points to file
                        num_available = self.total_points - self._last_write_index
                        if num_available > 0:
                            data_string = self.generate_data_string(self._last_write_index,
                                                                    self._last_write_index + num_available)
                            with open(self.result_file, 'a') as f:
                                f.write(data_string)
                            self._last_write_index += num_available
                    elif self.write_mode == 'once':
                        # Write all data to file
                        with open(self.result_file, 'a') as f:
                            data_strings = [self.get_row_string(i) for i in range(self.total_points)]
                            data_string = ''.join(data_strings)
                            f.write(data_string)


                # # If using one-time write, write all data to file
                # if result_file is not None and self.write_mode == 'once':
                #     with open(result_file, 'a') as f:
                #         data_string = ''
                #         for index in range(len(self.frequencies)):
                #             data_string += self.get_row_string(index)
                #         f.write(data_string)

                # Close result file
                if self.result_file is not None and self.write_mode == 'continuous':
                    self._active_file.close()
            else:
                # Measure next point
                self.measure_point(self.frequencies[self.frequency_index])
        elif self.gc_readzstatus == GamryCOM.ReadZStatusRetry and self.passes < self.max_passes:
            # Pstat settings need adjustment to obtain good data.
            # Print message, clear acquired_points, and retry (ReadZ will handle settings adjustments)
            print('Retry at frequency {:.2e} for reason: {}'.format(frequency, self.dtaq.StatusMessage()))
            self.acquired_points = []
            self.dtaq.Measure(frequency, self.ac_amp_req)  # self.measure_point would reset passes
        elif self.passes == self.max_passes:
            # Hit max number of passes without successful result
            # Skip to next point
            self.frequency_index += 1
            self.measure_point(self.frequencies[self.frequency_index])
        elif self.gc_readzstatus == GamryCOM.ReadZStatusError:
            # Error. Print message and close handle
            print(self.dtaq.StatusMessage())
            self.close_handle()

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
        self.pstat.SetIchRange(3.0)
        self.pstat.SetIchRangeMode(False)
        # self.pstat.SetIchFilter(2.5)  # 5?
        self.pstat.SetVchRange(3.0)
        self.pstat.SetVchRangeMode(False)

        # Offset enable seems to cause issues with high-frequency measurements
        # self.pstat.SetIchOffsetEnable(True)
        # self.pstat.SetVchOffsetEnable(True)

        # self.pstat.SetVchFilter(5)  # Causes "Invalid EIS result thought to be good" status at first frequency
        self.pstat.SetAchRange(3.0)

        self.pstat.SetIERange(0.03)
        self.pstat.SetIERangeMode(False)
        # self.pstat.SetIERangeLowerLimit(NIL)
        self.pstat.SetAnalogOut(0.0)

        self.pstat.SetPosFeedEnable(False)
        self.pstat.SetIruptMode(GamryCOM.IruptOff)

        # Mode-dependent settings
        if self.mode == 'galvanostatic':
            self.pstat.SetCASpeed(2)  # Normal
        elif self.mode == 'potentiostatic':
            self.pstat.SetCASpeed(3)  # MedFast
            self.pstat.SetVoltage(self.dc_amp_req)

            IERange = self.pstat.TestIERange((abs(self.dc_amp_req) + (2 ** 0.5) * self.ac_amp_req) / self.z_guess)
            print('IERange:', IERange)
            self.pstat.SetIERange(IERange)

        # Initialize dtaq to pstat
        self.dtaq.Init(self.pstat)
        self.dtaq.SetSpeed(getattr(GamryCOM, self.gc_readzspeed))
        self.dtaq.SetGain(1.0)
        self.dtaq.SetINoise(0.0)
        self.dtaq.SetVNoise(0.0)
        self.dtaq.SetIENoise(0.0)
        self.dtaq.SetZmod(self.z_guess)

        if self.mode == 'galvanostatic':
            self.dtaq.SetIdc(self.dc_amp_req)

            IERange = self.pstat.TestIERange(abs(self.dc_amp_req) + (2 ** 0.5) * self.ac_amp_req)  # Max current
            Rm = self.pstat.IEResistor(IERange)

            Sdc = Rm * self.dc_amp_req  # Voltage signal which generates I signal
            # Sac = Rm * self.i_ac_req

            # find the DC Voltage for VGS
            self.pstat.SetIERange(IERange)
            self.pstat.SetVoltage(Sdc)
            self.pstat.SetCell(GamryCOM.CellOn)  # turn the cell on
            time.sleep(3)

            print("Finding Vch Range...")
            self.pstat.FindVchRange()
            print('Sdc:', Sdc)
            print('Measured v:', self.pstat.MeasureV())
            # self.dtaq.SetVdc(self.pstat.MeasureV())
        elif self.mode == 'potentiostatic':
            self.dtaq.SetIdc(self.pstat.MeasureI())

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

        self.passes = 0
        self.dtaq.Measure(frequency, self.ac_amp_req)

    def run(self, pstat, frequencies, dc_amp, ac_amp, z_guess, max_passes=10, timeout=1000,
            result_file=None,
            show_plot=False, **plot_kw):
        # Store measurement parameters
        self.frequencies = frequencies
        self.dc_amp_req = dc_amp
        self.ac_amp_req = ac_amp
        self.z_guess = z_guess
        self.max_passes = max_passes

        self.pstat = pstat
        self.show_plot = show_plot

        self.result_file = result_file
        self._last_write_index = 0
        self.total_points = 0

        # Initialize data array
        self.z_data = np.zeros((len(frequencies), 10))

        # Create client connection and open pstat
        self.open_connection()

        try:
            # Set pstat settings and initialize dtaq
            self.initialize_pstat()

            print('running')

            if show_plot:
                self.ani = self.run_plot_animation(**plot_kw)

            self.start_time = time.time()

            if result_file is not None:
                # Write header
                with open(result_file, 'w+') as f:
                    f.write(self.generate_header_text())

                if self.write_mode == 'continuous':
                    self._active_file = open(result_file, 'a')
                else:
                    self._active_file = None

            # pstat.SetCell(GamryCOM.CellOn)

            # Start measurement at first point. All subsequent points are triggered by DataDone event
            self.frequency_index = 0
            self.measure_point(frequencies[self.frequency_index])

        except Exception as e:
            self.terminate()
            raise gamry_error_decoder(e)

        self.PumpEvents(timeout)
        # print('Zreal:', self.dtaq.Zreal())  # impedance MUST be extracted BEFORE pstat.Close()

        # Close connection if still active (may have been terminated prematurely)
        if self.connection_active:
            self.close_connection()

        # Stop plot animation once data collection complete
        if show_plot:
            self.ani.pause()

        print('Run time: {:.2f} s'.format(time.time() - self.start_time))

        # super().run(pstat, timeout, show_plot)

    # --------------------------
    # Plotting
    # --------------------------
    def run_plot_animation(self, plot_type='all', plot_interval=1.0):
        if plot_type == 'all':
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            nyquist_ax = axes[0]
            bode_axes = axes[1:]
        elif plot_type == 'nyquist':
            fig, ax = plt.subplots(figsize=(4, 3))
            nyquist_ax = ax
        elif plot_type == 'bode':
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            bode_axes = axes

        live_axes = []
        if plot_type in ('all', 'nyquist'):
            # Set axis_extend_ratios to same value to maintain aspect ratio
            lax_n = LiveAxes(nyquist_ax, axis_extend_ratio={'x': 0.25, 'y': 0.25})

            # Initialize axis limits with appropriate aspect ratio
            lax_n.ax.set_xlim(0, self.z_guess * 1.25)
            lax_n.ax.set_ylim(0, self.z_guess * 0.1)
            axis_limits = get_nyquist_limits(lax_n.ax, np.array([self.z_guess * 0.1, self.z_guess * (1 + 0.5j)]))
            lax_n.ax.set_xlim(axis_limits['x'])
            lax_n.ax.set_ylim(axis_limits['y'])

            # Data update function
            def update_nyquist(frame):
                x = self.z_data[:self.frequency_index, 2]
                y = -self.z_data[:self.frequency_index, 3]
                return x, y

            # Create artist
            lax_n.add_line_artist('z', update_nyquist, marker='.', ms=10, alpha=0.5, ls='')

            # Axis labels
            lax_n.ax.set_xlabel('$Z^\prime$ ($\Omega$)')
            lax_n.ax.set_ylabel('$-Z^{\prime\prime}$ ($\Omega$)')

            live_axes.append(lax_n)

        if plot_type in ('all', 'bode'):
            lax_b1 = LiveAxes(bode_axes[0])
            lax_b2 = LiveAxes(bode_axes[1])

            # Initialize axis limits
            for ax in bode_axes:
                ax.set_xlim(np.min(self.frequencies) / 5, np.max(self.frequencies) * 5)
                ax.set_xscale('log')
            bode_axes[0].set_ylim(0, self.z_guess * 1.25)  # modulus
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

            # Create artists
            lax_b1.add_line_artist('modulus', update_modulus, marker='.', ms=10, alpha=0.5, ls='')
            lax_b2.add_line_artist('phase', update_phase, marker='.', ms=10, alpha=0.5, ls='')

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

        live_axes[0].add_text_artist('status', 0.025, 0.025, update_status, ha='left', va='bottom',
                                     transform=live_axes[0].ax.transAxes)

        fig.tight_layout()

        lfig = LiveFigure(live_axes)

        return lfig.run(frames=1000, interval=plot_interval * 1000)

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
    def z_dataframe(self):
        if self.z_data is not None:
            return pd.DataFrame(self.z_data, columns=['Time', 'Freq', 'Zreal', 'Zimag', 'Zsig', 'Zmod', 'Zphz',
                                                      'Idc', 'Vdc', 'IERange']
                                )
        else:
            return None

    # -----------------------------
    # File writing
    # -----------------------------
    def get_row_string(self, frequency_index):
        row_data = [frequency_index] + list(rel_round(self.z_data[frequency_index], self.write_precision))
        row_string = '\t' + '\t'.join([str(x) for x in row_data]) + '\n'

        return row_string


# Cook column lookup
# -------------------
dtaq_cook_columns = {
    'GamryDtaqChrono': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Q',
        'Vsig',
        'Ach',
        'IERange',
        'Overload',
        'StopTest'
    ],
    'GamryDtaqOcv': [
        'Time',
        'Vf',
        'Vm',
        'Vsig',
        'Ach',
        'Overload',
        'StopTest',
        # Undocumented columns
        'Ignore',
        'Ignore',
        'Ignore'
    ],
    'GamryDtaqEis': [
        'I',
        'V'
    ],
    'GamryReadZ': [
        'I',
        'V'
    ],
}

dtaq_header_info = {
    'GamryDtaqOcv': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'T', 'Vf', 'Vm', 'Ach'],
        'units': ['#', 's', 'V vs. Ref.', 'V', 'V'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Vm', 'Ach']
    },
    'GamryReadZ': {
        'preceding': 'ZCURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Freq', 'Zreal', 'Zimag', 'Zsig', 'Zmod', 'Zphz', 'Idc', 'Vdc', 'IERange'],
        'units': ['#', 's', 'Hz', 'ohm', 'ohm', 'V', 'ohm', '°', 'A', 'V', '#'],
        'cook_columns': ['INDEX']
    }
}