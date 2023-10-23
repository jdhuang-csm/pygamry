import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import comtypes.client as client
from comtypes.client._events import _handles_type, ctypes
import gc
import warnings

from .file_utils import get_file_time, read_last_line, get_decimation_index
from .utils import gamry_error_decoder, check_write_mode, check_control_mode, rel_round, time_format_code
from . import signals
from .animation import LiveAxes, LiveFigure, LiveNyquist
from .plotting import get_nyquist_limits
from .filters import filter_chrono_signal

GamryCOM = client.GetModule(['{BD962F0D-A990-4823-9CF5-284D1CDD9C6D}', 1, 0])


def get_pstat(family='Interface', retry=5, device_index=0):
    """
    Get potentiostat
    :param str family: potentiostat family. Options: 'Interface', 'Reference'
    :return:
    """
    pstat = None
    iteration = 0
    while pstat is None:
        try:
            devices = client.CreateObject('GamryCOM.GamryDeviceList')
            print(devices.EnumSections())

            if family.lower() == 'interface':
                obj_string = 'GamryCOM.GamryPC6Pstat'
            elif family.lower() == 'reference':
                obj_string = 'GamryCOM.GamryPC5Pstat'
            else:
                raise ValueError(f'Invalid family argument {family}')

            pstat = client.CreateObject(obj_string)
            pstat.Init(devices.EnumSections()[device_index])
            return pstat
        except Exception as err:
            pstat = None
            iteration += 1

            if iteration == retry:
                print('Could not find an available potentiostat')
                raise (err)
            else:
                print('No pstat available. Retrying in 1 s')
                time.sleep(1)


class GamryDtaqEventSink(object):
    def __init__(self, dtaq_class, tag, axes_def, start_with_cell_off=True, leave_cell_on=False,
                 write_precision=6, write_mode='interval', write_interval=1, str_construct_method='pandas',
                 test_id='Title',
                 exp_notes=None):

        self.dtaq_class = dtaq_class
        self.tag = tag
        self.test_id = test_id
        self.axes_def = axes_def

        # Create dtaq source
        self.dtaq = client.CreateObject(f'GamryCOM.{dtaq_class}')

        self.pstat = None
        self.connection = None
        self.connection_active = False
        self.measurement_complete = False
        self.start_with_cell_off = start_with_cell_off
        self.leave_cell_on = leave_cell_on

        self.signal = None
        self.signal_params = {}  # empty dict to make get method accessible
        self.expected_duration = None
        self.i_req = None  # Requested current
        self.v_req = None  # Requested voltage

        # Create data containers
        self.acquired_points = []
        self.total_points = None
        self._new_count = None
        # self.counts = []
        # self.count_times = []

        self.start_time = None
        self.data_time_offset = 0  # Time offset based on time "0" in acquired_points
        # self.show_plot = None
        self.fig = None
        self.axes = None
        self.live_fig = None
        self.ani = None

        # Event handling
        self.handler_routine = None
        self.hevt = None  # Handle for PumpEvents

        # Result file and write options
        self.result_file = None
        self.append_to_file = None
        self.file_start_time = None
        self.start_index_from = None
        self.file_time_offset = None
        self.write_mode = write_mode
        self._active_file = None
        self.write_precision = write_precision
        self.write_interval = write_interval
        self.str_construct_method = str_construct_method
        self.exp_notes = exp_notes
        self._last_write_index = None

        # kst
        self.kst_file = None
        self._last_kst_write_index = None

    # -----------------------------
    # Configuration
    # -----------------------------
    @property
    def pstat_ctrlmode(self):
        gc_ctrlmode = self.pstat.CtrlMode()
        gc_map = get_gc_to_str_map(gc_string_dict['CtrlMode'])
        return gc_map[gc_ctrlmode]

    # COM methods
    # --------------------------
    def cook(self, num_points=1024):
        count = 1
        tot_count = 0
        while count > 0:
            try:
                count, points = self.dtaq.Cook(num_points)
            except Exception as e:
                raise gamry_error_decoder(e)
            # The columns exposed by GamryDtaq.Cook vary by dtaq and are
            # documented in the Toolkit Reference Manual.
            self.acquired_points.extend(zip(*points))

            tot_count += count

        self.total_points += tot_count

        return tot_count

    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        # print('DataAvailable fired')
        self._new_count = self.cook()

        self.write_to_files(self._new_count, False)
        # if new_count > 0 and self.result_file is not None:
        #     if self.write_mode == 'continuous':
        #         # Write all newly received data points
        #         data_string = self.generate_data_string(self._last_write_index, self._last_write_index + new_count,
        #                                                 method='pandas')
        #         self._active_file.write(data_string)
        #         self._last_write_index += new_count
        #     elif self.write_mode == 'interval':
        #         # If a sufficient number of new points are available, write them all to the file
        #         num_available = self.total_points - self._last_write_index
        #         if num_available >= self.write_interval:
        #             data_string = self.generate_data_string(self._last_write_index,
        #                                                     self._last_write_index + num_available)
        #             with open(self.result_file, 'a') as f:
        #                 f.write(data_string)
        #             self._last_write_index += num_available

        # print('OnDataAvailable finished')

    def _IGamryDtaqEvents_OnDataDone(self, this):
        self._new_count = self.cook()  # a final cook

        print('DataDone')

        # Update measurement status to complete
        self.measurement_complete = True

        # Final write
        # self.write_to_files(new_count, True)
        # Moved final write to run_main. This ensures that file is written even if DataDone event is never fired
        # (e.g. if timed out or measurement error occurred)

        # If specific current or voltage was not requested, measure current/voltage at end of experiment
        if self.pstat_ctrlmode == 'PstatMode' and self.v_req is None:
            self.v_req = self.pstat.MeasureV()
        elif self.pstat_ctrlmode == 'GstatMode' and self.i_req is None:
            self.i_req = self.pstat.MeasureI()

        # Close handle to terminate PumpEvents and close connection
        self.close_connection()
        print('Connection closed')

    # Event control
    # --------------------------------
    def PumpEvents(self, timeout):
        """
        Code based on comtypes.client._events.PumpEvents
        Modified to enable early termination when desired conditions are met (e.g. data collection finished)
        """

        # Store handle as instance attribute to enable other methods to close handle prior to timeout
        self.hevt = ctypes.windll.kernel32.CreateEventA(None, True, False, None)
        handles = _handles_type(self.hevt)
        RPC_S_CALLPENDING = -2147417835

        def handler_routine(dwCtrlType):
            if dwCtrlType == 0:  # CTRL+C
                ctypes.windll.kernel32.SetEvent(self.hevt)
                return 1
            return 0

        self.handler_routine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(handler_routine)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(self.handler_routine, 1)

        try:
            try:
                res = ctypes.oledll.ole32.CoWaitForMultipleHandles(0,
                                                                   int(timeout * 1000),
                                                                   len(handles), handles,
                                                                   ctypes.byref(ctypes.c_ulong()))
                print(res)
            except WindowsError as details:
                if details.winerror != RPC_S_CALLPENDING:  # timeout expired
                    raise
            else:
                # KeyboardInterrupt - stop measurement
                print('Received KeyboardInterrupt')
                self.close_connection()
        finally:
            self.close_handle()

    def close_handle(self):
        """
        Method to close handle and terminate PumpEvents
        :return:
        """
        ctypes.windll.kernel32.CloseHandle(self.hevt)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(self.handler_routine, 0)

    def run_main(self, pstat, result_file=None, kst_file=None, append_to_file=False, timeout=1000,
                 show_plot=False, plot_interval=10, repeats=1):
        """
        Main run method to be called by subclasses
        :param pstat: potentiostat instance
        :param str result_file: path to file to which results will be saved. If None, results will not be written to
        file, but will be available in the data_array attribute
        :param bool append_to_file: if True, append results to result_file. result_file must contain results from a
        previous measurement of the same type
        :param timeout:
        :param show_plot:
        :param plot_interval:
        :return:
        """
        # TODO: implement safe run with num_retries and exception catch
        # Requirements:
        # If measurement_complete = False, retry up until num_retries
        # Catch any exception. Don't allow script to exit. Print exceptions as caught
        t0 = time.time()
        self.pstat = pstat

        # # Clear acquired points
        # self.acquired_points = []
        # self.total_points = 0
        #
        # # Reset measurement status
        # self.measurement_complete = False
        #
        # # Reset last write index
        # self._last_write_index = 0

        # TODO: handle append_to_file when repeats > 1.
        # File params
        self.append_to_file = append_to_file
        self.kst_file = kst_file

        try:
            # Open connection and pstat - must be done before initialize_pstat
            self.open_connection()

            # Configure pstat for requested measurement
            self.initialize_pstat()
            print('init time: {:.3f}'.format(time.time() - t0))
            # Run
            print('running')

            for rep in range(repeats):
                loop_start = time.time()
                if repeats > 1:
                    print(f'Rep {rep}')

                # If measurement is to be repeated, add a suffix to the file name to differentiate repetitions
                if repeats > 1 and result_file is not None:
                    ext_index = -result_file[::-1].find('.') - 1
                    current_file = '{}_{}{}'.format(result_file[:ext_index], rep, result_file[ext_index:])
                else:
                    current_file = result_file
                self.result_file = current_file

                # Clear acquired points
                self.acquired_points = []
                self.total_points = 0

                # Reset measurement status
                self.measurement_complete = False

                # Reset last write index
                self._last_write_index = 0
                self._last_kst_write_index = 0

                # Re-open connection and pstat - must be done at each repetition after the first
                if rep > 0:
                    self.open_connection()

                # Record start time
                self.start_time = time.time()

                # Begin measurement. Starts running immediately, but we only start receiving events
                # once we call PumpEvents
                self.dtaq.Run()

                # Create result file
                self.get_file_offset_info(append_to_file, self.result_file)
                if self.result_file is not None:
                    # Write header
                    if not append_to_file:
                        with open(self.result_file, 'w+') as f:
                            f.write(self.generate_header_text())

                    if self.write_mode == 'continuous':
                        # Keep file open for continuous writing
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
                    while len(self.acquired_points) == 0:
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
                    # # If not plotting, pump events continuously
                    # self.PumpEvents(timeout)
                    # Seems to work better if pump events is called repeatedly with short timeout
                    # Otherwise, PumpEvents will sometimes "hang" even after closing the connection
                    while self.connection_active:
                        self.PumpEvents(1.0)
                        time.sleep(0.25)

                        # Check for timeout
                        if time.time() - self.start_time > timeout:
                            print('Timed out')
                            break
                print('PumpEvents done')

                print('dtaq run time: {:.3f} s'.format(time.time() - self.start_time))

                # Final write to file after DataDone or timeout
                ws = time.time()
                self.write_to_files(self._new_count, True)
                print('Final write time: {:.3f}'.format(time.time() - ws))
                print('loop time: {:.3f}'.format(time.time() - loop_start))
        except Exception as e:
            # If any exception raised, terminate measurement and disconnect
            self.terminate()
            # Final write to file
            self.write_to_files(self._new_count, True)
            raise gamry_error_decoder(e)

        tt = time.time()
        # Close connection if still active (can happen if measurement timed out)
        if self.connection_active:
            self.close_connection()

        # Stop plot animation once data collection complete
        if show_plot:
            self.ani.pause()

        # Garbage collection
        gc.collect()
        print('Cleanup time: {:.3f}'.format(time.time() - tt))

        print('Run time: {:.2f} s'.format(time.time() - self.start_time))

    def open_connection(self):
        """
        Create connection and open pstat
        :return:
        """
        self.connection = client.GetEvents(self.dtaq, self)
        self.pstat.Open()
        self.connection_active = True

        if self.start_with_cell_off:
            self.pstat.SetCell(GamryCOM.CellOff)

    def close_connection(self):
        """
        Delete connection and collect garbage at end of measurement. Called at DataDone
        If leave_cell_on is False, turn cell off and close pstat. Otherwise, leave cell on and pstat open.
        :return:
        """
        # Turn cell off unless directed to leave on
        if self.leave_cell_on:
            # Leave cell on with requested current or voltage
            pass
            # if self.pstat_ctrlmode == 'PstatMode':
            #     self.pstat.SetVoltage(self.v_req)
            #     print('Set voltage to {}'.format(self.v_req))
            # elif self.pstat_ctrlmode == 'GstatMode':
            #     # Set IE range to requested current
            #     self.pstat.SetIERange(self.i_req)
            #     # Get measurement resistor
            #     R_ie = self.pstat.IEResistor(self.pstat.IERange())
            #     # Set voltage to produce requested current
            #     self.pstat.SetVoltage(self.i_req * R_ie)
            #     print('Set voltage to {}'.format(self.i_req * R_ie))
        else:
            # TUrn cell off and close potentiostat
            self.pstat.SetCell(GamryCOM.CellOff)
            self.pstat.Close()

        # Close handle
        self.close_handle()

        # Delete connection
        del self.connection
        self.connection = None
        self.connection_active = False

        # Garbage collection - moved to run_main to avoid deleting reference to active file
        # gc.collect()

    def terminate(self):
        """Terminate prematurely - called if exception encountered during measurement"""
        # Stop measurement
        self.dtaq.Stop()

        # Close file if open
        # if self._active_file is not None:
        #     self._active_file.close()
        #     self._active_file = None

        # Close connection
        self.close_connection()

        # Close potentiostat to ensure new scripts/runs have access to it
        if self.pstat.TestIsOpen():
            if not self.leave_cell_on:
                self.pstat.SetCell(GamryCOM.CellOff)
            self.pstat.Close()

    # ------------------------------------
    # Subclass method placeholders
    # ------------------------------------
    def initialize_figure(self):
        self.fig, self.axes = plt.subplots(1, len(self.axes_def), figsize=(len(self.axes_def) * 4.5, 3.25))
        self.axes = np.atleast_1d(self.axes)

        # Define single data update function
        x_col_index = np.empty(len(self.axes_def), dtype=int)
        y_col_index = np.empty(len(self.axes_def), dtype=int)

        def update_data(frame, ax_index):
            if len(self.data_array) > 0:
                x = self.data_array[:, x_col_index[ax_index]]
                y = self.data_array[:, y_col_index[ax_index]]
            else:
                x = [np.nan]
                y = [np.nan]
            return x, y

        live_axes = []

        for i, (ax, ax_def) in enumerate(zip(self.axes, self.axes_def)):
            ax.set_title(ax_def['title'])

            if ax_def['type'] == 'y(t)':
                # Plot data column vs. time
                # Set x-axis parameters

                # Get x column index
                x_col_index[i] = 0

                # Set x label
                ax.set_xlabel('$t$ (s)')

                # Initialize x-axis limits
                # For longer experiments, start with x-axis no longer than 1 hour
                ax.set_xlim(self.data_time_offset, min(3600, self.expected_duration))

            elif ax_def['type'] == 'y(x)':
                # Plot data y column vs. x data column
                # Get x column index
                x_col_index[i] = self.cook_columns.index(ax_def['x_column'])

                # Set x label
                ax.set_xlabel(ax_def.get('x_label', ax_def['x_column']))

                # Initialize x-axis limits
                if len(self.acquired_points) > 0:
                    # Set based on initial data point(s)
                    x_min = np.min(self.data_array[:, x_col_index[i]])
                    x_max = np.max(self.data_array[:, x_col_index[i]])
                    x_range = max(x_max - x_min, max(abs(x_max), abs(x_min)) * 1e-2)
                    ax.set_xlim(x_min - x_range * 0.25, x_max + x_range * 0.25)
                else:
                    ax.set_xlim(-0.001, 0.001)

            # Get y column index
            y_col_index[i] = self.cook_columns.index(ax_def['y_column'])

            # Set y label
            ax.set_ylabel(ax_def.get('y_label', ax_def['y_column']))

            # Initialize y-axis limits
            if len(self.acquired_points) > 0:
                # Set based on initial data point(s)
                y_min = np.min(self.data_array[:, y_col_index[i]])
                y_max = np.max(self.data_array[:, y_col_index[i]])
                y_range = max(y_max - y_min, max(abs(y_max), abs(y_min)) * 1e-2)
                ax.set_ylim(y_min - y_range * 0.25, y_max + y_range * 0.25)
            else:
                ax.set_ylim(-0.001, 0.001)

            # Set sci limits
            ax.ticklabel_format(scilimits=(-3, 3))

            # Create LiveAxes instance
            lax = LiveAxes(ax)  # fixed_xlim=(0, self.signal_params['duration']))
            lax.add_line_artist(ax_def['y_column'], update_data, data_update_kwargs={'ax_index': i},
                                marker='.', ms=4, alpha=0.6, ls='')

            live_axes.append(lax)

        self.fig.tight_layout()

        self.live_fig = LiveFigure(live_axes)

    def run_plot_animation(self, plot_interval):
        return self.live_fig.run()

    def plot_data(self):
        self.initialize_figure()
        self.live_fig.plot_static()

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
    def num_points(self):
        return len(self.acquired_points)

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
        return pd.DataFrame(write_data, index=pd.RangeIndex(start_index, end_index), columns=np.array(cook_cols))

    def get_kst_dataframe(self, start_index, end_index):
        """
        DataFrame formatted for Kst
        :return:
        """
        try:
            kst_data = self.data_array[start_index:end_index, self.kst_column_index].astype(float)
        except TypeError:
            kst_data = self.data_array[start_index:end_index, self.kst_column_index]
        return pd.DataFrame(kst_data, index=pd.RangeIndex(start_index, end_index), columns=self.kst_columns)

    # ---------------------------------
    # File writing
    # ---------------------------------
    # Header creation
    def format_start_date_time(self):
        # Offset start timestamp by data offset to ensure that (timestamp + elapsed time) gives actual measurement time
        gmtime = time.gmtime(self.start_time - self.data_time_offset)
        # Get date string
        date_string = time.strftime(time_format_code.split(' ')[0], gmtime)
        # Get time string
        time_dec = (self.start_time - self.data_time_offset) % 1  # decimal seconds
        if self.write_precision is not None:
            time_dec_str = str(rel_round(time_dec, self.write_precision))[2:2 + self.write_precision]
        else:
            time_dec_str = str(time_dec)
        time_string = time.strftime('{}.{}'.format(time_format_code.split(' ')[1], time_dec_str), gmtime)

        return date_string, time_string

    def get_date_time_text(self):
        """Get formatted start date/time text to write to file header"""
        date_string, time_string = self.format_start_date_time()
        text = f'DATE\tLABEL\t{date_string}\tDate\n'
        text += f'TIME\tLABEL\t{time_string}\tTime\n'

        return text

    def get_data_header(self):
        """
        Get header for data table
        :return:
        """
        header_info = dtaq_header_info[self.dtaq_class]
        name_row = '\t' + '\t'.join(header_info['columns']) + '\n'
        unit_row = '\t' + '\t'.join(header_info['units']) + '\n'
        header_text = header_info['preceding'] + '\n' + name_row + unit_row

        return header_text

    def get_dtaq_header(self):
        """
        Get dtaq class-specific header text that is either (a) required to make file readable by Echem Analyst
        or (b) important information for measurement type. Define at subclass level
        :return:
        """
        return ''

    def get_notes_text(self):
        """Format notes text for file header"""
        if self.exp_notes is None:
            text = ''
        else:
            num_lines = len(self.exp_notes.split('\n'))
            text = f'NOTES\tNOTES\t{num_lines}\tNotes...\n'
            # Indent notes text
            text += '\t' + self.exp_notes.replace('\n', '\n\t') + '\n'
        return text

    def generate_header_text(self):
        """Generate full file header"""
        text = 'EXPLAIN\n' + f'TAG\t{self.tag}\n' + f'TITLE\tLABEL\t{self.test_id}\tTest Identifier\n' + \
               self.get_date_time_text() + self.get_dtaq_header() + self.get_notes_text() + self.get_data_header()
        return text

    def generate_kst_header(self):
        col_labels = self.kst_columns
        col_labels = ['Index'] + col_labels  # add index column label
        return '\t'.join(col_labels) + '\n'

    # Data writing
    # def get_row_string(self, row_index):
    #     # Apply offsets
    #     # Start from last index previously written to file
    #     write_index = row_index + self.start_index_from
    #     write_row = self.data_array[row_index]
    #     if self.append_to_file:
    #         # Offset times by time delta
    #         write_row[0] += self.file_time_offset
    #
    #     col_index = self.column_index_to_write
    #     row_data = [write_index] + list(rel_round(self.data_array[row_index, col_index], self.write_precision))
    #     row_string = '\t'.join([str(x) for x in row_data]) + '\n'
    #
    #     return row_string

    def generate_data_string(self, data_func, start_index, end_index, indent=True):
        # if method == 'join':
        #     # Join individual row strings
        #     data_strings = [self.get_row_string(i) for i in range(start_index, end_index)]
        #     data_string = ''.join(data_strings)
        # elif method == 'pandas':

        # Use pandas dataframe to generate csv text
        df = data_func(start_index, end_index)
        if self.append_to_file:
            # Apply offsets
            df = df.set_index(df.index + self.start_index_from)
            if 'Time' in df.columns:
                df['Time'] = df['Time'] + self.file_time_offset
        data_string = df.to_csv(None, sep='\t', header=False, lineterminator='\n',
                                float_format=f'%.{self.write_precision + 1}g')

        # Pad left with tabs
        if indent:
            data_string = '\t' + data_string.replace('\n', '\n\t')[:-1]

        return data_string

    # def generate_kst_string(self, start_index, end_index):
    #     # Use pandas dataframe to generate csv text
    #     df = self.kst_dataframe.loc[start_index:end_index, :]
    #     if self.append_to_file:
    #         # Apply offsets
    #         df = df.set_index(df.index + self.start_index_from)
    #         df['Time'] = df['Time'] + self.file_time_offset
    #     data_string = df.to_csv(None, sep='\t', header=False, lineterminator='\n',
    #                             float_format=f'%.{self.write_precision + 1}g')
    #
    #     return data_string

    def write_to_files(self, new_count, is_final_write):
        # print('active_file:', self._active_file)
        self._last_write_index = self.write_to_file(self.get_dataframe_to_write, self.result_file,
                                                    self._active_file, self._last_write_index,
                                                    new_count, is_final_write, indent=True)

        self._last_kst_write_index = self.write_to_file(self.get_kst_dataframe, self.kst_file,
                                                        None, self._last_kst_write_index, new_count,
                                                        is_final_write, indent=False)

        # Purge the active file attribute if finished writing
        if is_final_write:
            self._active_file = None

    def write_to_file(self, data_func, destination_file, active_file, last_write_index, new_count,
                      is_final_write, indent):

        if destination_file is not None:
            if is_final_write:
                if self.write_mode == 'continuous':
                    # Write all newly received data points and close the active file
                    if new_count > 0:
                        data_string = self.generate_data_string(data_func, last_write_index,
                                                                last_write_index + new_count, indent)
                        if active_file is not None:
                            active_file.write(data_string)
                        else:
                            # If no active_file exists, open the file, write, and close
                            # This is the case for Kst files
                            with open(destination_file, 'a') as f:
                                f.write(data_string)
                    # Close file
                    if active_file is not None:
                        active_file.close()
                    last_write_index += new_count
                elif self.write_mode == 'interval':
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
            elif new_count > 0:
                if self.write_mode == 'continuous':
                    # Write all newly received data points. Don't close the active file
                    data_string = self.generate_data_string(data_func, last_write_index,
                                                            last_write_index + new_count, indent)
                    if active_file is not None:
                        active_file.write(data_string)
                    else:
                        # If no active_file exists, open the file, write, and close
                        # This is the case for Kst files
                        with open(destination_file, 'a') as f:
                            f.write(data_string)
                    last_write_index += new_count
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

    @property
    def column_index_to_write(self):
        """Get indices of cook columns to write to result file"""
        cook_cols = dtaq_header_info[self.dtaq_class]['cook_columns'][1:]  # Skip INDEX
        index = [self.cook_columns.index(col) for col in cook_cols]

        return np.array(index)

    @property
    def kst_columns(self):
        return dtaq_header_info.get(self.dtaq_class, {}).get('kst_columns', [])

    @property
    def kst_column_index(self):
        return [self.cook_columns.index(col) for col in self.kst_columns]

    def get_file_offset_info(self, append_to_file, result_file):
        if result_file is not None:
            if append_to_file:
                # Get start time from file timestamp
                self.file_start_time = get_file_time(result_file)
                # Get last index written to file
                last_line = read_last_line(result_file)
                self.start_index_from = int(last_line.split('\t')[1]) + 1
            else:
                self.file_start_time = self.start_time
                self.start_index_from = 0
            self.file_time_offset = self.start_time - self.file_start_time - self.data_time_offset

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


# =========================================================
# Potentiostatic scan
# =========================================================
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


# =========================================================
# Galvanostatic scan
# =========================================================
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


# =========================================================
# Polarization curve
# =========================================================
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


# =========================================================
# Chrono
# =========================================================
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

    def get_step_times(self):
        """Determine step times from signal"""
        if self.signal_params['signal_class'] == 'Dstep':
            step_times = np.array([0, self.signal_params['t_step1']])
        elif self.signal_params['signal_class'] == 'Mstep':
            step_times = self.signal_params['t_init'] + \
                         np.arange(0, self.signal_params['n_steps']) * self.signal_params['t_step']
        elif self.signal_params['signal_class'] == 'Array':
            step_times = self.signal_params['step_times']
        else:
            step_times = None

        return step_times

    def get_step_index(self):
        step_times = self.get_step_times()
        times = self.data_array[:, self.cook_columns.index('Time')]

        def pos_delta(x, x0):
            out = np.empty(len(x))
            out[x < x0] = np.inf
            out[x >= x0] = x[x >= x0] - x0
            return out

        return np.array([np.argmin(pos_delta(times, st)) for st in step_times])

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


# =========================================================
# ReadZ (EIS)
# =========================================================
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
        
        # Get recommended value baed on mode
        if self.mode == 'galv':
            recommended = False
        else:
            recommended = True
            
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
        if self.mode == 'galv':
            self.pstat.SetIchOffsetEnable(True)
        else:
            self.pstat.SetIchOffsetEnable(False)
        # if self.dc_amp_req < 0:
        # self.pstat.SetVchOffsetEnable(True)

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


def estimate_eis_duration(frequencies):
    cycle_time = 1 / frequencies
    max_cycles = get_max_eis_cycles(frequencies)
    freq_times = np.maximum(cycle_time * max_cycles, 1)
    return np.sum(freq_times)


# ================================
# GamryCOM mapping
# ================================
def get_gc_to_str_map(string_list):
    return {getattr(GamryCOM, string): string for string in string_list}


def get_str_to_gc_map(string_list):
    return {string: getattr(GamryCOM, string) for string in string_list}


gc_string_dict = {
    'CtrlMode': ['GstatMode', 'PstatMode'],
}

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
    'GamryDtaqCpiv': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Vsig',
        'Ach',
        'IERange',
        'Overload',
        'StopTest'
    ],
    'GamryDtaqCiiv': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Vsig',
        'Ach',
        'IERange',
        'Overload',
        'StopTest'
    ],
    'GamryDtaqPwr': [
        'Time',
        'Vf',
        'Vu',
        'Im',
        'Pwr',
        'R',
        'Vsig',
        'Ach',
        'Temp',
        'IERange',
        'Overload',
        'StopTest',
        'StopTest2'
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
        'columns': ['Pt', 'Time', 'Vf', 'Vm', 'Ach'],
        'units': ['#', 's', 'V vs. Ref.', 'V', 'V'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Vm', 'Ach'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'OCV'},
        'kst_columns': ['Time', 'Vf']
    },
    'GamryDtaqCpiv': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Sig', 'Ach', 'IERange'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'V', 'V', '#'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'V', 'Im': 'I'},
        'kst_columns': ['Time', 'Vf', 'Im']
    },
    'GamryDtaqCiiv': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Sig', 'Ach', 'IERange'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'V', 'V', '#'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'V', 'Im': 'I'},
        'kst_columns': ['Time', 'Vf', 'Im']
    },
    'GamryDtaqPwr': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Pwr', 'Sig', 'Ach', 'IERange', 'ImExpected'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'W', 'V', 'V', '#', 'A'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Pwr', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Vf': 'V', 'Im': 'I', 'Pwr': 'P'},
        'kst_columns': ['Vf', 'Im', 'Pwr']
    },
    'GamryReadZ': {
        'preceding': 'ZCURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Freq', 'Zreal', 'Zimag', 'Zsig', 'Zmod', 'Zphz', 'Idc', 'Vdc', 'IERange'],
        'units': ['#', 's', 'Hz', 'ohm', 'ohm', 'V', 'ohm', '°', 'A', 'V', '#'],
        'cook_columns': ['INDEX'],
        # 'kst_map': {'Freq': 'f', 'Zreal': "Z'", 'Zimag': "Z''", 'Zmod': '|Z|', 'Zphz': 'Phase'},
        'kst_columns': ['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphz']
    },
    'GamryDtaqChrono': {
        'preceding': 'CURVE\tTABLE',
        'columns': ['Pt', 'Time', 'Vf', 'Im', 'Vu', 'Sig', 'Ach', 'IERange'],
        'units': ['#', 's', 'V vs. Ref.', 'A', 'V', 'V', 'V', '#'],
        'cook_columns': ['INDEX', 'Time', 'Vf', 'Im', 'Vu', 'Vsig', 'Ach', 'IERange'],
        # 'kst_map': {'Time': 'Time', 'Vf': 'V', 'Im': 'I'},
        'kst_columns': ['Time', 'Vf', 'Im']
    }
}
