import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import comtypes.client as client
from comtypes.client._events import _handles_type, ctypes
import gc

from .config import GamryCOM, dtaq_cook_columns, dtaq_header_info, gc_string_dict, get_gc_to_str_map

from ..file_utils import get_file_time, read_last_line
from ..utils import gamry_error_decoder, check_write_mode, rel_round, time_format_code
from ..animation import LiveAxes, LiveFigure

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







