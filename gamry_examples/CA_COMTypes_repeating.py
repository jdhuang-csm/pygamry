__author__ = "Dan Cook"
__credits__ = ["Dan Cook"]
__version__ = "7.8.4"
__status__ = "Example Only, Toolkit"

"""Runs a CA experiment"""

import time
import gc
import comtypes
import comtypes.client as client
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

GamryCOM = client.GetModule(['{BD962F0D-A990-4823-9CF5-284D1CDD9C6D}', 1, 0])


class GamryCOMError(Exception):
    pass

def gamry_error_decoder(e):
    if isinstance(e, comtypes.COMError):
        hresult = 2**32+e.args[0]
        if hresult & 0x20000000:
            return GamryCOMError('0x{0:08x}: {1}'.format(2**32+e.args[0], e.args[1]))
    return e


#inital settings for pstat. Sets DC Offset here based on dc setup parameter
def initializepstat(pstat):
    pstat.SetCtrlMode(GamryCOM.GstatMode)
    # pstat.SetCell(GamryCOM.CellOff)
    pstat.SetIEStability(GamryCOM.StabilityNorm)
    pstat.SetVchRangeMode(True)
    pstat.SetVchRange(10.0)
    # pstat.SetVoltage(0)
    # time.sleep(1)


class GamryDtaqEvents(object):
    def __init__(self, dtaq):
        self.dtaq = dtaq
        self.acquired_points = []

    def cook(self):
        count = 1
        while count > 0:
            count, points = self.dtaq.Cook(1024)
            # The columns exposed by GamryDtaq.Cook vary by dtaq and are
            # documented in the Toolkit Reference Manual.
            self.acquired_points.extend(zip(*points))

    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        self.cook()
        print("made it to data available")

    def _IGamryDtaqEvents_OnDataDone(self, this):
        print("made it to data done")
        self.cook()  # a final cook

        # Clear active status
        global active
        active = False

        # time.sleep(1.0)
        # stopacq()
#

def stopacq():

    global active
    global connection

    active = False

    print(dtaqsink.acquired_points)
    print(len(dtaqsink.acquired_points))

    pstat.SetCell(GamryCOM.CellOff)
    time.sleep(1)
    pstat.Close()
    # del connection
    gc.collect()
    return


def run(vinit, vstep, tinit, tstep, sample, n_steps):
    global dtaq
    global signal
    global dtaqsink
    global pstat
    global connection
    global active
    print("made it to run")

    # signal and dtaq object creation
    signal = client.CreateObject('GamryCOM.GamrySignalMstep')
    dtaq = client.CreateObject('GamryCOM.GamryDtaqChrono')

    dtaqsink = GamryDtaqEvents(dtaq)
    # connection = client.GetEvents(dtaq, dtaqsink)
    pstat = client.CreateObject('GamryCOM.GamryPC6Pstat')
    devices = client.CreateObject('GamryCOM.GamryDeviceList')
    pstat.Init(devices.EnumSections()[0])  # grab first pstat

    # Initialize pstat and dtaq
    pstat.Open()
    pstat.SetCell(GamryCOM.CellOff)
    pstat.SetVoltage(0)
    # initializepstat(pstat)

    # dtaq.Init(pstat, GamryCOM.ChronoPot)

    pstat.SetCell(GamryCOM.CellOn)

    # Initialize data array list
    data_arrays = []
    sleep_times = []
    sleep_currents = []
    sleep_voltages = []

    # Get start time
    start_time = time.time()

    # Loop through steps, increasing starting voltage with each step
    for i in range(n_steps):
        print(f'Running step {i}')
        # Clear acquired points
        dtaqsink.acquired_points = []
        # Set active status
        active = True

        connection = client.GetEvents(dtaq, dtaqsink)

        pstat.Open()
        initializepstat(pstat)
        dtaq.Init(pstat, GamryCOM.ChronoPot)

        # Configure next step signal
        signal.Init(pstat, vinit + i * vstep, vstep, tinit, tstep, 1, sample, GamryCOM.GstatMode)
        pstat.SetSignal(signal)

        # Get step offset time
        step_offset_time = time.time() - start_time

        # Turn cell on
        pstat.SetCell(GamryCOM.CellOn)

        print('dtaq running')
        t0 = time.time()
        dtaq.Run(True)
        # time.sleep(5)
        print('PumpEvents starting')
        while active:
            client.PumpEvents(1)
            time.sleep(0.1)
        print('Run time: {:.2f} s'.format(time.time() - t0))

        del connection
        gc.collect()

        # Append array with offset time
        tmp_data = np.array(dtaqsink.acquired_points)
        tmp_data[:, 0] += step_offset_time
        data_arrays.append(tmp_data)

        step_end_time = time.time()
        while time.time() - step_end_time < 2:
            sleep_currents.append(pstat.MeasureI())
            sleep_voltages.append(pstat.MeasureV())
            sleep_times.append(time.time() - start_time)

    # Close pstat once all steps complete
    stopacq()
    print("made it to run end")

    sleep_data = np.array([sleep_times, sleep_voltages, sleep_currents]).T

    return data_arrays, sleep_data


# active = True

#CA Setup Parameters
vinit = 0
tinit = 1
vstep = 1e-4
tstep = 5
sample = 0.1
n_steps = 2

if __name__ == "__main__":
    try:
        data_arrays, sleep_data = run(vinit, vstep, tinit, tstep, sample, n_steps)
        print("made it to try")

    except Exception as e:
        raise gamry_error_decoder(e)


# Define cook columns
cook_columns = [
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
]

# Construct dataframe of collected data
data_array = np.vstack(data_arrays)
data_df = pd.DataFrame(data_array, columns=cook_columns)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

axes[0].plot(data_df['Time'], data_df['Im'], marker='.')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('I (A)')

axes[0].plot(sleep_data[:, 0], sleep_data[:, 2])

axes[1].plot(data_df['Time'], data_df['Vf'], marker='.')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('V (V)')

axes[1].plot(sleep_data[:, 0], sleep_data[:, 1])

fig.tight_layout()