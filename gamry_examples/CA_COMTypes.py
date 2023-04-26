__author__ = "Dan Cook"
__credits__ = ["Dan Cook"]
__version__ = "7.8.4"
__status__ = "Example Only, Toolkit"

"""Runs a CA experiment"""

import time
import gc
import ctypes
from math import log10
import comtypes
import comtypes.client as client

GamryCOM=client.GetModule(['{BD962F0D-A990-4823-9CF5-284D1CDD9C6D}', 1, 0])


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
    pstat.SetCtrlMode(GamryCOM.PstatMode)
    pstat.SetCell(GamryCOM.CellOff)
    pstat.SetIEStability(GamryCOM.StabilityNorm)
    pstat.SetVchRangeMode (True)
    pstat.SetVchRange (10.0)

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
        print("made it to data avalible")

    def _IGamryDtaqEvents_OnDataDone(self, this):
        print("made it to data done")
        self.cook()  # a final cook
        time.sleep(2.0)
        stopacq()

def stopacq():

    global active
    global connection

    active = False

    print(dtaqsink.acquired_points)
    print(len(dtaqsink.acquired_points))

    pstat.SetCell(GamryCOM.CellOff)
    time.sleep(1)
    pstat.Close()
    del connection
    gc.collect()
    return

def run(vinit, tinit, vstep1, tstep1, vstep2, tstep2, sample):
    global dtaq
    global signal
    global dtaqsink
    global pstat
    global connection
    global dtaqsink
    print("made it to run")

    # signal and dtaq object creation
    signal = client.CreateObject('GamryCOM.GamrySignalDstep')
    dtaq = client.CreateObject('GamryCOM.GamryDtaqChrono')

    dtaqsink = GamryDtaqEvents(dtaq)
    connection = client.GetEvents(dtaq, dtaqsink)
    pstat = client.CreateObject('GamryCOM.GamryPC5Pstat')
    devices = client.CreateObject('GamryCOM.GamryDeviceList')
    pstat.Init(devices.EnumSections()[0])  # grab first pstat

    pstat.Open()


    signal.Init(pstat, vinit, tinit, vstep1, tstep1, vstep2, tstep2, sample, GamryCOM.PstatMode)
    initializepstat(pstat)

    dtaq.Init(pstat, GamryCOM.ChronoAmp)
    pstat.SetSignal(signal)
    pstat.SetCell(GamryCOM.CellOn)

    dtaq.Run(True)
    print("made it to run end")

active = True

#CA Setup Parameters
vinit = 0.5
tinit= 10
vstep1= 0.5
tstep1=20
vstep2=6.6
tstep2=20
sample=0.01



if __name__ == "__main__":
    try:
        run(vinit, tinit, vstep1, tstep1, vstep2, tstep2, sample)
        print("made it to try")
        while active == True:
            client.PumpEvents(1)
            time.sleep(0.1)

    except Exception as e:
        raise gamry_error_decoder(e)