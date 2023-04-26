# -*- coding: utf-8 -*-
# Get comtypes from:
# sourceforge -- http://sourceforge.net/projects/comtypes/files/comtypes/
# or
# PyPI -- https://pypi.python.org/pypi/comtypes
from __future__ import print_function
import comtypes
import comtypes.client as client

# Alternatively:
#GamryCOM=client.GetModule(r'C:\Program Files\Gamry Instruments\Framework 6\GamryCOM.exe')

# utilities: #####################
class GamryCOMError(Exception):
    pass

def gamry_error_decoder(e):
    if isinstance(e, comtypes.COMError):
        hresult = 2**32+e.args[0]
        if hresult & 0x20000000:
            return GamryCOMError('0x{0:08x}: {1}'.format(2**32+e.args[0], e.args[1]))
    return e

class GamryDtaqEvents(object):
    def __init__(self, dtaq):
        self.dtaq = dtaq
        self.acquired_points = []
        
    def cook(self):
        count = 1
        while count > 0:
            count, points = self.dtaq.Cook(10)
            # The columns exposed by GamryDtaq.Cook vary by dtaq and are
            # documented in the Toolkit Reference Manual.
            self.acquired_points.extend(zip(*points))
        
    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        self.cook()

    def _IGamryDtaqEvents_OnDataDone(self, this):
        self.cook() # a final cook
        # TODO:  indicate completion to enclosing code?
###############################

devices=client.CreateObject('GamryCOM.GamryDeviceList')
print(devices.EnumSections())

pstat=client.CreateObject('GamryCOM.GamryPC6Pstat')
pstat.Init(devices.EnumSections()[0]) # grab first pstat

pstat.Open()

dtaqcpiv=client.CreateObject('GamryCOM.GamryDtaqCpiv')
dtaqcpiv.Init(pstat)

sigramp=client.CreateObject('GamryCOM.GamrySignalRamp')
sigramp.Init(pstat, -0.25, 0.25, 1, 0.01, GamryCOM.PstatMode)

pstat.SetSignal(sigramp)
pstat.SetCell(GamryCOM.CellOn)

dtaqsink = GamryDtaqEvents(dtaqcpiv)

# Use the following code to discover events:
#client.ShowEvents(dtaqcpiv)
connection = client.GetEvents(dtaqcpiv, dtaqsink)

try:
    dtaqcpiv.Run(True)
except Exception as e:
    raise gamry_error_decoder(e)

client.PumpEvents(1)
print(len(dtaqsink.acquired_points))

print(dtaqsink.acquired_points)

del connection

pstat.Close()
