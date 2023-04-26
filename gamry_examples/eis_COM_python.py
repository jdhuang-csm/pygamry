__author__ = "Dan Cook"
__credits__ = ["Dan Cook", "Abe Krebs"]
__version__ = "7.07"
__status__ = "Example Only, Toolkit"

"""Runs a Potentiostatic EIS experiment"""

import time
import gc
import ctypes
from math import log10
import comtypes
import comtypes.client as client


GamryCOM=client.GetModule(['{BD962F0D-A990-4823-9CF5-284D1CDD9C6D}', 1, 0])
#Alternatively you can point to 'C:\Program Files (x86)\Gamry Instruments\Framework\GamryCom.exe'

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
    pstat.SetIEStability(GamryCOM.StabilityFast)
    pstat.SetVoltage(dc)

#class object that handles events fired from GamryCom
class GamryReadZEvents(object):
    def __init__(self, dtaq):
        self.dtaq = dtaq
        self.acquired_points = []

    #unlike other experiments, cook gets the lissajous data points. Zmod, Zphz, etc must be collected after DataDone is fired
    def cook(self):
        count = 1
        while count > 0:
            count, points = self.dtaq.Cook(1024)
            self.acquired_points.extend(zip(*points))

    #cooks lissajous data points when DataAvalible event is fired
    def _IGamryReadZEvents_OnDataAvailable(self, this):
        self.cook()

    #The bread and butter of ReadZ. All readz.measure() is called from here after the very first frequency point
    def _IGamryReadZEvents_OnDataDone(self, this, status1):
        global status
        global freq
        global point
        global initfreq
        global loginc
        global passes

        status = readz.StatusMessage() #string based message about data point

        print('At ' + str(freq) + ' Hz the measurement status is')
        print(status)

        datatime = time.time() - starttime

        if status1 == 0: #data acceptable, ie an impedance value was able to be obtained. Does not indicate quality

            writedata(point, datatime, readz.Zfreq(), readz.Zreal(), readz.Zimag(), readz.Zsig(), readz.Zmod(), readz.Zphz(), readz.Idc(), readz.Vdc(), readz.IERange())

            point = point + 1
            freq = 10**(log10(initialf) + (point * loginc))
            if point > maxpoints: # we have acquired all data points over specified freq range. Clean up and exit
                stopacq()
                return
            else: #still more data points to be collected. Measure impednace at next point
                readz.Measure(freq, ac)
                return

        if status1 == 1: #impedance value could not be determined. Retry 10 times and throw an error. At each re-try the instrument makes automatic changes to try and get a value the next pass. Give user an option to cancel experiment, retry point, or move to next point
            if passes > 10:
                result1 = mbox('Measurement Error', 'Unable to take measurement at {} Hz'.format(freq), 6)
                if result1 == 2:  # abort
                    stopacq()
                if result1 == 10:  # retry current frequency
                    passes = 0
                    readz.Measure(freq, ac)
                if result1 == 11:  # move to next frequency point
                    passes = 0
                    point = point + 1
                    freq = 10 ** (log10(initialf) + (point * loginc))
                    readz.Measure(freq, ac)
            else:
                passes = passes + 1
                readz.Measure(freq, ac)

        else: #impedance value could not be determined. Catch all
            result = mbox('Measurement Error', 'Bad Measurement at {} Hz'.format(freq), 6)
            if result == 2: #abort
                stopacq()
            if result == 10: #retry current frequency
                readz.Measure(freq, ac)
            if result == 11: #move to next frequency point
                point = point + 1
                freq = 10 ** (log10(initialf) + (point * loginc))
                readz.Measure(freq, ac)
            return

#determines number of data points to be taken based on frequency range and points per decade setup parameters
def eispoints():
    return round(0.5 + (abs(log10(finalf) - log10(initialf))*ptsperdec))

#runs the first EIS point after grabing and initializing a pstat from device list. All other frequency points are triggered to run once the DataDone event is fired from GamryCom
def run(initfreq, finfreq, ac):

    global freq
    global finalf
    global initialf
    global starttime
    global passes

    pstat.Init(devices.EnumSections()[0])  # grab first pstat

    pstat.Open()

    writeheader()

    readz.Init(pstat)
    initializepstat(pstat)

    pstat.SetCell(GamryCOM.CellOn)

    passes = 0
    freq = initfreq
    finalf = finfreq
    initialf = initfreq
    starttime = time.time()
    readz.Measure(freq, ac)
    return

#shuts of pstat cell switch, tells comtypes to stop pumping events, ends connection to GamryCom, cleans up
def stopacq():

    global active
    global connection

    pstat.SetCell(GamryCOM.CellOff)
    time.sleep(1)
    pstat.Close()
    datafile.close()
    print("done")
    del connection
    gc.collect()

    active = False

    return

#sets up a cancel, retry, continue
def mbox(title, text, style):

    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

#write header file to text file so Analyst can utilize the data file
def writeheader():

    datafile.write('EXPLAIN\n' 'TAG\tEISPOT\n' 'TITLE\tLABEL\tPotentiostatic EIS\tTest &Identifier\n' 'ZCURVE\tTABLE\n' '\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange\n' '\t#\ts\tHz\tohm\tohm\tV\tohm\t°\tA\tV\t#\n')

#write each ReadZ data to file
def writedata(pt, time, freq, zreal, zimg, zsig, zmod, zphz, idc, vdc, ier):

    datafile.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(pt, time, freq, zreal, zimg, zsig, zmod, zphz, idc, vdc, ier))


###########################################################################################
# readz object creation via GamryCOM
# Creates the comtypes sink that is used in GetEvents
# Creates a pstat object and device list object

readz = client.CreateObject('GamryCOM.GamryReadZ')

dtaqsink = GamryReadZEvents(readz)
connection = client.GetEvents(readz, dtaqsink)
status1 = GamryCOM.gcREADZSTATUS
pstat = client.CreateObject('GamryCOM.GamryPC6Pstat')
devices = client.CreateObject('GamryCOM.GamryDeviceList')

active = True

############################################################################################
#Setup parameters for EIS run
directory = "C:\\Users\\Lab User\\Documents\\jdhuang\\"
filename = "sample EIS"
initfreq = 1000000.0 #initial frequency to run
initialf = initfreq
finfreq = 1000  #run to this frequency
finalf = finfreq
ac = 0.02 #AC voltage, in Volts
dc = 0.02 #dc voltage, in Volts
ptsperdec = 10 #amount of data points to take per decade of frequency

############################################################################################
#book keeping variables
point = 0

loginc = 1/(ptsperdec)
if initfreq > finfreq:
    loginc = -loginc

maxpoints = eispoints()
completename = directory + filename + ".DTA"
datafile = open(completename, 'a')
starttime = 0


############################################################################################

#run here
#events from GamryCom are pumped back every second while data acquisition is active
print('name:', __name__)
if __name__ == "__main__":
    try:
        start = time.time()
        run(initialf, finalf, ac)
        while active == True:
            client.PumpEvents(1)
            time.sleep(0.1)
        print('run time: {:.2f} s'.format(time.time() - start))
    except Exception as e:
        raise gamry_error_decoder(e)














