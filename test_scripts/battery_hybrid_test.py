import time
import numpy as np
import os

from pygamry.dtaq import get_pstat, DtaqOcv
from pygamry.sequencers import HybridSequencer

data_path = 'C:\\Users\\jdhuang\\Documents\\Gamry_data\\Batteries\\Molicel_M35A\\Cell1\\230406'
# data_path = '..\\test_data'

pstat = get_pstat()

seq = HybridSequencer()
seq.configure_decimation('write', 100, 10, 2, 0.1)

dt_ocv = DtaqOcv()

s_rms = 0.1
print('s_rms:', s_rms)

t_step = 1
t_sample = 1e-3

file_suffix = '_Hybrid_Irms=100mA_tstep=10s_tsample=1ms_Rest=10s'


# Single measurement: triple step
# --------------------------------
# Equilibrate at OCV
print('Running OCV...')
dt_ocv.run(pstat, duration=200, t_sample=1, result_file=os.path.join(data_path, f'OCV{file_suffix}.DTA'),
           show_plot=False)


# Run hybrid measurement
seq.configure_triple_step(0, s_rms, 0.5, t_step, t_sample, np.logspace(5, 2, 31), init_sign=-1)
seq.run(pstat, True, data_path=data_path, file_suffix=file_suffix, rest_time=10)
