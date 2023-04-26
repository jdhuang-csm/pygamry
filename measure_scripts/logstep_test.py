import os
import time
import numpy as np
import run_functions as rf
from pygamry.dtaq import DtaqChrono, get_pstat


pstat = get_pstat()

chrono = DtaqChrono('galv', write_mode='once', write_precision=6, exp_notes=None,
                        leave_cell_on=False, start_with_cell_off=True)

v_rms = 0.01
decimate = True
s_init = 0
t_init = 0.1
t_long = 1
t_sample = 1e-4
num_scales = 4
file_suffix = 'LogStepDecimated'

data_path = 'C:\\Users\\jdhuang\\Documents\\Gamry_data\\Pub\\Fine-Coarse\\220320_220311-2t\\manual\\run2'
kst_path = 'C:\\Users\\jdhuang\\Documents\\Gamry_data\\Kst_dashboard'

# Test current
v_oc = rf.test_ocv(pstat)
s_rms = rf.find_current(pstat, v_oc + v_rms, 2.0)
s_half_step = s_rms * np.sqrt(2)
time.sleep(1)  # rest

# Configure decimation
if decimate:
    chrono.configure_decimation('write', 20, 10, 2, 0.05)

# Configure step signal
chrono.configure_logstep_signal(s_init, s_half_step, t_init, t_long, t_sample, num_scales)

# Get result file
if chrono.mode == 'galv':
    tag_letter = 'P'
else:
    tag_letter = 'A'

result_file = os.path.join(data_path, 'CHRONO{}_{}.DTA'.format(tag_letter, file_suffix))

if kst_path is not None:
    kst_file = os.path.join(kst_path, 'Kst_IVT.DTA')
else:
    kst_file = None

print('Running CHRONO')
chrono.run(pstat, result_file=result_file, kst_file=kst_file,
         decimate=decimate)
print('CHRONO done\n')
