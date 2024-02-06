import time
import numpy as np
import os


import run_functions as rf

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqChrono, HybridSequencer

data_path = 'C:\\Users\\jdhuang\\Documents\\Gamry_data\\Batteries\\Molicel_M35A\\Cell1\\230406'
# data_path = '..\\test_data'

pstat = get_pstat()

sequencer = HybridSequencer(mode='galv')
sequencer.configure_decimation('write', 100, 10, 2, 0.1)

dt_chrono = DtaqChrono(mode='galv')

i_dc = -0.7
equil_time = 30


file_suffix = 'Hybrid_Idc=-700mA_Irms=125mA_tstep=16s_tsample=1ms_Rest=2s'


# Configure hybrid
if sequencer.mode != 'galv':
    raise ValueError("run_hybrid expects a sequencer with mode = 'galv';"
                     f" received sequencer with mode {sequencer.mode}")

# Configure decimation
sequencer.configure_decimation('write', 25, 50, 2, 0.1)

# Get EIS frequencies
eis_freq = rf.get_eis_frequencies(1e5, 100, 10)

# Configure chrono step
i_init = i_dc
t_init = 1
i_rms = 0.125
t_step = 16
t_sample = 1e-3
rest_time = 2

print('i_rms: {:.2g} A'.format(i_rms))

sequencer.configure_geo_step(i_init, i_rms, t_init, t_short=1e-2,
                             t_long=t_step, t_sample=t_sample,
                             num_scales=4, steps_per_scale=2,
                             frequencies=eis_freq, end_at_init=False)

# Condition to i_dc
if i_dc != 0:
    dt_chrono.configure_mstep_signal(0, i_dc, 1, equil_time, 1e-3, n_steps=1)
    dt_chrono.leave_cell_on = True
    start_with_cell_off = False

    dt_chrono.configure_decimation('write', 10, 5, 2, 1)

    print('Conditioning at {:.3f} A for {:.0f} s...'.format(i_dc, equil_time))
    chrono_file = os.path.join(data_path, f'Conditioning_{file_suffix}.DTA')
    dt_chrono.run(pstat, result_file=chrono_file, decimate=True)
else:
    start_with_cell_off = True

print('Running HYBRID')
sequencer.run(pstat, decimate=True, data_path=data_path,
              file_suffix=file_suffix, rest_time=rest_time, filter_response=False,
              start_with_cell_off=start_with_cell_off,
              show_plot=False)
print('HYBRID done\n')
