import time
import numpy as np
import os

from pygamry.dtaq import get_pstat, DtaqReadZ, HybridSequencer

data_path = 'C:\\Users\\jdhuang\\Documents\\Gamry_data\\Batteries\\Molicel_M35A\\Cell1'
# data_path = '..\\test_data'

pstat = get_pstat()

seq = HybridSequencer()
seq.configure_decimation('write', 10, 10, 2, 0.5)

R_est = 0.05
s_rms = 0.01 / R_est
print('s_rms:', s_rms)


# Single measurement: triple step
# --------------------------------
# seq.configure_triple_step(0, s_rms, 0.05, 1.5, 1e-4, np.logspace(5, 3, 21))
seq.configure_triple_step(0, s_rms, 1, 10, 1e-3, np.logspace(5, 2, 31), init_sign=-1)

seq.run(pstat, True, data_path=data_path, file_suffix='_HybridTest')

# # Single measurement: hzh step
# # --------------------------------
# seq.configure_hzh_step(0, 5e-5, 0.1, 5, 1e-4, np.logspace(5, 3, 11))
# seq.run(pstat, True, data_path=data_path, file_suffix='_HybridHZHTest')

# # Single measurement: fz step
# # --------------------------------
# seq.configure_fz_step(0, 5e-5, 0.1, 5, 1e-4, np.logspace(5, 3, 11))
# seq.run(pstat, True, data_path=data_path, file_suffix='_HybridFZTest')

# Staircase: hzh step
# ------------------------
# run_path = os.path.join(data_path, 'hybrid_2d_ec', 'FZ_step')

# file_suffix = '_HybridStaircase'

# # Pre-staircase EIS
# dt_eis = DtaqReadZ('galvanostatic', write_mode='continuous', leave_cell_on=False)
# full_frequencies=np.logspace(5, -1, 61)
# dt_eis.run(pstat, full_frequencies, 0, abs(s_rms), R_est, timeout=350,
#            result_file=os.path.join(run_path, f'EISGALV{file_suffix}_Pre.DTA')
#            )

# num_steps = 20
# seq.configure_staircase(0, s_rms, 0.1, 1.5, 1e-4, np.logspace(5, 3, 21), num_steps, 'fz')
# seq.run_staircase(pstat, True, data_path=run_path, file_suffix=file_suffix, equil_time=5)
#                   # run_full_eis_pre=False, run_full_eis_post=True, full_frequencies=np.logspace(5, -1, 61))

# # Post-staircase EIS
# s_dc_final = num_steps * 2 * np.sqrt(2) * s_rms
# dt_eis.start_with_cell_off = False
# dt_eis.leave_cell_on = False
# dt_eis.run(pstat, full_frequencies, s_dc_final, abs(s_rms), R_est, timeout=350,
#            result_file=os.path.join(run_path, f'EISGALV{file_suffix}_Post.DTA')
#            )

# Staircase: fz step
# ------------------------
# print('s_rms:', s_rms)
# print('s_step:', s_rms * np.sqrt(2) * 2)
# seq.configure_staircase(0, -s_rms, 0.1, 5, 1e-4, np.logspace(5, 3, 11), 2, 'fz')
# seq.run_staircase(pstat, True, data_path=data_path, file_suffix='_HybridStaircaseFZ', equil_time=2)

# pstat.Open()
# dc_amp = 2 * s_rms * np.sqrt(2) * 20
# print(dc_amp)
# ie_range = pstat.TestIERange(dc_amp)
# R_ie = pstat.IEResistor(ie_range)
# v_dc = R_ie * dc_amp
# pstat.SetIERange(ie_range)
# pstat.SetVoltage(v_dc)
# time.sleep(20)
#
# seq.dt_eis.run(pstat, np.logspace(5, -1, 61), dc_amp, s_rms, 1, timeout=400,
#                result_file=os.path.join(run_path, 'EISGALV_HybridStaircaseHZH_tstep=0.75s_Post.DTA'))
#
# pstat.SetCell(0)
# pstat.Close()